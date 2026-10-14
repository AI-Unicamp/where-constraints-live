"""
Experiment: product (fuzzy union) vs weighted sum for primitive selection in CSG.

A/B test comparing two aggregation modes in SelectingLeaf:
  A) weighted_sum: output = Σ w_k · occ_k(p)        [current/baseline]
  B) product_fuzzy: output = 1 - Π_k (1 - w_k · occ_k(p))  [proposed]

Hypothesis (from NAS experiments):
  - Both give similar selection quality (sine provides exploration in both)
  - Product gives less parameter drift during phase 1 because gradient to
    primitive k's params is multiplicatively suppressed when w_k ~ 0
  - This means phase 2 starts from better initialization -> faster convergence
  - Product is well-typed on [0,1] occupancies (unlike NAS where unbounded
    features caused compression artifacts)

Key metrics:
  - Phase 1 IoU at transition (selection quality)
  - Phase 2 convergence speed (parameter preservation payoff)
  - Final IoU (overall quality)
  - Primitive type distribution (do they select differently?)
"""

import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import mcubes
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import trimesh

# Optional: for accurate SDF computation
try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    print("Warning: libigl not found. Using approximate SDF (less accurate).")

# DEVICE SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# CHECKPOINT MANAGER
@dataclass
class TrainingState:
    epoch: int
    losses: List[float]
    ious: List[float]
    best_iou: float
    best_epoch: int
    phase: int = 1
    phase1_epochs: int = 0
    selected_indices: Optional[List[int]] = None
    # IoU at the phase transition
    transition_iou: float = 0.0

class CheckpointManager:
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.checkpoint_path = os.path.join(output_folder, 'checkpoint.pt')
        self.best_model_path = os.path.join(output_folder, 'best_model.pt')
        self.state_path = os.path.join(output_folder, 'training_state.json')
        os.makedirs(output_folder, exist_ok=True)
    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
             state: TrainingState):
        selected_indices = None
        if hasattr(model, '_selected_global_indices') and model._selected_global_indices is not None:
            selected_indices = model._selected_global_indices
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': state.epoch,
            'losses': state.losses,
            'ious': state.ious,
            'best_iou': state.best_iou,
            'best_epoch': state.best_epoch,
            'phase': state.phase,
            'phase1_epochs': state.phase1_epochs,
            'selected_indices': selected_indices,
            'temperature': model.temperature if hasattr(model, 'temperature') else None,
            'transition_iou': state.transition_iou,
        }, self.checkpoint_path)
        
        if state.ious and state.ious[-1] >= state.best_iou:
            torch.save(model.state_dict(), self.best_model_path)
        
        with open(self.state_path, 'w') as f:
            json.dump({
                'epoch': state.epoch,
                'phase': state.phase,
                'phase1_epochs': state.phase1_epochs,
                'best_iou': state.best_iou,
                'best_epoch': state.best_epoch,
                'final_loss': state.losses[-1] if state.losses else None,
                'selected_indices': selected_indices,
                'transition_iou': state.transition_iou,
            }, f, indent=2)
    
    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
             ) -> Optional[TrainingState]:
        if not os.path.exists(self.checkpoint_path):
            return None
        
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('phase', 1) == 2 and checkpoint.get('selected_indices') is not None:
            model.phase = 2
            model._selected_global_indices = checkpoint['selected_indices']
            for i, leaf in enumerate(model.leaves):
                local_idx = checkpoint['selected_indices'][i] % len(model.primitive_types)
                leaf.fixed_idx = local_idx
        
        if checkpoint.get('temperature') is not None:
            model.temperature = checkpoint['temperature']
        
        return TrainingState(
            epoch=checkpoint['epoch'],
            losses=checkpoint['losses'],
            ious=checkpoint['ious'],
            best_iou=checkpoint['best_iou'],
            best_epoch=checkpoint['best_epoch'],
            phase=checkpoint.get('phase', 1),
            phase1_epochs=checkpoint.get('phase1_epochs', 0),
            selected_indices=checkpoint.get('selected_indices'),
            transition_iou=checkpoint.get('transition_iou', 0.0),
        )
    
    def load_best(self, model: nn.Module) -> bool:
        if os.path.exists(self.best_model_path):
            model.load_state_dict(torch.load(self.best_model_path, map_location=device, weights_only=True))
            return True
        return False


# MESH SDF CLASS
class MeshSDF:
    def __init__(self, mesh_path: str, normalize: bool = True, margin: float = 0.1,
                 auto_repair: bool = True):
        print(f"Loading mesh: {mesh_path}")
        self.mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(
                [g for g in self.mesh.geometry.values()]
            )
        if not self.mesh.is_watertight and auto_repair:
            self.mesh = self._repair_mesh(self.mesh, mesh_path)
        if normalize:
            center = self.mesh.bounding_box.centroid
            self.mesh.vertices -= center
            max_extent = np.abs(self.mesh.vertices).max()
            scale = (1.0 - margin) / max_extent
            self.mesh.vertices *= scale
        self.vertices = np.ascontiguousarray(self.mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.mesh.faces, dtype=np.int64)
        print(f"  Vertices: {len(self.vertices):,}, Faces: {len(self.faces):,}")
        print(f"  Bounds: [{self.vertices.min():.3f}, {self.vertices.max():.3f}]")
        print(f"  Watertight: {self.mesh.is_watertight}")

    def _repair_mesh(self, mesh, original_path):
        print("  Mesh is not watertight. Attempting repair...")
        try:
            import pymeshfix
            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair(verbose=False)
            verts = meshfix.mesh.points
            faces_raw = meshfix.mesh.faces
            faces = faces_raw.reshape(-1, 4)[:, 1:4] if faces_raw.ndim == 1 else faces_raw
            repaired = trimesh.Trimesh(vertices=verts, faces=faces)
            if repaired.is_watertight:
                print("  ✓ PyMeshFix repair successful!")
                return repaired
        except Exception:
            pass
        try:
            from scipy import ndimage
            resolution = 128
            voxels = mesh.voxelized(pitch=2.0 / resolution)
            voxel_grid = ndimage.binary_fill_holes(voxels.matrix.astype(np.float32)).astype(np.float32)
            voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.3)
            voxel_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
            vertices, triangles = mcubes.marching_cubes(voxel_grid, 0.5)
            vertices = (vertices - 1) / resolution * (mesh.vertices.max() - mesh.vertices.min()) + mesh.vertices.min()
            repaired = trimesh.Trimesh(vertices=vertices, faces=triangles)
            print(f"  ✓ Voxelization repair (watertight: {repaired.is_watertight})")
            return repaired
        except Exception as e:
            print(f"  Repair failed: {e}")
        return mesh

    def query(self, points: np.ndarray) -> np.ndarray:
        points = np.ascontiguousarray(points, dtype=np.float64)
        if HAS_IGL:
            sdf, _, _, _ = igl.signed_distance(points, self.vertices, self.faces)
            return sdf.astype(np.float32)
        else:
            closest_points, distance, _ = self.mesh.nearest.on_surface(points)
            inside = self.mesh.contains(points)
            sdf = distance.copy()
            sdf[inside] *= -1
            return sdf.astype(np.float32)

    def query_sdf_grid(self, resolution: int) -> np.ndarray:
        idx = np.linspace(-1, 1, resolution)
        x, y, z = np.meshgrid(idx, idx, idx, indexing='ij')
        grid_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        return self.query(grid_points).reshape(resolution, resolution, resolution)

    def sample_points(self, num_samples: int, surface_ratio: float = 0.4,
                      near_ratio: float = 0.4, near_std: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
        num_surface = int(num_samples * surface_ratio)
        num_near = int(num_samples * near_ratio)
        num_random = num_samples - num_surface - num_near
        points_surface, _ = trimesh.sample.sample_surface(self.mesh, num_surface)
        points_near_base, _ = trimesh.sample.sample_surface(self.mesh, num_near)
        points_near = np.clip(points_near_base + np.random.randn(num_near, 3) * near_std, -0.99, 0.99)
        points_random = np.random.uniform(-1, 1, (num_random, 3))
        points = np.vstack([points_surface, points_near, points_random]).astype(np.float32)
        sdf = self.query(points)
        occupancy = (sdf < 0).astype(np.float32)
        perm = np.random.permutation(len(points))
        return points[perm], occupancy[perm]

def create_mesh_target_func(mesh_sdf: MeshSDF) -> Callable:
    def target_func(grid_points: torch.Tensor) -> torch.Tensor:
        points_np = grid_points.detach().cpu().numpy()
        sdf = mesh_sdf.query(points_np)
        return torch.from_numpy((sdf < 0).astype(np.float32)).to(device)
    return target_func


# UTILITY FUNCTIONS
def sample_3D_grid(resolution: int, low: float = -1, high: float = 1) -> torch.Tensor:
    idx = torch.linspace(low, high, resolution, device=device)
    x, y, z = torch.meshgrid(idx, idx, idx, indexing='ij')
    return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

def marching_cubes_from_sdf(filepath, sdf_grid, resolution):
    if sdf_grid.min() >= 0 or sdf_grid.max() <= 0:
        return None
    vertices, triangles = mcubes.marching_cubes(sdf_grid, 0)
    vertices = vertices / (resolution - 1) * 2 - 1
    triangles = triangles.copy()
    triangles[:, [0, 1]] = triangles[:, [1, 0]]
    mcubes.export_obj(vertices, triangles, filepath)
    return vertices, triangles

def marching_cubes_from_occupancy(filepath, occupancy, resolution):
    occ = occupancy.detach().cpu().numpy().reshape(resolution, resolution, resolution)
    if occ.min() >= 0.5 or occ.max() <= 0.5:
        return None
    pseudo_sdf = -(occ * 2 - 1)
    vertices, triangles = mcubes.marching_cubes(pseudo_sdf, 0)
    vertices = vertices / (resolution - 1) * 2 - 1
    triangles = triangles.copy()
    triangles[:, [0, 1]] = triangles[:, [1, 0]]
    mcubes.export_obj(vertices, triangles, filepath)
    return vertices, triangles

def compute_iou(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).float()
    gt_bin = (gt > threshold).float()
    intersection = (pred_bin * gt_bin).sum()
    union = ((pred_bin + gt_bin) > 0).float().sum()
    return (intersection / (union + 1e-8)).item()


# SDF PRIMITIVES
MIN_SIZE = 0.02

def euler_to_rotation_matrix(angles):
    cx, sx = torch.cos(angles[0]), torch.sin(angles[0])
    cy, sy = torch.cos(angles[1]), torch.sin(angles[1])
    cz, sz = torch.cos(angles[2]), torch.sin(angles[2])
    R = torch.zeros(3, 3, device=angles.device, dtype=angles.dtype)
    R[0, 0] = cy * cz;  R[0, 1] = cz * sx * sy - cx * sz;  R[0, 2] = cx * cz * sy + sx * sz
    R[1, 0] = cy * sz;  R[1, 1] = cx * cz + sx * sy * sz;   R[1, 2] = cx * sy * sz - cz * sx
    R[2, 0] = -sy;      R[2, 1] = cy * sx;                   R[2, 2] = cx * cy
    return R

def sdf_block(p, center, half_size, rotation):
    p_local = torch.matmul(p - center, rotation.T)
    d = torch.abs(p_local) - half_size
    return torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(dim=-1)[0], max=0)

def sdf_sphere(p, center, radius):
    return torch.sqrt(((p - center)**2).sum(-1) + 1e-8) - radius

def sdf_cylinder(p, center, radius, half_height, rotation):
    p_local = torch.matmul(p - center, rotation.T)
    d_radial = torch.sqrt(p_local[:, 0]**2 + p_local[:, 2]**2 + 1e-8) - radius
    d_height = torch.abs(p_local[:, 1]) - half_height
    return torch.sqrt(torch.clamp(d_radial, min=0)**2 + torch.clamp(d_height, min=0)**2 + 1e-8) + \
           torch.clamp(torch.maximum(d_radial, d_height), max=0)

def sdf_wedge(p, center, half_size, rotation):
    p_local = torch.matmul(p - center, rotation.T)
    hy, hz = half_size[1], half_size[2]
    d = torch.abs(p_local) - half_size
    box_sdf = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(dim=-1)[0], max=0)
    inv_norm = 1.0 / torch.sqrt(1.0/(hy**2) + 1.0/(hz**2) + 1e-8)
    plane_sdf = (p_local[:, 1]/hy + p_local[:, 2]/hz - 1.0) * inv_norm
    return torch.maximum(box_sdf, plane_sdf)

def sdf_corner_wedge(p, center, half_size, rotation):
    p_local = torch.matmul(p - center, rotation.T)
    hx, hy, hz = half_size[0], half_size[1], half_size[2]
    d = torch.abs(p_local) - half_size
    box_sdf = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(dim=-1)[0], max=0)
    inv_norm1 = 1.0 / torch.sqrt(1.0/(hx**2) + 1.0/(hy**2) + 1e-8)
    plane1_sdf = (p_local[:, 0]/hx + p_local[:, 1]/hy - 1.0) * inv_norm1
    inv_norm2 = 1.0 / torch.sqrt(1.0/(hz**2) + 1.0/(hy**2) + 1e-8)
    plane2_sdf = (p_local[:, 2]/hz + p_local[:, 1]/hy - 1.0) * inv_norm2
    return torch.maximum(torch.maximum(box_sdf, plane1_sdf), plane2_sdf)


# PRIMITIVE POOL
class PrimitivePool(nn.Module):
    PARAM_COUNTS = {'block': 9, 'sphere': 4, 'cylinder': 8, 'wedge': 9, 'corner_wedge': 9}
    
    def __init__(self, primitive_types, init_scale=0.5):
        super().__init__()
        self.types = primitive_types
        self.num_primitives = len(primitive_types)
        total_params = sum(self.PARAM_COUNTS[t] for t in primitive_types)
        self.params = nn.Parameter(torch.randn(total_params) * init_scale)
        self.sharpness = nn.Parameter(torch.ones(self.num_primitives) * 20.0)
        self.param_indices = []
        idx = 0
        for t in primitive_types:
            count = self.PARAM_COUNTS[t]
            self.param_indices.append((idx, idx + count))
            idx += count

    def get_sdf(self, p, prim_idx):
        ptype = self.types[prim_idx]
        start, end = self.param_indices[prim_idx]
        params = self.params[start:end]
        if ptype == 'block':
            return sdf_block(p, params[:3], params[3:6].abs() + MIN_SIZE, euler_to_rotation_matrix(params[6:9]))
        elif ptype == 'sphere':
            return sdf_sphere(p, params[:3], params[3].abs() + MIN_SIZE)
        elif ptype == 'cylinder':
            return sdf_cylinder(p, params[:3], params[3].abs() + MIN_SIZE, params[4].abs() + MIN_SIZE, euler_to_rotation_matrix(params[5:8]))
        elif ptype == 'wedge':
            return sdf_wedge(p, params[:3], params[3:6].abs() + MIN_SIZE, euler_to_rotation_matrix(params[6:9]))
        elif ptype == 'corner_wedge':
            return sdf_corner_wedge(p, params[:3], params[3:6].abs() + MIN_SIZE, euler_to_rotation_matrix(params[6:9]))

    def get_occupancy(self, p, prim_idx):
        sdf = self.get_sdf(p, prim_idx)
        s = self.sharpness[prim_idx].abs().clamp(min=1.0)
        return torch.sigmoid(-sdf * s)

    def get_occupancies_batch(self, p, indices):
        return torch.stack([self.get_occupancy(p, i) for i in indices], dim=-1)

    def get_all_occupancies(self, p):
        return self.get_occupancies_batch(p, list(range(self.num_primitives)))


# SELECTING LEAF (two aggregation modes)
class SelectingLeaf(nn.Module):
    """
    Fuzzy primitive selector with two aggregation modes:
    
    'weighted_sum':  output = Σ w_k · occ_k(p)
        Standard linear mixture. Rich gradient signal but causes
        cross-primitive parameter interference.
    
    'product_fuzzy': output = 1 - Π_k (1 - w_k · occ_k(p))
        Fuzzy union over weighted occupancies. When w_k ~ 0, gradient
        to primitive k is multiplicatively suppressed (not just scaled).
        Stays in [0,1] for [0,1] occupancies, so no compression is needed.
    """
    
    def __init__(self, num_primitives: int, omega: float = 10.0, 
                 init_scale: float = 0.3, agg_mode: str = 'weighted_sum'):
        super().__init__()
        self.num_primitives = num_primitives
        self.omega = omega
        self.agg_mode = agg_mode
        self.selection_logits = nn.Parameter(torch.randn(num_primitives) * init_scale)
        self.fixed_idx: Optional[int] = None
    
    def get_selection_weights(self, temperature: float = 1000.0) -> torch.Tensor:
        logits = torch.sin(self.omega * self.selection_logits)
        return F.softmax(logits * temperature, dim=-1)

    def get_selected_idx(self) -> int:
        return self.get_selection_weights(0.01).argmax().item()
    
    def forward(self, occupancies: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if self.fixed_idx is not None:
            return occupancies[:, self.fixed_idx]
        
        weights = self.get_selection_weights(temperature)
        
        if self.agg_mode == 'weighted_sum':
            # Baseline: linear mixture
            return (occupancies * weights).sum(dim=-1)
        
        elif self.agg_mode == 'product_fuzzy':
            # Proposed: fuzzy union over weighted occupancies
            # output = 1 - Π_k (1 - w_k · occ_k)
            # When w_j->1 and others->0: reduces to occ_j exactly
            # Gradient to prim_k params ∝ w_k · Π_{j≠k}(1 - w_j·occ_j)
            #   -> multiplicatively suppressed when w_k small
            complement = torch.ones(occupancies.shape[0], device=occupancies.device)
            for k in range(self.num_primitives):
                complement = complement * (1.0 - weights[k] * occupancies[:, k])
            return 1.0 - complement
        
        else:
            raise ValueError(f"Unknown agg_mode: {self.agg_mode}")
    
    def fix_selection(self) -> int:
        self.fixed_idx = self.get_selected_idx()
        return self.fixed_idx


# TWO-PHASE CSG TREE
class TwoPhaseCSGTree(nn.Module):
    def __init__(self, depth: int = 4, primitive_types=None,
                 init_scale: float = 0.3, omega: float = 10.0,
                 agg_mode: str = 'weighted_sum'):
        super().__init__()
        self.depth = depth
        self.omega = omega
        self.agg_mode = agg_mode
        self.num_leaves = 2 ** depth
        self.num_internal = 2 ** depth - 1
        self.phase = 1
        self.temperature = 100.0
        
        if primitive_types is None:
            primitive_types = ['block', 'sphere', 'cylinder', 'wedge', 'corner_wedge']
        self.primitive_types = primitive_types
        self.n_types = len(primitive_types)
        
        all_types = primitive_types * self.num_leaves
        self.primitive_pool = PrimitivePool(all_types, init_scale)
        
        # Pass agg_mode to each leaf
        self.leaves = nn.ModuleList([
            SelectingLeaf(self.n_types, omega, init_scale, agg_mode=agg_mode)
            for _ in range(self.num_leaves)
        ])
        
        self.boolean_params = nn.Parameter(
            torch.randn(self.num_internal, 4) * init_scale
        )
        self._selected_global_indices = None

    def get_boolean_weights(self):
        return F.softmax(torch.sin(self.omega * self.boolean_params) * 1000.0, dim=-1)

    def _boolean_combine(self, left, right, weights):
        c0, c1, c2, c3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3], weights[:, 3:4]
        return (c1 + c2) * left + (c1 + c3) * right + (c0 - c1 - c2 - c3) * left * right

    def forward_phase1(self, p):
        all_occs = self.primitive_pool.get_all_occupancies(p)
        leaf_outputs = []
        for i, leaf in enumerate(self.leaves):
            start, end = i * self.n_types, (i + 1) * self.n_types
            leaf_outputs.append(leaf(all_occs[:, start:end], self.temperature))
        return self._tree_combine(leaf_outputs)

    def forward_phase2(self, p):
        selected_occs = self.primitive_pool.get_occupancies_batch(p, self._selected_global_indices)
        leaf_outputs = [selected_occs[:, i] for i in range(self.num_leaves)]
        return self._tree_combine(leaf_outputs)

    def _tree_combine(self, leaf_outputs):
        current = torch.stack(leaf_outputs, dim=0)
        bool_weights = self.get_boolean_weights()
        bool_idx = self.num_internal - 1
        for level in range(self.depth - 1, -1, -1):
            num_nodes = 2 ** level
            left, right = current[0::2], current[1::2]
            bw = bool_weights[bool_idx - num_nodes + 1:bool_idx + 1]
            current = self._boolean_combine(left, right, bw)
            bool_idx -= num_nodes
        return current[0]

    def forward(self, p):
        if self.phase == 2 and self._selected_global_indices is not None:
            return self.forward_phase2(p)
        return self.forward_phase1(p)

    def set_temperature(self, temp):
        self.temperature = temp

    def transition_to_phase2(self):
        self._selected_global_indices = []
        stats = {
            'selections': [],
            'primitives_before': self.primitive_pool.num_primitives,
            'type_counts': {t: 0 for t in self.primitive_types}
        }
        for i, leaf in enumerate(self.leaves):
            local_idx = leaf.fix_selection()
            global_idx = i * self.n_types + local_idx
            ptype = self.primitive_types[local_idx]
            self._selected_global_indices.append(global_idx)
            stats['selections'].append({'leaf': i, 'type': ptype, 'local_idx': local_idx, 'global_idx': global_idx})
            stats['type_counts'][ptype] += 1
        self.phase = 2
        stats['primitives_used'] = len(set(self._selected_global_indices))
        return stats

    def count_total_primitives(self):
        return self.primitive_pool.num_primitives


# SYNTHETIC TARGETS
def create_target_sphere_with_hole(grid_points):
    sphere_sdf = torch.sqrt((grid_points**2).sum(-1) + 1e-8) - 0.7
    cyl_sdf = torch.sqrt(grid_points[:, 0]**2 + grid_points[:, 2]**2 + 1e-8) - 0.2
    return (torch.maximum(sphere_sdf, -cyl_sdf) < 0).float()

def create_target_complex_csg(grid_points):
    d = torch.abs(grid_points) - 0.5
    box_sdf = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(dim=-1)[0], max=0)
    sphere_center = torch.tensor([0.3, 0.3, 0.3], device=device)
    sphere_sdf = torch.sqrt(((grid_points - sphere_center)**2).sum(-1) + 1e-8) - 0.35
    union_sdf = torch.minimum(box_sdf, sphere_sdf)
    cyl_y = torch.sqrt(grid_points[:, 0]**2 + grid_points[:, 2]**2 + 1e-8) - 0.15
    cyl_x = torch.sqrt(grid_points[:, 1]**2 + grid_points[:, 2]**2 + 1e-8) - 0.15
    return (torch.maximum(torch.maximum(union_sdf, -cyl_y), -cyl_x) < 0).float()


# PARAMETER DRIFT MEASUREMENT
def measure_parameter_drift(model_initial_params: Dict[str, torch.Tensor],
                            model: TwoPhaseCSGTree) -> Dict[str, float]:
    """
    Measure how much primitive parameters drifted during phase 1.
    
    For each leaf, compare selected primitive's current params to its
    initial values. High drift = more corruption from cross-primitive
    interference during soft mixing.
    """
    current_params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    # Focus on primitive pool params
    init_pool = model_initial_params.get('primitive_pool.params', None)
    curr_pool = current_params.get('primitive_pool.params', None)
    
    if init_pool is None or curr_pool is None:
        return {'total_drift': -1.0}
    
    # Per-selected-primitive drift
    drifts = []
    if model._selected_global_indices is not None:
        for global_idx in model._selected_global_indices:
            start, end = model.primitive_pool.param_indices[global_idx]
            init_p = init_pool[start:end]
            curr_p = curr_pool[start:end]
            drift = (init_p - curr_p).pow(2).mean().sqrt().item()
            drifts.append(drift)
    
    # Also measure drift of NON-selected primitives (should be lower for product)
    all_selected = set(model._selected_global_indices) if model._selected_global_indices else set()
    non_selected_drifts = []
    for idx in range(model.primitive_pool.num_primitives):
        if idx not in all_selected:
            start, end = model.primitive_pool.param_indices[idx]
            init_p = init_pool[start:end]
            curr_p = curr_pool[start:end]
            drift = (init_p - curr_p).pow(2).mean().sqrt().item()
            non_selected_drifts.append(drift)
    
    return {
        'selected_drift_mean': np.mean(drifts) if drifts else 0.0,
        'selected_drift_std': np.std(drifts) if drifts else 0.0,
        'non_selected_drift_mean': np.mean(non_selected_drifts) if non_selected_drifts else 0.0,
        'non_selected_drift_std': np.std(non_selected_drifts) if non_selected_drifts else 0.0,
    }


# TRAINING (single run, returns detailed metrics)
def train_single_run(
    agg_mode: str,
    target_func: Callable,
    mesh_sdf: Optional[MeshSDF],
    seed: int,
    depth: int = 4,
    primitive_types: List[str] = None,
    phase1_epochs: int = 10000,
    phase1_lr: float = 1e-3,
    phase1_temp_start: float = 0.3,
    phase1_temp_end: float = 2000.0,
    phase2_epochs: int = 25000,
    phase2_lr: float = 5e-4,
    num_samples: int = 1024*100,
    resample_every: int = 10,
    eval_every: int = 100,
    resolution: int = 128,
    output_folder: str = "./output/",
) -> Dict[str, Any]:
    """Run one complete training and return all metrics."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    
    if primitive_types is None:
        primitive_types = ['block', 'sphere', 'cylinder', 'wedge', 'corner_wedge']
    
    # Create model
    model = TwoPhaseCSGTree(
        depth=depth,
        primitive_types=primitive_types,
        init_scale=0.3,
        omega=10.0,
        agg_mode=agg_mode,
    ).to(device)
    
    # Save initial params for drift measurement
    initial_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {agg_mode} | seed={seed} | depth={depth}")
    print(f"  Leaves: {model.num_leaves}, Primitives: {model.count_total_primitives()}")
    print(f"  Phase 1: {phase1_epochs} epochs, Phase 2: {phase2_epochs} epochs")
    print(f"{'='*60}")
    
    # Eval grid
    eval_grid = sample_3D_grid(resolution, -1, 1)
    eval_gt = target_func(eval_grid)
    
    loss_func = nn.MSELoss()
    
    # Collect metrics
    metrics = {
        'agg_mode': agg_mode,
        'seed': seed,
        'depth': depth,
        'phase1_losses': [],
        'phase1_ious': [],
        'phase2_losses': [],
        'phase2_ious': [],
        'transition_iou': 0.0,
        'transition_type_counts': {},
        'param_drift': {},
        'final_iou': 0.0,
        'best_iou': 0.0,
        'best_epoch': 0,
        'phase2_iou_at_steps': {},  # IoU at specific phase2 checkpoints
    }
    
    # PHASE 1
    optimizer = torch.optim.Adam(model.parameters(), lr=phase1_lr)
    P, gt = None, None
    best_iou = 0.0
    
    pbar = tqdm(range(phase1_epochs), desc=f"P1 {agg_mode}", leave=False)
    for epoch in pbar:
        progress = epoch / max(phase1_epochs - 1, 1)
        temp = phase1_temp_start * (phase1_temp_end / phase1_temp_start) ** progress
        model.set_temperature(temp)
        
        if epoch % resample_every == 0 or P is None:
            if mesh_sdf is not None:
                points_np, occ_np = mesh_sdf.sample_points(num_samples, surface_ratio=0.3, near_ratio=0.5, near_std=0.03)
                P = torch.from_numpy(points_np).to(device)
                gt = torch.from_numpy(occ_np).to(device)
            else:
                P = torch.rand(num_samples, 3, device=device) * 2 - 1
                gt = target_func(P)
        
        pred = model(P)
        loss = loss_func(pred, gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics['phase1_losses'].append(loss.item())
        
        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                pred_full = torch.cat([model(b) for b in torch.split(eval_grid, 32**3)], dim=0)
                iou = compute_iou(pred_full, eval_gt)
                metrics['phase1_ious'].append(iou)
                if iou > best_iou:
                    best_iou = iou
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'IoU': f'{iou:.3f}', 'T': f'{temp:.1f}'})
    
    # Measure transition IoU
    with torch.no_grad():
        pred_full = torch.cat([model(b) for b in torch.split(eval_grid, 32**3)], dim=0)
        transition_iou = compute_iou(pred_full, eval_gt)
    metrics['transition_iou'] = transition_iou
    
    # Save phase 1 mesh
    try:
        marching_cubes_from_occupancy(os.path.join(output_folder, 'phase1_final.obj'), pred_full, resolution)
    except:
        pass
    
    # Transition
    stats = model.transition_to_phase2()
    metrics['transition_type_counts'] = stats['type_counts']
    
    # Measure parameter drift
    metrics['param_drift'] = measure_parameter_drift(initial_params, model)
    
    print(f"\n  Phase 1 done: IoU={transition_iou:.4f}")
    print(f"  Types: {stats['type_counts']}")
    print(f"  Drift (selected): {metrics['param_drift']['selected_drift_mean']:.4f} ± {metrics['param_drift']['selected_drift_std']:.4f}")
    print(f"  Drift (non-sel):  {metrics['param_drift']['non_selected_drift_mean']:.4f} ± {metrics['param_drift']['non_selected_drift_std']:.4f}")
    
    # PHASE 2
    optimizer = torch.optim.Adam(model.parameters(), lr=phase2_lr)
    P, gt = None, None
    best_iou_p2 = transition_iou
    
    phase2_checkpoints = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
    
    pbar = tqdm(range(phase2_epochs), desc=f"P2 {agg_mode}", leave=False)
    for epoch in pbar:
        if epoch % resample_every == 0 or P is None:
            if mesh_sdf is not None:
                points_np, occ_np = mesh_sdf.sample_points(num_samples, surface_ratio=0.3, near_ratio=0.5, near_std=0.03)
                P = torch.from_numpy(points_np).to(device)
                gt = torch.from_numpy(occ_np).to(device)
            else:
                P = torch.rand(num_samples, 3, device=device) * 2 - 1
                gt = target_func(P)
        
        pred = model(P)
        loss = loss_func(pred, gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics['phase2_losses'].append(loss.item())
        
        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                pred_full = torch.cat([model(b) for b in torch.split(eval_grid, 32**3)], dim=0)
                iou = compute_iou(pred_full, eval_gt)
                metrics['phase2_ious'].append(iou)
                if iou > best_iou_p2:
                    best_iou_p2 = iou
                    metrics['best_epoch'] = epoch
                    torch.save(model.state_dict(), os.path.join(output_folder, 'best_model.pt'))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'IoU': f'{iou:.3f}', 'best': f'{best_iou_p2:.3f}'})
        
        if (epoch + 1) in phase2_checkpoints:
            with torch.no_grad():
                pred_full = torch.cat([model(b) for b in torch.split(eval_grid, 32**3)], dim=0)
                iou = compute_iou(pred_full, eval_gt)
            metrics['phase2_iou_at_steps'][epoch + 1] = iou
            print(f"  P2 checkpoint {epoch+1}: IoU={iou:.4f}")
    
    # Final eval
    with torch.no_grad():
        pred_full = torch.cat([model(b) for b in torch.split(eval_grid, 32**3)], dim=0)
        final_iou = compute_iou(pred_full, eval_gt)
    metrics['final_iou'] = final_iou
    metrics['best_iou'] = best_iou_p2
    
    # Save final mesh
    try:
        marching_cubes_from_occupancy(os.path.join(output_folder, 'final.obj'), pred_full, resolution)
    except:
        pass
    
    # Save metrics
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
        json.dump(make_serializable(metrics), f, indent=2)
    
    print(f"\n  RESULT: {agg_mode} seed={seed}")
    print(f"  Transition IoU: {transition_iou:.4f}")
    print(f"  Final IoU:      {final_iou:.4f}")
    print(f"  Best IoU:       {best_iou_p2:.4f}")
    
    return metrics


# COMPARISON PLOTS
def plot_comparison(all_results: Dict[str, List[Dict]], output_folder: str, eval_every: int):
    """Generate comparison plots between aggregation modes."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Product Fuzzy vs Weighted Sum: Primitive Selection A/B Test', fontsize=14, fontweight='bold')
    
    colors = {'weighted_sum': '#2196F3', 'product_fuzzy': '#FF5722'}
    labels = {'weighted_sum': 'Weighted Sum (baseline)', 'product_fuzzy': 'Product Fuzzy (proposed)'}
    
    # --- Plot 1: Phase 1 IoU curves ---
    ax = axes[0, 0]
    for mode, runs in all_results.items():
        all_ious = [r['phase1_ious'] for r in runs]
        min_len = min(len(x) for x in all_ious)
        all_ious = np.array([x[:min_len] for x in all_ious])
        mean = all_ious.mean(axis=0)
        std = all_ious.std(axis=0)
        x = np.arange(len(mean)) * eval_every
        ax.plot(x, mean, color=colors[mode], label=labels[mode], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.2)
    ax.set_xlabel('Phase 1 Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Phase 1: Selection Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Phase 2 IoU curves ---
    ax = axes[0, 1]
    for mode, runs in all_results.items():
        all_ious = [r['phase2_ious'] for r in runs]
        min_len = min(len(x) for x in all_ious)
        all_ious = np.array([x[:min_len] for x in all_ious])
        mean = all_ious.mean(axis=0)
        std = all_ious.std(axis=0)
        x = np.arange(len(mean)) * eval_every
        ax.plot(x, mean, color=colors[mode], label=labels[mode], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.2)
    ax.set_xlabel('Phase 2 Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Phase 2: Convergence Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: Phase 2 Loss curves ---
    ax = axes[0, 2]
    for mode, runs in all_results.items():
        all_losses = [r['phase2_losses'] for r in runs]
        min_len = min(len(x) for x in all_losses)
        # Subsample for plotting
        step = max(1, min_len // 500)
        all_losses = np.array([x[:min_len:step] for x in all_losses])
        mean = all_losses.mean(axis=0)
        std = all_losses.std(axis=0)
        x = np.arange(len(mean)) * step
        ax.semilogy(x, mean, color=colors[mode], label=labels[mode], linewidth=2)
        ax.fill_between(x, np.clip(mean - std, 1e-6, None), mean + std, color=colors[mode], alpha=0.2)
    ax.set_xlabel('Phase 2 Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('Phase 2: Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 4: Parameter Drift (bar chart) ---
    ax = axes[1, 0]
    modes = list(all_results.keys())
    x_pos = np.arange(len(modes))
    width = 0.35
    
    selected_means = []
    nonsel_means = []
    selected_stds = []
    nonsel_stds = []
    
    for mode in modes:
        sel_drifts = [r['param_drift']['selected_drift_mean'] for r in all_results[mode]]
        nonsel_drifts = [r['param_drift']['non_selected_drift_mean'] for r in all_results[mode]]
        selected_means.append(np.mean(sel_drifts))
        nonsel_means.append(np.mean(nonsel_drifts))
        selected_stds.append(np.std(sel_drifts))
        nonsel_stds.append(np.std(nonsel_drifts))
    
    bars1 = ax.bar(x_pos - width/2, selected_means, width, yerr=selected_stds, 
                   label='Selected prims', color=[colors[m] for m in modes], alpha=0.8,
                   capsize=5)
    bars2 = ax.bar(x_pos + width/2, nonsel_means, width, yerr=nonsel_stds,
                   label='Non-selected prims', color=[colors[m] for m in modes], alpha=0.4,
                   capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[m].split(' (')[0] for m in modes])
    ax.set_ylabel('RMS Parameter Drift')
    ax.set_title('Parameter Drift During Phase 1')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 5: Key metrics comparison (bar chart) ---
    ax = axes[1, 1]
    metric_names = ['Transition\nIoU', 'Final\nIoU', 'Best\nIoU']
    x_pos = np.arange(len(metric_names))
    width = 0.35
    
    for i, mode in enumerate(modes):
        vals = [
            np.mean([r['transition_iou'] for r in all_results[mode]]),
            np.mean([r['final_iou'] for r in all_results[mode]]),
            np.mean([r['best_iou'] for r in all_results[mode]]),
        ]
        errs = [
            np.std([r['transition_iou'] for r in all_results[mode]]),
            np.std([r['final_iou'] for r in all_results[mode]]),
            np.std([r['best_iou'] for r in all_results[mode]]),
        ]
        offset = (i - 0.5) * width
        ax.bar(x_pos + offset, vals, width, yerr=errs, label=labels[mode],
               color=colors[mode], alpha=0.8, capsize=5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('IoU')
    ax.set_title('Key Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 6: Type distribution ---
    ax = axes[1, 2]
    all_types = list(all_results[modes[0]][0]['transition_type_counts'].keys())
    x_pos = np.arange(len(all_types))
    width = 0.35
    
    for i, mode in enumerate(modes):
        counts = []
        stds = []
        for t in all_types:
            vals = [r['transition_type_counts'].get(t, 0) for r in all_results[mode]]
            counts.append(np.mean(vals))
            stds.append(np.std(vals))
        offset = (i - 0.5) * width
        ax.bar(x_pos + offset, counts, width, yerr=stds, label=labels[mode],
               color=colors[mode], alpha=0.8, capsize=3)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([t.replace('_', '\n') for t in all_types], fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title('Primitive Type Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comparison.png'), dpi=150)
    plt.close()
    print(f"\nComparison plot saved: {os.path.join(output_folder, 'comparison.png')}")


# MAIN: A/B EXPERIMENT
def main():
    # EXPERIMENT CONFIGURATION
    MESH_PATH = "./bunny.obj"
    OUTPUT_ROOT = "./output_product_vs_weightedsum/"
    RESOLUTION = 128
    DEPTH = 6              # 2^6 = 64 leaves
    
    # Phase 1: Selection
    PHASE1_EPOCHS = 10000
    PHASE1_LR = 1e-3
    PHASE1_TEMP_START = 0.3
    PHASE1_TEMP_END = 2000.0
    
    # Phase 2: Refinement
    PHASE2_EPOCHS = 25000
    PHASE2_LR = 5e-4
    
    NUM_SAMPLES = 1024 * 100
    EVAL_EVERY = 100
    
    # Seeds for statistical significance
    SEEDS = [42, 123, 456]
    
    # Conditions to compare
    AGG_MODES = ['weighted_sum', 'product_fuzzy']
    
    primitive_types = ['block', 'sphere', 'cylinder', 'wedge', 'corner_wedge']
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print(f"{'='*60}")
    print("EXPERIMENT: Product Fuzzy vs Weighted Sum")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Depth: {DEPTH} ({2**DEPTH} leaves)")
    print(f"Phase 1: {PHASE1_EPOCHS} epochs")
    print(f"Phase 2: {PHASE2_EPOCHS} epochs")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {AGG_MODES}")
    print(f"{'='*60}")
    
    # Load target
    mesh_sdf = None
    if os.path.exists(MESH_PATH):
        print(f"\nLoading mesh: {MESH_PATH}")
        mesh_sdf = MeshSDF(MESH_PATH, normalize=True, margin=0.1)
        target_func = create_mesh_target_func(mesh_sdf)
        # Save ground truth once
        gt_sdf_grid = mesh_sdf.query_sdf_grid(RESOLUTION)
        marching_cubes_from_sdf(os.path.join(OUTPUT_ROOT, "00_ground_truth.obj"), gt_sdf_grid, RESOLUTION)
    else:
        print(f"\nMesh not found: {MESH_PATH}")
        print("Using synthetic target (complex CSG)")
        target_func = create_target_complex_csg
        # Save ground truth
        eval_grid = sample_3D_grid(RESOLUTION, -1, 1)
        gt_occ = target_func(eval_grid)
        marching_cubes_from_occupancy(os.path.join(OUTPUT_ROOT, "00_ground_truth.obj"), gt_occ, RESOLUTION)
    
    # Run all conditions
    all_results = {mode: [] for mode in AGG_MODES}
    
    for seed in SEEDS:
        for mode in AGG_MODES:
            run_folder = os.path.join(OUTPUT_ROOT, f"{mode}_seed{seed}")
            
            t0 = time.time()
            metrics = train_single_run(
                agg_mode=mode,
                target_func=target_func,
                mesh_sdf=mesh_sdf,
                seed=seed,
                depth=DEPTH,
                primitive_types=primitive_types,
                phase1_epochs=PHASE1_EPOCHS,
                phase1_lr=PHASE1_LR,
                phase1_temp_start=PHASE1_TEMP_START,
                phase1_temp_end=PHASE1_TEMP_END,
                phase2_epochs=PHASE2_EPOCHS,
                phase2_lr=PHASE2_LR,
                num_samples=NUM_SAMPLES,
                resample_every=10,
                eval_every=EVAL_EVERY,
                resolution=RESOLUTION,
                output_folder=run_folder,
            )
            metrics['wall_time'] = time.time() - t0
            all_results[mode].append(metrics)
            
            print(f"\n  Wall time: {metrics['wall_time']:.1f}s")
    
    # SUMMARY TABLE
    print(f"\n\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Weighted Sum':>18} {'Product Fuzzy':>18}")
    print(f"{'-'*70}")
    
    for metric_name, key in [
        ('Transition IoU', 'transition_iou'),
        ('Final IoU', 'final_iou'),
        ('Best IoU', 'best_iou'),
    ]:
        ws_vals = [r[key] for r in all_results['weighted_sum']]
        pf_vals = [r[key] for r in all_results['product_fuzzy']]
        print(f"{metric_name:<30} {np.mean(ws_vals):>7.4f} ± {np.std(ws_vals):.4f}  {np.mean(pf_vals):>7.4f} ± {np.std(pf_vals):.4f}")
    
    # Drift
    for drift_key, label in [
        ('selected_drift_mean', 'Selected Prim Drift'),
        ('non_selected_drift_mean', 'Non-Selected Prim Drift'),
    ]:
        ws_vals = [r['param_drift'][drift_key] for r in all_results['weighted_sum']]
        pf_vals = [r['param_drift'][drift_key] for r in all_results['product_fuzzy']]
        print(f"{label:<30} {np.mean(ws_vals):>7.4f} ± {np.std(ws_vals):.4f}  {np.mean(pf_vals):>7.4f} ± {np.std(pf_vals):.4f}")
    
    # Phase 2 convergence at checkpoints
    print(f"\n{'Phase 2 IoU at step:':<30}")
    for step in [500, 1000, 2000, 5000, 10000]:
        ws_vals = [r['phase2_iou_at_steps'].get(step, 0) for r in all_results['weighted_sum']]
        pf_vals = [r['phase2_iou_at_steps'].get(step, 0) for r in all_results['product_fuzzy']]
        if any(v > 0 for v in ws_vals + pf_vals):
            print(f"  Step {step:<24} {np.mean(ws_vals):>7.4f} ± {np.std(ws_vals):.4f}  {np.mean(pf_vals):>7.4f} ± {np.std(pf_vals):.4f}")
    
    # Wall time
    ws_times = [r['wall_time'] for r in all_results['weighted_sum']]
    pf_times = [r['wall_time'] for r in all_results['product_fuzzy']]
    print(f"\n{'Wall Time (s)':<30} {np.mean(ws_times):>7.1f} ± {np.std(ws_times):.1f}    {np.mean(pf_times):>7.1f} ± {np.std(pf_times):.1f}")
    
    print(f"{'='*70}")
    
    # Generate comparison plot
    plot_comparison(all_results, OUTPUT_ROOT, EVAL_EVERY)
    
    # Save full results
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open(os.path.join(OUTPUT_ROOT, 'full_results.json'), 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    
    print(f"\nAll results saved to: {OUTPUT_ROOT}")
    print("Done!")


if __name__ == "__main__":
    main()
