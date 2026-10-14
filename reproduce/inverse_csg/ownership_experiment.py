"""
Ownership-weighted per-leaf auxiliary loss for CSG primitive type selection.

Idea
----
The Boolean operations define an "ownership" mask for each leaf:
under union, du/dx = 1-y (leaf x matters where its sibling is absent);
under intersection, di/dx = y (leaf x matters where its sibling is present).

These masks are the Boolean derivatives already computed in the forward and
backward passes. Weighting the per-leaf auxiliary losses by ownership
evaluates each primitive type only where it controls the output boundary.
Without it, a leaf's SDF is compared with the whole target SDF, although each
leaf explains only about 1/K of the boundary (the limitation of the per-leaf
metrics in metric_experiment_v2.py).

Conditions
----------
1. baseline        - occupancy MSE only
2. uniform_sdf     - Per-leaf SDF matching, uniform weight (= metric_v2's hybrid_sdf)
3. ownership_sdf   - Per-leaf SDF matching, weighted by ownership mask
4. ownership_both  - Ownership-weighted SDF + normal matching

Usage:
    python ownership_experiment.py
    python ownership_experiment.py --mesh bunny.obj
    python ownership_experiment.py --mesh bunny.obj --depth 6
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Callable

# Optional mesh SDF
try:
    import trimesh
    import mcubes
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# SDF PRIMITIVES
MIN_SIZE = 0.02

def euler_to_rotation_matrix(angles):
    cx, sx = torch.cos(angles[0]), torch.sin(angles[0])
    cy, sy = torch.cos(angles[1]), torch.sin(angles[1])
    cz, sz = torch.cos(angles[2]), torch.sin(angles[2])
    R = torch.zeros(3, 3, device=angles.device, dtype=angles.dtype)
    R[0,0]=cy*cz; R[0,1]=cz*sx*sy-cx*sz; R[0,2]=cx*cz*sy+sx*sz
    R[1,0]=cy*sz; R[1,1]=cx*cz+sx*sy*sz; R[1,2]=cx*sy*sz-cz*sx
    R[2,0]=-sy;   R[2,1]=cy*sx;           R[2,2]=cx*cy
    return R

def sdf_block(p, center, half_size, rotation):
    pl = torch.matmul(p - center, rotation.T)
    d = torch.abs(pl) - half_size
    return torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(-1)[0], max=0)

def sdf_sphere(p, center, radius):
    return torch.sqrt(((p - center)**2).sum(-1) + 1e-8) - radius

def sdf_cylinder(p, center, radius, half_height, rotation):
    pl = torch.matmul(p - center, rotation.T)
    dr = torch.sqrt(pl[:,0]**2 + pl[:,2]**2 + 1e-8) - radius
    dh = torch.abs(pl[:,1]) - half_height
    return torch.sqrt(torch.clamp(dr,min=0)**2 + torch.clamp(dh,min=0)**2 + 1e-8) + torch.clamp(torch.maximum(dr,dh), max=0)

def sdf_wedge(p, center, half_size, rotation):
    pl = torch.matmul(p - center, rotation.T)
    hy, hz = half_size[1], half_size[2]
    d = torch.abs(pl) - half_size
    box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(-1)[0], max=0)
    inv = 1.0 / torch.sqrt(1.0/(hy**2) + 1.0/(hz**2) + 1e-8)
    return torch.maximum(box, (pl[:,1]/hy + pl[:,2]/hz - 1.0) * inv)

def sdf_corner_wedge(p, center, half_size, rotation):
    pl = torch.matmul(p - center, rotation.T)
    hx, hy, hz = half_size[0], half_size[1], half_size[2]
    d = torch.abs(pl) - half_size
    box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1) + 1e-8) + torch.clamp(d.max(-1)[0], max=0)
    i1 = 1.0/torch.sqrt(1.0/(hx**2)+1.0/(hy**2)+1e-8)
    i2 = 1.0/torch.sqrt(1.0/(hz**2)+1.0/(hy**2)+1e-8)
    p1 = (pl[:,0]/hx + pl[:,1]/hy - 1.0) * i1
    p2 = (pl[:,2]/hz + pl[:,1]/hy - 1.0) * i2
    return torch.maximum(torch.maximum(box, p1), p2)


# PRIMITIVE POOL
class PrimitivePool(nn.Module):
    PARAM_COUNTS = {'block':9, 'sphere':4, 'cylinder':8, 'wedge':9, 'corner_wedge':9}
    
    def __init__(self, primitive_types, init_scale=0.5):
        super().__init__()
        self.types = primitive_types
        self.num_primitives = len(primitive_types)
        total = sum(self.PARAM_COUNTS[t] for t in primitive_types)
        self.params = nn.Parameter(torch.randn(total) * init_scale)
        self.sharpness = nn.Parameter(torch.ones(self.num_primitives) * 20.0)
        self.param_indices = []
        idx = 0
        for t in primitive_types:
            c = self.PARAM_COUNTS[t]
            self.param_indices.append((idx, idx+c))
            idx += c
    
    def get_sdf(self, p, prim_idx):
        pt = self.types[prim_idx]
        s, e = self.param_indices[prim_idx]
        par = self.params[s:e]
        if pt == 'block':
            return sdf_block(p, par[:3], par[3:6].abs()+MIN_SIZE, euler_to_rotation_matrix(par[6:9]))
        elif pt == 'sphere':
            return sdf_sphere(p, par[:3], par[3].abs()+MIN_SIZE)
        elif pt == 'cylinder':
            return sdf_cylinder(p, par[:3], par[3].abs()+MIN_SIZE, par[4].abs()+MIN_SIZE, euler_to_rotation_matrix(par[5:8]))
        elif pt == 'wedge':
            return sdf_wedge(p, par[:3], par[3:6].abs()+MIN_SIZE, euler_to_rotation_matrix(par[6:9]))
        elif pt == 'corner_wedge':
            return sdf_corner_wedge(p, par[:3], par[3:6].abs()+MIN_SIZE, euler_to_rotation_matrix(par[6:9]))
    
    def get_occupancy(self, p, prim_idx):
        sdf = self.get_sdf(p, prim_idx)
        s = self.sharpness[prim_idx].abs().clamp(min=1.0)
        return torch.sigmoid(-sdf * s)
    
    def get_all_occupancies(self, p):
        return torch.stack([self.get_occupancy(p, i) for i in range(self.num_primitives)], dim=-1)
    
    def get_all_sdfs(self, p):
        return torch.stack([self.get_sdf(p, i) for i in range(self.num_primitives)], dim=-1)


# SELECTING LEAF
class SelectingLeaf(nn.Module):
    def __init__(self, num_primitives, omega=10.0, init_scale=0.3):
        super().__init__()
        self.num_primitives = num_primitives
        self.omega = omega
        self.selection_logits = nn.Parameter(torch.randn(num_primitives) * init_scale)
        self.fixed_idx = None
    
    def get_selection_weights(self, temperature=1000.0):
        logits = torch.sin(self.omega * self.selection_logits)
        return F.softmax(logits * temperature, dim=-1)
    
    def get_selected_idx(self):
        return self.get_selection_weights(0.01).argmax().item()
    
    def forward(self, occupancies, temperature=1.0):
        if self.fixed_idx is not None:
            return occupancies[:, self.fixed_idx]
        weights = self.get_selection_weights(temperature)
        return (occupancies * weights).sum(dim=-1)
    
    def fix_selection(self):
        self.fixed_idx = self.get_selected_idx()
        return self.fixed_idx


# CSG TREE WITH OWNERSHIP COMPUTATION
class OwnershipCSGTree(nn.Module):
    """
    CSG tree that can compute per-leaf ownership masks.
    
    Ownership mask for leaf i = ∂(tree_output)/∂(leaf_i_output)
    computed via top-down chain rule through boolean nodes.
    
    At each boolean node with output = (c1+c2)*L + (c1+c3)*R + (c0-c1-c2-c3)*L*R:
        ∂out/∂L = (c1+c2) + (c0-c1-c2-c3)*R   [matters where R is low for union]
        ∂out/∂R = (c1+c3) + (c0-c1-c2-c3)*L   [matters where L is low for union]
    """
    
    def __init__(self, depth=4, primitive_types=None, init_scale=0.3, omega=10.0):
        super().__init__()
        self.depth = depth
        self.omega = omega
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
        
        self.leaves = nn.ModuleList([
            SelectingLeaf(self.n_types, omega, init_scale)
            for _ in range(self.num_leaves)
        ])
        
        self.boolean_params = nn.Parameter(torch.randn(self.num_internal, 4) * init_scale)
        self._selected_global_indices = None
    
    def get_boolean_weights(self):
        return F.softmax(torch.sin(self.omega * self.boolean_params) * 1000.0, dim=-1)
    
    def _boolean_combine(self, left, right, weights):
        c0 = weights[:, 0:1]; c1 = weights[:, 1:2]
        c2 = weights[:, 2:3]; c3 = weights[:, 3:4]
        return (c1 + c2) * left + (c1 + c3) * right + (c0 - c1 - c2 - c3) * left * right
    
    def forward(self, p):
        """Standard forward pass."""
        if self.phase == 2 and self._selected_global_indices is not None:
            occs = self.primitive_pool.get_all_occupancies(p)
            selected = torch.stack([occs[:, idx] for idx in self._selected_global_indices], dim=0)
            leaf_outputs = [selected[i] for i in range(self.num_leaves)]
            return self._tree_combine(leaf_outputs)
        
        all_occs = self.primitive_pool.get_all_occupancies(p)
        leaf_outputs = []
        for i, leaf in enumerate(self.leaves):
            start, end = i * self.n_types, (i + 1) * self.n_types
            leaf_outputs.append(leaf(all_occs[:, start:end], self.temperature))
        return self._tree_combine(leaf_outputs)
    
    def forward_with_ownership(self, p):
        """
        Forward pass that also computes ownership masks for each leaf.
        
        Returns: (tree_output, ownership_masks, all_sdfs)
            tree_output: [N] occupancy
            ownership_masks: [num_leaves, N] per-leaf ownership weight
            all_sdfs: [N, total_primitives] all primitive SDFs
        """
        all_occs = self.primitive_pool.get_all_occupancies(p)
        all_sdfs = self.primitive_pool.get_all_sdfs(p)
        
        # Compute leaf outputs
        leaf_outputs = []
        for i, leaf in enumerate(self.leaves):
            start, end = i * self.n_types, (i + 1) * self.n_types
            leaf_outputs.append(leaf(all_occs[:, start:end], self.temperature))
        
        # Bottom-up tree combination, recording intermediate values
        N = p.shape[0]
        current = torch.stack(leaf_outputs, dim=0)  # [num_leaves, N]
        bool_weights = self.get_boolean_weights()
        
        # Store (left_vals, right_vals, bw) at each level for ownership computation
        level_data = []
        bool_idx = self.num_internal - 1
        for level in range(self.depth - 1, -1, -1):
            num_nodes = 2 ** level
            left, right = current[0::2], current[1::2]
            bw = bool_weights[bool_idx - num_nodes + 1:bool_idx + 1]
            # Detach for ownership computation (ownership is a mask, not a gradient target)
            level_data.append((left.detach(), right.detach(), bw.detach()))
            current = self._boolean_combine(left, right, bw)
            bool_idx -= num_nodes
        
        tree_output = current[0]
        
        # Top-down ownership computation
        # level_data is stored bottom-first, we need top-first
        level_data = level_data[::-1]  # Now: [root_level, ..., leaf_level]
        
        ownership = torch.ones(1, N, device=p.device)  # Root has ownership 1.0
        
        for left_vals, right_vals, bw in level_data:
            # bw: [num_nodes, 4]
            # left_vals, right_vals: [num_nodes, N]
            c0 = bw[:, 0:1]  # [num_nodes, 1]
            c1 = bw[:, 1:2]
            c2 = bw[:, 2:3]
            c3 = bw[:, 3:4]
            
            # Boolean derivatives
            # output = (c1+c2)*L + (c1+c3)*R + (c0-c1-c2-c3)*L*R
            # ∂out/∂L = (c1+c2) + (c0-c1-c2-c3)*R
            # ∂out/∂R = (c1+c3) + (c0-c1-c2-c3)*L
            d_left = (c1 + c2) + (c0 - c1 - c2 - c3) * right_vals   # [num_nodes, N]
            d_right = (c1 + c3) + (c0 - c1 - c2 - c3) * left_vals   # [num_nodes, N]
            
            # Propagate ownership to children
            left_ownership = ownership * d_left    # [num_nodes, N]
            right_ownership = ownership * d_right  # [num_nodes, N]
            
            # Interleave: [left_0, right_0, left_1, right_1, ...]
            new_ownership = torch.zeros(2 * ownership.shape[0], N, device=p.device)
            new_ownership[0::2] = left_ownership
            new_ownership[1::2] = right_ownership
            ownership = new_ownership
        
        # ownership is now [num_leaves, N]
        return tree_output, ownership, all_sdfs
    
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
    
    def transition_to_phase2(self):
        self._selected_global_indices = []
        type_counts = {t: 0 for t in self.primitive_types}
        for i, leaf in enumerate(self.leaves):
            local_idx = leaf.fix_selection()
            global_idx = i * self.n_types + local_idx
            ptype = self.primitive_types[local_idx]
            self._selected_global_indices.append(global_idx)
            type_counts[ptype] += 1
        self.phase = 2
        return type_counts
    
    def get_type_distribution(self):
        types = []
        for i, leaf in enumerate(self.leaves):
            if leaf.fixed_idx is not None:
                idx = leaf.fixed_idx
            else:
                idx = leaf.get_selected_idx()
            types.append(self.primitive_types[idx])
        return types


# SYNTHETIC TARGET (differentiable SDF)
class SyntheticTarget:
    """Box + sphere - 2 cylinders."""
    def sdf(self, p):
        d = torch.abs(p) - 0.5
        box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8) + torch.clamp(d.max(-1)[0],max=0)
        sc = torch.tensor([0.3,0.3,0.3], device=p.device)
        sph = torch.sqrt(((p-sc)**2).sum(-1)+1e-8) - 0.35
        union = torch.minimum(box, sph)
        cy = torch.sqrt(p[:,0]**2+p[:,2]**2+1e-8) - 0.15
        cx = torch.sqrt(p[:,1]**2+p[:,2]**2+1e-8) - 0.15
        return torch.maximum(torch.maximum(union, -cy), -cx)
    
    def occupancy(self, p):
        return (self.sdf(p) < 0).float()
    
    def sdf_gradient(self, p):
        pg = p.detach().requires_grad_(True)
        s = self.sdf(pg)
        g = torch.autograd.grad(s.sum(), pg, create_graph=False)[0]
        return g.detach()
    
    def sample_points(self, n, surface_ratio=0.3, near_ratio=0.5, near_std=0.03):
        ns = int(n * surface_ratio); nn_ = int(n * near_ratio); nr = n - ns - nn_
        cands = torch.rand(n*10, 3, device=device)*2-1
        sdf_vals = self.sdf(cands).abs()
        _, si = sdf_vals.topk(ns, largest=False)
        pts_s = cands[si]
        _, ni = sdf_vals.topk(nn_, largest=False)
        pts_n = cands[ni] + torch.randn(nn_, 3, device=device) * near_std
        pts_n = pts_n.clamp(-0.99, 0.99)
        pts_r = torch.rand(nr, 3, device=device)*2-1
        P = torch.cat([pts_s, pts_n, pts_r], 0)
        P = P[torch.randperm(P.shape[0])]
        return P, self.occupancy(P)


# MESH TARGET WRAPPER
class MeshTarget:
    """Wraps mesh to provide sdf(), sample_points(), sdf_gradient().
    Includes a mesh repair pipeline (pymeshfix, pymeshlab, voxelization fallback)."""
    def __init__(self, mesh_path):
        self.mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate([g for g in self.mesh.geometry.values()])
        
        # Repair if not watertight
        if not self.mesh.is_watertight:
            self.mesh = self._repair_mesh(self.mesh)
        
        # Normalize to [-0.9, 0.9]
        c = self.mesh.bounding_box.centroid
        self.mesh.vertices -= c
        s = 0.9 / np.abs(self.mesh.vertices).max()
        self.mesh.vertices *= s
        self.vertices = np.ascontiguousarray(self.mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.mesh.faces, dtype=np.int64)
        print(f"Mesh: {len(self.vertices)} verts, {len(self.faces)} faces, "
              f"bounds [{self.vertices.min():.2f}, {self.vertices.max():.2f}], "
              f"watertight={self.mesh.is_watertight}")
    
    def _repair_mesh(self, mesh):
        print("  Mesh not watertight. Attempting repair...")
        # Method 1: pymeshfix
        try:
            import pymeshfix
            print("  Trying PyMeshFix...")
            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair(verbose=False)
            verts = meshfix.mesh.points
            faces_raw = meshfix.mesh.faces
            faces = faces_raw.reshape(-1, 4)[:, 1:4] if faces_raw.ndim == 1 else faces_raw
            repaired = trimesh.Trimesh(vertices=verts, faces=faces)
            if repaired.is_watertight:
                print("  ✓ PyMeshFix repair successful!")
                return repaired
        except ImportError:
            print("  PyMeshFix not installed")
        except Exception as e:
            print(f"  PyMeshFix failed: {e}")
        
        # Method 2: pymeshlab
        try:
            import pymeshlab
            print("  Trying PyMeshLab...")
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_close_holes(maxholesize=1000)
            ms.meshing_repair_non_manifold_edges()
            m = ms.current_mesh()
            repaired = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
            if repaired.is_watertight:
                print("  ✓ PyMeshLab repair successful!")
                return repaired
        except ImportError:
            print("  PyMeshLab not installed")
        except Exception as e:
            print(f"  PyMeshLab failed: {e}")
        
        # Method 3: voxelization fallback
        try:
            from scipy import ndimage
            print("  Trying voxelization repair...")
            resolution = 128
            voxels = mesh.voxelized(pitch=2.0 / resolution)
            voxel_grid = ndimage.binary_fill_holes(voxels.matrix.astype(np.float32)).astype(np.float32)
            voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.3)
            voxel_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
            vertices, triangles = mcubes.marching_cubes(voxel_grid, 0.5)
            vertices = (vertices - 1) / resolution * (mesh.vertices.max() - mesh.vertices.min()) + mesh.vertices.min()
            repaired = trimesh.Trimesh(vertices=vertices, faces=triangles)
            print(f"  ✓ Voxelization repair (watertight={repaired.is_watertight})")
            return repaired
        except Exception as e:
            print(f"  Voxelization failed: {e}")
        
        print("  ⚠ All repairs failed. Using original mesh.")
        return mesh
    
    def _query_np(self, pts_np):
        pts_np = np.ascontiguousarray(pts_np, dtype=np.float64)
        if HAS_IGL:
            sdf, _, _, _ = igl.signed_distance(pts_np, self.vertices, self.faces)
            return sdf.astype(np.float32)
        cl, dist, _ = self.mesh.nearest.on_surface(pts_np)
        inside = self.mesh.contains(pts_np)
        sdf = dist.copy(); sdf[inside] *= -1
        return sdf.astype(np.float32)
    
    def sdf(self, p):
        pts_np = p.detach().cpu().numpy()
        sdf_np = self._query_np(pts_np)
        return torch.from_numpy(sdf_np).to(p.device)
    
    def occupancy(self, p):
        return (self.sdf(p) < 0).float()
    
    def sdf_gradient(self, p):
        """Finite-difference gradient of SDF."""
        eps = 0.003
        pts = p.detach().cpu().numpy()
        grad = np.zeros_like(pts)
        for dim in range(3):
            offset = np.zeros_like(pts); offset[:, dim] = eps
            grad[:, dim] = (self._query_np(pts+offset) - self._query_np(pts-offset)) / (2*eps)
        return torch.from_numpy(grad.astype(np.float32)).to(p.device)
    
    def sample_points(self, n, surface_ratio=0.3, near_ratio=0.5, near_std=0.03):
        ns = int(n*surface_ratio); nn_ = int(n*near_ratio); nr = n-ns-nn_
        ps, _ = trimesh.sample.sample_surface(self.mesh, ns)
        pn, _ = trimesh.sample.sample_surface(self.mesh, nn_)
        pn = np.clip(pn + np.random.randn(nn_, 3)*near_std, -0.99, 0.99)
        pr = np.random.uniform(-1, 1, (nr, 3))
        pts = np.vstack([ps, pn, pr]).astype(np.float32)
        occ = (self._query_np(pts) < 0).astype(np.float32)
        perm = np.random.permutation(len(pts))
        P = torch.from_numpy(pts[perm]).to(device)
        gt = torch.from_numpy(occ[perm]).to(device)
        return P, gt


# IoU
def compute_iou(pred, gt, threshold=0.5):
    p = (pred > threshold).float(); g = (gt > threshold).float()
    inter = (p * g).sum(); union = ((p + g) > 0).float().sum()
    return (inter / (union + 1e-8)).item()


# METRIC FUNCTIONS

def metric_baseline(model, P, gt_occ, target, surf_pts, **kw):
    """Standard MSE on occupancy through tree. No auxiliary loss."""
    pred = model(P)
    return F.mse_loss(pred, gt_occ)


def metric_uniform_sdf(model, P, gt_occ, target, surf_pts, **kw):
    """Per-leaf SDF matching with UNIFORM weight (no ownership).
    This is equivalent to hybrid_sdf from metric_v2."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    # Per-leaf SDF at surface points (uniform weight)
    all_sdfs = model.primitive_pool.get_all_sdfs(surf_pts)
    gt_sdf = target.sdf(surf_pts).detach().clamp(-0.5, 0.5)
    
    leaf_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        weighted_sdf = (all_sdfs[:, start:end] * weights).sum(dim=-1)
        leaf_loss = leaf_loss + F.mse_loss(weighted_sdf.clamp(-0.5, 0.5), gt_sdf)
    
    leaf_loss = leaf_loss / model.num_leaves
    return tree_loss + 0.1 * leaf_loss


def metric_ownership_sdf(model, P, gt_occ, target, surf_pts, **kw):
    """Per-leaf SDF matching weighted by OWNERSHIP masks.
    Each leaf's SDF is only compared to target where that leaf matters."""
    # Tree output + ownership
    tree_out, ownership, all_sdfs_full = model.forward_with_ownership(P)
    tree_loss = F.mse_loss(tree_out, gt_occ)
    
    # Compute per-leaf SDF at surface points
    all_sdfs_surf = model.primitive_pool.get_all_sdfs(surf_pts)
    gt_sdf_surf = target.sdf(surf_pts).detach().clamp(-0.5, 0.5)
    
    # Also need ownership at surface points
    # Run ownership computation on surface points
    _, ownership_surf, _ = model.forward_with_ownership(surf_pts)
    ownership_surf = ownership_surf.detach().abs()  # [num_leaves, n_surf]
    
    # Normalize ownership per leaf so it sums to ~1 (relative weighting)
    ownership_surf = ownership_surf / (ownership_surf.sum(dim=1, keepdim=True) + 1e-8) * ownership_surf.shape[1]
    
    leaf_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        weighted_sdf = (all_sdfs_surf[:, start:end] * weights).sum(dim=-1)
        
        # Ownership-weighted SDF error
        own_w = ownership_surf[i]  # [n_surf]
        err = (weighted_sdf.clamp(-0.5, 0.5) - gt_sdf_surf) ** 2
        leaf_loss = leaf_loss + (own_w * err).mean()
    
    leaf_loss = leaf_loss / model.num_leaves
    return tree_loss + 0.1 * leaf_loss


def metric_ownership_both(model, P, gt_occ, target, surf_pts, **kw):
    """Ownership-weighted SDF + normal matching per leaf."""
    tree_out, ownership, all_sdfs_full = model.forward_with_ownership(P)
    tree_loss = F.mse_loss(tree_out, gt_occ)
    
    all_sdfs_surf = model.primitive_pool.get_all_sdfs(surf_pts)
    gt_sdf_surf = target.sdf(surf_pts).detach().clamp(-0.5, 0.5)
    
    _, ownership_surf, _ = model.forward_with_ownership(surf_pts)
    ownership_surf = ownership_surf.detach().abs()
    ownership_surf = ownership_surf / (ownership_surf.sum(dim=1, keepdim=True) + 1e-8) * ownership_surf.shape[1]
    
    # Target normals at surface points
    gt_normals = target.sdf_gradient(surf_pts)
    gt_normals = gt_normals / (gt_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Finite-diff normals for each leaf's weighted SDF
    eps = 0.003
    dx = torch.zeros_like(surf_pts); dx[:, 0] = eps
    dy = torch.zeros_like(surf_pts); dy[:, 1] = eps
    dz = torch.zeros_like(surf_pts); dz[:, 2] = eps
    all_sdfs_7 = [all_sdfs_surf]
    for offset in [dx, -dx, dy, -dy, dz, -dz]:
        all_sdfs_7.append(model.primitive_pool.get_all_sdfs(surf_pts + offset))
    
    sdf_loss = 0
    normal_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        own_w = ownership_surf[i]
        
        # SDF loss
        weighted_sdf = (all_sdfs_7[0][:, start:end] * weights).sum(dim=-1)
        sdf_err = (weighted_sdf.clamp(-0.5, 0.5) - gt_sdf_surf) ** 2
        sdf_loss = sdf_loss + (own_w * sdf_err).mean()
        
        # Normal loss via finite differences
        def wsdf(idx):
            return (all_sdfs_7[idx][:, start:end] * weights).sum(dim=-1)
        gx = (wsdf(1) - wsdf(2)) / (2*eps)
        gy = (wsdf(3) - wsdf(4)) / (2*eps)
        gz = (wsdf(5) - wsdf(6)) / (2*eps)
        pred_n = torch.stack([gx, gy, gz], dim=-1)
        pred_n = pred_n / (pred_n.norm(dim=-1, keepdim=True) + 1e-8)
        
        cos_sim = F.cosine_similarity(pred_n, gt_normals, dim=-1)
        normal_loss = normal_loss + (own_w * (1 - cos_sim)).mean()
    
    sdf_loss = sdf_loss / model.num_leaves
    normal_loss = normal_loss / model.num_leaves
    
    return tree_loss + 0.1 * sdf_loss + 0.1 * normal_loss


# TRAINING
def train_one_config(metric_name, metric_fn, seed, target,
                     depth=4, phase1_epochs=1500, phase2_epochs=3000,
                     num_samples=8192, n_surface=1024,
                     eval_resolution=64, phase1_lr=1e-3, phase2_lr=5e-4,
                     temp_start=0.3, temp_end=2000.0,
                     resample_every=10, eval_every=300):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = OwnershipCSGTree(depth=depth, init_scale=0.3, omega=10.0).to(device)
    
    idx = torch.linspace(-1, 1, eval_resolution, device=device)
    x, y, z = torch.meshgrid(idx, idx, idx, indexing='ij')
    eval_grid = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    eval_gt = target.occupancy(eval_grid).detach()
    
    results = {
        'metric': metric_name, 'seed': seed,
        'phase1_ious': [], 'phase2_ious': [],
        'type_distribution': {}, 'type_selections': [],
        'transition_iou': 0, 'final_iou': 0, 'best_iou': 0, 'total_time': 0,
    }
    
    t_start = time.time()
    
    # Phase 1
    optimizer = torch.optim.Adam(model.parameters(), lr=phase1_lr)
    P, gt_occ, surf_pts = None, None, None
    
    for epoch in range(phase1_epochs):
        progress = epoch / max(phase1_epochs - 1, 1)
        model.temperature = temp_start * (temp_end / temp_start) ** progress
        
        if epoch % resample_every == 0 or P is None:
            P, gt_occ = target.sample_points(num_samples)
            if isinstance(P, np.ndarray):
                P = torch.from_numpy(P).to(device)
                gt_occ = torch.from_numpy(gt_occ).to(device)
            P, gt_occ = P.detach(), gt_occ.detach()
            
            # Sample near-surface points for auxiliary loss
            gt_sdf_all = target.sdf(P).detach()
            _, near_idx = gt_sdf_all.abs().topk(n_surface, largest=False)
            surf_pts = P[near_idx].detach()
        
        loss = metric_fn(model, P, gt_occ, target, surf_pts)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                pred = torch.cat([model(b) for b in eval_grid.split(8192)], 0)
                iou = compute_iou(pred, eval_gt)
            results['phase1_ious'].append(iou)
    
    # Record selections
    types = model.get_type_distribution()
    results['type_selections'] = types
    for t in model.primitive_types:
        results['type_distribution'][t] = types.count(t)
    
    with torch.no_grad():
        pred = torch.cat([model(b) for b in eval_grid.split(8192)], 0)
        results['transition_iou'] = compute_iou(pred, eval_gt)
    
    # Phase 2 (always plain MSE)
    model.transition_to_phase2()
    optimizer = torch.optim.Adam(model.parameters(), lr=phase2_lr)
    best_iou = results['transition_iou']
    
    for epoch in range(phase2_epochs):
        if epoch % resample_every == 0 or P is None:
            P, gt_occ = target.sample_points(num_samples)
            if isinstance(P, np.ndarray):
                P = torch.from_numpy(P).to(device)
                gt_occ = torch.from_numpy(gt_occ).to(device)
            P, gt_occ = P.detach(), gt_occ.detach()
        
        pred = model(P)
        loss = F.mse_loss(pred, gt_occ)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                pred = torch.cat([model(b) for b in eval_grid.split(8192)], 0)
                iou = compute_iou(pred, eval_gt)
            results['phase2_ious'].append(iou)
            if iou > best_iou:
                best_iou = iou
    
    results['final_iou'] = results['phase2_ious'][-1] if results['phase2_ious'] else results['transition_iou']
    results['best_iou'] = best_iou
    results['total_time'] = time.time() - t_start
    return results


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=None, help='Path to target mesh')
    parser.add_argument('--depth', type=int, default=4, help='Tree depth')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123], help='Random seeds')
    parser.add_argument('--phase1', type=int, default=1500, help='Phase 1 epochs')
    parser.add_argument('--phase2', type=int, default=3000, help='Phase 2 epochs')
    args = parser.parse_args()
    
    # Setup target
    if args.mesh and os.path.exists(args.mesh):
        target = MeshTarget(args.mesh)
    else:
        target = SyntheticTarget()
        print("Using synthetic target (box + sphere - 2 cylinders)")
    
    metrics = {
        'baseline':       metric_baseline,
        'uniform_sdf':    metric_uniform_sdf,
        'ownership_sdf':  metric_ownership_sdf,
        'ownership_both': metric_ownership_both,
    }
    
    seeds = args.seeds
    total_runs = len(metrics) * len(seeds)
    
    print(f"\n{'='*70}")
    print("OWNERSHIP EXPERIMENT: Does boundary ownership help type selection?")
    print(f"{'='*70}")
    print(f"  Metrics: {list(metrics.keys())}")
    print(f"  Seeds: {seeds}")
    print(f"  Depth: {args.depth} ({2**args.depth} leaves)")
    print(f"  Phase 1: {args.phase1}, Phase 2: {args.phase2}")
    print(f"  Total runs: {total_runs}")
    print()
    
    all_results = []
    
    for mi, (name, fn) in enumerate(metrics.items()):
        for si, seed in enumerate(seeds):
            run_idx = mi * len(seeds) + si + 1
            print(f"[{run_idx}/{total_runs}] {name} (seed={seed})", end=" ", flush=True)
            
            try:
                result = train_one_config(
                    metric_name=name, metric_fn=fn, seed=seed, target=target,
                    depth=args.depth,
                    phase1_epochs=args.phase1, phase2_epochs=args.phase2,
                )
                all_results.append(result)
                print(f"→ trans={result['transition_iou']:.4f} "
                      f"final={result['final_iou']:.4f} "
                      f"best={result['best_iou']:.4f} "
                      f"t={result['total_time']:.0f}s "
                      f"types={result['type_distribution']}")
            except Exception as e:
                print(f"→ FAILED: {e}")
                import traceback; traceback.print_exc()
    
    # SUMMARY
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    by_metric = defaultdict(list)
    for r in all_results:
        by_metric[r['metric']].append(r)
    
    baseline_best = 0
    print(f"\n{'Metric':<22s} {'Trans':>8s} {'Final':>8s} {'Best':>8s} {'Time':>7s}  Types")
    print("-"*90)
    
    for name in metrics.keys():
        if name not in by_metric: continue
        runs = by_metric[name]
        trans = np.mean([r['transition_iou'] for r in runs])
        final = np.mean([r['final_iou'] for r in runs])
        best = np.mean([r['best_iou'] for r in runs])
        t = np.mean([r['total_time'] for r in runs])
        if name == 'baseline': baseline_best = best
        
        agg = defaultdict(int)
        for r in runs:
            for typ, cnt in r['type_distribution'].items():
                agg[typ] += cnt
        type_str = " ".join(f"{t[:3]}={c}" for t, c in sorted(agg.items()))
        print(f"{name:<22s} {trans:>8.4f} {final:>8.4f} {best:>8.4f} {t:>6.0f}s  {type_str}")
    
    # Ranked
    rows = []
    for name in metrics.keys():
        if name not in by_metric: continue
        runs = by_metric[name]
        best = np.mean([r['best_iou'] for r in runs])
        rows.append((name, best))
    rows.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*70}")
    print("RANKED BY BEST IoU:")
    print(f"{'='*70}")
    for rank, (name, best) in enumerate(rows, 1):
        delta = best - baseline_best
        marker = " <<<" if delta > 0.005 else (" >" if delta > 0 else "")
        print(f"  #{rank}: {name:<22s} best={best:.4f} (Δ={delta:+.4f}){marker}")
    
    # Type diversity
    print(f"\n{'='*70}")
    print("TYPE DIVERSITY:")
    print(f"{'='*70}")
    from collections import Counter
    baseline_types = None
    for name in metrics.keys():
        if name not in by_metric: continue
        runs = by_metric[name]
        all_types = []
        for r in runs:
            all_types.extend(r['type_selections'])
        if name == 'baseline':
            baseline_types = all_types
        counts = Counter(all_types)
        total = sum(counts.values())
        entropy = -sum((c/total)*np.log2(c/total+1e-10) for c in counts.values())
        diff = 0
        if baseline_types and len(all_types) == len(baseline_types):
            diff = sum(1 for a, b in zip(all_types, baseline_types) if a != b)
        print(f"  {name:<22s} entropy={entropy:.2f}/2.32  diff_from_baseline={diff}/{len(all_types)}")
    
    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open('ownership_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved to ownership_results.json")


if __name__ == "__main__":
    main()