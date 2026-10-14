"""
Metric experiment v2: per-leaf auxiliary metrics for CSG primitive type selection.

Background (v1)
---------------
In v1 every metric was computed on the tree output. With MSE, BCE, focal or
dice alike, the gradient reaching each leaf's selection weights only says where
the blended output should increase or decrease, which does not discriminate
between primitive types. The gradient-based v1 metrics also used
target.occupancy() = (sdf < 0).float(), which has no gradient.

Approach (v2)
-------------
Auxiliary losses are computed per leaf, before Boolean composition, where each
candidate primitive's SDF is well defined and differs between types (after the
Boolean combination the type information is lost). Each candidate's SDF and
normals are compared with the target's, weighted by the selection weights.
Near the target surface, different primitive types give different SDF values
and normals: a sphere near a flat face has a large SDF error and radial
normals, a box a small error and axis-aligned normals.

Metrics
-------
Tree-level (controls):
  1. baseline          - MSE occupancy through the tree
  2. tree_normal_fixed - normal matching on the tree output (differentiable SDF)

Per-leaf:
  3. leaf_sdf          - Per-leaf SDF vs target SDF at surface points
  4. leaf_normal       - Per-leaf normal matching via finite differences
  5. leaf_sdf_normal   - Both SDF + normal per-leaf

Hybrid (tree + per-leaf):
  6. hybrid_sdf        - Baseline + per-leaf SDF auxiliary
  7. hybrid_normal     - Baseline + per-leaf normal auxiliary
  8. hybrid_both       - Baseline + per-leaf SDF + normal

Other:
  9. contrastive       - Reward winner, penalize losers per-leaf
  10. adversarial      - Baseline but resample to emphasize high-error regions
  11. leaf_curvature   - Match SDF Laplacian (curvature) per-leaf
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Callable
from collections import defaultdict

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ROTATION & SDF PRIMITIVES
MIN_SIZE = 0.02

def euler_to_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    cx, sx = torch.cos(angles[0]), torch.sin(angles[0])
    cy, sy = torch.cos(angles[1]), torch.sin(angles[1])
    cz, sz = torch.cos(angles[2]), torch.sin(angles[2])
    R = torch.zeros(3, 3, device=angles.device, dtype=angles.dtype)
    R[0, 0] = cy * cz; R[0, 1] = cz * sx * sy - cx * sz; R[0, 2] = cx * cz * sy + sx * sz
    R[1, 0] = cy * sz; R[1, 1] = cx * cz + sx * sy * sz; R[1, 2] = cx * sy * sz - cz * sx
    R[2, 0] = -sy;     R[2, 1] = cy * sx;                 R[2, 2] = cx * cy
    return R

def sdf_block(p, center, half_size, rotation):
    p_local = torch.matmul(p - center, rotation.T)
    d = torch.abs(p_local) - half_size
    outside = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8)
    inside = torch.clamp(d.max(dim=-1)[0], max=0)
    return outside + inside

def sdf_sphere(p, center, radius):
    return torch.sqrt(((p - center)**2).sum(-1) + 1e-8) - radius

def sdf_cylinder(p, center, radius, half_height, rotation):
    p_local = torch.matmul(p - center, rotation.T)
    d_radial = torch.sqrt(p_local[:, 0]**2 + p_local[:, 2]**2 + 1e-8) - radius
    d_height = torch.abs(p_local[:, 1]) - half_height
    outside = torch.sqrt(torch.clamp(d_radial, min=0)**2 + torch.clamp(d_height, min=0)**2 + 1e-8)
    inside = torch.clamp(torch.maximum(d_radial, d_height), max=0)
    return outside + inside

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
    PARAM_COUNTS = {
        'block': 9, 'sphere': 4, 'cylinder': 8, 'wedge': 9, 'corner_wedge': 9,
    }
    
    def __init__(self, primitive_types: List[str], init_scale: float = 0.5):
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


# CSG TREE
class CSGTree(nn.Module):
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
        c0, c1, c2, c3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3], weights[:, 3:4]
        return (c1 + c2) * left + (c1 + c3) * right + (c0 - c1 - c2 - c3) * left * right
    
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
            selected_occs = torch.stack([
                self.primitive_pool.get_occupancy(p, idx) 
                for idx in self._selected_global_indices
            ], dim=-1)
            leaf_outputs = [selected_occs[:, i] for i in range(self.num_leaves)]
            return self._tree_combine(leaf_outputs)
        
        all_occs = self.primitive_pool.get_all_occupancies(p)
        leaf_outputs = []
        for i, leaf in enumerate(self.leaves):
            start, end = i * self.n_types, (i + 1) * self.n_types
            leaf_outputs.append(leaf(all_occs[:, start:end], self.temperature))
        return self._tree_combine(leaf_outputs)
    
    def get_leaf_weighted_sdfs(self, p):
        """Get each leaf's selection-weighted SDF output (bypassing tree).
        Returns: [num_leaves, num_points] tensor."""
        all_sdfs = self.primitive_pool.get_all_sdfs(p)
        leaf_sdfs = []
        for i, leaf in enumerate(self.leaves):
            start, end = i * self.n_types, (i + 1) * self.n_types
            weights = leaf.get_selection_weights(self.temperature)
            weighted = (all_sdfs[:, start:end] * weights).sum(dim=-1)
            leaf_sdfs.append(weighted)
        return torch.stack(leaf_sdfs, dim=0)
    
    def get_per_candidate_sdfs(self, p):
        """Get each candidate's SDF at each leaf.
        Returns: [num_leaves, n_types, num_points] tensor."""
        all_sdfs = self.primitive_pool.get_all_sdfs(p)
        result = []
        for i in range(self.num_leaves):
            start, end = i * self.n_types, (i + 1) * self.n_types
            result.append(all_sdfs[:, start:end].T)  # [n_types, num_points]
        return torch.stack(result, dim=0)  # [num_leaves, n_types, num_points]
    
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
    """Box + sphere - 2 cylinders. Provides occupancy AND differentiable SDF."""
    
    def sdf(self, p):
        """Differentiable SDF."""
        d = torch.abs(p) - 0.5
        box_sdf = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + \
                  torch.clamp(d.max(dim=-1)[0], max=0)
        sphere_center = torch.tensor([0.3, 0.3, 0.3], device=p.device)
        sphere_sdf = torch.sqrt(((p - sphere_center)**2).sum(-1) + 1e-8) - 0.35
        union_sdf = torch.minimum(box_sdf, sphere_sdf)
        cyl_y = torch.sqrt(p[:, 0]**2 + p[:, 2]**2 + 1e-8) - 0.15
        cyl_x = torch.sqrt(p[:, 1]**2 + p[:, 2]**2 + 1e-8) - 0.15
        return torch.maximum(torch.maximum(union_sdf, -cyl_y), -cyl_x)
    
    def occupancy(self, p):
        return (self.sdf(p) < 0).float()
    
    def sdf_gradient(self, p):
        """Compute SDF gradient (surface normal direction) via autograd.
        Differentiates through the SDF, not the binary occupancy."""
        p_grad = p.detach().requires_grad_(True)
        s = self.sdf(p_grad)
        grad = torch.autograd.grad(s.sum(), p_grad, create_graph=False)[0]
        return grad.detach()
    
    def sample_points(self, n, surface_ratio=0.3, near_ratio=0.5, near_std=0.03):
        n_surface = int(n * surface_ratio)
        n_near = int(n * near_ratio)
        n_random = n - n_surface - n_near
        
        candidates = torch.rand(n * 10, 3, device=device) * 2 - 1
        sdf_vals = self.sdf(candidates).abs()
        
        _, surface_idx = sdf_vals.topk(n_surface, largest=False)
        pts_surface = candidates[surface_idx]
        
        _, near_base_idx = sdf_vals.topk(n_near, largest=False)
        pts_near = candidates[near_base_idx] + torch.randn(n_near, 3, device=device) * near_std
        pts_near = pts_near.clamp(-0.99, 0.99)
        
        pts_random = torch.rand(n_random, 3, device=device) * 2 - 1
        
        P = torch.cat([pts_surface, pts_near, pts_random], dim=0)
        P = P[torch.randperm(P.shape[0])]
        return P, self.occupancy(P)
    
    def sample_near_surface(self, n, threshold=0.05):
        """Sample points specifically near the target surface."""
        candidates = torch.rand(n * 20, 3, device=device) * 2 - 1
        sdf_vals = self.sdf(candidates)
        mask = sdf_vals.abs() < threshold
        if mask.sum() < n:
            # Fallback: take closest n points
            _, idx = sdf_vals.abs().topk(n, largest=False)
            return candidates[idx]
        return candidates[mask][:n]


# UTILITY: Finite-difference normals
def finite_diff_normals(sdf_fn, p, eps=0.002):
    """Compute SDF gradient via finite differences. Works for any SDF function.
    sdf_fn: callable(p) -> [N] tensor of SDF values
    p: [N, 3] points
    Returns: [N, 3] gradient vectors (unnormalized normals)
    """
    dx = torch.zeros_like(p); dx[:, 0] = eps
    dy = torch.zeros_like(p); dy[:, 1] = eps
    dz = torch.zeros_like(p); dz[:, 2] = eps
    
    gx = (sdf_fn(p + dx) - sdf_fn(p - dx)) / (2 * eps)
    gy = (sdf_fn(p + dy) - sdf_fn(p - dy)) / (2 * eps)
    gz = (sdf_fn(p + dz) - sdf_fn(p - dz)) / (2 * eps)
    
    return torch.stack([gx, gy, gz], dim=-1)


# IoU
def compute_iou(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).float()
    gt_bin = (gt > threshold).float()
    intersection = (pred_bin * gt_bin).sum()
    union = ((pred_bin + gt_bin) > 0).float().sum()
    return (intersection / (union + 1e-8)).item()


# METRIC FUNCTIONS

# --- TREE-LEVEL CONTROLS ---

def metric_baseline(model, P, gt_occ, target, **kw):
    """Standard MSE on occupancy through tree."""
    pred = model(P)
    return F.mse_loss(pred, gt_occ)


def metric_tree_normal_fixed(model, P, gt_occ, target, **kw):
    """Normal matching on the tree output, differentiating through the smooth SDF
    (not through the binary occupancy (sdf<0).float())."""
    # Tree occupancy loss
    pred = model(P)
    occ_loss = F.mse_loss(pred, gt_occ)
    
    # Normal matching on subset of near-surface points
    n_normal = min(1024, P.shape[0])
    gt_sdf_vals = target.sdf(P).detach()
    _, near_idx = gt_sdf_vals.abs().topk(n_normal, largest=False)
    P_near = P[near_idx].detach()
    
    # Target normals from the SDF gradient (not the occupancy).
    gt_normals = target.sdf_gradient(P_near)
    gt_normals = gt_normals / (gt_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Predicted normals via finite differences on tree output's underlying SDF
    # We use the occupancy output and finite-diff it as a proxy
    def tree_occ(p):
        return model(p)
    pred_normals = finite_diff_normals(tree_occ, P_near, eps=0.003)
    pred_normals = pred_normals / (pred_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    normal_loss = (1 - F.cosine_similarity(pred_normals, gt_normals, dim=-1)).mean()
    
    return occ_loss + 0.1 * normal_loss


# --- PER-LEAF METRICS (bypass the tree) ---

def metric_leaf_sdf(model, P, gt_occ, target, **kw):
    """Per-leaf SDF regression at near-surface points, bypassing tree.
    Each leaf's selection-weighted SDF is compared directly to target SDF.
    Type-discriminative because different types produce different SDF fields."""
    # Tree loss for overall optimization
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    # Per-leaf SDF matching at near-surface points
    n_surf = min(2048, P.shape[0])
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_surf, largest=False)
    P_surf = P[near_idx]
    gt_sdf_surf = target.sdf(P_surf)  # analytic target: gradient path is kept
    
    # Get each leaf's weighted SDF at surface points
    leaf_sdfs = model.get_leaf_weighted_sdfs(P_surf)  # [num_leaves, n_surf]
    
    # Each leaf tries to match target SDF (clamp to avoid huge values)
    leaf_loss = 0
    for i in range(model.num_leaves):
        leaf_loss = leaf_loss + F.mse_loss(
            leaf_sdfs[i].clamp(-0.5, 0.5), 
            gt_sdf_surf.clamp(-0.5, 0.5).detach()
        )
    leaf_loss = leaf_loss / model.num_leaves
    
    return tree_loss + 0.3 * leaf_loss


def metric_leaf_normal(model, P, gt_occ, target, **kw):
    """Per-leaf normal matching via finite differences.
    Each leaf's weighted SDF normals compared to target normals at surface.
    Type-discriminative: sphere normals are radial, box normals are axial."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    # Get near-surface points
    n_norm = min(512, P.shape[0])  # Smaller batch for finite diff cost
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_norm, largest=False)
    P_surf = P[near_idx].detach()
    
    # Target normals (computed once)
    gt_normals = target.sdf_gradient(P_surf)
    gt_normals = gt_normals / (gt_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-leaf normals via finite differences on each leaf's weighted SDF
    eps = 0.003
    dx = torch.zeros_like(P_surf); dx[:, 0] = eps
    dy = torch.zeros_like(P_surf); dy[:, 1] = eps
    dz = torch.zeros_like(P_surf); dz[:, 2] = eps
    
    # Evaluate all primitive SDFs at all 7 positions (cost: 7 primitive evaluations)
    points_7 = [P_surf, P_surf + dx, P_surf - dx, P_surf + dy, P_surf - dy, P_surf + dz, P_surf - dz]
    all_sdfs_7 = [model.primitive_pool.get_all_sdfs(p) for p in points_7]
    
    leaf_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        
        # Weighted SDF at each of 7 positions
        def weighted_sdf_at(idx):
            return (all_sdfs_7[idx][:, start:end] * weights).sum(dim=-1)
        
        # Finite difference normals
        gx = (weighted_sdf_at(1) - weighted_sdf_at(2)) / (2 * eps)
        gy = (weighted_sdf_at(3) - weighted_sdf_at(4)) / (2 * eps)
        gz = (weighted_sdf_at(5) - weighted_sdf_at(6)) / (2 * eps)
        pred_normal = torch.stack([gx, gy, gz], dim=-1)
        pred_normal = pred_normal / (pred_normal.norm(dim=-1, keepdim=True) + 1e-8)
        
        cos_sim = F.cosine_similarity(pred_normal, gt_normals, dim=-1)
        leaf_loss = leaf_loss + (1 - cos_sim).mean()
    
    leaf_loss = leaf_loss / model.num_leaves
    return tree_loss + 0.2 * leaf_loss


def metric_leaf_sdf_normal(model, P, gt_occ, target, **kw):
    """Combined per-leaf SDF + normal matching."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    # Near-surface points
    n_pts = min(512, P.shape[0])
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_pts, largest=False)
    P_surf = P[near_idx].detach()
    gt_sdf_surf = target.sdf(P_surf).detach()
    
    # Target normals
    gt_normals = target.sdf_gradient(P_surf)
    gt_normals = gt_normals / (gt_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Finite diff setup
    eps = 0.003
    dx = torch.zeros_like(P_surf); dx[:, 0] = eps
    dy = torch.zeros_like(P_surf); dy[:, 1] = eps
    dz = torch.zeros_like(P_surf); dz[:, 2] = eps
    
    points_7 = [P_surf, P_surf + dx, P_surf - dx, P_surf + dy, P_surf - dy, P_surf + dz, P_surf - dz]
    all_sdfs_7 = [model.primitive_pool.get_all_sdfs(p) for p in points_7]
    
    sdf_loss = 0
    normal_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        
        # SDF matching
        weighted_sdf_center = (all_sdfs_7[0][:, start:end] * weights).sum(dim=-1)
        sdf_loss = sdf_loss + F.mse_loss(
            weighted_sdf_center.clamp(-0.5, 0.5), gt_sdf_surf.clamp(-0.5, 0.5)
        )
        
        # Normal matching
        def wsdf(idx):
            return (all_sdfs_7[idx][:, start:end] * weights).sum(dim=-1)
        gx = (wsdf(1) - wsdf(2)) / (2 * eps)
        gy = (wsdf(3) - wsdf(4)) / (2 * eps)
        gz = (wsdf(5) - wsdf(6)) / (2 * eps)
        pred_n = torch.stack([gx, gy, gz], dim=-1)
        pred_n = pred_n / (pred_n.norm(dim=-1, keepdim=True) + 1e-8)
        normal_loss = normal_loss + (1 - F.cosine_similarity(pred_n, gt_normals, dim=-1)).mean()
    
    sdf_loss = sdf_loss / model.num_leaves
    normal_loss = normal_loss / model.num_leaves
    
    return tree_loss + 0.2 * sdf_loss + 0.2 * normal_loss


# --- HYBRID (tree + best per-leaf) ---

def metric_hybrid_sdf(model, P, gt_occ, target, **kw):
    """Baseline + per-leaf SDF (lighter weight)."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    n_surf = min(2048, P.shape[0])
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_surf, largest=False)
    P_surf = P[near_idx]
    gt_sdf_surf = target.sdf(P_surf).detach()
    
    leaf_sdfs = model.get_leaf_weighted_sdfs(P_surf)
    leaf_loss = 0
    for i in range(model.num_leaves):
        leaf_loss = leaf_loss + F.mse_loss(
            leaf_sdfs[i].clamp(-0.5, 0.5), gt_sdf_surf.clamp(-0.5, 0.5)
        )
    leaf_loss = leaf_loss / model.num_leaves
    
    return tree_loss + 0.1 * leaf_loss


def metric_hybrid_normal(model, P, gt_occ, target, **kw):
    """Baseline + per-leaf normal matching (lighter weight)."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    n_norm = min(512, P.shape[0])
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_norm, largest=False)
    P_surf = P[near_idx].detach()
    
    gt_normals = target.sdf_gradient(P_surf)
    gt_normals = gt_normals / (gt_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    eps = 0.003
    dx = torch.zeros_like(P_surf); dx[:, 0] = eps
    dy = torch.zeros_like(P_surf); dy[:, 1] = eps
    dz = torch.zeros_like(P_surf); dz[:, 2] = eps
    
    points_7 = [P_surf, P_surf + dx, P_surf - dx, P_surf + dy, P_surf - dy, P_surf + dz, P_surf - dz]
    all_sdfs_7 = [model.primitive_pool.get_all_sdfs(p) for p in points_7]
    
    leaf_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        def wsdf(idx):
            return (all_sdfs_7[idx][:, start:end] * weights).sum(dim=-1)
        gx = (wsdf(1) - wsdf(2)) / (2 * eps)
        gy = (wsdf(3) - wsdf(4)) / (2 * eps)
        gz = (wsdf(5) - wsdf(6)) / (2 * eps)
        pred_n = torch.stack([gx, gy, gz], dim=-1)
        pred_n = pred_n / (pred_n.norm(dim=-1, keepdim=True) + 1e-8)
        leaf_loss = leaf_loss + (1 - F.cosine_similarity(pred_n, gt_normals, dim=-1)).mean()
    
    leaf_loss = leaf_loss / model.num_leaves
    return tree_loss + 0.1 * leaf_loss


def metric_hybrid_both(model, P, gt_occ, target, **kw):
    """Baseline + per-leaf SDF + normal."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    n_pts = min(512, P.shape[0])
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_pts, largest=False)
    P_surf = P[near_idx].detach()
    gt_sdf_surf = target.sdf(P_surf).detach()
    gt_normals = target.sdf_gradient(P_surf)
    gt_normals = gt_normals / (gt_normals.norm(dim=-1, keepdim=True) + 1e-8)
    
    eps = 0.003
    dx = torch.zeros_like(P_surf); dx[:, 0] = eps
    dy = torch.zeros_like(P_surf); dy[:, 1] = eps
    dz = torch.zeros_like(P_surf); dz[:, 2] = eps
    
    points_7 = [P_surf, P_surf + dx, P_surf - dx, P_surf + dy, P_surf - dy, P_surf + dz, P_surf - dz]
    all_sdfs_7 = [model.primitive_pool.get_all_sdfs(p) for p in points_7]
    
    sdf_loss = 0
    normal_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        wsdf_center = (all_sdfs_7[0][:, start:end] * weights).sum(dim=-1)
        sdf_loss = sdf_loss + F.mse_loss(wsdf_center.clamp(-0.5, 0.5), gt_sdf_surf.clamp(-0.5, 0.5))
        
        def wsdf(idx):
            return (all_sdfs_7[idx][:, start:end] * weights).sum(dim=-1)
        gx = (wsdf(1) - wsdf(2)) / (2 * eps)
        gy = (wsdf(3) - wsdf(4)) / (2 * eps)
        gz = (wsdf(5) - wsdf(6)) / (2 * eps)
        pred_n = torch.stack([gx, gy, gz], dim=-1)
        pred_n = pred_n / (pred_n.norm(dim=-1, keepdim=True) + 1e-8)
        normal_loss = normal_loss + (1 - F.cosine_similarity(pred_n, gt_normals, dim=-1)).mean()
    
    sdf_loss = sdf_loss / model.num_leaves
    normal_loss = normal_loss / model.num_leaves
    return tree_loss + 0.1 * sdf_loss + 0.1 * normal_loss


# --- OTHER METRICS ---

def metric_contrastive(model, P, gt_occ, target, **kw):
    """Contrastive per-leaf: reward winner's SDF match, penalize losers'.
    At each leaf, the best-matching candidate should have much lower SDF error
    than the others, which strengthens the type-discriminative signal."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    n_pts = min(1024, P.shape[0])
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_pts, largest=False)
    P_surf = P[near_idx]
    gt_sdf_surf = target.sdf(P_surf).detach().clamp(-0.5, 0.5)
    
    # Per-candidate SDF errors at each leaf
    per_cand = model.get_per_candidate_sdfs(P_surf)  # [num_leaves, n_types, n_pts]
    
    contrastive_loss = 0
    for i, leaf in enumerate(model.leaves):
        weights = leaf.get_selection_weights(model.temperature)  # [n_types]
        
        # Each candidate's SDF error
        errors = ((per_cand[i].clamp(-0.5, 0.5) - gt_sdf_surf) ** 2).mean(dim=-1)  # [n_types]
        
        # Errors weighted by selection weights: pushes weight toward
        # low-error candidates (weighted contrastive term).
        contrastive_loss = contrastive_loss + (weights * errors).sum()
    
    contrastive_loss = contrastive_loss / model.num_leaves
    return tree_loss + 0.3 * contrastive_loss


def metric_adversarial(model, P, gt_occ, target, **kw):
    """Baseline but oversample where current model is worst.
    A sampling strategy rather than a different metric:
    focuses the loss on the points where primitives are hardest to tell apart."""
    pred = model(P)
    
    # Compute per-point error
    errors = (pred - gt_occ).pow(2)
    
    # Weight: higher error = higher weight (soft hard-example mining)
    weights = errors.detach()
    weights = weights / (weights.mean() + 1e-8)
    weights = 1.0 + 2.0 * weights  # Baseline weight 1, hard examples up to 3x
    
    return (weights * errors).mean()


def metric_leaf_curvature(model, P, gt_occ, target, **kw):
    """Per-leaf curvature matching via SDF Laplacian.
    Different types have characteristic curvatures:
    sphere=constant, box=zero on faces, cylinder=one-directional.
    Uses finite differences for second derivatives."""
    pred = model(P)
    tree_loss = F.mse_loss(pred, gt_occ)
    
    n_pts = min(256, P.shape[0])  # Small batch - second derivatives are expensive
    gt_sdf_all = target.sdf(P).detach()
    _, near_idx = gt_sdf_all.abs().topk(n_pts, largest=False)
    P_surf = P[near_idx].detach()
    
    # Target curvature (Laplacian of SDF)
    eps = 0.005
    gt_sdf_center = target.sdf(P_surf)
    gt_laplacian = torch.zeros(n_pts, device=device)
    for dim in range(3):
        offset = torch.zeros_like(P_surf)
        offset[:, dim] = eps
        gt_laplacian += (target.sdf(P_surf + offset) + target.sdf(P_surf - offset) - 2 * gt_sdf_center) / (eps**2)
    gt_laplacian = gt_laplacian.detach()
    
    # Per-leaf curvature via finite differences
    all_sdfs_offsets = {}
    all_sdfs_offsets['center'] = model.primitive_pool.get_all_sdfs(P_surf)
    for dim in range(3):
        offset = torch.zeros_like(P_surf)
        offset[:, dim] = eps
        all_sdfs_offsets[f'+{dim}'] = model.primitive_pool.get_all_sdfs(P_surf + offset)
        all_sdfs_offsets[f'-{dim}'] = model.primitive_pool.get_all_sdfs(P_surf - offset)
    
    curv_loss = 0
    for i, leaf in enumerate(model.leaves):
        start, end = i * model.n_types, (i + 1) * model.n_types
        weights = leaf.get_selection_weights(model.temperature)
        
        s_center = (all_sdfs_offsets['center'][:, start:end] * weights).sum(dim=-1)
        pred_laplacian = torch.zeros(n_pts, device=device)
        for dim in range(3):
            s_plus = (all_sdfs_offsets[f'+{dim}'][:, start:end] * weights).sum(dim=-1)
            s_minus = (all_sdfs_offsets[f'-{dim}'][:, start:end] * weights).sum(dim=-1)
            pred_laplacian += (s_plus + s_minus - 2 * s_center) / (eps**2)
        
        curv_loss = curv_loss + F.mse_loss(pred_laplacian, gt_laplacian)
    
    curv_loss = curv_loss / model.num_leaves
    return tree_loss + 0.05 * curv_loss


# TRAINING
def train_one_config(
    metric_name: str,
    metric_fn: Callable,
    seed: int,
    target: SyntheticTarget,
    depth: int = 4,
    phase1_epochs: int = 1500,
    phase2_epochs: int = 3000,
    num_samples: int = 8192,
    eval_resolution: int = 64,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 5e-4,
    temp_start: float = 0.3,
    temp_end: float = 2000.0,
    resample_every: int = 10,
    eval_every: int = 300,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = CSGTree(depth=depth, init_scale=0.3, omega=10.0).to(device)
    
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
    P, gt_occ = None, None
    
    for epoch in range(phase1_epochs):
        progress = epoch / max(phase1_epochs - 1, 1)
        model.temperature = temp_start * (temp_end / temp_start) ** progress
        
        if epoch % resample_every == 0 or P is None:
            P, gt_occ = target.sample_points(num_samples)
            P, gt_occ = P.detach(), gt_occ.detach()
        
        try:
            loss = metric_fn(model, P, gt_occ, target)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                loss = F.mse_loss(model(P), gt_occ)
            else:
                raise
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                pred = torch.cat([model(b) for b in eval_grid.split(8192)], dim=0)
                iou = compute_iou(pred, eval_gt)
            results['phase1_ious'].append(iou)
    
    # Record selections
    types = model.get_type_distribution()
    results['type_selections'] = types
    for t in model.primitive_types:
        results['type_distribution'][t] = types.count(t)
    
    with torch.no_grad():
        pred = torch.cat([model(b) for b in eval_grid.split(8192)], dim=0)
        results['transition_iou'] = compute_iou(pred, eval_gt)
    
    # Phase 2 (always baseline MSE)
    model.transition_to_phase2()
    optimizer = torch.optim.Adam(model.parameters(), lr=phase2_lr)
    best_iou = results['transition_iou']
    
    for epoch in range(phase2_epochs):
        if epoch % resample_every == 0 or P is None:
            P, gt_occ = target.sample_points(num_samples)
            P, gt_occ = P.detach(), gt_occ.detach()
        
        pred = model(P)
        loss = F.mse_loss(pred, gt_occ)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                pred = torch.cat([model(b) for b in eval_grid.split(8192)], dim=0)
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
    print("=" * 70)
    print("METRIC EXPERIMENT v2: PER-LEAF AUXILIARY METRICS")
    print("=" * 70)
    
    target = SyntheticTarget()
    
    metrics = {
        # Controls (tree-level)
        'baseline':           metric_baseline,
        'tree_normal_fixed':  metric_tree_normal_fixed,
        # Per-leaf
        'leaf_sdf':           metric_leaf_sdf,
        'leaf_normal':        metric_leaf_normal,
        'leaf_sdf_normal':    metric_leaf_sdf_normal,
        # Hybrid
        'hybrid_sdf':         metric_hybrid_sdf,
        'hybrid_normal':      metric_hybrid_normal,
        'hybrid_both':        metric_hybrid_both,
        # Other
        'contrastive':        metric_contrastive,
        'adversarial':        metric_adversarial,
        'leaf_curvature':     metric_leaf_curvature,
    }
    
    seeds = [42, 123]
    depth = 4
    phase1_epochs = 1500
    phase2_epochs = 3000
    num_samples = 8192
    
    total_runs = len(metrics) * len(seeds)
    print(f"\nConfig:")
    print(f"  Metrics: {len(metrics)} ({', '.join(metrics.keys())})")
    print(f"  Seeds: {seeds}")
    print(f"  Depth: {depth} ({2**depth} leaves)")
    print(f"  Phase 1: {phase1_epochs} epochs")
    print(f"  Phase 2: {phase2_epochs} epochs")
    print(f"  Samples: {num_samples}")
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
                    depth=depth, phase1_epochs=phase1_epochs, phase2_epochs=phase2_epochs,
                    num_samples=num_samples,
                )
                all_results.append(result)
                print(f"→ trans={result['transition_iou']:.4f} "
                      f"final={result['final_iou']:.4f} "
                      f"best={result['best_iou']:.4f} "
                      f"t={result['total_time']:.0f}s "
                      f"types={result['type_distribution']}")
            except Exception as e:
                print(f"→ FAILED: {e}")
                import traceback
                traceback.print_exc()
    
    # SUMMARY
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    by_metric = defaultdict(list)
    for r in all_results:
        by_metric[r['metric']].append(r)
    
    print(f"\n{'Metric':<22s} {'Trans IoU':>10s} {'Final IoU':>10s} {'Best IoU':>10s} {'Time':>8s}  Type Dist")
    print("-" * 100)
    
    summary_rows = []
    baseline_best = 0
    for name in metrics.keys():
        if name not in by_metric:
            continue
        runs = by_metric[name]
        trans = np.mean([r['transition_iou'] for r in runs])
        final = np.mean([r['final_iou'] for r in runs])
        best = np.mean([r['best_iou'] for r in runs])
        t = np.mean([r['total_time'] for r in runs])
        
        if name == 'baseline':
            baseline_best = best
        
        agg_types = defaultdict(int)
        for r in runs:
            for typ, cnt in r['type_distribution'].items():
                agg_types[typ] += cnt
        type_str = " ".join(f"{t[:3]}={c}" for t, c in sorted(agg_types.items()))
        
        summary_rows.append((name, trans, final, best, t, type_str))
        print(f"{name:<22s} {trans:>10.4f} {final:>10.4f} {best:>10.4f} {t:>7.0f}s  {type_str}")
    
    # Ranked
    summary_rows.sort(key=lambda x: x[3], reverse=True)
    print(f"\n{'='*100}")
    print("RANKED BY BEST IoU:")
    print(f"{'='*100}")
    for rank, (name, trans, final, best, t, type_str) in enumerate(summary_rows, 1):
        delta = best - baseline_best
        marker = " <<<" if delta > 0.005 else (" >" if delta > 0 else "")
        print(f"  #{rank}: {name:<22s} best={best:.4f} (Δ={delta:+.4f} vs baseline)  "
              f"final={final:.4f}  trans={trans:.4f}{marker}")
    
    # Type diversity
    print(f"\n{'='*100}")
    print("TYPE DIVERSITY")
    print(f"{'='*100}")
    
    baseline_types = None
    for name in metrics.keys():
        if name not in by_metric:
            continue
        runs = by_metric[name]
        all_types_flat = []
        for r in runs:
            all_types_flat.extend(r['type_selections'])
        
        if name == 'baseline':
            baseline_types = all_types_flat
        
        from collections import Counter
        counts = Counter(all_types_flat)
        total = sum(counts.values())
        entropy = -sum((c/total) * np.log2(c/total + 1e-10) for c in counts.values())
        
        diff = 0
        if baseline_types and len(all_types_flat) == len(baseline_types):
            diff = sum(1 for a, b in zip(all_types_flat, baseline_types) if a != b)
        
        print(f"  {name:<22s} entropy={entropy:.2f}/2.32  "
              f"diff_from_baseline={diff}/{len(all_types_flat)}")
    
    # Save
    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    out_path = "metric_v2_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()