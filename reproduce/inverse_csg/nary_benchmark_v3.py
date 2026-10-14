#!/usr/bin/env python3
"""
  N-ary fuzzy CSG, v3: Bc type selection.

Hypothesis: Bc product union for type blending gives gradient shielding
(losing types train slowly where the winner dominates and faster where it
is absent), which allows continuous type selection.

Three methods:

  A) BASELINE          -  overcomplete discrete pool (16x5=80 prims)
  G) BC-TYPE + SINE    -  each prim has all 5 types, Bc product union
                        for type selection, sine for switchability
  H) SOFTMAX-TYPE      -  same architecture but weighted-sum blending
                        (control: separates the effect of Bc from that
                        of having several types per slot)

Key comparison:
  G vs A -> does continuous type selection beat overcomplete discrete?
  G vs H -> does Bc beat the weighted sum?

Usage:
  python nary_benchmark_v3.py --target complex_csg --epochs 15000
  python nary_benchmark_v3.py --mesh ./bunny.obj --epochs 30000
"""

import os, sys, time, math, json, argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import mcubes
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import igl; HAS_IGL = True
except ImportError:
    HAS_IGL = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# MESH SDF
class MeshSDF:
    def __init__(self, mesh_path, normalize=True, margin=0.1):
        print(f"Loading mesh: {mesh_path}")
        self.mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(list(self.mesh.geometry.values()))
        if not self.mesh.is_watertight:
            self.mesh = self._repair(self.mesh)
        if normalize:
            self.mesh.vertices -= self.mesh.bounding_box.centroid
            s = (1.0 - margin) / np.abs(self.mesh.vertices).max()
            self.mesh.vertices *= s
        self.vertices = np.ascontiguousarray(self.mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.mesh.faces, dtype=np.int64)
        print(f"  {len(self.vertices):,} verts, watertight={self.mesh.is_watertight}")

    def _repair(self, mesh):
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mf.repair(verbose=False)
            f = mf.mesh.faces
            f = f.reshape(-1, 4)[:, 1:4] if f.ndim == 1 else f
            r = trimesh.Trimesh(vertices=mf.mesh.points, faces=f)
            if r.is_watertight: print("  ✓ Repaired"); return r
        except Exception: pass
        try:
            from scipy import ndimage
            res = 128
            vox = mesh.voxelized(pitch=2.0/res)
            vg = ndimage.binary_fill_holes(vox.matrix).astype(np.float32)
            vg = ndimage.gaussian_filter(vg, sigma=0.3)
            vg = np.pad(vg, 1, mode='constant', constant_values=0)
            v, t = mcubes.marching_cubes(vg, 0.5)
            v = (v-1)/res*(mesh.vertices.max()-mesh.vertices.min())+mesh.vertices.min()
            return trimesh.Trimesh(vertices=v, faces=t)
        except Exception: pass
        return mesh

    def query(self, points):
        points = np.ascontiguousarray(points, dtype=np.float64)
        if HAS_IGL:
            sdf, _, _, _ = igl.signed_distance(points, self.vertices, self.faces)
            return sdf.astype(np.float32)
        _, dist, _ = self.mesh.nearest.on_surface(points)
        sdf = dist.copy(); sdf[self.mesh.contains(points)] *= -1
        return sdf.astype(np.float32)

    def query_sdf_grid(self, resolution):
        idx = np.linspace(-1, 1, resolution)
        x, y, z = np.meshgrid(idx, idx, idx, indexing='ij')
        return self.query(np.stack([x.flatten(), y.flatten(), z.flatten()], 1)
                         ).reshape(resolution, resolution, resolution)

    def sample_points(self, n, surface_ratio=0.3, near_ratio=0.5, near_std=0.03):
        ns = int(n*surface_ratio); nn_ = int(n*near_ratio); nr = n-ns-nn_
        ps, _ = trimesh.sample.sample_surface(self.mesh, ns)
        pn, _ = trimesh.sample.sample_surface(self.mesh, nn_)
        pn = np.clip(pn + np.random.randn(nn_, 3)*near_std, -0.99, 0.99)
        pr = np.random.uniform(-1, 1, (nr, 3))
        pts = np.vstack([ps, pn, pr]).astype(np.float32)
        occ = (self.query(pts) < 0).astype(np.float32)
        perm = np.random.permutation(len(pts))
        return pts[perm], occ[perm]

# UTILITIES
MIN_SIZE = 0.02

def sample_3D_grid(resolution, low=-1.0, high=1.0):
    idx = torch.linspace(low, high, resolution, device=device)
    x, y, z = torch.meshgrid(idx, idx, idx, indexing='ij')
    return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

def mc_from_occ(filepath, occ, res):
    o = occ.detach().cpu().numpy().reshape(res, res, res)
    if o.min() >= 0.5 or o.max() <= 0.5: return None
    v, t = mcubes.marching_cubes(-(o*2-1), 0)
    v = v/(res-1)*2-1; t = t.copy(); t[:,[0,1]] = t[:,[1,0]]
    mcubes.export_obj(v, t, filepath); return v, t

def mc_from_sdf(filepath, sdf_grid, res):
    if sdf_grid.min() >= 0 or sdf_grid.max() <= 0: return None
    v, t = mcubes.marching_cubes(sdf_grid, 0)
    v = v/(res-1)*2-1; t = t.copy(); t[:,[0,1]] = t[:,[1,0]]
    mcubes.export_obj(v, t, filepath); return v, t

def compute_iou(pred, gt, threshold=0.5):
    p = (pred > threshold).float(); g = (gt > threshold).float()
    return ((p*g).sum() / (((p+g)>0).float().sum()+1e-8)).item()

def compute_fscore(pred, gt, threshold=0.5):
    p = (pred > threshold).float(); g = (gt > threshold).float()
    tp = (p*g).sum()
    return (2*tp / (p.sum()+g.sum()+1e-8)).item()

def batched_forward(model, grid, chunk=32**3, **kw):
    with torch.no_grad():
        return torch.cat([model(b, **kw) for b in torch.split(grid, chunk)], 0)

# SDF PRIMITIVES (all types, local coordinates)
def euler_to_R(a):
    cx,sx = torch.cos(a[0]),torch.sin(a[0])
    cy,sy = torch.cos(a[1]),torch.sin(a[1])
    cz,sz = torch.cos(a[2]),torch.sin(a[2])
    R = torch.zeros(3,3,device=a.device,dtype=a.dtype)
    R[0,0]=cy*cz; R[0,1]=cz*sx*sy-cx*sz; R[0,2]=cx*cz*sy+sx*sz
    R[1,0]=cy*sz; R[1,1]=cx*cz+sx*sy*sz; R[1,2]=cx*sy*sz-cz*sx
    R[2,0]=-sy;   R[2,1]=cy*sx;          R[2,2]=cx*cy
    return R

def sdf_sphere_local(pl, r):
    """Sphere SDF in local coords. r = radius."""
    return torch.sqrt((pl**2).sum(-1)+1e-8) - (r.abs()+MIN_SIZE)

def sdf_block_local(pl, half_size):
    """Box SDF in local coords. half_size = (hx, hy, hz)."""
    h = half_size.abs()+MIN_SIZE
    d = torch.abs(pl) - h
    return torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)

def sdf_cylinder_local(pl, r, hh):
    """Cylinder SDF in local coords. Axis = Y. r=radius, hh=half-height."""
    r_ = r.abs()+MIN_SIZE; hh_ = hh.abs()+MIN_SIZE
    dr = torch.sqrt(pl[:,0]**2+pl[:,2]**2+1e-8) - r_
    dh = torch.abs(pl[:,1]) - hh_
    return torch.sqrt(torch.clamp(dr,min=0)**2+torch.clamp(dh,min=0)**2+1e-8)+\
           torch.clamp(torch.maximum(dr,dh),max=0)

def sdf_wedge_local(pl, half_size):
    """Wedge SDF in local coords."""
    h = half_size.abs()+MIN_SIZE
    d = torch.abs(pl) - h
    box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    inv = 1.0/torch.sqrt(1.0/(h[1]**2)+1.0/(h[2]**2)+1e-8)
    plane = (pl[:,1]/h[1]+pl[:,2]/h[2]-1.0)*inv
    return torch.maximum(box,plane)

def sdf_corner_wedge_local(pl, half_size):
    """Corner wedge SDF in local coords."""
    h = half_size.abs()+MIN_SIZE; hx,hy,hz = h[0],h[1],h[2]
    d = torch.abs(pl) - h
    box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    inv1 = 1.0/torch.sqrt(1.0/(hx**2)+1.0/(hy**2)+1e-8)
    p1 = (pl[:,0]/hx+pl[:,1]/hy-1.0)*inv1
    inv2 = 1.0/torch.sqrt(1.0/(hz**2)+1.0/(hy**2)+1e-8)
    p2 = (pl[:,2]/hz+pl[:,1]/hy-1.0)*inv2
    return torch.maximum(torch.maximum(box,p1),p2)

# also need standalone SDF for baseline pool
def compute_prim_sdf(ptype, pr, pts):
    if ptype=='block':    return sdf_block_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
    if ptype=='sphere':   return sdf_sphere_local(pts-pr[:3],pr[3])
    if ptype=='cylinder':
        pl=torch.matmul(pts-pr[:3],euler_to_R(pr[5:8]).T)
        return sdf_cylinder_local(pl,pr[3],pr[4])
    if ptype=='wedge':    return sdf_wedge_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
    if ptype=='corner_wedge': return sdf_corner_wedge_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])

# METHOD A: BASELINE - discrete overcomplete pool
class DiscretePool(nn.Module):
    PC = {'block':9,'sphere':4,'cylinder':8,'wedge':9,'corner_wedge':9}
    def __init__(self, types, init_scale=0.5):
        super().__init__()
        self.types = types; self.num_primitives = len(types)
        total = sum(self.PC[t] for t in types)
        self.params = nn.Parameter(torch.randn(total)*init_scale)
        self.sharpness = nn.Parameter(torch.ones(len(types))*20.0)
        self.param_indices = []; idx = 0
        for t in types:
            c = self.PC[t]; self.param_indices.append((idx,idx+c)); idx += c
    def get_occ(self, p, i):
        s,e = self.param_indices[i]
        sdf = compute_prim_sdf(self.types[i], self.params[s:e], p)
        return torch.sigmoid(-sdf*self.sharpness[i].abs().clamp(min=1.0))
    def get_all_occs(self, p):
        return torch.stack([self.get_occ(p,i) for i in range(self.num_primitives)], dim=-1)
    def get_center(self, i):
        s,_ = self.param_indices[i]; return self.params.data[s:s+3]

class BaselineCSG(nn.Module):
    """Method A: standard overcomplete discrete-type pool."""
    def __init__(self, prims_per_type=16, prim_types=None, num_groups=16,
                 max_group_size=8):
        super().__init__()
        if prim_types is None:
            prim_types = ['block','sphere','cylinder','wedge','corner_wedge']
        all_types = []
        for t in prim_types: all_types.extend([t]*prims_per_type)
        self.pool = DiscretePool(all_types, 0.3)
        self.N = len(all_types); self.K = num_groups
        self.prim_types = prim_types; self.prims_per_type = prims_per_type
        self.max_group_size = max_group_size
        self.m_pos = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 3.0)
        self.m_neg = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 5.0)
        self.act = nn.Parameter(torch.randn(num_groups)*0.3)

    def forward(self, p, **kw):
        occs = self.pool.get_all_occs(p)
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def forward_hard_dnf(self, p, thr=0.5):
        occs = self.pool.get_all_occs(p)
        mp = (torch.sigmoid(self.m_pos)>thr).float()
        mn = (torch.sigmoid(self.m_neg)>thr).float()
        a = (torch.sigmoid(self.act)>thr).float(); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def structure_loss(self, ls=1e-4, lb=1e-5, lc=1e-4, lgs=1e-5, **kw):
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act)
        sparse = ls*(mp.sum()+mn.sum()+a.sum())
        binary = lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum())
        conflict = lc*(mp*mn).sum()
        excess = torch.clamp(mp.sum(0)+mn.sum(0)-self.max_group_size, min=0)
        return sparse+binary+conflict+lgs*(excess**2).sum()

    def get_stats(self, thr=0.3):
        mp = torch.sigmoid(self.m_pos).detach(); mn = torch.sigmoid(self.m_neg).detach()
        a = torch.sigmoid(self.act).detach()
        used = ((mp.max(1)[0]>thr)|(mn.max(1)[0]>thr)).sum().item()
        ag = (a>thr).sum().item()
        types = {}
        for i in range(self.N):
            if mp[i].max()>thr or mn[i].max()>thr:
                t = self.pool.types[i]; types[t] = types.get(t,0)+1
        return {'prims_used':used, 'groups_active':ag, 'type_dist':types, 'N':self.N}

    def print_status(self, thr=0.3):
        s = self.get_stats(thr)
        print(f"  Prims: {s['prims_used']}/{self.N}, Groups: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}")


# BC-TYPE POOL: each prim has all types, Bc product selects
TYPE_NAMES = ['sphere', 'block', 'cylinder', 'wedge', 'corner_wedge']
N_TYPES = len(TYPE_NAMES)

class BcTypePool(nn.Module):
    """
    Each primitive slot computes SDFs for all 5 types in shared local
    coordinates, then blends occupancies via Bc product union.
    
    The Bc product gives gradient shielding: the dominant type gets
    most of the gradient, losers get spatially-appropriate residual
    gradient (more where the winner is weak, less where it's strong).
    This means losing types' geometry slowly improves in regions
    where they might be useful, which allows type switching.
    
    Contrast with weighted sum: loser gets w_k * grad regardless of
    spatial context, changing its geometry uniformly everywhere.
    """
    # Per-type intrinsic param counts (beyond shared center+rotation)
    # sphere: radius(1)
    # block: half_size(3)  
    # cylinder: radius(1) + half_height(1)
    # wedge: half_size(3)
    # corner_wedge: half_size(3)
    TYPE_INTRINSIC = [1, 3, 2, 3, 3]  # = 12 total per prim

    def __init__(self, num_primitives, omega=10.0, init_scale=0.3):
        super().__init__()
        self.N = num_primitives
        self.omega = omega

        # Shared pose: center(3) + rotation(3) per primitive
        self.centers = nn.Parameter(torch.randn(num_primitives, 3) * init_scale)
        self.rotations = nn.Parameter(torch.randn(num_primitives, 3) * 0.1)

        # Per-type intrinsic parameters
        # Stored as flat tensor per primitive: [sphere(1), block(3), cyl(2), wedge(3), cwedge(3)]
        total_intrinsic = sum(self.TYPE_INTRINSIC)  # 12
        self.intrinsic = nn.Parameter(torch.randn(num_primitives, total_intrinsic) * init_scale)

        # Type selection logits (sine-parameterized for Bc, plain for softmax)
        self.type_logits = nn.Parameter(torch.randn(num_primitives, N_TYPES) * 0.3)

        # Sharpness
        self.sharpness = nn.Parameter(torch.ones(num_primitives) * 20.0)

    def _get_intrinsic(self, i, type_idx):
        """Get intrinsic params for primitive i, type type_idx."""
        offsets = [0]; 
        for c in self.TYPE_INTRINSIC: offsets.append(offsets[-1]+c)
        s, e = offsets[type_idx], offsets[type_idx+1]
        return self.intrinsic[i, s:e]

    def _compute_type_sdfs(self, p, i):
        """Compute SDF for all 5 types of primitive i."""
        center = self.centers[i]
        R = euler_to_R(self.rotations[i])
        pl = torch.matmul(p - center, R.T)  # Shared local coords

        sdfs = []
        # sphere
        r = self._get_intrinsic(i, 0)  # (1,)
        sdfs.append(sdf_sphere_local(pl, r[0]))
        # block
        h = self._get_intrinsic(i, 1)  # (3,)
        sdfs.append(sdf_block_local(pl, h))
        # cylinder
        cr = self._get_intrinsic(i, 2)  # (2,)
        sdfs.append(sdf_cylinder_local(pl, cr[0], cr[1]))
        # wedge
        wh = self._get_intrinsic(i, 3)  # (3,)
        sdfs.append(sdf_wedge_local(pl, wh))
        # corner_wedge
        cwh = self._get_intrinsic(i, 4)  # (3,)
        sdfs.append(sdf_corner_wedge_local(pl, cwh))

        return sdfs  # list of 5 tensors, each [P]

    def get_occ_bc(self, p, i, temperature=1.0):
        """
        Bc product union type blending.
        
        occ = 1 - Π_k [1 - w_k * occ_k]
        
        Where w_k = sigmoid(sin(ω * logit_k) * temperature).
        When w_k ~ 1 for one type: occ ~ occ_k (that type)
        When w_k ~ 0: factor = 1 (no contribution)
        
        Gradient: ∂occ/∂θ_k ∝ Π_{j≠k}(1 - w_j * occ_j) · w_k · ∂occ_k/∂θ_k
        When another type j has high w_j·occ_j, this product shrinks -> shielding.
        """
        sdfs = self._compute_type_sdfs(p, i)
        sharp = self.sharpness[i].abs().clamp(min=1.0)

        # Type weights via sine + sigmoid
        logits = self.type_logits[i]  # [K_types]
        weights = torch.sigmoid(torch.sin(self.omega * logits) * temperature)  # [K_types]

        # Bc product union over types
        log_occ = torch.zeros(len(p), device=p.device)
        for k in range(N_TYPES):
            occ_k = torch.sigmoid(-sdfs[k] * sharp)      # [P]
            factor = 1.0 - weights[k] * occ_k              # [P]
            log_occ = log_occ + torch.log(factor.clamp(min=1e-10))

        return 1.0 - torch.exp(log_occ)  # [P]

    def get_occ_softmax(self, p, i, temperature=1.0):
        """
        Weighted-sum type blending (control method).
        
        occ = Σ softmax(logits * temperature)_k · occ_k
        
        Gradient: ∂occ/∂θ_k = w_k · ∂occ_k/∂θ_k
        w_k is input-independent -> no spatial shielding.
        """
        sdfs = self._compute_type_sdfs(p, i)
        sharp = self.sharpness[i].abs().clamp(min=1.0)

        logits = self.type_logits[i]
        weights = torch.softmax(logits * temperature, dim=0)  # [K_types]

        occ = torch.zeros(len(p), device=p.device)
        for k in range(N_TYPES):
            occ_k = torch.sigmoid(-sdfs[k] * sharp)
            occ = occ + weights[k] * occ_k

        return occ.clamp(0, 1)

    def get_all_occs(self, p, mode='bc', temperature=1.0):
        """Get occupancies for all primitives."""
        if mode == 'bc':
            return torch.stack([self.get_occ_bc(p, i, temperature)
                               for i in range(self.N)], dim=-1)
        else:
            return torch.stack([self.get_occ_softmax(p, i, temperature)
                               for i in range(self.N)], dim=-1)

    def get_center(self, i):
        return self.centers.data[i]

    def get_dominant_type(self, i, temperature=20.0):
        """Return the dominant type name for primitive i."""
        logits = self.type_logits[i].detach()
        weights = torch.sigmoid(torch.sin(self.omega * logits) * temperature)
        idx = weights.argmax().item()
        return TYPE_NAMES[idx]

    def type_entropy(self, temperature=1.0):
        """Average entropy of type selection across primitives (lower = more committed)."""
        logits = self.type_logits.detach()
        if temperature > 0:
            weights = torch.sigmoid(torch.sin(self.omega * logits) * temperature)
        else:
            weights = torch.softmax(logits * 20.0, dim=1)
        # Normalize to distribution
        w_norm = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(w_norm * torch.log(w_norm + 1e-8)).sum(dim=1)
        return entropy.mean().item()


# METHOD G: Bc-TYPE CSG (hypothesis)
class BcTypeCSG(nn.Module):
    """
    N-ary fuzzy aggregation with Bc-type pool.
    Each primitive has all 5 types, Bc product selects which one.
    Groups/membership work identically to baseline.
    """
    def __init__(self, num_primitives=50, num_groups=16, omega=10.0,
                 max_group_size=8):
        super().__init__()
        self.pool = BcTypePool(num_primitives, omega)
        self.N = num_primitives; self.K = num_groups
        self.max_group_size = max_group_size
        self.m_pos = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 3.0)
        self.m_neg = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 5.0)
        self.act = nn.Parameter(torch.randn(num_groups)*0.3)

    def forward(self, p, temperature=1.0, **kw):
        occs = self.pool.get_all_occs(p, mode='bc', temperature=temperature)
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def forward_hard_dnf(self, p, thr=0.5):
        # Hard DNF at high temperature (committed types)
        occs = self.pool.get_all_occs(p, mode='bc', temperature=20.0)
        mp = (torch.sigmoid(self.m_pos)>thr).float()
        mn = (torch.sigmoid(self.m_neg)>thr).float()
        a = (torch.sigmoid(self.act)>thr).float(); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def structure_loss(self, ls=1e-4, lb=1e-5, lc=1e-4, lgs=1e-5,
                       lt=5e-5, temperature=1.0, **kw):
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act)
        sparse = ls*(mp.sum()+mn.sum()+a.sum())
        binary = lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum())
        conflict = lc*(mp*mn).sum()
        excess = torch.clamp(mp.sum(0)+mn.sum(0)-self.max_group_size, min=0)
        gs = lgs*(excess**2).sum()
        # Type sparsity: push type weights toward one-hot
        tw = torch.sigmoid(torch.sin(self.pool.omega * self.pool.type_logits) * temperature)
        type_sparse = lt * tw.sum()  # L1 on type weights
        type_binary = lt * (tw * (1-tw)).sum()  # Push to 0/1
        return sparse+binary+conflict+gs+type_sparse+type_binary

    def get_stats(self, thr=0.3):
        mp = torch.sigmoid(self.m_pos).detach(); mn = torch.sigmoid(self.m_neg).detach()
        a = torch.sigmoid(self.act).detach()
        used = ((mp.max(1)[0]>thr)|(mn.max(1)[0]>thr)).sum().item()
        ag = (a>thr).sum().item()
        types = {}
        for i in range(self.N):
            if mp[i].max()>thr or mn[i].max()>thr:
                t = self.pool.get_dominant_type(i); types[t] = types.get(t,0)+1
        entropy = self.pool.type_entropy()
        return {'prims_used':used, 'groups_active':ag, 'type_dist':types,
                'type_entropy':entropy, 'N':self.N}

    def print_status(self, thr=0.3):
        s = self.get_stats(thr)
        print(f"  Prims: {s['prims_used']}/{self.N}, Groups: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}")
        print(f"  Type entropy: {s['type_entropy']:.3f} (0=committed, 1.6=uniform)")


# METHOD H: SOFTMAX-TYPE CSG (control)
class SoftmaxTypeCSG(nn.Module):
    """Same as G but weighted-sum type blending instead of Bc product.
    Control for the effect of Bc shielding."""
    def __init__(self, num_primitives=50, num_groups=16, max_group_size=8):
        super().__init__()
        self.pool = BcTypePool(num_primitives, omega=10.0)
        self.N = num_primitives; self.K = num_groups
        self.max_group_size = max_group_size
        self.m_pos = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 3.0)
        self.m_neg = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 5.0)
        self.act = nn.Parameter(torch.randn(num_groups)*0.3)

    def forward(self, p, temperature=1.0, **kw):
        occs = self.pool.get_all_occs(p, mode='softmax', temperature=temperature)
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def forward_hard_dnf(self, p, thr=0.5):
        occs = self.pool.get_all_occs(p, mode='softmax', temperature=20.0)
        mp = (torch.sigmoid(self.m_pos)>thr).float()
        mn = (torch.sigmoid(self.m_neg)>thr).float()
        a = (torch.sigmoid(self.act)>thr).float(); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def structure_loss(self, ls=1e-4, lb=1e-5, lc=1e-4, lgs=1e-5, **kw):
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act)
        sparse = ls*(mp.sum()+mn.sum()+a.sum())
        binary = lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum())
        conflict = lc*(mp*mn).sum()
        excess = torch.clamp(mp.sum(0)+mn.sum(0)-self.max_group_size, min=0)
        return sparse+binary+conflict+lgs*(excess**2).sum()

    def get_stats(self, thr=0.3):
        mp = torch.sigmoid(self.m_pos).detach(); mn = torch.sigmoid(self.m_neg).detach()
        a = torch.sigmoid(self.act).detach()
        used = ((mp.max(1)[0]>thr)|(mn.max(1)[0]>thr)).sum().item()
        ag = (a>thr).sum().item()
        types = {}
        for i in range(self.N):
            if mp[i].max()>thr or mn[i].max()>thr:
                # For softmax: argmax of weights
                w = torch.softmax(self.pool.type_logits[i].detach()*20.0, dim=0)
                t = TYPE_NAMES[w.argmax().item()]; types[t] = types.get(t,0)+1
        return {'prims_used':used, 'groups_active':ag, 'type_dist':types, 'N':self.N}

    def print_status(self, thr=0.3):
        s = self.get_stats(thr)
        print(f"  Prims: {s['prims_used']}/{self.N}, Groups: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}")


# INITIALIZATION
def surface_init(model, mesh_sdf):
    """FPS surface initialization - works for any model with a pool."""
    N = model.N
    pts, _ = trimesh.sample.sample_surface(mesh_sdf.mesh, N*3)
    sel = [0]; dists = np.full(len(pts), np.inf)
    for _ in range(N-1):
        d = np.linalg.norm(pts-pts[sel[-1]], axis=1)
        dists = np.minimum(dists, d); sel.append(dists.argmax())
    bbox = mesh_sdf.mesh.bounding_box.extents.max()
    ts = bbox/(N**(1/3))

    with torch.no_grad():
        pool = model.pool
        if hasattr(pool, 'centers'):
            # BcTypePool
            for i in range(N):
                pool.centers.data[i] = torch.from_numpy(pts[sel[i]].astype(np.float32))
                # Initialize intrinsic params to reasonable sizes
                pool.intrinsic.data[i, 0] = ts*0.5  # sphere radius
                pool.intrinsic.data[i, 1:4] = ts*0.4  # block half_size
                pool.intrinsic.data[i, 4] = ts*0.3  # cylinder radius
                pool.intrinsic.data[i, 5] = ts*0.5  # cylinder half_height
                pool.intrinsic.data[i, 6:9] = ts*0.4  # wedge half_size
                pool.intrinsic.data[i, 9:12] = ts*0.4  # corner_wedge half_size
        else:
            # DiscretePool
            for i in range(N):
                s,_ = pool.param_indices[i]
                pool.params.data[s:s+3] = torch.from_numpy(pts[sel[i]].astype(np.float32))
                t = pool.types[i]
                if t=='sphere': pool.params.data[s+3]=ts*0.5
                elif t=='cylinder':
                    pool.params.data[s+3]=ts*0.3; pool.params.data[s+4]=ts*0.5
                elif t in ('block','wedge','corner_wedge'):
                    for j in range(3): pool.params.data[s+3+j]=ts*0.4

def spatial_group_init(model, mesh_sdf):
    try: from scipy.cluster.vq import kmeans2
    except ImportError: return
    centers = np.array([model.pool.get_center(i).cpu().numpy() for i in range(model.N)])
    nc = min(model.K, model.N//2)
    _, labels = kmeans2(centers, nc, minit='++')
    with torch.no_grad():
        for i in range(model.N):
            model.m_pos.data[i, labels[i]] = -1.0
        for k in range(nc):
            if (labels==k).sum()>0: model.act.data[k] = 1.0

# SYNTHETIC TARGETS
def target_complex_csg(p):
    d=torch.abs(p)-0.5
    box=torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    c=torch.tensor([0.3,0.3,0.3],device=p.device)
    sph=torch.sqrt(((p-c)**2).sum(-1)+1e-8)-0.35
    u=torch.minimum(box,sph)
    cy=torch.sqrt(p[:,0]**2+p[:,2]**2+1e-8)-0.15
    cx=torch.sqrt(p[:,1]**2+p[:,2]**2+1e-8)-0.15
    return (torch.maximum(torch.maximum(u,-cy),-cx)<0).float()
def target_steinmetz(p):
    cx=torch.sqrt(p[:,1]**2+p[:,2]**2+1e-8)-0.5
    cy=torch.sqrt(p[:,0]**2+p[:,2]**2+1e-8)-0.5
    cz=torch.sqrt(p[:,0]**2+p[:,1]**2+1e-8)-0.5
    return (torch.maximum(torch.maximum(cx,cy),cz)<0).float()
def target_sphere_minus_cyl(p):
    s=torch.sqrt((p**2).sum(-1)+1e-8)-0.7
    c=torch.sqrt(p[:,0]**2+p[:,2]**2+1e-8)-0.2
    return (torch.maximum(s,-c)<0).float()
TARGETS = {'complex_csg':target_complex_csg, 'steinmetz':target_steinmetz,
           'sphere_minus_cylinder':target_sphere_minus_cyl}

# CHECKPOINTING
@dataclass
class TState:
    epoch:int=0; losses:List[float]=field(default_factory=list)
    ious:List[float]=field(default_factory=list); best_iou:float=0.0; best_epoch:int=0

class CkptMgr:
    def __init__(self, folder):
        self.folder=folder; os.makedirs(folder, exist_ok=True)
        self.ckpt=os.path.join(folder,'checkpoint.pt')
        self.best=os.path.join(folder,'best_model.pt')
    def save(self, model, optim, state):
        torch.save({'model':model.state_dict(),'optim':optim.state_dict(),
                     'state':{'epoch':state.epoch,'losses':state.losses[-5000:],
                              'ious':state.ious,'best_iou':state.best_iou,
                              'best_epoch':state.best_epoch}}, self.ckpt)
        if state.ious and state.ious[-1]>=state.best_iou:
            torch.save(model.state_dict(), self.best)
    def load(self, model, optim=None):
        if not os.path.exists(self.ckpt): return None
        ck=torch.load(self.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        if optim:
            try: optim.load_state_dict(ck['optim'])
            except: pass
        return TState(**ck['state'])
    def load_best(self, model):
        if os.path.exists(self.best):
            model.load_state_dict(torch.load(self.best, map_location=device, weights_only=True))
            return True
        return False

# TRAINING
def train(model, target_func, mesh_sdf, epochs, output_folder,
          lr=1e-3, num_samples=100000, resample_every=10,
          sparsity_warmup=3000, sparsity_rampup=5000,
          ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
          ba_start=15000, ba_end=40000, ba_max=5e-4,
          eval_every=200, save_every=2000, resolution=128,
          resume=True, desc="Training",
          use_temperature=False, temp_start=1.0, temp_end=20.0):
    """Unified training loop with temperature annealing for type selection."""
    os.makedirs(output_folder, exist_ok=True)
    ckpt = CkptMgr(output_folder); loss_fn = nn.MSELoss()
    eval_grid = sample_3D_grid(resolution); eval_gt = target_func(eval_grid)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    state = None
    if resume:
        state = ckpt.load(model, optimizer)
        if state and state.epoch >= epochs:
            print(f"  Already trained ({state.epoch} ep, best={state.best_iou:.4f})")
            ckpt.load_best(model); return state, eval_grid, eval_gt
        if state: print(f"  Resuming from epoch {state.epoch}")
    if state is None: state = TState()

    P, gt = None, None
    pbar = tqdm(range(state.epoch, epochs), initial=state.epoch, total=epochs,
                desc=f"  {desc}", leave=False)

    for epoch in pbar:
        # Temperature schedule
        progress = epoch / max(epochs-1, 1)
        temperature = temp_start + (temp_end - temp_start) * progress if use_temperature else 1.0

        if epoch % resample_every == 0 or P is None:
            if mesh_sdf is not None:
                pts, occ = mesh_sdf.sample_points(num_samples)
                P = torch.from_numpy(pts).to(device); gt = torch.from_numpy(occ).to(device)
            else:
                P = torch.rand(num_samples,3,device=device)*2-1; gt = target_func(P)

        pred = model(P, temperature=temperature)
        recon = loss_fn(pred, gt)

        # Regularization schedule
        if epoch<sparsity_warmup: sw=0.0
        elif epoch<sparsity_warmup+sparsity_rampup:
            sw=(epoch-sparsity_warmup)/sparsity_rampup
        else: sw=1.0
        if epoch<ba_start: bm=1.0
        elif epoch<ba_end:
            bm=1.0+(epoch-ba_start)/(ba_end-ba_start)*(ba_max/max(lb,1e-10)-1.0)
        else: bm=ba_max/max(lb,1e-10)

        struct = model.structure_loss(ls=ls*sw, lb=lb*bm*sw, lc=lc*sw, lgs=lgs*sw,
                                       temperature=temperature)
        loss = recon+struct

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        state.epoch=epoch+1; state.losses.append(loss.item())

        if (epoch+1)%eval_every==0:
            eval_temp = temp_end if use_temperature else 1.0
            with torch.no_grad():
                pf = batched_forward(model, eval_grid, temperature=eval_temp)
                iou = compute_iou(pf, eval_gt)
            state.ious.append(iou)
            if iou>state.best_iou:
                state.best_iou=iou; state.best_epoch=epoch+1
                torch.save(model.state_dict(), ckpt.best)
            s = model.get_stats()
            pbar.set_postfix(loss=f'{recon.item():.4f}', IoU=f'{iou:.3f}',
                           best=f'{state.best_iou:.3f}', g=s['groups_active'],
                           p=s['prims_used'])

        if (epoch+1)%save_every==0: ckpt.save(model, optimizer, state)

    ckpt.save(model, optimizer, state); ckpt.load_best(model)
    return state, eval_grid, eval_gt

# EVALUATION
def full_eval(model, eval_grid, eval_gt, output_folder, resolution,
              label="", eval_temperature=1.0):
    os.makedirs(output_folder, exist_ok=True)
    pred = batched_forward(model, eval_grid, temperature=eval_temperature)
    iou = compute_iou(pred,eval_gt); mse = ((pred-eval_gt)**2).mean().item()
    fs = compute_fscore(pred,eval_gt)
    mc_from_occ(os.path.join(output_folder,'mesh_soft.obj'), pred, resolution)
    dnf_iou=0.0
    if hasattr(model,'forward_hard_dnf'):
        with torch.no_grad():
            dp = torch.cat([model.forward_hard_dnf(b) for b in torch.split(eval_grid,32**3)],0)
        dnf_iou=compute_iou(dp,eval_gt)
        mc_from_occ(os.path.join(output_folder,'mesh_hard_dnf.obj'), dp, resolution)
    stats = model.get_stats() if hasattr(model,'get_stats') else {}
    results = {'iou':iou,'dnf_iou':dnf_iou,'mse':mse,'fscore':fs}
    results.update({k:v for k,v in stats.items() if not isinstance(v,(list,dict))})
    if 'type_dist' in stats: results['type_dist'] = stats['type_dist']
    with open(os.path.join(output_folder,'results.json'),'w') as f:
        json.dump(results, f, indent=2)
    print(f"  {label}: IoU={iou:.4f}, DNF={dnf_iou:.4f}, MSE={mse:.6f}, "
          f"F={fs:.4f}, prims={stats.get('prims_used','?')}, "
          f"groups={stats.get('groups_active','?')}")
    return results

# COMPARISON PLOT
def save_plot(results, output_path):
    methods = list(results.keys())
    n = len(methods)
    colors = ['#4C72B0','#55A868','#C44E52','#8172B2']
    fig, axes = plt.subplots(1, 2, figsize=(5+n*1.5, 5))
    for ax_i, (met, title) in enumerate([('iou','Soft IoU'),('dnf_iou','Hard DNF IoU')]):
        ax = axes[ax_i]
        vals = [results[m].get(met,0) for m in methods]
        bars = ax.bar(range(n), vals, color=colors[:n], edgecolor='white')
        ax.set_xticks(range(n))
        ax.set_xticklabels([m.replace('_','\n') for m in methods], fontsize=8)
        ax.set_ylabel('IoU'); ax.set_title(title)
        if any(v>0 for v in vals):
            ax.set_ylim(min(v for v in vals if v>0)-0.03, min(max(vals)+0.015,1.0))
        for b,v in zip(bars,vals):
            if v>0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                            f'{v:.4f}', ha='center', fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()

# MAIN
def main():
    parser = argparse.ArgumentParser(description='V3: Bc Type Selection Hypothesis')
    parser.add_argument('--mesh', type=str, default=None)
    parser.add_argument('--target', type=str, default='complex_csg')
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--baseline-prims', type=int, default=12,
                        help='Prims per type for baseline (×5 types)')
    parser.add_argument('--bc-prims', type=int, default=50,
                        help='Total prims for Bc/softmax methods')
    parser.add_argument('--groups', type=int, default=14)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--output', type=str, default='./benchmark_v3/')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--omega', type=float, default=10.0,
                        help='Sine frequency for Bc type selection')
    args = parser.parse_args()
    resume = not args.no_resume

    # Target
    mesh_sdf = None
    if args.mesh and os.path.exists(args.mesh):
        mesh_sdf = MeshSDF(args.mesh)
        target_func = lambda p: torch.from_numpy(
            (mesh_sdf.query(p.cpu().numpy())<0).astype(np.float32)).to(device)
        target_name = os.path.splitext(os.path.basename(args.mesh))[0]
    elif args.target in TARGETS:
        target_func = TARGETS[args.target]; target_name = args.target
    else:
        print(f"No target. Use --mesh or --target {list(TARGETS.keys())}"); return

    OUT = os.path.join(args.output, target_name); os.makedirs(OUT, exist_ok=True)

    n_baseline = args.baseline_prims * 5
    print(f"\n{'═'*65}")
    print(f"  V3: Bc TYPE SELECTION — {target_name}")
    print(f"{'═'*65}")
    print(f"  A) Baseline: {n_baseline} prims ({args.baseline_prims}×5 types)")
    print(f"  G) Bc-type:  {args.bc_prims} prims (each has all 5 types, Bc+sine)")
    print(f"  H) Softmax:  {args.bc_prims} prims (each has all 5 types, weighted sum)")
    print(f"  Groups: {args.groups}, Epochs: {args.epochs}, ω={args.omega}")
    print(f"  Device: {device}")
    print(f"{'═'*65}\n")

    eval_grid = sample_3D_grid(args.resolution)
    eval_gt = target_func(eval_grid)
    if mesh_sdf:
        mc_from_sdf(os.path.join(OUT,'00_gt.obj'),
                     mesh_sdf.query_sdf_grid(args.resolution), args.resolution)
    else:
        mc_from_occ(os.path.join(OUT,'00_gt.obj'), eval_gt, args.resolution)

    reg = dict(sparsity_warmup=max(args.epochs//20,1500),
               sparsity_rampup=max(args.epochs//12,2000),
               ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
               ba_start=args.epochs//4, ba_end=int(args.epochs*0.75), ba_max=5e-4)
    all_results = {}

    # A) BASELINE
    print(f"\n{'─'*65}\n  A) BASELINE ({n_baseline} discrete prims)\n{'─'*65}")
    torch.manual_seed(42); np.random.seed(42)
    model_a = BaselineCSG(prims_per_type=args.baseline_prims, num_groups=args.groups).to(device)
    if mesh_sdf: surface_init(model_a, mesh_sdf); spatial_group_init(model_a, mesh_sdf)
    t0=time.time()
    train(model_a, target_func, mesh_sdf, args.epochs,
          os.path.join(OUT,'A_baseline'),
          num_samples=args.samples, resolution=args.resolution,
          resume=resume, desc="Baseline", **reg)
    res_a = full_eval(model_a, eval_grid, eval_gt,
                      os.path.join(OUT,'A_baseline'), args.resolution, "BASELINE")
    res_a['time_s']=time.time()-t0
    all_results['A_baseline']=res_a; model_a.print_status()

    # G) BC-TYPE + SINE
    print(f"\n{'─'*65}\n  G) BC-TYPE + SINE ({args.bc_prims} prims, Bc product)\n{'─'*65}")
    torch.manual_seed(42); np.random.seed(42)
    model_g = BcTypeCSG(num_primitives=args.bc_prims, num_groups=args.groups,
                         omega=args.omega).to(device)
    if mesh_sdf: surface_init(model_g, mesh_sdf); spatial_group_init(model_g, mesh_sdf)
    t0=time.time()
    train(model_g, target_func, mesh_sdf, args.epochs,
          os.path.join(OUT,'G_bc_type'),
          num_samples=args.samples, resolution=args.resolution,
          resume=resume, desc="Bc-Type",
          use_temperature=True, temp_start=1.0, temp_end=20.0, **reg)
    res_g = full_eval(model_g, eval_grid, eval_gt,
                      os.path.join(OUT,'G_bc_type'), args.resolution,
                      "BC-TYPE", eval_temperature=20.0)
    res_g['time_s']=time.time()-t0
    all_results['G_bc_type']=res_g; model_g.print_status()

    # H) SOFTMAX-TYPE (control)
    print(f"\n{'─'*65}\n  H) SOFTMAX-TYPE ({args.bc_prims} prims, weighted sum)\n{'─'*65}")
    torch.manual_seed(42); np.random.seed(42)
    model_h = SoftmaxTypeCSG(num_primitives=args.bc_prims, num_groups=args.groups).to(device)
    if mesh_sdf: surface_init(model_h, mesh_sdf); spatial_group_init(model_h, mesh_sdf)
    t0=time.time()
    train(model_h, target_func, mesh_sdf, args.epochs,
          os.path.join(OUT,'H_softmax_type'),
          num_samples=args.samples, resolution=args.resolution,
          resume=resume, desc="Softmax-Type",
          use_temperature=True, temp_start=1.0, temp_end=20.0, **reg)
    res_h = full_eval(model_h, eval_grid, eval_gt,
                      os.path.join(OUT,'H_softmax_type'), args.resolution,
                      "SOFTMAX-TYPE", eval_temperature=20.0)
    res_h['time_s']=time.time()-t0
    all_results['H_softmax_type']=res_h; model_h.print_status()

    # COMPARISON
    print(f"\n{'━'*65}")
    print(f"  COMPARISON — {target_name}")
    print(f"{'━'*65}")
    print(f"\n  {'Method':<22} {'Soft IoU':>9} {'DNF IoU':>9} {'MSE':>10} "
          f"{'Prims':>6} {'Groups':>7} {'Time':>7}")
    print(f"  {'─'*72}")
    for m in all_results:
        r=all_results[m]
        print(f"  {m:<22} {r['iou']:>9.4f} {r.get('dnf_iou',0):>9.4f} "
              f"{r['mse']:>10.6f} "
              f"{r.get('prims_used','?'):>6} {r.get('groups_active','?'):>7} "
              f"{r.get('time_s',0):>6.0f}s")
    print(f"  {'─'*72}")

    # Key comparisons
    print(f"\n  KEY HYPOTHESIS TESTS:")
    if 'A_baseline' in all_results and 'G_bc_type' in all_results:
        d = all_results['G_bc_type']['iou'] - all_results['A_baseline']['iou']
        print(f"    G vs A (Bc-type vs discrete pool): {d:+.4f} IoU")
        if d > 0:
            print(f"    → ✓ Continuous Bc type selection beats overcomplete discrete pool")
        else:
            print(f"    → ✗ Overcomplete pool wins — type selection not needed")

    if 'G_bc_type' in all_results and 'H_softmax_type' in all_results:
        d = all_results['G_bc_type']['iou'] - all_results['H_softmax_type']['iou']
        print(f"    G vs H (Bc vs softmax blending): {d:+.4f} IoU")
        if d > 0:
            print(f"    → ✓ Bc gradient shielding essential (matches NAS prediction)")
        else:
            print(f"    → ✗ Softmax works as well as Bc (shielding not needed here)")

    if 'G_bc_type' in all_results:
        s = all_results['G_bc_type']
        td = s.get('type_dist', {})
        ent = s.get('type_entropy', -1)
        print(f"\n    Bc-type learned types: {td}")
        print(f"    Type entropy: {ent:.3f} (0=committed, 1.6=uniform)")
        if len(td) >= 3:
            print(f"    → Types differentiated: {len(td)} distinct types used")
        else:
            print(f"    → Poor differentiation: only {len(td)} types emerged")

    print(f"{'━'*65}")

    save_plot(all_results, os.path.join(OUT,'comparison_v3.png'))
    with open(os.path.join(OUT,'results_v3.json'),'w') as f:
        json.dump({m:{k:v for k,v in r.items() if not isinstance(v,torch.Tensor)}
                   for m,r in all_results.items()}, f, indent=2)

    print(f"\n  Output: {OUT}/")
    print("Done!")

if __name__=='__main__':
    main()
