#!/usr/bin/env python3
"""
  N-ary fuzzy CSG, v4: zonotope primitive family.

Each primitive is a rounded zonotope: the Minkowski sum of a ball
of radius R with K line segments along learned directions eᵢ.

Support function:  h(n) = R + Σᵢ aᵢ |n · eᵢ|

  R > 0       : rounding radius (sphere component)
  aᵢ >= 0      : generator extents (line segment half-lengths)
  eᵢ ∈ S²     : generator directions (learned)

Sphere:   all aᵢ ~ 0, R large
Box:      3 orthogonal eᵢ with large aᵢ, R small
Capsule:  1 large aᵢ, R large
Cylinder: 1 large aᵢ (height), R moderate (approximation)

Moving between these shapes changes only a few parameters.

Convex implicit via LSE at D fixed evaluation directions:
  φ(x) = (1/τ) [log Σ_d exp(τ(n_d · x_local - h(n_d))) - log D]

Stage 1: single-primitive fitting
Stage 2: full n-ary fuzzy aggregation

Usage:
  python nary_benchmark_v4.py --stage 1 --epochs 5000 --K 3
  python nary_benchmark_v4.py --stage 1 --epochs 5000 --K 5
  python nary_benchmark_v4.py --stage 2 --mesh ./bunny.obj --epochs 30000
"""

import os, sys, time, math, json, argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

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


# UTILITIES
MIN_SIZE = 0.02

def fibonacci_sphere(n):
    golden = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n, dtype=torch.float32)
    theta = 2 * math.pi * indices / golden
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)

def sample_3D_grid(resolution):
    idx = torch.linspace(-1, 1, resolution, device=device)
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

def euler_to_R(a):
    cx,sx = torch.cos(a[0]),torch.sin(a[0])
    cy,sy = torch.cos(a[1]),torch.sin(a[1])
    cz,sz = torch.cos(a[2]),torch.sin(a[2])
    R = torch.zeros(3,3,device=a.device,dtype=a.dtype)
    R[0,0]=cy*cz; R[0,1]=cz*sx*sy-cx*sz; R[0,2]=cx*cz*sy+sx*sz
    R[1,0]=cy*sz; R[1,1]=cx*cz+sx*sy*sz; R[1,2]=cx*sy*sz-cz*sx
    R[2,0]=-sy;   R[2,1]=cy*sx;          R[2,2]=cx*cy
    return R

def inv_softplus(x):
    if x > 20.0: return x
    return math.log(math.expm1(x))

def normalize_to_sphere(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


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


# ZONOTOPE PRIMITIVE
class ZonotopePrimitive(nn.Module):
    """
    A rounded zonotope: Minkowski sum of K line segments + ball.

    Support function: h(n) = R + Σᵢ aᵢ |n · eᵢ|

    Parameters:
      center    [3]    : position
      rotation  [3]    : Euler angles
      R_logit   scalar : rounding radius (through softplus)
      axis_raw  [K,3]  : generator directions (projected to S²)
      ext_logit [K]    : generator extents (through softplus)
      sharpness scalar : sigmoid sharpness for occupancy

    Total: 8 + 4K parameters per primitive.
    K=3 -> 20 params. K=5 -> 28 params.
    """
    def __init__(self, K=3, D=64, tau=30.0, init_radius=0.2):
        super().__init__()
        self.K_gen = K
        self.D = D
        self.tau = tau
        self.log_D = math.log(D)

        # Fixed evaluation directions on S²
        dirs = fibonacci_sphere(D)
        self.register_buffer('directions', dirs)  # [D, 3]

        # Pose
        self.center = nn.Parameter(torch.zeros(3))
        self.rotation = nn.Parameter(torch.randn(3) * 0.1)

        # Rounding radius R > 0
        # Initialize so R ~ init_radius (start as sphere)
        self.R_logit = nn.Parameter(torch.tensor(inv_softplus(init_radius)))

        # Generator axes on S² (learned directions)
        # Initialize near coordinate axes for K>=3, spread for K<3
        self.axis_raw = nn.Parameter(self._init_axes(K))

        # Generator extents aᵢ >= 0
        # Initialize near zero (start as sphere, grow into box/etc)
        self.ext_logit = nn.Parameter(torch.zeros(K) - 1.0)  # softplus(-1) ~ 0.31

        # Sigmoid sharpness
        self.sharpness = nn.Parameter(torch.tensor(20.0))

    def _init_axes(self, K):
        """Initialize generator axes. First 3 near coordinate axes."""
        axes = torch.randn(K, 3) * 0.3
        if K >= 1: axes[0] = torch.tensor([1.0, 0.0, 0.0]) + torch.randn(3) * 0.05
        if K >= 2: axes[1] = torch.tensor([0.0, 1.0, 0.0]) + torch.randn(3) * 0.05
        if K >= 3: axes[2] = torch.tensor([0.0, 0.0, 1.0]) + torch.randn(3) * 0.05
        # Additional axes start diagonal-ish
        for i in range(3, K):
            axes[i] = torch.randn(3)
        return axes

    def get_R(self):
        return F.softplus(self.R_logit)

    def get_axes(self):
        return normalize_to_sphere(self.axis_raw)  # [K, 3]

    def get_extents(self):
        return F.softplus(self.ext_logit)  # [K]

    def support_at_dirs(self):
        """Evaluate support function h(n_d) at all D directions.
        h(n) = R + Σᵢ aᵢ |n · eᵢ|
        """
        R = self.get_R()             # scalar
        e = self.get_axes()           # [K, 3]
        a = self.get_extents()        # [K]

        # |n_d · eᵢ| for all directions and generators
        dots = torch.matmul(self.directions, e.T).abs()  # [D, K]

        # h(n_d) = R + Σᵢ aᵢ |n_d · eᵢ|
        h = R + torch.matmul(dots, a)  # [D]
        return h

    def implicit(self, p, tau=None):
        """Convex implicit. Negative inside, positive outside."""
        if tau is None: tau = self.tau

        rot = euler_to_R(self.rotation)
        pl = torch.matmul(p - self.center, rot.T)  # [P, 3]

        h = self.support_at_dirs()  # [D]

        dots = torch.matmul(pl, self.directions.T)  # [P, D]

        exponents = tau * (dots - h.unsqueeze(0))  # [P, D]
        max_exp = exponents.max(dim=1, keepdim=True)[0]
        phi = (1.0 / tau) * (
            torch.log(torch.exp(exponents - max_exp).sum(dim=1) + 1e-10)
            + max_exp.squeeze(1)
            - self.log_D
        )
        return phi

    def occupancy(self, p, tau=None):
        phi = self.implicit(p, tau)
        sharp = self.sharpness.abs().clamp(min=1.0)
        return torch.sigmoid(-phi * sharp)

    def size_loss(self):
        R = self.get_R()
        a = self.get_extents()
        return R.pow(2) + a.pow(2).sum()

    def describe(self):
        R = self.get_R().detach().item()
        e = self.get_axes().detach()
        a = self.get_extents().detach()

        # Check orthogonality of first 3 axes
        ortho = 0.0
        if self.K_gen >= 3:
            dots = torch.matmul(e[:3], e[:3].T)
            mask = 1.0 - torch.eye(3, device=dots.device)
            ortho = (dots.abs() * mask).sum().item() / 6.0

        active = (a > 0.05).sum().item()
        total_extent = a.sum().item()
        ratio = total_extent / (R + 1e-8)

        if ratio < 0.1:
            label = 'sphere'
        elif active <= 1 and ratio > 0.5:
            label = 'capsule'
        elif active == 2:
            label = 'plate'
        elif active >= 3 and ratio > 1.0:
            label = 'box-like'
        else:
            label = 'zonotope'

        return {
            'R': R,
            'extents': [f'{v:.3f}' for v in a.tolist()],
            'active_gen': int(active),
            'extent_sum': total_extent,
            'R_ratio': ratio,
            'ortho_err': ortho,
            'label': label
        }


# ZONOTOPE POOL
class ZonotopePool(nn.Module):
    def __init__(self, num_primitives, K=3, D=64, tau=30.0):
        super().__init__()
        self.N = num_primitives
        self.K = K
        self.primitives = nn.ModuleList([
            ZonotopePrimitive(K=K, D=D, tau=tau)
            for _ in range(num_primitives)
        ])

    def get_occ(self, p, i, tau=None):
        return self.primitives[i].occupancy(p, tau)

    def get_all_occs(self, p, tau=None):
        return torch.stack([self.get_occ(p, i, tau) for i in range(self.N)], dim=-1)

    def get_center(self, i):
        return self.primitives[i].center.data

    def total_size_loss(self):
        return sum(p.size_loss() for p in self.primitives) / self.N

    def describe_shape(self, i):
        return self.primitives[i].describe()['label']


# DISCRETE-TYPE PRIMITIVES (baseline)
def sdf_block_local(pl, half_size):
    h = half_size.abs()+MIN_SIZE; d = torch.abs(pl)-h
    return torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
def sdf_sphere_local(pl, r):
    return torch.sqrt((pl**2).sum(-1)+1e-8)-(r.abs()+MIN_SIZE)
def sdf_cylinder_local(pl, r, hh):
    r_=r.abs()+MIN_SIZE; hh_=hh.abs()+MIN_SIZE
    dr=torch.sqrt(pl[:,0]**2+pl[:,2]**2+1e-8)-r_
    dh=torch.abs(pl[:,1])-hh_
    return torch.sqrt(torch.clamp(dr,min=0)**2+torch.clamp(dh,min=0)**2+1e-8)+\
           torch.clamp(torch.maximum(dr,dh),max=0)
def sdf_wedge_local(pl, half_size):
    h=half_size.abs()+MIN_SIZE; d=torch.abs(pl)-h
    box=torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    inv=1.0/torch.sqrt(1.0/(h[1]**2)+1.0/(h[2]**2)+1e-8)
    plane=(pl[:,1]/h[1]+pl[:,2]/h[2]-1.0)*inv
    return torch.maximum(box,plane)
def sdf_corner_wedge_local(pl, half_size):
    h=half_size.abs()+MIN_SIZE; hx,hy,hz=h[0],h[1],h[2]
    d=torch.abs(pl)-h
    box=torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    inv1=1.0/torch.sqrt(1.0/(hx**2)+1.0/(hy**2)+1e-8)
    p1=(pl[:,0]/hx+pl[:,1]/hy-1.0)*inv1
    inv2=1.0/torch.sqrt(1.0/(hz**2)+1.0/(hy**2)+1e-8)
    p2=(pl[:,2]/hz+pl[:,1]/hy-1.0)*inv2
    return torch.maximum(torch.maximum(box,p1),p2)
def compute_prim_sdf(ptype, pr, pts):
    if ptype=='block':    return sdf_block_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
    if ptype=='sphere':   return sdf_sphere_local(pts-pr[:3],pr[3])
    if ptype=='cylinder':
        pl=torch.matmul(pts-pr[:3],euler_to_R(pr[5:8]).T)
        return sdf_cylinder_local(pl,pr[3],pr[4])
    if ptype=='wedge':    return sdf_wedge_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
    if ptype=='corner_wedge': return sdf_corner_wedge_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])

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


# CSG MODELS
class BaselineCSG(nn.Module):
    def __init__(self, prims_per_type=12, prim_types=None, num_groups=16, max_group_size=8):
        super().__init__()
        if prim_types is None:
            prim_types = ['block','sphere','cylinder','wedge','corner_wedge']
        all_types = []
        for t in prim_types: all_types.extend([t]*prims_per_type)
        self.pool = DiscretePool(all_types, 0.3)
        self.N = len(all_types); self.K = num_groups; self.max_group_size = max_group_size
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
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg); a = torch.sigmoid(self.act)
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


class ZonotopeCSG(nn.Module):
    """N-ary fuzzy aggregation with zonotope pool."""
    def __init__(self, num_primitives=50, num_groups=16, K=3, D=64,
                 tau=30.0, max_group_size=8):
        super().__init__()
        self.pool = ZonotopePool(num_primitives, K=K, D=D, tau=tau)
        self.N = num_primitives; self.K_groups = num_groups; self.max_group_size = max_group_size
        self.m_pos = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 3.0)
        self.m_neg = nn.Parameter(torch.randn(self.N, num_groups)*0.3 - 5.0)
        self.act = nn.Parameter(torch.randn(num_groups)*0.3)

    def forward(self, p, tau=None, **kw):
        occs = self.pool.get_all_occs(p, tau)
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg)
        a = torch.sigmoid(self.act); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def forward_hard_dnf(self, p, tau=None, thr=0.5):
        occs = self.pool.get_all_occs(p, tau)
        mp = (torch.sigmoid(self.m_pos)>thr).float()
        mn = (torch.sigmoid(self.m_neg)>thr).float()
        a = (torch.sigmoid(self.act)>thr).float(); o = occs.unsqueeze(-1)
        fp = 1.0-mp.unsqueeze(0)*(1.0-o); fn = 1.0-mn.unsqueeze(0)*o
        lg = torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag = a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

    def structure_loss(self, ls=1e-4, lb=1e-5, lc=1e-4, lgs=1e-5,
                       l_size=1e-5, **kw):
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg); a = torch.sigmoid(self.act)
        sparse = ls*(mp.sum()+mn.sum()+a.sum())
        binary = lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum())
        conflict = lc*(mp*mn).sum()
        excess = torch.clamp(mp.sum(0)+mn.sum(0)-self.max_group_size, min=0)
        gs = lgs*(excess**2).sum()
        size = l_size * self.pool.total_size_loss()
        return sparse+binary+conflict+gs+size

    def get_stats(self, thr=0.3):
        mp = torch.sigmoid(self.m_pos).detach(); mn = torch.sigmoid(self.m_neg).detach()
        a = torch.sigmoid(self.act).detach()
        used = ((mp.max(1)[0]>thr)|(mn.max(1)[0]>thr)).sum().item()
        ag = (a>thr).sum().item()
        shapes = {}
        for i in range(self.N):
            if mp[i].max()>thr or mn[i].max()>thr:
                s = self.pool.describe_shape(i)
                shapes[s] = shapes.get(s,0)+1
        return {'prims_used':used, 'groups_active':ag, 'shape_dist':shapes, 'N':self.N}

    def print_status(self, thr=0.3):
        s = self.get_stats(thr)
        print(f"  Prims: {s['prims_used']}/{self.N}, Groups: {s['groups_active']}/{self.K_groups}")
        print(f"  Shapes: {s['shape_dist']}")
        mp = torch.sigmoid(self.m_pos).detach(); mn = torch.sigmoid(self.m_neg).detach()
        for i in range(min(5, self.N)):
            if mp[i].max() > thr or mn[i].max() > thr:
                d = self.pool.primitives[i].describe()
                print(f"    P{i}: R={d['R']:.3f} ext={d['extents']} "
                      f"gen={d['active_gen']}/{self.pool.K} -> {d['label']}")


# STAGE 1: SINGLE-PRIMITIVE FITTING
def make_target_sdf(shape, half=0.3):
    if shape == 'sphere':
        def f(p): return (torch.sqrt((p**2).sum(-1)+1e-8) - half < 0).float()
    elif shape == 'box':
        def f(p):
            d = torch.abs(p) - half
            sdf = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
            return (sdf < 0).float()
    elif shape == 'cylinder':
        def f(p):
            dr = torch.sqrt(p[:,0]**2+p[:,2]**2+1e-8) - half*0.7
            dh = torch.abs(p[:,1]) - half
            sdf = torch.sqrt(torch.clamp(dr,min=0)**2+torch.clamp(dh,min=0)**2+1e-8)+\
                  torch.clamp(torch.maximum(dr,dh),max=0)
            return (sdf < 0).float()
    elif shape == 'wedge':
        def f(p):
            h = half; d = torch.abs(p) - h
            box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
            inv = 1.0/math.sqrt(2.0/(h**2))
            plane = (p[:,1]/h+p[:,2]/h-1.0)*inv
            return (torch.maximum(box, plane) < 0).float()
    elif shape == 'corner_wedge':
        def f(p):
            h = half; d = torch.abs(p) - h
            box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
            inv = 1.0/math.sqrt(2.0/(h**2))
            p1 = (p[:,0]/h+p[:,1]/h-1.0)*inv
            p2 = (p[:,2]/h+p[:,1]/h-1.0)*inv
            return (torch.maximum(torch.maximum(box,p1),p2) < 0).float()
    else:
        raise ValueError(f"Unknown: {shape}")
    return f


def run_stage1(args):
    shapes = ['sphere', 'box', 'cylinder', 'wedge', 'corner_wedge']
    OUT = os.path.join(args.output, 'stage1')
    os.makedirs(OUT, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  STAGE 1: ZONOTOPE SINGLE-PRIMITIVE FITTING")
    print(f"{'═'*65}")
    print(f"  K={args.K} generators, D={args.D} eval dirs, τ={args.tau}")
    print(f"  Params per prim: {8 + 4*args.K}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'═'*65}\n")

    resolution = 96
    eval_grid = sample_3D_grid(resolution)
    results = {}

    for shape in shapes:
        print(f"\n  {'─'*50}\n  {shape}\n  {'─'*50}")
        torch.manual_seed(42)
        target_fn = make_target_sdf(shape)
        eval_gt = target_fn(eval_grid)
        mc_from_occ(os.path.join(OUT, f'target_{shape}.obj'), eval_gt, resolution)

        prim = ZonotopePrimitive(K=args.K, D=args.D, tau=args.tau,
                                  init_radius=0.2).to(device)
        optimizer = torch.optim.Adam(prim.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=2e-4)
        loss_fn = nn.MSELoss()
        best_iou = 0.0

        pbar = tqdm(range(args.epochs), desc=f"  {shape}", leave=False)
        for epoch in pbar:
            P = torch.rand(50000, 3, device=device) * 2 - 1
            gt = target_fn(P)
            pred = prim.occupancy(P)
            recon = loss_fn(pred, gt)
            reg = 1e-5 * prim.size_loss()
            loss = recon + reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prim.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    pf = batched_forward(lambda p, **kw: prim.occupancy(p), eval_grid)
                    iou = compute_iou(pf, eval_gt)
                if iou > best_iou: best_iou = iou
                pbar.set_postfix(IoU=f'{iou:.3f}', best=f'{best_iou:.3f}')

        with torch.no_grad():
            pf = batched_forward(lambda p, **kw: prim.occupancy(p), eval_grid)
            iou = compute_iou(pf, eval_gt)
            mc_from_occ(os.path.join(OUT, f'fit_{shape}.obj'), pf, resolution)

        desc = prim.describe()
        print(f"  {shape}: IoU={iou:.4f}, R={desc['R']:.3f}, ext={desc['extents']}, "
              f"gen={desc['active_gen']}/{args.K} -> {desc['label']}")
        results[shape] = {'iou': iou, **desc}

    print(f"\n{'━'*65}")
    print(f"  STAGE 1 SUMMARY (K={args.K}, D={args.D}, τ={args.tau})")
    print(f"{'━'*65}")
    print(f"  {'Shape':<16} {'IoU':>8} {'R':>6} {'Extents':>24} {'Gen':>5} {'Label':>12}")
    print(f"  {'─'*75}")
    for shape in shapes:
        r = results[shape]
        ext = r.get('extents', ['?'])
        print(f"  {shape:<16} {r['iou']:>8.4f} {r['R']:>6.3f} {str(ext):>24} "
              f"{r['active_gen']:>3}/{args.K} {r['label']:>12}")
    print(f"  {'─'*75}")

    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return results


# STAGE 2: FULL N-ARY BENCHMARK
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

def surface_init(model, mesh_sdf):
    N = model.N
    pts, _ = trimesh.sample.sample_surface(mesh_sdf.mesh, N*3)
    sel = [0]; dists = np.full(len(pts), np.inf)
    for _ in range(N-1):
        d = np.linalg.norm(pts-pts[sel[-1]], axis=1)
        dists = np.minimum(dists, d); sel.append(dists.argmax())
    with torch.no_grad():
        pool = model.pool
        if hasattr(pool, 'primitives') and hasattr(pool.primitives[0], 'R_logit'):
            for i in range(N):
                pool.primitives[i].center.data = torch.from_numpy(
                    pts[sel[i]].astype(np.float32)).to(device)
        else:
            bbox = mesh_sdf.mesh.bounding_box.extents.max()
            ts = bbox/(N**(1/3))
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
    pool = model.pool
    if hasattr(pool, 'primitives'):
        centers = np.array([pool.primitives[i].center.data.cpu().numpy() for i in range(model.N)])
    else:
        centers = np.array([pool.get_center(i).cpu().numpy() for i in range(model.N)])
    nc = min(model.K_groups if hasattr(model,'K_groups') else model.K, model.N//2)
    _, labels = kmeans2(centers, nc, minit='++')
    with torch.no_grad():
        for i in range(model.N):
            model.m_pos.data[i, labels[i]] = -1.0
        for k in range(nc):
            if (labels==k).sum()>0: model.act.data[k] = 1.0

def train_stage2(model, target_func, mesh_sdf, epochs, output_folder,
                 lr=1e-3, num_samples=100000, resample_every=10,
                 sparsity_warmup=3000, sparsity_rampup=5000,
                 ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
                 ba_start=15000, ba_end=40000, ba_max=5e-4,
                 eval_every=200, save_every=2000, resolution=128,
                 resume=True, desc="Training",
                 tau_start=15.0, tau_end=50.0):
    os.makedirs(output_folder, exist_ok=True)
    ckpt = CkptMgr(output_folder); loss_fn = nn.MSELoss()
    eval_grid = sample_3D_grid(resolution); eval_gt = target_func(eval_grid)
    uses_tau = isinstance(model, ZonotopeCSG)
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
        progress = epoch / max(epochs-1, 1)
        tau = tau_start + (tau_end - tau_start) * progress if uses_tau else None
        if epoch % resample_every == 0 or P is None:
            if mesh_sdf is not None:
                pts, occ = mesh_sdf.sample_points(num_samples)
                P = torch.from_numpy(pts).to(device); gt = torch.from_numpy(occ).to(device)
            else:
                P = torch.rand(num_samples,3,device=device)*2-1; gt = target_func(P)
        pred = model(P, tau=tau) if uses_tau else model(P)
        recon = loss_fn(pred, gt)
        if epoch<sparsity_warmup: sw=0.0
        elif epoch<sparsity_warmup+sparsity_rampup: sw=(epoch-sparsity_warmup)/sparsity_rampup
        else: sw=1.0
        if epoch<ba_start: bm=1.0
        elif epoch<ba_end: bm=1.0+(epoch-ba_start)/(ba_end-ba_start)*(ba_max/max(lb,1e-10)-1.0)
        else: bm=ba_max/max(lb,1e-10)
        struct = model.structure_loss(ls=ls*sw, lb=lb*bm*sw, lc=lc*sw, lgs=lgs*sw)
        loss = recon+struct
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        state.epoch=epoch+1; state.losses.append(loss.item())
        if (epoch+1)%eval_every==0:
            with torch.no_grad():
                pf = batched_forward(model, eval_grid, tau=tau_end) if uses_tau else batched_forward(model, eval_grid)
                iou = compute_iou(pf, eval_gt)
            state.ious.append(iou)
            if iou>state.best_iou:
                state.best_iou=iou; state.best_epoch=epoch+1
                torch.save(model.state_dict(), ckpt.best)
            s = model.get_stats()
            pbar.set_postfix(loss=f'{recon.item():.4f}', IoU=f'{iou:.3f}',
                           best=f'{state.best_iou:.3f}', g=s['groups_active'], p=s['prims_used'])
        if (epoch+1)%save_every==0: ckpt.save(model, optimizer, state)
    ckpt.save(model, optimizer, state); ckpt.load_best(model)
    return state, eval_grid, eval_gt

def full_eval(model, eval_grid, eval_gt, output_folder, resolution, label="", tau=50.0):
    os.makedirs(output_folder, exist_ok=True)
    uses_tau = isinstance(model, ZonotopeCSG)
    pred = batched_forward(model, eval_grid, tau=tau) if uses_tau else batched_forward(model, eval_grid)
    iou = compute_iou(pred,eval_gt); mse = ((pred-eval_gt)**2).mean().item()
    fs = compute_fscore(pred,eval_gt)
    mc_from_occ(os.path.join(output_folder,'mesh_soft.obj'), pred, resolution)
    dnf_iou=0.0
    if hasattr(model,'forward_hard_dnf'):
        with torch.no_grad():
            if uses_tau:
                dp = torch.cat([model.forward_hard_dnf(b, tau=tau) for b in torch.split(eval_grid,32**3)],0)
            else:
                dp = torch.cat([model.forward_hard_dnf(b) for b in torch.split(eval_grid,32**3)],0)
        dnf_iou=compute_iou(dp,eval_gt)
        mc_from_occ(os.path.join(output_folder,'mesh_hard_dnf.obj'), dp, resolution)
    stats = model.get_stats() if hasattr(model,'get_stats') else {}
    results = {'iou':iou,'dnf_iou':dnf_iou,'mse':mse,'fscore':fs}
    results.update({k:v for k,v in stats.items() if not isinstance(v,(list,dict))})
    if 'type_dist' in stats: results['type_dist'] = stats['type_dist']
    if 'shape_dist' in stats: results['shape_dist'] = stats['shape_dist']
    with open(os.path.join(output_folder,'results.json'),'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  {label}: IoU={iou:.4f}, DNF={dnf_iou:.4f}, MSE={mse:.6f}, "
          f"F={fs:.4f}, prims={stats.get('prims_used','?')}, "
          f"groups={stats.get('groups_active','?')}")
    return results

def run_stage2(args):
    mesh_sdf = None
    if args.mesh and os.path.exists(args.mesh):
        mesh_sdf = MeshSDF(args.mesh)
        target_func = lambda p: torch.from_numpy(
            (mesh_sdf.query(p.cpu().numpy())<0).astype(np.float32)).to(device)
        target_name = os.path.splitext(os.path.basename(args.mesh))[0]
    else:
        print("Stage 2 requires --mesh"); return
    OUT = os.path.join(args.output, target_name); os.makedirs(OUT, exist_ok=True)
    n_baseline = args.baseline_prims * 5; resume = not args.no_resume
    print(f"\n{'═'*65}")
    print(f"  V4: ZONOTOPE FAMILY — {target_name}")
    print(f"{'═'*65}")
    print(f"  A) Baseline:  {n_baseline} prims ({args.baseline_prims}×5 types)")
    print(f"  I) Zonotope:  {args.zono_prims} prims (K={args.K} gen, D={args.D})")
    print(f"  Groups: {args.groups}, Epochs: {args.epochs}")
    print(f"  τ schedule: {args.tau_init} → {args.tau_max}")
    print(f"  Device: {device}")
    print(f"{'═'*65}\n")
    eval_grid = sample_3D_grid(args.resolution); eval_gt = target_func(eval_grid)
    if mesh_sdf:
        mc_from_sdf(os.path.join(OUT,'00_gt.obj'), mesh_sdf.query_sdf_grid(args.resolution), args.resolution)
    reg = dict(sparsity_warmup=max(args.epochs//20,1500), sparsity_rampup=max(args.epochs//12,2000),
               ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
               ba_start=args.epochs//4, ba_end=int(args.epochs*0.75), ba_max=5e-4)
    all_results = {}

    print(f"\n{'─'*65}\n  A) BASELINE ({n_baseline} discrete prims)\n{'─'*65}")
    torch.manual_seed(42); np.random.seed(42)
    model_a = BaselineCSG(prims_per_type=args.baseline_prims, num_groups=args.groups).to(device)
    surface_init(model_a, mesh_sdf); spatial_group_init(model_a, mesh_sdf)
    t0=time.time()
    train_stage2(model_a, target_func, mesh_sdf, args.epochs, os.path.join(OUT,'A_baseline'),
                 num_samples=args.samples, resolution=args.resolution, resume=resume, desc="Baseline", **reg)
    res_a = full_eval(model_a, eval_grid, eval_gt, os.path.join(OUT,'A_baseline'), args.resolution, "BASELINE")
    res_a['time_s']=time.time()-t0; all_results['A_baseline']=res_a; model_a.print_status()

    print(f"\n{'─'*65}\n  I) ZONOTOPE ({args.zono_prims} prims, K={args.K})\n{'─'*65}")
    torch.manual_seed(42); np.random.seed(42)
    model_i = ZonotopeCSG(num_primitives=args.zono_prims, num_groups=args.groups,
                           K=args.K, D=args.D, tau=args.tau).to(device)
    surface_init(model_i, mesh_sdf); spatial_group_init(model_i, mesh_sdf)
    t0=time.time()
    train_stage2(model_i, target_func, mesh_sdf, args.epochs, os.path.join(OUT,'I_zonotope'),
                 num_samples=args.samples, resolution=args.resolution, resume=resume, desc="Zonotope",
                 tau_start=args.tau_init, tau_end=args.tau_max, **reg)
    res_i = full_eval(model_i, eval_grid, eval_gt, os.path.join(OUT,'I_zonotope'),
                      args.resolution, "ZONOTOPE", tau=args.tau_max)
    res_i['time_s']=time.time()-t0; all_results['I_zonotope']=res_i; model_i.print_status()

    print(f"\n{'━'*65}\n  COMPARISON — {target_name}\n{'━'*65}")
    print(f"\n  {'Method':<22} {'Soft IoU':>9} {'DNF IoU':>9} {'MSE':>10} {'Prims':>6} {'Groups':>7} {'Time':>7}")
    print(f"  {'─'*72}")
    for m in all_results:
        r=all_results[m]
        print(f"  {m:<22} {r['iou']:>9.4f} {r.get('dnf_iou',0):>9.4f} {r['mse']:>10.6f} "
              f"{r.get('prims_used','?'):>6} {r.get('groups_active','?'):>7} {r.get('time_s',0):>6.0f}s")
    print(f"  {'─'*72}")
    with open(os.path.join(OUT,'results_v4.json'),'w') as f:
        json.dump({m:{k:v for k,v in r.items() if not isinstance(v,torch.Tensor)}
                   for m,r in all_results.items()}, f, indent=2, default=str)
    print(f"\n  Output: {OUT}/")


# MAIN
def main():
    parser = argparse.ArgumentParser(description='V4: Zonotope Family')
    parser.add_argument('--stage', type=int, default=1, choices=[1,2])
    parser.add_argument('--mesh', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--K', type=int, default=3, help='Generator axes per primitive')
    parser.add_argument('--D', type=int, default=64, help='Evaluation directions on S²')
    parser.add_argument('--tau', type=float, default=30.0, help='LSE sharpness')
    parser.add_argument('--baseline-prims', type=int, default=12)
    parser.add_argument('--zono-prims', type=int, default=50)
    parser.add_argument('--groups', type=int, default=16)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--output', type=str, default='./benchmark_v4/')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--tau-init', type=float, default=15.0)
    parser.add_argument('--tau-max', type=float, default=50.0)
    args = parser.parse_args()
    if args.stage == 1:
        run_stage1(args)
    elif args.stage == 2:
        if args.epochs == 5000: args.epochs = 30000
        run_stage2(args)

if __name__=='__main__':
    main()