#!/usr/bin/env python3
"""
Inverse CSG mesh benchmark (solver core used by the Chapter 3 budget sweep).

Purpose
-------
Benchmark for inverse primitive fitting methods in which training may use a
differentiable type relaxation, while the hard outputs of this script commit
each leaf/slot to exactly one primitive type:

    leaf_i(x) = q_{i, argmax_t type_weight_it}(x)

(csg_leaf_budget_ablation_stable_fixed.py adds the multi-hot hard extraction.)

Covers, on watertight/repairable meshes:
  * N-ary literal-DNF product aggregation;
  * CSG-Stump aggregation with WC/WI/WU connection matrices (reimplementation);
  * pairwise binary-tree CSG aggregation;
  * product-union vs clipped-sum leaf/union baselines;
  * discrete fixed-type primitive leaves vs learned type leaves;
  * sine-softmax type annealing (one-hot training relaxation);
  * sine-sigmoid multi-hot type relaxation as an explicit ablation only;
  * volume-equivalent radius normalization;
  * soft-vs-hard hardening gaps;
  * convergence over seeds and meshes;
  * optional GPU-resident training-point cache for fast long runs;
  * .obj export for target, soft, hard-leaf, and hard-program reconstructions.

Key definitions
---------------
A BC/learned-type leaf during SOFT training may be:

    sum:     leaf_i = sum_t w_it q_it
    product: leaf_i = 1 - prod_t (1 - w_it q_it)

where w_it comes from one of:

    sine_softmax:       w_it = softmax_t(beta * sin(omega * alpha_it))
    plain_softmax:      w_it = softmax_t(beta * alpha_it)
    sine_sigmoid_multi: w_it = sigmoid(beta * sin(omega * alpha_it))  # multi-hot ablation

The hard metrics of this script use exactly one primitive type per leaf:

    t_i* = argmax_t w_it,
    leaf_i^hard = q_i,t_i*

so no product-union ensemble remains inside a leaf in the hard IoU.

Typical commands
----------------
Quick smoke run:

    python siggraph_csg_benchmark_v3.py --meshes bunny --methods nary_bc_vol tree_bc_vol \
      --epochs 2000 --seeds 1 --resolution 48 --samples 40000 --batch-size 8192 \
      --export-objs --output ./csg_smoke

Main convergence benchmark:

    python siggraph_csg_benchmark_v3.py --meshes bunny fandisk --seeds 3 --epochs 30000 \
      --methods nary_discrete nary_bc_vol tree_discrete tree_bc_vol \
      --baseline-prims 12 --bc-prims 50 --groups 16 --resolution 128 \
      --samples 100000 --batch-size 32768 --output ./csg_bench --export-objs

Regenerate plots/summary from curves:

    python siggraph_csg_benchmark_v3.py --meshes bunny fandisk --plot-only --output ./csg_bench
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Marching-cubes fallback.
try:
    import mcubes  # type: ignore
except Exception:  # pragma: no cover
    from skimage import measure
    class _MCubesStub:
        @staticmethod
        def marching_cubes(vol, iso):
            v, f, _, _ = measure.marching_cubes(vol, level=iso)
            return v, f.astype(np.int32)
        @staticmethod
        def export_obj(vertices, triangles, filepath):
            with open(filepath, "w") as fh:
                for x, y, z in vertices:
                    fh.write(f"v {x} {y} {z}\n")
                for a, b, c in triangles:
                    fh.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")
    mcubes = _MCubesStub()

PI = math.pi
EPS = 1e-8
MIN_SIZE = 0.02
TYPE_NAMES = ["sphere", "box", "cylinder", "wedge", "corner_wedge"]
N_TYPES = len(TYPE_NAMES)
MAX_PARAM = 3

# Stable mesh registry; only used if the name is requested and file is missing.
MESH_REGISTRY = {
    "bunny": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
    "fandisk": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/fandisk.obj",
    "rocker-arm": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/rocker-arm.obj",
    "armadillo": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj",
    "spot": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj",
    "cow": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/cow.obj",
    "teapot": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj",
}

# Basic utilities


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sample_grid(resolution: int, device: torch.device) -> torch.Tensor:
    a = torch.linspace(-1.0, 1.0, resolution, device=device)
    x, y, z = torch.meshgrid(a, a, a, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)


def euler_to_R(a: torch.Tensor) -> torch.Tensor:
    """Euler angles (...,3) -> rotation matrices (...,3,3), row-vector convention."""
    cx, sx = torch.cos(a[..., 0]), torch.sin(a[..., 0])
    cy, sy = torch.cos(a[..., 1]), torch.sin(a[..., 1])
    cz, sz = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    R = torch.zeros(*a.shape[:-1], 3, 3, device=a.device, dtype=a.dtype)
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = cz * sx * sy - cx * sz
    R[..., 0, 2] = cx * cz * sy + sx * sz
    R[..., 1, 0] = cy * sz
    R[..., 1, 1] = cx * cz + sx * sy * sz
    R[..., 1, 2] = cx * sy * sz - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cx * cy
    return R


def pos_param(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw) + MIN_SIZE


def inv_pos_param(val: torch.Tensor) -> torch.Tensor:
    x = (val - MIN_SIZE).clamp_min(1e-5)
    return torch.log(torch.exp(x).clamp_max(1e6) - 1.0).clamp(-8, 8)

# SDF primitives

def sdf_sphere(pl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r = p[..., 0].abs() + MIN_SIZE
    return torch.sqrt((pl * pl).sum(-1) + EPS) - r


def sdf_box(pl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    h = p[..., :3].abs() + MIN_SIZE
    d = torch.abs(pl) - h
    outside = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + EPS)
    inside = torch.clamp(d.max(-1).values, max=0)
    return outside + inside


def sdf_cylinder(pl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r = p[..., 0].abs() + MIN_SIZE
    hh = p[..., 1].abs() + MIN_SIZE
    dr = torch.sqrt(pl[..., 0] ** 2 + pl[..., 2] ** 2 + EPS) - r
    dy = torch.abs(pl[..., 1]) - hh
    outside = torch.sqrt(torch.clamp(dr, min=0) ** 2 + torch.clamp(dy, min=0) ** 2 + EPS)
    inside = torch.clamp(torch.maximum(dr, dy), max=0)
    return outside + inside


def sdf_wedge(pl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    h = p[..., :3].abs() + MIN_SIZE
    box = sdf_box(pl, p)
    inv = 1.0 / torch.sqrt(1.0 / (h[..., 1] ** 2) + 1.0 / (h[..., 2] ** 2) + EPS)
    plane = (pl[..., 1] / h[..., 1] + pl[..., 2] / h[..., 2] - 1.0) * inv
    return torch.maximum(box, plane)


def sdf_corner_wedge(pl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    h = p[..., :3].abs() + MIN_SIZE
    box = sdf_box(pl, p)
    inv1 = 1.0 / torch.sqrt(1.0 / (h[..., 0] ** 2) + 1.0 / (h[..., 1] ** 2) + EPS)
    inv2 = 1.0 / torch.sqrt(1.0 / (h[..., 2] ** 2) + 1.0 / (h[..., 1] ** 2) + EPS)
    p1 = (pl[..., 0] / h[..., 0] + pl[..., 1] / h[..., 1] - 1.0) * inv1
    p2 = (pl[..., 2] / h[..., 2] + pl[..., 1] / h[..., 1] - 1.0) * inv2
    return torch.maximum(torch.maximum(box, p1), p2)


def sdf_by_type(type_id: int, pl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if type_id == 0:
        return sdf_sphere(pl, p)
    if type_id == 1:
        return sdf_box(pl, p)
    if type_id == 2:
        return sdf_cylinder(pl, p)
    if type_id == 3:
        return sdf_wedge(pl, p)
    if type_id == 4:
        return sdf_corner_wedge(pl, p)
    raise ValueError(type_id)


def volume_equiv_radius(type_id: int, p: torch.Tensor) -> torch.Tensor:
    """Return equivalent sphere radius r_eff=(3V/4pi)^(1/3)."""
    pp = p.abs() + MIN_SIZE
    if type_id == 0:  # sphere
        return pp[..., 0]
    if type_id == 1:  # box half sizes: V=8abc => r_eff=(6abc/pi)^(1/3)
        a, b, c = pp[..., 0], pp[..., 1], pp[..., 2]
        return torch.pow(6.0 * a * b * c / PI + EPS, 1.0 / 3.0)
    if type_id == 2:  # cylinder half-height h: V=2*pi*r^2*h => r_eff=(1.5 r^2 h)^(1/3)
        r, h = pp[..., 0], pp[..., 1]
        return torch.pow(1.5 * r * r * h + EPS, 1.0 / 3.0)
    if type_id == 3:  # wedge approx half box: V~4abc => r_eff=(3abc/pi)^(1/3)
        a, b, c = pp[..., 0], pp[..., 1], pp[..., 2]
        return torch.pow(3.0 * a * b * c / PI + EPS, 1.0 / 3.0)
    if type_id == 4:  # corner wedge approx quarter box: V~2abc
        a, b, c = pp[..., 0], pp[..., 1], pp[..., 2]
        return torch.pow(1.5 * a * b * c / PI + EPS, 1.0 / 3.0)
    raise ValueError(type_id)


def occ_from_sdf(sdf: torch.Tensor, sharpness: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
    if scale is not None:
        sdf = sdf / scale.clamp_min(0.035)
    return torch.sigmoid(-sdf * sharpness.clamp_min(1.0))

# Mesh handling: watertight repair + occupancy/SDF queries

class MeshSDF:
    def __init__(self, mesh_path: str, normalize: bool = True, margin: float = 0.10):
        import trimesh
        self.trimesh = trimesh
        self.mesh_path = mesh_path
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not mesh.is_watertight:
            mesh = self._repair(mesh)
        if normalize:
            mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
            scale = (1.0 - margin) / (np.abs(mesh.vertices).max() + 1e-12)
            mesh.vertices = mesh.vertices * scale
        self.mesh = mesh
        self.vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        print(f"  mesh: {len(self.vertices):,} verts, {len(self.faces):,} faces, watertight={mesh.is_watertight}")

    def _repair(self, mesh):
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mf.repair(verbose=False)
            faces = mf.mesh.faces
            faces = faces.reshape(-1, 4)[:, 1:4] if getattr(faces, "ndim", 2) == 1 else faces
            repaired = self.trimesh.Trimesh(vertices=mf.mesh.points, faces=faces, process=True)
            if repaired.is_watertight:
                print("  repaired with pymeshfix")
                return repaired
        except Exception:
            pass
        try:
            from scipy import ndimage
            res = 128
            pitch = max(mesh.extents) / res
            vox = mesh.voxelized(pitch=pitch)
            mat = ndimage.binary_fill_holes(vox.matrix).astype(np.float32)
            mat = ndimage.gaussian_filter(mat, sigma=0.35)
            mat = np.pad(mat, 1, mode="constant")
            verts, faces = mcubes.marching_cubes(mat, 0.5)
            verts = (verts - 1.0) * vox.pitch + vox.origin
            repaired = self.trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            print(f"  repaired by voxel fill; watertight={repaired.is_watertight}")
            return repaired
        except Exception as e:
            print(f"  warning: repair failed ({e}); continuing with original mesh")
            return mesh

    def _winding_occupancy_fallback(self, points_np: np.ndarray, point_chunk: int = 2048, tri_chunk: int = 4096) -> np.ndarray:
        """Slow watertight-mesh inside test that does not need rtree/igl.

        Computes the generalized winding number by summing triangle solid
        angles. This is intended as a fallback for small smoke tests or systems
        without libigl/rtree. For large benchmark meshes, install libigl or rtree
        for speed.
        """
        V = self.vertices.astype(np.float64)
        F = self.faces.astype(np.int64)
        A_all, B_all, C_all = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        out = np.zeros(len(points_np), dtype=np.float32)
        for ps in range(0, len(points_np), point_chunk):
            P = points_np[ps:ps + point_chunk].astype(np.float64)
            total = np.zeros(len(P), dtype=np.float64)
            for ts in range(0, len(F), tri_chunk):
                A = A_all[ts:ts + tri_chunk][None, :, :] - P[:, None, :]
                B = B_all[ts:ts + tri_chunk][None, :, :] - P[:, None, :]
                C = C_all[ts:ts + tri_chunk][None, :, :] - P[:, None, :]
                la = np.linalg.norm(A, axis=-1) + 1e-12
                lb = np.linalg.norm(B, axis=-1) + 1e-12
                lc = np.linalg.norm(C, axis=-1) + 1e-12
                num = np.einsum('...i,...i->...', A, np.cross(B, C))
                den = (la * lb * lc
                       + np.einsum('...i,...i->...', A, B) * lc
                       + np.einsum('...i,...i->...', B, C) * la
                       + np.einsum('...i,...i->...', C, A) * lb)
                total += np.sum(2.0 * np.arctan2(num, den), axis=1)
            winding = total / (4.0 * np.pi)
            out[ps:ps + len(P)] = (np.abs(winding) > 0.5).astype(np.float32)
        return out

    def signed_distance(self, points_np: np.ndarray) -> np.ndarray:
        points_np = np.ascontiguousarray(points_np, dtype=np.float64)
        try:
            import igl
            sdf, _, _, _ = igl.signed_distance(points_np, self.vertices, self.faces)
            return sdf.astype(np.float32)
        except Exception:
            pass
        try:
            _, dist, _ = self.mesh.nearest.on_surface(points_np)
            sign = np.ones(len(points_np), dtype=np.float32)
            sign[self.mesh.contains(points_np)] = -1.0
            return (dist.astype(np.float32) * sign).astype(np.float32)
        except Exception:
            # Last-resort fallback when neither libigl nor trimesh+rtree is available.
            # Distance is approximated by nearest vertex; sign uses generalized winding.
            from scipy.spatial import cKDTree
            tree = cKDTree(self.vertices)
            dist, _ = tree.query(points_np)
            occ = self._winding_occupancy_fallback(points_np)
            sign = np.where(occ > 0.5, -1.0, 1.0)
            return (dist.astype(np.float32) * sign.astype(np.float32)).astype(np.float32)

    def occupancy(self, points_np: np.ndarray) -> np.ndarray:
        points_np = np.ascontiguousarray(points_np, dtype=np.float64)
        try:
            import igl
            sdf, _, _, _ = igl.signed_distance(points_np, self.vertices, self.faces)
            return (sdf < 0).astype(np.float32)
        except Exception:
            pass
        try:
            return self.mesh.contains(points_np).astype(np.float32)
        except Exception:
            return self._winding_occupancy_fallback(points_np)

    def query_sdf_grid(self, resolution: int) -> np.ndarray:
        a = np.linspace(-1.0, 1.0, resolution)
        x, y, z = np.meshgrid(a, a, a, indexing="ij")
        pts = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
        return self.signed_distance(pts).reshape(resolution, resolution, resolution)

    def sample_points(self, n: int, balanced: bool = True, near_std: float = 0.035) -> Tuple[np.ndarray, np.ndarray]:
        ns = int(0.35 * n)
        nn = int(0.45 * n)
        nr = n - ns - nn
        ps, _ = self.trimesh.sample.sample_surface(self.mesh, ns)
        pn, _ = self.trimesh.sample.sample_surface(self.mesh, nn)
        pn = np.clip(pn + np.random.randn(nn, 3) * near_std, -1, 1)
        pr = np.random.uniform(-1, 1, (nr, 3))
        pts = np.vstack([ps, pn, pr]).astype(np.float32)
        occ = self.occupancy(pts)
        if balanced:
            pos = np.where(occ > 0.5)[0]
            neg = np.where(occ <= 0.5)[0]
            if len(pos) > 0 and len(neg) > 0:
                npos = n // 2
                nneg = n - npos
                sel = np.concatenate([
                    np.random.choice(pos, npos, replace=True),
                    np.random.choice(neg, nneg, replace=True),
                ])
                np.random.shuffle(sel)
                return pts[sel], occ[sel]
        perm = np.random.permutation(len(pts))[:n]
        return pts[perm], occ[perm]


def download_mesh(name: str, meshes_dir: str) -> str:
    if name not in MESH_REGISTRY:
        raise ValueError(f"unknown mesh registry key {name}; available={list(MESH_REGISTRY)}")
    os.makedirs(meshes_dir, exist_ok=True)
    fp = os.path.join(meshes_dir, f"{name}.obj")
    if os.path.exists(fp) and os.path.getsize(fp) > 1024:
        return fp
    url = MESH_REGISTRY[name]
    tmp = fp + ".tmp"
    print(f"  downloading {name}: {url}")
    urllib.request.urlretrieve(url, tmp)
    if os.path.getsize(tmp) < 1024:
        raise RuntimeError(f"downloaded file too small: {tmp}")
    os.replace(tmp, fp)
    return fp


def resolve_mesh_specs(specs: Sequence[str], meshes_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for spec in specs:
        hits = sorted(glob.glob(spec))
        if hits:
            for h in hits:
                out.append((h, os.path.splitext(os.path.basename(h))[0]))
            continue
        if os.path.exists(spec):
            out.append((spec, os.path.splitext(os.path.basename(spec))[0]))
            continue
        if spec in MESH_REGISTRY:
            p = download_mesh(spec, meshes_dir)
            out.append((p, spec))
            continue
        cand = os.path.join(meshes_dir, spec if spec.endswith(".obj") else spec + ".obj")
        if os.path.exists(cand):
            out.append((cand, os.path.splitext(os.path.basename(cand))[0]))
            continue
        raise ValueError(f"mesh spec not found: {spec}")
    return out

# Exports and metrics

def compute_iou(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    p = (pred > threshold).float()
    g = (gt > threshold).float()
    return float(((p * g).sum() / (((p + g) > 0).float().sum() + 1e-8)).item())


def compute_fscore(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    p = (pred > threshold).float()
    g = (gt > threshold).float()
    tp = (p * g).sum()
    return float((2 * tp / (p.sum() + g.sum() + 1e-8)).item())


def export_occ_obj(filepath: str, occ: torch.Tensor, res: int, threshold: float = 0.5) -> bool:
    arr = occ.detach().float().cpu().numpy().reshape(res, res, res)
    if arr.min() >= threshold or arr.max() <= threshold:
        return False
    verts, faces = mcubes.marching_cubes(-(arr * 2 - 1), 0.0)
    verts = verts / (res - 1) * 2.0 - 1.0
    faces = faces.copy(); faces[:, [0, 1]] = faces[:, [1, 0]]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mcubes.export_obj(verts, faces, filepath)
    return True


# Primitive leaf pool

@dataclass
class PoolCfg:
    n_leaves: int
    mode: str = "fixed_cycle"          # fixed_cycle, sine_softmax, plain_softmax, sine_sigmoid_multi
    type_agg: str = "product"          # product, sum
    volume_norm: bool = True
    omega: float = 10.0
    init_scale: float = 0.3


class PrimitivePool(nn.Module):
    def __init__(self, cfg: PoolCfg):
        super().__init__()
        self.cfg = cfg
        N = cfg.n_leaves
        fixed = torch.tensor([i % N_TYPES for i in range(N)], dtype=torch.long)
        self.register_buffer("fixed_types", fixed)
        self.centers = nn.Parameter(torch.randn(N, 3) * cfg.init_scale)
        self.euler = nn.Parameter(torch.randn(N, 3) * 0.2)
        # Each leaf has params for every type. Hard evaluation uses only one type.
        self.raw_params = nn.Parameter(torch.randn(N, N_TYPES, MAX_PARAM) * 0.15 - 1.2)
        self.raw_sharp = nn.Parameter(torch.ones(N) * 2.5)
        self.type_logits = nn.Parameter(torch.randn(N, N_TYPES) * 0.05)

    def type_weights(self, beta: float, hard: bool = False) -> torch.Tensor:
        N = self.cfg.n_leaves
        if self.cfg.mode == "fixed_cycle":
            w = torch.zeros(N, N_TYPES, device=self.centers.device, dtype=self.centers.dtype)
            w.scatter_(1, self.fixed_types.reshape(-1, 1), 1.0)
            return w
        if self.cfg.mode == "sine_softmax":
            logits = beta * torch.sin(self.cfg.omega * self.type_logits)
            if hard:
                idx = torch.argmax(logits, dim=-1, keepdim=True)
                w = torch.zeros_like(logits); w.scatter_(1, idx, 1.0); return w
            return torch.softmax(logits, dim=-1)
        if self.cfg.mode == "plain_softmax":
            logits = beta * self.type_logits
            if hard:
                idx = torch.argmax(logits, dim=-1, keepdim=True)
                w = torch.zeros_like(logits); w.scatter_(1, idx, 1.0); return w
            return torch.softmax(logits, dim=-1)
        if self.cfg.mode == "sine_sigmoid_multi":
            logits = beta * torch.sin(self.cfg.omega * self.type_logits)
            if hard:
                idx = torch.argmax(logits, dim=-1, keepdim=True)
                w = torch.zeros_like(logits); w.scatter_(1, idx, 1.0); return w
            return torch.sigmoid(logits)  # multi-hot training ablation
        raise ValueError(self.cfg.mode)

    def local_coords(self, pts: torch.Tensor) -> torch.Tensor:
        # (B,3) -> (N,B,3)
        R = euler_to_R(self.euler)
        return torch.einsum("nbd,nde->nbe", pts.unsqueeze(0) - self.centers.unsqueeze(1), R)

    def occs_by_type(self, pts: torch.Tensor) -> torch.Tensor:
        """Return occupancies (N,B,T)."""
        local = self.local_coords(pts)
        sharp = (F.softplus(self.raw_sharp) + 2.0).reshape(-1, 1)
        out = []
        params = pos_param(self.raw_params)
        for t in range(N_TYPES):
            p = params[:, t, :].reshape(self.cfg.n_leaves, 1, MAX_PARAM)
            sdf = sdf_by_type(t, local, p)
            scale = volume_equiv_radius(t, p).reshape(self.cfg.n_leaves, 1) if self.cfg.volume_norm else None
            out.append(occ_from_sdf(sdf, sharp, scale))
        return torch.stack(out, dim=-1)

    def get_occs(self, pts: torch.Tensor, beta: float, hard_leaf: bool = False) -> torch.Tensor:
        occ_stack = self.occs_by_type(pts)  # (N,B,T)
        w = self.type_weights(beta, hard=hard_leaf)  # (N,T)
        if hard_leaf:
            # Exactly one primitive type per leaf.
            return (occ_stack * w.unsqueeze(1)).sum(-1)
        if self.cfg.type_agg == "sum":
            return (occ_stack * w.unsqueeze(1)).sum(-1).clamp(0, 1)
        return 1.0 - torch.prod((1.0 - occ_stack * w.unsqueeze(1)).clamp_min(1e-6), dim=-1)

    @torch.no_grad()
    def dominant_types(self, beta: float) -> torch.Tensor:
        return torch.argmax(self.type_weights(beta, hard=False), dim=-1)

    def type_entropy(self, beta: float) -> torch.Tensor:
        if self.cfg.mode == "fixed_cycle":
            return torch.tensor(0.0, device=self.centers.device)
        w = self.type_weights(beta, hard=False)
        w_norm = w / (w.sum(dim=-1, keepdim=True) + EPS)
        return (-(w_norm * torch.log(w_norm + EPS)).sum(-1)).mean()

# Models: literal DNF, CSG-Stump, product union, sum union, pairwise tree

class BaseCSGModel(nn.Module):
    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        raise NotImplementedError
    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        raise NotImplementedError
    def init_from_mesh(self, mesh: MeshSDF) -> None:
        raise NotImplementedError


def dnf_forward(occs: torch.Tensor, m_pos: torch.Tensor, m_neg: torch.Tensor, act: torch.Tensor,
                hard_bool: bool = False, thr: float = 0.5) -> torch.Tensor:
    # occs: (N,B); m_pos/m_neg: (N,C), act: (C)
    if hard_bool:
        mp = (torch.sigmoid(m_pos) > thr).float()
        mn = (torch.sigmoid(m_neg) > thr).float()
        a = (torch.sigmoid(act) > thr).float()
        # Valid hard projection: an active empty intersection clause evaluates to
        # the universe and can make the entire DNF spuriously full.  Drop such
        # clauses in hard evaluation (a projection to a valid hard CSG program;
        # training is unaffected).
        nonempty = ((mp + mn).sum(dim=0) > 0).float()
        a = a * nonempty
    else:
        mp = torch.sigmoid(m_pos)
        mn = torch.sigmoid(m_neg)
        a = torch.sigmoid(act)
    o = occs.unsqueeze(-1)  # (N,B,1)
    fp = 1.0 - mp.unsqueeze(1) * (1.0 - o)  # selected positive literal q, unselected neutral 1
    fn = 1.0 - mn.unsqueeze(1) * o          # selected negative literal 1-q, unselected neutral 1
    clause = torch.exp(torch.log(fp.clamp_min(1e-8)).sum(0) + torch.log(fn.clamp_min(1e-8)).sum(0))  # (B,C)
    ag = (a.unsqueeze(0) * clause).clamp(0, 1)
    return (1.0 - torch.prod((1.0 - ag).clamp_min(1e-6), dim=-1)).clamp(0, 1)


def soft_min_star(x: torch.Tensor, dim: int, psi: float) -> torch.Tensor:
    """CSG-Stump Net relaxed min*: sum softmax(-psi*x) * x.

    Paper CSG-Stump uses min for intersections and proposes a weighted-softmax
    relaxation to avoid vanishing gradients through hard min/max. This returns
    a soft approximation to min along dim; larger psi is harder.
    """
    w = torch.softmax(-psi * x, dim=dim)
    return (w * x).sum(dim=dim)


def soft_max_star(x: torch.Tensor, dim: int, psi: float) -> torch.Tensor:
    """CSG-Stump Net relaxed max*: sum softmax(psi*x) * x."""
    w = torch.softmax(psi * x, dim=dim)
    return (w * x).sum(dim=dim)


def hard_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.min(x, dim=dim).values


def hard_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.max(x, dim=dim).values


class PaperStumpModel(BaseCSGModel):
    """CSG-Stump (Ren et al. 2021): complement -> intersection -> union.

    This differs from the DNF model above. The DNF model has clause-local
    positive and negative literal matrices m_pos/m_neg. The CSG-Stump paper
    defines exactly three connection matrices:

        WC: (N,)   choose primitive or complement at the complement layer
        WI: (N,C) choose which complemented leaves enter each intersection node
        WU: (C,)   choose which intersection nodes enter the final union

    Soft training follows the paper's differentiable relaxation:
        F_i = WC_i*(1-O_i) + (1-WC_i)*O_i
        X_ic = WI_ic*F_i + (1-WI_ic)*1
        S_c = min*_i X_ic
        Y_c = WU_c*S_c + (1-WU_c)*0
        T = max*_c Y_c

    Hard reporting thresholds WC/WI/WU at 0.5 and uses exact min/max, while
    hard leaves always commit to one primitive type.
    """
    def __init__(self, n_leaves: int, n_groups: int, pool_mode: str, type_agg: str,
                 volume_norm: bool, omega: float, psi: float = 20.0):
        super().__init__()
        self.pool = PrimitivePool(PoolCfg(n_leaves=n_leaves, mode=pool_mode, type_agg=type_agg,
                                          volume_norm=volume_norm, omega=omega))
        self.N = n_leaves
        self.C = n_groups
        self.psi = float(psi)
        # Paper matrices: WC, WI, WU. Logits are optimized; sigmoid gives [0,1].
        self.wc = nn.Parameter(torch.ones(n_leaves) * logit(0.05))       # mostly no complement
        self.wi = nn.Parameter(torch.randn(n_leaves, n_groups) * 0.2 - 5.0)
        self.wu = nn.Parameter(torch.ones(n_groups) * logit(0.65))
        self._init_connections()

    def _init_connections(self) -> None:
        with torch.no_grad():
            self.wc.fill_(logit(0.03))
            self.wi.fill_(-6.0)
            self.wu.fill_(logit(0.65))
            # Start with spatially distributed singleton-ish intersection nodes;
            # this avoids the all-ones intersection-node degeneracy.
            for i in range(self.N):
                self.wi[i, i % self.C] = logit(0.45)

    def _stump(self, q: torch.Tensor, hard_bool: bool, psi: Optional[float] = None) -> torch.Tensor:
        # q: (N,B)
        if hard_bool:
            wc = (torch.sigmoid(self.wc) > 0.5).float()                 # (N,)
            wi = (torch.sigmoid(self.wi) > 0.5).float()                 # (N,C)
            wu = (torch.sigmoid(self.wu) > 0.5).float()                 # (C,)
            nonempty = (wi.sum(dim=0) > 0).float()
            wu = wu * nonempty
            Fleaf = wc.reshape(-1, 1) * (1.0 - q) + (1.0 - wc.reshape(-1, 1)) * q
            X = wi.unsqueeze(1) * Fleaf.unsqueeze(-1) + (1.0 - wi.unsqueeze(1)) * 1.0
            S = hard_min(X, dim=0)                                      # (B,C)
            Y = wu.reshape(1, -1) * S
            return hard_max(Y, dim=-1).clamp(0, 1)
        wc = torch.sigmoid(self.wc)
        wi = torch.sigmoid(self.wi)
        wu = torch.sigmoid(self.wu)
        Fleaf = wc.reshape(-1, 1) * (1.0 - q) + (1.0 - wc.reshape(-1, 1)) * q
        X = wi.unsqueeze(1) * Fleaf.unsqueeze(-1) + (1.0 - wi.unsqueeze(1)) * 1.0
        ps = self.psi if psi is None else float(psi)
        S = soft_min_star(X, dim=0, psi=ps)                              # (B,C)
        Y = wu.reshape(1, -1) * S
        return soft_max_star(Y, dim=-1, psi=ps).clamp(0, 1)

    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        q_soft = self.pool.get_occs(pts, beta, hard_leaf=False)
        q_hard = self.pool.get_occs(pts, beta, hard_leaf=True)
        return {
            "soft": self._stump(q_soft, hard_bool=False),
            "hardleaf_softbool": self._stump(q_hard, hard_bool=False),
            "hard": self._stump(q_hard, hard_bool=True),
        }

    def forward(self, pts: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self.forward_modes(pts, beta)["soft"]

    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        wc = torch.sigmoid(self.wc); wi = torch.sigmoid(self.wi); wu = torch.sigmoid(self.wu)
        sparse = args.lambda_sparse * warm * (wi.sum() + wu.sum())
        # Complement is not sparse in the same way: complementing is a choice, not usage.
        binary = args.lambda_binary * warm * ((wc * (1 - wc)).sum() + (wi * (1 - wi)).sum() + (wu * (1 - wu)).sum())
        excess = torch.clamp(wi.sum(0) - args.max_group_size, min=0)
        group = args.lambda_group * warm * (excess ** 2).sum()
        type_ent = args.lambda_type_entropy * warm * self.pool.type_entropy(beta)
        return sparse + binary + group + type_ent

    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        wi = torch.sigmoid(self.wi).detach(); wu = torch.sigmoid(self.wu).detach(); wc = torch.sigmoid(self.wc).detach()
        used = ((wi.max(1).values > threshold) & (wu.max() > threshold)).sum().item()
        groups = (wu > threshold).sum().item()
        dom = self.pool.dominant_types(beta).detach().cpu().numpy().tolist()
        td = {name: int(sum(1 for t in dom if t == k)) for k, name in enumerate(TYPE_NAMES)}
        return {"prims_used": int(used), "groups_active": int(groups), "type_dist": td,
                "type_entropy": float(self.pool.type_entropy(beta).detach().cpu().item()),
                "complements_active": int((wc > 0.5).sum().item())}

    @torch.no_grad()
    def init_from_mesh(self, mesh: MeshSDF) -> None:
        surface_init_pool(self.pool, mesh)
        spatial_group_init_paper_stump(self, mesh)


class NaryDNFModel(BaseCSGModel):
    def __init__(self, n_leaves: int, n_groups: int, pool_mode: str, type_agg: str, volume_norm: bool, omega: float):
        super().__init__()
        self.pool = PrimitivePool(PoolCfg(n_leaves=n_leaves, mode=pool_mode, type_agg=type_agg,
                                          volume_norm=volume_norm, omega=omega))
        self.N = n_leaves; self.C = n_groups
        self.m_pos = nn.Parameter(torch.randn(n_leaves, n_groups) * 0.25 - 4.0)
        self.m_neg = nn.Parameter(torch.randn(n_leaves, n_groups) * 0.25 - 5.0)
        self.act = nn.Parameter(torch.randn(n_groups) * 0.25 - 3.0)
        self._init_groups()

    def _init_groups(self) -> None:
        with torch.no_grad():
            self.m_pos.fill_(-5.0); self.m_neg.fill_(-6.0); self.act.fill_(-4.0)
            for i in range(self.N):
                self.m_pos[i, i % self.C] = logit(0.35)
            self.act[:] = logit(0.65)

    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        q_soft = self.pool.get_occs(pts, beta, hard_leaf=False)
        q_hard = self.pool.get_occs(pts, beta, hard_leaf=True)
        return {
            "soft": dnf_forward(q_soft, self.m_pos, self.m_neg, self.act, hard_bool=False),
            "hardleaf_softbool": dnf_forward(q_hard, self.m_pos, self.m_neg, self.act, hard_bool=False),
            "hard": dnf_forward(q_hard, self.m_pos, self.m_neg, self.act, hard_bool=True),
        }

    def forward(self, pts: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self.forward_modes(pts, beta)["soft"]

    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        mp = torch.sigmoid(self.m_pos); mn = torch.sigmoid(self.m_neg); a = torch.sigmoid(self.act)
        sparse = args.lambda_sparse * warm * (mp.sum() + mn.sum() + a.sum())
        binary = args.lambda_binary * warm * ((mp * (1 - mp)).sum() + (mn * (1 - mn)).sum() + (a * (1 - a)).sum())
        conflict = args.lambda_conflict * warm * (mp * mn).sum()
        excess = torch.clamp(mp.sum(0) + mn.sum(0) - args.max_group_size, min=0)
        group = args.lambda_group * warm * (excess ** 2).sum()
        type_ent = args.lambda_type_entropy * warm * self.pool.type_entropy(beta)
        return sparse + binary + conflict + group + type_ent

    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        mp = torch.sigmoid(self.m_pos).detach(); mn = torch.sigmoid(self.m_neg).detach(); a = torch.sigmoid(self.act).detach()
        used = ((mp.max(1).values > threshold) | (mn.max(1).values > threshold)).sum().item()
        groups = (a > threshold).sum().item()
        dom = self.pool.dominant_types(beta).detach().cpu().numpy().tolist()
        td = {name: int(sum(1 for t in dom if t == k)) for k, name in enumerate(TYPE_NAMES)}
        return {"prims_used": int(used), "groups_active": int(groups), "type_dist": td,
                "type_entropy": float(self.pool.type_entropy(beta).detach().cpu().item())}

    @torch.no_grad()
    def init_from_mesh(self, mesh: MeshSDF) -> None:
        surface_init_pool(self.pool, mesh)
        spatial_group_init_dnf(self, mesh)


class ProperCSGStumpModel(BaseCSGModel):
    """Fuzzy CSG-Stump: complement layer, intersection layer, union layer.

    This follows the CSG-Stump normal form:
        F_i = c_i (1 - q_i) + (1-c_i) q_i                  [complement layer WC]
        S_c = prod_i [1 - w^I_{ic} (1 - F_i)]             [intersection layer WI]
        T   = 1 - prod_c [1 - w^U_c S_c]                  [union layer WU]

    Unlike NaryDNFModel, the complement choice c_i is global per leaf, not
    per-clause positive/negative membership.
    """
    def __init__(self, n_leaves: int, n_groups: int, pool_mode: str, type_agg: str, volume_norm: bool, omega: float):
        super().__init__()
        self.pool = PrimitivePool(PoolCfg(n_leaves=n_leaves, mode=pool_mode, type_agg=type_agg,
                                          volume_norm=volume_norm, omega=omega))
        self.N = n_leaves
        self.C = n_groups
        self.comp = nn.Parameter(torch.randn(n_leaves) * 0.15 - 5.0)      # WC: default non-complement
        self.inter = nn.Parameter(torch.randn(n_leaves, n_groups) * 0.25 - 5.0)  # WI
        self.union = nn.Parameter(torch.randn(n_groups) * 0.25 - 4.0)     # WU
        self._init_groups()

    def _init_groups(self) -> None:
        with torch.no_grad():
            self.comp.fill_(-6.0)
            self.inter.fill_(-5.0)
            self.union.fill_(-4.0)
            for i in range(self.N):
                self.inter[i, i % self.C] = logit(0.35)
            self.union[:] = logit(0.65)

    def _stump_forward(self, occs: torch.Tensor, hard_bool: bool = False, thr: float = 0.5) -> torch.Tensor:
        # occs: (N,B). CSG-Stump with product t-norm/t-conorm.
        if hard_bool:
            c = (torch.sigmoid(self.comp) > thr).float()
            wi = (torch.sigmoid(self.inter) > thr).float()
            wu = (torch.sigmoid(self.union) > thr).float()
            # Valid hard projection: an active intersection node selecting no
            # literals is the universe U in CSG-Stump.  Unless the target is the
            # entire bounding box, this is an extraction artifact; drop it.
            nonempty = (wi.sum(dim=0) > 0).float()
            wu = wu * nonempty
        else:
            c = torch.sigmoid(self.comp)
            wi = torch.sigmoid(self.inter)
            wu = torch.sigmoid(self.union)
        # Complement layer: one global literal per leaf.
        Fleaf = (1.0 - c).reshape(-1, 1) * occs + c.reshape(-1, 1) * (1.0 - occs)  # (N,B)
        # Intersection layer: selected literals contribute F_i; unselected contribute universe=1.
        lit = 1.0 - wi.unsqueeze(1) * (1.0 - Fleaf.unsqueeze(-1))  # (N,B,C)
        clause = torch.exp(torch.log(lit.clamp_min(1e-8)).sum(0))  # (B,C)
        # Union layer: selected clauses contribute; unselected contribute empty=0.
        ag = (wu.unsqueeze(0) * clause).clamp(0, 1)
        return (1.0 - torch.prod((1.0 - ag).clamp_min(1e-6), dim=-1)).clamp(0, 1)

    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        q_soft = self.pool.get_occs(pts, beta, hard_leaf=False)
        q_hard = self.pool.get_occs(pts, beta, hard_leaf=True)
        return {
            "soft": self._stump_forward(q_soft, hard_bool=False),
            "hardleaf_softbool": self._stump_forward(q_hard, hard_bool=False),
            "hard": self._stump_forward(q_hard, hard_bool=True),
        }

    def forward(self, pts: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self.forward_modes(pts, beta)["soft"]

    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        c = torch.sigmoid(self.comp)
        wi = torch.sigmoid(self.inter)
        wu = torch.sigmoid(self.union)
        sparse = args.lambda_sparse * warm * (wi.sum() + wu.sum())
        binary = args.lambda_binary * warm * ((c * (1 - c)).sum() + (wi * (1 - wi)).sum() + (wu * (1 - wu)).sum())
        # CSG-Stump has no pos/neg conflict term; cap the clause arity instead.
        excess = torch.clamp(wi.sum(0) - args.max_group_size, min=0)
        group = args.lambda_group * warm * (excess ** 2).sum()
        type_ent = args.lambda_type_entropy * warm * self.pool.type_entropy(beta)
        return sparse + binary + group + type_ent

    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        wi = torch.sigmoid(self.inter).detach()
        wu = torch.sigmoid(self.union).detach()
        used = (wi.max(1).values > threshold).sum().item()
        groups = (wu > threshold).sum().item()
        comp_used = (torch.sigmoid(self.comp).detach() > threshold).sum().item()
        dom = self.pool.dominant_types(beta).detach().cpu().numpy().tolist()
        td = {name: int(sum(1 for t in dom if t == k)) for k, name in enumerate(TYPE_NAMES)}
        return {"prims_used": int(used), "groups_active": int(groups), "complements_active": int(comp_used),
                "type_dist": td, "type_entropy": float(self.pool.type_entropy(beta).detach().cpu().item())}

    @torch.no_grad()
    def init_from_mesh(self, mesh: MeshSDF) -> None:
        surface_init_pool(self.pool, mesh)
        # Spatially initialize WI/WU similarly to DNF, but no per-clause neg literals.
        self.comp.fill_(-6.0)
        self.inter.fill_(-5.0)
        self.union.fill_(-4.0)
        for i in range(self.N):
            self.inter[i, i % self.C] = logit(0.35)
        self.union[:] = logit(0.65)


class ProductUnionModel(BaseCSGModel):
    def __init__(self, n_leaves: int, pool_mode: str, type_agg: str, volume_norm: bool, omega: float, sum_mode: bool = False):
        super().__init__()
        self.pool = PrimitivePool(PoolCfg(n_leaves=n_leaves, mode=pool_mode, type_agg=type_agg,
                                          volume_norm=volume_norm, omega=omega))
        self.N = n_leaves
        self.gates = nn.Parameter(torch.ones(n_leaves) * logit(0.35))
        self.sum_mode = sum_mode

    def _combine(self, q: torch.Tensor, hard_bool: bool) -> torch.Tensor:
        g = (torch.sigmoid(self.gates) > 0.5).float() if hard_bool else torch.sigmoid(self.gates)
        if self.sum_mode:
            return (g.reshape(-1, 1) * q).sum(0).clamp(0, 1)
        return (1.0 - torch.prod((1.0 - g.reshape(-1, 1) * q).clamp_min(1e-6), dim=0)).clamp(0, 1)

    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        q_soft = self.pool.get_occs(pts, beta, hard_leaf=False)
        q_hard = self.pool.get_occs(pts, beta, hard_leaf=True)
        return {
            "soft": self._combine(q_soft, hard_bool=False),
            "hardleaf_softbool": self._combine(q_hard, hard_bool=False),
            "hard": self._combine(q_hard, hard_bool=True),
        }

    def forward(self, pts: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self.forward_modes(pts, beta)["soft"]

    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        g = torch.sigmoid(self.gates)
        return (args.lambda_sparse * warm * g.sum() + args.lambda_binary * warm * (g * (1 - g)).sum()
                + args.lambda_type_entropy * warm * self.pool.type_entropy(beta))

    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        g = torch.sigmoid(self.gates).detach()
        dom = self.pool.dominant_types(beta).detach().cpu().numpy().tolist()
        td = {name: int(sum(1 for t in dom if t == k)) for k, name in enumerate(TYPE_NAMES)}
        return {"prims_used": int((g > threshold).sum().item()), "groups_active": 0, "type_dist": td,
                "type_entropy": float(self.pool.type_entropy(beta).detach().cpu().item())}

    @torch.no_grad()
    def init_from_mesh(self, mesh: MeshSDF) -> None:
        surface_init_pool(self.pool, mesh)


class BinaryTreeModel(BaseCSGModel):
    def __init__(self, n_leaves: int, pool_mode: str, type_agg: str, volume_norm: bool, omega: float,
                 tree_ops: str = "csg3"):
        super().__init__()
        self.pool = PrimitivePool(PoolCfg(n_leaves=n_leaves, mode=pool_mode, type_agg=type_agg,
                                          volume_norm=volume_norm, omega=omega))
        self.N = n_leaves
        self.tree_ops = tree_ops
        self.n_ops = 3 if tree_ops == "csg3" else 5
        self.gates = nn.Parameter(torch.ones(n_leaves) * logit(0.55))
        self.op_logits = nn.Parameter(torch.randn(max(n_leaves - 1, 1), self.n_ops) * 0.05)

    def _combine_pair(self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        union = 1.0 - (1.0 - a) * (1.0 - b)
        inter = a * b
        diff_ab = a * (1.0 - b)
        if self.n_ops == 3:
            return (w[0] * union + w[1] * inter + w[2] * diff_ab).clamp(0, 1)
        diff_ba = b * (1.0 - a)
        # pass-left is a prune/skip operation; useful as explicit tree-pruning ablation.
        return (w[0] * union + w[1] * inter + w[2] * diff_ab + w[3] * diff_ba + w[4] * a).clamp(0, 1)

    def _tree_reduce(self, q: torch.Tensor, hard_bool: bool) -> torch.Tensor:
        # q: (N,B). Balanced binary tree, internal nodes consume op_logits in BFS-like order.
        g = (torch.sigmoid(self.gates) > 0.5).float() if hard_bool else torch.sigmoid(self.gates)
        vals: List[torch.Tensor] = [(g[i] * q[i]).clamp(0, 1) for i in range(self.N)]
        node = 0
        while len(vals) > 1:
            new_vals: List[torch.Tensor] = []
            for i in range(0, len(vals), 2):
                if i + 1 == len(vals):
                    new_vals.append(vals[i])
                else:
                    logits = self.op_logits[node]
                    if hard_bool:
                        idx = torch.argmax(logits)
                        w = torch.zeros_like(logits); w[idx] = 1.0
                    else:
                        w = torch.softmax(logits, dim=-1)
                    new_vals.append(self._combine_pair(vals[i], vals[i+1], w))
                    node += 1
            vals = new_vals
        return vals[0]

    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        q_soft = self.pool.get_occs(pts, beta, hard_leaf=False)
        q_hard = self.pool.get_occs(pts, beta, hard_leaf=True)
        return {
            "soft": self._tree_reduce(q_soft, hard_bool=False),
            "hardleaf_softbool": self._tree_reduce(q_hard, hard_bool=False),
            "hard": self._tree_reduce(q_hard, hard_bool=True),
        }

    def forward(self, pts: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return self.forward_modes(pts, beta)["soft"]

    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        g = torch.sigmoid(self.gates)
        ow = torch.softmax(self.op_logits, dim=-1)
        op_ent = -(ow * torch.log(ow + EPS)).sum(-1).mean()
        return (args.lambda_sparse * warm * g.sum() + args.lambda_binary * warm * (g * (1 - g)).sum()
                + args.lambda_type_entropy * warm * self.pool.type_entropy(beta)
                + args.lambda_op_entropy * warm * op_ent)

    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        g = torch.sigmoid(self.gates).detach()
        dom = self.pool.dominant_types(beta).detach().cpu().numpy().tolist()
        td = {name: int(sum(1 for t in dom if t == k)) for k, name in enumerate(TYPE_NAMES)}
        ow = torch.softmax(self.op_logits, dim=-1).detach()
        ops = torch.argmax(ow, dim=-1).cpu().numpy().tolist()
        return {"prims_used": int((g > threshold).sum().item()), "groups_active": int(len(ops)), "type_dist": td,
                "type_entropy": float(self.pool.type_entropy(beta).detach().cpu().item()),
                "op_entropy": float((-(ow * torch.log(ow + EPS)).sum(-1)).mean().item()),
                "op_counts": {str(k): int(sum(1 for o in ops if o == k)) for k in range(self.n_ops)}}

    @torch.no_grad()
    def init_from_mesh(self, mesh: MeshSDF) -> None:
        surface_init_pool(self.pool, mesh)

# Initialization

@torch.no_grad()
def surface_init_pool(pool: PrimitivePool, mesh: MeshSDF) -> None:
    N = pool.cfg.n_leaves
    pts, _ = mesh.trimesh.sample.sample_surface(mesh.mesh, max(N * 4, N + 1))
    if len(pts) < N:
        pts = np.random.uniform(-0.5, 0.5, (N, 3))
    # Farthest-point sample among surface samples.
    sel = [int(np.random.randint(0, len(pts)))]
    dists = np.full(len(pts), np.inf)
    for _ in range(N - 1):
        d = np.linalg.norm(pts - pts[sel[-1]], axis=1)
        dists = np.minimum(dists, d)
        sel.append(int(np.argmax(dists)))
    centers = torch.tensor(pts[sel[:N]].astype(np.float32), device=pool.centers.device)
    # Sort by x/y/z so a balanced tree has roughly spatial leaf order.
    order = torch.argsort(centers[:, 0] + 0.37 * centers[:, 1] + 0.13 * centers[:, 2])
    centers = centers[order]
    pool.centers.copy_(centers)
    pool.euler.copy_(torch.randn_like(pool.euler) * 0.1)
    bbox = float(max(mesh.mesh.bounding_box.extents))
    base = max(bbox / max(N ** (1.0 / 3.0), 1.0) * 0.45, 0.05)
    params = torch.tensor([base, base * 0.8, base * 0.65], device=pool.centers.device)
    inv = inv_pos_param(params)
    pool.raw_params.fill_(-4.0)
    for i in range(N):
        for t in range(N_TYPES):
            pool.raw_params[i, t, :].copy_(inv)
    # Initialize fixed/learned types uniformly cyclic; for learned logits, slightly prefer cyclic type.
    pool.type_logits.zero_()
    for i in range(N):
        pool.type_logits[i, int(pool.fixed_types[i].item())] = 0.25


@torch.no_grad()
def spatial_group_init_dnf(model: NaryDNFModel, mesh: MeshSDF) -> None:
    # Clause assignment by current center order, as in nary_benchmark_v7.py.
    C = model.C
    model.m_pos.fill_(-5.0); model.m_neg.fill_(-6.0); model.act.fill_(-4.0)
    for i in range(model.N):
        model.m_pos[i, i % C] = logit(0.35)
    model.act[:] = logit(0.65)


@torch.no_grad()
def spatial_group_init_paper_stump(model: PaperStumpModel, mesh: MeshSDF) -> None:
    # As in the DNF initialization: activate small spatially distributed
    # intersections. WC remains mostly non-complemented, matching the paper's
    # common setting WC=0 for many ShapeNet-like shapes.
    C = model.C
    model.wc.fill_(logit(0.03))
    model.wi.fill_(-6.0)
    model.wu.fill_(logit(0.65))
    for i in range(model.N):
        model.wi[i, i % C] = logit(0.45)

# Training / eval


def make_training_cache(mesh_sdf: MeshSDF, n: int, device: torch.device, balanced: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Precompute a GPU-resident training set.

    Mesh inside/outside queries are usually CPU-bound; for long optimizer runs it
    is faster, and gives a fixed sample set, to precompute a large stratified cache and
    draw random minibatches from it on the GPU. Set --cache-points 0 to fall back
    to periodic CPU resampling.
    """
    if n <= 0:
        return None, None
    print(f"  building training cache: {n:,} points")
    pts, occ = mesh_sdf.sample_points(n, balanced=balanced)
    P = torch.tensor(pts, device=device, dtype=torch.float32)
    Y = torch.tensor(occ, device=device, dtype=torch.float32)
    pos = int((Y > 0.5).sum().item())
    print(f"  cache occupancy: {pos:,}/{n:,} inside ({pos/max(n,1):.3f})")
    return P, Y


def beta_schedule(epoch: int, epochs: int, beta_start: float, beta_end: float, warmup_frac: float = 0.05) -> float:
    if epochs <= 1:
        return beta_end
    t = epoch / float(max(epochs - 1, 1))
    if t < warmup_frac:
        return beta_start
    u = (t - warmup_frac) / max(1e-8, (1.0 - warmup_frac))
    # Exponential ramp is more stable than a sudden jump.
    return float(beta_start * ((beta_end / beta_start) ** u))


def reg_warm(epoch: int, epochs: int, warmup: int, ramp: int) -> float:
    if epoch < warmup:
        return 0.0
    if epoch < warmup + ramp:
        return float((epoch - warmup) / max(ramp, 1))
    return 1.0


def make_model(method: str, args) -> BaseCSGModel:
    # Fixed-cycle methods use one fixed type per leaf. BC methods use learned one-hot type selection.
    if method == "nary_discrete":
        return NaryDNFModel(args.baseline_prims * N_TYPES, args.groups, "fixed_cycle", "sum", args.volume_norm, args.omega)
    if method == "nary_bc_vol":
        return NaryDNFModel(args.bc_prims, args.groups, args.type_selector, args.type_agg, args.volume_norm, args.omega)
    if method == "stump_discrete":
        return ProperCSGStumpModel(args.baseline_prims * N_TYPES, args.groups, "fixed_cycle", "sum", args.volume_norm, args.omega)
    if method == "stump_bc_vol":
        return ProperCSGStumpModel(args.bc_prims, args.groups, args.type_selector, args.type_agg, args.volume_norm, args.omega)
    if method == "tree_discrete":
        return BinaryTreeModel(args.baseline_prims * N_TYPES, "fixed_cycle", "sum", args.volume_norm, args.omega, args.tree_ops)
    if method == "tree_bc_vol":
        return BinaryTreeModel(args.bc_prims, args.type_selector, args.type_agg, args.volume_norm, args.omega, args.tree_ops)
    if method == "union_product_bc":
        return ProductUnionModel(args.bc_prims, args.type_selector, args.type_agg, args.volume_norm, args.omega, sum_mode=False)
    if method == "union_sum_bc":
        return ProductUnionModel(args.bc_prims, args.type_selector, args.type_agg, args.volume_norm, args.omega, sum_mode=True)
    if method == "nary_bc_multihot":
        return NaryDNFModel(args.bc_prims, args.groups, "sine_sigmoid_multi", args.type_agg, args.volume_norm, args.omega)
    if method == "paper_stump_discrete":
        return PaperStumpModel(args.baseline_prims * N_TYPES, args.groups, "fixed_cycle", "sum", args.volume_norm, args.omega, args.stump_psi)
    if method == "paper_stump_bc_vol":
        return PaperStumpModel(args.bc_prims, args.groups, args.type_selector, args.type_agg, args.volume_norm, args.omega, args.stump_psi)
    raise ValueError(f"unknown method {method}")


# Hard-aware extraction/training helpers


# Plotting / summaries


# CLI


