#!/usr/bin/env python3
"""
Benchmark: n-ary DNF vs pairwise tree (as full_benchmark_with_levels.py, plus the per-level
tree reduction _reduce_tree_with_levels used by the responsibility experiments).

Takes a mesh (or analytic target). For each, runs:

  N-ARY DNF:
    A) Discrete baseline (12x5 = 60 prims)
    N) Bc + vol-norm   (50 prims)

  PAIRWISE TREE (Liu et al. architecture):
    TA) Discrete baseline (60 leaves)
    TN) Bc + vol-norm   (50 leaves)

Plus, if --controlled, runs the 6-target type-fairness suite.

Usage:
  python full_benchmark_with_levels.py --mesh ./bunny.obj --epochs 30000
  python full_benchmark_with_levels.py --mesh ./bracket.obj --epochs 30000
  python full_benchmark_with_levels.py --controlled --epochs 12000
  python full_benchmark_with_levels.py --mesh ./bunny.obj --controlled --epochs 30000
"""

import os, sys, time, math, json, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib.util
import types as _types
if importlib.util.find_spec('mcubes') is None:
    from skimage import measure
    mcubes_stub = _types.ModuleType('mcubes')
    def _marching_cubes(vol, iso):
        v, f, _, _ = measure.marching_cubes(vol, level=iso)
        return v, f.astype(np.int32)
    def _export_obj(vertices, triangles, filepath):
        with open(filepath, 'w') as fh:
            for x, y, z in vertices:
                fh.write(f'v {x} {y} {z}\n')
            for a, b, c in triangles:
                fh.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')
    mcubes_stub.marching_cubes = _marching_cubes
    mcubes_stub.export_obj = _export_obj
    sys.modules['mcubes'] = mcubes_stub

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nary_benchmark_v7 import DiscretePool, BcTypePool, BcVariantCSG, surface_init, TYPE_NAMES

# TREE ARCHITECTURE: Pairwise boolean reduction (Liu et al.)
class _TreeMixin:
    """Shared tree reduction logic for all tree models."""

    def _bool_merge(self, x, y, params, temperature=1.0):
        w = torch.softmax(torch.sin(self.omega * params) * temperature, dim=-1)
        c0, c1, c2, c3 = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
        return (c1 + c2) * x + (c1 + c3) * y + (c0 - c1 - c2 - c3) * x * y

    def _reduce_tree(self, leaves, temperature=1.0, hard=False):
        """Balanced binary tree reduction: pair leaves left-to-right, repeat."""
        cur = list(torch.unbind(leaves, dim=1))
        idx = 0
        while len(cur) > 1:
            nxt = []
            j = 0
            while j + 1 < len(cur):
                if hard:
                    w = torch.softmax(
                        torch.sin(self.omega * self.boolean_params[idx]) * 20.0, dim=-1)
                    one = torch.zeros_like(w)
                    one[w.argmax()] = 1.0
                    c0, c1, c2, c3 = one[0], one[1], one[2], one[3]
                    merged = ((c1+c2)*cur[j] + (c1+c3)*cur[j+1]
                              + (c0-c1-c2-c3)*cur[j]*cur[j+1])
                else:
                    merged = self._bool_merge(
                        cur[j], cur[j+1], self.boolean_params[idx], temperature)
                nxt.append(merged)
                idx += 1
                j += 2
            if j < len(cur):
                nxt.append(cur[j])
            cur = nxt
        return cur[0]

    def _reduce_tree_with_levels(self, leaves, temperature=1.0):
        """
        Return tree levels from leaves upward.

        levels[0] = leaves            [B, N]
        levels[1] = first merge       [B, ceil(N/2)]
        ...
        levels[-1] = root             [B, 1]

        Note:
        Each returned level tensor is the exact tensor used to compute the next one,
        so autograd can differentiate root w.r.t. any intermediate level.
        """
        levels = [leaves]
        cur = leaves
        idx = 0

        while cur.shape[1] > 1:
            cols = []
            j = 0
            m = cur.shape[1]

            while j + 1 < m:
                merged = self._bool_merge(
                    cur[:, j], cur[:, j + 1], self.boolean_params[idx], temperature
                )
                cols.append(merged.unsqueeze(1))
                idx += 1
                j += 2

            if j < m:
                # carry odd leftover forward unchanged
                cols.append(cur[:, j:j+1])

            cur = torch.cat(cols, dim=1)
            levels.append(cur)

        return levels


class TreeDiscreteCSG(nn.Module, _TreeMixin):
    """Liu et al. architecture: discrete typed primitives + pairwise tree."""

    def __init__(self, prims_per_type=12, prim_types=None, omega=10.0):
        super().__init__()
        if prim_types is None:
            prim_types = TYPE_NAMES
        all_types = []
        for t in prim_types:
            all_types.extend([t] * prims_per_type)
        self.pool = DiscretePool(all_types, 0.3)
        self.N = len(all_types)
        self.omega = omega
        self.boolean_params = nn.Parameter(torch.randn(self.N - 1, 4) * 0.3)

    def forward(self, p, temperature=1.0, **kw):
        return self._reduce_tree(self.pool.get_all_occs(p), temperature)

    def forward_hard_dnf(self, p, thr=0.5):
        return self._reduce_tree(self.pool.get_all_occs(p), hard=True)

    def structure_loss(self, ls=1e-4, lb=1e-5, **kw):
        w = torch.softmax(torch.sin(self.omega * self.boolean_params), dim=-1)
        return ls * w.sum() + lb * (w * (1 - w)).sum()

    def get_stats(self, thr=0.3):
        types = {}
        for t in self.pool.types:
            types[t] = types.get(t, 0) + 1
        return {'prims_used': self.N, 'groups_active': self.N - 1,
                'type_dist': types, 'N': self.N}

    def print_status(self, thr=0.3):
        s = self.get_stats(thr)
        print(f"  Leaves: {self.N}, Booleans: {self.N-1}")
        print(f"  Types: {s['type_dist']}")


class TreeBcCSG(nn.Module, _TreeMixin):
    """Tree architecture with Bc type selection (any occ mode)."""

    def __init__(self, num_primitives=50, omega=10.0, occ_mode='bc_vol_norm'):
        super().__init__()
        self.pool = BcTypePool(num_primitives, omega)
        self.N = num_primitives
        self.omega = omega
        self.occ_mode = occ_mode
        self.boolean_params = nn.Parameter(torch.randn(self.N - 1, 4) * 0.3)

    def forward(self, p, temperature=1.0, **kw):
        leaves = self.pool.get_all_occs(p, mode=self.occ_mode, temperature=temperature)
        return self._reduce_tree(leaves, temperature)

    def forward_hard_dnf(self, p, thr=0.5):
        leaves = self.pool.get_all_occs(p, mode=self.occ_mode, temperature=20.0)
        return self._reduce_tree(leaves, hard=True)

    def structure_loss(self, ls=1e-4, lb=1e-5, lt=5e-5, temperature=1.0, **kw):
        w = torch.softmax(torch.sin(self.omega * self.boolean_params), dim=-1)
        base = ls * w.sum() + lb * (w * (1 - w)).sum()
        tw = torch.sigmoid(
            torch.sin(self.pool.omega * self.pool.type_logits) * temperature)
        return base + lt * tw.sum() + lt * (tw * (1 - tw)).sum()

    def get_stats(self, thr=0.3):
        types = {}
        for i in range(self.N):
            t = self.pool.get_dominant_type(i)
            types[t] = types.get(t, 0) + 1
        ent = self.pool.type_entropy()
        return {'prims_used': self.N, 'groups_active': self.N - 1,
                'type_dist': types, 'type_entropy': ent, 'N': self.N}

    def print_status(self, thr=0.3):
        s = self.get_stats(thr)
        print(f"  Leaves: {self.N}, Booleans: {self.N-1}")
        print(f"  Types: {s['type_dist']}")
        print(f"  Type entropy: {s['type_entropy']:.3f}")


# CONVENIENCE WRAPPERS
class NaryBcVolCSG(BcVariantCSG):
    def __init__(self, num_primitives=50, num_groups=16, omega=10.0):
        super().__init__(num_primitives=num_primitives, num_groups=num_groups,
                         omega=omega, max_group_size=8, occ_mode='bc_vol_norm')

class TreeBcVolCSG(TreeBcCSG):
    def __init__(self, num_primitives=50, omega=10.0):
        super().__init__(num_primitives, omega, occ_mode='bc_vol_norm')


# TREE-COMPATIBLE INIT
def tree_surface_init(model, mesh_sdf):
    """Same as surface_init but works with tree models that have .pool."""
    surface_init(model, mesh_sdf)

def tree_spatial_init(model, mesh_sdf):
    """No group init for tree (no DNF groups), but we can init booleans to union."""
    # Initialize all booleans toward union: sin(omega * x) -> [0,1,0,0]
    # union is index 1 in [intersection, union, left_diff, right_diff]
    with torch.no_grad():
        lo = -math.pi / (2.0 * model.omega)
        hi = +math.pi / (2.0 * model.omega)
        for i in range(model.boolean_params.shape[0]):
            model.boolean_params.data[i] = torch.tensor([lo, hi, lo, lo])


# ANALYTIC TARGETS
def _target_complex_csg(p):
    d = torch.abs(p) - 0.5
    box = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1)+1e-8) + \
          torch.clamp(d.max(-1)[0], max=0)
    c = torch.tensor([0.3, 0.3, 0.3], device=p.device)
    sph = torch.sqrt(((p - c)**2).sum(-1)+1e-8) - 0.35
    u = torch.minimum(box, sph)
    cy = torch.sqrt(p[:, 0]**2 + p[:, 2]**2 + 1e-8) - 0.15
    cx = torch.sqrt(p[:, 1]**2 + p[:, 2]**2 + 1e-8) - 0.15
    return (torch.maximum(torch.maximum(u, -cy), -cx) < 0).float()

TARGETS = {'complex_csg': _target_complex_csg}


# SAVE PLOT


# MESH BENCHMARK


# MAIN


