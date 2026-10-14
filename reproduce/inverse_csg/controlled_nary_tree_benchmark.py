#!/usr/bin/env python3
"""
Controlled inverse-CSG benchmark.

Extends nary_benchmark_v7.py with a controlled factorial suite over targets
that have known ground-truth CSG decompositions. The suite separates three
questions:
- Boolean choice with oracle primitives (the fuzzy Boolean operators of
  Liu et al.);
- type choice with oracle geometry (Bc + volume normalisation);
- tree vs n-ary with free geometry, once the first two are established.

Factorial design (10 conditions):

  TREE ARCHITECTURE:
    T_perfect_tree          all frozen GT                    (ceiling)
    T_find_op_tree          frozen geom+type, learn boolean  (can it find the op?)
    T_find_type_bcvol_tree  frozen geom, learn type          (can it find types?)
    T_find_both_bcvol_tree  frozen geom, learn type+boolean  (can it find both?)
    T_wrong_type_bcvol_tree same but wrong type init         (can it recover?)

  N-ARY DNF ARCHITECTURE:
    N_perfect_dnf           all frozen GT                    (ceiling)
    N_find_membership_dnf   frozen geom+type, learn member.  (can it find the boolean?)
    N_find_type_bcvol_dnf   frozen geom, learn type          (can it find types?)
    N_find_both_bcvol_dnf   frozen geom, learn type+member.  (can it find both?)
    N_wrong_type_bcvol_dnf  same but wrong type init         (can it recover?)

Modes
-----
--mode controlled    Runs controlled tests on analytic targets
--mode full          Runs full free-geometry benchmark
--mode both          Runs both

Examples
--------
python controlled_nary_tree_benchmark.py \\
    --mode controlled --controlled-target box_sphere_minus_2cyl --epochs 12000
"""

import os
import sys
import time
import math
import json
import copy
import argparse
from typing import Dict, List, Optional

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

from nary_benchmark_v7 import MeshSDF, sample_3D_grid, mc_from_occ, mc_from_sdf, compute_iou, compute_fscore, batched_forward, euler_to_R, sdf_sphere_local, sdf_block_local, sdf_cylinder_local, sdf_wedge_local, sdf_corner_wedge_local, BcTypePool, BaselineCSG, BcVariantCSG, surface_init, spatial_group_init, train, full_eval, device, TYPE_NAMES

# Convenience wrappers matching V7's BcVariantCSG
class BcTypeCSG(BcVariantCSG):
    def __init__(self, num_primitives=50, num_groups=16, omega=10.0, max_group_size=8):
        super().__init__(num_primitives=num_primitives, num_groups=num_groups,
                         omega=omega, max_group_size=max_group_size, occ_mode='bc')

class BcVolCSG(BcVariantCSG):
    def __init__(self, num_primitives=50, num_groups=16, omega=10.0, max_group_size=8):
        super().__init__(num_primitives=num_primitives, num_groups=num_groups,
                         omega=omega, max_group_size=max_group_size, occ_mode='bc_vol_norm')

def save_plot(results, output_path):
    """Minimal plot fallback."""
    import matplotlib.pyplot as plt
    methods = list(results.keys())
    n = len(methods)
    fig, ax = plt.subplots(figsize=(max(8, n*0.8), 5))
    vals = [results[m].get('iou', 0) for m in methods]
    ax.bar(range(n), vals)
    ax.set_xticks(range(n))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100); plt.close()

MIN_SIZE = 0.02

# Simple analytic targets for full suite
def _target_complex_csg(p):
    d = torch.abs(p) - 0.5
    box = torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    c = torch.tensor([0.3,0.3,0.3],device=p.device)
    sph = torch.sqrt(((p-c)**2).sum(-1)+1e-8)-0.35
    u = torch.minimum(box,sph)
    cy = torch.sqrt(p[:,0]**2+p[:,2]**2+1e-8)-0.15
    cx = torch.sqrt(p[:,1]**2+p[:,2]**2+1e-8)-0.15
    return (torch.maximum(torch.maximum(u,-cy),-cx)<0).float()

TARGETS = {'complex_csg': _target_complex_csg}


# Controlled targets with known GT decompositions
def _leaf(type_name, center, size, rot=(0.0, 0.0, 0.0)):
    return {'type': type_name, 'center': list(center), 'size': list(size), 'rot': list(rot)}

CONTROLLED_TARGETS = {
    'sphere_minus_cylinder': {
        'desc': 'sphere - cylinder',
        'leaves': [
            _leaf('sphere', (0.0, 0.0, 0.0), (0.70, 0.70, 0.70)),
            _leaf('cylinder', (0.0, 0.0, 0.0), (0.20, 1.10, 0.20)),
        ],
        'tree_ops': ['left_diff_right'],
        'dnf_groups': [{'pos': [0], 'neg': [1]}],
    },
    'box_sphere_minus_2cyl': {
        'desc': '(box ∪ sphere) - (cyl_x ∪ cyl_y)',
        'leaves': [
            _leaf('block',    (0.10, 0.10, 0.00), (0.55, 0.42, 0.42)),
            _leaf('sphere',   (-0.15, -0.05, -0.05), (0.45, 0.45, 0.45)),
            _leaf('cylinder', (0.00, 0.00, 0.00), (0.18, 1.10, 0.18), (0.0, 0.0, 0.0)),
            _leaf('cylinder', (0.00, 0.00, 0.00), (0.18, 1.10, 0.18), (0.0, 0.0, math.pi / 2.0)),
        ],
        'tree_ops': ['union', 'union', 'left_diff_right'],
        'dnf_groups': [
            {'pos': [0], 'neg': [2, 3]},
            {'pos': [1], 'neg': [2, 3]},
        ],
    },
}
CONTROLLED_TARGETS['complex_csg'] = CONTROLLED_TARGETS['box_sphere_minus_2cyl']

OP_NAMES = ['intersection', 'union', 'left_diff_right', 'right_diff_left']
OP_TO_INDEX = {n: i for i, n in enumerate(OP_NAMES)}
TYPE_TO_INDEX = {n: i for i, n in enumerate(TYPE_NAMES)}
WRONG_TYPE = {
    'sphere': 'block', 'block': 'sphere', 'cylinder': 'wedge',
    'wedge': 'cylinder', 'corner_wedge': 'sphere',
}

def _clone_spec(spec):
    return copy.deepcopy(spec)


# GT target evaluation
def leaf_sdf_from_spec(p, leaf):
    center = torch.tensor(leaf['center'], device=p.device, dtype=p.dtype)
    rot = torch.tensor(leaf['rot'], device=p.device, dtype=p.dtype)
    size = torch.tensor(leaf['size'], device=p.device, dtype=p.dtype)
    t = leaf['type']
    if t == 'sphere':
        return sdf_sphere_local(p - center, size[0])
    R = euler_to_R(rot)
    pl = torch.matmul(p - center, R.T)
    if t == 'block': return sdf_block_local(pl, size)
    if t == 'cylinder': return sdf_cylinder_local(pl, size[0], size[1])
    if t == 'wedge': return sdf_wedge_local(pl, size)
    if t == 'corner_wedge': return sdf_corner_wedge_local(pl, size)
    raise ValueError(f'Unknown type: {t}')

def hard_sdf_op(name, a, b):
    if name == 'union': return torch.minimum(a, b)
    if name == 'intersection': return torch.maximum(a, b)
    if name == 'left_diff_right': return torch.maximum(a, -b)
    if name == 'right_diff_left': return torch.maximum(b, -a)
    raise ValueError(name)

def reduce_hard_tree(leaf_sdfs, ops):
    cur = list(leaf_sdfs); op_idx = 0
    while len(cur) > 1:
        nxt = []; j = 0
        while j + 1 < len(cur):
            nxt.append(hard_sdf_op(ops[op_idx], cur[j], cur[j+1]))
            op_idx += 1; j += 2
        if j < len(cur): nxt.append(cur[j])
        cur = nxt
    return cur[0]

def controlled_target_occ(name, p):
    spec = CONTROLLED_TARGETS[name]
    leaf_sdfs = [leaf_sdf_from_spec(p, leaf) for leaf in spec['leaves']]
    return (reduce_hard_tree(leaf_sdfs, spec['tree_ops']) < 0).float()


# Helpers for sine-discrete initialization
def sine_onehot_raw(num_classes, target_idx, omega, device_, dtype_):
    lo = -math.pi / (2.0 * omega)
    hi = +math.pi / (2.0 * omega)
    x = torch.full((num_classes,), lo, device=device_, dtype=dtype_)
    x[target_idx] = hi
    return x

def set_requires_grad_(tensor_or_param, flag):
    if isinstance(tensor_or_param, torch.nn.Parameter):
        tensor_or_param.requires_grad_(flag)
    else:
        raise TypeError(type(tensor_or_param))


# Controlled leaf init for Bc-type pool
def size3_from_leaf(leaf):
    size = torch.tensor(leaf['size'], dtype=torch.float32)
    if leaf['type'] == 'sphere':
        r = size[0]; return torch.tensor([r, r, r], dtype=torch.float32)
    if leaf['type'] == 'cylinder':
        return torch.tensor([size[0], size[1], size[2]], dtype=torch.float32)
    return size.clone()

def sphere_radius_from_size3(s):
    return (s[0] * s[1] * s[2]).clamp(min=MIN_SIZE**3).pow(1.0/3.0)

def cylinder_from_size3(s):
    r = torch.sqrt((s[0] * s[2]).clamp(min=MIN_SIZE**2))
    h = s[1].clamp(min=MIN_SIZE)
    return torch.stack([r, h])

def init_bc_pool_from_oracle(pool, leaves, wrong_type_init=False, sharpness=30.0):
    with torch.no_grad():
        for i, leaf in enumerate(leaves):
            center = torch.tensor(leaf['center'], dtype=torch.float32, device=pool.centers.device)
            rot = torch.tensor(leaf['rot'], dtype=torch.float32, device=pool.rotations.device)
            s3 = size3_from_leaf(leaf).to(pool.intrinsic.device)
            pool.centers[i].copy_(center)
            pool.rotations[i].copy_(rot)
            pool.sharpness[i].fill_(sharpness)
            pool.intrinsic[i, 0] = sphere_radius_from_size3(s3)
            pool.intrinsic[i, 1:4] = s3
            pool.intrinsic[i, 4:6] = cylinder_from_size3(s3)
            pool.intrinsic[i, 6:9] = s3
            pool.intrinsic[i, 9:12] = s3
            tname = WRONG_TYPE[leaf['type']] if wrong_type_init else leaf['type']
            tidx = TYPE_TO_INDEX[tname]
            pool.type_logits[i].copy_(sine_onehot_raw(5, tidx, pool.omega,
                                                      pool.type_logits.device,
                                                      pool.type_logits.dtype))

def freeze_bc_pool_geometry(pool, freeze_geometry=True, freeze_sharpness=True, freeze_type=False):
    set_requires_grad_(pool.centers, not freeze_geometry)
    set_requires_grad_(pool.rotations, not freeze_geometry)
    set_requires_grad_(pool.intrinsic, not freeze_geometry)
    set_requires_grad_(pool.sharpness, not freeze_sharpness)
    set_requires_grad_(pool.type_logits, not freeze_type)


# Controlled models: pairwise tree
class FixedDiscreteLeaves(nn.Module):
    def __init__(self, leaves, sharpness=40.0):
        super().__init__()
        self.leaves = _clone_spec({'leaves': leaves})['leaves']
        self.sharpness = sharpness
    def get_all_occs(self, p):
        occs = []
        for leaf in self.leaves:
            sdf = leaf_sdf_from_spec(p, leaf)
            occs.append(torch.sigmoid(-sdf * self.sharpness))
        return torch.stack(occs, dim=-1)

class ControlledTreeDiscrete(nn.Module):
    def __init__(self, spec, train_boolean):
        super().__init__()
        self.spec = _clone_spec(spec)
        self.leaf_module = FixedDiscreteLeaves(self.spec['leaves'])
        self.omega = 10.0; self.N = len(self.spec['leaves'])
        self.boolean_params = nn.Parameter(torch.zeros(self.N-1, 4), requires_grad=train_boolean)
        self._init_boolean_gt()
    def _init_boolean_gt(self):
        with torch.no_grad():
            for i, opname in enumerate(self.spec['tree_ops']):
                idx = OP_TO_INDEX[opname]
                self.boolean_params[i].copy_(sine_onehot_raw(4, idx, self.omega,
                    self.boolean_params.device, self.boolean_params.dtype))
    def _bool_merge(self, x, y, params, temperature=1.0):
        w = torch.softmax(torch.sin(self.omega * params) * temperature, dim=-1)
        c0,c1,c2,c3 = w[...,0],w[...,1],w[...,2],w[...,3]
        return (c1+c2)*x + (c1+c3)*y + (c0-c1-c2-c3)*x*y
    def _reduce(self, leaves, temperature=1.0, hard=False):
        cur = list(torch.unbind(leaves, dim=1)); idx = 0
        while len(cur) > 1:
            nxt = []; j = 0
            while j+1 < len(cur):
                if hard:
                    w = torch.softmax(torch.sin(self.omega*self.boolean_params[idx])*20.0, dim=-1)
                    one = torch.zeros_like(w); one[w.argmax()] = 1.0
                    c0,c1,c2,c3 = one[0],one[1],one[2],one[3]
                    merged = (c1+c2)*cur[j] + (c1+c3)*cur[j+1] + (c0-c1-c2-c3)*cur[j]*cur[j+1]
                else:
                    merged = self._bool_merge(cur[j], cur[j+1], self.boolean_params[idx], temperature)
                nxt.append(merged); idx += 1; j += 2
            if j < len(cur): nxt.append(cur[j])
            cur = nxt
        return cur[0]
    def forward(self, p, temperature=1.0, **kw):
        return self._reduce(self.leaf_module.get_all_occs(p), temperature=temperature)
    def forward_hard_dnf(self, p, thr=0.5):
        return self._reduce(self.leaf_module.get_all_occs(p), hard=True)
    def structure_loss(self, ls=0.0, lb=0.0, **kw):
        if not self.boolean_params.requires_grad:
            return self.boolean_params.sum() * 0.0
        w = torch.softmax(torch.sin(self.omega*self.boolean_params), dim=-1)
        return ls*w.sum() + lb*(w*(1-w)).sum()
    def get_stats(self, thr=0.3):
        types = {}
        for leaf in self.spec['leaves']:
            types[leaf['type']] = types.get(leaf['type'], 0) + 1
        return {'prims_used': self.N, 'groups_active': self.N-1, 'type_dist': types, 'type_entropy': 0.0, 'N': self.N}
    def print_status(self, thr=0.3):
        print(f"  Leaves: {self.N}, internal booleans: {self.N-1}")
        print(f"  Types: {self.get_stats()['type_dist']}")

class ControlledTreeBcVol(nn.Module):
    def __init__(self, spec, train_type, train_boolean, wrong_type_init=False):
        super().__init__()
        self.spec = _clone_spec(spec); self.omega = 10.0; self.N = len(self.spec['leaves'])
        self.pool = BcTypePool(self.N, omega=self.omega)
        init_bc_pool_from_oracle(self.pool, self.spec['leaves'], wrong_type_init=wrong_type_init)
        freeze_bc_pool_geometry(self.pool, freeze_geometry=True, freeze_sharpness=True, freeze_type=not train_type)
        self.boolean_params = nn.Parameter(torch.zeros(self.N-1, 4), requires_grad=train_boolean)
        self._init_boolean_gt()
    def _init_boolean_gt(self):
        with torch.no_grad():
            for i, opname in enumerate(self.spec['tree_ops']):
                idx = OP_TO_INDEX[opname]
                self.boolean_params[i].copy_(sine_onehot_raw(4, idx, self.omega,
                    self.boolean_params.device, self.boolean_params.dtype))
    def _bool_merge(self, x, y, params, temperature=1.0):
        w = torch.softmax(torch.sin(self.omega*params)*temperature, dim=-1)
        c0,c1,c2,c3 = w[...,0],w[...,1],w[...,2],w[...,3]
        return (c1+c2)*x + (c1+c3)*y + (c0-c1-c2-c3)*x*y
    def _reduce(self, leaves, temperature=1.0, hard=False):
        cur = list(torch.unbind(leaves, dim=1)); idx = 0
        while len(cur) > 1:
            nxt = []; j = 0
            while j+1 < len(cur):
                if hard:
                    w = torch.softmax(torch.sin(self.omega*self.boolean_params[idx])*20.0, dim=-1)
                    one = torch.zeros_like(w); one[w.argmax()] = 1.0
                    c0,c1,c2,c3 = one[0],one[1],one[2],one[3]
                    merged = (c1+c2)*cur[j] + (c1+c3)*cur[j+1] + (c0-c1-c2-c3)*cur[j]*cur[j+1]
                else:
                    merged = self._bool_merge(cur[j], cur[j+1], self.boolean_params[idx], temperature)
                nxt.append(merged); idx += 1; j += 2
            if j < len(cur): nxt.append(cur[j])
            cur = nxt
        return cur[0]
    def forward(self, p, temperature=1.0, **kw):
        return self._reduce(self.pool.get_all_occs(p, mode='bc_vol_norm', temperature=temperature), temperature=temperature)
    def forward_hard_dnf(self, p, thr=0.5):
        return self._reduce(self.pool.get_all_occs(p, mode='bc_vol_norm', temperature=20.0), hard=True)
    def structure_loss(self, ls=1e-5, lb=1e-5, lt=5e-5, temperature=1.0, **kw):
        z = self.boolean_params.sum()*0.0 + self.pool.type_logits.sum()*0.0
        out = z
        if self.boolean_params.requires_grad:
            w = torch.softmax(torch.sin(self.omega*self.boolean_params), dim=-1)
            out = out + ls*w.sum() + lb*(w*(1-w)).sum()
        if self.pool.type_logits.requires_grad:
            tw = torch.sigmoid(torch.sin(self.pool.omega*self.pool.type_logits)*temperature)
            out = out + lt*tw.sum() + lt*(tw*(1-tw)).sum()
        return out
    def get_stats(self, thr=0.3):
        types = {}
        for i in range(self.N):
            t = self.pool.get_dominant_type(i); types[t] = types.get(t,0)+1
        ent = self.pool.type_entropy()
        return {'prims_used': self.N, 'groups_active': self.N-1, 'type_dist': types, 'type_entropy': ent, 'N': self.N}
    def print_status(self, thr=0.3):
        s = self.get_stats()
        print(f"  Leaves: {self.N}, internal booleans: {self.N-1}")
        print(f"  Types: {s['type_dist']}, entropy: {s['type_entropy']:.3f}")


# Controlled models: n-ary DNF
class ControlledNaryDiscrete(nn.Module):
    """Fixed GT leaves, optionally learn membership (m_pos, m_neg, act)."""
    def __init__(self, spec, train_membership):
        super().__init__()
        self.spec = _clone_spec(spec)
        self.leaf_module = FixedDiscreteLeaves(self.spec['leaves'])
        self.N = len(self.spec['leaves']); self.K = len(self.spec['dnf_groups'])
        self.m_pos = nn.Parameter(torch.full((self.N, self.K), -6.0), requires_grad=train_membership)
        self.m_neg = nn.Parameter(torch.full((self.N, self.K), -6.0), requires_grad=train_membership)
        self.act = nn.Parameter(torch.full((self.K,), -6.0), requires_grad=train_membership)
        self._init_gt_membership()
    def _init_gt_membership(self):
        with torch.no_grad():
            for g, clause in enumerate(self.spec['dnf_groups']):
                self.act[g] = 6.0
                for i in clause['pos']: self.m_pos[i,g] = 6.0
                for i in clause['neg']: self.m_neg[i,g] = 6.0
    def _aggregate(self, occs):
        mp=torch.sigmoid(self.m_pos);mn=torch.sigmoid(self.m_neg);a=torch.sigmoid(self.act)
        o=occs.unsqueeze(-1)
        fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
        lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag=a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)
    def forward(self, p, temperature=1.0, **kw):
        return self._aggregate(self.leaf_module.get_all_occs(p))
    def forward_hard_dnf(self, p, thr=0.5):
        occs=self.leaf_module.get_all_occs(p)
        mp=(torch.sigmoid(self.m_pos)>thr).float();mn=(torch.sigmoid(self.m_neg)>thr).float()
        a=(torch.sigmoid(self.act)>thr).float();o=occs.unsqueeze(-1)
        fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
        lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag=a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)
    def structure_loss(self, ls=1e-5, lb=1e-5, lc=1e-5, **kw):
        z = self.m_pos.sum()*0.0 + self.m_neg.sum()*0.0 + self.act.sum()*0.0
        if not self.m_pos.requires_grad: return z
        mp=torch.sigmoid(self.m_pos);mn=torch.sigmoid(self.m_neg);a=torch.sigmoid(self.act)
        return z + ls*(mp.sum()+mn.sum()+a.sum()) + lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum()) + lc*(mp*mn).sum()
    def get_stats(self, thr=0.3):
        types = {}
        for leaf in self.spec['leaves']: types[leaf['type']] = types.get(leaf['type'],0)+1
        active = (torch.sigmoid(self.act).detach()>thr).sum().item()
        return {'prims_used': self.N, 'groups_active': active, 'type_dist': types, 'type_entropy': 0.0, 'N': self.N}
    def print_status(self, thr=0.3):
        s=self.get_stats()
        print(f"  Prims: {self.N}, groups active: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}")

class ControlledNaryBcVol(nn.Module):
    """Bc+vol-norm leaves with oracle geometry, optionally learn type + membership."""
    def __init__(self, spec, train_type, train_membership, wrong_type_init=False):
        super().__init__()
        self.spec = _clone_spec(spec)
        self.N = len(self.spec['leaves']); self.K = len(self.spec['dnf_groups'])
        self.pool = BcTypePool(self.N, omega=10.0)
        init_bc_pool_from_oracle(self.pool, self.spec['leaves'], wrong_type_init=wrong_type_init)
        freeze_bc_pool_geometry(self.pool, freeze_geometry=True, freeze_sharpness=True, freeze_type=not train_type)
        self.m_pos = nn.Parameter(torch.full((self.N, self.K), -6.0), requires_grad=train_membership)
        self.m_neg = nn.Parameter(torch.full((self.N, self.K), -6.0), requires_grad=train_membership)
        self.act = nn.Parameter(torch.full((self.K,), -6.0), requires_grad=train_membership)
        self._init_gt_membership()
    def _init_gt_membership(self):
        with torch.no_grad():
            for g, clause in enumerate(self.spec['dnf_groups']):
                self.act[g] = 6.0
                for i in clause['pos']: self.m_pos[i,g] = 6.0
                for i in clause['neg']: self.m_neg[i,g] = 6.0
    def _aggregate(self, occs):
        mp=torch.sigmoid(self.m_pos);mn=torch.sigmoid(self.m_neg);a=torch.sigmoid(self.act)
        o=occs.unsqueeze(-1)
        fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
        lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag=a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)
    def forward(self, p, temperature=1.0, **kw):
        return self._aggregate(self.pool.get_all_occs(p, mode='bc_vol_norm', temperature=temperature))
    def forward_hard_dnf(self, p, thr=0.5):
        occs=self.pool.get_all_occs(p, mode='bc_vol_norm', temperature=20.0)
        mp=(torch.sigmoid(self.m_pos)>thr).float();mn=(torch.sigmoid(self.m_neg)>thr).float()
        a=(torch.sigmoid(self.act)>thr).float();o=occs.unsqueeze(-1)
        fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
        lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag=a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)
    def structure_loss(self, ls=1e-5, lb=1e-5, lc=1e-5, lt=5e-5, temperature=1.0, **kw):
        z = self.m_pos.sum()*0.0 + self.m_neg.sum()*0.0 + self.act.sum()*0.0 + self.pool.type_logits.sum()*0.0
        out = z
        if self.m_pos.requires_grad:
            mp=torch.sigmoid(self.m_pos);mn=torch.sigmoid(self.m_neg);a=torch.sigmoid(self.act)
            out = out + ls*(mp.sum()+mn.sum()+a.sum()) + lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum()) + lc*(mp*mn).sum()
        if self.pool.type_logits.requires_grad:
            tw = torch.sigmoid(torch.sin(self.pool.omega*self.pool.type_logits)*temperature)
            out = out + lt*tw.sum() + lt*(tw*(1-tw)).sum()
        return out
    def get_stats(self, thr=0.3):
        active = (torch.sigmoid(self.act).detach()>thr).sum().item()
        types = {}
        for i in range(self.N): t=self.pool.get_dominant_type(i); types[t]=types.get(t,0)+1
        ent = self.pool.type_entropy()
        return {'prims_used': self.N, 'groups_active': active, 'type_dist': types, 'type_entropy': ent, 'N': self.N}
    def print_status(self, thr=0.3):
        s=self.get_stats()
        print(f"  Prims: {self.N}, groups active: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}, entropy: {s['type_entropy']:.3f}")


# Evaluation + training helpers
def controlled_eval(model, target_name, output_folder, resolution, label, eval_temperature=1.0):
    os.makedirs(output_folder, exist_ok=True)
    eval_grid = sample_3D_grid(resolution)
    eval_gt = controlled_target_occ(target_name, eval_grid)
    pred = batched_forward(model, eval_grid, temperature=eval_temperature)
    iou = compute_iou(pred, eval_gt)
    mse = ((pred - eval_gt)**2).mean().item()
    fs = compute_fscore(pred, eval_gt)
    mc_from_occ(os.path.join(output_folder, 'mesh_soft.obj'), pred, resolution)
    dnf_iou = 0.0
    if hasattr(model, 'forward_hard_dnf'):
        with torch.no_grad():
            dp = torch.cat([model.forward_hard_dnf(b) for b in torch.split(eval_grid, 32**3)], 0)
        dnf_iou = compute_iou(dp, eval_gt)
        mc_from_occ(os.path.join(output_folder, 'mesh_hard.obj'), dp, resolution)
    stats = model.get_stats()
    results = {'iou': iou, 'dnf_iou': dnf_iou, 'mse': mse, 'fscore': fs}
    results.update({k:v for k,v in stats.items() if not isinstance(v, (list, dict))})
    results['type_dist'] = stats.get('type_dist', {})
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  {label}: IoU={iou:.4f}, Hard={dnf_iou:.4f}, MSE={mse:.6f}, F={fs:.4f}")
    return results

def num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def maybe_train_controlled(model, target_name, output_folder, epochs, resolution, samples, desc,
                           use_temperature=True, resume=True):
    if num_trainable_params(model) == 0:
        print(f"  {desc}: no trainable parameters, skipping optimization")
        return None
    target_func = lambda p: controlled_target_occ(target_name, p)
    reg = dict(
        sparsity_warmup=max(epochs//20, 200), sparsity_rampup=max(epochs//12, 400),
        ls=1e-5, lb=1e-5, lc=1e-5, lgs=1e-5,
        ba_start=epochs//4, ba_end=int(epochs*0.75), ba_max=5e-5,
    )
    train(model, target_func, None, epochs, output_folder,
          num_samples=samples, resolution=resolution, resume=resume, desc=desc,
          use_temperature=use_temperature, temp_start=1.0, temp_end=20.0, **reg)


# Controlled benchmark suite - 10 conditions
def run_controlled_suite(args):
    target_name = args.controlled_target
    if target_name not in CONTROLLED_TARGETS:
        raise ValueError(f"Unknown target {target_name}")
    spec = CONTROLLED_TARGETS[target_name]
    out_root = os.path.join(args.output, 'controlled', target_name)
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'═'*74}")
    print(f"  CONTROLLED FACTORIAL SUITE — {target_name}")
    print(f"{'═'*74}")
    print(f"  Target: {spec['desc']}")
    print(f"  Leaves: {len(spec['leaves'])}, Tree ops: {spec['tree_ops']}, "
          f"DNF groups: {len(spec['dnf_groups'])}")
    print(f"")
    print(f"  TREE CONDITIONS:")
    print(f"    T_perfect         all frozen (ceiling)")
    print(f"    T_find_op         learn boolean only")
    print(f"    T_find_type       learn type only (Bc+vol)")
    print(f"    T_find_both       learn type + boolean")
    print(f"    T_wrong_type      wrong type init, learn both")
    print(f"  N-ARY CONDITIONS:")
    print(f"    N_perfect         all frozen (ceiling)")
    print(f"    N_find_membership learn membership only")
    print(f"    N_find_type       learn type only (Bc+vol)")
    print(f"    N_find_both       learn type + membership")
    print(f"    N_wrong_type      wrong type init, learn both")
    print(f"")
    print(f"  Epochs: {args.epochs}, Samples: {args.samples}, Res: {args.resolution}")
    print(f"  Device: {device}")
    print(f"{'═'*74}\n")

    eval_grid = sample_3D_grid(args.resolution)
    eval_gt = controlled_target_occ(target_name, eval_grid)
    mc_from_occ(os.path.join(out_root, '00_gt.obj'), eval_gt, args.resolution)
    all_results = {}

    # 10 conditions
    runs = [
        # TREE: 5 conditions
        ('T_perfect_tree',
         lambda: ControlledTreeDiscrete(spec, train_boolean=False),
         False, False),
        ('T_find_op_tree',
         lambda: ControlledTreeDiscrete(spec, train_boolean=True),
         True, False),
        ('T_find_type_bcvol_tree',
         lambda: ControlledTreeBcVol(spec, train_type=True, train_boolean=False),
         True, True),
        ('T_find_both_bcvol_tree',
         lambda: ControlledTreeBcVol(spec, train_type=True, train_boolean=True),
         True, True),
        ('T_wrong_type_bcvol_tree',
         lambda: ControlledTreeBcVol(spec, train_type=True, train_boolean=True, wrong_type_init=True),
         True, True),
        # N-ARY: 5 conditions
        ('N_perfect_dnf',
         lambda: ControlledNaryDiscrete(spec, train_membership=False),
         False, False),
        ('N_find_membership_dnf',
         lambda: ControlledNaryDiscrete(spec, train_membership=True),
         True, False),
        ('N_find_type_bcvol_dnf',
         lambda: ControlledNaryBcVol(spec, train_type=True, train_membership=False),
         True, True),
        ('N_find_both_bcvol_dnf',
         lambda: ControlledNaryBcVol(spec, train_type=True, train_membership=True),
         True, True),
        ('N_wrong_type_bcvol_dnf',
         lambda: ControlledNaryBcVol(spec, train_type=True, train_membership=True, wrong_type_init=True),
         True, True),
    ]

    for label, make_model, do_train, use_temp in runs:
        print(f"\n{'─'*74}\n  {label}\n{'─'*74}")
        torch.manual_seed(42); np.random.seed(42)
        model = make_model().to(device)
        print(f"  Trainable params: {num_trainable_params(model):,}")
        t0 = time.time()
        if do_train:
            maybe_train_controlled(model, target_name, os.path.join(out_root, label),
                                   args.epochs, args.resolution, args.samples,
                                   desc=label, use_temperature=use_temp,
                                   resume=(not args.no_resume))
        res = controlled_eval(model, target_name, os.path.join(out_root, label),
                              args.resolution, label,
                              eval_temperature=20.0 if use_temp else 1.0)
        res['time_s'] = time.time() - t0
        all_results[label] = res
        model.print_status()

    # Comparison table
    print(f"\n{'━'*74}")
    print(f"  CONTROLLED COMPARISON — {target_name}")
    print(f"{'━'*74}")
    print(f"\n  {'Method':<28} {'Soft IoU':>9} {'Hard IoU':>9} {'MSE':>10} {'Time':>7}")
    print(f"  {'─'*68}")
    for m, r in all_results.items():
        print(f"  {m:<28} {r['iou']:>9.4f} {r.get('dnf_iou',0):>9.4f} "
              f"{r['mse']:>10.6f} {r.get('time_s',0):>6.0f}s")
    print(f"  {'─'*68}")

    # Key comparisons
    def dd(a, b):
        if a in all_results and b in all_results:
            return all_results[a]['iou'] - all_results[b]['iou']
        return None

    print(f"\n  DIAGNOSTICS:")
    comparisons = [
        # Ceilings
        ('T_perfect_tree', None, 'Tree ceiling (all frozen)'),
        ('N_perfect_dnf', None, 'N-ary ceiling (all frozen)'),
        # Boolean learning
        ('T_find_op_tree', 'T_perfect_tree', 'Tree: can it find the boolean?'),
        ('N_find_membership_dnf', 'N_perfect_dnf', 'N-ary: can it find the membership?'),
        # Type learning
        ('T_find_type_bcvol_tree', 'T_perfect_tree', 'Tree: can Bc+vol find types?'),
        ('N_find_type_bcvol_dnf', 'N_perfect_dnf', 'N-ary: can Bc+vol find types?'),
        # Both
        ('T_find_both_bcvol_tree', 'T_perfect_tree', 'Tree: type+boolean together'),
        ('N_find_both_bcvol_dnf', 'N_perfect_dnf', 'N-ary: type+membership together'),
        # Recovery from wrong init
        ('T_wrong_type_bcvol_tree', 'T_find_both_bcvol_tree', 'Tree: wrong init recovery cost'),
        ('N_wrong_type_bcvol_dnf', 'N_find_both_bcvol_dnf', 'N-ary: wrong init recovery cost'),
        # Tree vs N-ary
        ('T_find_both_bcvol_tree', 'N_find_both_bcvol_dnf', 'Tree vs N-ary (both learn)'),
        ('T_find_op_tree', 'N_find_membership_dnf', 'Tree boolean vs N-ary membership'),
    ]
    for a, b, msg in comparisons:
        if b is None:
            if a in all_results:
                print(f"    {msg:<48}: {all_results[a]['iou']:.4f}")
        else:
            d = dd(a, b)
            if d is not None:
                sym = '✓' if abs(d) < 0.005 else ('+' if d > 0 else '−')
                print(f"    {msg:<48}: {d:+.4f} [{sym}]")

    # Type distributions
    print(f"\n  TYPE DISTRIBUTIONS:")
    for m, r in all_results.items():
        td = r.get('type_dist', {})
        ent = r.get('type_entropy', -1)
        ent_str = f", ent={ent:.3f}" if ent >= 0 else ""
        print(f"    {m:<28} {td}{ent_str}")

    save_plot(all_results, os.path.join(out_root, 'comparison.png'))
    with open(os.path.join(out_root, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nOutput: {out_root}/")
    return all_results


# Full free-geometry benchmark (delegates to nary_benchmark_v7)
def run_full_suite(args):
    mesh_sdf = None
    if args.mesh and os.path.exists(args.mesh):
        mesh_sdf = MeshSDF(args.mesh)
        target_func = lambda p: torch.from_numpy((mesh_sdf.query(p.cpu().numpy())<0).astype(np.float32)).to(device)
        target_name = os.path.splitext(os.path.basename(args.mesh))[0]
    elif args.target in TARGETS:
        target_func = TARGETS[args.target]; target_name = args.target
    else:
        raise ValueError(f"No target. Use --mesh or --target {list(TARGETS.keys())}")

    OUT = os.path.join(args.output, 'full', target_name); os.makedirs(OUT, exist_ok=True)
    eval_grid = sample_3D_grid(args.resolution); eval_gt = target_func(eval_grid)
    if mesh_sdf:
        mc_from_sdf(os.path.join(OUT, '00_gt.obj'), mesh_sdf.query_sdf_grid(args.resolution), args.resolution)
    else:
        mc_from_occ(os.path.join(OUT, '00_gt.obj'), eval_gt, args.resolution)

    print(f"\n{'═'*74}\n  FULL N-ARY vs TREE — {target_name}\n{'═'*74}")
    reg = dict(sparsity_warmup=max(args.epochs//20,1500), sparsity_rampup=max(args.epochs//12,2000),
               ls=2e-4,lb=5e-5,lc=2e-4,lgs=1e-5,
               ba_start=args.epochs//4,ba_end=int(args.epochs*0.75),ba_max=5e-4)
    all_results = {}

    full_runs = [
        ('A_nary_discrete', lambda: BaselineCSG(prims_per_type=args.baseline_prims, num_groups=args.groups).to(device), False),
        ('G_nary_bc', lambda: BcTypeCSG(num_primitives=args.bc_prims, num_groups=args.groups, omega=args.omega).to(device), True),
        ('K_nary_bc_vol', lambda: BcVolCSG(num_primitives=args.bc_prims, num_groups=args.groups, omega=args.omega).to(device), True),
    ]
    for label, make_model, use_temp in full_runs:
        print(f"\n{'─'*74}\n  {label}\n{'─'*74}")
        torch.manual_seed(42); np.random.seed(42)
        model = make_model()
        if mesh_sdf: surface_init(model, mesh_sdf); spatial_group_init(model, mesh_sdf)
        t0 = time.time()
        train(model, target_func, mesh_sdf, args.epochs, os.path.join(OUT, label),
              num_samples=args.samples, resolution=args.resolution,
              resume=(not args.no_resume), desc=label,
              use_temperature=use_temp, temp_start=1.0, temp_end=20.0, **reg)
        res = full_eval(model, eval_grid, eval_gt, os.path.join(OUT, label),
                        args.resolution, label=label, eval_temperature=20.0 if use_temp else 1.0)
        res['time_s'] = time.time()-t0; all_results[label] = res; model.print_status()

    print(f"\n{'━'*74}\n  FULL COMPARISON — {target_name}\n{'━'*74}")
    for m,r in all_results.items():
        print(f"  {m:<20} IoU={r['iou']:.4f} Hard={r.get('dnf_iou',0):.4f}")
    with open(os.path.join(OUT,'results.json'),'w') as f: json.dump(all_results,f,indent=2)
    print(f"\nOutput: {OUT}/")
    return all_results


# Main
def main():
    parser = argparse.ArgumentParser(description='Controlled + full inverse-CSG benchmark')
    parser.add_argument('--mode', type=str, default='controlled', choices=['controlled','full','both'])
    parser.add_argument('--controlled-target', type=str, default='box_sphere_minus_2cyl',
                        choices=list(CONTROLLED_TARGETS.keys()))
    parser.add_argument('--mesh', type=str, default=None)
    parser.add_argument('--target', type=str, default='complex_csg')
    parser.add_argument('--epochs', type=int, default=12000)
    parser.add_argument('--baseline-prims', type=int, default=10)
    parser.add_argument('--bc-prims', type=int, default=50)
    parser.add_argument('--groups', type=int, default=14)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--output', type=str, default='./benchmark_controlled/')
    parser.add_argument('--omega', type=float, default=10.0)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    if args.mode in ('controlled', 'both'):
        run_controlled_suite(args)
    if args.mode in ('full', 'both'):
        run_full_suite(args)
    print('Done!')

if __name__ == '__main__':
    main()
