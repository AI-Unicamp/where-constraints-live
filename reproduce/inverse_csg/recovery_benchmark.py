"""
Limits of the fuzzy relaxation in inverse CSG: type recovery experiment.

Tests:
  1. CORRELATION: Does boolean recovery predict type recovery?
  2. JOINT SWEEP: Does fixing booleans via discrete sweep unlock type recovery?
  3. CEILING: What's the best achievable with brute-force / greedy enumeration?

METHODS:
  - gt_family_only:  only GT types, soft selection -> phase 2 (difficulty control)
  - soft_baseline:   all 5 types, soft selection -> phase 2 (no intervention)
  - cf_standard:     all 5 -> counterfactual type sweep -> phase 2
  - cf_joint:        all 5 -> bool sweep + type sweep (repeat) -> phase 2
  - ceiling_search:  all 5 -> enumerate type+op combos -> phase 2
                     (exhaustive for depth 1, greedy subtree for depth 2)

DUAL PHASE 2:
  Each run produces two phase 2 results:
  - frozen:  booleans locked, geometry only  (clean causal test)
  - full:    booleans free to drift          (practical result)

Usage:
    python recovery_benchmark.py
    python recovery_benchmark.py --seeds 42 123 456 789 --phase1 2500
"""

import os, json, argparse, time as _time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# SDF PRIMITIVES
MIN_SIZE = 0.02

def euler_to_rotation_matrix(angles):
    cx, sx = torch.cos(angles[0]), torch.sin(angles[0])
    cy, sy = torch.cos(angles[1]), torch.sin(angles[1])
    cz, sz = torch.cos(angles[2]), torch.sin(angles[2])
    R = torch.zeros(3, 3, device=angles.device, dtype=angles.dtype)
    R[0,0]=cy*cz;  R[0,1]=cz*sx*sy-cx*sz;  R[0,2]=cx*cz*sy+sx*sz
    R[1,0]=cy*sz;  R[1,1]=cx*cz+sx*sy*sz;   R[1,2]=cx*sy*sz-cz*sx
    R[2,0]=-sy;    R[2,1]=cy*sx;             R[2,2]=cx*cy
    return R

def sdf_block(p, center, half_size, rotation):
    pl = torch.matmul(p - center, rotation.T)
    d = torch.abs(pl) - half_size
    return torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + \
           torch.clamp(d.max(-1)[0], max=0)

def sdf_sphere(p, center, radius):
    return torch.sqrt(((p - center)**2).sum(-1) + 1e-8) - radius

def sdf_cylinder(p, center, radius, half_height, rotation):
    pl = torch.matmul(p - center, rotation.T)
    dr = torch.sqrt(pl[:,0]**2 + pl[:,2]**2 + 1e-8) - radius
    dh = torch.abs(pl[:,1]) - half_height
    return torch.sqrt(torch.clamp(dr, min=0)**2 + torch.clamp(dh, min=0)**2 + 1e-8) + \
           torch.clamp(torch.maximum(dr, dh), max=0)

def sdf_wedge(p, center, half_size, rotation):
    pl = torch.matmul(p - center, rotation.T)
    hy, hz = half_size[1], half_size[2]
    d = torch.abs(pl) - half_size
    box = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + \
          torch.clamp(d.max(-1)[0], max=0)
    inv = 1.0 / torch.sqrt(1.0/(hy**2) + 1.0/(hz**2) + 1e-8)
    return torch.maximum(box, (pl[:,1]/hy + pl[:,2]/hz - 1.0) * inv)

def sdf_corner_wedge(p, center, half_size, rotation):
    pl = torch.matmul(p - center, rotation.T)
    hx, hy, hz = half_size[0], half_size[1], half_size[2]
    d = torch.abs(pl) - half_size
    box = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + 1e-8) + \
          torch.clamp(d.max(-1)[0], max=0)
    i1 = 1.0 / torch.sqrt(1.0/(hx**2) + 1.0/(hy**2) + 1e-8)
    i2 = 1.0 / torch.sqrt(1.0/(hz**2) + 1.0/(hy**2) + 1e-8)
    return torch.maximum(
        torch.maximum(box, (pl[:,0]/hx + pl[:,1]/hy - 1.0)*i1),
        (pl[:,2]/hz + pl[:,1]/hy - 1.0)*i2)

PRIM_TYPES = ['block', 'sphere', 'cylinder', 'wedge', 'corner_wedge']
OP_NAMES = ['intersection', 'union', 'difference', 'rev_difference']

def eval_sdf(ptype, p, center, size, rotation):
    c = torch.tensor(center, device=p.device, dtype=torch.float32)
    R = euler_to_rotation_matrix(torch.tensor(rotation, device=p.device, dtype=torch.float32))
    if ptype == 'sphere':
        return sdf_sphere(p, c, torch.tensor(size[0], device=p.device))
    elif ptype == 'block':
        return sdf_block(p, c, torch.tensor(size, device=p.device), R)
    elif ptype == 'cylinder':
        return sdf_cylinder(p, c, torch.tensor(size[0], device=p.device),
                           torch.tensor(size[1], device=p.device), R)
    elif ptype == 'wedge':
        return sdf_wedge(p, c, torch.tensor(size, device=p.device), R)
    elif ptype == 'corner_wedge':
        return sdf_corner_wedge(p, c, torch.tensor(size, device=p.device), R)


# TARGETS
def make_target_occupancy(td, p):
    return (make_target_sdf(td, p) < 0).float()

def make_target_sdf(td, p):
    def ev(n, p):
        if n['type'] == 'primitive':
            return eval_sdf(n['prim_type'], p, n['center'], n['size'], n['rotation'])
        L = ev(n['left'], p); R_ = ev(n['right'], p)
        op = n['op']
        if op == 'union':          return torch.minimum(L, R_)
        elif op == 'intersection': return torch.maximum(L, R_)
        elif op == 'difference':   return torch.maximum(L, -R_)
        else:                      return torch.maximum(-L, R_)
    return ev(td, p)

TARGETS = {
    'sphere_union_box': {
        'type':'boolean','op':'union',
        'left':{'type':'primitive','prim_type':'sphere',
                'center':[-0.3,0,0],'size':[0.55],'rotation':[0,0,0]},
        'right':{'type':'primitive','prim_type':'block',
                 'center':[0.3,0,0],'size':[0.4,0.4,0.4],'rotation':[0,0,0]},
        'gt_types':['sphere','block'],'gt_ops':['union'],
        'depth':1,'description':'sphere ∪ box',
    },
    'box_minus_cylinder': {
        'type':'boolean','op':'difference',
        'left':{'type':'primitive','prim_type':'block',
                'center':[0,0,0],'size':[0.6,0.6,0.6],'rotation':[0,0,0]},
        'right':{'type':'primitive','prim_type':'cylinder',
                 'center':[0,0,0],'size':[0.3,0.7],'rotation':[0,0,0]},
        'gt_types':['block','cylinder'],'gt_ops':['difference'],
        'depth':1,'description':'box − cylinder',
    },
    'sphere_minus_sphere': {
        'type':'boolean','op':'difference',
        'left':{'type':'primitive','prim_type':'sphere',
                'center':[0,0,0],'size':[0.7],'rotation':[0,0,0]},
        'right':{'type':'primitive','prim_type':'sphere',
                 'center':[0,0.3,0],'size':[0.55],'rotation':[0,0,0]},
        'gt_types':['sphere','sphere'],'gt_ops':['difference'],
        'depth':1,'description':'sphere − sphere',
    },
    'union3_mixed': {
        'type':'boolean','op':'union',
        'left':{
            'type':'boolean','op':'union',
            'left':{'type':'primitive','prim_type':'block',
                    'center':[0,-0.3,0],'size':[0.6,0.15,0.6],'rotation':[0,0,0]},
            'right':{'type':'primitive','prim_type':'cylinder',
                     'center':[0,0.15,0],'size':[0.2,0.4],'rotation':[0,0,0]},
        },
        'right':{'type':'primitive','prim_type':'sphere',
                 'center':[0,0.55,0],'size':[0.25],'rotation':[0,0,0]},
        'gt_types':['block','cylinder','sphere'],'gt_ops':['union','union'],
        'depth':2,'description':'block ∪ cyl ∪ sphere',
    },
    'complex_4prim': {
        'type':'boolean','op':'union',
        'left':{
            'type':'boolean','op':'difference',
            'left':{'type':'primitive','prim_type':'block',
                    'center':[0,0,0],'size':[0.55,0.55,0.55],'rotation':[0,0,0]},
            'right':{'type':'primitive','prim_type':'cylinder',
                     'center':[0,0,0],'size':[0.25,0.7],'rotation':[0,0,0]},
        },
        'right':{
            'type':'boolean','op':'union',
            'left':{'type':'primitive','prim_type':'sphere',
                    'center':[0.4,0.4,0.4],'size':[0.3],'rotation':[0,0,0]},
            'right':{'type':'primitive','prim_type':'wedge',
                     'center':[-0.35,-0.35,0],'size':[0.35,0.35,0.35],'rotation':[0,0,0]},
        },
        'gt_types':['block','cylinder','sphere','wedge'],
        'gt_ops':['union','difference','union'],
        'depth':2,'description':'(box−cyl) ∪ (sph∪wedge)',
    },
}


# MODEL
class PrimitivePool(nn.Module):
    PC = {'block':9,'sphere':4,'cylinder':8,'wedge':9,'corner_wedge':9}
    def __init__(self, types, scale=0.5):
        super().__init__()
        self.types = types; self.num = len(types)
        total = sum(self.PC[t] for t in types)
        self.params = nn.Parameter(torch.randn(total)*scale)
        self.sharpness = nn.Parameter(torch.ones(self.num)*20.0)
        self.pidx = []; idx = 0
        for t in types:
            c = self.PC[t]; self.pidx.append((idx,idx+c)); idx += c

    def sdf(self, p, i):
        pt = self.types[i]; s,e = self.pidx[i]; par = self.params[s:e]
        if pt=='block':
            return sdf_block(p,par[:3],par[3:6].abs()+MIN_SIZE,
                           euler_to_rotation_matrix(par[6:9]))
        elif pt=='sphere':
            return sdf_sphere(p,par[:3],par[3].abs()+MIN_SIZE)
        elif pt=='cylinder':
            return sdf_cylinder(p,par[:3],par[3].abs()+MIN_SIZE,
                              par[4].abs()+MIN_SIZE,
                              euler_to_rotation_matrix(par[5:8]))
        elif pt=='wedge':
            return sdf_wedge(p,par[:3],par[3:6].abs()+MIN_SIZE,
                           euler_to_rotation_matrix(par[6:9]))
        elif pt=='corner_wedge':
            return sdf_corner_wedge(p,par[:3],par[3:6].abs()+MIN_SIZE,
                                   euler_to_rotation_matrix(par[6:9]))

    def occ(self, p, i):
        return torch.sigmoid(-self.sdf(p,i)*self.sharpness[i].abs().clamp(min=1.0))


class Leaf(nn.Module):
    def __init__(self, n, omega=10.0, scale=0.3):
        super().__init__()
        self.n = n; self.omega = omega
        self.logits = nn.Parameter(torch.randn(n)*scale)
        self.fixed = None
    def weights(self, temp=1000.0):
        return F.softmax(torch.sin(self.omega*self.logits)*temp, -1)
    def best(self):
        return self.weights(0.01).argmax().item()
    def forward(self, occ, temp=1.0):
        if self.fixed is not None: return occ[:,self.fixed]
        return (occ*self.weights(temp)).sum(-1)
    def fix(self):
        self.fixed = self.best(); return self.fixed


class CSGTree(nn.Module):
    def __init__(self, depth=2, ptypes=None, scale=0.3, omega=10.0):
        super().__init__()
        self.depth = depth; self.omega = omega
        self.nl = 2**depth; self.ni = 2**depth-1
        self.phase = 1; self.temperature = 100.0
        if ptypes is None: ptypes = PRIM_TYPES
        self.ptypes = ptypes; self.nt = len(ptypes)
        self.pool = PrimitivePool(ptypes*self.nl, scale)
        self.leaves = nn.ModuleList([Leaf(self.nt,omega,scale) for _ in range(self.nl)])
        self.bparams = nn.Parameter(torch.randn(self.ni,4)*scale)
        self._sel = None

    def bweights(self):
        return F.softmax(torch.sin(self.omega*self.bparams)*1000.0, -1)

    def _combine(self, L, R, w):
        c0,c1,c2,c3 = w[:,0:1],w[:,1:2],w[:,2:3],w[:,3:4]
        return (c1+c2)*L + (c1+c3)*R + (c0-c1-c2-c3)*L*R

    def _tree(self, lvs):
        cur = torch.stack(lvs, 0)
        bw = self.bweights(); bi = self.ni-1
        for lv in range(self.depth-1,-1,-1):
            nn_ = 2**lv; l,r = cur[0::2],cur[1::2]
            cur = self._combine(l,r,bw[bi-nn_+1:bi+1]); bi -= nn_
        return cur[0]

    def forward(self, p):
        if self.phase==2 and self._sel:
            occs = torch.stack([self.pool.occ(p,i) for i in self._sel],-1)
            return self._tree([occs[:,i] for i in range(self.nl)])
        all_o = torch.stack([self.pool.occ(p,i) for i in range(self.pool.num)],-1)
        lvs = []
        for i,leaf in enumerate(self.leaves):
            s,e = i*self.nt,(i+1)*self.nt
            lvs.append(leaf(all_o[:,s:e],self.temperature))
        return self._tree(lvs)

    def transition(self):
        self._sel = []
        for i,leaf in enumerate(self.leaves):
            li = leaf.fix(); self._sel.append(i*self.nt+li)
        self.phase = 2

    def get_types(self):
        return [self.ptypes[leaf.fixed if leaf.fixed is not None else leaf.best()]
                for leaf in self.leaves]

    def get_ops(self):
        bw = self.bweights().detach()
        return [OP_NAMES[bw[i].argmax().item()] for i in range(self.ni)]


# UTILITIES
def compute_iou(pred, gt):
    p=(pred>0.5).float(); g=(gt>0.5).float()
    return ((p*g).sum()/((p+g>0).float().sum()+1e-8)).item()

def sample_near_surface(sdf_fn, n, sr=0.3, nr=0.5, std=0.03):
    ns=int(n*sr); nn_=int(n*nr); nr_=n-ns-nn_
    cand=torch.rand(n*10,3,device=device)*2-1
    with torch.no_grad(): sv=sdf_fn(cand).abs()
    _,si=sv.topk(ns,largest=False); ps=cand[si]
    _,ni=sv.topk(nn_,largest=False)
    pn=(cand[ni]+torch.randn(nn_,3,device=device)*std).clamp(-0.99,0.99)
    pr=torch.rand(nr_,3,device=device)*2-1
    P=torch.cat([ps,pn,pr],0); return P[torch.randperm(P.shape[0])]

def eval_model(model, grid, bs=8192):
    with torch.no_grad():
        return torch.cat([model(b) for b in grid.split(bs)],0)

def score_type(sel, gt):
    gc=Counter(gt); sc=Counter(sel)
    return sum(min(c,sc.get(t,0)) for t,c in gc.items())/max(len(gt),1)

def score_ops(sel, gt):
    if not gt: return 1.0
    gc=Counter(gt); sc=Counter(sel)
    return sum(min(c,sc.get(o,0)) for o,c in gc.items())/len(gt)


# STATE MANAGEMENT
def _apply_phase2(model, indices):
    """Set phase 2 state explicitly. Always call after load_state_dict."""
    model.phase = 2
    model._sel = list(indices)
    for i, leaf in enumerate(model.leaves):
        leaf.fixed = indices[i] % model.nt

def _snap(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


# GEOMETRY TRANSFER
def extract_anchor(model, gi):
    """Extract effective (center, aniso_scale, rotation) from primitive.
    Returns sizes AFTER abs()+MIN_SIZE, i.e. the true geometric sizes."""
    pt = model.pool.types[gi]
    s, _ = model.pool.pidx[gi]; par = model.pool.params.data
    center = par[s:s+3].clone()
    if pt == 'sphere':
        r = par[s+3].abs().item() + MIN_SIZE
        return center, torch.tensor([r,r,r], device=par.device), torch.zeros(3, device=par.device)
    elif pt == 'cylinder':
        r = par[s+3].abs().item() + MIN_SIZE
        h = par[s+4].abs().item() + MIN_SIZE
        return center, torch.tensor([r,h,r], device=par.device), par[s+5:s+8].clone()
    else:
        return center, par[s+3:s+6].abs().clone()+MIN_SIZE, par[s+6:s+9].clone()


def inject_anchor(model, gi, center, scale, rot):
    """Write effective geometry into primitive params.
    Subtracts MIN_SIZE so that abs(raw)+MIN_SIZE reproduces the intended scale."""
    pt = model.pool.types[gi]
    ds, _ = model.pool.pidx[gi]
    with torch.no_grad():
        model.pool.params.data[ds:ds+3] = center
        if pt == 'sphere':
            r = (scale[0]*scale[1]*scale[2]).pow(1.0/3.0)
            model.pool.params.data[ds+3] = r - MIN_SIZE
        elif pt == 'cylinder':
            model.pool.params.data[ds+3] = (scale[0]+scale[2])/2.0 - MIN_SIZE
            model.pool.params.data[ds+4] = scale[1] - MIN_SIZE
            model.pool.params.data[ds+5:ds+8] = rot
        else:
            model.pool.params.data[ds+3:ds+6] = scale - MIN_SIZE
            model.pool.params.data[ds+6:ds+9] = rot


def force_boolean(model, node_idx, op_idx):
    pos = np.pi / (2.0 * model.omega)
    with torch.no_grad():
        model.bparams.data[node_idx, :] = -pos
        model.bparams.data[node_idx, op_idx] = pos

def get_boolean_idx(model, ni):
    return model.bweights()[ni].argmax().item()

def warm_start_all_candidates(model):
    """Copy selected primitive geometry to all non-selected candidates."""
    for i in range(model.nl):
        sel_gi = model._sel[i]
        center, scale, rot = extract_anchor(model, sel_gi)
        for t_idx in range(model.nt):
            gi = i * model.nt + t_idx
            if gi != sel_gi:
                inject_anchor(model, gi, center, scale, rot)


# TRAINING
def run_phase1(model, target_fn, sdf_fn, epochs, num_samples,
               lr=1e-3, temp_start=0.3, temp_end=2000.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    P = None
    for ep in range(epochs):
        prog = ep / max(epochs-1, 1)
        model.temperature = temp_start * (temp_end/temp_start)**prog
        if ep%10==0 or P is None:
            P = sample_near_surface(sdf_fn, num_samples)
            gt = target_fn(P).detach()
        loss = F.mse_loss(model(P), gt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def run_phase2(model, target_fn, sdf_fn, eval_grid, eval_gt,
               epochs, num_samples, lr=5e-4, freeze_bools=False):
    """Phase 2 with optional boolean freezing for clean causal test."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    P = None; best = 0.0
    for ep in range(epochs):
        if ep%10==0 or P is None:
            P = sample_near_surface(sdf_fn, num_samples)
            gt = target_fn(P).detach()
        loss = F.mse_loss(model(P), gt)
        opt.zero_grad(set_to_none=True); loss.backward()
        if freeze_bools:
            with torch.no_grad():
                if model.bparams.grad is not None:
                    model.bparams.grad.zero_()
                for leaf in model.leaves:
                    if leaf.logits.grad is not None:
                        leaf.logits.grad.zero_()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep+1)%300==0:
            iou = compute_iou(eval_model(model, eval_grid), eval_gt)
            best = max(best, iou)
    iou = compute_iou(eval_model(model, eval_grid), eval_gt)
    return max(best, iou)


def optimize_geometry(model, target_fn, sdf_fn, epochs, num_samples, lr=1e-3):
    """Geometry only: freeze booleans and leaf selections."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    P = None
    for ep in range(epochs):
        if ep%50==0 or P is None:
            P = sample_near_surface(sdf_fn, num_samples)
            gt = target_fn(P).detach()
        loss = F.mse_loss(model(P), gt)
        opt.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            if model.bparams.grad is not None:
                model.bparams.grad.zero_()
            for leaf in model.leaves:
                if leaf.logits.grad is not None:
                    leaf.logits.grad.zero_()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


# COUNTERFACTUAL TYPE SWEEP (changes accepted only if they beat the no-op baseline)
def sweep_types(model, target_fn, sdf_fn, eval_grid, eval_gt,
                num_samples=8192, epochs=200, lr=1e-3,
                threshold=0.001, passes=2):
    """Coordinate descent over leaf types.
    Accept logic: compare ALL candidates against noop_iou baseline.
    Type change accepted only if best_alternative > noop_iou + threshold."""
    base = _snap(model)
    indices = list(model._sel)
    total_changes = 0

    for _ in range(passes):
        pass_ch = 0
        for li in range(model.nl):
            cur_gi = indices[li]
            cur_ti = cur_gi % model.nt

            # Baseline: current config, no optimization
            model.load_state_dict(base)
            _apply_phase2(model, indices)
            noop_iou = compute_iou(eval_model(model, eval_grid), eval_gt)

            center, scale, rot = extract_anchor(model, cur_gi)

            # Evaluate ALL candidates (including current type)
            best_overall_iou = -1.0
            best_overall_ti = cur_ti
            best_overall_state = None
            cur_type_iou = -1.0
            cur_type_state = None

            for ti in range(model.nt):
                cand_gi = li * model.nt + ti

                model.load_state_dict(base)
                _apply_phase2(model, indices)
                model._sel[li] = cand_gi
                model.leaves[li].fixed = ti

                # Current type keeps trained params; others get warm-start
                if ti != cur_ti:
                    inject_anchor(model, cand_gi, center, scale, rot)

                cs, ce = model.pool.pidx[cand_gi]
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                P = None
                for ep in range(epochs):
                    if ep%50==0 or P is None:
                        P = sample_near_surface(sdf_fn, num_samples)
                        gt = target_fn(P).detach()
                    loss = F.mse_loss(model(P), gt)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    with torch.no_grad():
                        if model.pool.params.grad is not None:
                            mask = torch.zeros_like(model.pool.params.grad)
                            mask[cs:ce] = 1.0
                            model.pool.params.grad *= mask
                        if model.pool.sharpness.grad is not None:
                            sm = torch.zeros_like(model.pool.sharpness.grad)
                            sm[cand_gi] = 1.0
                            model.pool.sharpness.grad *= sm
                        if model.bparams.grad is not None:
                            model.bparams.grad.zero_()
                        for leaf in model.leaves:
                            if leaf.logits.grad is not None:
                                leaf.logits.grad.zero_()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                trial_iou = compute_iou(eval_model(model, eval_grid), eval_gt)

                # Track current type separately
                if ti == cur_ti:
                    cur_type_iou = trial_iou
                    cur_type_state = _snap(model)

                if trial_iou > best_overall_iou:
                    best_overall_iou = trial_iou
                    best_overall_ti = ti
                    best_overall_state = _snap(model)

            # Decision: compare against noop baseline
            accepted_ti = cur_ti
            accepted_state = None

            if best_overall_ti != cur_ti and best_overall_iou > noop_iou + threshold:
                # Type change: best alternative beats baseline by threshold
                accepted_ti = best_overall_ti
                accepted_state = best_overall_state
                pass_ch += 1; total_changes += 1
            elif cur_type_state is not None and cur_type_iou > noop_iou:
                # Same type, but re-optimized params are better
                accepted_state = cur_type_state

            if accepted_state is not None:
                model.load_state_dict(accepted_state)
            else:
                model.load_state_dict(base)
            indices[li] = li * model.nt + accepted_ti
            _apply_phase2(model, indices)
            base = _snap(model)

        if pass_ch == 0:
            break

    return total_changes


# BOOLEAN SWEEP (changes accepted only if they beat the no-op baseline)
def sweep_booleans(model, target_fn, sdf_fn, eval_grid, eval_gt,
                   num_samples=8192, epochs=200, lr=1e-3,
                   threshold=0.001):
    """Try all 4 boolean ops at each internal node.
    Accept logic: compare against noop_iou baseline."""
    indices = list(model._sel)
    base = _snap(model)
    total_changes = 0

    for ni in range(model.ni):
        cur_op = get_boolean_idx(model, ni)

        # Baseline
        model.load_state_dict(base)
        _apply_phase2(model, indices)
        noop_iou = compute_iou(eval_model(model, eval_grid), eval_gt)

        best_overall_iou = -1.0
        best_overall_op = cur_op
        best_overall_state = None
        cur_op_iou = -1.0
        cur_op_state = None

        for oi in range(4):
            model.load_state_dict(base)
            _apply_phase2(model, indices)
            force_boolean(model, ni, oi)
            optimize_geometry(model, target_fn, sdf_fn, epochs, num_samples, lr)
            force_boolean(model, ni, oi)
            trial_iou = compute_iou(eval_model(model, eval_grid), eval_gt)

            if oi == cur_op:
                cur_op_iou = trial_iou
                cur_op_state = _snap(model)

            if trial_iou > best_overall_iou:
                best_overall_iou = trial_iou
                best_overall_op = oi
                best_overall_state = _snap(model)

        # Decision against noop baseline
        accepted_op = cur_op
        accepted_state = None

        if best_overall_op != cur_op and best_overall_iou > noop_iou + threshold:
            accepted_op = best_overall_op
            accepted_state = best_overall_state
            total_changes += 1
        elif cur_op_state is not None and cur_op_iou > noop_iou:
            accepted_state = cur_op_state

        if accepted_state is not None:
            model.load_state_dict(accepted_state)
        else:
            model.load_state_dict(base)
        _apply_phase2(model, indices)
        base = _snap(model)

    return total_changes


# CEILING SEARCH
def ceiling_search(model, target_fn, sdf_fn, eval_grid, eval_gt,
                   num_samples=8192, epochs_per_combo=500, lr=1e-3):
    """
    Depth 1: exhaustive (5² x 4 = 100 combos).
    Depth 2: greedy subtree sweep (NOT exhaustive over 640k combos).

    Tree indexing (depth 2):
      bparams[1] combines leaves 0,1 (left subtree)
      bparams[2] combines leaves 2,3 (right subtree)
      bparams[0] is root
    """
    assert model.phase == 2 and model._sel is not None

    warm_start_all_candidates(model)
    for ni in range(model.ni):
        force_boolean(model, ni, get_boolean_idx(model, ni))
    warm = _snap(model)
    nt = model.nt

    def try_combo(type_indices, op_indices, base_state):
        model.load_state_dict(base_state)
        sel = [li * nt + ti for li, ti in enumerate(type_indices)]
        _apply_phase2(model, sel)
        for ni, oi in enumerate(op_indices):
            force_boolean(model, ni, oi)
        optimize_geometry(model, target_fn, sdf_fn, epochs_per_combo,
                         num_samples, lr)
        for ni, oi in enumerate(op_indices):
            force_boolean(model, ni, oi)
        iou = compute_iou(eval_model(model, eval_grid), eval_gt)
        return iou, _snap(model), sel

    if model.depth == 1:
        best_iou = -1.0; best_state = None; best_sel = None
        best_types = None; best_ops = None
        for t0 in range(nt):
            for t1 in range(nt):
                for oi in range(4):
                    iou, state, sel = try_combo([t0,t1], [oi], warm)
                    if iou > best_iou:
                        best_iou=iou; best_state=state; best_sel=sel
                        best_types=[t0,t1]; best_ops=[oi]
        model.load_state_dict(best_state)
        _apply_phase2(model, best_sel)
        return {
            'iou': best_iou,
            'types': [model.ptypes[t] for t in best_types],
            'ops': [OP_NAMES[o] for o in best_ops],
            'n_combos': nt*nt*4,
            'search': 'exhaustive',
        }

    elif model.depth == 2:
        cur_types = [model._sel[li] % nt for li in range(model.nl)]
        cur_ops = [get_boolean_idx(model, ni) for ni in range(model.ni)]

        # Step 1: left subtree (leaves 0,1 + bparams[1])
        best_left = None; best_left_iou = -1.0
        for t0 in range(nt):
            for t1 in range(nt):
                for oi in range(4):
                    types = [t0, t1, cur_types[2], cur_types[3]]
                    ops = [cur_ops[0], oi, cur_ops[2]]
                    iou, state, sel = try_combo(types, ops, warm)
                    if iou > best_left_iou:
                        best_left_iou = iou
                        best_left = {'t':[t0,t1],'op':oi,'state':state,'sel':sel}

        # Step 2: right subtree (leaves 2,3 + bparams[2])
        best_right = None; best_right_iou = -1.0
        for t2 in range(nt):
            for t3 in range(nt):
                for oi in range(4):
                    types = [best_left['t'][0], best_left['t'][1], t2, t3]
                    ops = [cur_ops[0], best_left['op'], oi]
                    iou, state, sel = try_combo(types, ops, best_left['state'])
                    if iou > best_right_iou:
                        best_right_iou = iou
                        best_right = {'t':[t2,t3],'op':oi,'state':state,'sel':sel}

        # Step 3: root (bparams[0])
        all_t = best_left['t'] + best_right['t']
        best_root = None; best_root_iou = -1.0
        for oi in range(4):
            ops = [oi, best_left['op'], best_right['op']]
            iou, state, sel = try_combo(all_t, ops, best_right['state'])
            if iou > best_root_iou:
                best_root_iou = iou
                best_root = {'op':oi,'state':state,'sel':sel}

        model.load_state_dict(best_root['state'])
        _apply_phase2(model, best_root['sel'])
        return {
            'iou': best_root_iou,
            'types': [model.ptypes[t] for t in all_t],
            'ops': [OP_NAMES[o] for o in
                    [best_root['op'], best_left['op'], best_right['op']]],
            'n_combos': nt*nt*4 + nt*nt*4 + 4,
            'search': 'greedy_subtree',
        }


# RUN ONE EXPERIMENT (two phase-2 variants)
def run_one(method, tdef, seed, eval_grid, eval_gt, ptypes,
            phase1_epochs=2500, phase2_epochs=5000,
            num_samples=16384, cf_epochs=200, cf_passes=2,
            ceiling_epochs=500):

    torch.manual_seed(seed); np.random.seed(seed)
    t0 = _time.time()

    def target_fn(p): return make_target_occupancy(tdef, p)
    def sdf_fn(p): return make_target_sdf(tdef, p)

    model = CSGTree(depth=tdef['depth'], ptypes=ptypes,
                    scale=0.3, omega=10.0).to(device)

    # Phase 1
    run_phase1(model, target_fn, sdf_fn, phase1_epochs, num_samples)
    trans_iou = compute_iou(eval_model(model, eval_grid), eval_gt)
    types_p1 = model.get_types()
    ops_p1 = model.get_ops()

    # Transition
    model.transition()

    # Intervention
    changes = 0; ceil_info = None

    if method in ('gt_family_only', 'soft_baseline'):
        pass
    elif method == 'cf_standard':
        changes = sweep_types(model, target_fn, sdf_fn, eval_grid, eval_gt,
                             num_samples, cf_epochs, passes=cf_passes)
    elif method == 'cf_joint':
        for _ in range(3):
            bc = sweep_booleans(model, target_fn, sdf_fn, eval_grid, eval_gt,
                               num_samples, cf_epochs)
            tc = sweep_types(model, target_fn, sdf_fn, eval_grid, eval_gt,
                            num_samples, cf_epochs, passes=1)
            changes += bc + tc
            if bc + tc == 0: break
    elif method == 'ceiling_search':
        ceil_info = ceiling_search(
            model, target_fn, sdf_fn, eval_grid, eval_gt,
            num_samples, ceiling_epochs)

    # Post-treatment snapshot (before any phase 2)
    types_post = model.get_types()
    ops_post = model.get_ops()
    post_iou = compute_iou(eval_model(model, eval_grid), eval_gt)
    post_snap = _snap(model)
    post_indices = list(model._sel)

    # Phase 2A: frozen booleans (clean causal test)
    model.load_state_dict(post_snap)
    _apply_phase2(model, post_indices)
    frozen_iou = run_phase2(model, target_fn, sdf_fn, eval_grid, eval_gt,
                            phase2_epochs, num_samples, freeze_bools=True)
    frozen_iou = max(frozen_iou, post_iou)
    types_frozen = model.get_types()
    ops_frozen = model.get_ops()

    # Phase 2B: full optimization (booleans can drift)
    model.load_state_dict(post_snap)
    _apply_phase2(model, post_indices)
    full_iou = run_phase2(model, target_fn, sdf_fn, eval_grid, eval_gt,
                          phase2_epochs, num_samples, freeze_bools=False)
    full_iou = max(full_iou, post_iou)
    types_full = model.get_types()
    ops_full = model.get_ops()

    elapsed = _time.time() - t0

    return {
        'method': method, 'seed': seed,
        'target': tdef['description'], 'depth': tdef['depth'],
        'gt_types': tdef['gt_types'], 'gt_ops': tdef['gt_ops'],
        'ptypes_used': ptypes,

        # Phase 1
        'trans_iou': trans_iou,
        'types_p1': types_p1, 'ops_p1': ops_p1,
        'p1_type_recovery': score_type(types_p1, tdef['gt_types']),
        'p1_op_recovery': score_ops(ops_p1, tdef['gt_ops']),

        # Post-treatment
        'post_iou': post_iou,
        'types_post': types_post, 'ops_post': ops_post,
        'post_type_recovery': score_type(types_post, tdef['gt_types']),
        'post_op_recovery': score_ops(ops_post, tdef['gt_ops']),
        'post_structural': float(
            score_type(types_post, tdef['gt_types']) >= 1-1e-6 and
            score_ops(ops_post, tdef['gt_ops']) >= 1-1e-6),

        # Phase 2 frozen
        'frozen_iou': frozen_iou,
        'types_frozen': types_frozen, 'ops_frozen': ops_frozen,
        'frozen_type_recovery': score_type(types_frozen, tdef['gt_types']),
        'frozen_op_recovery': score_ops(ops_frozen, tdef['gt_ops']),
        'frozen_structural': float(
            score_type(types_frozen, tdef['gt_types']) >= 1-1e-6 and
            score_ops(ops_frozen, tdef['gt_ops']) >= 1-1e-6),

        # Phase 2 full
        'best_iou': full_iou,
        'types_final': types_full, 'ops_final': ops_full,
        'type_recovery': score_type(types_full, tdef['gt_types']),
        'op_recovery': score_ops(ops_full, tdef['gt_ops']),
        'structural_recovery': float(
            score_type(types_full, tdef['gt_types']) >= 1-1e-6 and
            score_ops(ops_full, tdef['gt_ops']) >= 1-1e-6),

        'changes': changes, 'ceil_info': ceil_info, 'elapsed_s': elapsed,
    }


# ANALYSIS
def _correlation_table(results, methods, tr_key, or_key, label):
    """Print P(type_correct | ops_correct) vs P(type_correct | ops_wrong)."""
    print(f"\n  {label}:")
    print(f"  {'Method':<20s} │ {'N':>3s} │  O✓ {'N':>3s} {'P(T|O✓)':>8s} │  "
          f"O✗ {'N':>3s} {'P(T|O✗)':>8s} │ {'Δ':>7s}")
    print(f"  {'─'*74}")
    for m in methods:
        runs = [r for r in results if r['method'] == m]
        if not runs: continue
        oc = [r for r in runs if r[or_key] >= 1-1e-6]
        oi = [r for r in runs if r[or_key] < 1-1e-6]
        pt_oc = np.mean([r[tr_key] for r in oc]) if oc else float('nan')
        pt_oi = np.mean([r[tr_key] for r in oi]) if oi else float('nan')
        delta = pt_oc - pt_oi if oc and oi else float('nan')
        def fmt(v): return f"{v:.0%}" if not np.isnan(v) else "   -"
        print(f"  {m:<20s} │ {len(runs):>3d} │     {len(oc):>3d} {fmt(pt_oc):>8s} │     "
              f"{len(oi):>3d} {fmt(pt_oi):>8s} │ {fmt(delta):>7s}")


def analyze_correlation(results, methods):
    print(f"\n{'='*80}")
    print("BOOLEAN-CONDITIONED TYPE RECOVERY")
    print(f"{'='*80}")
    print(f"  Key question: when booleans are correct, are types also correct?")
    print(f"  If P(T|O✓) >> P(T|O✗), remaining failures are upstream boolean failures.")

    _correlation_table(results, methods,
        'post_type_recovery', 'post_op_recovery',
        'POST-TREATMENT (intervention effect, before any phase 2)')

    _correlation_table(results, methods,
        'frozen_type_recovery', 'frozen_op_recovery',
        'AFTER FROZEN PHASE 2 (geometry refined, booleans locked)')

    _correlation_table(results, methods,
        'type_recovery', 'op_recovery',
        'AFTER FULL PHASE 2 (booleans may have drifted)')

    # Per-run detail for cf_standard and cf_joint
    for m in ['cf_standard', 'cf_joint']:
        runs = [r for r in results if r['method'] == m]
        if not runs: continue
        print(f"\n  Per-run detail ({m}):")
        nops_max = max(len(r['gt_ops']) for r in runs)
        print(f"  {'Target':<19s} {'Sd':>4s} │ P1 {'OR':>3s}/{'TR':>3s} │ "
              f"Post {'OR':>3s}/{'TR':>3s} │ Frzn {'OR':>3s}/{'TR':>3s} │ "
              f"Full {'OR':>3s}/{'TR':>3s} {'IoU':>7s} │ Types")
        print(f"  {'─'*100}")
        for r in sorted(runs, key=lambda x: (x['target'], x['seed'])):
            print(f"  {r['target']:<19s} {r['seed']:>4d} │    "
                  f"{r['p1_op_recovery']:.0%}/{r['p1_type_recovery']:.0%}  │ "
                  f"     {r['post_op_recovery']:.0%}/{r['post_type_recovery']:.0%}  │ "
                  f"     {r['frozen_op_recovery']:.0%}/{r['frozen_type_recovery']:.0%}  │ "
                  f"     {r['op_recovery']:.0%}/{r['type_recovery']:.0%} "
                  f"{r['best_iou']:>7.4f} │ "
                  f"{','.join(r['types_final'])}")


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789])
    parser.add_argument('--phase1', type=int, default=2500)
    parser.add_argument('--phase2', type=int, default=5000)
    parser.add_argument('--cf-epochs', type=int, default=200)
    parser.add_argument('--cf-passes', type=int, default=2)
    parser.add_argument('--ceiling-epochs', type=int, default=500)
    parser.add_argument('--samples', type=int, default=16384)
    parser.add_argument('--eval-res', type=int, default=64)
    parser.add_argument('--skip-ceiling', action='store_true')
    args = parser.parse_args()

    res = args.eval_res
    idx = torch.linspace(-1,1,res,device=device)
    x,y,z = torch.meshgrid(idx,idx,idx,indexing='ij')
    eval_grid = torch.stack([x.flatten(),y.flatten(),z.flatten()],1)

    METHODS = ['gt_family_only', 'soft_baseline', 'cf_standard', 'cf_joint']
    if not args.skip_ceiling:
        METHODS.append('ceiling_search')

    total = sum(len(METHODS)*len(args.seeds) for _ in TARGETS)

    print(f"\n{'='*80}")
    print("TYPE RECOVERY EXPERIMENT")
    print(f"{'='*80}")
    print(f"  Targets:  {list(TARGETS.keys())}")
    print(f"  Methods:  {METHODS}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Phase 1: {args.phase1}, Phase 2: {args.phase2} (×2: frozen + full)")
    print(f"  CF: {args.cf_epochs} ep/trial, {args.cf_passes} passes")
    if not args.skip_ceiling:
        print(f"  Ceiling: {args.ceiling_epochs} ep/combo "
              f"(exhaustive d1, greedy subtree d2)")
    print(f"  Total runs: {total}")
    print()

    all_results = []
    run_idx = 0

    for tname, tdef in TARGETS.items():
        eval_gt = make_target_occupancy(tdef, eval_grid).detach()
        fill = eval_gt.mean().item()
        d = tdef['depth']
        gt_unique = list(sorted(set(tdef['gt_types'])))

        print(f"\n--- {tdef['description']} ({fill*100:.1f}%, d={d}, {2**d}L) ---")
        print(f"    GT: types={tdef['gt_types']}, ops={tdef['gt_ops']}")

        for method in METHODS:
            for seed in args.seeds:
                run_idx += 1
                ptypes = gt_unique if method == 'gt_family_only' else PRIM_TYPES
                lib = 'GT' if method == 'gt_family_only' else 'ALL5'
                print(f"  [{run_idx}/{total}] {method} s={seed} [{lib}]",
                      end=" ", flush=True)

                try:
                    r = run_one(
                        method, tdef, seed, eval_grid, eval_gt, ptypes,
                        phase1_epochs=args.phase1, phase2_epochs=args.phase2,
                        num_samples=args.samples,
                        cf_epochs=args.cf_epochs, cf_passes=args.cf_passes,
                        ceiling_epochs=args.ceiling_epochs)
                    r['target_name'] = tname
                    all_results.append(r)

                    ci = ""
                    if r['ceil_info']:
                        ci = f" {r['ceil_info']['search']}({r['ceil_info']['n_combos']})"
                    print(f"→ frz={r['frozen_iou']:.4f} full={r['best_iou']:.4f} "
                          f"T={r['type_recovery']:.0%} O={r['op_recovery']:.0%} "
                          f"ch={r['changes']} {r['elapsed_s']:.0f}s{ci}")

                except Exception as e:
                    print(f"→ FAILED: {e}")
                    import traceback; traceback.print_exc()

    # METHOD COMPARISON
    print(f"\n{'='*80}")
    print("METHOD COMPARISON (overall)")
    print(f"{'='*80}")
    print(f"  {'Method':<20s} │ {'Frz IoU':>8s} {'Full IoU':>9s} │ "
          f"{'Post TR':>7s} {'Frz TR':>7s} {'Full TR':>7s} │ "
          f"{'Post OR':>7s} {'Frz OR':>7s} {'Full OR':>7s} │ {'Time':>5s}")
    print(f"  {'─'*100}")

    for m in METHODS:
        runs = [r for r in all_results if r['method']==m]
        if not runs: continue
        def av(k): return np.mean([r[k] for r in runs])
        print(f"  {m:<20s} │ {av('frozen_iou'):>8.4f} {av('best_iou'):>9.4f} │ "
              f"{av('post_type_recovery'):>6.0%} {av('frozen_type_recovery'):>7.0%} "
              f"{av('type_recovery'):>7.0%} │ "
              f"{av('post_op_recovery'):>6.0%} {av('frozen_op_recovery'):>7.0%} "
              f"{av('op_recovery'):>7.0%} │ {av('elapsed_s'):>4.0f}s")

    # Per target
    print(f"\n{'='*80}")
    print("METHOD COMPARISON (per target, frozen phase 2)")
    print(f"{'='*80}")

    for tname, tdef in TARGETS.items():
        print(f"\n  {tdef['description']} (GT: {tdef['gt_types']}, {tdef['gt_ops']})")
        print(f"  {'Method':<20s} {'IoU':>7s} {'TR':>5s} {'OR':>5s} {'SR':>5s}  Types")
        print(f"  {'-'*70}")
        for m in METHODS:
            runs = [r for r in all_results
                    if r['target_name']==tname and r['method']==m]
            if not runs: continue
            avg_iou = np.mean([r['frozen_iou'] for r in runs])
            avg_tr = np.mean([r['frozen_type_recovery'] for r in runs])
            avg_or = np.mean([r['frozen_op_recovery'] for r in runs])
            avg_sr = np.mean([r['frozen_structural'] for r in runs])
            all_types = []
            for r in runs: all_types.extend(r['types_frozen'])
            tc = Counter(all_types)
            ts = " ".join(f"{t[:3]}={c}" for t,c in sorted(tc.items()))
            tag = " ← control" if m == 'gt_family_only' else ""
            print(f"  {m:<20s} {avg_iou:>7.4f} {avg_tr:>4.0%} "
                  f"{avg_or:>5.0%} {avg_sr:>4.0%}  [{ts}]{tag}")

    # CORRELATION
    analyze_correlation(all_results, METHODS)

    # BOOLEAN DRIFT: frozen vs full phase 2
    print(f"\n{'='*80}")
    print("BOOLEAN DRIFT ANALYSIS (frozen vs full phase 2)")
    print(f"{'='*80}")
    print(f"  Does unfreezing booleans in phase 2 help or hurt?")
    print()
    print(f"  {'Method':<20s} │ {'Frz IoU':>8s} {'Full IoU':>9s} {'Δ IoU':>7s} │ "
          f"{'Frz TR':>7s} {'Full TR':>7s} │ {'Frz OR':>7s} {'Full OR':>7s}")
    print(f"  {'─'*85}")
    for m in METHODS:
        runs = [r for r in all_results if r['method']==m]
        if not runs: continue
        def av(k): return np.mean([r[k] for r in runs])
        d_iou = av('best_iou') - av('frozen_iou')
        print(f"  {m:<20s} │ {av('frozen_iou'):>8.4f} {av('best_iou'):>9.4f} "
              f"{d_iou:>+7.4f} │ "
              f"{av('frozen_type_recovery'):>6.0%} {av('type_recovery'):>7.0%} │ "
              f"{av('frozen_op_recovery'):>6.0%} {av('op_recovery'):>7.0%}")

    # TREATMENT EFFECT
    print(f"\n{'='*80}")
    print("TREATMENT EFFECT (phase 1 → post-treatment → frozen phase 2)")
    print(f"{'='*80}")
    print(f"  {'Method':<20s} │ P1 {'TR':>3s}/{'OR':>3s} │ Post {'TR':>3s}/{'OR':>3s} │ "
          f"Frozen {'TR':>3s}/{'OR':>3s} │ P1→Post {'ΔTR':>5s} │ Post→Frz {'ΔTR':>5s}")
    print(f"  {'─'*90}")
    for m in METHODS:
        runs = [r for r in all_results if r['method']==m]
        if not runs: continue
        def av(k): return np.mean([r[k] for r in runs])
        d1 = av('post_type_recovery') - av('p1_type_recovery')
        d2 = av('frozen_type_recovery') - av('post_type_recovery')
        print(f"  {m:<20s} │    {av('p1_type_recovery'):.0%}/{av('p1_op_recovery'):.0%}  │ "
              f"     {av('post_type_recovery'):.0%}/{av('post_op_recovery'):.0%}  │ "
              f"      {av('frozen_type_recovery'):.0%}/{av('frozen_op_recovery'):.0%}  │ "
              f"         {d1:>+4.0%} │          {d2:>+4.0%}")

    # CEILING COMPARISON
    ceil_runs = [r for r in all_results if r['method']=='ceiling_search']
    if ceil_runs:
        print(f"\n{'='*80}")
        print("CEILING vs CF_JOINT (frozen phase 2)")
        print(f"{'='*80}")
        print(f"  {'Target':<19s} {'Search':>14s} │ {'Ceil IoU':>9s} {'Jnt IoU':>8s} "
              f"{'Gap':>6s} │ {'Ceil TR':>8s} {'Jnt TR':>7s} │ {'Ceil SR':>8s} {'Jnt SR':>7s}")
        print(f"  {'─'*95}")

        for tname in TARGETS:
            ex = [r for r in ceil_runs if r['target_name']==tname]
            jt = [r for r in all_results
                  if r['target_name']==tname and r['method']=='cf_joint']
            if not ex or not jt: continue
            search = ex[0]['ceil_info']['search'] if ex[0]['ceil_info'] else '?'
            ei = np.mean([r['frozen_iou'] for r in ex])
            ji = np.mean([r['frozen_iou'] for r in jt])
            et = np.mean([r['frozen_type_recovery'] for r in ex])
            jt_ = np.mean([r['frozen_type_recovery'] for r in jt])
            es = np.mean([r['frozen_structural'] for r in ex])
            js = np.mean([r['frozen_structural'] for r in jt])
            print(f"  {TARGETS[tname]['description']:<19s} {search:>14s} │ "
                  f"{ei:>9.4f} {ji:>8.4f} {ei-ji:>6.4f} │ "
                  f"{et:>7.0%} {jt_:>7.0%} │ {es:>7.0%} {js:>7.0%}")

    # FULL TABLE
    print(f"\n{'='*80}")
    print("FULL RESULTS (frozen phase 2)")
    print(f"{'='*80}")
    print(f"  {'Target':<17s} {'Method':<15s} {'Sd':>4s} {'IoU':>7s} "
          f"{'TR':>4s} {'OR':>4s} {'S':>2s} {'Ch':>3s} {'Types':>28s}  Ops")
    print(f"  {'-'*105}")

    for tname in TARGETS:
        tdef = TARGETS[tname]; nops = len(tdef['gt_ops'])
        for r in sorted(
            [r for r in all_results if r['target_name']==tname],
            key=lambda x: (METHODS.index(x['method']) if x['method'] in METHODS else 99,
                          x['seed'])):
            s = '✓' if r['frozen_structural'] > 0.5 else '✗'
            print(f"  {tdef['description']:<17s} {r['method']:<15s} "
                  f"{r['seed']:>4d} {r['frozen_iou']:>7.4f} "
                  f"{r['frozen_type_recovery']:.0%} {r['frozen_op_recovery']:.0%}  "
                  f"{s} {r['changes']:>3d} "
                  f"{','.join(r['types_frozen']):>28s}  "
                  f"{','.join(r['ops_frozen'][:nops])}")

    # SAVE
    def convert(obj):
        if isinstance(obj, (np.floating,np.float64,np.float32)): return float(obj)
        if isinstance(obj, (np.integer,np.int64,np.int32)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open('type_recovery_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved to type_recovery_results.json")


if __name__ == "__main__":
    main()