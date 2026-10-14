#!/usr/bin/env python3
"""
Budget sweep of the inverse CSG methods (Chapter 3, 432 runs).

Imports the solver core from siggraph_csg_benchmark_v3.py (model classes,
primitive SDFs, initialization, training-point cache, mesh repair/occupancy,
exporters) and adds:
  * one-hot vs multi-hot leaf evaluation;
  * available/active physical primitive accounting;
  * optional legacy methods when their modules are present;
  * zipped/Cartesian budget sweeps;
  * plots of best IoU vs epoch and budget curves.

The solver core is used unchanged, so every method in the sweep shares the same
training and evaluation code.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Solver core (same directory).
try:
    import siggraph_csg_benchmark_v3 as v3
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import siggraph_csg_benchmark_v3.py. Place this script next "
        "to the working v3 benchmark file."
    ) from e

N_TYPES = v3.N_TYPES
TYPE_NAMES = v3.TYPE_NAMES
EPS = v3.EPS

# Utilities

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(s: Optional[str], default: int) -> List[int]:
    if s is None or str(s).strip() == "":
        return [int(default)]
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def expand_budget_lists(base: List[int], bc: List[int], groups: List[int], grid: bool) -> List[Tuple[int, int, int]]:
    if grid:
        return [(b, c, g) for b, c, g in itertools.product(base, bc, groups)]
    n = max(len(base), len(bc), len(groups))
    def ex(xs: List[int]) -> List[int]:
        if len(xs) == n:
            return xs
        if len(xs) == 1:
            return xs * n
        raise ValueError("Budget lists must have equal length unless one is singleton. Use --budget-grid for Cartesian product.")
    return list(zip(ex(base), ex(bc), ex(groups)))


def method_label(method: str) -> str:
    return method.replace("_product_vol", "_prod").replace("_multihot", "_mh").replace("_onehot", "_oh")


def stabilize_structure_init(model, args) -> None:
    """Stable Boolean-structure initialization for small-budget sweeps.

    The v3 DNF/Stump initializes selected literal/intersection masks at
    probability 0.35. At very small budgets this makes selected literals behave
    like a large constant outside the primitive, so the model can predict the
    universe and then drive all union gates below the hard threshold.

    For the budget sweep, selected Boolean connections are initialized closer to
    hard literals while keeping the same v3 model, losses, and training loop.
    Pass --strict-v3-init to disable this and reproduce the v3 initializer.
    """
    if getattr(args, "strict_v3_init", False):
        return
    lit = v3.logit(float(args.init_lit_prob))
    off = float(args.init_off_logit)
    union = v3.logit(float(args.init_union_prob))
    comp = v3.logit(float(args.init_complement_prob))
    with torch.no_grad():
        if isinstance(model, v3.NaryDNFModel):
            model.m_pos.fill_(off)
            model.m_neg.fill_(off - 1.0)
            model.act.fill_(union)
            for i in range(model.N):
                model.m_pos[i, i % model.C] = lit
        elif isinstance(model, v3.ProperCSGStumpModel):
            model.comp.fill_(comp)
            model.inter.fill_(off)
            model.union.fill_(union)
            for i in range(model.N):
                model.inter[i, i % model.C] = lit
        elif isinstance(model, v3.PaperStumpModel):
            model.wc.fill_(comp)
            model.wi.fill_(off)
            model.wu.fill_(union)
            for i in range(model.N):
                model.wi[i, i % model.C] = lit
        elif isinstance(model, v3.BinaryTreeModel):
            model.gates.fill_(v3.logit(float(args.init_tree_gate_prob)))
        elif isinstance(model, v3.ProductUnionModel):
            model.gates.fill_(v3.logit(float(args.init_union_leaf_prob)))


# Legacy original methods (optional)

class LegacyAdapter(v3.BaseCSGModel):
    def __init__(self, legacy_model, init_fn, use_temperature: bool):
        super().__init__()
        self.legacy = legacy_model
        self.init_fn = init_fn
        self.use_temperature = use_temperature

    def to(self, *args, **kwargs):
        self.legacy.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        return self.legacy.parameters()

    def forward(self, pts: torch.Tensor, beta: float = 1.0):
        if self.use_temperature:
            return self.legacy(pts, temperature=beta)
        # Older discrete models accept a temperature argument but ignore it.
        try:
            return self.legacy(pts, temperature=beta)
        except TypeError:
            return self.legacy(pts)

    def forward_modes(self, pts: torch.Tensor, beta: float) -> Dict[str, torch.Tensor]:
        soft = self.forward(pts, beta)
        if hasattr(self.legacy, "forward_hard_dnf"):
            hard = self.legacy.forward_hard_dnf(pts)
        else:
            hard = soft
        return {"soft": soft, "hardleaf_softbool": hard, "hard": hard}

    def structure_loss(self, warm: float, beta: float, args) -> torch.Tensor:
        # Same call signature as train_with_logging in the older scripts.
        try:
            return self.legacy.structure_loss(
                ls=args.lambda_sparse * warm,
                lb=args.lambda_binary * warm,
                lc=args.lambda_conflict * warm,
                lgs=args.lambda_group * warm,
                temperature=beta,
            )
        except TypeError:
            try:
                return self.legacy.structure_loss(
                    args.lambda_sparse * warm,
                    args.lambda_binary * warm,
                    args.lambda_conflict * warm,
                    args.lambda_group * warm,
                    beta,
                )
            except TypeError:
                return torch.zeros((), device=next(self.legacy.parameters()).device)

    def stats(self, beta: float, threshold: float = 0.35) -> Dict[str, object]:
        if hasattr(self.legacy, "get_stats"):
            s = self.legacy.get_stats()
            return {
                "prims_used": int(s.get("prims_used", 0)),
                "groups_active": int(s.get("groups_active", 0)),
                "type_entropy": float(s.get("type_entropy", 0.0)),
            }
        return {"prims_used": 0, "groups_active": 0, "type_entropy": 0.0}

    def init_from_mesh(self, mesh):
        if self.init_fn is not None:
            self.init_fn(self.legacy, mesh)


def try_make_legacy(method: str, args, device: torch.device) -> Optional[LegacyAdapter]:
    try:
        nb = importlib.import_module("nary_benchmark_v7")
        fb = importlib.import_module("full_benchmark_with_levels")
    except Exception:
        return None
    try:
        if method == "legacy_nary_discrete":
            model = nb.BaselineCSG(prims_per_type=args.baseline_prims, num_groups=args.groups).to(device)
            def init(m, ms):
                nb.surface_init(m, ms); nb.spatial_group_init(m, ms)
            return LegacyAdapter(model, init, use_temperature=False)
        if method == "legacy_nary_bc_vol":
            model = fb.NaryBcVolCSG(num_primitives=args.bc_prims, num_groups=args.groups, omega=args.omega).to(device)
            def init(m, ms):
                nb.surface_init(m, ms); nb.spatial_group_init(m, ms)
            return LegacyAdapter(model, init, use_temperature=True)
        if method == "legacy_tree_discrete":
            model = fb.TreeDiscreteCSG(prims_per_type=args.baseline_prims, omega=args.omega).to(device)
            def init(m, ms):
                fb.tree_surface_init(m, ms); fb.tree_spatial_init(m, ms)
            return LegacyAdapter(model, init, use_temperature=False)
        if method == "legacy_tree_bc_vol":
            model = fb.TreeBcVolCSG(num_primitives=args.bc_prims, omega=args.omega).to(device)
            def init(m, ms):
                fb.tree_surface_init(m, ms); fb.tree_spatial_init(m, ms)
            return LegacyAdapter(model, init, use_temperature=True)
    except Exception as e:
        print(f"warning: constructing {method} failed: {e}")
        return None
    return None

# Method construction on top of v3 classes

def method_leaf_mode(method: str) -> str:
    if "multihot" in method:
        return "multihot"
    if "onehot" in method:
        return "onehot"
    if method in {"nary_bc_vol", "tree_bc_vol", "stump_bc_vol", "paper_stump_bc_vol"}:
        return "onehot"
    return "discrete"


def method_available_capacity(method: str, args) -> int:
    leaf_mode = method_leaf_mode(method)
    if "legacy" in method:
        # Original BC methods had learned primitive banks; count by BC leaves for the x-axis,
        # but note legacy hard semantics are reported separately.
        if "bc" in method:
            return int(args.bc_prims)
        return int(args.baseline_prims * N_TYPES)
    if leaf_mode == "multihot":
        return int(args.bc_prims * N_TYPES)
    if leaf_mode == "onehot":
        return int(args.bc_prims)
    return int(args.baseline_prims * N_TYPES)


def make_model(method: str, args, device: torch.device) -> v3.BaseCSGModel:
    aliases = {
        "nary_onehot_product_vol": "nary_bc_vol",
        "nary_onehot_sum_vol": "nary_onehot_sum_vol",
        "nary_multihot_product_vol": "nary_bc_multihot",
        "nary_multihot_sum_vol": "nary_multihot_sum_vol",
        "tree_onehot_product_vol": "tree_bc_vol",
        "tree_multihot_product_vol": "tree_multihot_product_vol",
        "stump_onehot_product_vol": "stump_bc_vol",
        "stump_multihot_product_vol": "stump_multihot_product_vol",
        "paper_stump_onehot_product_vol": "paper_stump_bc_vol",
        "paper_stump_multihot_product_vol": "paper_stump_multihot_product_vol",
        "union_onehot_product_vol": "union_product_bc",
        "union_onehot_sum_vol": "union_sum_bc",
        "union_multihot_product_vol": "union_multihot_product_vol",
        "union_multihot_sum_vol": "union_multihot_sum_vol",
    }
    m = aliases.get(method, method)
    if m.startswith("legacy_"):
        leg = try_make_legacy(m, args, device)
        if leg is not None:
            return leg
        if not args.allow_legacy_fallback:
            raise RuntimeError(
                f"Requested {m}, but legacy modules nary_benchmark_v7.py/full_benchmark_with_levels.py were not found. "
                "Place them next to this script or pass --allow-legacy-fallback for approximate smoke tests."
            )
        print(f"warning: {m} unavailable; using closest v3 fallback")
        m = {"legacy_nary_discrete": "nary_discrete", "legacy_nary_bc_vol": "nary_bc_vol",
             "legacy_tree_discrete": "tree_discrete", "legacy_tree_bc_vol": "tree_bc_vol"}[m]

    # Direct v3 methods.
    if m in {"nary_discrete", "nary_bc_vol", "tree_discrete", "tree_bc_vol", "stump_discrete", "stump_bc_vol",
             "paper_stump_discrete", "paper_stump_bc_vol", "union_product_bc", "union_sum_bc", "nary_bc_multihot"}:
        return v3.make_model(m, args)

    # New controlled variants.
    volume_norm = args.volume_norm and ("_novol" not in method)
    agg = "sum" if "_sum_" in method or method.endswith("_sum_vol") or "_sum" in method else "product"
    pool_mode = "sine_sigmoid_multi" if "multihot" in method else args.type_selector
    if method.startswith("nary_"):
        return v3.NaryDNFModel(args.bc_prims, args.groups, pool_mode, agg, volume_norm, args.omega)
    if method.startswith("tree_"):
        return v3.BinaryTreeModel(args.bc_prims, pool_mode, agg, volume_norm, args.omega, args.tree_ops)
    if method.startswith("stump_"):
        return v3.ProperCSGStumpModel(args.bc_prims, args.groups, pool_mode, agg, volume_norm, args.omega)
    if method.startswith("paper_stump_"):
        return v3.PaperStumpModel(args.bc_prims, args.groups, pool_mode, agg, volume_norm, args.omega, args.stump_psi)
    if method.startswith("union_"):
        return v3.ProductUnionModel(args.bc_prims, pool_mode, agg, volume_norm, args.omega, sum_mode=(agg == "sum"))
    raise ValueError(f"unknown method {method}")

# Multi-hot leaf evaluation and active counts

def type_multihot_mask(pool: v3.PrimitivePool, beta: float, thr: float) -> torch.Tensor:
    if pool.cfg.mode == "fixed_cycle":
        return pool.type_weights(beta, hard=True)
    if pool.cfg.mode == "sine_sigmoid_multi":
        logits = beta * torch.sin(pool.cfg.omega * pool.type_logits)
        return (torch.sigmoid(logits) > thr).float()
    # For softmax selectors, a multi-hot hard extraction at threshold >0.5 normally
    # collapses to one type, since the method was not trained for compound leaves.
    w = pool.type_weights(beta, hard=False)
    return (w > thr).float()


def leaf_occs_from_stack(pool: v3.PrimitivePool, occ_stack: torch.Tensor, beta: float, mode: str, thr: float) -> torch.Tensor:
    if mode == "soft":
        w = pool.type_weights(beta, hard=False)
        if pool.cfg.type_agg == "sum":
            return (occ_stack * w.unsqueeze(1)).sum(-1).clamp(0, 1)
        return 1.0 - torch.prod((1.0 - occ_stack * w.unsqueeze(1)).clamp_min(1e-6), dim=-1)
    if mode == "onehot":
        w = pool.type_weights(beta, hard=True)
        return (occ_stack * w.unsqueeze(1)).sum(-1)
    if mode == "multihot":
        mask = type_multihot_mask(pool, beta, thr)
        # Guarantee at least one type per leaf: if all thresholds are false, keep argmax.
        none = (mask.sum(-1, keepdim=True) <= 0)
        if none.any():
            arg = torch.argmax(pool.type_weights(beta, hard=False), dim=-1, keepdim=True)
            one = torch.zeros_like(mask); one.scatter_(1, arg, 1.0)
            mask = torch.where(none, one, mask)
        if pool.cfg.type_agg == "sum":
            return (occ_stack * mask.unsqueeze(1)).sum(-1).clamp(0, 1)
        return 1.0 - torch.prod((1.0 - occ_stack * mask.unsqueeze(1)).clamp_min(1e-6), dim=-1)
    raise ValueError(mode)


def paper_stump_exact_hard(model: v3.PaperStumpModel, q: torch.Tensor, thr: float) -> torch.Tensor:
    wc = (torch.sigmoid(model.wc) > thr).float()
    wi = (torch.sigmoid(model.wi) > thr).float()
    wu = (torch.sigmoid(model.wu) > thr).float()
    Fleaf = wc.reshape(-1, 1) * (1.0 - q) + (1.0 - wc.reshape(-1, 1)) * q
    X = wi.unsqueeze(1) * Fleaf.unsqueeze(-1) + (1.0 - wi.unsqueeze(1)) * 1.0
    S = torch.min(X, dim=0).values
    Y = wu.reshape(1, -1) * S
    return torch.max(Y, dim=-1).values.clamp(0, 1)


def combine_from_leaves(model: v3.BaseCSGModel, q: torch.Tensor, mode: str, thr: float) -> torch.Tensor:
    hard = mode == "hard"
    if isinstance(model, v3.NaryDNFModel):
        return v3.dnf_forward(q, model.m_pos, model.m_neg, model.act, hard_bool=hard, thr=thr)
    if isinstance(model, v3.ProperCSGStumpModel):
        return model._stump_forward(q, hard_bool=hard, thr=thr)
    if isinstance(model, v3.PaperStumpModel):
        if hard:
            return paper_stump_exact_hard(model, q, thr)
        return model._stump(q, hard_bool=False)
    if isinstance(model, v3.ProductUnionModel):
        return model._combine(q, hard_bool=hard)
    if isinstance(model, v3.BinaryTreeModel):
        return model._tree_reduce(q, hard_bool=hard)
    # Legacy adapter or unknown.
    m = model.forward_modes(q, 1.0)  # not used normally
    return m.get("hard", m["soft"])


def forward_extended(model: v3.BaseCSGModel, pts: torch.Tensor, beta: float, type_thr: float) -> Dict[str, torch.Tensor]:
    if isinstance(model, LegacyAdapter):
        m = model.forward_modes(pts, beta)
        hard = m.get("hard", m["soft"])
        return {"soft": m["soft"], "hard_onehot": hard, "hard_multihot": hard,
                "hardleaf_onehot_softbool": m.get("hardleaf_softbool", hard), "hardleaf_multihot_softbool": m.get("hardleaf_softbool", hard)}
    if not hasattr(model, "pool"):
        m = model.forward_modes(pts, beta)
        hard = m.get("hard", m["soft"])
        return {"soft": m["soft"], "hard_onehot": hard, "hard_multihot": hard,
                "hardleaf_onehot_softbool": m.get("hardleaf_softbool", hard), "hardleaf_multihot_softbool": m.get("hardleaf_softbool", hard)}
    pool = getattr(model, "pool")
    occ_stack = pool.occs_by_type(pts)
    q_soft = leaf_occs_from_stack(pool, occ_stack, beta, "soft", type_thr)
    q_one = leaf_occs_from_stack(pool, occ_stack, beta, "onehot", type_thr)
    q_multi = leaf_occs_from_stack(pool, occ_stack, beta, "multihot", type_thr)
    return {
        "soft": combine_from_leaves(model, q_soft, "soft", type_thr),
        "hardleaf_onehot_softbool": combine_from_leaves(model, q_one, "soft", type_thr),
        "hard_onehot": combine_from_leaves(model, q_one, "hard", type_thr),
        "hardleaf_multihot_softbool": combine_from_leaves(model, q_multi, "soft", type_thr),
        "hard_multihot": combine_from_leaves(model, q_multi, "hard", type_thr),
    }


def batched_forward_extended(model: v3.BaseCSGModel, grid: torch.Tensor, beta: float, chunk: int, type_thr: float) -> Dict[str, torch.Tensor]:
    outs: Dict[str, List[torch.Tensor]] = {"soft": [], "hardleaf_onehot_softbool": [], "hard_onehot": [], "hardleaf_multihot_softbool": [], "hard_multihot": []}
    with torch.no_grad():
        for b in torch.split(grid, chunk):
            m = forward_extended(model, b, beta, type_thr)
            for k in outs:
                outs[k].append(m[k].detach())
    return {k: torch.cat(v, dim=0) for k, v in outs.items()}


def compute_metrics_extended(model: v3.BaseCSGModel, grid: torch.Tensor, gt: torch.Tensor, beta: float, chunk: int, type_thr: float) -> Dict[str, float]:
    m = batched_forward_extended(model, grid, beta, chunk, type_thr)
    out: Dict[str, float] = {}
    for k, pred in m.items():
        out[f"{k}_iou"] = v3.compute_iou(pred, gt)
        out[f"{k}_mse"] = float(((pred - gt) ** 2).mean().item())
        out[f"{k}_fscore"] = v3.compute_fscore(pred, gt)
    out["hardening_gap_onehot"] = out["soft_iou"] - out["hard_onehot_iou"]
    out["hardening_gap_multihot"] = out["soft_iou"] - out["hard_multihot_iou"]
    return out


def active_counts(model: v3.BaseCSGModel, method: str, beta: float, thr: float, args) -> Dict[str, object]:
    available = method_available_capacity(method, args)
    if isinstance(model, LegacyAdapter):
        s = model.stats(beta, thr)
        return {"available_physical_primitives": available, "active_physical_primitives": int(s.get("prims_used", 0)),
                "active_leaves": int(s.get("prims_used", 0)), "groups_active": int(s.get("groups_active", 0)),
                "active_physical_primitives_onehot": int(s.get("prims_used", 0)), "active_physical_primitives_multihot": int(s.get("prims_used", 0)),
                "capacity_leaf_type_pairs_onehot": available, "capacity_leaf_type_pairs_multihot": available,
                "type_entropy": float(s.get("type_entropy", 0.0))}
    if not hasattr(model, "pool"):
        return {"available_physical_primitives": available, "active_physical_primitives": 0, "active_leaves": 0,
                "groups_active": 0, "active_physical_primitives_onehot": 0, "active_physical_primitives_multihot": 0,
                "capacity_leaf_type_pairs_onehot": available, "capacity_leaf_type_pairs_multihot": available,
                "type_entropy": 0.0}
    pool: v3.PrimitivePool = getattr(model, "pool")
    base_stats = model.stats(beta, threshold=thr) if hasattr(model, "stats") else {}
    used_leaves = int(base_stats.get("prims_used", 0))
    # Count leaf/type pairs only for leaves used by Boolean structure.
    if pool.cfg.mode == "fixed_cycle":
        type_pairs_multi = used_leaves
    else:
        mask = type_multihot_mask(pool, beta, thr)
        if isinstance(model, v3.NaryDNFModel):
            lit_used = ((torch.sigmoid(model.m_pos) > thr) | (torch.sigmoid(model.m_neg) > thr)).float()
            leaf_used = (lit_used.max(dim=1).values > 0).float()
        elif isinstance(model, v3.ProperCSGStumpModel):
            leaf_used = ((torch.sigmoid(model.inter) > thr).max(dim=1).values > 0).float()
        elif isinstance(model, v3.PaperStumpModel):
            leaf_used = ((torch.sigmoid(model.wi) > thr).max(dim=1).values > 0).float()
        elif isinstance(model, v3.ProductUnionModel) or isinstance(model, v3.BinaryTreeModel):
            leaf_used = (torch.sigmoid(model.gates) > thr).float()
        else:
            leaf_used = torch.ones(pool.cfg.n_leaves, device=pool.centers.device)
        type_pairs_multi = int((mask * leaf_used.reshape(-1, 1)).sum().detach().cpu().item())
    if method_leaf_mode(method) == "multihot":
        active = type_pairs_multi
    else:
        active = used_leaves
    return {
        "available_physical_primitives": available,
        "active_physical_primitives": active,
        "active_leaves": used_leaves,
        "groups_active": int(base_stats.get("groups_active", 0)),
        "active_physical_primitives_onehot": used_leaves,
        "active_physical_primitives_multihot": type_pairs_multi,
        "capacity_leaf_type_pairs_onehot": int(getattr(pool.cfg, "n_leaves", args.bc_prims)),
        "capacity_leaf_type_pairs_multihot": int(getattr(pool.cfg, "n_leaves", args.bc_prims) * (1 if pool.cfg.mode == "fixed_cycle" else N_TYPES)),
        "type_entropy": float(base_stats.get("type_entropy", 0.0)),
    }

# Training

@dataclass
class RunResult:
    final_metrics: Dict[str, float]
    best_soft: float
    best_hard_onehot: float
    best_hard_multihot: float
    stats: Dict[str, object]
    wall_s: float


def train_one_run(mesh_sdf, method: str, seed: int, out_dir: str, args, device: torch.device) -> RunResult:
    os.makedirs(out_dir, exist_ok=True)
    seed_all(seed)
    run_name = os.path.basename(out_dir)
    model = make_model(method, args, device).to(device)
    model.init_from_mesh(mesh_sdf)
    stabilize_structure_init(model, args)

    cache_P, cache_Y = v3.make_training_cache(mesh_sdf, args.cache_points, device, balanced=not args.no_balance)
    eval_grid = v3.sample_grid(args.resolution, device)
    eval_gt = torch.tensor(mesh_sdf.occupancy(eval_grid.detach().cpu().numpy()), device=device, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.03)
    csv_path = os.path.join(out_dir, "curve.csv")
    header = ["epoch", "beta", "loss", "soft_iou", "hard_onehot_iou", "hard_multihot_iou", "best_soft_iou", "best_hard_onehot_iou", "best_hard_multihot_iou",
              "available_physical_primitives", "active_physical_primitives", "active_leaves", "groups_active", "type_entropy", "wall_s"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    P = None; Y = None
    best_soft = 0.0; best_h1 = 0.0; best_hm = 0.0
    t0 = time.time()
    last_metrics: Dict[str, float] = {}
    last_stats: Dict[str, object] = {}

    degenerate_count = 0
    for ep in range(args.epochs):
        beta = v3.beta_schedule(ep, args.epochs, args.beta_start, args.beta_end)
        if cache_P is not None:
            P, Y = cache_P, cache_Y
        elif ep % args.resample_every == 0 or P is None:
            pts, occ = mesh_sdf.sample_points(args.samples, balanced=not args.no_balance)
            P = torch.tensor(pts, device=device, dtype=torch.float32)
            Y = torch.tensor(occ, device=device, dtype=torch.float32)
        if args.batch_size > 0 and args.batch_size < P.shape[0]:
            idx = torch.randint(0, P.shape[0], (args.batch_size,), device=device)
            p_batch = P[idx]; y_batch = Y[idx]
        else:
            p_batch, y_batch = P, Y

        # Soft training path of the v3 core.
        pred = model.forward_modes(p_batch, beta)["soft"]
        if args.loss == "bce":
            recon = F.binary_cross_entropy(pred.clamp(1e-5, 1 - 1e-5), y_batch)
        else:
            recon = F.mse_loss(pred, y_batch)
        warm = v3.reg_warm(ep, args.epochs, args.sparsity_warmup, args.sparsity_rampup)
        reg = model.structure_loss(warm, beta, args)
        loss = recon + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            metrics = compute_metrics_extended(model, eval_grid, eval_gt, beta=args.beta_end, chunk=args.eval_chunk, type_thr=args.active_thr)
            stats = active_counts(model, method, args.beta_end, args.active_thr, args)
            best_soft = max(best_soft, metrics["soft_iou"])
            best_h1 = max(best_h1, metrics["hard_onehot_iou"])
            best_hm = max(best_hm, metrics["hard_multihot_iou"])
            last_metrics, last_stats = metrics, stats
            wall = time.time() - t0
            row = [ep + 1, f"{beta:.6f}", f"{loss.item():.8f}", f"{metrics['soft_iou']:.6f}", f"{metrics['hard_onehot_iou']:.6f}", f"{metrics['hard_multihot_iou']:.6f}",
                   f"{best_soft:.6f}", f"{best_h1:.6f}", f"{best_hm:.6f}", int(stats["available_physical_primitives"]), int(stats["active_physical_primitives"]),
                   int(stats["active_leaves"]), int(stats["groups_active"]), f"{float(stats['type_entropy']):.6f}", f"{wall:.2f}"]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"[{run_name}] ep {ep+1:6d} beta={beta:6.2f} soft={metrics['soft_iou']:.4f} onehot={metrics['hard_onehot_iou']:.4f} multi={metrics['hard_multihot_iou']:.4f} avail={int(stats['available_physical_primitives'])} active={int(stats['active_physical_primitives'])} g={int(stats['groups_active'])} ent={float(stats['type_entropy']):.3f}")

            # Optional fail-fast for clearly degenerate tiny budgets; disabled by default.
            if args.early_degenerate_stop > 0 and ep + 1 >= args.early_degenerate_after:
                if metrics["soft_iou"] <= args.early_degenerate_iou and metrics["hard_onehot_iou"] <= 1e-6 and metrics["hard_multihot_iou"] <= 1e-6:
                    degenerate_count += 1
                else:
                    degenerate_count = 0
                if degenerate_count >= args.early_degenerate_stop:
                    print(f"[{run_name}] early degenerate stop after {degenerate_count} bad evals")
                    break

    # Export if requested.
    if args.export_objs and last_metrics:
        try:
            export_dir = os.path.join(out_dir, "objs")
            os.makedirs(export_dir, exist_ok=True)
            modes = batched_forward_extended(model, eval_grid, args.beta_end, args.eval_chunk, args.active_thr)
            v3.export_occ_obj(os.path.join(export_dir, "recon_soft.obj"), modes["soft"], args.resolution)
            v3.export_occ_obj(os.path.join(export_dir, "recon_hard_onehot.obj"), modes["hard_onehot"], args.resolution)
            v3.export_occ_obj(os.path.join(export_dir, "recon_hard_multihot.obj"), modes["hard_multihot"], args.resolution)
        except Exception as e:
            print(f"warning: OBJ export failed for {run_name}: {e}")

    return RunResult(last_metrics, best_soft, best_h1, best_hm, last_stats, time.time() - t0)

# Suites, plotting, main

def default_methods_for_suite(suite: str) -> List[str]:
    if suite == "original_and_leaf":
        return ["legacy_nary_discrete", "legacy_nary_bc_vol", "legacy_tree_discrete", "legacy_tree_bc_vol",
                "nary_discrete", "nary_bc_vol", "nary_onehot_product_vol", "nary_multihot_product_vol", "tree_discrete", "tree_bc_vol", "tree_multihot_product_vol"]
    if suite == "budget_siggraph":
        return ["nary_onehot_product_vol", "nary_multihot_product_vol", "tree_onehot_product_vol", "tree_multihot_product_vol",
                "stump_onehot_product_vol", "stump_multihot_product_vol", "paper_stump_onehot_product_vol", "paper_stump_multihot_product_vol"]
    if suite == "leaf_ablation":
        return ["nary_bc_vol", "nary_onehot_product_vol", "nary_multihot_product_vol", "nary_onehot_sum_vol", "nary_multihot_sum_vol",
                "tree_bc_vol", "tree_multihot_product_vol"]
    return ["nary_discrete", "nary_bc_vol", "nary_multihot_product_vol", "tree_discrete", "tree_bc_vol", "tree_multihot_product_vol"]


def run_name(method: str, b: int, bc: int, g: int, seed: int) -> str:
    return f"{method}__base{b}_bc{bc}_g{g}_seed{seed}"


def generate_plots(mesh_out: str) -> None:
    if plt is None:
        return
    plots = os.path.join(mesh_out, "plots")
    os.makedirs(plots, exist_ok=True)
    curves = os.path.join(mesh_out, "curves")
    rows = []
    for fn in os.listdir(curves) if os.path.isdir(curves) else []:
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(curves, fn)
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        stem = fn[:-4]
        # method contains __base...
        method = stem.split("__base")[0]
        try:
            avail = int(data[-1]["available_physical_primitives"])
        except Exception:
            avail = 0
        for d in data:
            d["method"] = method; d["avail"] = avail
        rows.extend(data)
    if not rows:
        return

    # Best IoU per epoch for all methods/budgets.
    for metric, ylabel in [("best_soft_iou", "Best Soft IoU"), ("best_hard_onehot_iou", "Best Hard One-Hot IoU"), ("best_hard_multihot_iou", "Best Hard Multi-Hot IoU")]:
        fig, ax = plt.subplots(figsize=(11, 7))
        grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
        for r in rows:
            grouped.setdefault((r["method"], int(r["avail"])), []).append(r)
        for (m, avail), rs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            rs = sorted(rs, key=lambda r: int(r["epoch"]))
            xs = [int(r["epoch"]) for r in rs]
            ys = [float(r.get(metric, "nan")) for r in rs]
            ax.plot(xs, ys, linewidth=1.5, label=f"{method_label(m)} A={avail}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} by available physical primitive budget")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(plots, f"{metric}_per_epoch_all.png"), dpi=150); plt.close(fig)

    # Final budget summary plots.
    final = []
    for fn in os.listdir(curves) if os.path.isdir(curves) else []:
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(curves, fn), newline="") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        d = data[-1]
        d["method"] = fn[:-4].split("__base")[0]
        final.append(d)
    for metric, ylabel in [("best_soft_iou", "Best Soft IoU"), ("best_hard_onehot_iou", "Best Hard One-Hot IoU"), ("best_hard_multihot_iou", "Best Hard Multi-Hot IoU"), ("active_physical_primitives", "Active Physical Primitives")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for m in sorted(set(r["method"] for r in final)):
            rs = [r for r in final if r["method"] == m]
            # Average seeds with same avail.
            vals: Dict[int, List[float]] = {}
            for r in rs:
                vals.setdefault(int(r["available_physical_primitives"]), []).append(float(r[metric]))
            xs = sorted(vals)
            ys = [float(np.mean(vals[x])) for x in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=method_label(m))
        ax.set_xlabel("Available physical primitive budget"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(plots, f"budget_{metric}.png"), dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stable CSG primitive/leaf budget ablation using working v3 solver core.")
    ap.add_argument("--meshes", nargs="+", default=["bunny"])
    ap.add_argument("--meshes-dir", default="./meshes")
    ap.add_argument("--fetch-meshes", action="store_true")
    ap.add_argument("--output", default="./csg_budget_stable")
    ap.add_argument("--suite", choices=["core", "leaf_ablation", "original_and_leaf", "budget_siggraph"], default="budget_siggraph")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--allow-legacy-fallback", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--cache-points", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--resample-every", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--eval-chunk", type=int, default=32768)
    ap.add_argument("--baseline-prims", type=int, default=12)
    ap.add_argument("--bc-prims", type=int, default=50)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--baseline-prims-list", default=None)
    ap.add_argument("--bc-prims-list", default=None)
    ap.add_argument("--groups-list", default=None)
    ap.add_argument("--budget-grid", action="store_true")
    ap.add_argument("--stump-psi", type=float, default=20.0)
    ap.add_argument("--max-group-size", type=float, default=8.0)
    ap.add_argument("--active-thr", type=float, default=0.5)
    ap.add_argument("--tree-ops", choices=["csg3", "csg5"], default="csg3")
    ap.add_argument("--type-selector", choices=["sine_softmax", "plain_softmax", "sine_sigmoid_multi"], default="sine_softmax")
    ap.add_argument("--type-agg", choices=["product", "sum"], default="product")
    ap.add_argument("--omega", type=float, default=10.0)
    ap.add_argument("--beta-start", type=float, default=1.0)
    ap.add_argument("--beta-end", type=float, default=20.0)
    ap.add_argument("--volume-norm", action="store_true", default=True)
    ap.add_argument("--no-volume-norm", dest="volume_norm", action="store_false")
    ap.add_argument("--loss", choices=["mse", "bce"], default="mse")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sparsity-warmup", type=int, default=None)
    ap.add_argument("--sparsity-rampup", type=int, default=None)
    ap.add_argument("--lambda-sparse", type=float, default=2e-4)
    ap.add_argument("--lambda-binary", type=float, default=5e-5)
    ap.add_argument("--lambda-conflict", type=float, default=2e-4)
    ap.add_argument("--lambda-group", type=float, default=1e-5)
    ap.add_argument("--lambda-type-entropy", type=float, default=2e-4)
    ap.add_argument("--lambda-op-entropy", type=float, default=1e-4)
    ap.add_argument("--strict-v3-init", action="store_true", help="use v3's original 0.35 literal initialization; otherwise use stable low-budget init")
    ap.add_argument("--init-lit-prob", type=float, default=0.85, help="selected literal/intersection probability for stable init")
    ap.add_argument("--init-union-prob", type=float, default=0.65, help="initial union/clause activation probability")
    ap.add_argument("--init-complement-prob", type=float, default=0.03, help="initial complement probability for stump models")
    ap.add_argument("--init-off-logit", type=float, default=-7.0, help="logit for unselected Boolean connections")
    ap.add_argument("--init-tree-gate-prob", type=float, default=0.65, help="initial active leaf gate probability for tree models")
    ap.add_argument("--init-union-leaf-prob", type=float, default=0.35, help="initial active leaf gate probability for union-only models")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--export-objs", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--early-degenerate-stop", type=int, default=0, help="Stop a run after this many consecutive degenerate evals; 0 disables.")
    ap.add_argument("--early-degenerate-after", type=int, default=4000)
    ap.add_argument("--early-degenerate-iou", type=float, default=0.16)
    args = ap.parse_args()

    if args.sparsity_warmup is None:
        args.sparsity_warmup = max(args.epochs // 20, 1500)
    if args.sparsity_rampup is None:
        args.sparsity_rampup = max(args.epochs // 12, 2000)

    device = v3.get_device(args.device)
    if args.fetch_meshes:
        for name in v3.MESH_REGISTRY:
            v3.download_mesh(name, args.meshes_dir)
        return
    methods = args.methods if args.methods else default_methods_for_suite(args.suite)
    budgets = expand_budget_lists(parse_int_list(args.baseline_prims_list, args.baseline_prims), parse_int_list(args.bc_prims_list, args.bc_prims), parse_int_list(args.groups_list, args.groups), args.budget_grid)

    print("=" * 100)
    print("Stable CSG Leaf/Primitive Budget Ablation")
    print(f"Device: {device}")
    print(f"Methods: {methods}")
    print(f"Budgets: {budgets}")
    print("Training core: working siggraph_csg_benchmark_v3 classes/functions")
    print("=" * 100)

    os.makedirs(args.output, exist_ok=True)
    resolved = v3.resolve_mesh_specs(args.meshes, args.meshes_dir)
    for mesh_path, mesh_name in resolved:
        mesh_out = os.path.join(args.output, mesh_name)
        os.makedirs(mesh_out, exist_ok=True)
        curves_dir = os.path.join(mesh_out, "curves"); os.makedirs(curves_dir, exist_ok=True)
        runs_dir = os.path.join(mesh_out, "runs"); os.makedirs(runs_dir, exist_ok=True)
        if args.plot_only:
            generate_plots(mesh_out); continue
        print("\n" + "#" * 100)
        print(f"Mesh: {mesh_name} ({mesh_path})")
        print("#" * 100)
        mesh_sdf = v3.MeshSDF(mesh_path)
        for b, bc, g in budgets:
            print("\n" + "=" * 90)
            print(f"Budget: baseline_prims={b}/type, bc_leaves={bc}, groups={g}")
            print(f"  fixed discrete available = {b*N_TYPES}")
            print(f"  one-hot BC available    = {bc}")
            print(f"  multi-hot BC available  = {bc*N_TYPES}")
            print("=" * 90)
            for seed in range(args.seeds):
                for method in methods:
                    # Budget-specific arguments, set in place as the v3 core expects.
                    args.baseline_prims = int(b); args.bc_prims = int(bc); args.groups = int(g)
                    rn = run_name(method, b, bc, g, seed)
                    run_dir = os.path.join(runs_dir, rn)
                    curve_target = os.path.join(curves_dir, rn + ".csv")
                    if os.path.exists(curve_target):
                        print(f"skip existing {rn}")
                        continue
                    print("\n" + "-" * 80)
                    print(f"Run: mesh={mesh_name} method={method} budget=base{b}/bc{bc}/g{g} seed={seed}")
                    print("-" * 80)
                    res = train_one_run(mesh_sdf, method, seed * 1000 + 42, run_dir, args, device)
                    # Copy curve to flat curves dir for plotting.
                    src = os.path.join(run_dir, "curve.csv")
                    if os.path.exists(src):
                        import shutil
                        shutil.copyfile(src, curve_target)
                    with open(os.path.join(run_dir, "result.json"), "w") as f:
                        json.dump({"method": method, "mesh": mesh_name, "budget": {"baseline_prims": b, "bc_prims": bc, "groups": g},
                                   "best_soft_iou": res.best_soft, "best_hard_onehot_iou": res.best_hard_onehot, "best_hard_multihot_iou": res.best_hard_multihot,
                                   "final_metrics": res.final_metrics, "stats": res.stats, "wall_s": res.wall_s}, f, indent=2)
                    generate_plots(mesh_out)
        generate_plots(mesh_out)
    print("\nDone.")

if __name__ == "__main__":
    main()
