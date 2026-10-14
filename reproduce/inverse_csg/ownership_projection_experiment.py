#!/usr/bin/env python3
"""
Ownership experiment for hard-extractable inverse CSG.

Imports nary_benchmark_v7.py and full_benchmark_with_levels.py from the same folder.

Steps:
  1. Load a trained TreeBcVolCSG checkpoint.
  2. Check that the checkpoint reproduces its IoU.
  3. Compute Aumann-Shapley / integrated-gradient volume ownership
     through the fuzzy Boolean tree.
  4. Check ownership completeness numerically: sum_i W_i(x) ~= F(x).
  5. Optionally compute boundary ownership on target surface samples,
        B_i(x) = ||grad_x o_i(x)|| * |dF/do_i|,
     used for visualization only.
  6. Optionally project each owned volume region back to one valid typed
     primitive and evaluate hard IoU / pruning.

Typical usage:
  python ownership_projection_experiment.py \
    --mesh bunny.obj \
    --checkpoint path/to/tree_bc_vol_best_model.pt \
    --output ./ownership_bunny \
    --model tree \
    --n-primitives 50

  python ownership_projection_experiment.py \
    --mesh bunny.obj \
    --checkpoint path/to/tree_bc_vol_best_model.pt \
    --output ./ownership_bunny_project \
    --model tree \
    --run-projection \
    --projection-steps 150 \
    --projection-types all \
    --prune-mass-frac 0.002
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nary_benchmark_v7 import (
    MeshSDF,
    sample_3D_grid,
    compute_iou,
    batched_forward,
    device,
    TYPE_NAMES,
    MIN_SIZE,
    euler_to_R,
    sdf_sphere_local,
    sdf_block_local,
    sdf_cylinder_local,
    sdf_wedge_local,
    sdf_corner_wedge_local,
    BcVariantCSG,
)

from full_benchmark_with_levels import TreeBcVolCSG, TreeBcCSG


# Basic model / checkpoint helpers

def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint object: {type(ckpt)}")

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        stripped = {}
        for k, v in state.items():
            stripped[k[len("module."):] if k.startswith("module.") else k] = v
        model.load_state_dict(stripped, strict=True)


def make_model(args) -> nn.Module:
    if args.model == "tree":
        if args.tree_leaf_mode == "bc_vol_norm":
            model = TreeBcVolCSG(num_primitives=args.n_primitives, omega=args.omega)
        else:
            model = TreeBcCSG(
                num_primitives=args.n_primitives,
                omega=args.omega,
                occ_mode=args.tree_leaf_mode,
            )
    elif args.model == "flat":
        model = BcVariantCSG(
            num_primitives=args.n_primitives,
            num_groups=args.n_groups,
            omega=args.omega,
            occ_mode=args.tree_leaf_mode,
        )
    else:
        raise ValueError(args.model)
    return model.to(device)


def target_func_from_mesh(mesh_sdf: MeshSDF):
    def target_func(pts: torch.Tensor) -> torch.Tensor:
        occ = (mesh_sdf.query(pts.detach().cpu().numpy()) < 0).astype(np.float32)
        return torch.from_numpy(occ).to(pts.device)
    return target_func


def verify_iou(model: nn.Module, target_func, resolution: int, temperature: float) -> Dict[str, float]:
    eval_grid = sample_3D_grid(resolution).to(device)
    eval_gt = target_func(eval_grid)
    with torch.no_grad():
        pred_soft = batched_forward(model, eval_grid, temperature=temperature)
        soft_iou = compute_iou(pred_soft, eval_gt)
        threshold_iou = compute_iou((pred_soft > 0.5).float(), eval_gt)

        if hasattr(model, "forward_hard_dnf"):
            chunks = []
            for b in torch.split(eval_grid, 32 ** 3):
                chunks.append(model.forward_hard_dnf(b))
            pred_hard = torch.cat(chunks, dim=0)
            hard_iou = compute_iou(pred_hard, eval_gt)
        else:
            hard_iou = threshold_iou

    return {"soft_iou": soft_iou, "threshold_iou": threshold_iou, "hard_iou": hard_iou}


# Tree/flat leaf interface

def get_leaf_occs(model: nn.Module, pts: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return leaf occupancy tensor with shape (B, N)."""
    if hasattr(model, "pool") and hasattr(model.pool, "get_all_occs"):
        mode = getattr(model, "occ_mode", "bc_vol_norm")
        return model.pool.get_all_occs(pts, mode=mode, temperature=temperature)
    raise RuntimeError("Model does not expose model.pool.get_all_occs")


def reduce_from_leaves(model: nn.Module, leaves: torch.Tensor, temperature: float, hard: bool = False) -> torch.Tensor:
    """Run the actual CSG aggregator from precomputed leaves."""
    if hasattr(model, "_reduce_tree"):
        return model._reduce_tree(leaves, temperature=temperature, hard=hard)

    # Flat DNF fallback for BcVariantCSG.
    if hasattr(model, "m_pos") and hasattr(model, "m_neg") and hasattr(model, "act"):
        omega = model.omega
        if hard:
            mp = (torch.sigmoid(torch.sin(omega * model.m_pos) * 20.0) > 0.5).float()
            mn = (torch.sigmoid(torch.sin(omega * model.m_neg) * 20.0) > 0.5).float()
            a = (torch.sigmoid(torch.sin(omega * model.act) * 20.0) > 0.5).float()
        else:
            mp = torch.sigmoid(torch.sin(omega * model.m_pos) * temperature)
            mn = torch.sigmoid(torch.sin(omega * model.m_neg) * temperature)
            a = torch.sigmoid(torch.sin(omega * model.act) * temperature)

        o = leaves.unsqueeze(-1)  # (B, N, K)
        fp = 1.0 - mp.unsqueeze(0) * (1.0 - o)
        fn = 1.0 - mn.unsqueeze(0) * o
        lg = torch.log(fp.clamp_min(1e-10)).sum(dim=1) + torch.log(fn.clamp_min(1e-10)).sum(dim=1)
        ag = a.unsqueeze(0) * torch.exp(lg)
        return (1.0 - (1.0 - ag.clamp(0, 1)).prod(dim=1)).clamp(0, 1)

    raise RuntimeError("Do not know how to reduce leaves for this model")


def get_num_prims(model: nn.Module) -> int:
    if hasattr(model, "N"):
        return int(model.N)
    if hasattr(model, "pool") and hasattr(model.pool, "N"):
        return int(model.pool.N)
    raise RuntimeError("Could not infer primitive count")


# Aumann-Shapley / integrated volume ownership

@dataclass
class OwnershipResult:
    W: torch.Tensor
    F: torch.Tensor
    endpoint: torch.Tensor
    comp_rel_l1: float
    comp_max_abs: float
    endpoint_rel_l1: float


def gauss_legendre_01(n: int, device_) -> Tuple[torch.Tensor, torch.Tensor]:
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1.0)
    ww = 0.5 * w
    return (torch.tensor(t, dtype=torch.float32, device=device_),
            torch.tensor(ww, dtype=torch.float32, device=device_))


def aumann_shapley_chunk(model: nn.Module, leaf_occs: torch.Tensor, temperature: float, quadrature: int) -> OwnershipResult:
    leaf_occs = leaf_occs.detach()

    with torch.no_grad():
        F_final = reduce_from_leaves(model, leaf_occs, temperature=temperature, hard=False).detach()

    taus, weights = gauss_legendre_01(quadrature, leaf_occs.device)
    integrated_grad = torch.zeros_like(leaf_occs)

    for tau, wt in zip(taus, weights):
        z = (tau * leaf_occs).detach().clone().requires_grad_(True)
        F_tau = reduce_from_leaves(model, z, temperature=temperature, hard=False)
        grad = torch.autograd.grad(F_tau.sum(), z, retain_graph=False, create_graph=False)[0]
        integrated_grad = integrated_grad + wt * grad.detach()

    W = leaf_occs * integrated_grad

    z1 = leaf_occs.detach().clone().requires_grad_(True)
    F1 = reduce_from_leaves(model, z1, temperature=temperature, hard=False)
    grad1 = torch.autograd.grad(F1.sum(), z1, retain_graph=False, create_graph=False)[0]
    endpoint = leaf_occs * grad1.detach()

    W_sum = W.sum(dim=1)
    endpoint_sum = endpoint.sum(dim=1)
    denom = F_final.abs().mean().item() + 1e-8
    return OwnershipResult(
        W=W.detach(),
        F=F_final.detach(),
        endpoint=endpoint.detach(),
        comp_rel_l1=(W_sum - F_final).abs().mean().item() / denom,
        endpoint_rel_l1=(endpoint_sum - F_final).abs().mean().item() / denom,
        comp_max_abs=(W_sum - F_final).abs().max().item(),
    )


def compute_volume_ownership(model: nn.Module, pts: torch.Tensor, temperature: float, quadrature: int, chunk_size: int) -> OwnershipResult:
    all_W, all_F, all_E = [], [], []
    for chunk in torch.split(pts, chunk_size):
        with torch.no_grad():
            leaves = get_leaf_occs(model, chunk, temperature=temperature)
        res = aumann_shapley_chunk(model, leaves, temperature=temperature, quadrature=quadrature)
        all_W.append(res.W.cpu())
        all_F.append(res.F.cpu())
        all_E.append(res.endpoint.cpu())

    W = torch.cat(all_W, dim=0)
    F = torch.cat(all_F, dim=0)
    endpoint = torch.cat(all_E, dim=0)
    denom = F.abs().mean().item() + 1e-8
    return OwnershipResult(
        W=W,
        F=F,
        endpoint=endpoint,
        comp_rel_l1=(W.sum(1) - F).abs().mean().item() / denom,
        endpoint_rel_l1=(endpoint.sum(1) - F).abs().mean().item() / denom,
        comp_max_abs=(W.sum(1) - F).abs().max().item(),
    )


# Boundary ownership for visualization

def sample_target_surface(mesh_sdf: MeshSDF, n: int) -> torch.Tensor:
    pts_np, _ = mesh_sdf.sample_points(n, surface_ratio=1.0, near_ratio=0.0)
    return torch.from_numpy(pts_np).float().to(device)


def compute_boundary_scores(model: nn.Module, surface_pts: torch.Tensor, temperature: float, chunk_size: int, exact_spatial_grad: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_scores, all_signs, all_leaves = [], [], []
    N = get_num_prims(model)

    for chunk in torch.split(surface_pts, chunk_size):
        pts_req = chunk.detach().clone().requires_grad_(True)
        leaves = get_leaf_occs(model, pts_req, temperature=temperature)

        leaf_var = leaves.detach().clone().requires_grad_(True)
        F_leaf = reduce_from_leaves(model, leaf_var, temperature=temperature, hard=False)
        dF_dleaf = torch.autograd.grad(F_leaf.sum(), leaf_var, retain_graph=False, create_graph=False)[0]

        if exact_spatial_grad:
            grad_norm_cols = []
            for i in range(N):
                gi = torch.autograd.grad(
                    leaves[:, i].sum(), pts_req,
                    retain_graph=True, create_graph=False, allow_unused=False
                )[0]
                grad_norm_cols.append(gi.norm(dim=-1))
            grad_norm = torch.stack(grad_norm_cols, dim=1)
        else:
            grad_norm = leaves.detach() * (1.0 - leaves.detach())

        score = grad_norm.detach() * dF_dleaf.detach().abs()
        all_scores.append(score.cpu())
        all_signs.append(dF_dleaf.detach().cpu())
        all_leaves.append(leaves.detach().cpu())

    return torch.cat(all_scores, 0), torch.cat(all_signs, 0), torch.cat(all_leaves, 0)


def owner_from_scores(scores: torch.Tensor, decisive_ratio: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    top2, idx = scores.topk(2, dim=1)
    owner = idx[:, 0]
    decisive = top2[:, 0] > decisive_ratio * (top2[:, 1] + 1e-12)
    return owner, decisive


def save_colored_pointcloud(path: str, pts: torch.Tensor, owner: torch.Tensor, decisive: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pts_np = pts.detach().cpu().numpy()
    owner_np = owner.detach().cpu().numpy()
    decisive_np = decisive.detach().cpu().numpy()

    rng = np.random.default_rng(123)
    n_colors = int(owner_np.max()) + 1 if len(owner_np) else 1
    palette = rng.uniform(0.15, 1.0, size=(max(n_colors, 8), 3))
    colors = np.zeros((len(pts_np), 3), dtype=np.float32)
    for i in range(len(pts_np)):
        colors[i] = palette[owner_np[i] % len(palette)] if decisive_np[i] else np.array([0.55, 0.55, 0.55])

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_np)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts_np, colors):
            r, g, b = (np.clip(c, 0, 1) * 255).astype(np.uint8)
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(r)} {int(g)} {int(b)}\n")


def summarize_owner_distribution(owner: torch.Tensor, decisive: torch.Tensor, n_prims: int) -> Dict:
    counts = torch.bincount(owner, minlength=n_prims).float()
    frac = counts / max(float(owner.numel()), 1.0)
    active = int((frac > 0.005).sum().item())
    top_vals, top_idx = torch.topk(frac, min(10, n_prims))
    return {
        "decisive_fraction": float(decisive.float().mean().item()),
        "active_owners_gt_0p5pct": active,
        "top_owner_indices": [int(i) for i in top_idx.cpu().tolist()],
        "top_owner_fracs": [float(v) for v in top_vals.cpu().tolist()],
    }


# Projection back to valid typed primitives

TYPE_INTRINSIC = [1, 3, 2, 3, 3]

def type_offsets() -> List[int]:
    out = [0]
    for c in TYPE_INTRINSIC:
        out.append(out[-1] + c)
    return out


def sdf_for_type(ptype: int, pts: torch.Tensor, center: torch.Tensor, rot: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
    R = euler_to_R(rot)
    pl = torch.matmul(pts - center, R.T)
    if ptype == 0:
        return sdf_sphere_local(pl, intr[0])
    if ptype == 1:
        return sdf_block_local(pl, intr)
    if ptype == 2:
        return sdf_cylinder_local(pl, intr[0], intr[1])
    if ptype == 3:
        return sdf_wedge_local(pl, intr)
    return sdf_corner_wedge_local(pl, intr)


def vol_radius_for_type(ptype: int, intr: torch.Tensor) -> torch.Tensor:
    s = intr.abs() + MIN_SIZE
    if ptype == 0:
        return s[0]
    if ptype == 1:
        return (6.0 * s.prod() / math.pi).pow(1.0 / 3.0)
    if ptype == 2:
        return (1.5 * s[0] * s[0] * s[1]).pow(1.0 / 3.0)
    if ptype == 3:
        return (3.0 * s.prod() / math.pi).pow(1.0 / 3.0)
    return (1.5 * s.prod() / math.pi).pow(1.0 / 3.0)


def primitive_phi(ptype: int, pts: torch.Tensor, center: torch.Tensor, rot: torch.Tensor, intr: torch.Tensor, sharpness, vol_norm: bool = True) -> torch.Tensor:
    sdf = sdf_for_type(ptype, pts, center, rot, intr)
    if vol_norm:
        sdf = sdf / (vol_radius_for_type(ptype, intr) + 1e-8)
    sharp = torch.as_tensor(sharpness, dtype=pts.dtype, device=pts.device).abs().clamp(min=1.0)
    return torch.sigmoid(-sdf * sharp)


@dataclass
class ProjectedPrimitive:
    active: bool
    ptype: int = -1
    center: Optional[torch.Tensor] = None
    rot: Optional[torch.Tensor] = None
    intr: Optional[torch.Tensor] = None
    sharpness: float = 20.0
    loss: float = float("inf")
    mass: float = 0.0


def dominant_type(model: nn.Module, i: int) -> int:
    if hasattr(model.pool, "get_dominant_type"):
        t = model.pool.get_dominant_type(i)
        return TYPE_NAMES.index(t) if isinstance(t, str) else int(t)
    if hasattr(model.pool, "type_logits"):
        weights = torch.sigmoid(torch.sin(model.pool.omega * model.pool.type_logits[i]) * 20.0)
        return int(weights.argmax().item())
    return 0


def init_intr_for_type_from_pool(model: nn.Module, i: int, ptype: int) -> torch.Tensor:
    off = type_offsets()
    if hasattr(model.pool, "intrinsic"):
        return model.pool.intrinsic[i, off[ptype]:off[ptype+1]].detach().clone()
    return torch.ones(TYPE_INTRINSIC[ptype], device=device) * 0.2


def project_one_slot(model: nn.Module, i: int, pts: torch.Tensor, target_occ: torch.Tensor, owned_weight: torch.Tensor, args) -> ProjectedPrimitive:
    mass = float(owned_weight.mean().item())
    if owned_weight.max().item() <= 1e-8 or owned_weight.sum().item() <= 1e-8:
        return ProjectedPrimitive(active=False, mass=mass)

    if args.projection_types == "dominant":
        types = [dominant_type(model, i)]
    elif args.projection_types == "current+neighbors":
        dt = dominant_type(model, i)
        types = sorted(set([dt, 0, 1, 2]))
    else:
        types = list(range(len(TYPE_NAMES)))

    base_center = model.pool.centers[i].detach().clone()
    base_rot = model.pool.rotations[i].detach().clone()
    base_sharp = float(model.pool.sharpness[i].detach().abs().clamp(min=1.0).item()) if hasattr(model.pool, "sharpness") else 20.0

    best = ProjectedPrimitive(active=False, mass=mass)
    w = owned_weight.detach().clamp(min=0)
    w_norm = w / (w.mean() + 1e-8)
    y = target_occ.detach().float()

    for ptype in types:
        center = nn.Parameter(base_center.clone())
        rot = nn.Parameter(base_rot.clone())
        intr = nn.Parameter(init_intr_for_type_from_pool(model, i, ptype).clone())
        opt = torch.optim.Adam([center, rot, intr], lr=args.projection_lr)

        for _ in range(args.projection_steps):
            opt.zero_grad(set_to_none=True)
            phi = primitive_phi(ptype, pts, center, rot, intr, base_sharp, vol_norm=True)
            fit = (w_norm * (1.0 - phi).pow(2)).mean()
            spill = ((1.0 - y) * phi.pow(2)).mean()
            loss = fit + args.spill_weight * spill + 1e-5 * intr.abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([center, rot, intr], 1.0)
            opt.step()
            with torch.no_grad():
                intr.data = intr.data.clamp(min=-3.0, max=3.0)

        with torch.no_grad():
            phi = primitive_phi(ptype, pts, center, rot, intr, base_sharp, vol_norm=True)
            fit = (w_norm * (1.0 - phi).pow(2)).mean()
            spill = ((1.0 - y) * phi.pow(2)).mean()
            loss_val = float((fit + args.spill_weight * spill).item())

        if loss_val < best.loss:
            best = ProjectedPrimitive(True, ptype, center.detach().clone(), rot.detach().clone(), intr.detach().clone(), base_sharp, loss_val, mass)

    return best


def run_projection(model: nn.Module, pts: torch.Tensor, target_occ: torch.Tensor, W: torch.Tensor, args) -> Tuple[List[ProjectedPrimitive], Dict]:
    N = W.shape[1]
    pos_W = W.clamp(min=0)
    denom = pos_W.sum(dim=1, keepdim=True).clamp_min(1e-8)
    owned = target_occ.float().unsqueeze(1) * pos_W / denom
    mass_per_slot = owned.mean(dim=0)
    max_mass = mass_per_slot.max().item() + 1e-12
    active_mask = mass_per_slot > args.prune_mass_frac * max_mass

    projected = []
    print("\nProjection:")
    print(f"  Active before projection: {int(active_mask.sum().item())}/{N}")
    for i in range(N):
        if not bool(active_mask[i].item()):
            projected.append(ProjectedPrimitive(active=False, mass=float(mass_per_slot[i].item())))
            continue
        print(f"  slot {i:03d}: mass={mass_per_slot[i].item():.4e}, dominant={TYPE_NAMES[dominant_type(model, i)]}")
        pp = project_one_slot(model, i, pts, target_occ, owned[:, i], args)
        projected.append(pp)
        if pp.active:
            print(f"    -> {TYPE_NAMES[pp.ptype]} loss={pp.loss:.5f}")

    return projected, {
        "active_projected": int(sum(pp.active for pp in projected)),
        "mass_threshold_relative": args.prune_mass_frac,
        "projected": [
            {"active": pp.active, "type": TYPE_NAMES[pp.ptype] if pp.active else None, "loss": pp.loss, "mass": pp.mass}
            for pp in projected
        ],
    }


def eval_projected_program(model: nn.Module, projected: List[ProjectedPrimitive], grid: torch.Tensor, temperature: float, chunk_size: int = 32768) -> torch.Tensor:
    outs = []
    N = len(projected)
    for chunk in torch.split(grid, chunk_size):
        leaves = torch.zeros((chunk.shape[0], N), device=chunk.device)
        for i, pp in enumerate(projected):
            if pp.active:
                leaves[:, i] = primitive_phi(pp.ptype, chunk, pp.center.to(chunk.device), pp.rot.to(chunk.device), pp.intr.to(chunk.device), pp.sharpness, vol_norm=True)
        with torch.no_grad():
            outs.append(reduce_from_leaves(model, leaves, temperature=temperature, hard=True).detach().cpu())
    return torch.cat(outs, dim=0).to(grid.device)


# Main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="./ownership_experiment")
    p.add_argument("--model", choices=["tree", "flat"], default="tree")
    p.add_argument("--tree-leaf-mode", default="bc_vol_norm")
    p.add_argument("--n-primitives", type=int, default=50)
    p.add_argument("--n-groups", type=int, default=16)
    p.add_argument("--omega", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--eval-resolution", type=int, default=96)
    p.add_argument("--min-iou", type=float, default=0.90)
    p.add_argument("--allow-low-iou", action="store_true")

    p.add_argument("--num-volume-samples", type=int, default=50000)
    p.add_argument("--volume-chunk", type=int, default=4096)
    p.add_argument("--quadrature", type=int, default=12)

    p.add_argument("--num-surface-samples", type=int, default=8000)
    p.add_argument("--surface-chunk", type=int, default=512)
    p.add_argument("--boundary-exact-grad", action="store_true")
    p.add_argument("--skip-boundary", action="store_true")

    p.add_argument("--run-projection", action="store_true")
    p.add_argument("--projection-samples", type=int, default=30000)
    p.add_argument("--projection-steps", type=int, default=150)
    p.add_argument("--projection-lr", type=float, default=2e-2)
    p.add_argument("--projection-types", choices=["dominant", "current+neighbors", "all"], default="all")
    p.add_argument("--spill-weight", type=float, default=0.5)
    p.add_argument("--prune-mass-frac", type=float, default=0.002)
    p.add_argument("--projected-eval-resolution", type=int, default=96)

    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 78)
    print("OWNERSHIP PROJECTION EXPERIMENT")
    print("=" * 78)
    print(f"mesh:       {args.mesh}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"model:      {args.model}")
    print(f"output:     {args.output}")

    mesh_sdf = MeshSDF(args.mesh)
    target_func = target_func_from_mesh(mesh_sdf)
    model = make_model(args)
    load_checkpoint_into_model(model, args.checkpoint)
    model.eval()

    iou = verify_iou(model, target_func, args.eval_resolution, args.temperature)
    print("\nCheckpoint IoU:")
    print(f"  soft      {iou['soft_iou']:.4f}")
    print(f"  threshold {iou['threshold_iou']:.4f}")
    print(f"  hard      {iou['hard_iou']:.4f}")

    if max(iou["soft_iou"], iou["hard_iou"]) < args.min_iou and not args.allow_low_iou:
        print("\nERROR: checkpoint IoU is too low for ownership analysis.")
        print("Likely causes: wrong checkpoint, wrong model type, or wrong leaf occ mode.")
        print("Use --allow-low-iou only for debugging.")
        with open(os.path.join(args.output, "failed_low_iou.json"), "w") as f:
            json.dump(iou, f, indent=2)
        return

    print("\nSampling volume points...")
    pts_np, occ_np = mesh_sdf.sample_points(args.num_volume_samples)
    pts = torch.from_numpy(pts_np).float().to(device)
    gt = torch.from_numpy(occ_np).float().to(device)

    print("Computing Aumann-Shapley volume ownership...")
    vol_res = compute_volume_ownership(model, pts, args.temperature, args.quadrature, args.volume_chunk)
    W, Fvals, endpoint = vol_res.W, vol_res.F, vol_res.endpoint

    print("\nVolume ownership completeness:")
    print(f"  Aumann-Shapley rel L1 error: {vol_res.comp_rel_l1:.6e}")
    print(f"  Aumann-Shapley max abs error: {vol_res.comp_max_abs:.6e}")
    print(f"  Endpoint responsibility rel L1 error: {vol_res.endpoint_rel_l1:.6e}")

    W_pos = W.clamp(min=0)
    owner_vol = W_pos.argmax(dim=1)
    mass = W_pos.mean(dim=0)
    mass_frac = mass / (mass.sum() + 1e-12)
    top_vals, top_idx = torch.topk(mass_frac, min(10, mass_frac.numel()))
    print("\nTop positive volume owners:")
    for v, i in zip(top_vals.tolist(), top_idx.tolist()):
        print(f"  slot {i:03d}: {100*v:6.2f}%  type={TYPE_NAMES[dominant_type(model, i)]}")

    np.savez_compressed(
        os.path.join(args.output, "volume_ownership.npz"),
        points=pts.detach().cpu().numpy(),
        target=gt.detach().cpu().numpy(),
        F=Fvals.numpy(),
        W=W.numpy(),
        endpoint=endpoint.numpy(),
        owner=owner_vol.numpy(),
    )

    metrics = {
        "checkpoint_iou": iou,
        "volume_completeness": {
            "as_rel_l1": vol_res.comp_rel_l1,
            "as_max_abs": vol_res.comp_max_abs,
            "endpoint_rel_l1": vol_res.endpoint_rel_l1,
        },
        "top_volume_owners": [
            {"slot": int(i), "mass_frac": float(v), "dominant_type": TYPE_NAMES[dominant_type(model, int(i))]}
            for v, i in zip(top_vals.tolist(), top_idx.tolist())
        ],
    }

    if not args.skip_boundary:
        print("\nSampling target surface for boundary ownership visualization...")
        surf_pts = sample_target_surface(mesh_sdf, args.num_surface_samples)
        print("Computing boundary scores...")
        B, signed, leaves = compute_boundary_scores(model, surf_pts, args.temperature, args.surface_chunk, args.boundary_exact_grad)
        owner_b, decisive_b = owner_from_scores(B, decisive_ratio=2.0)
        b_summary = summarize_owner_distribution(owner_b, decisive_b, get_num_prims(model))
        print("\nBoundary ownership summary:")
        print(f"  decisive fraction: {b_summary['decisive_fraction']:.4f}")
        print(f"  active owners >0.5%: {b_summary['active_owners_gt_0p5pct']}")
        for idx, frac in zip(b_summary["top_owner_indices"], b_summary["top_owner_fracs"]):
            if frac > 0:
                print(f"  slot {idx:03d}: {100*frac:6.2f}% surface  type={TYPE_NAMES[dominant_type(model, idx)]}")
        ply_path = os.path.join(args.output, "boundary_ownership_target_surface.ply")
        save_colored_pointcloud(ply_path, surf_pts.detach().cpu(), owner_b, decisive_b)
        print(f"  saved: {ply_path}")
        np.savez_compressed(
            os.path.join(args.output, "boundary_ownership.npz"),
            points=surf_pts.detach().cpu().numpy(),
            B=B.numpy(),
            signed_grad=signed.numpy(),
            leaves=leaves.numpy(),
            owner=owner_b.numpy(),
            decisive=decisive_b.numpy(),
        )
        metrics["boundary"] = b_summary
        metrics["boundary_ply"] = ply_path

    if args.run_projection:
        print("\nRunning ownership projection...")
        if args.projection_samples != args.num_volume_samples:
            p_np, o_np = mesh_sdf.sample_points(args.projection_samples)
            p_pts = torch.from_numpy(p_np).float().to(device)
            p_gt = torch.from_numpy(o_np).float().to(device)
            p_res = compute_volume_ownership(model, p_pts, args.temperature, args.quadrature, args.volume_chunk)
            p_W = p_res.W.to(device)
        else:
            p_pts, p_gt, p_W = pts, gt, W.to(device)

        projected, proj_summary = run_projection(model, p_pts, p_gt, p_W, args)
        metrics["projection"] = proj_summary

        print("\nEvaluating projected hard program...")
        eval_grid = sample_3D_grid(args.projected_eval_resolution).to(device)
        eval_gt = target_func(eval_grid)
        pred_proj = eval_projected_program(model, projected, eval_grid, args.temperature)
        proj_iou = compute_iou(pred_proj, eval_gt)
        print(f"  projected hard IoU: {proj_iou:.4f}")
        metrics["projection"]["hard_iou"] = float(proj_iou)

        proj_json = []
        for i, pp in enumerate(projected):
            if not pp.active:
                proj_json.append({"slot": i, "active": False, "mass": pp.mass})
            else:
                proj_json.append({
                    "slot": i,
                    "active": True,
                    "type": TYPE_NAMES[pp.ptype],
                    "center": pp.center.cpu().tolist(),
                    "rotation": pp.rot.cpu().tolist(),
                    "intrinsic": pp.intr.cpu().tolist(),
                    "sharpness": pp.sharpness,
                    "loss": pp.loss,
                    "mass": pp.mass,
                })
        with open(os.path.join(args.output, "projected_primitives.json"), "w") as f:
            json.dump(proj_json, f, indent=2)

    with open(os.path.join(args.output, "ownership_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(f"Metrics: {os.path.join(args.output, 'ownership_metrics.json')}")


if __name__ == "__main__":
    main()
