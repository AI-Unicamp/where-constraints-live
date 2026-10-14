#!/usr/bin/env python3
"""
Calibrated-field ablation for typed inverse CSG.

Imports `nary_benchmark_v7.py` and `full_benchmark_with_levels.py` from the same folder.
Compares alternatives to the volume-radius SDF calibration:

  modes defined in nary_benchmark_v7.py:
    bc             : product-union over unnormalized SDF occupancies
    bc_size_norm   : mean-parameter size normalization
    bc_vol_norm    : volume-equivalent-radius normalization
    bc_area_norm   : surface-area-equivalent normalization
    bc_kj_norm     : size * local spatial-gradient normalization
    attn_gate      : input-dependent additive attention baseline

  modes added by this script:
    bc_tube_norm       : rho_T = 3V/A  (transition-shell / tube scale)
    bc_blend_norm      : rho = rho_V^(1-alpha) * rho_T^alpha
    bc_jac_static_norm : volume scale times a detached static Jacobian-energy factor

Modes (--mode):

  1. diagnostic: no training. Compares calibrations on canonical primitives by
     transition mass, spatial gradient energy, and finite-difference parameter
     Jacobian energy.

  2. synthetic: trains small BcVariantCSG models on analytic targets with known
     shape programs.

  3. mesh: trains flat or tree models on a mesh with the selected calibration
     modes. The added modes are supported for the flat model; for the tree model
     they require TreeBcCSG(num_primitives, omega, occ_mode) in full_benchmark_with_levels.py.

Examples
--------

# 1) cheap diagnostic, no training
python calibrated_field_ablation.py --mode diagnostic --output ./calib_diag

# 2) synthetic type/fit ablation
python calibrated_field_ablation.py --mode synthetic --case sphere_union_box \
  --epochs 4000 --n-primitives 12 \
  --occ-modes bc bc_vol_norm bc_area_norm bc_kj_norm bc_tube_norm bc_blend_norm \
  --output ./calib_synth

# 3) mesh ablation on flat DNF
python calibrated_field_ablation.py --mode mesh --mesh bunny.obj \
  --epochs 30000 --n-primitives 50 --n-groups 16 \
  --occ-modes bc_vol_norm bc_tube_norm bc_blend_norm bc_area_norm bc_kj_norm \
  --output ./calib_bunny_flat

# 4) small smoke test
python calibrated_field_ablation.py --mode synthetic --case sphere_union_box \
  --epochs 200 --n-primitives 8 --occ-modes bc_vol_norm bc_tube_norm

Notes
-----
- bc_blend_norm with alpha=0.5 gives rho = sqrt(rho_vol * rho_tube), the
  geometric mean of the volume and tube scales.

- bc_jac_static_norm computes a detached Jacobian-energy scale once, at
  initialization, and is used as a diagnostic.

- The added modes are installed on BcTypePool by `patch_bc_type_pool()`.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Repo imports
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from nary_benchmark_v7 import (
        MeshSDF, sample_3D_grid, compute_iou, batched_forward,
        sdf_sphere_local, sdf_block_local, sdf_cylinder_local,
        sdf_wedge_local, sdf_corner_wedge_local, euler_to_R,
        device, TYPE_NAMES, N_TYPES, MIN_SIZE,
        train as train_full,
        BcVariantCSG,
        surface_init as v7_surface_init,
        spatial_group_init as v7_spatial_group_init,
    )
except Exception as e:
    raise RuntimeError(
        "Could not import nary_benchmark_v7.py. Put this script next to your repo files.\n"
        f"Original error: {e}"
    )

try:
    # Prefer general tree class if available. Some versions only expose TreeBcVolCSG.
    from full_benchmark_with_levels import TreeBcCSG, TreeBcVolCSG, tree_surface_init, tree_spatial_init
except Exception:
    TreeBcCSG = None
    try:
        from full_benchmark_with_levels import TreeBcVolCSG, tree_surface_init, tree_spatial_init
    except Exception:
        TreeBcVolCSG = None
        tree_surface_init = None
        tree_spatial_init = None


# Utilities

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


# Patch BcTypePool with experimental modes

def _safe_get_area_scales(pool, i: int) -> torch.Tensor:
    if hasattr(pool, "get_type_area_scale"):
        return pool.get_type_area_scale(i).clamp(min=MIN_SIZE)
    raise AttributeError("Pool has no get_type_area_scale(i). Cannot use tube/blend modes.")


def _safe_get_vol_scales(pool, i: int) -> torch.Tensor:
    if hasattr(pool, "get_type_vol_radius_scale"):
        return pool.get_type_vol_radius_scale(i).clamp(min=MIN_SIZE)
    raise AttributeError("Pool has no get_type_vol_radius_scale(i). Cannot use tube/blend modes.")


def _tube_scales_from_vol_area(pool, i: int, clamp: Tuple[float, float] = (0.03, 3.0)) -> torch.Tensor:
    """rho_T = 3V/A.

    If vol scale is rho_V = (3V/4pi)^(1/3) and area scale is
    rho_A = sqrt(A/4pi), then 3V/A = rho_V^3 / rho_A^2.
    """
    rho_v = _safe_get_vol_scales(pool, i)
    rho_a = _safe_get_area_scales(pool, i)
    rho_t = (rho_v ** 3) / (rho_a ** 2 + 1e-8)
    return rho_t.clamp(min=clamp[0], max=clamp[1])


def _blend_scales(pool, i: int, alpha: float) -> torch.Tensor:
    rho_v = _safe_get_vol_scales(pool, i)
    rho_t = _tube_scales_from_vol_area(pool, i)
    return (rho_v.clamp_min(1e-8) ** (1.0 - alpha) * rho_t.clamp_min(1e-8) ** alpha).clamp(min=MIN_SIZE)


def _get_type_weights(pool, i: int, temperature: float) -> torch.Tensor:
    if hasattr(pool, "get_type_weights_bc"):
        return pool.get_type_weights_bc(i, temperature)
    # fallback for compatible implementations
    return torch.sigmoid(torch.sin(pool.omega * pool.type_logits[i]) * temperature)


def _bc_aggregate_from_sdfs(pool, p: torch.Tensor, i: int, scales: torch.Tensor, temperature: float) -> torch.Tensor:
    sdfs = pool._compute_type_sdfs(p, i)  # expected (N_TYPES, B)
    sharp = pool.sharpness[i].abs().clamp(min=1.0)
    weights = _get_type_weights(pool, i, temperature)
    log_occ = torch.zeros(len(p), device=p.device)
    for k in range(N_TYPES):
        occ_k = torch.sigmoid(-sdfs[k] / scales[k].clamp(min=MIN_SIZE) * sharp)
        log_occ = log_occ + torch.log((1.0 - weights[k] * occ_k).clamp(min=1e-10))
    return 1.0 - torch.exp(log_occ)


def _get_occ_bc_tube_norm(self, p, i, temperature=1.0):
    scales = _tube_scales_from_vol_area(self, i)
    return _bc_aggregate_from_sdfs(self, p, i, scales, temperature)


def _get_occ_bc_blend_norm(self, p, i, temperature=1.0):
    alpha = getattr(self, "calib_blend_alpha", 0.5)
    scales = _blend_scales(self, i, alpha)
    return _bc_aggregate_from_sdfs(self, p, i, scales, temperature)


def _get_occ_bc_jac_static_norm(self, p, i, temperature=1.0):
    base = _safe_get_vol_scales(self, i)
    if not hasattr(self, "jacobian_scale_factors"):
        # No calibration prepared; behave like volume norm but warn once.
        if not getattr(self, "_warned_missing_jac", False):
            print("[WARN] bc_jac_static_norm used before prepare_static_jacobian_scales(); falling back to bc_vol_norm")
            self._warned_missing_jac = True
        scales = base
    else:
        factors = self.jacobian_scale_factors[i].to(base.device).clamp(0.25, 4.0)
        scales = (base * factors).clamp(min=MIN_SIZE)
    return _bc_aggregate_from_sdfs(self, p, i, scales, temperature)


def patch_bc_type_pool() -> None:
    """Monkey-patch the BcTypePool class used by BcVariantCSG / TreeBcCSG.

    bc_area_norm and bc_kj_norm stay with the original dispatcher; this patch
    adds the tube, blend and static-Jacobian modes.
    """
    dummy = BcVariantCSG(num_primitives=1, num_groups=1, occ_mode="bc_vol_norm")
    cls = dummy.pool.__class__

    if getattr(cls, "_calibration_patch_applied", False):
        return

    original_get_all_occs = cls.get_all_occs

    def patched_get_all_occs(self, p, mode="bc", temperature=1.0):
        if mode == "bc_tube_norm":
            fn = _get_occ_bc_tube_norm.__get__(self, cls)
            return torch.stack([fn(p, i, temperature) for i in range(self.N)], dim=-1)
        if mode == "bc_blend_norm":
            fn = _get_occ_bc_blend_norm.__get__(self, cls)
            return torch.stack([fn(p, i, temperature) for i in range(self.N)], dim=-1)
        if mode == "bc_jac_static_norm":
            fn = _get_occ_bc_jac_static_norm.__get__(self, cls)
            return torch.stack([fn(p, i, temperature) for i in range(self.N)], dim=-1)
        return original_get_all_occs(self, p, mode=mode, temperature=temperature)

    cls.get_occ_bc_tube_norm = _get_occ_bc_tube_norm
    cls.get_occ_bc_blend_norm = _get_occ_bc_blend_norm
    cls.get_occ_bc_jac_static_norm = _get_occ_bc_jac_static_norm
    cls.get_all_occs = patched_get_all_occs
    cls._calibration_patch_applied = True


# Static Jacobian calibration for bc_jac_static_norm

def prepare_static_jacobian_scales(model, sample_pts: torch.Tensor, n_calib_pts: int = 512,
                                   temperature: float = 1.0, clamp=(0.25, 4.0)) -> torch.Tensor:
    """Compute detached per-slot/type Jacobian energy factors.

    For each slot i and type k, measure the gradient energy of that type's
    volume-normalized occupancy with respect to the pool parameters. Then use
        factor = sqrt(G_ik / median_G)
    so high-gradient fields get a larger denominator and low-gradient fields
    get a smaller denominator.

    The factors are static and detached (a preconditioning ablation, not a
    second-order method).
    """
    pool = model.pool
    pts = sample_pts[:min(n_calib_pts, len(sample_pts))].to(device)
    factors = torch.ones(pool.N, N_TYPES, device=device)
    energies = torch.zeros(pool.N, N_TYPES, device=device)

    # Keep graph but avoid changing params.
    for i in range(pool.N):
        for k in range(N_TYPES):
            try:
                sdfs = pool._compute_type_sdfs(pts, i)
                vol_scales = _safe_get_vol_scales(pool, i)
                sharp = pool.sharpness[i].abs().clamp(min=1.0)
                occ = torch.sigmoid(-sdfs[k] / vol_scales[k].clamp(min=MIN_SIZE) * sharp)
                # Use mean square occupancy as scalar. Grad wrt all pool params;
                # only slot i/type-k-related params should receive significant grad.
                scalar = (occ ** 2).mean()
                grads = torch.autograd.grad(
                    scalar,
                    [pool.centers, pool.rotations, pool.intrinsic, pool.sharpness],
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )
                e = torch.tensor(0.0, device=device)
                for g in grads:
                    if g is not None:
                        # only slot i row if tensor has first dimension N
                        if g.ndim > 0 and g.shape[0] == pool.N:
                            e = e + (g[i].detach() ** 2).sum()
                        else:
                            e = e + (g.detach() ** 2).sum()
                energies[i, k] = e.clamp_min(1e-12)
            except Exception:
                energies[i, k] = 1.0

    med = torch.median(energies[energies > 0]).clamp_min(1e-12)
    factors = torch.sqrt(energies / med).clamp(min=clamp[0], max=clamp[1]).detach()
    # Store as plain attribute; not a parameter.
    pool.jacobian_scale_factors = factors
    return factors


# Analytic / canonical diagnostic

@dataclass
class PrimitiveSpec:
    name: str
    ptype: str
    params: Tuple[float, ...]
    box_radius: float = 1.2


def sdf_spec(spec: PrimitiveSpec, pts: torch.Tensor) -> torch.Tensor:
    # canonical centered, axis-aligned primitives for diagnostics
    if spec.ptype == "sphere":
        (r,) = spec.params
        return torch.sqrt((pts ** 2).sum(-1) + 1e-8) - r
    if spec.ptype == "box":
        hx, hy, hz = spec.params
        h = torch.tensor([hx, hy, hz], device=pts.device, dtype=pts.dtype)
        d = pts.abs() - h
        return torch.sqrt(d.clamp(min=0).pow(2).sum(-1) + 1e-8) + d.max(-1)[0].clamp(max=0)
    if spec.ptype == "cylinder":
        r, hh = spec.params
        dr = torch.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2 + 1e-8) - r
        dh = pts[:, 1].abs() - hh
        return torch.sqrt(dr.clamp(min=0) ** 2 + dh.clamp(min=0) ** 2 + 1e-8) + torch.maximum(dr, dh).clamp(max=0)
    raise ValueError(spec.ptype)


def exact_volume_area(spec: PrimitiveSpec) -> Tuple[float, float]:
    if spec.ptype == "sphere":
        (r,) = spec.params
        return 4.0 / 3.0 * math.pi * r ** 3, 4.0 * math.pi * r ** 2
    if spec.ptype == "box":
        hx, hy, hz = spec.params
        V = 8.0 * hx * hy * hz
        A = 8.0 * (hx * hy + hx * hz + hy * hz)  # box side lengths are 2h
        return V, A
    if spec.ptype == "cylinder":
        r, hh = spec.params
        h = 2.0 * hh
        V = math.pi * r ** 2 * h
        A = 2.0 * math.pi * r * h + 2.0 * math.pi * r ** 2
        return V, A
    raise ValueError(spec.ptype)


def calib_scale(spec: PrimitiveSpec, mode: str, alpha: float = 0.5) -> float:
    V, A = exact_volume_area(spec)
    rho_v = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)
    rho_a = math.sqrt(A / (4.0 * math.pi))
    rho_t = 3.0 * V / max(A, 1e-8)
    # size scale: geometric mean of bounding half extents / simple fallback
    if spec.ptype == "sphere":
        rho_s = spec.params[0]
    elif spec.ptype == "box":
        hx, hy, hz = spec.params
        rho_s = (hx * hy * hz) ** (1.0 / 3.0)
    elif spec.ptype == "cylinder":
        r, hh = spec.params
        rho_s = (r * r * hh) ** (1.0 / 3.0)
    else:
        rho_s = rho_v
    if mode == "none":
        return 1.0
    if mode == "vol":
        return rho_v
    if mode == "area":
        return rho_a
    if mode == "tube":
        return rho_t
    if mode == "blend":
        return rho_v ** (1.0 - alpha) * rho_t ** alpha
    if mode == "size":
        return rho_s
    raise ValueError(mode)


def finite_diff_param_jac_energy(spec: PrimitiveSpec, pts: torch.Tensor, mode: str, alpha: float, beta: float = 20.0) -> float:
    """Finite-difference sensitivity of occupancy wrt intrinsic parameters."""
    base_scale = calib_scale(spec, mode, alpha)
    phi0 = torch.sigmoid(-sdf_spec(spec, pts) / base_scale * beta)
    params = list(spec.params)
    e_total = 0.0
    for j in range(len(params)):
        eps = max(1e-3, abs(params[j]) * 1e-3)
        pp = params.copy(); pp[j] += eps
        pm = params.copy(); pm[j] = max(1e-4, pm[j] - eps)
        spec_p = PrimitiveSpec(spec.name, spec.ptype, tuple(pp), spec.box_radius)
        spec_m = PrimitiveSpec(spec.name, spec.ptype, tuple(pm), spec.box_radius)
        sp = calib_scale(spec_p, mode, alpha)
        sm = calib_scale(spec_m, mode, alpha)
        phip = torch.sigmoid(-sdf_spec(spec_p, pts) / sp * beta)
        phim = torch.sigmoid(-sdf_spec(spec_m, pts) / sm * beta)
        deriv = (phip - phim) / (2.0 * eps)
        e_total += float((deriv ** 2).mean().detach().cpu().item())
    return e_total


def run_diagnostic(args) -> Dict:
    ensure_dir(args.output)
    specs = [
        PrimitiveSpec("sphere_small", "sphere", (0.20,)),
        PrimitiveSpec("sphere_med", "sphere", (0.35,)),
        PrimitiveSpec("box_cube", "box", (0.28, 0.28, 0.28)),
        PrimitiveSpec("box_plate", "box", (0.45, 0.08, 0.30)),
        PrimitiveSpec("box_rod", "box", (0.08, 0.08, 0.55)),
        PrimitiveSpec("cyl_short", "cylinder", (0.24, 0.20)),
        PrimitiveSpec("cyl_long", "cylinder", (0.09, 0.55)),
    ]
    modes = ["none", "size", "vol", "area", "tube", "blend"]
    alpha = args.blend_alpha
    beta = args.beta

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    pts = (torch.rand(args.diag_points, 3, generator=rng, device=device) * 2.0 - 1.0) * args.diag_box
    pts_for_grad = pts.detach().clone().requires_grad_(True)

    rows = []
    for spec in specs:
        for mode in modes:
            scale = calib_scale(spec, mode, alpha)
            sdf = sdf_spec(spec, pts_for_grad)
            phi = torch.sigmoid(-sdf / scale * beta)
            gx = torch.autograd.grad(phi.sum(), pts_for_grad, retain_graph=True, create_graph=False)[0]
            V, A = exact_volume_area(spec)
            row = {
                "spec": spec.name,
                "ptype": spec.ptype,
                "mode": mode,
                "scale": scale,
                "V": V,
                "A": A,
                "rho_vol": calib_scale(spec, "vol", alpha),
                "rho_tube": calib_scale(spec, "tube", alpha),
                "phi_mean": float(phi.mean().detach().cpu().item()),
                "transition_mean": float((phi * (1.0 - phi)).mean().detach().cpu().item()),
                "spatial_grad_energy": float((gx.norm(dim=-1) ** 2).mean().detach().cpu().item()),
                "param_jac_energy": finite_diff_param_jac_energy(spec, pts.detach(), mode, alpha, beta),
            }
            rows.append(row)

    # Summary coefficient of variation by metric/mode
    summary = []
    for mode in modes:
        rr = [r for r in rows if r["mode"] == mode]
        for metric in ["transition_mean", "spatial_grad_energy", "param_jac_energy"]:
            vals = np.array([r[metric] for r in rr], dtype=np.float64)
            mean = float(vals.mean())
            cv = float(vals.std() / (mean + 1e-12))
            summary.append({"mode": mode, "metric": metric, "mean": mean, "cv": cv})

    out = {"rows": rows, "summary": summary, "alpha": alpha, "beta": beta}
    with open(os.path.join(args.output, "diagnostic.json"), "w") as f:
        json.dump(out, f, indent=2)
    # CSVs for easy spreadsheet plotting
    with open(os.path.join(args.output, "diagnostic_rows.csv"), "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(os.path.join(args.output, "diagnostic_summary.csv"), "w") as f:
        keys = list(summary[0].keys())
        f.write(",".join(keys) + "\n")
        for r in summary:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Diagnostic written to {args.output}/diagnostic_summary.csv")
    print("Lower CV means better cross-shape balance for that metric.")
    for r in summary:
        print(f"  {r['mode']:<6} {r['metric']:<20} CV={r['cv']:.3f}")
    return out


# Synthetic targets

class GTPrimitive:
    def __init__(self, ptype: str, params: Tuple):
        self.ptype = ptype
        self.params = params

    def sdf(self, pts: torch.Tensor) -> torch.Tensor:
        if self.ptype == "sphere":
            center, radius = self.params
            c = torch.tensor(center, dtype=torch.float32, device=pts.device)
            return torch.sqrt(((pts - c) ** 2).sum(-1) + 1e-8) - radius
        if self.ptype == "box":
            center, half_extents = self.params
            c = torch.tensor(center, dtype=torch.float32, device=pts.device)
            h = torch.tensor(half_extents, dtype=torch.float32, device=pts.device)
            d = (pts - c).abs() - h
            return torch.sqrt(d.clamp(min=0).pow(2).sum(-1) + 1e-8) + d.max(-1)[0].clamp(max=0)
        if self.ptype == "cylinder":
            center, (radius, half_height) = self.params
            c = torch.tensor(center, dtype=torch.float32, device=pts.device)
            local = pts - c
            dr = torch.sqrt(local[:, 0] ** 2 + local[:, 2] ** 2 + 1e-8) - radius
            dh = local[:, 1].abs() - half_height
            return torch.sqrt(dr.clamp(min=0) ** 2 + dh.clamp(min=0) ** 2 + 1e-8) + torch.maximum(dr, dh).clamp(max=0)
        raise ValueError(self.ptype)


def build_synthetic_target(case_name: str) -> Tuple[List[GTPrimitive], Callable[[torch.Tensor], torch.Tensor]]:
    if case_name == "sphere_union_box":
        gt = [
            GTPrimitive("sphere", ((0.25, 0.0, 0.0), 0.30)),
            GTPrimitive("box", ((-0.20, 0.0, 0.0), (0.25, 0.20, 0.20))),
        ]
        def target(pts):
            sdfs = torch.stack([g.sdf(pts) for g in gt], 0)
            return (sdfs.min(0)[0] < 0).float()
        return gt, target
    if case_name == "elongated_cylinder":
        gt = [GTPrimitive("cylinder", ((0.0, 0.0, 0.0), (0.09, 0.65)))]
        def target(pts):
            return (gt[0].sdf(pts) < 0).float()
        return gt, target
    if case_name == "thin_box":
        gt = [GTPrimitive("box", ((0.0, 0.0, 0.0), (0.55, 0.06, 0.25)))]
        def target(pts):
            return (gt[0].sdf(pts) < 0).float()
        return gt, target
    if case_name == "five_primitives":
        gt = [
            GTPrimitive("sphere", ((0.40, 0.0, 0.0), 0.22)),
            GTPrimitive("sphere", ((-0.40, 0.0, 0.0), 0.22)),
            GTPrimitive("box", ((0.0, 0.30, 0.0), (0.15, 0.15, 0.15))),
            GTPrimitive("box", ((0.0, -0.30, 0.0), (0.15, 0.15, 0.15))),
            GTPrimitive("cylinder", ((0.0, 0.0, 0.0), (0.10, 0.40))),
        ]
        def target(pts):
            sdfs = torch.stack([g.sdf(pts) for g in gt], 0)
            return (sdfs.min(0)[0] < 0).float()
        return gt, target
    raise ValueError(case_name)


# Training runners

def build_model(kind: str, occ_mode: str, n_primitives: int, n_groups: int, omega: float):
    if kind == "flat":
        return BcVariantCSG(num_primitives=n_primitives, num_groups=n_groups, omega=omega, occ_mode=occ_mode).to(device)
    if kind == "tree":
        if TreeBcCSG is not None:
            return TreeBcCSG(num_primitives=n_primitives, omega=omega, occ_mode=occ_mode).to(device)
        if occ_mode != "bc_vol_norm":
            raise RuntimeError("Your full_benchmark_with_levels.py does not expose TreeBcCSG; custom tree modes unavailable.")
        return TreeBcVolCSG(num_primitives=n_primitives, omega=omega).to(device)
    raise ValueError(kind)


def init_model_if_mesh(model, mesh_sdf, kind: str):
    if mesh_sdf is None:
        return
    if kind == "flat":
        v7_surface_init(model, mesh_sdf)
        v7_spatial_group_init(model, mesh_sdf)
    else:
        if tree_surface_init is not None:
            tree_surface_init(model, mesh_sdf)
        if tree_spatial_init is not None:
            tree_spatial_init(model, mesh_sdf)


def train_one(args, occ_mode: str, target_func, mesh_sdf=None) -> Dict:
    out_dir = os.path.join(args.output, f"{args.model}_{occ_mode}_seed{args.seed}")
    ensure_dir(out_dir)
    set_seed(args.seed)
    model = build_model(args.model, occ_mode, args.n_primitives, args.n_groups, args.omega)
    model.occ_mode = occ_mode  # make sure patched mode string is visible
    if hasattr(model, "pool"):
        model.pool.calib_blend_alpha = args.blend_alpha

    # Static Jacobian factor preparation, if requested.
    if occ_mode == "bc_jac_static_norm" and hasattr(model, "pool"):
        calib_pts = torch.rand(args.jac_calib_points, 3, device=device) * 2.0 - 1.0
        if mesh_sdf is not None:
            try:
                pts_np, _ = mesh_sdf.sample_points(args.jac_calib_points)
                calib_pts = torch.from_numpy(pts_np).float().to(device)
            except Exception:
                pass
        print("Preparing static Jacobian calibration factors...")
        factors = prepare_static_jacobian_scales(model, calib_pts, n_calib_pts=args.jac_calib_points)
        print(f"  jac factors: min={factors.min().item():.3f}, median={factors.median().item():.3f}, max={factors.max().item():.3f}")

    init_model_if_mesh(model, mesh_sdf, args.model)

    reg = dict(
        sparsity_warmup=max(args.epochs // 20, 500 if args.epochs < 5000 else 1500),
        sparsity_rampup=max(args.epochs // 12, 750 if args.epochs < 5000 else 2000),
        ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
        ba_start=args.epochs // 4,
        ba_end=int(args.epochs * 0.75),
        ba_max=5e-4,
    )

    t0 = time.time()
    state, _, _ = train_full(
        model, target_func, mesh_sdf, args.epochs, out_dir,
        num_samples=args.num_samples,
        resolution=args.eval_resolution,
        resume=False,
        desc=f"{args.model}-{occ_mode}",
        use_temperature=True,
        temp_start=1.0,
        temp_end=20.0,
        eval_every=args.eval_every,
        **reg,
    )
    wall = time.time() - t0

    eval_grid = sample_3D_grid(args.eval_resolution)
    if mesh_sdf is not None:
        eval_gt = torch.from_numpy((mesh_sdf.query(eval_grid.cpu().numpy()) < 0).astype(np.float32)).to(device)
    else:
        eval_gt = target_func(eval_grid)

    model.eval()
    with torch.no_grad():
        pred_soft = batched_forward(model, eval_grid, temperature=20.0)
        soft_iou = compute_iou(pred_soft, eval_gt)
        if hasattr(model, "forward_hard_dnf"):
            chunks = [model.forward_hard_dnf(b) for b in torch.split(eval_grid, 32 ** 3)]
            pred_hard = torch.cat(chunks, 0)
            hard_iou = compute_iou(pred_hard, eval_gt)
        else:
            hard_iou = compute_iou((pred_soft > 0.5).float(), eval_gt)

    stats = model.get_stats() if hasattr(model, "get_stats") else {}
    result = {
        "occ_mode": occ_mode,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "wall_s": wall,
        "soft_iou": float(soft_iou),
        "hard_iou": float(hard_iou),
        "gap": float(soft_iou - hard_iou),
        "stats": stats,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"RESULT {occ_mode}: soft={soft_iou:.4f} hard={hard_iou:.4f} gap={soft_iou-hard_iou:.4f} wall={wall:.1f}s")
    return result


def run_training_ablation(args) -> Dict:
    ensure_dir(args.output)
    patch_bc_type_pool()
    if args.mode == "synthetic":
        _, target_func = build_synthetic_target(args.case)
        mesh_sdf = None
    elif args.mode == "mesh":
        mesh_sdf = MeshSDF(args.mesh)
        def target_func(pts):
            return torch.from_numpy((mesh_sdf.query(pts.cpu().numpy()) < 0).astype(np.float32)).to(device)
    else:
        raise ValueError(args.mode)

    results = []
    for occ_mode in args.occ_modes:
        results.append(train_one(args, occ_mode, target_func, mesh_sdf))

    summary = {"results": results, "args": vars(args)}
    with open(os.path.join(args.output, "calibration_ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(args.output, "calibration_ablation_summary.csv"), "w") as f:
        f.write("occ_mode,model,seed,epochs,soft_iou,hard_iou,gap,wall_s\n")
        for r in results:
            f.write(f"{r['occ_mode']},{r['model']},{r['seed']},{r['epochs']},{r['soft_iou']},{r['hard_iou']},{r['gap']},{r['wall_s']}\n")
    print("\nSUMMARY")
    print("occ_mode                 soft     hard      gap     wall")
    for r in results:
        print(f"{r['occ_mode']:<24} {r['soft_iou']:.4f}  {r['hard_iou']:.4f}  {r['gap']:.4f}  {r['wall_s']:.1f}")
    return summary


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Calibration field ablations for typed inverse CSG")
    p.add_argument("--mode", choices=["diagnostic", "synthetic", "mesh"], default="diagnostic")
    p.add_argument("--output", default="./calibration_ablation")
    p.add_argument("--seed", type=int, default=0)

    # Diagnostic args
    p.add_argument("--diag-points", type=int, default=20000)
    p.add_argument("--diag-box", type=float, default=1.2)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--blend-alpha", type=float, default=0.5,
                   help="alpha in rho = rho_vol^(1-alpha) * rho_tube^alpha")

    # Training args
    p.add_argument("--case", default="sphere_union_box",
                   choices=["sphere_union_box", "elongated_cylinder", "thin_box", "five_primitives"])
    p.add_argument("--mesh", default="bunny.obj")
    p.add_argument("--model", choices=["flat", "tree"], default="flat")
    p.add_argument("--occ-modes", nargs="+", default=["bc", "bc_vol_norm", "bc_area_norm", "bc_kj_norm", "bc_tube_norm", "bc_blend_norm"],
                   help="Modes to run. Existing: bc bc_size_norm bc_vol_norm bc_area_norm bc_kj_norm attn_gate. Patched: bc_tube_norm bc_blend_norm bc_jac_static_norm")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--n-primitives", type=int, default=12)
    p.add_argument("--n-groups", type=int, default=8)
    p.add_argument("--omega", type=float, default=10.0)
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--eval-resolution", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--jac-calib-points", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output)
    if args.mode == "diagnostic":
        run_diagnostic(args)
    else:
        run_training_ablation(args)


if __name__ == "__main__":
    main()
