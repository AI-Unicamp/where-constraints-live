#!/usr/bin/env python3
"""
Internal-node pair-competition experiment for inverse CSG.

The reported statistics include the effect of the `--min-second-ratio` filter
(how many faces it keeps and removes).

At each requested tree level l, computes node responsibilities

    rho_n(x) = |dS(x) / d u_n(x)|

at face barycenters of the reconstructed surface, takes the top-2 node indices,
and labels each face by the unordered pair (i, j). It then measures coherence of
those pair labels using face adjacency.

Requirements:
- full_benchmark_with_levels.py must define TreeBcVolCSG
- TreeBcVolCSG must inherit _reduce_tree_with_levels(...) from _TreeMixin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import torch

# mcubes fallback
import importlib.util
import types as _types
if importlib.util.find_spec("mcubes") is None:
    from skimage import measure
    mcubes_stub = _types.ModuleType("mcubes")

    def _mc(vol, iso):
        v, f, _, _ = measure.marching_cubes(vol, level=iso)
        return v, f.astype(np.int32)

    def _export(vertices, triangles, filepath):
        with open(filepath, "w", encoding="utf-8") as fh:
            for x, y, z in vertices:
                fh.write(f"v {x} {y} {z}\n")
            for a, b, c in triangles:
                fh.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")

    mcubes_stub.marching_cubes = _mc
    mcubes_stub.export_obj = _export
    sys.modules["mcubes"] = mcubes_stub

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nary_benchmark_v7 import batched_forward, device, sample_3D_grid
from full_benchmark_with_levels import TreeBcVolCSG


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)


def collect_tree_levels(model: torch.nn.Module, leaves: torch.Tensor, temperature: float):
    if not hasattr(model, "_reduce_tree_with_levels"):
        raise RuntimeError("Model needs _reduce_tree_with_levels(leaves, temperature)")
    levels = model._reduce_tree_with_levels(leaves, temperature=temperature)
    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        raise TypeError("_reduce_tree_with_levels must return a non-empty list/tuple")
    root = levels[-1]
    if root.ndim == 2 and root.shape[1] == 1:
        root = root[:, 0]
    elif root.ndim != 1:
        raise ValueError(f"Root tensor should be [B] or [B,1], got {tuple(root.shape)}")
    return list(levels), root


def validate_tree_levels(model: torch.nn.Module, temperature: float):
    pts = torch.rand(4, 3, device=device) * 2 - 1
    leaves = model.pool.get_all_occs(pts, mode=model.occ_mode, temperature=temperature)
    leaves = leaves.detach().requires_grad_(True)

    levels, root = collect_tree_levels(model, leaves, temperature=temperature)
    root_ref = model._reduce_tree(leaves, temperature=temperature)
    if root_ref.ndim > 1:
        root_ref = root_ref.squeeze(-1)

    max_abs_diff = float((root - root_ref).abs().max().item())
    if max_abs_diff > 1e-6:
        raise RuntimeError(f"_reduce_tree_with_levels root mismatch: {max_abs_diff:.3e}")

    widths: Dict[int, int] = {}
    prev = None
    for i, t in enumerate(levels):
        if t.ndim != 2:
            raise ValueError(f"Level {i} should have shape [B, W], got {tuple(t.shape)}")
        widths[i] = int(t.shape[1])
        if prev is not None:
            expected = (prev + 1) // 2
            if widths[i] != expected:
                raise RuntimeError(
                    f"Level widths inconsistent: level {i-1} has {prev}, level {i} has {widths[i]}, expected {expected}"
                )
        prev = widths[i]
    return len(levels) - 1, widths


def extract_surface(model: torch.nn.Module, resolution: int, temperature: float, threshold: float):
    grid = sample_3D_grid(resolution)
    with torch.no_grad():
        occ = batched_forward(model, grid, temperature=temperature)
    vol = occ.cpu().numpy().reshape(resolution, resolution, resolution)
    import mcubes
    verts, tris = mcubes.marching_cubes(vol, threshold)
    verts = verts / (resolution - 1) * 2 - 1
    return verts.astype(np.float32), tris.astype(np.int32)


def face_barycenters_and_areas(verts: np.ndarray, tris: np.ndarray):
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    bary = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return bary.astype(np.float32), areas.astype(np.float32)


def build_face_adjacency(tris: np.ndarray):
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, tri in enumerate(tris):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_faces[e].append(fi)
    adj: Dict[int, set[int]] = defaultdict(set)
    for faces in edge_to_faces.values():
        if len(faces) < 2:
            continue
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                f0, f1 = faces[i], faces[j]
                adj[f0].add(f1)
                adj[f1].add(f0)
    return adj


def compute_level_responsibilities(model, points, temperature, levels_to_keep, chunk_size):
    model.eval()
    n_points = int(points.shape[0])
    max_level, width_by_level = validate_tree_levels(model, temperature=temperature)
    for level in levels_to_keep:
        if level < 0 or level > max_level:
            raise ValueError(f"Requested level {level}, but valid levels are 0..{max_level}")

    rho_by_level = {level: torch.zeros(n_points, width_by_level[level], device=device) for level in levels_to_keep}
    S_all = torch.zeros(n_points, device=device)
    sorted_levels = sorted(levels_to_keep)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        p_chunk = points[start:end]
        leaves = model.pool.get_all_occs(p_chunk, mode=model.occ_mode, temperature=temperature)
        leaves = leaves.detach().requires_grad_(True)
        levels, root = collect_tree_levels(model, leaves, temperature=temperature)
        S_all[start:end] = root.detach()
        for idx, level in enumerate(sorted_levels):
            nodes = levels[level]
            grad_nodes = torch.autograd.grad(
                root.sum(), nodes,
                retain_graph=(idx < len(sorted_levels) - 1),
                create_graph=False,
            )[0]
            rho_by_level[level][start:end] = grad_nodes.detach().abs()

    return rho_by_level, S_all, {level: width_by_level[level] for level in levels_to_keep}


def connected_components_by_label(labels: np.ndarray, adj: Dict[int, set[int]], valid_mask: np.ndarray):
    visited = np.zeros(len(labels), dtype=bool)
    comps_per_label: Dict[int, int] = defaultdict(int)
    largest_count_per_label: Dict[int, int] = defaultdict(int)
    component_faces: Dict[int, List[List[int]]] = defaultdict(list)

    for start in range(len(labels)):
        if visited[start] or not valid_mask[start]:
            continue
        lbl = int(labels[start])
        comps_per_label[lbl] += 1
        comp = []
        q = deque([start])
        visited[start] = True
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if visited[nb] or not valid_mask[nb]:
                    continue
                if int(labels[nb]) == lbl:
                    visited[nb] = True
                    q.append(nb)
        component_faces[lbl].append(comp)
        largest_count_per_label[lbl] = max(largest_count_per_label[lbl], len(comp))

    return dict(comps_per_label), dict(largest_count_per_label), component_faces


def analyze_pair_variant(
    rho_np: np.ndarray,
    face_adj,
    face_areas: np.ndarray,
    min_rho_quantile: float,
    min_second_ratio: float,
):
    order = np.argsort(rho_np, axis=1)[:, ::-1]
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if rho_np.shape[1] > 1 else order[:, 0]

    sorted_rho = np.take_along_axis(rho_np, order, axis=1)
    top1 = sorted_rho[:, 0]
    top2 = sorted_rho[:, 1] if rho_np.shape[1] > 1 else np.zeros_like(top1)
    top3 = sorted_rho[:, 2] if rho_np.shape[1] > 2 else np.zeros_like(top1)

    # Unordered pair label
    a = np.minimum(top1_idx, top2_idx)
    b = np.maximum(top1_idx, top2_idx)
    pair_codes = a.astype(np.int64) * int(rho_np.shape[1]) + b.astype(np.int64)

    eps = 1e-12
    top1_thr = float(np.quantile(top1, min_rho_quantile))
    second_ratio = top2 / np.maximum(top1, eps)
    valid_mask = (top1 > top1_thr) & (second_ratio >= min_second_ratio)

    pair_stability = (top2 - top3) / np.maximum(top1, eps)
    valid_pair_stability = pair_stability[valid_mask]

    comps_per_label, largest_count_per_label, component_faces = connected_components_by_label(pair_codes, face_adj, valid_mask)

    area_per_label: Dict[int, float] = defaultdict(float)
    for i, lbl in enumerate(pair_codes):
        if valid_mask[i]:
            area_per_label[int(lbl)] += float(face_areas[i])

    largest_area_per_label: Dict[int, float] = defaultdict(float)
    for lbl, comps in component_faces.items():
        best = 0.0
        for comp in comps:
            area = float(face_areas[np.array(comp, dtype=np.int64)].sum())
            if area > best:
                best = area
        largest_area_per_label[int(lbl)] = best

    coverage_count = {}
    coverage_area = {}
    label_count_total: Dict[int, int] = defaultdict(int)
    for lbl in pair_codes[valid_mask]:
        label_count_total[int(lbl)] += 1
    for lbl, big in largest_count_per_label.items():
        tot = label_count_total.get(int(lbl), 0)
        coverage_count[int(lbl)] = (big / tot) if tot > 0 else 0.0
    for lbl, big_area in largest_area_per_label.items():
        tot_area = area_per_label.get(int(lbl), 0.0)
        coverage_area[int(lbl)] = (big_area / tot_area) if tot_area > 0 else 0.0

    seam_thresholds = {}
    default_thr = 0.0
    default_seam = np.zeros_like(valid_mask, dtype=bool)
    if len(valid_pair_stability) > 0:
        for q in [0.05, 0.10, 0.20, 0.30]:
            thr = float(np.quantile(valid_pair_stability, q))
            seam_mask = (pair_stability < thr) & valid_mask
            seam_thresholds[f"q{int(q * 100)}"] = {
                "threshold": thr,
                "fraction_valid": float(seam_mask[valid_mask].mean()) if valid_mask.any() else 0.0,
                "count": int(seam_mask.sum()),
            }
        default_thr = float(np.quantile(valid_pair_stability, 0.10))
        default_seam = (pair_stability < default_thr) & valid_mask

    avg_comps = float(np.mean(list(comps_per_label.values()))) if comps_per_label else 0.0
    max_comps = int(max(comps_per_label.values())) if comps_per_label else 0
    avg_cov_count = float(np.mean(list(coverage_count.values()))) if coverage_count else 0.0
    avg_cov_area = float(np.mean(list(coverage_area.values()))) if coverage_area else 0.0

    pair_names = {}
    for code in comps_per_label.keys():
        i = int(code) // int(rho_np.shape[1])
        j = int(code) % int(rho_np.shape[1])
        pair_names[int(code)] = [int(i), int(j)]

    return {
        "pair_codes": pair_codes,
        "valid_mask": valid_mask,
        "default_seam": default_seam,
        "pair_stability": pair_stability,
        "second_ratio": second_ratio,
        "stats": {
            "n_faces": int(rho_np.shape[0]),
            "n_valid": int(valid_mask.sum()),
            "valid_fraction": float(valid_mask.mean()),
            "top1_threshold": float(top1_thr),
            "min_second_ratio": float(min_second_ratio),
            "second_ratio_median_all": float(np.median(second_ratio)),
            "second_ratio_median_valid": float(np.median(second_ratio[valid_mask])) if valid_mask.any() else 0.0,
            "n_active_pairs": int(len(comps_per_label)),
            "avg_components_per_pair": avg_comps,
            "max_components_per_pair": max_comps,
            "avg_largest_component_coverage_count": avg_cov_count,
            "avg_largest_component_coverage_area": avg_cov_area,
            "pair_stability_median_valid": float(np.median(valid_pair_stability)) if len(valid_pair_stability) else 0.0,
            "default_seam_threshold": default_thr,
            "default_seam_fraction_valid": float(default_seam[valid_mask].mean()) if valid_mask.any() else 0.0,
            "components_per_pair": {int(k): int(v) for k, v in comps_per_label.items()},
            "largest_component_per_pair_count": {int(k): int(v) for k, v in largest_count_per_label.items()},
            "coverage_largest_per_pair_count": {int(k): float(v) for k, v in coverage_count.items()},
            "coverage_largest_per_pair_area": {int(k): float(v) for k, v in coverage_area.items()},
            "area_per_pair": {int(k): float(v) for k, v in area_per_label.items()},
            "pair_names": pair_names,
            "seam_thresholds": seam_thresholds,
        },
    }


def make_color_map(n_active: int):
    if n_active <= 10:
        return colormaps.get_cmap("tab10")
    if n_active <= 20:
        return colormaps.get_cmap("tab20")
    return colormaps.get_cmap("hsv")


def pair_colors(pair_codes: np.ndarray, seam_mask: np.ndarray, valid_mask: np.ndarray):
    unique_codes = sorted(set(pair_codes[valid_mask].tolist())) if valid_mask.any() else []
    code_to_idx = {code: i for i, code in enumerate(unique_codes)}
    cmap = make_color_map(max(len(unique_codes), 1))
    invalid_gray = np.array([170, 170, 170], dtype=np.uint8)
    colors = np.zeros((len(pair_codes), 3), dtype=np.uint8)
    for i, code in enumerate(pair_codes):
        if not valid_mask[i]:
            colors[i] = invalid_gray
        elif seam_mask[i]:
            colors[i] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            t = 0.0 if len(unique_codes) <= 1 else code_to_idx[int(code)] / max(len(unique_codes) - 1, 1)
            colors[i] = (255 * np.array(cmap(t)[:3])).astype(np.uint8)
    return colors


def render_face_barycenters(face_pts, pair_codes, seam_mask, valid_mask, pair_stability, out_dir, suffix, max_points=30000):
    n = len(face_pts)
    keep = np.arange(n) if n <= max_points else np.sort(np.random.choice(n, max_points, replace=False))
    pts = face_pts[keep]
    colors = pair_colors(pair_codes[keep], seam_mask[keep], valid_mask[keep]) / 255.0
    colors_rgba = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)

    fig = plt.figure(figsize=(16, 12))
    views = [(20, 30), (20, 120), (20, 210), (20, 300)]
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors_rgba, s=2, alpha=0.9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view {i}")
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
    plt.suptitle(f"Pair competition {suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pair_barycenters_{suffix}.png"), dpi=150)
    plt.close()

    valid_stab = pair_stability[valid_mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(valid_stab):
        ax.hist(valid_stab, bins=80, color="steelblue", edgecolor="black")
        ax.axvline(np.quantile(valid_stab, 0.10), color="red", linestyle="--", label="10th percentile (valid only)")
        ax.legend()
    ax.set_xlabel("Pair-stability margin (top2 - top3) / top1")
    ax.set_ylabel("Number of valid faces")
    ax.set_title(f"Pair-stability distribution {suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pair_stability_{suffix}.png"), dpi=150)
    plt.close()


def print_summary(level: int, width: int, stats: Dict[str, object]):
    print(f"\n  -- level {level} ({width} nodes) --")
    print(f"    Faces:                         {stats['n_faces']} ({stats['n_valid']} valid, {100*stats['valid_fraction']:.1f}% kept)")
    print(f"    Min second ratio:              {stats['min_second_ratio']:.3f}")
    print(f"    Second-ratio median (all):     {stats['second_ratio_median_all']:.4f}")
    print(f"    Second-ratio median (valid):   {stats['second_ratio_median_valid']:.4f}")
    print(f"    Active pairs:                  {stats['n_active_pairs']}")
    print(f"    Avg components/pair:           {stats['avg_components_per_pair']:.2f}")
    print(f"    Max components/pair:           {stats['max_components_per_pair']}")
    print(f"    Avg largest-comp cover count:  {100 * stats['avg_largest_component_coverage_count']:.1f}%")
    print(f"    Avg largest-comp cover area:   {100 * stats['avg_largest_component_coverage_area']:.1f}%")
    print(f"    Pair-stability median:         {stats['pair_stability_median_valid']:.4e}")
    print(f"    Default seam frac:             {100 * stats['default_seam_fraction_valid']:.1f}% of valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bc-prims", type=int, default=100)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-rho-quantile", type=float, default=0.05)
    parser.add_argument("--min-second-ratio", type=float, default=0.0)
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--render-max-points", type=int, default=30000)
    parser.add_argument("--output", type=str, default="./internal_node_pair_competition_fixed/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 70}")
    print("  INTERNAL-NODE PAIR-COMPETITION EXPERIMENT (FIXED)")
    print(f"{'=' * 70}")
    print(f"  Checkpoint:         {args.ckpt}")
    print(f"  Resolution:         {args.resolution}")
    print(f"  Levels:             {args.levels}")
    print(f"  Min second ratio:   {args.min_second_ratio}")
    print(f"  Chunk size:         {args.chunk_size}")
    print(f"  Output:             {args.output}")
    print(f"{'=' * 70}\n")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = TreeBcVolCSG(num_primitives=args.bc_prims, omega=args.omega).to(device)
    load_checkpoint(model, args.ckpt)
    print("  Loaded checkpoint")

    print(f"\n  Extracting reconstructed surface at {args.resolution}^3...")
    verts, tris = extract_surface(model, resolution=args.resolution, temperature=args.temperature, threshold=args.threshold)
    print(f"  Mesh: {len(verts)} vertices, {len(tris)} triangles")

    import mcubes
    mcubes.export_obj(verts, tris, os.path.join(args.output, "surface.obj"))

    face_pts, face_areas = face_barycenters_and_areas(verts, tris)
    face_adj = build_face_adjacency(tris)
    face_pts_t = torch.from_numpy(face_pts).to(device)

    print("\n  Computing internal-node responsibilities at face barycenters...")
    rho_by_level, S_t, width_by_level = compute_level_responsibilities(
        model, face_pts_t,
        temperature=args.temperature,
        levels_to_keep=args.levels,
        chunk_size=args.chunk_size,
    )
    S = S_t.cpu().numpy()
    print(f"  S range: [{S.min():.3f}, {S.max():.3f}]")
    print(f"  S mean:  {S.mean():.3f}")

    all_stats = {}
    for level in sorted(rho_by_level.keys()):
        rho_np = rho_by_level[level].cpu().numpy()
        result = analyze_pair_variant(
            rho_np,
            face_adj=face_adj,
            face_areas=face_areas,
            min_rho_quantile=args.min_rho_quantile,
            min_second_ratio=args.min_second_ratio,
        )
        stats = result["stats"]
        print_summary(level, width_by_level[level], stats)

        level_dir = os.path.join(args.output, f"level_{level:02d}")
        os.makedirs(level_dir, exist_ok=True)
        render_face_barycenters(
            face_pts,
            result["pair_codes"],
            result["default_seam"],
            result["valid_mask"],
            result["pair_stability"],
            level_dir,
            suffix=f"level_{level:02d}",
            max_points=args.render_max_points,
        )
        with open(os.path.join(level_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        np.savez(
            os.path.join(level_dir, "raw.npz"),
            verts=verts,
            tris=tris,
            face_points=face_pts,
            face_areas=face_areas,
            S=S,
            rho=rho_np,
            pair_codes=result["pair_codes"],
            valid=result["valid_mask"],
            pair_stability=result["pair_stability"],
            second_ratio=result["second_ratio"],
        )
        all_stats[f"level_{level}"] = stats

    with open(os.path.join(args.output, "all_levels_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n  Done. See {args.output}\n")


if __name__ == "__main__":
    main()
