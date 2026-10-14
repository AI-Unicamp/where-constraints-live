#!/usr/bin/env python3
"""
Internal-node responsibility experiment (face-based, with a root-consistency
check); faster variant used for the responsibility_results_fast/ data.

What it tests
-------------
Hypothesis:
    Surface ownership is not piecewise constant at the leaf level,
    but may be piecewise constant at the subtree / branch level.

For a chosen tree level ℓ, define node responsibility on the reconstructed
surface as

    rho_n(x) = | dS(x) / d u_n(x) |

where u_n(x) is the output of internal node n at level ℓ and S(x) is the final
shape occupancy.

Responsibilities are evaluated at face barycenters (not vertices); each face is
assigned an argmax owner, and the script measures:
- connected components on face adjacency
- largest-component coverage per owner
- relative top-1 / top-2 responsibility margin
- seam fraction under quantile thresholds

Design choices
--------------
- internal-node ownership instead of leaf ownership
- face-based patches instead of vertex-based labels
- no random mesh subsampling in the analysis path
- explicit root-consistency check against model._reduce_tree(...)
- comparison across multiple levels by default

Model requirement
-----------------
The model must provide one helper that returns level tensors from leaves upward:

    def _reduce_tree_with_levels(self, leaves, temperature=20.0):
        # leaves: [B, N]
        # return [level0, level1, ..., root]
        # where level0 is the input leaves tensor,
        # level1 has N/2 nodes, ..., root has shape [B, 1]

Example usage
-------------
Default levels:

    python responsibility_field_fast.py \
        --ckpt benchmark_convergence/runs/TN_tree_bc_vol_seed0/best_model.pt

Specific levels:

    python responsibility_field_fast.py \
        --ckpt benchmark_convergence/runs/TN_tree_bc_vol_seed0/best_model.pt \
        --levels 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

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


# Model helpers

def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)


def collect_tree_levels(
    model: torch.nn.Module,
    leaves: torch.Tensor,
    temperature: float,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if not hasattr(model, "_reduce_tree_with_levels"):
        raise RuntimeError(
            "TreeBcVolCSG needs a helper `_reduce_tree_with_levels(leaves, temperature)`\n"
            "that returns [level0, level1, ..., root]."
        )
    levels = model._reduce_tree_with_levels(leaves, temperature=temperature)
    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        raise TypeError("_reduce_tree_with_levels must return a non-empty list/tuple")
    root = levels[-1]
    if root.ndim == 2 and root.shape[1] == 1:
        root = root[:, 0]
    elif root.ndim != 1:
        raise ValueError(f"Root tensor should be [B] or [B,1], got {tuple(root.shape)}")
    return list(levels), root


def validate_tree_levels(
    model: torch.nn.Module,
    temperature: float,
) -> Tuple[int, Dict[int, int]]:
    # Dry run on a few random points.
    pts = torch.rand(4, 3, device=device) * 2 - 1
    leaves = model.pool.get_all_occs(pts, mode=model.occ_mode, temperature=temperature)
    leaves = leaves.detach().requires_grad_(True)

    levels, root = collect_tree_levels(model, leaves, temperature=temperature)
    root_ref = model._reduce_tree(leaves, temperature=temperature)
    if root_ref.ndim > 1:
        root_ref = root_ref.squeeze(-1)

    max_abs_diff = float((root - root_ref).abs().max().item())
    if max_abs_diff > 1e-6:
        raise RuntimeError(
            f"_reduce_tree_with_levels root does not match _reduce_tree. max abs diff = {max_abs_diff:.3e}"
        )

    widths: Dict[int, int] = {}
    prev = None
    for i, t in enumerate(levels):
        if t.ndim != 2:
            raise ValueError(f"Level {i} should have shape [B, W], got {tuple(t.shape)}")
        widths[i] = int(t.shape[1])
        if prev is not None and widths[i] != max(1, prev // 2):
            raise RuntimeError(
                f"Level widths are inconsistent: level {i-1} has {prev}, level {i} has {widths[i]}"
            )
        prev = widths[i]
    return len(levels) - 1, widths


# Surface extraction + face geometry

def extract_surface(
    model: torch.nn.Module,
    resolution: int = 96,
    temperature: float = 20.0,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    grid = sample_3D_grid(resolution)
    with torch.no_grad():
        occ = batched_forward(model, grid, temperature=temperature)
    vol = occ.cpu().numpy().reshape(resolution, resolution, resolution)

    import mcubes
    verts, tris = mcubes.marching_cubes(vol, threshold)
    verts = verts / (resolution - 1) * 2 - 1
    return verts.astype(np.float32), tris.astype(np.int32)


def face_barycenters_and_areas(verts: np.ndarray, tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    bary = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return bary.astype(np.float32), areas.astype(np.float32)


def build_face_adjacency(tris: np.ndarray) -> Dict[int, set[int]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, tri in enumerate(tris):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            e = (u, v) if u < v else (v, u)
            edge_to_faces[e].append(fi)

    adj: Dict[int, set[int]] = defaultdict(set)
    for _, faces in edge_to_faces.items():
        if len(faces) < 2:
            continue
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                f0, f1 = faces[i], faces[j]
                adj[f0].add(f1)
                adj[f1].add(f0)
    return adj


# Responsibility computation

def compute_level_responsibilities(
    model: torch.nn.Module,
    points: torch.Tensor,
    temperature: float,
    levels_to_keep: Sequence[int],
    chunk_size: int,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, Dict[int, int]]:
    model.eval()
    n_points = int(points.shape[0])

    # Width discovery + validation
    max_level, width_by_level = validate_tree_levels(model, temperature=temperature)
    for level in levels_to_keep:
        if level < 0 or level > max_level:
            raise ValueError(f"Requested level {level}, but valid levels are 0..{max_level}")

    rho_by_level = {
        level: torch.zeros(n_points, width_by_level[level], device=device)
        for level in levels_to_keep
    }
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
                root.sum(),
                nodes,
                retain_graph=(idx < len(sorted_levels) - 1),
                create_graph=False,
            )[0]
            rho_by_level[level][start:end] = grad_nodes.detach().abs()

    return rho_by_level, S_all, {level: width_by_level[level] for level in levels_to_keep}


# Face-based analysis

def connected_components_by_label(
    labels: np.ndarray,
    adj: Dict[int, set[int]],
    valid_mask: np.ndarray,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    visited = np.zeros(len(labels), dtype=bool)
    comps_per_label: Dict[int, int] = defaultdict(int)
    largest_per_label: Dict[int, int] = defaultdict(int)

    for start in range(len(labels)):
        if visited[start] or not valid_mask[start]:
            continue
        lbl = int(labels[start])
        comps_per_label[lbl] += 1
        comp_size = 0
        q = deque([start])
        visited[start] = True
        while q:
            cur = q.popleft()
            comp_size += 1
            for nb in adj[cur]:
                if visited[nb] or not valid_mask[nb]:
                    continue
                if int(labels[nb]) == lbl:
                    visited[nb] = True
                    q.append(nb)
        largest_per_label[lbl] = max(largest_per_label[lbl], comp_size)
    return dict(comps_per_label), dict(largest_per_label)


def analyze_face_variant(
    rho_np: np.ndarray,
    face_adj: Dict[int, set[int]],
    face_areas: np.ndarray,
    min_rho_quantile: float = 0.05,
) -> Dict[str, object]:
    sorted_rho = np.sort(rho_np, axis=1)[:, ::-1]
    top1 = sorted_rho[:, 0]
    top2 = sorted_rho[:, 1] if rho_np.shape[1] > 1 else np.zeros_like(top1)
    labels = np.argmax(rho_np, axis=1)

    eps = 1e-12
    raw_margin = top1 - top2
    rel_margin = raw_margin / np.maximum(top1, eps)

    rho_thr = float(np.quantile(top1, min_rho_quantile))
    valid_mask = top1 > rho_thr
    valid_rel = rel_margin[valid_mask]

    comps_per_label, largest_per_label = connected_components_by_label(labels, face_adj, valid_mask)

    area_per_label: Dict[int, float] = defaultdict(float)
    for i, lbl in enumerate(labels):
        if valid_mask[i]:
            area_per_label[int(lbl)] += float(face_areas[i])

    coverage = {}
    # largest_component_per_label is measured in face count (not area).
    label_count = Counter(labels[valid_mask].tolist())
    for lbl, big in largest_per_label.items():
        tot = label_count.get(lbl, 0)
        coverage[int(lbl)] = (big / tot) if tot > 0 else 0.0

    seam_thresholds = {}
    default_thr = 0.0
    default_seam = np.zeros_like(valid_mask, dtype=bool)
    if len(valid_rel) > 0:
        for q in [0.05, 0.10, 0.20, 0.30]:
            thr = float(np.quantile(valid_rel, q))
            seam_mask = (rel_margin < thr) & valid_mask
            seam_thresholds[f"q{int(q * 100)}"] = {
                "threshold": thr,
                "fraction_valid": float(seam_mask[valid_mask].mean()) if valid_mask.any() else 0.0,
                "count": int(seam_mask.sum()),
            }
        default_thr = float(np.quantile(valid_rel, 0.10))
        default_seam = (rel_margin < default_thr) & valid_mask

    avg_comps = float(np.mean(list(comps_per_label.values()))) if comps_per_label else 0.0
    max_comps = int(max(comps_per_label.values())) if comps_per_label else 0
    avg_cov = float(np.mean(list(coverage.values()))) if coverage else 0.0

    return {
        "labels": labels,
        "valid_mask": valid_mask,
        "default_seam": default_seam,
        "rel_margin": rel_margin,
        "stats": {
            "n_faces": int(rho_np.shape[0]),
            "n_valid": int(valid_mask.sum()),
            "n_active_nodes": int(len(comps_per_label)),
            "avg_components_per_label": avg_comps,
            "max_components_per_label": max_comps,
            "avg_largest_component_coverage": avg_cov,
            "rel_margin_median_valid": float(np.median(valid_rel)) if len(valid_rel) else 0.0,
            "default_seam_threshold": default_thr,
            "default_seam_fraction_valid": float(default_seam[valid_mask].mean()) if valid_mask.any() else 0.0,
            "components_per_label": {int(k): int(v) for k, v in comps_per_label.items()},
            "largest_component_per_label": {int(k): int(v) for k, v in largest_per_label.items()},
            "coverage_largest_per_label": {int(k): float(v) for k, v in coverage.items()},
            "area_per_label": {int(k): float(v) for k, v in area_per_label.items()},
            "seam_thresholds": seam_thresholds,
        },
    }


# Visualization

def make_color_map(n_active: int):
    if n_active <= 10:
        return colormaps.get_cmap("tab10")
    if n_active <= 20:
        return colormaps.get_cmap("tab20")
    return colormaps.get_cmap("hsv")


def face_colors(labels: np.ndarray, seam_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    unique_labels = sorted(set(labels[valid_mask].tolist())) if valid_mask.any() else []
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    cmap = make_color_map(max(len(unique_labels), 1))
    invalid_gray = np.array([170, 170, 170], dtype=np.uint8)

    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        if not valid_mask[i]:
            colors[i] = invalid_gray
        elif seam_mask[i]:
            colors[i] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            # even spacing over the colormap
            t = 0.0 if len(unique_labels) <= 1 else label_to_idx[int(lbl)] / max(len(unique_labels) - 1, 1)
            colors[i] = (255 * np.array(cmap(t)[:3])).astype(np.uint8)
    return colors


def save_face_colored_obj(
    path: str,
    verts: np.ndarray,
    tris: np.ndarray,
    labels: np.ndarray,
    seam_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    # OBJ with face colors stored in comments (many viewers ignore them; the
    # labels are also saved in raw.npz and the PNG renders).
    colors = face_colors(labels, seam_mask, valid_mask)
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for i, tri in enumerate(tris):
            c = colors[i]
            f.write(f"# face_color {int(c[0])} {int(c[1])} {int(c[2])}\n")
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


def render_face_barycenters(
    face_pts: np.ndarray,
    labels: np.ndarray,
    seam_mask: np.ndarray,
    valid_mask: np.ndarray,
    rel_margin: np.ndarray,
    out_dir: str,
    suffix: str,
    max_points: int = 30000,
) -> None:
    n = len(face_pts)
    keep = np.arange(n) if n <= max_points else np.sort(np.random.choice(n, max_points, replace=False))
    pts = face_pts[keep]
    labels_k = labels[keep]
    seam_k = seam_mask[keep]
    valid_k = valid_mask[keep]
    colors = face_colors(labels_k, seam_k, valid_k) / 255.0
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
    plt.suptitle(f"Internal-node ownership {suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"face_barycenters_{suffix}.png"), dpi=150)
    plt.close()

    valid_rel = rel_margin[valid_mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(valid_rel):
        ax.hist(valid_rel, bins=80, color="steelblue", edgecolor="black")
        ax.axvline(np.quantile(valid_rel, 0.10), color="red", linestyle="--", label="10th percentile (valid only)")
        ax.legend()
    ax.set_xlabel("Relative margin (top1 - top2) / top1")
    ax.set_ylabel("Number of valid faces")
    ax.set_title(f"Margin distribution {suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"margin_histogram_{suffix}.png"), dpi=150)
    plt.close()


# Main

def print_summary(level: int, width: int, stats: Dict[str, object]) -> None:
    print(f"\n  -- level {level} ({width} nodes) --")
    print(f"    Faces:                   {stats['n_faces']} ({stats['n_valid']} valid)")
    print(f"    Active nodes:            {stats['n_active_nodes']}")
    print(f"    Avg components/node:     {stats['avg_components_per_label']:.2f}")
    print(f"    Max components/node:     {stats['max_components_per_label']}")
    print(f"    Avg largest-comp cover:  {100 * stats['avg_largest_component_coverage']:.1f}%")
    print(f"    Rel margin median:       {stats['rel_margin_median_valid']:.4e}")
    print(f"    Default seam frac:       {100 * stats['default_seam_fraction_valid']:.1f}% of valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bc-prims", type=int, default=100)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-rho-quantile", type=float, default=0.05)
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--render-max-points", type=int, default=30000)
    parser.add_argument("--output", type=str, default="./internal_node_results_correct/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 70}")
    print("  INTERNAL-NODE RESPONSIBILITY EXPERIMENT (CORRECT)")
    print(f"{'=' * 70}")
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  Resolution:  {args.resolution}")
    print(f"  Levels:      {args.levels}")
    print(f"  Chunk size:  {args.chunk_size}")
    print(f"  Output:      {args.output}")
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
        model,
        face_pts_t,
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
        result = analyze_face_variant(
            rho_np,
            face_adj=face_adj,
            face_areas=face_areas,
            min_rho_quantile=args.min_rho_quantile,
        )
        stats = result["stats"]
        print_summary(level, width_by_level[level], stats)

        level_dir = os.path.join(args.output, f"level_{level:02d}")
        os.makedirs(level_dir, exist_ok=True)
        save_face_colored_obj(
            os.path.join(level_dir, "surface_face_colors.obj"),
            verts,
            tris,
            result["labels"],
            result["default_seam"],
            result["valid_mask"],
        )
        render_face_barycenters(
            face_pts,
            result["labels"],
            result["default_seam"],
            result["valid_mask"],
            result["rel_margin"],
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
            labels=result["labels"],
            valid=result["valid_mask"],
            rel_margin=result["rel_margin"],
        )
        all_stats[f"level_{level}"] = stats

    with open(os.path.join(args.output, "all_levels_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n  Done. See {args.output}\n")


if __name__ == "__main__":
    main()
