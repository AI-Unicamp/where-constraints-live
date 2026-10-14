r"""Multi-target 2D dissection: Zhou chain search + multi-assembly BACO.

Zhou & Wang's algorithm extends to dissections of more than two figures;
zhou_multi.cpp implements the chain generalization (refinement sweeps
copy-paste forward and backward along the figure sequence; EM is either the
chain sum over consecutive pairs or, with --em all-pairs, the sum over all
figure pairs; for two figures both reduce to the original).

This script drives the full pipeline and generalizes the exp358
common-boundary optimizer to N assemblies: every piece is one shared loop
living in the frame of shape 0, and each assembly t contributes contact and
exterior constraints mapped through that piece's isometry into assembly t.

On top of exp358 the optimizer adds:
  * per-(piece, assembly>0) SE(2) pose deltas (theta, dx, dy) composed with
    the discrete Zhou isometry, so pieces can rotate/translate continuously
    instead of being pinned to 90-degree poses;
  * an ICP-style outer loop that re-pairs contact correspondences and
    re-projects exterior samples onto the actual target outline (point to
    curve instead of point to frozen pixel);
  * a one-sided signed-distance penalty that pushes samples back inside
    each target silhouette (attacks outside%);
  * an explicit Jacobian sparsity pattern so scipy's trf solver converges
    in seconds instead of stalling on dense finite differences.

Stages (via --stage, default all):
  prepare  equal-area binary masks for the figure list
  build    compile zhou_multi.cpp (MSVC + stb headers; rebuilds on change)
  search   run the chain search (use --search-timeout; detach long runs)
  baco     multi-assembly common-boundary optimization at --pieces
  render   styled SVG/PNG, one panel per assembly

Example (run from dissection/):
  python experiments\exp462_zhou_multitarget_baco_2d.py ^
      --shapes a.png,b.png,c.png --tag bear_cat_fish_k6 ^
      -N 400 -R 100 --max-level 8 --pieces 6
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           distance_transform_edt)
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from _common import ROOT, parser
from lattice2d import (
    PixelIsometry,
    _decode_color_map,
    _find_vcvars64,
    _download_if_missing,
    geometry_match_labels,
    STB_IMAGE_URL,
    STB_IMAGE_WRITE_URL,
)
from exp358_zhou_duncan_common_boundary_optimizer import (
    align_loop,
    contour_loop,
    forward_points,
    inverse_points,
    nearest_indices,
    relation_edges,
    resample_closed,
    write_csv,
)

RESULTS = ROOT / "outputs" / "zhou_multitarget"
EPS = 1e-9
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
           "#b07aa1", "#9c755f"]


# ----------------------------------------------------------------- inputs --
def load_fg(path: Path, *, white_dist: int = 22, fill_holes: bool = True,
            alpha_threshold: int = 128) -> np.ndarray:
    """Load a binary foreground mask.

    PNGs with real transparency use alpha as the silhouette. Opaque images
    use the exp361 distance-from-white rule, which captures pale filled
    silhouettes as well as black masks. Hole filling remains the default for
    outline/fill animal inputs, but wordmarks should pass fill_holes=False so
    counters such as the holes in lowercase e's are preserved.
    """
    arr = np.asarray(Image.open(path).convert("RGBA"))
    alpha = arr[..., 3]
    if np.any(alpha < 255):
        fg = alpha >= alpha_threshold
    else:
        rgb = arr[..., :3].astype(np.int16)
        fg = np.abs(rgb - 255).max(axis=2) > white_dist
    return binary_fill_holes(fg) if fill_holes else fg


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask)


def normalize_mask_area(mask: np.ndarray, target_area: int, size: int) -> np.ndarray:
    """Rescale the silhouette so its pixel count is ~target_area on a size^2
    canvas (same procedure as the thesis Duncan-mask prep)."""
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    scale = math.sqrt(float(target_area) / max(float(mask.sum()), 1.0))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    if max(new_w, new_h) > size - 6:
        s2 = (size - 6) / float(max(new_w, new_h))
        new_w = max(1, int(round(new_w * s2)))
        new_h = max(1, int(round(new_h * s2)))
    img = Image.fromarray(np.where(crop, 255, 0).astype(np.uint8), mode="L")
    img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
    out = np.zeros((size, size), dtype=bool)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    out[y0: y0 + new_h, x0: x0 + new_w] = np.asarray(img, dtype=np.uint8) > 0
    return out


def _neighbor_counts(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask.astype(np.int32), 1)
    return (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
            + pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:])


def set_exact_area(mask: np.ndarray, target: int, *, preserve_holes: bool = False) -> np.ndarray:
    """Trim or grow boundary pixels until the foreground count equals target.

    Growth fills concavities first (ring pixels with the most foreground
    neighbours) and trimming removes the most protruding boundary pixels, so
    the silhouette stays smooth instead of sprouting lexicographic spurs."""
    m = mask.copy()
    hole_guard = (binary_fill_holes(m) & ~m) if preserve_holes else None
    while int(m.sum()) > target:
        pts = np.argwhere(boundary_pixels(m))
        nc = _neighbor_counts(m)
        order = np.lexsort((pts[:, 1], pts[:, 0], nc[pts[:, 0], pts[:, 1]]))
        y, x = pts[order][0]
        m[y, x] = False
    while int(m.sum()) < target:
        ring = binary_dilation(m) & ~m
        if hole_guard is not None:
            ring &= ~hole_guard
        pts = np.argwhere(ring)
        if len(pts) == 0:
            break
        nc = _neighbor_counts(m)
        order = np.lexsort((pts[:, 1], pts[:, 0], -nc[pts[:, 0], pts[:, 1]]))
        y, x = pts[order][0]
        m[y, x] = True
    return m


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L").save(path)


def prepare_inputs(shape_paths: list[Path], names: list[str], out_dir: Path,
                   *, size: int, area: int, fill_holes: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []
    for path, name in zip(shape_paths, names):
        raw = load_fg(path, fill_holes=fill_holes)
        mask = set_exact_area(normalize_mask_area(raw, area, size), area,
                              preserve_holes=not fill_holes)
        out = out_dir / f"{name}.png"
        save_mask(out, mask)
        outs.append(out)
        print(f"prepare: {name} <- {path.name}  area={int(mask.sum())}")
    return outs


# ------------------------------------------------------------------ build --
def build_zhou_multi(build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent / "zhou_multi.cpp"
    src_text = src.read_text(encoding="utf-8")
    cpp = build_dir / "zhou_multi.cpp"
    exe = build_dir / "zhou_multi.exe"
    if exe.exists() and cpp.exists() and cpp.read_text(encoding="utf-8") == src_text:
        return exe
    cpp.write_text(src_text, encoding="utf-8")
    for header, url in (("stb_image.h", STB_IMAGE_URL), ("stb_image_write.h", STB_IMAGE_WRITE_URL)):
        dst = build_dir / header
        if dst.exists():
            continue
        for cand in ROOT.glob("outputs/*/*/zhou_build/" + header):
            dst.write_bytes(cand.read_bytes())
            break
        else:
            _download_if_missing(dst, url)
    vcvars = _find_vcvars64()
    if vcvars is None:
        raise RuntimeError("MSVC vcvars64.bat not found")
    bat = build_dir / "build_zhou_multi.bat"
    bat.write_text(
        "\n".join([
            "@echo off",
            f'call "{vcvars}"',
            f'cl /nologo /std:c++17 /O2 /EHsc /Fe:"{exe}" "{cpp}"',
        ]) + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(["cmd.exe", "/c", str(bat)], cwd=str(build_dir), text=True, capture_output=True)
    (build_dir / "build_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (build_dir / "build_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not exe.exists():
        raise RuntimeError(f"zhou_multi compile failed (rc={proc.returncode}); see {build_dir}\\build_stdout.txt")
    print(f"build: {exe}")
    return exe


# ----------------------------------------------------------------- search --
def run_search(exe: Path, mask_paths: list[Path], out_dir: Path, args) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe)] + [str(p.resolve()) for p in mask_paths] + [
        "-o", str(out_dir.resolve()),
        "-N", str(args.n_trials), "-R", str(args.refinements), "-Nb", str(args.nb),
        "--max-level", str(args.max_level), "--seed", str(args.seed), "-j", str(args.jobs),
        "--em", args.em, "--sweep", args.sweep,
        "--lambda0", str(args.lambda0), "--decay", str(args.decay),
    ]
    print("search:", " ".join(cmd))
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(out_dir), text=True, capture_output=True,
                              timeout=args.search_timeout if args.search_timeout > 0 else None)
        status = "ok" if proc.returncode == 0 else f"nonzero({proc.returncode})"
        (out_dir / "zhou_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
        (out_dir / "zhou_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        (out_dir / "zhou_stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="replace")
        (out_dir / "zhou_stderr.txt").write_text(exc.stderr or "", encoding="utf-8", errors="replace")
    diag = {"search_status": status, "search_runtime_s": time.perf_counter() - started}
    print(f"search: {status} ({diag['search_runtime_s']:.0f}s)")
    return diag


def label_files_for_pieces(zhou_dir: Path, pieces: int, n_shapes: int) -> list[Path] | None:
    for prefix in (f"level_{pieces}pcs_S", f"result_{pieces}pcs_S"):
        files = [zhou_dir / f"{prefix}{t}.png" for t in range(n_shapes)]
        if all(f.exists() for f in files):
            return files
    return None


def latest_label_files(zhou_dir: Path, n_shapes: int) -> tuple[int, list[Path]] | None:
    best: tuple[int, list[Path]] | None = None
    for f in zhou_dir.glob("level_*pcs_S0.png"):
        k = int(f.name.split("_")[1].replace("pcs", ""))
        files = label_files_for_pieces(zhou_dir, k, n_shapes)
        if files and (best is None or k > best[0]):
            best = (k, files)
    return best


# ------------------------------------------------------------------- baco --
def decode_label_png(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    colors = set(map(tuple, img.reshape(-1, 3).tolist()))
    cmap = _decode_color_map([tuple(map(int, c)) for c in colors if tuple(c) != (0, 0, 0)])
    lab = np.zeros(img.shape[:2], dtype=np.int32)
    for color, label in cmap.items():
        lab[np.all(img == np.asarray(color, dtype=np.uint8), axis=2)] = label
    return lab


@dataclass
class ContactGroup:
    """Seam samples between pieces a and b inside assembly t: loop sample
    ai[k] of a should sit at loop sample bi[k] of b (offset by the seam
    clearance along nrm, b's outward normal, when --clearance > 0). nrm is
    frozen per ICP outer iteration so the residual stays a simple offset
    spring with unchanged Jacobian structure."""
    t: int
    a: int
    b: int
    ai: np.ndarray
    bi: np.ndarray
    nrm: np.ndarray | None = None


@dataclass
class ProximityGroup:
    """One-sided non-penetration guards between ANY near pair of pieces in
    assembly t (adjacent or not), rebuilt each ICP outer iteration: sample
    ai[k] of intruder a must stay on the outside of host b's boundary at
    bi[k] (frozen outward normal nrm). This is what keeps junctions and
    non-seam piece pairs from interpenetrating -- the seam hinge never sees
    them, and these are the places the clip pass would otherwise cut."""
    t: int
    a: int
    b: int
    ai: np.ndarray
    bi: np.ndarray
    nrm: np.ndarray


@dataclass
class ExteriorGroup:
    """Loop samples ai of piece a that lie on the silhouette of assembly t;
    targets are their current projections onto that outline (assembly-t
    frame), refreshed every outer iteration."""
    t: int
    a: int
    ai: np.ndarray
    targets: np.ndarray


@dataclass
class MModel:
    labels: list[int]
    offsets: dict[int, int]                       # loop variable offsets
    pose_offsets: dict[tuple[int, int], int]      # (label, t>0) -> offset of (theta, dx, dy)
    n: int                                        # total variables
    samples: int
    n_assemblies: int
    loops: list[dict[int, np.ndarray]]            # per assembly, pulled back to frame 0
    init_loops: dict[int, np.ndarray]
    isos: list[dict[int, PixelIsometry] | None]   # isos[0] is None (identity)
    pose_centers: dict[tuple[int, int], np.ndarray]
    contacts: list[ContactGroup]
    exteriors: list[ExteriorGroup]
    sdf: list[np.ndarray]                         # signed distance per assembly (+ outside)
    rings: list                                   # shapely outline ring per assembly
    target_geoms: list[BaseGeometry]              # full target silhouettes, including components/holes
    prox: list[ProximityGroup]                    # rebuilt by update_correspondences


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float64)


def mask_geometry(mask: np.ndarray) -> BaseGeometry:
    """Exact pixel-cell geometry for a binary silhouette.

    Unlike contour_loop(mask), this preserves disconnected components and
    holes, which matters for wordmarks such as FEEC.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return Polygon()
    geom = unary_union([box(float(x), float(y), float(x + 1), float(y + 1))
                        for y, x in zip(ys, xs)])
    return geom if geom.is_valid else geom.buffer(0)


def bilinear(grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    x = np.clip(pts[:, 0], 0.0, w - 1.001)
    y = np.clip(pts[:, 1], 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    return (grid[y0, x0] * (1 - fx) * (1 - fy) + grid[y0, x0 + 1] * fx * (1 - fy)
            + grid[y0 + 1, x0] * (1 - fx) * fy + grid[y0 + 1, x0 + 1] * fx * fy)


def pose_params(model: MModel, x: np.ndarray, label: int, t: int) -> tuple[float, float, float]:
    off = model.pose_offsets[(label, t)]
    return float(x[off]), float(x[off + 1]), float(x[off + 2])


def map_label_points(model: MModel, x: np.ndarray, t: int, label: int, pts: np.ndarray) -> np.ndarray:
    """Frame-0 points -> assembly-t frame: the fixed discrete Zhou isometry
    composed with an optimizable SE(2) delta about the piece's initial
    centroid in t. Flip stays discrete; theta/dx/dy live in x."""
    if t == 0:
        return pts
    f = forward_points(model.isos[t][label], pts)
    th, dx, dy = pose_params(model, x, label, t)
    c = model.pose_centers[(label, t)]
    ca, sa = math.cos(th), math.sin(th)
    rel = f - c
    rot = np.column_stack([ca * rel[:, 0] - sa * rel[:, 1],
                           sa * rel[:, 0] + ca * rel[:, 1]])
    return rot + c + np.asarray([dx, dy], dtype=np.float64)


def build_multi_model(label_grids: list[np.ndarray], isos_list: list[dict[int, PixelIsometry] | None],
                      labels: list[int], fg_masks: list[np.ndarray], args) -> MModel:
    rng = np.random.default_rng(args.seed)
    n_asm = len(label_grids)
    S = args.samples_per_piece
    loops: list[dict[int, np.ndarray]] = [dict() for _ in range(n_asm)]
    init_loops: dict[int, np.ndarray] = {}
    for label in labels:
        loop0 = resample_closed(contour_loop(label_grids[0] == label), S)
        loops[0][label] = loop0
        acc = args.a_init_weight * loop0
        wsum = args.a_init_weight
        w_t = args.b_init_weight / max(n_asm - 1, 1)
        for t in range(1, n_asm):
            loop_t = resample_closed(contour_loop(label_grids[t] == label), S)
            pulled = align_loop(loop0, inverse_points(isos_list[t][label], loop_t))
            loops[t][label] = pulled
            acc = acc + w_t * pulled
            wsum += w_t
        init_loops[label] = acc / max(wsum, EPS)

    contacts: list[ContactGroup] = []
    exteriors: list[ExteriorGroup] = []
    label_set = set(labels)
    for t in range(n_asm):
        rel = relation_edges(label_grids[t], max_per_kind=args.max_relation_samples, rng=rng)
        for (owner, neighbor), pts in rel.items():
            if owner not in label_set:
                continue
            owner_pts = pts if t == 0 else inverse_points(isos_list[t][owner], pts)
            oi = nearest_indices(init_loops[owner], owner_pts).astype(np.int64)
            if neighbor == 0:
                exteriors.append(ExteriorGroup(t, owner, oi, pts.astype(np.float64).copy()))
            elif neighbor in label_set and owner < neighbor:
                neigh_pts = pts if t == 0 else inverse_points(isos_list[t][neighbor], pts)
                ji = nearest_indices(init_loops[neighbor], neigh_pts).astype(np.int64)
                contacts.append(ContactGroup(t, owner, neighbor, oi, ji))

    offsets: dict[int, int] = {}
    off = 0
    for label in labels:
        offsets[label] = off
        off += S * 2
    pose_offsets: dict[tuple[int, int], int] = {}
    for label in labels:
        for t in range(1, n_asm):
            pose_offsets[(label, t)] = off
            off += 3
    pose_centers = {
        (label, t): forward_points(isos_list[t][label], init_loops[label]).mean(axis=0)
        for label in labels for t in range(1, n_asm)
    }
    sdf = [signed_distance(m) for m in fg_masks]
    target_geoms = [mask_geometry(m) for m in fg_masks]
    rings = [g.boundary for g in target_geoms]
    return MModel(labels, offsets, pose_offsets, off, S, n_asm, loops, init_loops,
                  isos_list, pose_centers, contacts, exteriors, sdf, rings,
                  target_geoms, [])


def pack(model: MModel, loops: dict[int, np.ndarray]) -> np.ndarray:
    x = np.zeros(model.n, dtype=np.float64)
    for label in model.labels:
        off = model.offsets[label]
        x[off: off + len(loops[label]) * 2] = loops[label].reshape(-1)
    return x


def unpack(model: MModel, x: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for label in model.labels:
        off = model.offsets[label]
        out[label] = x[off: off + model.samples * 2].reshape((-1, 2))
    return out


def _mapped_loops(model: MModel, x: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    loops = unpack(model, x)
    out: dict[tuple[int, int], np.ndarray] = {}
    for t in range(model.n_assemblies):
        for label in model.labels:
            out[(t, label)] = map_label_points(model, x, t, label, loops[label])
    return out


def _outward_normals(model: MModel, t: int, dst: int, Q: np.ndarray,
                     di: np.ndarray) -> np.ndarray:
    """Unit outward normals of piece dst's mapped loop at samples di; loop
    orientation reverses under a flipped isometry."""
    S = model.samples
    tang = Q[(di + 1) % S] - Q[(di - 1) % S]
    orient = -1.0 if (t > 0 and model.isos[t][dst].flip) else 1.0
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]]) * orient
    return nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), EPS)


def _nopen_depths(mapped: dict, model: MModel, g: ContactGroup) -> list[np.ndarray]:
    """Penetration depths (px, >=0) of each contact sample into its neighbor
    piece, both directions. Outward normal of the neighbor's boundary is
    estimated from its adjacent samples; loop orientation reverses under a
    flipped isometry, so the normal sign tracks the iso's flip flag."""
    S = model.samples
    out: list[np.ndarray] = []
    for src, si, dst, di in ((g.a, g.ai, g.b, g.bi), (g.b, g.bi, g.a, g.ai)):
        P = mapped[(g.t, src)][si]
        Q = mapped[(g.t, dst)]
        nrm = _outward_normals(model, g.t, dst, Q, di)
        d = np.sum(nrm * (P - Q[di]), axis=1)   # >0 outside neighbor
        out.append(np.maximum(-d, 0.0))
    return out


def residuals(x: np.ndarray, model: MModel, args) -> np.ndarray:
    # NOTE: block order must match jacobian_sparsity() exactly.
    mapped = _mapped_loops(model, x)
    loops = unpack(model, x)
    res: list[np.ndarray] = []
    n_t = max(model.n_assemblies - 1, 1)
    for g in model.contacts:
        w = args.a_contact_weight if g.t == 0 else args.b_contact_weight
        delta = mapped[(g.t, g.a)][g.ai] - mapped[(g.t, g.b)][g.bi]
        if args.clearance > 0.0 and g.nrm is not None:
            delta = delta - args.clearance * g.nrm
        res.append(w * delta)
    for g in model.exteriors:
        w = args.a_exterior_weight if g.t == 0 else args.b_exterior_weight
        res.append(w * (mapped[(g.t, g.a)][g.ai] - g.targets))
    if args.nopen_weight > 0.0:
        for g in model.contacts:
            for depth in _nopen_depths(mapped, model, g):
                res.append(args.nopen_weight * depth)
    if args.prox_weight > 0.0:
        for g in model.prox:
            s = np.sum(g.nrm * (mapped[(g.t, g.a)][g.ai]
                                - mapped[(g.t, g.b)][g.bi]), axis=1)
            res.append(args.prox_weight * np.maximum(-s, 0.0))
    for label in model.labels:
        pts = loops[label]
        init = model.init_loops[label]
        res.append(args.shape_weight * (pts - init))
        res.append(args.a_shape_weight * (pts - model.loops[0][label]))
        for t in range(1, model.n_assemblies):
            res.append((args.b_shape_weight / n_t) * (pts - model.loops[t][label]))
        fair = np.roll(pts, 1, axis=0) - 2.0 * pts + np.roll(pts, -1, axis=0)
        res.append(args.fair_weight * fair)
        edge_len = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        target_len = np.linalg.norm(np.roll(init, -1, axis=0) - init, axis=1)
        res.append(args.edge_weight * (edge_len - target_len))
    if args.sdf_weight > 0.0:
        for t in range(model.n_assemblies):
            for label in model.labels:
                sd = bilinear(model.sdf[t], mapped[(t, label)])
                res.append(args.sdf_weight * np.maximum(sd, 0.0))
    for label in model.labels:
        for t in range(1, model.n_assemblies):
            th, dx, dy = pose_params(model, x, label, t)
            res.append(np.asarray([args.pose_rot_reg * th,
                                   args.pose_trans_reg * dx,
                                   args.pose_trans_reg * dy]))
    return np.concatenate([np.asarray(r, dtype=np.float64).reshape(-1) for r in res])


def jacobian_sparsity(model: MModel, args):
    """Boolean structure of d(residuals)/dx, mirroring residuals() block by
    block. Over-marking is allowed (it only loosens the FD column grouping);
    missing entries are not."""
    S = model.samples
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    r = 0

    def mark(rr: np.ndarray, cc: np.ndarray) -> None:
        rows.append(np.asarray(rr, dtype=np.int64))
        cols.append(np.asarray(cc, dtype=np.int64))

    def mark_pose(rr: np.ndarray, label: int, t: int) -> None:
        off = model.pose_offsets[(label, t)]
        for k in range(3):
            mark(rr, np.full(len(rr), off + k))

    for g in model.contacts:
        m = len(g.ai)
        a_off = model.offsets[g.a] + 2 * g.ai
        b_off = model.offsets[g.b] + 2 * g.bi
        for dr in (0, 1):
            rr = r + 2 * np.arange(m) + dr
            for c in (a_off, a_off + 1, b_off, b_off + 1):
                mark(rr, c)
            if g.t > 0:
                mark_pose(rr, g.a, g.t)
                mark_pose(rr, g.b, g.t)
        r += 2 * m
    for g in model.exteriors:
        m = len(g.ai)
        a_off = model.offsets[g.a] + 2 * g.ai
        for dr in (0, 1):
            rr = r + 2 * np.arange(m) + dr
            for c in (a_off, a_off + 1):
                mark(rr, c)
            if g.t > 0:
                mark_pose(rr, g.a, g.t)
        r += 2 * m
    if args.nopen_weight > 0.0:
        for g in model.contacts:
            m = len(g.ai)
            for src, si, dst, di in ((g.a, g.ai, g.b, g.bi), (g.b, g.bi, g.a, g.ai)):
                rr = r + np.arange(m)
                s_off = model.offsets[src] + 2 * si
                for c in (s_off, s_off + 1):
                    mark(rr, c)
                for nb in (-1, 0, 1):
                    d_off = model.offsets[dst] + 2 * ((di + nb) % S)
                    for dd in (0, 1):
                        mark(rr, d_off + dd)
                if g.t > 0:
                    mark_pose(rr, src, g.t)
                    mark_pose(rr, dst, g.t)
                r += m
    if args.prox_weight > 0.0:
        for g in model.prox:
            m = len(g.ai)
            rr = r + np.arange(m)
            a_off = model.offsets[g.a] + 2 * g.ai
            b_off = model.offsets[g.b] + 2 * g.bi
            for c in (a_off, a_off + 1, b_off, b_off + 1):
                mark(rr, c)
            if g.t > 0:
                mark_pose(rr, g.a, g.t)
                mark_pose(rr, g.b, g.t)
            r += m
    idx = np.arange(S)
    for label in model.labels:
        off = model.offsets[label]
        diag = off + np.arange(2 * S)
        for _ in range(2 + (model.n_assemblies - 1)):   # shape, a_shape, b_shape*(T-1)
            mark(r + np.arange(2 * S), diag)
            r += 2 * S
        for d in (0, 1):                                 # fairness: 3-sample stencil
            rr = r + 2 * idx + d
            for nb in (-1, 0, 1):
                j = (idx + nb) % S
                for dd in (0, 1):
                    mark(rr, off + 2 * j + dd)
        r += 2 * S
        rr = r + idx                                     # edge length: 2-sample stencil
        for nb in (0, 1):
            j = (idx + nb) % S
            for dd in (0, 1):
                mark(rr, off + 2 * j + dd)
        r += S
    if args.sdf_weight > 0.0:
        for t in range(model.n_assemblies):
            for label in model.labels:
                off = model.offsets[label]
                rr = r + idx
                for dd in (0, 1):
                    mark(rr, off + 2 * idx + dd)
                if t > 0:
                    mark_pose(rr, label, t)
                r += S
    for label in model.labels:
        for t in range(1, model.n_assemblies):
            off = model.pose_offsets[(label, t)]
            mark(r + np.arange(3), off + np.arange(3))
            r += 3
    rcat = np.concatenate(rows)
    ccat = np.concatenate(cols)
    return coo_matrix((np.ones(len(rcat), dtype=np.uint8), (rcat, ccat)),
                      shape=(r, model.n)).tocsr()


def update_correspondences(model: MModel, x: np.ndarray, args=None) -> None:
    """ICP outer step: re-pair each contact's neighbor sample by nearest
    mapped position (freezing the seam normal there for the clearance
    offset), re-project each exterior sample onto the actual target outline,
    and rebuild the all-pairs proximity guards."""
    mapped = _mapped_loops(model, x)
    new_contacts = []
    for g in model.contacts:
        bi = cKDTree(mapped[(g.t, g.b)]).query(mapped[(g.t, g.a)][g.ai])[1].astype(np.int64)
        nrm = _outward_normals(model, g.t, g.b, mapped[(g.t, g.b)], bi)
        new_contacts.append(ContactGroup(g.t, g.a, g.b, g.ai, bi, nrm))
    model.contacts = new_contacts
    model.prox = []
    if args is not None and args.prox_weight > 0.0:
        for t in range(model.n_assemblies):
            for a in model.labels:
                P = mapped[(t, a)]
                for b in model.labels:
                    if b == a:
                        continue
                    Q = mapped[(t, b)]
                    d, bi = cKDTree(Q).query(P)
                    sel = np.nonzero(d < args.prox_radius)[0]
                    if sel.size == 0:
                        continue
                    bi = bi[sel].astype(np.int64)
                    nrm = _outward_normals(model, t, b, Q, bi)
                    model.prox.append(ProximityGroup(t, a, b, sel.astype(np.int64),
                                                     bi, nrm))
    new_ext: list[ExteriorGroup] = []
    for g in model.exteriors:
        ring = model.rings[g.t]
        pts = mapped[(g.t, g.a)][g.ai]
        proj = np.asarray([nearest_points(ring, Point(float(p[0]), float(p[1])))[0].coords[0]
                           for p in pts],
                          dtype=np.float64)
        new_ext.append(ExteriorGroup(g.t, g.a, g.ai, proj))
    model.exteriors = new_ext


def contact_rms(x: np.ndarray, model: MModel) -> dict[str, float]:
    mapped = _mapped_loops(model, x)
    sums: dict[int, list[float]] = {t: [] for t in range(model.n_assemblies)}
    for g in model.contacts:
        d = np.linalg.norm(mapped[(g.t, g.a)][g.ai] - mapped[(g.t, g.b)][g.bi], axis=1)
        sums[g.t].extend(d.tolist())
    out: dict[str, float] = {}
    for t, vals in sums.items():
        rms = math.sqrt(sum(v * v for v in vals) / max(len(vals), 1))
        out[f"s{t}_contact_rms_px"] = float(rms)
        out[f"s{t}_contact_count"] = float(len(vals))
    return out


def pose_stats(model: MModel, x: np.ndarray) -> dict[str, float]:
    rot = [abs(pose_params(model, x, label, t)[0])
           for label in model.labels for t in range(1, model.n_assemblies)]
    trans = [math.hypot(*pose_params(model, x, label, t)[1:])
             for label in model.labels for t in range(1, model.n_assemblies)]
    if not rot:
        return {}
    return {
        "max_pose_rot_deg": math.degrees(max(rot)),
        "mean_pose_rot_deg": math.degrees(sum(rot) / len(rot)),
        "max_pose_trans_px": max(trans),
        "mean_pose_trans_px": sum(trans) / len(trans),
    }


def piece_polygon(points_px: np.ndarray) -> BaseGeometry:
    poly = Polygon(points_px)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def inv_map_points(model: MModel, x: np.ndarray, t: int, label: int, pts: np.ndarray) -> np.ndarray:
    """Inverse of map_label_points: assembly-t frame -> frame 0."""
    if t == 0:
        return pts
    th, dx, dy = pose_params(model, x, label, t)
    c = model.pose_centers[(label, t)]
    rel = pts - c - np.asarray([dx, dy], dtype=np.float64)
    ca, sa = math.cos(-th), math.sin(-th)
    rot = np.column_stack([ca * rel[:, 0] - sa * rel[:, 1],
                           sa * rel[:, 0] + ca * rel[:, 1]])
    return inverse_points(model.isos[t][label], rot + c)


def _poly_parts(geom: BaseGeometry) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_poly_parts(g))
        return out
    return []


def _transform_geom(geom: BaseGeometry, fn) -> BaseGeometry:
    """Apply a point-array map to every ring of a (Multi)Polygon."""
    parts = []
    for poly in _poly_parts(geom):
        ext = fn(np.asarray(poly.exterior.coords)[:-1])
        holes = [fn(np.asarray(r.coords)[:-1]) for r in poly.interiors]
        p = Polygon(ext, holes)
        if not p.is_valid:
            p = p.buffer(0)
        parts.append(p)
    return unary_union(parts) if parts else Polygon()


def map_geom(model: MModel, x: np.ndarray, t: int, label: int, geom: BaseGeometry) -> BaseGeometry:
    return _transform_geom(geom, lambda pts: map_label_points(model, x, t, label, pts))


def _drop_crumbs(geom: BaseGeometry, min_component: float) -> BaseGeometry:
    parts = _poly_parts(geom)
    if not parts:
        return Polygon()
    kept = [p for p in parts if p.area >= min_component]
    if not kept:
        kept = [max(parts, key=lambda p: p.area)]
    return unary_union(kept)


def clip_overlaps(model: MModel, x: np.ndarray, geoms: dict[int, BaseGeometry],
                  *, min_component: float = 1.0, buffer_eps: float = 1e-3,
                  ) -> tuple[dict[int, BaseGeometry], dict[str, float]]:
    """Make the piece set overlap-free in every assembly while preserving
    congruence: each overlap region is pulled back to frame 0 through the
    larger piece's rigid map and subtracted from the SHARED geometry, so all
    assemblies keep using one geometry per piece. Material is only removed,
    so no clip can create a new overlap and one sweep terminates with zero
    overlap everywhere; the removed slack reappears as gaps."""
    labels = model.labels
    clipped = {t: 0.0 for t in range(model.n_assemblies)}
    for t in range(model.n_assemblies):
        for ii in range(len(labels)):
            for jj in range(ii + 1, len(labels)):
                a, b = labels[ii], labels[jj]
                ov = map_geom(model, x, t, a, geoms[a]).intersection(
                    map_geom(model, x, t, b, geoms[b]))
                if ov.is_empty or float(ov.area) < 1e-9:
                    continue
                victim = a if float(geoms[a].area) >= float(geoms[b].area) else b
                region0 = _transform_geom(
                    ov.buffer(buffer_eps),
                    lambda pts: inv_map_points(model, x, t, victim, pts))
                geoms[victim] = _drop_crumbs(geoms[victim].difference(region0), min_component)
                clipped[t] += float(ov.area)
    for label in labels:
        n_parts = len(_poly_parts(geoms[label]))
        if n_parts > 1:
            print(f"clip: warning, piece {label} split into {n_parts} components")
    return geoms, {f"s{t}_clipped_overlap_px": v for t, v in clipped.items()}


def grow_into_free(model: MModel, x: np.ndarray, geoms: dict[int, BaseGeometry],
                   fg_masks: list[np.ndarray], *, rounds: int = 3, step: float = 0.8,
                   margin: float = 0.05) -> dict[int, BaseGeometry]:
    """Monotone dual of clip_overlaps: dilate each shared piece into the
    region that is free in EVERY assembly (inside that target silhouette and
    at least `margin` px away from every other piece). Added material is
    inside all targets and disjoint from all neighbors by construction, so
    overlap stays zero and outside does not grow, while clip scars and seam
    pinches refill with buffer-rounded boundaries. Pieces grow smallest
    first so large pieces do not hog the shared free space."""
    sils = model.target_geoms
    labels = model.labels
    for _ in range(max(0, rounds)):
        for label in sorted(labels, key=lambda l: float(geoms[l].area)):
            allowed: BaseGeometry | None = None
            for t in range(model.n_assemblies):
                others = unary_union([map_geom(model, x, t, l2, geoms[l2])
                                      for l2 in labels if l2 != label])
                free_t = sils[t].difference(others.buffer(margin))
                free_0 = free_t if t == 0 else _transform_geom(
                    free_t, lambda pts: inv_map_points(model, x, t, label, pts))
                allowed = free_0 if allowed is None else allowed.intersection(free_0)
            g = geoms[label]
            cand = g.union(g.buffer(step).intersection(allowed))
            if not cand.is_valid:
                cand = cand.buffer(0)
            # keep only growth attached to the piece (no floating islands)
            parts = [p for p in _poly_parts(cand) if p.intersects(g)]
            if parts:
                geoms[label] = unary_union(parts)
    return geoms


def assembly_metrics_geoms(mapped: dict[tuple[int, int], BaseGeometry], labels: list[int],
                           n_asm: int, fg_masks: list[np.ndarray]) -> dict[str, float]:
    """Per-assembly coverage diagnostics, all in pixel coordinates (areas are
    invariant under the y-down convention shared by shape and pieces)."""
    out: dict[str, float] = {}
    for t in range(n_asm):
        shape = mask_geometry(fg_masks[t])
        polys = [mapped[(t, label)] for label in labels]
        union = unary_union(polys)
        shape_area = float(shape.area)
        outside = float(union.difference(shape).area)
        uncovered = float(shape.difference(union).area)
        overlap = 0.0
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                overlap += float(polys[i].intersection(polys[j]).area)
        out[f"s{t}_shape_area"] = shape_area
        out[f"s{t}_outside_pct"] = 100.0 * outside / max(shape_area, EPS)
        out[f"s{t}_uncovered_pct"] = 100.0 * uncovered / max(shape_area, EPS)
        out[f"s{t}_overlap_pct"] = 100.0 * overlap / max(shape_area, EPS)
    return out


# ----------------------------------------------------------------- render --
def _svg_pts(points: np.ndarray, ox: float) -> str:
    return " ".join(f"{p[0] + ox:.2f},{p[1]:.2f}" for p in points)


def _svg_path(points: np.ndarray, ox: float) -> str:
    if len(points) == 0:
        return ""
    head = f"M {points[0, 0] + ox:.2f},{points[0, 1]:.2f}"
    tail = " ".join(f"L {p[0] + ox:.2f},{p[1]:.2f}" for p in points[1:])
    return f"{head} {tail} Z"


def _svg_path_for_geom(geom: BaseGeometry, ox: float) -> str:
    paths: list[str] = []
    for poly in _poly_parts(geom):
        paths.append(_svg_path(np.asarray(poly.exterior.coords)[:-1], ox))
        for ring in poly.interiors:
            paths.append(_svg_path(np.asarray(ring.coords)[:-1], ox))
    return " ".join(p for p in paths if p)


def export_multi_svg(model: MModel, mapped: dict[tuple[int, int], BaseGeometry],
                     fg_masks: list[np.ndarray], names: list[str], path: Path) -> None:
    h, w = fg_masks[0].shape
    gap = 14.0
    total_w = model.n_assemblies * w + (model.n_assemblies - 1) * gap
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{h + 22}" viewBox="0 0 {total_w:.0f} {h + 22}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for t in range(model.n_assemblies):
        ox = t * (w + gap)
        target_path = _svg_path_for_geom(model.target_geoms[t], ox)
        lines.append(f'<path d="{target_path}" fill="#f7f7f7" fill-rule="evenodd" stroke="#333" stroke-width="1.5"/>')
        for i, label in enumerate(model.labels):
            color = PALETTE[i % len(PALETTE)]
            piece_path = _svg_path_for_geom(mapped[(t, label)], ox)
            if piece_path:
                lines.append(f'<path d="{piece_path}" fill="{color}" fill-opacity="0.65" fill-rule="evenodd" stroke="#111" stroke-width="1"/>')
        lines.append(f'<text x="{ox + w / 2:.0f}" y="{h + 16}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">{names[t]}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_geom(ax, geom: BaseGeometry, *, facecolor: str, edgecolor: str,
               linewidth: float, alpha: float, zorder: int) -> None:
    for poly in _poly_parts(geom):
        ext = np.asarray(poly.exterior.coords)[:-1]
        ax.fill(ext[:, 0], ext[:, 1], facecolor=facecolor, edgecolor=edgecolor,
                linewidth=linewidth, alpha=alpha, zorder=zorder)
        for ring in poly.interiors:
            pts = np.asarray(ring.coords)[:-1]
            ax.fill(pts[:, 0], pts[:, 1], facecolor="white", edgecolor=edgecolor,
                    linewidth=linewidth, alpha=1.0, zorder=zorder + 1)


def export_multi_png(model: MModel, mapped: dict[tuple[int, int], BaseGeometry],
                     fg_masks: list[np.ndarray], names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = model.n_assemblies
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), dpi=200)
    if n == 1:
        axes = [axes]
    h, w = fg_masks[0].shape
    for t, ax in enumerate(axes):
        _plot_geom(ax, model.target_geoms[t], facecolor="#f7f7f7",
                   edgecolor="#333333", linewidth=1.4, alpha=1.0, zorder=0)
        for i, label in enumerate(model.labels):
            _plot_geom(ax, mapped[(t, label)], facecolor=PALETTE[i % len(PALETTE)],
                       edgecolor="#111111", linewidth=0.9, alpha=0.65, zorder=2)
        ax.set_title(names[t], fontsize=12)
        ax.set_xlim(-2, w + 2)
        ax.set_ylim(h + 2, -2)  # pixel y points down
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------- main --
def main() -> None:
    p = parser()
    p.add_argument("--shapes", required=True, help="comma-separated silhouette PNGs (>=2)")
    p.add_argument("--names", default="", help="comma-separated short names")
    p.add_argument("--tag", default="exp462_multitarget")
    p.add_argument("--stage", choices=["all", "prepare", "build", "search", "baco", "render"], default="all")
    p.add_argument("--size", type=int, default=215)
    p.add_argument("--area", type=int, default=4800)
    p.add_argument("--fill-holes", type=int, default=1,
                   help="fill enclosed holes in input silhouettes; use 0 for logos/wordmarks")
    p.add_argument("--n-trials", type=int, default=400)
    p.add_argument("--refinements", type=int, default=100)
    p.add_argument("--nb", type=int, default=30)
    p.add_argument("--max-level", type=int, default=8)
    p.add_argument("--seed", type=int, default=2012)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--search-timeout", type=float, default=0.0, help="seconds; 0 = no timeout")
    p.add_argument("--em", choices=["chain", "all-pairs"], default="all-pairs",
                   help="Zhou EM objective over consecutive pairs or all pairs")
    p.add_argument("--sweep", choices=["chain", "cycle"], default="cycle",
                   help="refinement sweep topology; cycle makes the endpoint "
                        "figures adjacent too (no-op for 2 figures)")
    # Area-balance term: level lambda = (lambda0/2) * decay^(level-1). The
    # paper default (1.0, 0.8) decays to ~0.1 by level 8, where a 33-px
    # sliver costs ~1.6 against mismatches of ~1300 - slivers survive. Use
    # e.g. --lambda0 10 --decay 1.0 for constant anti-sliver pressure.
    p.add_argument("--lambda0", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=0.8)
    p.add_argument("--pieces", type=int, default=0, help="K for BACO; 0 = latest level available")
    p.add_argument("--samples-per-piece", type=int, default=96)
    p.add_argument("--max-relation-samples", type=int, default=220)
    p.add_argument("--outer-iters", type=int, default=4,
                   help="ICP outer iterations (re-pair contacts, re-project exteriors)")
    p.add_argument("--baco-suffix", default="",
                   help="suffix for the baco output dir (variant runs side by side)")
    p.add_argument("--a-init-weight", type=float, default=0.65)
    p.add_argument("--b-init-weight", type=float, default=0.35)
    p.add_argument("--a-contact-weight", type=float, default=8.0)
    p.add_argument("--b-contact-weight", type=float, default=14.0)
    p.add_argument("--a-exterior-weight", type=float, default=5.0)
    p.add_argument("--b-exterior-weight", type=float, default=5.0)
    p.add_argument("--shape-weight", type=float, default=0.9)
    # Per-assembly shape pulls: assembly 0 gets a_shape_weight, each other
    # assembly gets b_shape_weight/(T-1). Weights 0.8/0.35 biased the
    # consensus shape toward frame 0, dumping the residual on the last
    # assembly; 0.4/0.8 is symmetric at T=3 (0.4 each).
    p.add_argument("--a-shape-weight", type=float, default=0.4)
    p.add_argument("--b-shape-weight", type=float, default=0.8)
    p.add_argument("--fair-weight", type=float, default=8.0)
    p.add_argument("--edge-weight", type=float, default=0.8)
    p.add_argument("--sdf-weight", type=float, default=2.0,
                   help="one-sided penalty on samples outside the target silhouette")
    p.add_argument("--nopen-weight", type=float, default=20.0,
                   help="one-sided penalty on contact samples penetrating the "
                        "neighboring piece; biases seams to close from the gap "
                        "side so pieces do not overlap")
    # Variational overlap handling: pull seams to a small uniform clearance
    # instead of coincidence (spends the irreducible slack as an even
    # hairline gap rather than holes), and guard ALL near piece pairs --
    # junctions and non-adjacent pieces -- with one-sided hinges so the clip
    # pass has nothing left to cut.
    p.add_argument("--clearance", type=float, default=0.0,
                   help="target seam gap (px) along the neighbor normal; "
                        "0 = pull to coincidence (old behavior)")
    p.add_argument("--prox-weight", type=float, default=0.0,
                   help="one-sided penalty for ANY piece pair within "
                        "--prox-radius (junctions, non-adjacent overlap); "
                        "0 disables")
    p.add_argument("--prox-radius", type=float, default=2.0,
                   help="detection radius (px) for proximity guards")
    p.add_argument("--clip-overlaps", type=int, default=1,
                   help="post-pass that subtracts every residual overlap region "
                        "from the shared piece geometry (congruence-preserving; "
                        "guarantees zero overlap in all assemblies); 0 disables")
    p.add_argument("--clip-min-component", type=float, default=1.0,
                   help="drop clipped-off fragments smaller than this (px^2)")
    p.add_argument("--grow-rounds", type=int, default=3,
                   help="after clipping, rounds of dilating pieces into space "
                        "free in every assembly (fills clip scars and seam "
                        "pinches; overlap stays zero); 0 disables")
    p.add_argument("--grow-step", type=float, default=0.8,
                   help="dilation radius per grow round (px)")
    p.add_argument("--grow-margin", type=float, default=0.05,
                   help="clearance kept from neighboring pieces while growing (px)")
    p.add_argument("--pose-rot-reg", type=float, default=2.0)
    p.add_argument("--pose-trans-reg", type=float, default=0.2)
    p.add_argument("--max-nfev", type=int, default=150, help="per outer iteration")
    p.add_argument("--loss", default="soft_l1")
    p.add_argument("--f-scale", type=float, default=2.0)
    args = p.parse_args()

    shape_paths = [Path(s.strip()) for s in args.shapes.split(",") if s.strip()]
    if len(shape_paths) < 2:
        raise ValueError("need at least two shapes")
    names = ([s.strip() for s in args.names.split(",") if s.strip()]
             if args.names else [p.stem for p in shape_paths])
    if len(names) != len(shape_paths):
        raise ValueError("--names count must match --shapes count")

    out_dir = RESULTS / args.tag
    masks_dir = out_dir / "masks"
    zhou_dir = out_dir / "zhou"
    build_dir = out_dir / "zhou_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    diag: dict[str, object] = {"tag": args.tag, "names": names}

    mask_paths = [masks_dir / f"{n}.png" for n in names]
    if args.stage in ("all", "prepare") and not all(m.exists() for m in mask_paths):
        mask_paths = prepare_inputs(shape_paths, names, masks_dir, size=args.size,
                                    area=args.area, fill_holes=bool(args.fill_holes))
    if args.stage == "prepare":
        return

    if args.stage in ("all", "build", "search"):
        exe = build_zhou_multi(build_dir)  # no-op if source unchanged and exe exists
        if args.stage == "build":
            return
        have_any = latest_label_files(zhou_dir, len(names)) is not None
        if args.stage == "search" or (args.stage == "all" and not have_any):
            diag.update(run_search(exe, mask_paths, zhou_dir, args))
        if args.stage == "search":
            (out_dir / "search_diag.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
            return

    # ---- baco + render ----
    n_shapes = len(names)
    if args.pieces > 0:
        files = label_files_for_pieces(zhou_dir, args.pieces, n_shapes)
        if files is None:
            raise FileNotFoundError(f"no level files for K={args.pieces} in {zhou_dir}")
        k = args.pieces
    else:
        found = latest_label_files(zhou_dir, n_shapes)
        if found is None:
            raise FileNotFoundError(f"no Zhou level outputs in {zhou_dir}")
        k, files = found
    print(f"baco: K={k}, labels from {[f.name for f in files]}")

    grids_raw = [decode_label_png(f) for f in files]
    fg_masks = [load_fg(m, fill_holes=bool(args.fill_holes)) for m in mask_paths]

    isos_list: list[dict[int, PixelIsometry] | None] = [None]
    grids: list[np.ndarray] = [grids_raw[0]]
    match_diags = []
    common = set(int(v) for v in np.unique(grids_raw[0])) - {0}
    for t in range(1, n_shapes):
        lt, isos, mdiag = geometry_match_labels(grids_raw[0], grids_raw[t])
        grids.append(lt)
        isos_list.append(isos)
        match_diags.append(mdiag)
        common &= set(int(v) for v in np.unique(lt)) - {0}
        common &= set(isos)
    labels = sorted(common)
    print(f"baco: {len(labels)} matched pieces across {n_shapes} assemblies")

    baco_dir = out_dir / f"baco_k{k}{args.baco_suffix}"
    baco_dir.mkdir(parents=True, exist_ok=True)

    model = build_multi_model(grids, isos_list, labels, fg_masks, args)
    x = pack(model, model.init_loops)
    start_rms = contact_rms(x, model)
    outer_history: list[dict[str, float]] = []
    total_nfev = 0
    last_status = 0.0
    for outer in range(max(1, args.outer_iters)):
        update_correspondences(model, x, args)
        sp = jacobian_sparsity(model, args)
        n_res = residuals(x, model, args).size
        if sp.shape[0] != n_res:
            raise RuntimeError(f"sparsity rows {sp.shape[0]} != residual rows {n_res}")
        res = least_squares(
            lambda z: residuals(z, model, args),
            x,
            max_nfev=args.max_nfev,
            verbose=0,
            loss=args.loss,
            f_scale=args.f_scale,
            x_scale="jac",
            jac_sparsity=sp,
            method="trf",
        )
        x = res.x
        total_nfev += int(res.nfev)
        last_status = float(res.status)
        entry = {"outer": float(outer), "cost": float(res.cost),
                 "nfev": float(res.nfev), **contact_rms(x, model)}
        outer_history.append(entry)
        print(f"baco outer {outer}: cost={res.cost:.1f} nfev={res.nfev} "
              + " ".join(f"s{t}_rms={entry[f's{t}_contact_rms_px']:.3f}" for t in range(n_shapes)))
    final_rms = contact_rms(x, model)
    poses = pose_stats(model, x)

    loops = unpack(model, x)
    final_geoms: dict[int, BaseGeometry] = {label: piece_polygon(loops[label]) for label in labels}
    clip_diag: dict[str, float] = {}
    if args.clip_overlaps:
        final_geoms, clip_diag = clip_overlaps(
            model, x, final_geoms, min_component=args.clip_min_component)
        print("clip: removed overlap px per assembly "
              + " ".join(f"s{t}={clip_diag[f's{t}_clipped_overlap_px']:.1f}"
                         for t in range(n_shapes)))
        if args.grow_rounds > 0:
            before = sum(float(g.area) for g in final_geoms.values())
            final_geoms = grow_into_free(model, x, final_geoms, fg_masks,
                                         rounds=args.grow_rounds, step=args.grow_step,
                                         margin=args.grow_margin)
            clip_diag["grow_added_px"] = sum(float(g.area) for g in final_geoms.values()) - before
            print(f"grow: refilled {clip_diag['grow_added_px']:.1f} px of free space")
    mapped_geoms = {(t, label): map_geom(model, x, t, label, final_geoms[label])
                    for t in range(n_shapes) for label in labels}
    metrics = assembly_metrics_geoms(mapped_geoms, labels, n_shapes, fg_masks)

    export_multi_svg(model, mapped_geoms, fg_masks, names, baco_dir / "assemblies.svg")
    export_multi_png(model, mapped_geoms, fg_masks, names, baco_dir / "assemblies.png")

    rows = []
    for label in labels:
        poly = final_geoms[label]
        rows.append({"label": label, "area": float(poly.area), "length": float(poly.length)})
    write_csv(baco_dir / "pieces.csv", rows)

    summary = {
        "tag": args.tag,
        "pieces": float(len(labels)),
        "n_assemblies": float(n_shapes),
        "runtime_s": time.perf_counter() - started,
        "outer_iters": float(len(outer_history)),
        "least_squares_cost": outer_history[-1]["cost"] if outer_history else 0.0,
        "least_squares_nfev": float(total_nfev),
        "least_squares_status": last_status,
        **poses,
        **clip_diag,
        **{f"start_{k2}": v for k2, v in start_rms.items()},
        **{f"final_{k2}": v for k2, v in final_rms.items()},
        **metrics,
    }
    write_csv(baco_dir / "summary.csv", [summary])
    payload = {
        "out_dir": str(out_dir),
        "names": names,
        "k": k,
        "label_files": [str(f) for f in files],
        "matching": match_diags,
        "outer_history": outer_history,
        "summary": summary,
        **diag,
    }
    (baco_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"out_dir": str(baco_dir), **summary}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
