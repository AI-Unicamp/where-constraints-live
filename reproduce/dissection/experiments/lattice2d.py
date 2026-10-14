from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import GeometryCollection, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from _common import ROOT, parser
from shape_benchmarks import load_benchmark
from src.continuous_parametric_solvers import Rigid2
from src.shapely_quantile_solvers import apply_rigid, clean_geom


RESULTS = ROOT / "outputs" / "zhou_seeded_vector_solve"
EPS = 1e-9
STB_IMAGE_URL = "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
STB_IMAGE_WRITE_URL = "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"


@dataclass(frozen=True)
class RasterMeta:
    width: int
    height: int
    minx: float
    miny: float
    maxx: float
    maxy: float
    scale: float


@dataclass(frozen=True)
class PixelIsometry:
    label: int
    rot: int
    flip: bool
    dy: int
    dx: int
    y0: int
    x0: int
    h: int
    w: int
    mismatch: int
    source_pixels: int
    target_pixels: int

    @property
    def name(self) -> str:
        f = "F" if self.flip else "N"
        return f"zhou_l{self.label}_r{self.rot}_{f}_dy{self.dy}_dx{self.dx}"


@dataclass(frozen=True)
class TransformSpec:
    name: str
    kind: str
    label: int
    pixel: PixelIsometry | None = None
    rigid: Rigid2 | None = None
    score_hint: float = 0.0


def _count_mask(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _shape_bounds(source: BaseGeometry, target: BaseGeometry, pad_frac: float) -> tuple[float, float, float, float]:
    minx = min(float(source.bounds[0]), float(target.bounds[0]))
    miny = min(float(source.bounds[1]), float(target.bounds[1]))
    maxx = max(float(source.bounds[2]), float(target.bounds[2]))
    maxy = max(float(source.bounds[3]), float(target.bounds[3]))
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    pad = pad_frac * max(dx, dy)
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def _raster_meta(source: BaseGeometry, target: BaseGeometry, resolution: int, pad_frac: float, *, square_size: int = 0) -> RasterMeta:
    minx, miny, maxx, maxy = _shape_bounds(source, target, pad_frac)
    if square_size > 0:
        span = max(maxx - minx, maxy - miny)
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        scale = span / float(square_size)
        half = 0.5 * square_size * scale
        minx, maxx = cx - half, cx + half
        miny, maxy = cy - half, cy + half
        width = height = square_size
    else:
        scale = max(maxx - minx, maxy - miny) / float(resolution)
        width = max(1, int(math.ceil((maxx - minx) / scale)))
        height = max(1, int(math.ceil((maxy - miny) / scale)))
        maxx = minx + width * scale
        maxy = miny + height * scale
    return RasterMeta(width, height, minx, miny, maxx, maxy, scale)


def _rasterize_geom(geom: BaseGeometry, meta: RasterMeta) -> np.ndarray:
    mask = np.zeros((meta.height, meta.width), dtype=bool)
    prepared = geom
    for y in range(meta.height):
        wy = meta.maxy - (y + 0.5) * meta.scale
        for x in range(meta.width):
            wx = meta.minx + (x + 0.5) * meta.scale
            if prepared.covers(Point(wx, wy)):
                mask[y, x] = True
    return mask


def _boundary_rank(mask: np.ndarray, geom: BaseGeometry, meta: RasterMeta, *, foreground: bool) -> list[tuple[float, int, int]]:
    pts: list[tuple[float, int, int]] = []
    boundary = geom.boundary
    ys, xs = np.nonzero(mask if foreground else ~mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        wx = meta.minx + (x + 0.5) * meta.scale
        wy = meta.maxy - (y + 0.5) * meta.scale
        pts.append((float(boundary.distance(Point(wx, wy))), y, x))
    pts.sort(key=lambda item: item[0])
    return pts


def _equalize_masks(a: np.ndarray, b: np.ndarray, source: BaseGeometry, target: BaseGeometry, meta: RasterMeta) -> dict[str, int]:
    ca, cb = _count_mask(a), _count_mask(b)
    diag = {"source_pixels_initial": ca, "target_pixels_initial": cb, "source_removed": 0, "target_removed": 0, "source_added": 0, "target_added": 0}
    if ca == cb:
        return diag
    if ca > cb:
        for _dist, y, x in _boundary_rank(a, source, meta, foreground=True)[: ca - cb]:
            a[y, x] = False
            diag["source_removed"] += 1
    else:
        for _dist, y, x in _boundary_rank(b, target, meta, foreground=True)[: cb - ca]:
            b[y, x] = False
            diag["target_removed"] += 1
    diag["source_pixels_final"] = _count_mask(a)
    diag["target_pixels_final"] = _count_mask(b)
    return diag


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    arr = np.where(mask, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def prepare_zhou_inputs(
    source: BaseGeometry,
    target: BaseGeometry,
    out_dir: Path,
    *,
    resolution: int,
    pad_frac: float,
    square_size: int = 0,
) -> tuple[Path, Path, RasterMeta, dict[str, int]]:
    meta = _raster_meta(source, target, resolution, pad_frac, square_size=square_size)
    a = _rasterize_geom(source, meta)
    b = _rasterize_geom(target, meta)
    diag = _equalize_masks(a, b, source, target, meta)
    a_path = out_dir / "zhou_input_A.png"
    b_path = out_dir / "zhou_input_B.png"
    _write_binary_png(a_path, a)
    _write_binary_png(b_path, b)
    (out_dir / "raster_meta.json").write_text(json.dumps({**meta.__dict__, **diag}, indent=2, sort_keys=True), encoding="utf-8")
    return a_path, b_path, meta, diag


def load_png_shapes(a_path: Path, b_path: Path, *, threshold: int = 128) -> tuple[BaseGeometry, BaseGeometry, RasterMeta, dict[str, int]]:
    a = np.asarray(Image.open(a_path).convert("L")) < threshold
    b = np.asarray(Image.open(b_path).convert("L")) < threshold
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    aa = np.zeros((h, w), dtype=bool)
    bb = np.zeros((h, w), dtype=bool)
    aa[: a.shape[0], : a.shape[1]] = a
    bb[: b.shape[0], : b.shape[1]] = b
    meta = RasterMeta(w, h, 0.0, 0.0, float(w), float(h), 1.0)
    source = _mask_to_geometry(aa, meta, min_area=0.0)
    target = _mask_to_geometry(bb, meta, min_area=0.0)
    diag = {"source_pixels_initial": _count_mask(aa), "target_pixels_initial": _count_mask(bb), "source_pixels_final": _count_mask(aa), "target_pixels_final": _count_mask(bb)}
    return source, target, meta, diag


def _find_vcvars64() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def _instrument_zhou_source_for_level_outputs(text: str) -> tuple[str, dict[str, object]]:
    """Keep Zhou's search unchanged, but save the best candidate at each level.

    The stock C++ program saves `best_overall`, which can be fewer than the
    requested cap. For a K=6 experiment we need the level-5 / 6-cluster labels.
    """
    diag = {"zhou_level_output_instrumented": 0}
    l1_anchor = '              << "  Split=" << std::setprecision(0) << obj1.split << std::endl;\n\n    if (obj1.mismatch'
    l1_insert = (
        '              << "  Split=" << std::setprecision(0) << obj1.split << std::endl;\n\n'
        '    try {\n'
        '        std::filesystem::create_directories(args.outdir);\n'
        '        save_labels_png(args.outdir + "/level_" + std::to_string(pcs) + "pcs_A.png", best.LA);\n'
        '        save_labels_png(args.outdir + "/level_" + std::to_string(pcs) + "pcs_B.png", best.LB);\n'
        '    } catch (const std::exception& e) {\n'
        '        std::cerr << "Save level error: " << e.what() << std::endl;\n'
        '    }\n\n'
        '    if (obj1.mismatch'
    )
    if l1_anchor in text:
        text = text.replace(l1_anchor, l1_insert, 1)
        diag["zhou_level_output_instrumented"] = int(diag["zhou_level_output_instrumented"]) + 1

    loop_anchor = '                  << "  Split=" << std::setprecision(0) << obj.split << std::endl;\n\n        if (obj.mismatch'
    loop_insert = (
        '                  << "  Split=" << std::setprecision(0) << obj.split << std::endl;\n\n'
        '        try {\n'
        '            std::filesystem::create_directories(args.outdir);\n'
        '            save_labels_png(args.outdir + "/level_" + std::to_string(pcs_cur) + "pcs_A.png", cur.LA);\n'
        '            save_labels_png(args.outdir + "/level_" + std::to_string(pcs_cur) + "pcs_B.png", cur.LB);\n'
        '        } catch (const std::exception& e) {\n'
        '            std::cerr << "Save level error: " << e.what() << std::endl;\n'
        '        }\n\n'
        '        if (obj.mismatch'
    )
    if loop_anchor in text:
        text = text.replace(loop_anchor, loop_insert, 1)
        diag["zhou_level_output_instrumented"] = int(diag["zhou_level_output_instrumented"]) + 1
    return text, diag


def build_zhou_binary(source_path: Path, build_dir: Path, *, download_stb: bool) -> tuple[Path | None, dict[str, object]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = build_dir / "zhou_faithful.cpp"
    if not source_path.exists():
        return None, {"build_status": "missing_source", "source_path": str(source_path)}
    source_text, instrument_diag = _instrument_zhou_source_for_level_outputs(source_path.read_text(encoding="utf-8", errors="replace"))
    cpp_path.write_text(source_text, encoding="utf-8")
    if download_stb:
        _download_if_missing(build_dir / "stb_image.h", STB_IMAGE_URL)
        _download_if_missing(build_dir / "stb_image_write.h", STB_IMAGE_WRITE_URL)
    missing_headers = [name for name in ("stb_image.h", "stb_image_write.h") if not (build_dir / name).exists()]
    if missing_headers:
        return None, {"build_status": "missing_stb_headers", "missing_headers": ",".join(missing_headers), "build_dir": str(build_dir), **instrument_diag}
    vcvars = _find_vcvars64()
    if vcvars is None:
        return None, {"build_status": "missing_msvc_vcvars", **instrument_diag}
    exe = build_dir / "zhou_faithful.exe"
    bat = build_dir / "build_zhou.bat"
    bat.write_text(
        "\n".join(
            [
                "@echo off",
                f'call "{vcvars}"',
                f'cl /nologo /std:c++17 /O2 /EHsc /Fe:"{exe}" "{cpp_path}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(["cmd.exe", "/c", str(bat)], cwd=str(build_dir), text=True, capture_output=True)
    (build_dir / "build_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (build_dir / "build_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not exe.exists():
        return None, {"build_status": "compile_failed", "returncode": proc.returncode, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:], **instrument_diag}
    return exe, {"build_status": "ok", "exe": str(exe), **instrument_diag}


def run_zhou(
    exe: Path,
    a_path: Path,
    b_path: Path,
    out_dir: Path,
    *,
    n_trials: int,
    refinements: int,
    nb: int,
    max_level: int,
    seed: int,
    jobs: int,
    timeout_s: float | None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        str(a_path),
        str(b_path),
        "-o",
        str(out_dir),
        "-N",
        str(n_trials),
        "-R",
        str(refinements),
        "-Nb",
        str(nb),
        "--max-level",
        str(max_level),
        "--seed",
        str(seed),
        "-j",
        str(jobs),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(out_dir), text=True, capture_output=True, timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        (out_dir / "zhou_stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="replace")
        (out_dir / "zhou_stderr.txt").write_text(exc.stderr or "", encoding="utf-8", errors="replace")
        return {"zhou_status": "timeout", "timeout_s": timeout_s, "runtime_s": time.perf_counter() - started, "cmd": cmd}
    (out_dir / "zhou_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (out_dir / "zhou_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "zhou_status": "ok" if proc.returncode == 0 else "nonzero",
        "returncode": proc.returncode,
        "runtime_s": time.perf_counter() - started,
        "timed_out": int(timed_out),
        "cmd": cmd,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }


def _latest_label_pair(out_dir: Path) -> tuple[Path | None, Path | None]:
    a_files = sorted(out_dir.glob("result_A_*pcs.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for a in a_files:
        b = out_dir / a.name.replace("result_A_", "result_B_")
        if b.exists():
            return a, b
    return None, None


def _label_pair_for_pieces(out_dir: Path, pieces: int) -> tuple[Path | None, Path | None]:
    exact_pairs = [
        (out_dir / f"level_{pieces}pcs_A.png", out_dir / f"level_{pieces}pcs_B.png"),
        (out_dir / f"result_A_{pieces}pcs.png", out_dir / f"result_B_{pieces}pcs.png"),
    ]
    for a, b in exact_pairs:
        if a.exists() and b.exists():
            return a, b
    return _latest_label_pair(out_dir)


def _label_color(label: int) -> tuple[int, int, int]:
    v = (label * 0x9E3779B1) & 0xFFFFFFFF
    v ^= v >> 16
    v = (v * 0x85EBCA6B) & 0xFFFFFFFF
    v ^= v >> 13
    v = (v * 0xC2B2AE35) & 0xFFFFFFFF
    v ^= v >> 16
    r = v & 0xFF
    g = (v >> 8) & 0xFF
    b = (v >> 16) & 0xFF
    if r < 24 and g < 24 and b < 24:
        r ^= 0x55
        g ^= 0xAA
        b ^= 0xCC
    return int(r), int(g), int(b)


def save_label_map_png(path: Path, labels: np.ndarray) -> None:
    """Save integer labels with the same deterministic palette as Zhou's C++ output."""
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        rgb[labels == label] = np.asarray(_label_color(label), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _decode_color_map(colors: list[tuple[int, int, int]], max_guess: int = 4096) -> dict[tuple[int, int, int], int]:
    inverse = {_label_color(i): i for i in range(1, max_guess + 1)}
    out: dict[tuple[int, int, int], int] = {}
    next_id = max_guess + 1
    for color in sorted(colors):
        if color == (0, 0, 0):
            continue
        if color in inverse:
            out[color] = inverse[color]
        else:
            out[color] = next_id
            next_id += 1
    return out


def decode_label_pngs(a_path: Path, b_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    a_img = np.asarray(Image.open(a_path).convert("RGB"))
    b_img = np.asarray(Image.open(b_path).convert("RGB"))
    if a_img.shape != b_img.shape:
        raise ValueError(f"Zhou label PNG sizes differ: {a_img.shape} vs {b_img.shape}")
    colors = set(map(tuple, a_img.reshape(-1, 3).tolist())) | set(map(tuple, b_img.reshape(-1, 3).tolist()))
    cmap = _decode_color_map([tuple(map(int, c)) for c in colors if tuple(c) != (0, 0, 0)])
    la = np.zeros(a_img.shape[:2], dtype=np.int32)
    lb = np.zeros(b_img.shape[:2], dtype=np.int32)
    for color, label in cmap.items():
        c = np.asarray(color, dtype=np.uint8)
        la[np.all(a_img == c, axis=2)] = label
        lb[np.all(b_img == c, axis=2)] = label
    diag = {
        "zhou_label_A": str(a_path),
        "zhou_label_B": str(b_path),
        "labels_A": int(len(set(np.unique(la).tolist()) - {0})),
        "labels_B": int(len(set(np.unique(lb).tolist()) - {0})),
        "labels_common": int(len((set(np.unique(la).tolist()) & set(np.unique(lb).tolist())) - {0})),
    }
    return la, lb, diag


def _mask_to_geometry(mask: np.ndarray, meta: RasterMeta, *, min_area: float) -> BaseGeometry:
    rects = []
    h, w = mask.shape
    for y in range(h):
        x = 0
        while x < w:
            if not mask[y, x]:
                x += 1
                continue
            x0 = x
            while x < w and mask[y, x]:
                x += 1
            x1 = x
            minx = meta.minx + x0 * meta.scale
            maxx = meta.minx + x1 * meta.scale
            miny = meta.maxy - (y + 1) * meta.scale
            maxy = meta.maxy - y * meta.scale
            rects.append(box(minx, miny, maxx, maxy))
    if not rects:
        return GeometryCollection()
    return clean_geom(unary_union(rects))


def _coords(mask: np.ndarray) -> np.ndarray:
    return np.column_stack(np.nonzero(mask)).astype(np.int32)


def _rotflip_coords(local: np.ndarray, h: int, w: int, rot: int, flip: bool) -> tuple[np.ndarray, int, int]:
    y = local[:, 0]
    x = local[:, 1]
    k = (rot // 90) % 4
    if k == 0:
        yy, xx = y.copy(), x.copy()
        hh, ww = h, w
    elif k == 1:
        yy, xx = x.copy(), (h - 1) - y
        hh, ww = w, h
    elif k == 2:
        yy, xx = (h - 1) - y, (w - 1) - x
        hh, ww = h, w
    else:
        yy, xx = (w - 1) - x, y.copy()
        hh, ww = w, h
    if flip:
        xx = (ww - 1) - xx
    return np.column_stack([yy, xx]).astype(np.int32), hh, ww


def best_pixel_isometry(label: int, a_mask: np.ndarray, b_mask: np.ndarray, *, shift_radius: int = 0) -> PixelIsometry | None:
    ayx = _coords(a_mask)
    byx = _coords(b_mask)
    if len(ayx) == 0 or len(byx) == 0:
        return None
    y0, x0 = ayx.min(axis=0).tolist()
    y1, x1 = ayx.max(axis=0).tolist()
    h = int(y1 - y0 + 1)
    w = int(x1 - x0 + 1)
    local = ayx - np.asarray([y0, x0], dtype=np.int32)
    cy_b, cx_b = byx.mean(axis=0)
    target = b_mask.astype(bool)
    H, W = b_mask.shape
    best: PixelIsometry | None = None
    lower = abs(len(ayx) - len(byx))
    for flip in (False, True):
        for rot in (0, 90, 180, 270):
            rcoords, _hh, _ww = _rotflip_coords(local, h, w, rot, flip)
            cy_a, cx_a = (rcoords + np.asarray([y0, x0])).mean(axis=0)
            dyf = cy_b - cy_a
            dxf = cx_b - cx_a
            base_snaps = {(math.floor(dyf), math.floor(dxf)), (math.floor(dyf), math.ceil(dxf)), (math.ceil(dyf), math.floor(dxf)), (math.ceil(dyf), math.ceil(dxf))}
            shifts: set[tuple[int, int]] = set()
            radius = max(0, int(shift_radius))
            for bdy, bdx in base_snaps:
                for oy in range(-radius, radius + 1):
                    for ox in range(-radius, radius + 1):
                        shifts.add((int(bdy) + oy, int(bdx) + ox))
            for dy, dx in sorted(shifts):
                    gy = rcoords[:, 0] + y0 + int(dy)
                    gx = rcoords[:, 1] + x0 + int(dx)
                    inside = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
                    overlap = int(np.count_nonzero(target[gy[inside], gx[inside]]))
                    mismatch = int(len(ayx) + len(byx) - 2 * overlap)
                    if best is None or mismatch < best.mismatch:
                        best = PixelIsometry(label, rot, flip, int(dy), int(dx), int(y0), int(x0), h, w, mismatch, int(len(ayx)), int(len(byx)))
                        if mismatch <= lower:
                            return best
    return best


def geometry_match_labels(la: np.ndarray, lb: np.ndarray) -> tuple[np.ndarray, dict[int, PixelIsometry], dict[str, object]]:
    """Rematch Zhou labels by geometry instead of trusting equal label IDs.

    Zhou's refinement uses matching internally, but the saved color labels are not
    a reliable correspondence between pieces. This pass builds a bipartite
    assignment using the same 8-isometry mismatch plus cheap shape-quality terms,
    then relabels target pieces into source-label IDs.
    """
    source_ids = sorted(int(x) for x in np.unique(la) if int(x) > 0)
    target_ids = sorted(int(x) for x in np.unique(lb) if int(x) > 0)
    if not source_ids or not target_ids:
        return lb.copy(), {}, {"geometry_matching": "empty"}
    iso_table: dict[tuple[int, int], PixelIsometry] = {}
    costs = np.full((len(source_ids), len(target_ids)), 1e6, dtype=float)
    source_stats: dict[int, tuple[int, int, int]] = {}
    target_stats: dict[int, tuple[int, int, int]] = {}
    for sid in source_ids:
        sm = la == sid
        source_stats[sid] = (_count_mask(sm), _perimeter_pixels(sm), _components_binary(sm))
    for tid in target_ids:
        tm = lb == tid
        target_stats[tid] = (_count_mask(tm), _perimeter_pixels(tm), _components_binary(tm))
    for i, sid in enumerate(source_ids):
        sm = la == sid
        sa, sp, sc = source_stats[sid]
        for j, tid in enumerate(target_ids):
            tm = lb == tid
            ta, tp, tc = target_stats[tid]
            iso = best_pixel_isometry(sid, sm, tm)
            if iso is None:
                continue
            iso_table[(sid, tid)] = iso
            mismatch = iso.mismatch / max(sa + ta, 1)
            area = abs(sa - ta) / max(sa + ta, 1)
            perim = abs(sp - tp) / max(sp + tp, 1)
            comps = abs(sc - tc)
            costs[i, j] = mismatch + 0.30 * area + 0.08 * perim + 0.05 * comps
    row_ind, col_ind = linear_sum_assignment(costs)
    lb_matched = np.zeros_like(lb, dtype=np.int32)
    isos: dict[int, PixelIsometry] = {}
    pairs: list[dict[str, object]] = []
    total_mismatch = 0
    total_source = 0
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        sid = source_ids[i]
        tid = target_ids[j]
        iso = iso_table.get((sid, tid))
        if iso is None:
            continue
        lb_matched[lb == tid] = sid
        isos[sid] = iso
        total_mismatch += iso.mismatch
        total_source += iso.source_pixels
        pairs.append(
            {
                "source_label": sid,
                "target_label": tid,
                "cost": float(costs[i, j]),
                "mismatch": int(iso.mismatch),
                "source_pixels": int(iso.source_pixels),
                "target_pixels": int(iso.target_pixels),
                "rot": int(iso.rot),
                "flip": bool(iso.flip),
                "dy": int(iso.dy),
                "dx": int(iso.dx),
            }
        )
    return lb_matched, isos, {
        "geometry_matching": "hungarian_8isometry_shape_cost",
        "geometry_match_pairs": pairs,
        "geometry_match_pair_count": len(pairs),
        "geometry_match_total_mismatch": int(total_mismatch),
        "geometry_match_mismatch_fraction": float(total_mismatch / max(total_source, 1)),
        "geometry_match_flip_labels": int(sum(1 for iso in isos.values() if iso.flip)),
    }


def _pixel_to_world(meta: RasterMeta, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return meta.minx + px * meta.scale, meta.maxy - py * meta.scale


def _world_to_pixel(meta: RasterMeta, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x - meta.minx) / meta.scale, (meta.maxy - y) / meta.scale


def _pixel_forward_xy(iso: PixelIsometry, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = px - iso.x0
    y = py - iso.y0
    k = (iso.rot // 90) % 4
    if k == 0:
        xr, yr = x, y
        ww = iso.w
    elif k == 1:
        xr, yr = iso.h - y, x
        ww = iso.h
    elif k == 2:
        xr, yr = iso.w - x, iso.h - y
        ww = iso.w
    else:
        xr, yr = y, iso.w - x
        ww = iso.h
    if iso.flip:
        xr = ww - xr
    return xr + iso.x0 + iso.dx, yr + iso.y0 + iso.dy


def _pixel_inverse_xy(iso: PixelIsometry, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Invert by undoing global shift, optional flip, then the 90-degree rotation.
    xr = px - iso.x0 - iso.dx
    yr = py - iso.y0 - iso.dy
    k = (iso.rot // 90) % 4
    ww = iso.w if k in (0, 2) else iso.h
    if iso.flip:
        xr = ww - xr
    if k == 0:
        x, y = xr, yr
    elif k == 1:
        x, y = yr, iso.h - xr
    elif k == 2:
        x, y = iso.w - xr, iso.h - yr
    else:
        x, y = iso.w - yr, xr
    return x + iso.x0, y + iso.y0


def apply_spec(geom: BaseGeometry, spec: TransformSpec, meta: RasterMeta) -> BaseGeometry:
    if geom.is_empty:
        return geom
    if spec.kind == "rigid" and spec.rigid is not None:
        return apply_rigid(geom, spec.rigid)
    if spec.pixel is None:
        return geom
    iso = spec.pixel

    def fn(x, y, z=None):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        px, py = _world_to_pixel(meta, xa, ya)
        px2, py2 = _pixel_forward_xy(iso, px, py)
        wx, wy = _pixel_to_world(meta, px2, py2)
        return wx, wy

    return clean_geom(shapely_transform(fn, geom))


def _components_binary(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    comps = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        comps += 1
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return comps


def _perimeter_pixels(mask: np.ndarray) -> int:
    p = np.pad(mask.astype(bool), 1, constant_values=False)
    center = p[1:-1, 1:-1]
    return int(np.count_nonzero(center & ~p[:-2, 1:-1]) + np.count_nonzero(center & ~p[2:, 1:-1]) + np.count_nonzero(center & ~p[1:-1, :-2]) + np.count_nonzero(center & ~p[1:-1, 2:]))


def main() -> None:
    """Run the two-figure lattice search of Zhou and Wang (2012) and keep the per-level labels.

    This is the lattice-search stage of the original exp335 driver (same arguments and defaults);
    the label PNGs it writes (zhou_run/level_<k>pcs_{A,B}.png) are the inputs of exp358 (BACO).
    """
    p = parser()
    p.add_argument("--pair", type=str, default="nike_to_shoe")
    p.add_argument("--input-mode", type=str, default="benchmark", choices=["benchmark", "png"])
    p.add_argument("--shape-a", type=str, default=str(ROOT.parent / "nike.png"))
    p.add_argument("--shape-b", type=str, default=str(ROOT.parent / "shoe.png"))
    p.add_argument("--simplify", type=float, default=2.5)
    p.add_argument("--resolution", type=int, default=180)
    p.add_argument("--square-size", type=int, default=0, help="force a square Zhou lattice, e.g. 215 for the Nike/shoe run")
    p.add_argument("--pad-frac", type=float, default=0.04)
    p.add_argument("--zhou-target-pieces", type=int, default=6, help="report the label pair of this level when available")
    p.add_argument("--zhou-source", type=str, default=str(Path(__file__).resolve().parent / "zhou2d.cpp"))
    p.add_argument("--zhou-bin", type=str, default="")
    p.add_argument("--build-zhou", action="store_true")
    p.add_argument("--download-stb", action="store_true")
    p.add_argument("--zhou-N", type=int, default=120)
    p.add_argument("--zhou-R", type=int, default=80)
    p.add_argument("--zhou-Nb", type=int, default=20)
    p.add_argument("--zhou-max-level", type=int, default=5)
    p.add_argument("--zhou-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=2012)
    p.add_argument("--zhou-timeout-s", type=float, default=0.0)
    p.add_argument("--tag", type=str, default="exp335_zhou_seeded_vector_solve")
    args = p.parse_args()

    out_dir = RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.input_mode == "benchmark":
        _benchmark, source, target = load_benchmark(args.pair, args.simplify)
        a_input, b_input, _meta, _raster = prepare_zhou_inputs(source, target, out_dir, resolution=args.resolution, pad_frac=args.pad_frac, square_size=args.square_size)
    else:
        a_input, b_input = Path(args.shape_a), Path(args.shape_b)

    exe: Path | None = Path(args.zhou_bin) if args.zhou_bin else None
    if exe is not None and not exe.exists():
        exe = None
    build_diag: dict[str, object] = {"build_status": "provided_binary" if exe else "not_requested"}
    if args.build_zhou:
        exe, build_diag = build_zhou_binary(Path(args.zhou_source), out_dir / "zhou_build", download_stb=args.download_stb)
    if exe is None:
        raise SystemExit(f"no lattice-search binary ({build_diag}); use --build-zhou or --zhou-bin")

    zhou_out = out_dir / "zhou_run"
    run_diag = run_zhou(
        exe, a_input, b_input, zhou_out,
        n_trials=args.zhou_N, refinements=args.zhou_R, nb=args.zhou_Nb, max_level=args.zhou_max_level,
        seed=args.seed, jobs=args.zhou_jobs, timeout_s=(args.zhou_timeout_s if args.zhou_timeout_s > 0 else None),
    )
    labels_a, labels_b = _label_pair_for_pieces(zhou_out, args.zhou_target_pieces)
    print({**build_diag, **run_diag, "labels_a": str(labels_a), "labels_b": str(labels_b)})


if __name__ == "__main__":
    main()
