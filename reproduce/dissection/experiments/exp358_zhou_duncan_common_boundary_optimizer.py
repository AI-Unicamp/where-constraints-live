from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from skimage import measure

from _common import ROOT, parser
from shape_benchmarks import load_benchmark
from lattice2d import (
    PixelIsometry,
    TransformSpec,
    _pixel_forward_xy,
    _pixel_inverse_xy,
    apply_spec,
    decode_label_pngs,
    geometry_match_labels,
    load_png_shapes,
    prepare_zhou_inputs,
)
from src.shapely_quantile_solvers import SPiece, SSolution, clean_geom, export_shapely_solution_svg, validate_shapely_solution


RESULTS = ROOT / "outputs" / "zhou_duncan_common_boundary_optimizer"
EPS = 1e-9


@dataclass(frozen=True)
class Contact:
    kind: str
    a_label: int
    a_idx: int
    b_label: int | None = None
    b_idx: int | None = None
    target: tuple[float, float] | None = None


@dataclass
class Model:
    labels: list[int]
    offsets: dict[int, int]
    n: int
    loops_a: dict[int, np.ndarray]
    loops_b: dict[int, np.ndarray]
    init_loops: dict[int, np.ndarray]
    isos: dict[int, PixelIsometry]
    a_contacts: list[Contact]
    b_contacts: list[Contact]
    a_exterior: list[Contact]
    b_exterior: list[Contact]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def contour_loop(mask: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(mask.astype(np.float32), 0.5, fully_connected="high")
    if not contours:
        raise ValueError("empty contour")
    c = max(contours, key=len)
    pts = np.column_stack([c[:, 1], c[:, 0]]).astype(np.float64)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    if signed_area(pts) < 0:
        pts = pts[::-1]
    return pts


def resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) < 3:
        raise ValueError("cannot resample fewer than 3 points")
    pts = np.vstack([points, points[0]])
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(seg.sum())
    if total <= EPS:
        raise ValueError("zero-length contour")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] <= t:
            j += 1
        denom = max(cum[j + 1] - cum[j], EPS)
        a = (t - cum[j]) / denom
        out[i] = (1.0 - a) * pts[j] + a * pts[j + 1]
    return out


def align_loop(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    n = len(reference)
    best = candidate
    best_cost = float("inf")
    for rev in (False, True):
        c = candidate[::-1].copy() if rev else candidate
        for shift in range(n):
            shifted = np.roll(c, shift, axis=0)
            cost = float(np.mean(np.sum((reference - shifted) ** 2, axis=1)))
            if cost < best_cost:
                best_cost = cost
                best = shifted
    if signed_area(best) * signed_area(reference) < 0:
        best = best[::-1]
    return best.copy()


def inverse_points(iso: PixelIsometry, points: np.ndarray) -> np.ndarray:
    px, py = _pixel_inverse_xy(iso, points[:, 0], points[:, 1])
    return np.column_stack([px, py]).astype(np.float64)


def forward_points(iso: PixelIsometry, points: np.ndarray) -> np.ndarray:
    px, py = _pixel_forward_xy(iso, points[:, 0], points[:, 1])
    return np.column_stack([px, py]).astype(np.float64)


def relation_edges(labels_img: np.ndarray, *, max_per_kind: int, rng: np.random.Generator) -> dict[tuple[int, int], np.ndarray]:
    rows: dict[tuple[int, int], list[tuple[float, float]]] = {}
    h, w = labels_img.shape
    left = labels_img[:, :-1]
    right = labels_img[:, 1:]
    ys, xs = np.nonzero(left != right)
    for y, x in zip(ys.tolist(), xs.tolist()):
        a = int(left[y, x])
        b = int(right[y, x])
        if a == b:
            continue
        rows.setdefault((a, b), []).append((float(x + 1.0), float(y + 0.5)))
        rows.setdefault((b, a), []).append((float(x + 1.0), float(y + 0.5)))
    up = labels_img[:-1, :]
    down = labels_img[1:, :]
    ys, xs = np.nonzero(up != down)
    for y, x in zip(ys.tolist(), xs.tolist()):
        a = int(up[y, x])
        b = int(down[y, x])
        if a == b:
            continue
        rows.setdefault((a, b), []).append((float(x + 0.5), float(y + 1.0)))
        rows.setdefault((b, a), []).append((float(x + 0.5), float(y + 1.0)))
    out: dict[tuple[int, int], np.ndarray] = {}
    for key, pts in rows.items():
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) > max_per_kind:
            arr = arr[rng.permutation(len(arr))[:max_per_kind]]
        out[key] = arr
    return out


def nearest_indices(loop: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.int32)
    tree = cKDTree(loop)
    _d, idx = tree.query(pts)
    return idx.astype(np.int32)


def build_model(la: np.ndarray, lb: np.ndarray, isos: dict[int, PixelIsometry], labels: list[int], args) -> Model:
    rng = np.random.default_rng(args.seed)
    loops_a: dict[int, np.ndarray] = {}
    loops_b: dict[int, np.ndarray] = {}
    init_loops: dict[int, np.ndarray] = {}
    for label in labels:
        a_loop = resample_closed(contour_loop(la == label), args.samples_per_piece)
        b_loop_b = resample_closed(contour_loop(lb == label), args.samples_per_piece)
        b_loop = align_loop(a_loop, inverse_points(isos[label], b_loop_b))
        init = (args.a_init_weight * a_loop + args.b_init_weight * b_loop) / max(args.a_init_weight + args.b_init_weight, EPS)
        loops_a[label] = a_loop
        loops_b[label] = b_loop
        init_loops[label] = init

    trees = {label: cKDTree(init_loops[label]) for label in labels}
    a_rel = relation_edges(la, max_per_kind=args.max_relation_samples, rng=rng)
    b_rel = relation_edges(lb, max_per_kind=args.max_relation_samples, rng=rng)
    a_contacts: list[Contact] = []
    b_contacts: list[Contact] = []
    a_exterior: list[Contact] = []
    b_exterior: list[Contact] = []
    label_set = set(labels)

    for (owner, neighbor), pts in a_rel.items():
        if owner not in label_set:
            continue
        oi = nearest_indices(init_loops[owner], pts)
        if neighbor == 0:
            for idx, p in zip(oi.tolist(), pts.tolist()):
                a_exterior.append(Contact("a_ext", owner, int(idx), target=(float(p[0]), float(p[1]))))
        elif neighbor in label_set and owner < neighbor:
            ji = nearest_indices(init_loops[neighbor], pts)
            for io, ij in zip(oi.tolist(), ji.tolist()):
                a_contacts.append(Contact("a_contact", owner, int(io), neighbor, int(ij)))

    for (owner, neighbor), pts_b in b_rel.items():
        if owner not in label_set:
            continue
        owner_pts = inverse_points(isos[owner], pts_b)
        oi = nearest_indices(init_loops[owner], owner_pts)
        if neighbor == 0:
            for idx, p in zip(oi.tolist(), pts_b.tolist()):
                b_exterior.append(Contact("b_ext", owner, int(idx), target=(float(p[0]), float(p[1]))))
        elif neighbor in label_set and owner < neighbor:
            neigh_pts = inverse_points(isos[neighbor], pts_b)
            ji = nearest_indices(init_loops[neighbor], neigh_pts)
            for io, ij in zip(oi.tolist(), ji.tolist()):
                b_contacts.append(Contact("b_contact", owner, int(io), neighbor, int(ij)))

    offsets: dict[int, int] = {}
    off = 0
    for label in labels:
        offsets[label] = off
        off += args.samples_per_piece * 2
    return Model(labels, offsets, off, loops_a, loops_b, init_loops, isos, a_contacts, b_contacts, a_exterior, b_exterior)


def pack(model: Model, loops: dict[int, np.ndarray]) -> np.ndarray:
    x = np.zeros(model.n, dtype=np.float64)
    for label in model.labels:
        off = model.offsets[label]
        x[off : off + len(loops[label]) * 2] = loops[label].reshape(-1)
    return x


def unpack(model: Model, x: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for label in model.labels:
        off = model.offsets[label]
        out[label] = x[off : off + len(model.init_loops[label]) * 2].reshape((-1, 2))
    return out


def loop_point(model: Model, x: np.ndarray, label: int, idx: int) -> np.ndarray:
    off = model.offsets[label] + idx * 2
    return x[off : off + 2]


def residuals(x: np.ndarray, model: Model, args) -> np.ndarray:
    res: list[np.ndarray] = []
    for c in model.a_contacts:
        p = loop_point(model, x, c.a_label, c.a_idx)
        q = loop_point(model, x, int(c.b_label), int(c.b_idx))
        res.append(args.a_contact_weight * (p - q))
    for c in model.b_contacts:
        p = forward_points(model.isos[c.a_label], loop_point(model, x, c.a_label, c.a_idx)[None, :])[0]
        q = forward_points(model.isos[int(c.b_label)], loop_point(model, x, int(c.b_label), int(c.b_idx))[None, :])[0]
        res.append(args.b_contact_weight * (p - q))
    for c in model.a_exterior:
        p = loop_point(model, x, c.a_label, c.a_idx)
        t = np.asarray(c.target, dtype=np.float64)
        res.append(args.a_exterior_weight * (p - t))
    for c in model.b_exterior:
        p = forward_points(model.isos[c.a_label], loop_point(model, x, c.a_label, c.a_idx)[None, :])[0]
        t = np.asarray(c.target, dtype=np.float64)
        res.append(args.b_exterior_weight * (p - t))
    for label in model.labels:
        pts = x[model.offsets[label] : model.offsets[label] + len(model.init_loops[label]) * 2].reshape((-1, 2))
        init = model.init_loops[label]
        a = model.loops_a[label]
        b = model.loops_b[label]
        res.append(args.shape_weight * (pts - init).reshape(-1))
        res.append(args.a_shape_weight * (pts - a).reshape(-1))
        res.append(args.b_shape_weight * (pts - b).reshape(-1))
        fair = np.roll(pts, 1, axis=0) - 2.0 * pts + np.roll(pts, -1, axis=0)
        res.append(args.fair_weight * fair.reshape(-1))
        edge_len = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        target_len = np.linalg.norm(np.roll(init, -1, axis=0) - init, axis=1)
        res.append(args.edge_weight * (edge_len - target_len))
    if not res:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate([r.reshape(-1) for r in res]).astype(np.float64)


def px_to_world(points: np.ndarray, meta) -> np.ndarray:
    x = meta.minx + points[:, 0] * meta.scale
    y = meta.maxy - points[:, 1] * meta.scale
    return np.column_stack([x, y])


def polygon_from_loop(points_px: np.ndarray, meta) -> BaseGeometry:
    pts = px_to_world(points_px, meta)
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return clean_geom(poly)


def make_solution(model: Model, x: np.ndarray, meta, min_area: float) -> SSolution:
    loops = unpack(model, x)
    pieces: list[SPiece] = []
    for label in model.labels:
        geom = polygon_from_loop(loops[label], meta)
        if geom.is_empty or float(geom.area) < min_area:
            continue
        spec = TransformSpec(model.isos[label].name, "pixel", label, pixel=model.isos[label])
        target_geom = clean_geom(apply_spec(geom, spec, meta))
        pieces.append(SPiece(len(pieces), f"common_boundary_l{label}", geom, target_geom, "zhou_duncan_common_boundary", (float(label),)))
    return SSolution("zhou_duncan_common_boundary", pieces, 0.0, "sampled shared piece boundaries with A/B contact equality constraints")


def contact_rms(x: np.ndarray, model: Model) -> dict[str, float]:
    def rms(vals: list[float]) -> float:
        return float(math.sqrt(sum(v * v for v in vals) / max(len(vals), 1)))

    a = []
    for c in model.a_contacts:
        a.append(float(np.linalg.norm(loop_point(model, x, c.a_label, c.a_idx) - loop_point(model, x, int(c.b_label), int(c.b_idx)))))
    b = []
    for c in model.b_contacts:
        p = forward_points(model.isos[c.a_label], loop_point(model, x, c.a_label, c.a_idx)[None, :])[0]
        q = forward_points(model.isos[int(c.b_label)], loop_point(model, x, int(c.b_label), int(c.b_idx))[None, :])[0]
        b.append(float(np.linalg.norm(p - q)))
    return {"a_contact_rms_px": rms(a), "b_contact_rms_px": rms(b), "a_contact_count": float(len(a)), "b_contact_count": float(len(b))}


def main() -> None:
    p = parser()
    p.add_argument("--pair", default="nike_to_shoe")
    p.add_argument("--input-mode", choices=["benchmark", "png"], default="benchmark")
    p.add_argument("--shape-a", default="")
    p.add_argument("--shape-b", default="")
    p.add_argument("--simplify", type=float, default=2.5)
    p.add_argument("--square-size", type=int, default=215)
    p.add_argument("--resolution", type=int, default=180)
    p.add_argument("--pad-frac", type=float, default=0.04)
    p.add_argument("--labels-a", required=True)
    p.add_argument("--labels-b", required=True)
    p.add_argument("--samples-per-piece", type=int, default=96)
    p.add_argument("--max-relation-samples", type=int, default=220)
    p.add_argument("--a-init-weight", type=float, default=0.65)
    p.add_argument("--b-init-weight", type=float, default=0.35)
    p.add_argument("--a-contact-weight", type=float, default=8.0)
    p.add_argument("--b-contact-weight", type=float, default=14.0)
    p.add_argument("--a-exterior-weight", type=float, default=3.5)
    p.add_argument("--b-exterior-weight", type=float, default=1.4)
    p.add_argument("--shape-weight", type=float, default=0.9)
    p.add_argument("--a-shape-weight", type=float, default=0.8)
    p.add_argument("--b-shape-weight", type=float, default=0.35)
    p.add_argument("--fair-weight", type=float, default=8.0)
    p.add_argument("--edge-weight", type=float, default=0.8)
    p.add_argument("--max-nfev", type=int, default=120)
    p.add_argument("--loss", default="soft_l1")
    p.add_argument("--f-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2012)
    p.add_argument("--min-piece-area-frac", type=float, default=0.0002)
    p.add_argument("--tag", default="exp358_zhou_duncan_common_boundary")
    args = p.parse_args()

    started = time.perf_counter()
    out_dir = RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.input_mode == "png":
        if not args.shape_a or not args.shape_b:
            raise ValueError("--input-mode png requires --shape-a and --shape-b")
        benchmark = f"{Path(args.shape_a).stem}_to_{Path(args.shape_b).stem}"
        source, target, meta, raster_diag = load_png_shapes(Path(args.shape_a), Path(args.shape_b))
    else:
        benchmark, source, target = load_benchmark(args.pair, args.simplify)
        _a, _b, meta, raster_diag = prepare_zhou_inputs(source, target, out_dir, resolution=args.resolution, pad_frac=args.pad_frac, square_size=args.square_size)
    la, lb_raw, label_diag = decode_label_pngs(Path(args.labels_a), Path(args.labels_b))
    lb, isos, match_diag = geometry_match_labels(la, lb_raw)
    labels = sorted((set(int(v) for v in np.unique(la)) & set(int(v) for v in np.unique(lb)) & set(isos)) - {0})
    model = build_model(la, lb, isos, labels, args)
    x0 = pack(model, model.init_loops)
    start = contact_rms(x0, model)
    res = least_squares(
        lambda z: residuals(z, model, args),
        x0,
        max_nfev=args.max_nfev,
        verbose=0,
        loss=args.loss,
        f_scale=args.f_scale,
        x_scale="jac",
    )
    final = contact_rms(res.x, model)
    min_shape = max(min(float(source.area), float(target.area)), EPS)
    sol = make_solution(model, res.x, meta, args.min_piece_area_frac * min_shape)
    valid, valid_diag = validate_shapely_solution(source, target, sol.pieces)
    cert = slack_certificate(source, target, sol.pieces if valid else [], k_total=max(6, len(sol.pieces)), min_component_area=max(1e-10, 1e-7 * min_shape))
    export_shapely_solution_svg(source, target, sol, out_dir / f"{benchmark}_common_boundary.svg")
    rows = []
    loops = unpack(model, res.x)
    for label in labels:
        rows.append(
            {
                "label": label,
                "area": float(polygon_from_loop(loops[label], meta).area),
                "length": float(polygon_from_loop(loops[label], meta).length),
            }
        )
    write_csv(out_dir / "pieces.csv", rows)
    summary = {
        "benchmark": benchmark,
        "runtime_s": time.perf_counter() - started,
        "pieces": float(len(sol.pieces)),
        "shared_area": float(sol.shared_area),
        "least_squares_cost": float(res.cost),
        "least_squares_nfev": float(res.nfev),
        "least_squares_status": float(res.status),
        "vector_valid_nonoverlap": bool(valid),
        "source_outside_area": float(valid_diag["source_outside_area"]),
        "target_outside_area": float(valid_diag["target_outside_area"]),
        "source_overlap_area": float(valid_diag["source_overlap_area"]),
        "target_overlap_area": float(valid_diag["target_overlap_area"]),
        "source_slack_components": float(cert.source_slack_components),
        "target_slack_components": float(cert.target_slack_components),
        "total_components": float(cert.total_components),
        **{f"start_{k}": v for k, v in start.items()},
        **{f"final_{k}": v for k, v in final.items()},
    }
    write_csv(out_dir / "summary.csv", [summary])
    payload = {
        "out_dir": str(out_dir),
        "input_mode": args.input_mode,
        "shape_a": args.shape_a,
        "shape_b": args.shape_b,
        "raster": raster_diag,
        "labels": label_diag,
        "matching": match_diag,
        "validate": valid_diag,
        "summary": summary,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    from src.shapely_duncan_slack import slack_certificate

    main()
