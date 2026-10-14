from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "zhou3d_common_boundary_optimizer"
EPS = 1e-9


@dataclass(frozen=True)
class Rot3:
    axis: tuple[int, int, int]
    sign: tuple[int, int, int]


@dataclass(frozen=True)
class Transform3:
    label: int
    target_label: int
    rot_index: int
    shift: tuple[int, int, int]
    bbox_min: tuple[int, int, int]
    bbox_dims: tuple[int, int, int]
    mismatch: float

    def matrix_and_offset(self) -> tuple[np.ndarray, np.ndarray]:
        rot = ROTS[self.rot_index]
        mat = np.zeros((3, 3), dtype=np.float64)
        d = np.zeros(3, dtype=np.float64)
        dims = np.asarray(self.bbox_dims, dtype=np.float64)
        mn = np.asarray(self.bbox_min, dtype=np.float64)
        sh = np.asarray(self.shift, dtype=np.float64)
        for out_axis, src_axis in enumerate(rot.axis):
            s = float(rot.sign[out_axis])
            mat[out_axis, src_axis] = s
            if s < 0:
                d[out_axis] = dims[src_axis] - 1.0
        off = d + mn + sh - mat @ mn
        return mat, off

    def forward(self, pts: np.ndarray) -> np.ndarray:
        mat, off = self.matrix_and_offset()
        return pts @ mat.T + off

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        mat, off = self.matrix_and_offset()
        return (pts - off) @ mat


@dataclass
class PieceMesh:
    label: int
    verts_a: np.ndarray
    verts_b_pulled: np.ndarray
    verts_init: np.ndarray
    b_nearest: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    neighbors: list[np.ndarray]


@dataclass(frozen=True)
class ContactSample:
    label_a: int
    idx_a: int
    label_b: int
    idx_b: int


@dataclass(frozen=True)
class ExteriorSample:
    label: int
    idx: int
    target: tuple[float, float, float]


@dataclass
class Model3:
    labels: list[int]
    offsets: dict[int, int]
    meshes: dict[int, PieceMesh]
    transforms: dict[int, Transform3]
    a_contacts: list[ContactSample]
    b_contacts: list[ContactSample]
    a_exterior: list[ExteriorSample]
    b_exterior: list[ExteriorSample]

    @property
    def variable_count(self) -> int:
        return sum(mesh.verts_init.size for mesh in self.meshes.values())


def make_rots() -> list[Rot3]:
    out: list[Rot3] = []
    for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        inv = int(p[0] > p[1]) + int(p[0] > p[2]) + int(p[1] > p[2])
        parity = -1 if inv % 2 else 1
        for s0 in (-1, 1):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    if parity * s0 * s1 * s2 == 1:
                        out.append(Rot3(p, (s0, s1, s2)))
    return out


ROTS = make_rots()


def label_color(label: int) -> tuple[int, int, int]:
    if label <= 0:
        return (255, 255, 255)
    v = (label * 0x9E3779B1) & 0xFFFFFFFF
    v ^= v >> 16
    v = (v * 0x85EBCA6B) & 0xFFFFFFFF
    v ^= v >> 13
    v = (v * 0xC2B2AE35) & 0xFFFFFFFF
    v ^= v >> 16
    r, g, b = v & 255, (v >> 8) & 255, (v >> 16) & 255
    if r < 24 and g < 24 and b < 24:
        r ^= 0x55
        g ^= 0xAA
        b ^= 0xCC
    return int(r), int(g), int(b)


def decode_label_stack(path: Path, *, max_label_guess: int = 512) -> np.ndarray:
    files = sorted(path.glob("*.png"))
    if not files:
        raise ValueError(f"no PNG slices in {path}")
    sidecar = path / "label_colors.json"
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        inverse = {tuple(int(v) for v in rgb.split(",")): int(label) for label, rgb in payload["label_to_rgb"].items()}
    else:
        inverse = {label_color(i): i for i in range(1, max_label_guess + 1)}
    slices: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        if shape is None:
            shape = img.shape[:2]
        elif img.shape[:2] != shape:
            raise ValueError(f"slice dimensions differ in {path}")
        lab = np.zeros(img.shape[:2], dtype=np.int32)
        flat = img.reshape(-1, 3)
        out = lab.reshape(-1)
        colors = np.unique(flat, axis=0)
        for c in colors:
            tup = (int(c[0]), int(c[1]), int(c[2]))
            if tup == (255, 255, 255):
                continue
            label = inverse.get(tup)
            if label is None:
                raise ValueError(f"unknown Zhou label color {tup} in {f}")
            out[np.all(flat == c, axis=1)] = label
        slices.append(lab)
    return np.stack(slices, axis=0)


def write_label_stack(path: Path, labels: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    all_labels = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    label_to_rgb: dict[int, tuple[int, int, int]] = {}
    used = {(255, 255, 255)}
    # Preserve historical colors when possible; fall back to a collision-free
    # integer RGB code for high-piece repaired stacks.
    for label in all_labels:
        c = label_color(label)
        if c in used:
            code = label
            while True:
                code += 1
                c = (code & 255, (code >> 8) & 255, (code >> 16) & 255)
                if c not in used and c != (255, 255, 255):
                    break
        used.add(c)
        label_to_rgb[label] = c
    for z in range(labels.shape[0]):
        rgb = np.full(labels.shape[1:] + (3,), 255, dtype=np.uint8)
        for label in sorted(int(v) for v in np.unique(labels[z]) if int(v) > 0):
            rgb[labels[z] == label] = np.asarray(label_to_rgb[label], dtype=np.uint8)
        Image.fromarray(rgb, "RGB").save(path / f"z_{z:04d}.png")
    (path / "label_colors.json").write_text(
        json.dumps(
            {"label_to_rgb": {str(k): ",".join(str(int(v)) for v in rgb) for k, rgb in label_to_rgb.items()}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def labels_in(vol: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(vol) if int(v) > 0)


def bbox_of_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros(3, dtype=np.int32), np.zeros(3, dtype=np.int32)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    return mn.astype(np.int32), (mx - mn + 1).astype(np.int32)


def rotate_local(local: np.ndarray, dims: np.ndarray, rot: Rot3) -> np.ndarray:
    out = np.empty_like(local)
    for out_axis, src_axis in enumerate(rot.axis):
        vals = local[:, src_axis]
        if rot.sign[out_axis] > 0:
            out[:, out_axis] = vals
        else:
            out[:, out_axis] = dims[src_axis] - 1 - vals
    return out


def snap_shifts(delta: np.ndarray) -> list[tuple[int, int, int]]:
    floors = np.floor(delta).astype(int)
    ceils = np.ceil(delta).astype(int)
    out = []
    for z in (floors[0], ceils[0]):
        for y in (floors[1], ceils[1]):
            for x in (floors[2], ceils[2]):
                item = (int(z), int(y), int(x))
                if item not in out:
                    out.append(item)
    return out


def pair_mismatch(a_mask: np.ndarray, b_mask: np.ndarray, label: int, target_label: int) -> Transform3:
    a = np.argwhere(a_mask)
    b = np.argwhere(b_mask)
    if len(a) == 0 or len(b) == 0:
        return Transform3(label, target_label, 0, (0, 0, 0), (0, 0, 0), (1, 1, 1), float(max(len(a), len(b))))
    mn, dims = bbox_of_mask(a_mask)
    local = a - mn[None, :]
    centroid_b = b.mean(axis=0)
    lower = abs(len(a) - len(b))
    best = Transform3(label, target_label, 0, (0, 0, 0), tuple(int(v) for v in mn), tuple(int(v) for v in dims), float("inf"))
    bd, bh, bw = b_mask.shape
    for ri, rot in enumerate(ROTS):
        rr = rotate_local(local, dims, rot)
        centroid_a = rr.mean(axis=0) + mn
        for shift in snap_shifts(centroid_b - centroid_a):
            placed = rr + mn[None, :] + np.asarray(shift, dtype=np.int32)[None, :]
            ok = (
                (0 <= placed[:, 0])
                & (placed[:, 0] < bd)
                & (0 <= placed[:, 1])
                & (placed[:, 1] < bh)
                & (0 <= placed[:, 2])
                & (placed[:, 2] < bw)
            )
            if np.any(ok):
                pp = placed[ok]
                overlap = int(np.count_nonzero(b_mask[pp[:, 0], pp[:, 1], pp[:, 2]]))
            else:
                overlap = 0
            mis = float(len(a) + len(b) - 2 * overlap)
            if mis < best.mismatch:
                best = Transform3(label, target_label, ri, tuple(int(v) for v in shift), tuple(int(v) for v in mn), tuple(int(v) for v in dims), mis)
                if mis <= lower + 1e-12:
                    return best
    return best


def geometry_match_labels_3d(la: np.ndarray, lb_raw: np.ndarray) -> tuple[np.ndarray, dict[int, Transform3], dict[str, object]]:
    a_ids = labels_in(la)
    b_ids = labels_in(lb_raw)
    if not a_ids or not b_ids:
        raise ValueError("empty label set")
    costs = np.full((len(a_ids), len(b_ids)), 1e18, dtype=np.float64)
    transforms: dict[tuple[int, int], Transform3] = {}
    for i, a_label in enumerate(a_ids):
        a_mask = la == a_label
        for j, b_label in enumerate(b_ids):
            tf = pair_mismatch(a_mask, lb_raw == b_label, a_label, b_label)
            transforms[(a_label, b_label)] = tf
            costs[i, j] = tf.mismatch
    rows, cols = linear_sum_assignment(costs)
    lb = np.zeros_like(lb_raw)
    out: dict[int, Transform3] = {}
    mapping = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        a_label = a_ids[r]
        b_label = b_ids[c]
        lb[lb_raw == b_label] = a_label
        out[a_label] = transforms[(a_label, b_label)]
        mapping.append({"source_label": a_label, "target_label": b_label, "mismatch": float(costs[r, c])})
    diag = {
        "source_labels": len(a_ids),
        "target_labels": len(b_ids),
        "matched_labels": len(out),
        "total_mismatch": float(sum(m["mismatch"] for m in mapping)),
        "proper_rotations": len(ROTS),
        "mapping": mapping,
    }
    return lb, out, diag


def extract_mesh(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
    verts, faces, _normals, _values = measure.marching_cubes(padded, level=0.5, allow_degenerate=False)
    verts = verts.astype(np.float64) - 1.0
    return verts, faces.astype(np.int32)


def mesh_edges_and_neighbors(faces: np.ndarray, nverts: int) -> tuple[np.ndarray, list[np.ndarray]]:
    edges: set[tuple[int, int]] = set()
    for tri in faces.tolist():
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    edge_arr = np.asarray(sorted(edges), dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    neigh = [set() for _ in range(nverts)]
    for a, b in edge_arr.tolist():
        neigh[a].add(b)
        neigh[b].add(a)
    return edge_arr, [np.asarray(sorted(s), dtype=np.int32) for s in neigh]


def build_piece_mesh(label: int, la: np.ndarray, lb: np.ndarray, tf: Transform3, args: argparse.Namespace) -> PieceMesh:
    va, faces = extract_mesh(la == label)
    vb, _fb = extract_mesh(lb == label)
    if len(va) == 0:
        raise ValueError(f"label {label} has empty source mesh")
    if len(vb) == 0:
        vb_pulled = va.copy()
    else:
        vb_pulled = tf.inverse(vb)
    if len(vb_pulled):
        tree = cKDTree(vb_pulled)
        _d, idx = tree.query(va)
        b_nearest = vb_pulled[idx].astype(np.float64)
    else:
        b_nearest = va.copy()
    denom = max(args.a_init_weight + args.b_init_weight, EPS)
    init = (args.a_init_weight * va + args.b_init_weight * b_nearest) / denom
    edges, neighbors = mesh_edges_and_neighbors(faces, len(va))
    return PieceMesh(label, va, vb_pulled, init, b_nearest, faces, edges, neighbors)


def sample_points(arr: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap > 0 and len(arr) > cap:
        return arr[rng.permutation(len(arr))[:cap]]
    return arr


def extract_relations(labels: np.ndarray, *, max_contact_samples: int, max_exterior_samples: int, rng: np.random.Generator) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, np.ndarray]]:
    contacts: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    exterior: dict[int, list[tuple[float, float, float]]] = {}
    d, h, w = labels.shape
    directions = [
        (1, 0, 0, np.array([0.5, 0.0, 0.0])),
        (0, 1, 0, np.array([0.0, 0.5, 0.0])),
        (0, 0, 1, np.array([0.0, 0.0, 0.5])),
    ]
    for dz, dy, dx, off in directions:
        a = labels[: d - dz if dz else d, : h - dy if dy else h, : w - dx if dx else w]
        b = labels[dz:, dy:, dx:]
        zz, yy, xx = np.nonzero(a != b)
        for z, y, x in zip(zz.tolist(), yy.tolist(), xx.tolist()):
            la = int(a[z, y, x])
            lb = int(b[z, y, x])
            ctr = np.array([float(z), float(y), float(x)]) + off
            if la > 0 and lb > 0:
                key = (la, lb) if la < lb else (lb, la)
                contacts.setdefault(key, []).append((float(ctr[0]), float(ctr[1]), float(ctr[2])))
            elif la > 0:
                exterior.setdefault(la, []).append((float(ctr[0]), float(ctr[1]), float(ctr[2])))
            elif lb > 0:
                exterior.setdefault(lb, []).append((float(ctr[0]), float(ctr[1]), float(ctr[2])))
    # Volume-border exterior faces.
    border_specs = [
        (0, 0, np.array([-0.5, 0.0, 0.0])),
        (0, d - 1, np.array([0.5, 0.0, 0.0])),
        (1, 0, np.array([0.0, -0.5, 0.0])),
        (1, h - 1, np.array([0.0, 0.5, 0.0])),
        (2, 0, np.array([0.0, 0.0, -0.5])),
        (2, w - 1, np.array([0.0, 0.0, 0.5])),
    ]
    for axis, idx, off in border_specs:
        if axis == 0:
            coords = np.argwhere(labels[idx, :, :] > 0)
            for y, x in coords.tolist():
                lab = int(labels[idx, y, x])
                ctr = np.array([float(idx), float(y), float(x)]) + off
                exterior.setdefault(lab, []).append(tuple(float(v) for v in ctr))
        elif axis == 1:
            coords = np.argwhere(labels[:, idx, :] > 0)
            for z, x in coords.tolist():
                lab = int(labels[z, idx, x])
                ctr = np.array([float(z), float(idx), float(x)]) + off
                exterior.setdefault(lab, []).append(tuple(float(v) for v in ctr))
        else:
            coords = np.argwhere(labels[:, :, idx] > 0)
            for z, y in coords.tolist():
                lab = int(labels[z, y, idx])
                ctr = np.array([float(z), float(y), float(idx)]) + off
                exterior.setdefault(lab, []).append(tuple(float(v) for v in ctr))
    c_out = {k: sample_points(np.asarray(v, dtype=np.float64), max_contact_samples, rng) for k, v in contacts.items()}
    e_out = {k: sample_points(np.asarray(v, dtype=np.float64), max_exterior_samples, rng) for k, v in exterior.items()}
    return c_out, e_out


def nearest_vertex(mesh: PieceMesh, pt: np.ndarray) -> int:
    tree = cKDTree(mesh.verts_init)
    _d, idx = tree.query(pt)
    return int(idx)


def build_model(la: np.ndarray, lb: np.ndarray, transforms: dict[int, Transform3], args: argparse.Namespace) -> Model3:
    rng = np.random.default_rng(args.seed)
    labels = sorted((set(labels_in(la)) & set(labels_in(lb)) & set(transforms)) - {0})
    meshes = {label: build_piece_mesh(label, la, lb, transforms[label], args) for label in labels}
    offsets: dict[int, int] = {}
    off = 0
    for label in labels:
        offsets[label] = off
        off += meshes[label].verts_init.size

    a_rel, a_ext_rel = extract_relations(la, max_contact_samples=args.max_contact_samples, max_exterior_samples=args.max_exterior_samples, rng=rng)
    b_rel, b_ext_rel = extract_relations(lb, max_contact_samples=args.max_contact_samples, max_exterior_samples=args.max_exterior_samples, rng=rng)

    a_contacts: list[ContactSample] = []
    b_contacts: list[ContactSample] = []
    a_exterior: list[ExteriorSample] = []
    b_exterior: list[ExteriorSample] = []
    label_set = set(labels)
    for (i, j), pts in a_rel.items():
        if i not in label_set or j not in label_set:
            continue
        for p in pts:
            a_contacts.append(ContactSample(i, nearest_vertex(meshes[i], p), j, nearest_vertex(meshes[j], p)))
    for i, pts in a_ext_rel.items():
        if i not in label_set:
            continue
        for p in pts:
            a_exterior.append(ExteriorSample(i, nearest_vertex(meshes[i], p), tuple(float(v) for v in p)))
    for (i, j), pts_b in b_rel.items():
        if i not in label_set or j not in label_set:
            continue
        for p_b in pts_b:
            pi = transforms[i].inverse(p_b[None, :])[0]
            pj = transforms[j].inverse(p_b[None, :])[0]
            b_contacts.append(ContactSample(i, nearest_vertex(meshes[i], pi), j, nearest_vertex(meshes[j], pj)))
    for i, pts_b in b_ext_rel.items():
        if i not in label_set:
            continue
        for p_b in pts_b:
            pi = transforms[i].inverse(p_b[None, :])[0]
            b_exterior.append(ExteriorSample(i, nearest_vertex(meshes[i], pi), tuple(float(v) for v in p_b)))
    return Model3(labels, offsets, meshes, transforms, a_contacts, b_contacts, a_exterior, b_exterior)


def pack(model: Model3, use: str = "init") -> np.ndarray:
    x = np.zeros(model.variable_count, dtype=np.float64)
    for label in model.labels:
        mesh = model.meshes[label]
        arr = mesh.verts_init if use == "init" else mesh.verts_a
        x[model.offsets[label] : model.offsets[label] + arr.size] = arr.reshape(-1)
    return x


def unpack_piece(model: Model3, x: np.ndarray, label: int) -> np.ndarray:
    mesh = model.meshes[label]
    off = model.offsets[label]
    return x[off : off + mesh.verts_init.size].reshape((-1, 3))


def var_index(model: Model3, label: int, vidx: int, comp: int) -> int:
    return model.offsets[label] + vidx * 3 + comp


def add_vector_equation(rows: list[int], cols: list[int], data: list[float], rhs: list[float], row_counter: list[int], terms: list[tuple[int, int, np.ndarray, float]], target: np.ndarray, weight: float) -> None:
    for comp in range(3):
        r = row_counter[0]
        row_counter[0] += 1
        for base, vidx, mat, sign in terms:
            for c in range(3):
                val = weight * sign * float(mat[comp, c])
                if abs(val) > 0.0:
                    rows.append(r)
                    cols.append(base + vidx * 3 + c)
                    data.append(val)
        rhs.append(weight * float(target[comp]))


def build_linear_system(model: Model3, args: argparse.Namespace) -> tuple[coo_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_counter = [0]
    eye = np.eye(3, dtype=np.float64)

    for c in model.a_contacts:
        add_vector_equation(
            rows,
            cols,
            data,
            rhs,
            row_counter,
            [(model.offsets[c.label_a], c.idx_a, eye, 1.0), (model.offsets[c.label_b], c.idx_b, eye, -1.0)],
            np.zeros(3),
            args.a_contact_weight,
        )
    for c in model.b_contacts:
        mi, oi = model.transforms[c.label_a].matrix_and_offset()
        mj, oj = model.transforms[c.label_b].matrix_and_offset()
        add_vector_equation(
            rows,
            cols,
            data,
            rhs,
            row_counter,
            [(model.offsets[c.label_a], c.idx_a, mi, 1.0), (model.offsets[c.label_b], c.idx_b, mj, -1.0)],
            oj - oi,
            args.b_contact_weight,
        )
    for e in model.a_exterior:
        add_vector_equation(
            rows,
            cols,
            data,
            rhs,
            row_counter,
            [(model.offsets[e.label], e.idx, eye, 1.0)],
            np.asarray(e.target, dtype=np.float64),
            args.a_exterior_weight,
        )
    for e in model.b_exterior:
        mat, off = model.transforms[e.label].matrix_and_offset()
        add_vector_equation(
            rows,
            cols,
            data,
            rhs,
            row_counter,
            [(model.offsets[e.label], e.idx, mat, 1.0)],
            np.asarray(e.target, dtype=np.float64) - off,
            args.b_exterior_weight,
        )
    for label in model.labels:
        mesh = model.meshes[label]
        base = model.offsets[label]
        for vidx, target in enumerate(mesh.verts_init):
            add_vector_equation(rows, cols, data, rhs, row_counter, [(base, vidx, eye, 1.0)], target, args.shape_weight)
        for vidx, target in enumerate(mesh.verts_a):
            add_vector_equation(rows, cols, data, rhs, row_counter, [(base, vidx, eye, 1.0)], target, args.a_shape_weight)
        for vidx, target in enumerate(mesh.b_nearest):
            add_vector_equation(rows, cols, data, rhs, row_counter, [(base, vidx, eye, 1.0)], target, args.b_shape_weight)
        if args.fair_weight > 0.0:
            init = mesh.verts_init
            for vidx, neigh in enumerate(mesh.neighbors):
                if len(neigh) == 0:
                    continue
                target = init[vidx] - init[neigh].mean(axis=0)
                terms: list[tuple[int, int, np.ndarray, float]] = [(base, vidx, eye, 1.0)]
                for n in neigh.tolist():
                    terms.append((base, int(n), eye, -1.0 / float(len(neigh))))
                add_vector_equation(rows, cols, data, rhs, row_counter, terms, target, args.fair_weight)

    mat = coo_matrix((data, (rows, cols)), shape=(row_counter[0], model.variable_count), dtype=np.float64)
    return mat.tocsr(), np.asarray(rhs, dtype=np.float64)


def solve_model(model: Model3, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    a, b = build_linear_system(model, args)
    result = lsqr(a, b, atol=args.lsqr_tol, btol=args.lsqr_tol, iter_lim=args.lsqr_iter)
    x = result[0]
    return x, {
        "lsqr_istop": float(result[1]),
        "lsqr_iterations": float(result[2]),
        "lsqr_r1norm": float(result[3]),
        "lsqr_r2norm": float(result[4]),
        "linear_rows": float(a.shape[0]),
        "linear_cols": float(a.shape[1]),
        "linear_nnz": float(a.nnz),
    }


def rms(vals: list[float]) -> float:
    return float(math.sqrt(sum(v * v for v in vals) / max(len(vals), 1)))


def contact_metrics(model: Model3, x: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    a_vals = []
    for c in model.a_contacts:
        vi = unpack_piece(model, x, c.label_a)[c.idx_a]
        vj = unpack_piece(model, x, c.label_b)[c.idx_b]
        a_vals.append(float(np.linalg.norm(vi - vj)))
    b_vals = []
    for c in model.b_contacts:
        vi = unpack_piece(model, x, c.label_a)[c.idx_a]
        vj = unpack_piece(model, x, c.label_b)[c.idx_b]
        b_vals.append(float(np.linalg.norm(model.transforms[c.label_a].forward(vi[None, :])[0] - model.transforms[c.label_b].forward(vj[None, :])[0])))
    ae = []
    for e in model.a_exterior:
        v = unpack_piece(model, x, e.label)[e.idx]
        ae.append(float(np.linalg.norm(v - np.asarray(e.target))))
    be = []
    for e in model.b_exterior:
        v = unpack_piece(model, x, e.label)[e.idx]
        be.append(float(np.linalg.norm(model.transforms[e.label].forward(v[None, :])[0] - np.asarray(e.target))))
    out.update(
        {
            "a_contact_rms": rms(a_vals),
            "b_contact_rms": rms(b_vals),
            "a_exterior_rms": rms(ae),
            "b_exterior_rms": rms(be),
            "a_contact_count": float(len(a_vals)),
            "b_contact_count": float(len(b_vals)),
            "a_exterior_count": float(len(ae)),
            "b_exterior_count": float(len(be)),
        }
    )
    disp = []
    for label in model.labels:
        verts = unpack_piece(model, x, label)
        disp.extend(np.linalg.norm(verts - model.meshes[label].verts_init, axis=1).tolist())
    out["template_displacement_rms"] = rms([float(v) for v in disp])
    return out


def write_obj(path: Path, meshes: dict[int, tuple[np.ndarray, np.ndarray]], *, transforms: dict[int, Transform3] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    offset = 1
    for label in sorted(meshes):
        verts, faces = meshes[label]
        if transforms is not None:
            verts = transforms[label].forward(verts)
        lines.append(f"o piece_{label}")
        for v in verts:
            lines.append(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}")
        for tri in faces:
            a, b, c = tri.tolist()
            lines.append(f"f {a + offset} {b + offset} {c + offset}")
        offset += len(verts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="3D Zhou-seeded common-boundary surface optimizer")
    p.add_argument("--labels-a", required=True)
    p.add_argument("--labels-b", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="exp384_zhou3d_common_boundary")
    p.add_argument("--seed", type=int, default=2012)
    p.add_argument("--max-contact-samples", type=int, default=450)
    p.add_argument("--max-exterior-samples", type=int, default=650)
    p.add_argument("--a-init-weight", type=float, default=0.70)
    p.add_argument("--b-init-weight", type=float, default=0.30)
    p.add_argument("--a-contact-weight", type=float, default=10.0)
    p.add_argument("--b-contact-weight", type=float, default=14.0)
    p.add_argument("--a-exterior-weight", type=float, default=2.5)
    p.add_argument("--b-exterior-weight", type=float, default=1.2)
    p.add_argument("--shape-weight", type=float, default=0.8)
    p.add_argument("--a-shape-weight", type=float, default=0.7)
    p.add_argument("--b-shape-weight", type=float, default=0.35)
    p.add_argument("--fair-weight", type=float, default=4.0)
    p.add_argument("--lsqr-tol", type=float, default=1e-7)
    p.add_argument("--lsqr-iter", type=int, default=1200)
    args = p.parse_args()

    start = time.perf_counter()
    out_dir = Path(args.out) if args.out else RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    la = decode_label_stack(Path(args.labels_a))
    lb_raw = decode_label_stack(Path(args.labels_b))
    lb, transforms, match_diag = geometry_match_labels_3d(la, lb_raw)
    write_label_stack(out_dir / "matched_labels_A", la)
    write_label_stack(out_dir / "matched_labels_B", lb)
    model = build_model(la, lb, transforms, args)
    x0 = pack(model, "init")
    start_metrics = contact_metrics(model, x0)
    x, solve_diag = solve_model(model, args)
    final_metrics = contact_metrics(model, x)

    meshes = {label: (unpack_piece(model, x, label), model.meshes[label].faces) for label in model.labels}
    write_obj(out_dir / "pieces_template.obj", meshes)
    write_obj(out_dir / "assembly_A.obj", meshes)
    write_obj(out_dir / "assembly_B.obj", meshes, transforms=transforms)
    init_meshes = {label: (model.meshes[label].verts_init, model.meshes[label].faces) for label in model.labels}
    write_obj(out_dir / "initial_template.obj", init_meshes)
    write_obj(out_dir / "initial_assembly_B.obj", init_meshes, transforms=transforms)

    piece_rows = []
    for label in model.labels:
        mesh = model.meshes[label]
        piece_rows.append(
            {
                "label": label,
                "vertices": len(mesh.verts_init),
                "faces": len(mesh.faces),
                "edges": len(mesh.edges),
                "zhou_mismatch": transforms[label].mismatch,
                "rot_index": transforms[label].rot_index,
                "shift_z": transforms[label].shift[0],
                "shift_y": transforms[label].shift[1],
                "shift_x": transforms[label].shift[2],
            }
        )
    write_csv(out_dir / "pieces.csv", piece_rows)

    summary = {
        "runtime_s": time.perf_counter() - start,
        "labels": float(len(model.labels)),
        "vertices_total": float(sum(len(model.meshes[label].verts_init) for label in model.labels)),
        "faces_total": float(sum(len(model.meshes[label].faces) for label in model.labels)),
        **{f"start_{k}": v for k, v in start_metrics.items()},
        **{f"final_{k}": v for k, v in final_metrics.items()},
        **solve_diag,
    }
    write_csv(out_dir / "summary.csv", [summary])
    payload = {
        "out_dir": str(out_dir),
        "labels_a": str(Path(args.labels_a)),
        "labels_b": str(Path(args.labels_b)),
        "matching": match_diag,
        "summary": summary,
        "weights": vars(args),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
