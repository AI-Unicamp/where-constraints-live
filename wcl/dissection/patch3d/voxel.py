"""Voxel label stacks of the 3D lattice search: decoding, piece matching (24 rotations), lattice transforms."""
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


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "outputs" / "zhou3d_common_boundary_optimizer"

EPS = 1e-9

ROTS = make_rots()

