"""The shared patch complex: lattice interface faces shared by the pieces, built once in the template frame."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import numpy as np

from .voxel import Transform3, labels_in


def vertex_keys_for_face(c: tuple[int, int, int], axis: int, sign: int) -> list[tuple[int, int, int]]:
    # Doubled coordinates: voxel centers are even, cube corners odd.
    z, y, x = c
    center = [2 * z, 2 * y, 2 * x]
    fixed = center[axis] + sign
    other = [a for a in (0, 1, 2) if a != axis]
    out = []
    for s0, s1 in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        key = center.copy()
        key[axis] = fixed
        key[other[0]] += s0
        key[other[1]] += s1
        out.append((int(key[0]), int(key[1]), int(key[2])))
    # The permutation (axis, other0, other1) is odd for axis == 1, which flips
    # the winding of the corner cycle; reverse so every face has a consistent
    # outward orientation regardless of axis.
    flip = (sign < 0) != (axis == 1)
    if flip:
        out = [out[0], out[3], out[2], out[1]]
    return out


@dataclass
class Face:
    axis: int
    cell_neg: tuple[int, int, int]
    cell_pos: tuple[int, int, int]
    label_neg: int
    label_pos: int
    corners: tuple[int, int, int, int]  # ordered, outward normal = +axis (neg side)


@dataclass
class Side:
    face: int
    label: int       # the piece this side belongs to
    other_cell: tuple[int, int, int]
    a_role: int      # label across the face in A (0 = exterior in A)
    b_role: int      # label across the pushed face in B (0 = exterior in B)
    own_ok: bool     # B grid agrees that the piece owns its own voxel there
    is_neg: bool     # True if the piece sits on the negative side of the face
    patch: int = -1


@dataclass
class Complex3:
    vert_keys: list[tuple[int, int, int]]
    vert_pos: np.ndarray                 # raw template positions (key / 2)
    faces: list[Face]
    sides: list[Side]
    labels: list[int]
    transforms: dict[int, Transform3]
    mats: dict[int, np.ndarray] = field(default_factory=dict)   # lattice (frozen: combinatorics)
    offs: dict[int, np.ndarray] = field(default_factory=dict)
    r_mats: dict[int, np.ndarray] = field(default_factory=dict)  # refined rigid (geometry)
    r_offs: dict[int, np.ndarray] = field(default_factory=dict)


def fwd(cx: "Complex3", label: int, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64) @ cx.r_mats[label].T + cx.r_offs[label]


def inv(cx: "Complex3", label: int, pts: np.ndarray) -> np.ndarray:
    return (np.asarray(pts, dtype=np.float64) - cx.r_offs[label]) @ cx.r_mats[label]


def label_at(vol: np.ndarray, cell: tuple[int, int, int]) -> int:
    z, y, x = cell
    if 0 <= z < vol.shape[0] and 0 <= y < vol.shape[1] and 0 <= x < vol.shape[2]:
        return int(vol[z, y, x])
    return 0


def transform_cell(mat: np.ndarray, off: np.ndarray, cell: tuple[int, int, int]) -> tuple[int, int, int]:
    p = mat @ np.asarray(cell, dtype=np.float64) + off
    return (int(round(p[0])), int(round(p[1])), int(round(p[2])))


def build_complex(la: np.ndarray, lb: np.ndarray, transforms: dict[int, Transform3]) -> Complex3:
    labels = sorted(set(labels_in(la)) & set(transforms))
    mats = {}
    offs = {}
    r_mats = {}
    r_offs = {}
    for lab in labels:
        m, o = transforms[lab].matrix_and_offset()
        mats[lab] = m
        offs[lab] = o
        r_mats[lab] = m.copy()
        r_offs[lab] = o.copy()

    vert_ids: dict[tuple[int, int, int], int] = {}
    vert_keys: list[tuple[int, int, int]] = []

    def vid(key: tuple[int, int, int]) -> int:
        got = vert_ids.get(key)
        if got is not None:
            return got
        idx = len(vert_keys)
        vert_ids[key] = idx
        vert_keys.append(key)
        return idx

    faces: list[Face] = []
    padded = np.pad(la, 1, mode="constant", constant_values=0)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, padded.shape[axis] - 1)
        hi[axis] = slice(1, padded.shape[axis])
        a = padded[tuple(lo)]
        b = padded[tuple(hi)]
        zz, yy, xx = np.nonzero(a != b)
        for z, y, x in zip(zz.tolist(), yy.tolist(), xx.tolist()):
            cell_neg = [z - 1, y - 1, x - 1]
            cell_pos = cell_neg.copy()
            cell_pos[axis] += 1
            ln = int(a[z, y, x])
            lp = int(b[z, y, x])
            keys = vertex_keys_for_face(tuple(cell_neg), axis, 1)
            faces.append(
                Face(
                    axis=axis,
                    cell_neg=tuple(int(v) for v in cell_neg),
                    cell_pos=tuple(int(v) for v in cell_pos),
                    label_neg=ln,
                    label_pos=lp,
                    corners=tuple(vid(k) for k in keys),
                )
            )

    sides: list[Side] = []
    label_set = set(labels)
    for fi, f in enumerate(faces):
        for own_label, own_cell, other_cell, other_label, is_neg in (
            (f.label_neg, f.cell_neg, f.cell_pos, f.label_pos, True),
            (f.label_pos, f.cell_pos, f.cell_neg, f.label_neg, False),
        ):
            if own_label <= 0 or own_label not in label_set:
                continue
            m = mats[own_label]
            o = offs[own_label]
            b_own = label_at(lb, transform_cell(m, o, own_cell))
            b_oth = label_at(lb, transform_cell(m, o, other_cell))
            sides.append(
                Side(
                    face=fi,
                    label=own_label,
                    other_cell=other_cell,
                    a_role=other_label,
                    b_role=b_oth,
                    own_ok=(b_own == own_label),
                    is_neg=is_neg,
                )
            )

    pos = np.asarray(vert_keys, dtype=np.float64) * 0.5
    return Complex3(vert_keys, pos, faces, sides, labels, transforms, mats, offs, r_mats, r_offs)


def transformed_corner_keys(cx: Complex3, side: Side) -> list[tuple[int, int, int]]:
    m = cx.mats[side.label]
    o = cx.offs[side.label]
    out = []
    for vidx in cx.faces[side.face].corners:
        key = np.asarray(cx.vert_keys[vidx], dtype=np.float64)
        kb = m @ key + 2.0 * o
        out.append((int(round(kb[0])), int(round(kb[1])), int(round(kb[2]))))
    return out


def pair_b_contacts(cx: Complex3) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Return (vertex relations, paired face count, unpaired face count).

    Each relation is (v, label_i, w, label_k) meaning T_i(x_v) == T_k(x_w):
    template vertex v of piece i and template vertex w of piece k are the
    same material contact point in assembly B.
    """
    buckets: dict[frozenset[tuple[int, int, int]], list[tuple[int, dict[tuple[int, int, int], int]]]] = {}
    for si, side in enumerate(cx.sides):
        if not side.own_ok or side.b_role <= 0 or side.b_role == side.label:
            continue
        keys_b = transformed_corner_keys(cx, side)
        corner_map = {kb: cx.faces[side.face].corners[j] for j, kb in enumerate(keys_b)}
        buckets.setdefault(frozenset(keys_b), []).append((si, corner_map))

    relations: list[tuple[int, int, int, int]] = []
    paired = 0
    unpaired = 0
    seen_rel: set[tuple[int, int, int, int]] = set()
    for entries in buckets.values():
        if len(entries) != 2:
            unpaired += len(entries)
            continue
        (si, map_i), (sk, map_k) = entries
        li = cx.sides[si].label
        lk = cx.sides[sk].label
        paired += 2
        for kb, v in map_i.items():
            w = map_k.get(kb)
            if w is None:
                continue
            rel = (v, li, w, lk)
            if rel not in seen_rel and (w, lk, v, li) not in seen_rel:
                seen_rel.add(rel)
                relations.append(rel)
    return relations, paired, unpaired


def face_tag(cx: Complex3, fi: int, side_of_face: dict[tuple[int, bool], int]) -> tuple:
    f = cx.faces[fi]
    def side_info(is_neg: bool) -> tuple:
        si = side_of_face.get((fi, is_neg))
        if si is None:
            return (-1, -1, True)
        s = cx.sides[si]
        return (s.label, s.b_role, s.own_ok)
    return (f.label_neg, f.label_pos) + side_info(True) + side_info(False)


def decompose_patches(cx: Complex3) -> tuple[np.ndarray, dict[int, tuple], dict[tuple[int, int], list[int]]]:
    side_of_face = {(s.face, s.is_neg): i for i, s in enumerate(cx.sides)}
    tags = [face_tag(cx, fi, side_of_face) for fi in range(len(cx.faces))]

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(cx.faces):
        c = f.corners
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])):
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(fi)

    patch = np.full(len(cx.faces), -1, dtype=np.int64)
    patch_tags: dict[int, tuple] = {}
    next_patch = 0
    for start in range(len(cx.faces)):
        if patch[start] >= 0:
            continue
        pid = next_patch
        next_patch += 1
        patch_tags[pid] = tags[start]
        stack = [start]
        patch[start] = pid
        while stack:
            fi = stack.pop()
            c = cx.faces[fi].corners
            for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])):
                key = (a, b) if a < b else (b, a)
                for nf in edge_faces[key]:
                    if patch[nf] < 0 and tags[nf] == tags[fi]:
                        patch[nf] = pid
                        stack.append(nf)
    for si, s in enumerate(cx.sides):
        s.patch = int(patch[s.face])
    return patch, patch_tags, edge_faces


def classify_vertices(cx: Complex3, patch: np.ndarray, edge_faces: dict[tuple[int, int], list[int]]) -> tuple[np.ndarray, list[np.ndarray], list[list[int]]]:
    """Return (vertex class, umbrella neighbor lists, seam neighbor lists).

    class 0 = patch interior (2D umbrella fairing)
    class 1 = seam vertex with exactly two seam neighbors (1D curve fairing)
    class 2 = corner / junction vertex (anchored only)
    """
    nverts = len(cx.vert_keys)
    vert_patches: list[set[int]] = [set() for _ in range(nverts)]
    neigh: list[set[int]] = [set() for _ in range(nverts)]
    for fi, f in enumerate(cx.faces):
        c = f.corners
        for v in c:
            vert_patches[v].add(int(patch[fi]))
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])):
            neigh[a].add(b)
            neigh[b].add(a)

    seam_edges: set[tuple[int, int]] = set()
    for (a, b), fl in edge_faces.items():
        if len({int(patch[fi]) for fi in fl}) >= 2:
            seam_edges.add((a, b))
    seam_neigh: list[set[int]] = [set() for _ in range(nverts)]
    for a, b in seam_edges:
        seam_neigh[a].add(b)
        seam_neigh[b].add(a)

    cls = np.zeros(nverts, dtype=np.int8)
    for v in range(nverts):
        np_count = len(vert_patches[v])
        if np_count <= 1:
            cls[v] = 0
        elif np_count == 2 and len(seam_neigh[v]) == 2:
            cls[v] = 1
        else:
            cls[v] = 2
    umbrellas = [np.asarray(sorted(neigh[v]), dtype=np.int64) for v in range(nverts)]
    seams = [sorted(seam_neigh[v]) for v in range(nverts)]
    return cls, umbrellas, seams


