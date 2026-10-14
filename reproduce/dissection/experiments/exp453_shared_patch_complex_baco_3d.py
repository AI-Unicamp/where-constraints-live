from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import igl
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

import exp384_zhou3d_common_boundary_optimizer as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "zhou3d_shared_patch_complex_baco"
EPS = 1e-9
EYE3 = np.eye(3, dtype=np.float64)


# Mesh projection (point -> closest point on the original input surface).


@dataclass
class MeshProjector:
    vertices_zyx: np.ndarray
    faces: np.ndarray
    chunk_size: int = 20000

    def project(self, pts_zyx: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_zyx, dtype=np.float64)
        if len(pts) == 0:
            return pts.copy()
        chunks: list[np.ndarray] = []
        for start in range(0, len(pts), max(1, int(self.chunk_size))):
            q = pts[start : start + max(1, int(self.chunk_size))]
            _sqr, _idx, closest = igl.point_mesh_squared_distance(q, self.vertices_zyx, self.faces)
            out = np.asarray(closest, dtype=np.float64)
            chunks.append(out.reshape((1, 3)) if out.ndim == 1 else out)
        return np.vstack(chunks)


def load_aligned_projector(path: Path) -> MeshProjector:
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a mesh")
    # Meshes prepared by exp450 are written with base.write_obj: OBJ xyz is
    # solver x/y/z, so internal z/y/x is the reversed coordinate order.
    verts_zyx = np.asarray(mesh.vertices, dtype=np.float64)[:, [2, 1, 0]]
    return MeshProjector(verts_zyx, np.asarray(mesh.faces, dtype=np.int32))


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Centers of the boundary faces of a voxel mask (z/y/x coordinates)."""
    inside = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    pts: list[np.ndarray] = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, inside.shape[axis] - 1)
        hi[axis] = slice(1, inside.shape[axis])
        a = inside[tuple(lo)]
        b = inside[tuple(hi)]
        zz, yy, xx = np.nonzero(a != b)
        if len(zz):
            p = np.column_stack([zz, yy, xx]).astype(np.float64) - 1.0
            p[:, axis] += 0.5
            pts.append(p)
    return np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with s R src + t ~= dst.

    With allow_scale=False this is a rigid Kabsch fit (s = 1). The
    certificate compares against the input meshes moved rigidly, not against
    meshes rescaled to fit the voxelization: independent per-mesh scales would
    break the equal-volume premise of the problem."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = dc.T @ sc / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_fix[2, 2] = -1.0
    rot = u @ s_fix @ vt
    if allow_scale:
        var_s = float(np.mean(np.sum(sc * sc, axis=1)))
        scale = float(np.trace(np.diag(d) @ s_fix) / max(var_s, EPS))
    else:
        scale = 1.0
    t = mu_d - scale * rot @ mu_s
    return scale, rot, t


def refine_projector_alignment(
    proj: MeshProjector, mask: np.ndarray, iters: int, allow_scale: bool = False
) -> tuple[MeshProjector, dict[str, float]]:
    """ICP: refine the mesh's fit to the voxel grid boundary.

    The prepared meshes were aligned by a uniform bbox fit, which can be off
    when the voxelization (an SDF-upsampled legacy mask) filled its bounding
    box differently. The label grid is the frame everything else lives in, so
    the mesh is moved, not the grid.

    Rigid by default: a similarity ICP would rescale the source and target
    meshes independently, so the certificate would compare against rescaled
    inputs at possibly inconsistent scales. Scale can be enabled with
    allow_scale=True; either way the total applied scale is reported.
    """
    bnd = mask_boundary_points(mask)
    verts = proj.vertices_zyx.copy()
    diag: dict[str, float] = {}
    if len(bnd) == 0 or iters <= 0:
        return proj, {"align_iters": 0.0, "align_scale_total": 1.0}
    scale_total = 1.0
    for it in range(int(iters)):
        cur = MeshProjector(verts, proj.faces)
        closest = cur.project(bnd)
        resid = np.linalg.norm(closest - bnd, axis=1)
        if it == 0:
            diag["align_rms_before"] = rms(resid)
        scale, rot, t = umeyama_similarity(closest, bnd, allow_scale=allow_scale)
        scale_total *= scale
        verts = (scale * (rot @ verts.T)).T + t
    cur = MeshProjector(verts, proj.faces)
    resid = np.linalg.norm(cur.project(bnd) - bnd, axis=1)
    diag["align_rms_after"] = rms(resid)
    diag["align_iters"] = float(iters)
    diag["align_scale_total"] = float(scale_total)
    return cur, diag


# Shared patch complex.
#
# The complex is built ONCE in the template (A) frame from the matched label
# grid. 2-cells are lattice interface faces between two different labels
# (background counts as label 0), so every internal contact face is a single
# shared object referenced by both incident pieces: A-assembly contact closure
# is structural, not a residual. Vertices are lattice cube corners stored in
# doubled integer coordinates and shared across all pieces.
#
# Because the Zhou transforms are exact lattice maps, pushing a face of piece
# i through T_i lands on a lattice face of the B grid. The face of piece k
# that claims the same B-grid face is its exact partner: this gives a
# face-level and vertex-level bijection for every B-assembly contact, the 3D
# analogue of align_loop's sample bijection. No nearest-neighbor matching and
# no harmonic cross-parameterization is needed; the lattice IS the common
# parameterization, and seams fall out as tag-change curves.


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
    transforms: dict[int, base.Transform3]
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


def build_complex(la: np.ndarray, lb: np.ndarray, transforms: dict[int, base.Transform3]) -> Complex3:
    labels = sorted(set(base.labels_in(la)) & set(transforms))
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


# B-contact pairing: exact lattice face bijection between pieces in assembly B.


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


# Affine union-find: turn consistent B-contact closure relations into shared
# variables (structural identities) instead of residuals.


class AffineUnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rel_m = [EYE3.copy() for _ in range(n)]  # x_v = M x_parent + t
        self.rel_t = [np.zeros(3) for _ in range(n)]
        self.inconsistent: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    def find(self, v: int) -> tuple[int, np.ndarray, np.ndarray]:
        chain = []
        node = v
        while self.parent[node] != node:
            chain.append(node)
            node = self.parent[node]
        root = node
        m = EYE3
        t = np.zeros(3)
        for node in reversed(chain):
            m = self.rel_m[node] @ m
            t = self.rel_m[node] @ t + self.rel_t[node]
            self.parent[node] = root
            self.rel_m[node] = m
            self.rel_t[node] = t
        return root, self.rel_m[v] if chain else EYE3.copy(), self.rel_t[v] if chain else np.zeros(3)

    def union(self, v: int, w: int, a: np.ndarray, b: np.ndarray) -> bool:
        """Add relation x_w = A x_v + b. Returns True if merged structurally."""
        rv, mv, tv = self.find(v)
        rw, mw, tw = self.find(w)
        if rv == rw:
            lhs_m = a @ mv
            lhs_t = a @ tv + b
            if np.allclose(lhs_m, mw, atol=1e-8) and np.allclose(lhs_t, tw, atol=1e-6):
                return True
            self.inconsistent.append((v, w, a, b))
            return False
        # x_rw = Mw^T (A Mv x_rv + A tv + b - tw); relation matrices are
        # products of cube rotations, hence orthogonal.
        self.parent[rw] = rv
        self.rel_m[rw] = mw.T @ (a @ mv)
        self.rel_t[rw] = mw.T @ (a @ tv + b - tw)
        return True


def merge_relations(
    cx: Complex3, relations: list[tuple[int, int, int, int]]
) -> tuple[AffineUnionFind, list[tuple[int, int, int, int]], int, int]:
    """Merge closure relations into shared variables; return the soft remainder.

    Relations are T_i(x_v) == T_k(x_w), rewritten x_w = A x_v + b. Three cases:
    trivially structural (v == w with identity map: the two pieces keep the
    same relative pose in B, so the B contact face IS the shared A face),
    merged structurally (distinct vertices identified through the affine
    union-find), or soft (a positional constraint: the relation pins
    the vertex to a fixed-point subspace of the relative transform, or it
    closed a cycle with a different map).
    """
    uf = AffineUnionFind(len(cx.vert_keys))
    soft: list[tuple[int, int, int, int]] = []
    merged = 0
    trivial = 0
    for rel in relations:
        v, li, w, lk = rel
        a = cx.r_mats[lk].T @ cx.r_mats[li]
        b = cx.r_mats[lk].T @ (cx.r_offs[li] - cx.r_offs[lk])
        if v == w:
            if np.allclose(a, EYE3, atol=1e-9) and np.allclose(b, 0.0, atol=1e-9):
                trivial += 1
            else:
                soft.append(rel)
            continue
        if uf.union(v, w, a, b):
            merged += 1
        else:
            soft.append(rel)
    return uf, soft, merged, trivial


# Sub-voxel transform refinement. The lattice T_i maximize VOXEL overlap; the
# optimal rigid placement of the continuous piece can differ by sub-voxel
# amounts. Alternation: fix transforms, solve vertices (LSQR); fix vertices,
# solve per-cluster rigid Procrustes against the B-side targets.
#
# Pieces connected by a closure relation while having IDENTICAL lattice
# transforms share template walls that persist in assembly B; giving them
# different refinements would break that shared wall, so they are clustered
# and refined together (their relative motion is locked by the geometry).


def build_transform_clusters(cx: Complex3, relations: list[tuple[int, int, int, int]]) -> list[list[int]]:
    parent = {lab: lab for lab in cx.labels}

    def find(lab: int) -> int:
        while parent[lab] != lab:
            parent[lab] = parent[parent[lab]]
            lab = parent[lab]
        return lab

    for v, li, w, lk in relations:
        if li == lk:
            continue
        if np.array_equal(cx.mats[li], cx.mats[lk]) and np.array_equal(cx.offs[li], cx.offs[lk]):
            parent[find(li)] = find(lk)
    groups: dict[int, list[int]] = {}
    for lab in cx.labels:
        groups.setdefault(find(lab), []).append(lab)
    return [sorted(g) for g in sorted(groups.values())]


def weighted_rigid_fit(src: np.ndarray, dst: np.ndarray, wts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Weighted Kabsch: rigid (R, t) with R src + t ~= dst. No scale."""
    if len(src) < 3:
        return None
    w = wts / max(float(wts.sum()), EPS)
    mu_s = (src * w[:, None]).sum(axis=0)
    mu_d = (dst * w[:, None]).sum(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc * w[:, None]).T @ sc
    u, d, vt = np.linalg.svd(cov)
    if d[1] < 1e-12:
        return None  # degenerate correspondence set
    s_fix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_fix[2, 2] = -1.0
    rot = u @ s_fix @ vt
    t = mu_d - rot @ mu_s
    return rot, t


def refine_transform_sweep(
    cx: Complex3,
    cur: np.ndarray,
    relations: list[tuple[int, int, int, int]],
    clusters: list[list[int]],
    eb_by_vert: dict[int, list[int]],
    target_proj: MeshProjector,
    w_ext: float,
    w_closure: float,
) -> None:
    cluster_of = {lab: ci for ci, group in enumerate(clusters) for lab in group}
    ext_pairs: dict[int, list[tuple[int, int]]] = {}
    for v, labs in eb_by_vert.items():
        for lab in labs:
            ext_pairs.setdefault(cluster_of[lab], []).append((v, lab))
    contact_pairs: dict[int, list[tuple[int, int, int, int]]] = {}
    for v, li, w, lk in relations:
        ci, ck = cluster_of[li], cluster_of[lk]
        if ci == ck:
            continue  # locked together; preserved exactly by the common update
        contact_pairs.setdefault(ci, []).append((v, li, w, lk))
        contact_pairs.setdefault(ck, []).append((w, lk, v, li))

    for ci, group in enumerate(clusters):
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        wts: list[float] = []
        ep = ext_pairs.get(ci, [])
        if ep:
            pts_t = np.asarray([cur[v] for v, _lab in ep])
            pts_b = np.vstack([fwd(cx, lab, cur[v][None, :]) for v, lab in ep])
            proj_b = target_proj.project(pts_b)
            src.extend(pts_t)
            dst.extend(proj_b)
            wts.extend([w_ext] * len(ep))
        for v, li, w, lk in contact_pairs.get(ci, []):
            src.append(cur[v])
            dst.append(fwd(cx, lk, cur[w][None, :])[0])
            wts.append(w_closure)
        fit = weighted_rigid_fit(np.asarray(src), np.asarray(dst), np.asarray(wts)) if src else None
        if fit is None:
            continue
        rot, t = fit
        for lab in group:
            cx.r_mats[lab] = rot
            cx.r_offs[lab] = t


def transform_refinement_diag(cx: Complex3) -> dict[str, float]:
    angles = []
    shifts = []
    for lab in cx.labels:
        rel = cx.mats[lab].T @ cx.r_mats[lab]
        c = (np.trace(rel) - 1.0) * 0.5
        angles.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        shifts.append(float(np.linalg.norm(cx.r_offs[lab] - cx.offs[lab])))
    return {
        "refine_rot_deg_max": float(max(angles, default=0.0)),
        "refine_rot_deg_mean": float(np.mean(angles)) if angles else 0.0,
        "refine_shift_max": float(max(shifts, default=0.0)),
        "refine_shift_mean": float(np.mean(shifts)) if shifts else 0.0,
    }


# Per-piece self-intersection check (Moller tri-tri, spatial hashed).
# Self-intersection is invariant under rigid motion, so checking the template
# covers both assemblies.


def find_triangle_intersections(
    verts: np.ndarray,
    tris: np.ndarray,
    skip_pair=None,
    cell: float = 1.0,
) -> list[tuple[int, int]]:
    """Transversal triangle-pair intersections (Moller, spatial hashed).

    skip_pair(a, b) -> True excludes a candidate pair (e.g. adjacency).
    The default skips pairs sharing a vertex INDEX, which is correct inside
    the complex's single shared vertex pool."""
    if len(tris) == 0:
        return []
    if skip_pair is None:
        def skip_pair(a: int, b: int) -> bool:
            return bool(set(tris[a]) & set(tris[b]))
    tv = verts[tris]  # (n, 3, 3)
    lo = tv.min(axis=1)
    hi = tv.max(axis=1)
    cells: dict[tuple[int, int, int], list[int]] = {}
    lo_i = np.floor(lo / cell).astype(np.int64)
    hi_i = np.floor(hi / cell).astype(np.int64)
    for ti in range(len(tris)):
        for z in range(lo_i[ti, 0], hi_i[ti, 0] + 1):
            for y in range(lo_i[ti, 1], hi_i[ti, 1] + 1):
                for x in range(lo_i[ti, 2], hi_i[ti, 2] + 1):
                    cells.setdefault((z, y, x), []).append(ti)
    cand: set[tuple[int, int]] = set()
    rejected: set[tuple[int, int]] = set()
    for items in cells.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a > b:
                    a, b = b, a
                if (a, b) in cand or (a, b) in rejected:
                    continue
                if skip_pair(a, b):
                    rejected.add((a, b))
                    continue
                cand.add((a, b))
    if not cand:
        return []
    pairs = np.asarray(sorted(cand), dtype=np.int64)
    # AABB reject.
    keep = np.all(lo[pairs[:, 0]] <= hi[pairs[:, 1]] + 1e-12, axis=1) & np.all(
        lo[pairs[:, 1]] <= hi[pairs[:, 0]] + 1e-12, axis=1
    )
    pairs = pairs[keep]
    if len(pairs) == 0:
        return []
    t1 = tv[pairs[:, 0]]
    t2 = tv[pairs[:, 1]]
    eps = 1e-9

    def plane_split(tri_pts: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
        d = -np.einsum("ij,ij->i", n, tri_pts[:, 0])
        dist = np.einsum("ij,ikj->ik", n, other) + d[:, None]
        return n, dist

    n2, d1 = plane_split(t2, t1)  # distances of t1 verts to t2 plane
    n1, d2 = plane_split(t1, t2)
    sep1 = np.all(d1 > eps, axis=1) | np.all(d1 < -eps, axis=1)
    sep2 = np.all(d2 > eps, axis=1) | np.all(d2 < -eps, axis=1)
    alive = ~(sep1 | sep2)
    coplanar = np.all(np.abs(d1) <= eps, axis=1)
    alive &= ~coplanar  # exact coplanar contact is not transversal intersection
    idx = np.nonzero(alive)[0]
    if len(idx) == 0:
        return []

    hits: list[tuple[int, int]] = []
    for k in idx.tolist():
        # Interval overlap along the plane-intersection line (Moller).
        line = np.cross(n1[k], n2[k])
        axis = int(np.argmax(np.abs(line)))

        def interval(tri_pts: np.ndarray, dist: np.ndarray) -> tuple[float, float] | None:
            proj = tri_pts[:, axis]
            pts = []
            for a in range(3):
                b = (a + 1) % 3
                da, db = dist[a], dist[b]
                if (da > eps and db < -eps) or (da < -eps and db > eps):
                    f = da / (da - db)
                    pts.append(proj[a] + f * (proj[b] - proj[a]))
                elif abs(da) <= eps:
                    pts.append(proj[a])
            if len(pts) < 2:
                return None
            return min(pts), max(pts)

        i1 = interval(t1[k], d1[k])
        i2 = interval(t2[k], d2[k])
        if i1 is None or i2 is None:
            continue
        if min(i1[1], i2[1]) - max(i1[0], i2[0]) > 1e-7:
            hits.append((int(pairs[k, 0]), int(pairs[k, 1])))
    return hits


def find_self_intersections(verts: np.ndarray, tris: np.ndarray, cell: float = 1.0) -> list[tuple[int, int]]:
    return find_triangle_intersections(verts, tris, skip_pair=None, cell=cell)


def count_self_intersections(verts: np.ndarray, tris: np.ndarray) -> int:
    return len(find_self_intersections(verts, tris))


# Cross-piece intersection detection.
#
# Besides the per-piece self-intersection check, interpenetration between
# different pieces is detected explicitly (otherwise it would only show up,
# aliased, in the sampled certificate). Two checks:
#
#   Assembly A (template frame): all pieces share one vertex pool, so pairs
#   of triangles from different pieces that share a vertex index are
#   legitimate contact (shared walls/edges) and are skipped; everything else
#   that intersects transversally is a foldover through a wall.
#
#   Assembly B: each piece's copy of the pool moves by its own motion, so
#   index sharing no longer implies geometric coincidence. Two (label, vertex)
#   occurrences are B-coincident iff they are related by a closure relation,
#   or they are the same template vertex used by two pieces with identical
#   refined motions. Triangle pairs whose vertex sets meet a common
#   B-coincidence class are legitimate contact and skipped; coplanar shared
#   contact faces are already excluded by the Moller coplanar test.
#
# Limitation: a pair that both shares a coincidence class and also crosses
# elsewhere is skipped; such folds near seams are still
# caught by the per-piece self check or the certificate.


def b_coincidence_groups(cx: Complex3, relations: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], int]:
    """Union-find over (label, vertex) occurrences that coincide in B."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(n: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(n, n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for v, li, w, lk in relations:
        union((li, v), (lk, w))
    # Same template vertex used by pieces whose refined motions are identical.
    vert_labels: dict[int, set[int]] = {}
    for s in cx.sides:
        for v in cx.faces[s.face].corners:
            vert_labels.setdefault(v, set()).add(s.label)
    same_motion: dict[tuple[int, int], bool] = {}
    for v, labs in vert_labels.items():
        labs_l = sorted(labs)
        for i in range(len(labs_l)):
            for j in range(i + 1, len(labs_l)):
                li, lk = labs_l[i], labs_l[j]
                key = (li, lk)
                ok = same_motion.get(key)
                if ok is None:
                    ok = np.array_equal(cx.r_mats[li], cx.r_mats[lk]) and np.array_equal(
                        cx.r_offs[li], cx.r_offs[lk]
                    )
                    same_motion[key] = ok
                if ok:
                    union((li, v), (lk, v))
    groups: dict[tuple[int, int], int] = {}
    roots: dict[tuple[int, int], int] = {}
    for n in list(parent):
        r = find(n)
        groups[n] = roots.setdefault(r, len(roots))
    return groups


def find_cross_intersections(
    cx: Complex3,
    cur: np.ndarray,
    piece_tris: dict[int, np.ndarray],
    *,
    transformed: bool,
    coincidence: dict[tuple[int, int], int] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Transversal intersections between triangles of DIFFERENT pieces.

    Returns (label_i, template_tri_row_i, label_k, template_tri_row_k).
    transformed=False checks assembly A (template frame, shared pool);
    transformed=True checks assembly B (per-piece moved copies)."""
    labels = [lab for lab in cx.labels if len(piece_tris[lab])]
    if not labels:
        return []
    tri_blocks: list[np.ndarray] = []
    tri_piece: list[int] = []
    tri_row: list[int] = []
    if not transformed:
        verts = cur
        for lab in labels:
            t = piece_tris[lab]
            tri_blocks.append(t)
            tri_piece.extend([lab] * len(t))
            tri_row.extend(range(len(t)))
    else:
        verts_blocks = []
        offset = 0
        for lab in labels:
            verts_blocks.append(fwd(cx, lab, cur))
            t = piece_tris[lab] + offset
            tri_blocks.append(t)
            tri_piece.extend([lab] * len(t))
            tri_row.extend(range(len(t)))
            offset += len(cur)
        verts = np.vstack(verts_blocks)
    tris = np.vstack(tri_blocks)
    tri_piece_arr = tri_piece
    npool = len(cur)

    if not transformed:
        def skip(a: int, b: int) -> bool:
            if tri_piece_arr[a] == tri_piece_arr[b]:
                return True  # self handled separately (rigid invariant)
            return bool(set(tris[a]) & set(tris[b]))
    else:
        coin = coincidence or {}

        def skip(a: int, b: int) -> bool:
            la, lb_ = tri_piece_arr[a], tri_piece_arr[b]
            if la == lb_:
                return True
            ga = {coin.get((la, int(v) % npool), -1 - int(v) % npool) for v in tris[a]}
            gb = {coin.get((lb_, int(v) % npool), -2_000_000_000) for v in tris[b]}
            return bool(ga & gb)

    hits = find_triangle_intersections(verts, tris, skip_pair=skip)
    return [
        (tri_piece_arr[a], tri_row[a], tri_piece_arr[b], tri_row[b])
        for a, b in hits
    ]


# Piece-piece non-penetration springs.
# The paired B-contacts cannot overlap (shared variables), but conflict and
# unpaired sides -- the Zhou mismatch surfaces -- have no partner and nothing
# kept them out of neighboring pieces. One-sided springs: any vertex strictly
# inside ANOTHER piece (winding number, all ordered pairs, both assemblies) is
# pulled toward the outside of that piece's surface. Targets are frozen per
# outer iteration, like the exterior pins; springs act on union-find roots, so
# structural closure stays exact. Preventing penetration during the solve
# also leaves less work for the untangle pass, which reduces coverage.


def piece_vertex_ids(cx: Complex3) -> dict[int, np.ndarray]:
    acc: dict[int, set[int]] = {lab: set() for lab in cx.labels}
    for s in cx.sides:
        acc[s.label].update(cx.faces[s.face].corners)
    return {lab: np.asarray(sorted(vs), dtype=np.int64) for lab, vs in acc.items()}


def penetration_springs(
    cx: Complex3,
    cur: np.ndarray,
    piece_vids: dict[int, np.ndarray],
    piece_tris: dict[int, np.ndarray],
    transformed: bool,
    min_depth: float,
    clearance: float,
    max_step: float = 0.0,
    exempt: set[int] | None = None,
) -> list[tuple[int, int, np.ndarray, float]]:
    """(vertex, owning label, target point in assembly frame, depth) for every
    vertex of a piece strictly inside another piece, deeper than min_depth.
    Exact shared-contact vertices sit ON the neighbor surface (depth ~0), so
    the depth guard keeps them out. max_step > 0 caps the pull per iteration:
    deep penetrations walk out over several ICP rounds instead of being yanked
    the full depth in one solve (which folds the surface faster than fairing
    can recover). exempt vertices are never pushed: a template variable moves
    identically in both assemblies, so pushing an A-pinned vertex out of a
    piece in B dents the source surface in A by the same amount."""
    springs: list[tuple[int, int, np.ndarray, float]] = []
    pos = {lab: (fwd(cx, lab, cur) if transformed else cur) for lab in cx.labels}
    for lab_k in cx.labels:
        tris_k = piece_tris[lab_k]
        if len(tris_k) == 0:
            continue
        verts_k = pos[lab_k]
        own = verts_k[piece_vids[lab_k]]
        lo = own.min(axis=0)
        hi = own.max(axis=0)
        proj_k: MeshProjector | None = None
        for lab_i in cx.labels:
            if lab_i == lab_k:
                continue
            vids = piece_vids[lab_i]
            if exempt:
                vids = vids[[v not in exempt for v in vids.tolist()]]
                if len(vids) == 0:
                    continue
            p = pos[lab_i][vids]
            box = np.all((p > lo) & (p < hi), axis=1)
            if not box.any():
                continue
            q = np.ascontiguousarray(p[box])
            qv = vids[box]
            # atleast_1d: a single-point query yields a 0-d winding mask, and
            # 0-d boolean indexing would ADD a dimension instead of selecting.
            ins = np.atleast_1d(inside_mesh(verts_k, tris_k, q))
            if not ins.any():
                continue
            if proj_k is None:
                proj_k = MeshProjector(verts_k, tris_k.astype(np.int32))
            qi = np.ascontiguousarray(q[ins])
            qvi = qv[ins]
            closest = proj_k.project(qi)
            delta = closest - qi
            depth = np.linalg.norm(delta, axis=1)
            for j in np.nonzero(depth > min_depth)[0]:
                d = float(depth[j])
                pull = d + clearance
                if max_step > 0.0:
                    pull = min(pull, max_step)
                tgt = qi[j] + delta[j] * (pull / d)
                springs.append((int(qvi[j]), lab_i, tgt, d))
    return springs


def untangle_pass(
    cx: Complex3,
    cur: np.ndarray,
    x: np.ndarray,
    vmap: list[tuple[int, np.ndarray, np.ndarray]],
    root_col: dict[int, int],
    umbrellas: list[np.ndarray],
    iters: int,
    coincidence: dict[tuple[int, int], int] | None = None,
    check_cross: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Resolve residual self-intersections AND cross-piece foldovers by
    relaxing the involved vertices toward their umbrella average. Nudges are
    applied to the union-find ROOT variables, so the structural closure
    identities stay exact. Both per-piece self-intersections and
    interpenetration between different pieces are relaxed."""
    piece_tris = {label: piece_triangles(cx, label)[0] for label in cx.labels}

    def measure() -> tuple[int, int, int, set[int]]:
        involved: set[int] = set()
        n_self = 0
        for tris in piece_tris.values():
            hits = find_self_intersections(cur, tris)
            n_self += len(hits)
            for a, b in hits:
                involved.update(int(t) for t in tris[a])
                involved.update(int(t) for t in tris[b])
        n_cross_a = n_cross_b = 0
        if check_cross:
            hits_a = find_cross_intersections(cx, cur, piece_tris, transformed=False)
            n_cross_a = len(hits_a)
            for li, ra, lk, rb in hits_a:
                involved.update(int(t) for t in piece_tris[li][ra])
                involved.update(int(t) for t in piece_tris[lk][rb])
            hits_b = find_cross_intersections(
                cx, cur, piece_tris, transformed=True, coincidence=coincidence
            )
            n_cross_b = len(hits_b)
            for li, ra, lk, rb in hits_b:
                involved.update(int(t) for t in piece_tris[li][ra])
                involved.update(int(t) for t in piece_tris[lk][rb])
        return n_self, n_cross_a, n_cross_b, involved

    n_self = n_cross_a = n_cross_b = 0
    for _ in range(max(0, int(iters))):
        n_self, n_cross_a, n_cross_b, involved = measure()
        if n_self + n_cross_a + n_cross_b == 0:
            break
        delta_acc: dict[int, np.ndarray] = {}
        cnt: dict[int, int] = {}
        for v in involved:
            nb = umbrellas[v]
            if len(nb) == 0:
                continue
            delta = 0.6 * (cur[nb].mean(axis=0) - cur[v])
            root, m, _t = vmap[v]
            delta_acc[root] = delta_acc.get(root, np.zeros(3)) + m.T @ delta
            cnt[root] = cnt.get(root, 0) + 1
        if not delta_acc:
            break
        for root, dsum in delta_acc.items():
            col = root_col[root] * 3
            x[col : col + 3] += dsum / float(cnt[root])
        cur = vertex_positions(vmap, root_col, x)
    if n_self + n_cross_a + n_cross_b > 0:
        n_self, n_cross_a, n_cross_b, _ = measure()
    return x, cur, {
        "self": int(n_self),
        "cross_a": int(n_cross_a),
        "cross_b": int(n_cross_b),
    }


# Patch decomposition: connected components of faces with identical dual
# (A-role, B-role) classification. Seam curves are the tag-change edges.


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


# Linear system on union-find roots.


class System:
    def __init__(self, vmap: list[tuple[int, np.ndarray, np.ndarray]], root_col: dict[int, int]) -> None:
        self.vmap = vmap
        self.root_col = root_col
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.rhs: list[float] = []
        self.nrows = 0

    def add(self, terms: list[tuple[int, np.ndarray]], rhs: np.ndarray, weight: float) -> None:
        acc: dict[int, np.ndarray] = {}
        r = np.asarray(rhs, dtype=np.float64).copy()
        for vidx, cmat in terms:
            root, m, t = self.vmap[vidx]
            cur = acc.get(root)
            add = cmat @ m
            acc[root] = add if cur is None else cur + add
            r = r - cmat @ t
        for comp in range(3):
            row = self.nrows
            self.nrows += 1
            for root, cmat in acc.items():
                basecol = self.root_col[root] * 3
                for c in range(3):
                    val = weight * float(cmat[comp, c])
                    if val != 0.0:
                        self.rows.append(row)
                        self.cols.append(basecol + c)
                        self.data.append(val)
            self.rhs.append(weight * float(r[comp]))

    def solve(self, tol: float, iters: int) -> tuple[np.ndarray, dict[str, float]]:
        ncols = 3 * len(self.root_col)
        mat = coo_matrix((self.data, (self.rows, self.cols)), shape=(self.nrows, ncols)).tocsr()
        result = lsqr(mat, np.asarray(self.rhs), atol=tol, btol=tol, iter_lim=iters)
        return result[0], {
            "lsqr_istop": float(result[1]),
            "lsqr_iterations": float(result[2]),
            "lsqr_r1norm": float(result[3]),
            "linear_rows": float(mat.shape[0]),
            "linear_cols": float(mat.shape[1]),
            "linear_nnz": float(mat.nnz),
        }


def vertex_positions(vmap: list[tuple[int, np.ndarray, np.ndarray]], root_col: dict[int, int], x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(vmap), 3), dtype=np.float64)
    for v, (root, m, t) in enumerate(vmap):
        xr = x[root_col[root] * 3 : root_col[root] * 3 + 3]
        out[v] = m @ xr + t
    return out


# Outputs and validation.


def piece_triangles(cx: Complex3, label: int) -> tuple[np.ndarray, list[str]]:
    tris: list[tuple[int, int, int]] = []
    tags: list[str] = []
    own = sorted((s for s in cx.sides if s.label == label), key=lambda s: (s.patch, s.face))
    for s in own:
        c = cx.faces[s.face].corners
        quad = list(c) if s.is_neg else list(c)[::-1]
        flag = "" if s.own_ok else "_conflict"
        if s.b_role == s.label:
            flag = "_binterior"
        tag = f"a{s.a_role}_b{s.b_role}{flag}_p{s.patch}"
        tris.append((quad[0], quad[1], quad[2]))
        tris.append((quad[0], quad[2], quad[3]))
        tags.extend((tag, tag))
    return np.asarray(tris, dtype=np.int64), tags


def write_tagged_obj(path: Path, cx: Complex3, verts: np.ndarray, *, transformed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    offset = 1
    for label in cx.labels:
        tris, tags = piece_triangles(cx, label)
        if len(tris) == 0:
            continue
        used = sorted(set(int(v) for tri in tris for v in tri))
        remap = {v: i for i, v in enumerate(used)}
        pv = verts[used]
        if transformed:
            pv = fwd(cx, label, pv)
        lines.append(f"o piece_{label}")
        for v in pv:
            lines.append(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}")
        current = None
        for tri, tag in zip(tris.tolist(), tags):
            if tag != current:
                current = tag
                lines.append(f"g piece_{label}_{tag}")
            a, b, c = (remap[int(t)] for t in tri)
            lines.append(f"f {a + offset} {b + offset} {c + offset}")
        offset += len(pv)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def piece_watertight_stats(cx: Complex3, label: int) -> dict[str, int]:
    uses: dict[tuple[int, int], int] = {}
    for s in cx.sides:
        if s.label != label:
            continue
        c = cx.faces[s.face].corners
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])):
            key = (a, b) if a < b else (b, a)
            uses[key] = uses.get(key, 0) + 1
    vals = list(uses.values())
    return {
        "edges": len(vals),
        "open_edges": sum(1 for v in vals if v == 1),
        "overused_edges": sum(1 for v in vals if v > 2),
    }


def tri_area_sum(verts: np.ndarray, tris: np.ndarray) -> float:
    if len(tris) == 0:
        return 0.0
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Signed volume of a closed triangle surface (divergence theorem)."""
    if len(tris) == 0:
        return 0.0
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def inside_mesh(
    verts_zyx: np.ndarray, faces: np.ndarray, queries_zyx: np.ndarray, orient_sign: float = 0.0
) -> np.ndarray:
    verts = np.ascontiguousarray(verts_zyx, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    queries = np.ascontiguousarray(queries_zyx, dtype=np.float64)
    if hasattr(igl, "fast_winding_number_for_meshes"):
        w = igl.fast_winding_number_for_meshes(verts, tris, queries)
    else:
        w = igl.fast_winding_number(verts, tris, queries)
    w = np.asarray(w)
    # abs(w) > 0.5 cannot distinguish a piece that covers a region from a
    # piece that is inside-out over it. The zyx storage order mirrors
    # handedness relative to the OBJ xyz frame, so the convention is signed
    # winding normalized by the mesh's global orientation (the sign of its
    # signed volume). orient_sign = 0 asks this function to derive it; a piece
    # whose orientation disagrees with the complex convention reads as not
    # covering, and is flagged separately.
    if orient_sign == 0.0:
        orient_sign = math.copysign(1.0, signed_volume(verts_zyx, faces)) if len(faces) else 1.0
    return (w * orient_sign) > 0.5


def validation_certificate(
    cx: Complex3,
    verts: np.ndarray,
    la: np.ndarray,
    lb: np.ndarray,
    source_proj: MeshProjector,
    target_proj: MeshProjector,
    dilate: int = 2,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Winding-number certificate.

    - Fractions are normalized by the shape volume (winding-inside count of
      the input mesh over the query set), i.e. fractions of the input solid.
      Values normalized by the dilated query count are also emitted, as
      *_fraction_of_queries, for comparison; they are smaller (on a 64^3
      bunny by roughly a factor of 1.2).
    - Signed winding with an explicit orientation convention: the complex's
      canonical piece orientation is the majority sign of the per-piece
      signed volumes; a piece whose sign disagrees is counted in
      pieces_orientation_flipped and does not count as covering.
    - Query dilation is configurable (default 2), so spill farther than one
      voxel outside the voxel hull is measured.
    - Per-pair overlap counts are returned as rows (assembly, label_i,
      label_k, voxels), localizing B-interpenetration by piece pair for
      feedback to the discrete stage.

    This is a sampled diagnostic, not a proof: voxel-center queries alias
    sub-voxel features. Exact volumes would require mesh Booleans, which the
    closed piece surfaces allow.
    """
    out: dict[str, float] = {}
    pair_rows: list[dict[str, object]] = []
    piece_meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    piece_signs: dict[int, float] = {}
    for label in cx.labels:
        tris, _tags = piece_triangles(cx, label)
        piece_meshes[label] = (verts, tris)
        v = signed_volume(verts, tris)  # orientation is rigid-invariant
        piece_signs[label] = math.copysign(1.0, v) if v != 0.0 else 0.0
    sign_sum = sum(piece_signs.values())
    canonical = math.copysign(1.0, sign_sum) if sign_sum != 0.0 else 1.0
    flipped = [lab for lab, s in piece_signs.items() if s != canonical]
    out["pieces_orientation_flipped"] = float(len(flipped))
    out["piece_orientation_sign"] = float(canonical)

    for name, vol, proj, transformed in (
        ("A", la, source_proj, False),
        ("B", lb, target_proj, True),
    ):
        fg = ndimage.binary_dilation(vol > 0, iterations=max(1, int(dilate)))
        queries = np.argwhere(fg).astype(np.float64)
        if len(queries) == 0:
            continue
        counts = np.zeros(len(queries), dtype=np.int32)
        inside_by_label: dict[int, np.ndarray] = {}
        for label in cx.labels:
            pv, tris = piece_meshes[label]
            pv2 = fwd(cx, label, pv) if transformed else pv
            ins = inside_mesh(pv2, tris, queries, orient_sign=canonical)
            inside_by_label[label] = ins
            counts += ins.astype(np.int32)
        shape_inside = inside_mesh(proj.vertices_zyx, proj.faces, queries)
        union = counts >= 1
        nq = float(len(queries))
        shape_n = float(np.count_nonzero(shape_inside))
        denom = max(shape_n, 1.0)
        unc = float(np.count_nonzero(shape_inside & ~union))
        spl = float(np.count_nonzero(union & ~shape_inside))
        ovl = float(np.count_nonzero(counts >= 2))
        out[f"certificate_{name}_queries"] = nq
        out[f"certificate_{name}_dilate"] = float(dilate)
        out[f"certificate_{name}_shape_voxels"] = shape_n
        out[f"certificate_{name}_uncovered_voxels"] = unc
        out[f"certificate_{name}_spill_voxels"] = spl
        out[f"certificate_{name}_overlap_voxels"] = ovl
        # Shape-volume-normalized fractions (the stated definition).
        out[f"certificate_{name}_uncovered_fraction"] = unc / denom
        out[f"certificate_{name}_spill_fraction"] = spl / denom
        out[f"certificate_{name}_overlap_fraction"] = ovl / denom
        # Legacy query-normalized values, for continuity with older runs.
        out[f"certificate_{name}_uncovered_fraction_of_queries"] = unc / nq
        out[f"certificate_{name}_spill_fraction_of_queries"] = spl / nq
        out[f"certificate_{name}_overlap_fraction_of_queries"] = ovl / nq
        # Per-pair overlap localization.
        labs = sorted(inside_by_label)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                n_ij = int(np.count_nonzero(inside_by_label[labs[i]] & inside_by_label[labs[j]]))
                if n_ij:
                    pair_rows.append(
                        {
                            "assembly": name,
                            "label_i": labs[i],
                            "label_k": labs[j],
                            "overlap_voxels": n_ij,
                            "overlap_fraction_of_shape": n_ij / denom,
                        }
                    )
    return out, pair_rows


def rms(vals: np.ndarray) -> float:
    if len(vals) == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(vals)))))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# Main pipeline.


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Shared patch complex 3D BACO: non-manifold B-rep of the Zhou partition with dual "
            "(A-role, B-role) classified patches, structural contact closure, seam-aware fairing, "
            "exterior pinning to the original input meshes, and a winding-number certificate."
        )
    )
    p.add_argument("--labels-a", required=True)
    p.add_argument("--labels-b", required=True)
    p.add_argument("--source-mesh", required=True)
    p.add_argument("--target-mesh", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="exp453_shared_patch_complex")
    p.add_argument("--outer-iters", type=int, default=4)
    p.add_argument("--align-iters", type=int, default=15)
    p.add_argument("--w-exterior-a", type=float, default=4.0)
    p.add_argument("--w-exterior-b", type=float, default=2.5)
    p.add_argument("--w-closure", type=float, default=25.0)
    p.add_argument("--w-fair", type=float, default=1.0)
    p.add_argument("--w-seam", type=float, default=1.0)
    p.add_argument("--w-anchor", type=float, default=0.12)
    p.add_argument("--w-nopen", type=float, default=3.0, help="One-sided non-penetration springs between every piece pair in both assemblies (0 disables). Vertices strictly inside another piece are pulled toward the outside of its surface; targets refresh every outer iteration.")
    p.add_argument("--nopen-clearance", type=float, default=0.05, help="Push-out distance beyond the penetrated surface (voxel units).")
    p.add_argument("--nopen-min-depth", type=float, default=0.02, help="Ignore penetrations shallower than this (guards machine-exact shared-contact vertices, which sit at depth ~0).")
    p.add_argument("--nopen-max-step", type=float, default=0.75, help="Cap on the outward pull per outer iteration (voxel units); deep penetrations walk out over several rounds instead of folding the surface in one solve.")
    p.add_argument("--nopen-start-iter", type=int, default=2, help="First outer iteration that adds penetration springs (the initial blend geometry gives unreliable winding numbers).")
    p.add_argument("--nopen-exempt-a-pinned", action=argparse.BooleanOptionalAction, default=True, help="Do not push A-exterior-pinned vertices in assembly B: those pushes dent the source surface (same template variable), trading A fidelity for B overlap.")
    p.add_argument("--lsqr-tol", type=float, default=1e-8)
    p.add_argument("--lsqr-iter", type=int, default=1500)
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refine-transforms", action=argparse.BooleanOptionalAction, default=False, help="Sub-voxel rigid refinement of the piece transforms (per persistent-contact cluster) alternated with the vertex solve.")
    p.add_argument("--cert-slack", type=float, default=0.01, help="Pass threshold on certificate fractions (uncovered/spill/overlap) per assembly.")
    p.add_argument("--untangle-iters", type=int, default=12, help="Post-solve local relaxation rounds to remove residual self- and cross-piece intersections (root-space, closure-preserving).")
    p.add_argument("--cross-untangle", action=argparse.BooleanOptionalAction, default=True, help="Also detect and relax cross-piece foldovers in both assemblies (FIX: previously only per-piece self-intersections were checked).")
    p.add_argument("--align-scale", action=argparse.BooleanOptionalAction, default=False, help="Allow uniform scale in the mesh-to-grid ICP. Default OFF (FIX): rigid alignment, so the certificate compares against the actual input meshes.")
    p.add_argument("--cert-dilate", type=int, default=2, help="Dilation radius (voxels) of the certificate query grid (was hardwired 1).")
    args = p.parse_args()

    start = time.perf_counter()
    out_dir = Path(args.out) if args.out else RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    la = base.decode_label_stack(Path(args.labels_a))
    lb_raw = base.decode_label_stack(Path(args.labels_b))
    lb, transforms, match_diag = base.geometry_match_labels_3d(la, lb_raw)
    base.write_label_stack(out_dir / "matched_labels_A", la)
    base.write_label_stack(out_dir / "matched_labels_B", lb)

    source_proj = load_aligned_projector(Path(args.source_mesh))
    target_proj = load_aligned_projector(Path(args.target_mesh))
    source_proj, align_a_diag = refine_projector_alignment(source_proj, la > 0, args.align_iters, allow_scale=args.align_scale)
    target_proj, align_b_diag = refine_projector_alignment(target_proj, lb > 0, args.align_iters, allow_scale=args.align_scale)
    base.write_obj(out_dir / "source_mesh_refined.obj", {1: (source_proj.vertices_zyx, source_proj.faces)})
    base.write_obj(out_dir / "target_mesh_refined.obj", {1: (target_proj.vertices_zyx, target_proj.faces)})

    cx = build_complex(la, lb, transforms)
    relations, paired_sides, unpaired_sides = pair_b_contacts(cx)
    uf, soft_relations, merged_count, trivial_count = merge_relations(cx, relations)
    patch, patch_tags, edge_faces = decompose_patches(cx)
    cls, umbrellas, seams = classify_vertices(cx, patch, edge_faces)
    clusters = build_transform_clusters(cx, relations)

    def rebuild_merge() -> tuple:
        u, s, m, t = merge_relations(cx, relations)
        vm = [u.find(v) for v in range(len(cx.vert_keys))]
        rt = sorted({r for r, _m, _t in vm})
        return u, s, m, t, vm, {r: i for i, r in enumerate(rt)}

    vmap = [uf.find(v) for v in range(len(cx.vert_keys))]
    roots = sorted({r for r, _m, _t in vmap})
    root_col = {r: i for i, r in enumerate(roots)}

    # Exterior pin sets.
    ea_verts: set[int] = set()
    eb_pairs: set[tuple[int, int]] = set()
    for s in cx.sides:
        corners = cx.faces[s.face].corners
        if s.a_role == 0:
            ea_verts.update(corners)
        if s.b_role == 0 and s.own_ok:
            eb_pairs.update((v, s.label) for v in corners)
    ea_list = sorted(ea_verts)
    eb_list = sorted(eb_pairs)
    eb_by_vert: dict[int, list[int]] = {}
    for v, lab in eb_list:
        eb_by_vert.setdefault(v, []).append(lab)

    piece_tris_all = {label: piece_triangles(cx, label)[0] for label in cx.labels}
    piece_vids = piece_vertex_ids(cx)
    nopen_a: list[tuple[int, int, np.ndarray, float]] = []
    nopen_b: list[tuple[int, int, np.ndarray, float]] = []

    # Initialization: pull exterior vertices onto the input surfaces.
    cur = cx.vert_pos.copy()
    last_x = None
    solve_diag: dict[str, float] = {}
    for it in range(max(0, int(args.outer_iters)) + 1):
        # Projection targets from current geometry.
        targets_a = {}
        if ea_list:
            idx = np.asarray(ea_list, dtype=np.int64)
            proj_pts = source_proj.project(cur[idx])
            targets_a = {v: proj_pts[i] for i, v in enumerate(ea_list)}
        targets_b = {}
        if eb_list:
            pts = np.asarray([fwd(cx, lab, cur[v][None, :])[0] for v, lab in eb_list])
            proj_pts = target_proj.project(pts)
            targets_b = {pair: proj_pts[i] for i, pair in enumerate(eb_list)}
        if it == 0:
            # Blend the raw lattice positions toward their pins, then solve.
            for v, t in targets_a.items():
                cur[v] = t
            accum: dict[int, list[np.ndarray]] = {}
            for (v, lab), tb in targets_b.items():
                accum.setdefault(v, []).append(inv(cx, lab, tb[None, :])[0])
            for v, pulls in accum.items():
                pulled = np.mean(np.asarray(pulls), axis=0)
                cur[v] = 0.5 * (cur[v] + pulled) if v in targets_a else pulled
            init_positions = cur.copy()
            continue

        if args.w_nopen > 0 and it >= args.nopen_start_iter:
            b_exempt = ea_verts if args.nopen_exempt_a_pinned else None
            nopen_a = penetration_springs(cx, cur, piece_vids, piece_tris_all, False, args.nopen_min_depth, args.nopen_clearance, args.nopen_max_step)
            nopen_b = penetration_springs(cx, cur, piece_vids, piece_tris_all, True, args.nopen_min_depth, args.nopen_clearance, args.nopen_max_step, b_exempt)

        sysm = System(vmap, root_col)
        for v, t in targets_a.items():
            sysm.add([(v, EYE3)], t, args.w_exterior_a)
        for (v, lab), tb in targets_b.items():
            sysm.add([(v, cx.r_mats[lab])], tb - cx.r_offs[lab], args.w_exterior_b)
        for v, _lab, tgt, _d in nopen_a:
            sysm.add([(v, EYE3)], tgt, args.w_nopen)
        for v, lab, tgt, _d in nopen_b:
            sysm.add([(v, cx.r_mats[lab])], tgt - cx.r_offs[lab], args.w_nopen)
        # Only relations that could NOT be merged structurally need residual rows.
        for v, li, w, lk in soft_relations:
            sysm.add(
                [(v, cx.r_mats[li]), (w, -cx.r_mats[lk])],
                cx.r_offs[lk] - cx.r_offs[li],
                args.w_closure,
            )
        for v in range(len(cx.vert_keys)):
            if cls[v] == 0 and len(umbrellas[v]) > 0:
                n = float(len(umbrellas[v]))
                terms = [(v, EYE3)] + [(int(u), -EYE3 / n) for u in umbrellas[v]]
                sysm.add(terms, np.zeros(3), args.w_fair)
            elif cls[v] == 1:
                a, b = seams[v]
                sysm.add([(v, EYE3), (a, -0.5 * EYE3), (b, -0.5 * EYE3)], np.zeros(3), args.w_seam)
            sysm.add([(v, EYE3)], init_positions[v], args.w_anchor)

        x, solve_diag = sysm.solve(args.lsqr_tol, args.lsqr_iter)
        cur = vertex_positions(vmap, root_col, x)
        last_x = x

        if args.refine_transforms and it < max(0, int(args.outer_iters)):
            refine_transform_sweep(
                cx, cur, relations, clusters, eb_by_vert, target_proj, args.w_exterior_b, args.w_closure
            )
            uf, soft_relations, merged_count, trivial_count, vmap, root_col = rebuild_merge()

    if last_x is None:
        # --outer-iters 0: no linear solve ran. Build a root vector from the
        # initialized positions so the untangle pass still has variables.
        last_x = np.zeros(3 * len(root_col), dtype=np.float64)
        for root, col in root_col.items():
            last_x[col * 3 : col * 3 + 3] = cur[root]
        cur = vertex_positions(vmap, root_col, last_x)

    # B-coincidence classes: which (label, vertex) occurrences are the same
    # geometric point in assembly B (needed to distinguish legitimate contact
    # from interpenetration in the cross-piece checks).
    coincidence = b_coincidence_groups(cx, relations)

    def intersection_snapshot() -> dict[str, int]:
        n_self = sum(len(find_self_intersections(cur, t)) for t in piece_tris_all.values())
        n_ca = n_cb = 0
        if args.cross_untangle:
            n_ca = len(find_cross_intersections(cx, cur, piece_tris_all, transformed=False))
            n_cb = len(
                find_cross_intersections(
                    cx, cur, piece_tris_all, transformed=True, coincidence=coincidence
                )
            )
        return {"self": n_self, "cross_a": n_ca, "cross_b": n_cb}

    before = intersection_snapshot()
    after = dict(before)
    if args.untangle_iters > 0:
        last_x, cur, after = untangle_pass(
            cx,
            cur,
            last_x,
            vmap,
            root_col,
            umbrellas,
            args.untangle_iters,
            coincidence=coincidence,
            check_cross=bool(args.cross_untangle),
        )
    untangled_resolved = float(
        (before["self"] + before["cross_a"] + before["cross_b"])
        - (after["self"] + after["cross_a"] + after["cross_b"])
    )

    # Metrics.
    metrics: dict[str, float] = {
        "untangle_resolved": untangled_resolved,
        "nopen_springs_a": float(len(nopen_a)),
        "nopen_springs_b": float(len(nopen_b)),
        "self_intersections_before_untangle": float(before["self"]),
        "cross_intersections_A_before_untangle": float(before["cross_a"]),
        "cross_intersections_B_before_untangle": float(before["cross_b"]),
        "cross_intersections_A_total": float(after["cross_a"]),
        "cross_intersections_B_total": float(after["cross_b"]),
    }
    if ea_list:
        idx = np.asarray(ea_list, dtype=np.int64)
        d = np.linalg.norm(source_proj.project(cur[idx]) - cur[idx], axis=1)
        metrics["exterior_a_rms"] = rms(d)
        metrics["exterior_a_max"] = float(d.max())
        metrics["exterior_a_vertices"] = float(len(idx))
    if eb_list:
        pts = np.asarray([fwd(cx, lab, cur[v][None, :])[0] for v, lab in eb_list])
        d = np.linalg.norm(target_proj.project(pts) - pts, axis=1)
        metrics["exterior_b_rms"] = rms(d)
        metrics["exterior_b_max"] = float(d.max())
        metrics["exterior_b_vertices"] = float(len(eb_list))
    closure_res = []
    for v, li, w, lk in relations:
        pa = fwd(cx, li, cur[v][None, :])[0]
        pb = fwd(cx, lk, cur[w][None, :])[0]
        closure_res.append(float(np.linalg.norm(pa - pb)))
    closure_res = np.asarray(closure_res)
    metrics["b_closure_relations"] = float(len(relations))
    metrics["b_closure_merged"] = float(merged_count)
    metrics["b_closure_trivial"] = float(trivial_count)
    metrics["b_closure_soft"] = float(len(soft_relations))
    metrics["b_closure_rms"] = rms(closure_res)
    metrics["b_closure_max"] = float(closure_res.max()) if len(closure_res) else 0.0
    # Exterior-in-both conflict: same template point pinned to both inputs.
    extext = [v for v in ea_verts if v in eb_by_vert]
    if extext:
        idx = np.asarray(sorted(extext), dtype=np.int64)
        pa = source_proj.project(cur[idx])
        gaps = []
        for i, v in enumerate(idx.tolist()):
            for lab in eb_by_vert[v]:
                tb = fwd(cx, lab, cur[v][None, :])
                pb = inv(cx, lab, target_proj.project(tb))[0]
                gaps.append(float(np.linalg.norm(pa[i] - pb)))
        metrics["extext_vertices"] = float(len(extext))
        metrics["extext_gap_rms"] = rms(np.asarray(gaps))
        metrics["extext_gap_max"] = float(max(gaps))
    conflict_sides = sum(1 for s in cx.sides if not s.own_ok)
    binterior_sides = sum(1 for s in cx.sides if s.b_role == s.label)
    metrics.update(
        {
            "faces": float(len(cx.faces)),
            "sides": float(len(cx.sides)),
            "vertices": float(len(cx.vert_keys)),
            "variables_after_merge": float(3 * len(roots)),
            "patches": float(len(patch_tags)),
            "sides_paired_b": float(paired_sides),
            "sides_unpaired_b": float(unpaired_sides),
            "sides_conflict_own": float(conflict_sides),
            "sides_b_interior": float(binterior_sides),
            "verts_interior": float(int(np.sum(cls == 0))),
            "verts_seam": float(int(np.sum(cls == 1))),
            "verts_corner": float(int(np.sum(cls == 2))),
        }
    )

    # Per-piece watertightness and self-intersections (rigid-invariant, so the
    # template check covers both assemblies).
    piece_rows = []
    for label in cx.labels:
        stats = piece_watertight_stats(cx, label)
        tris, _ = piece_triangles(cx, label)
        piece_rows.append(
            {
                "label": label,
                "faces": int(np.sum([1 for s in cx.sides if s.label == label])),
                "area_final": tri_area_sum(cur, tris),
                "signed_volume": signed_volume(cur, tris),
                "self_intersections": count_self_intersections(cur, tris),
                "zhou_mismatch": transforms[label].mismatch,
                **stats,
            }
        )
    metrics["open_edges_total"] = float(sum(r["open_edges"] for r in piece_rows))
    metrics["self_intersections_total"] = float(sum(r["self_intersections"] for r in piece_rows))
    metrics["transform_clusters"] = float(len(clusters))
    metrics.update(transform_refinement_diag(cx))

    # Per-patch reporting: localizes where the discrete solution is wrong, the
    # continuous analogue of Zhou's mismatch-cluster signal.
    dist_a: dict[int, float] = {}
    if ea_list:
        idx = np.asarray(ea_list, dtype=np.int64)
        d = np.linalg.norm(source_proj.project(cur[idx]) - cur[idx], axis=1)
        dist_a = {v: float(d[i]) for i, v in enumerate(ea_list)}
    dist_b: dict[tuple[int, int], float] = {}
    if eb_list:
        pts = np.asarray([fwd(cx, lab, cur[v][None, :])[0] for v, lab in eb_list])
        d = np.linalg.norm(target_proj.project(pts) - pts, axis=1)
        dist_b = {pair: float(d[i]) for i, pair in enumerate(eb_list)}

    patch_faces: dict[int, list[int]] = {}
    for fi in range(len(cx.faces)):
        patch_faces.setdefault(int(patch[fi]), []).append(fi)
    patch_rows = []
    for pid, tag in sorted(patch_tags.items()):
        ln, lp, sl_n, br_n, ok_n, sl_p, br_p, ok_p = tag
        fis = patch_faces.get(pid, [])
        verts_p = sorted({v for fi in fis for v in cx.faces[fi].corners})
        tris = np.asarray(
            [(c[0], c[1], c[2]) for fi in fis for c in (cx.faces[fi].corners,)]
            + [(c[0], c[2], c[3]) for fi in fis for c in (cx.faces[fi].corners,)],
            dtype=np.int64,
        )
        row: dict[str, object] = {
            "patch": pid,
            "label_neg": ln,
            "label_pos": lp,
            "b_role_neg_side": br_n,
            "b_role_pos_side": br_p,
            "own_ok_neg": ok_n,
            "own_ok_pos": ok_p,
            "faces": len(fis),
            "area_final": tri_area_sum(cur, tris) if len(tris) else 0.0,
            "is_conflict": int(
                (not ok_n and sl_n >= 0)
                or (not ok_p and sl_p >= 0)
                or (sl_n > 0 and br_n == sl_n)
                or (sl_p > 0 and br_p == sl_p)
            ),
        }
        da = [dist_a[v] for v in verts_p if v in dist_a]
        if da:
            row["exterior_a_mean"] = float(np.mean(da))
            row["exterior_a_max"] = float(np.max(da))
        db = []
        for v in verts_p:
            for lab in (sl_n, sl_p):
                if lab > 0 and (v, lab) in dist_b:
                    db.append(dist_b[(v, lab)])
        if db:
            row["exterior_b_mean"] = float(np.mean(db))
            row["exterior_b_max"] = float(np.max(db))
        if da and db:
            row["extext_patch"] = 1
        patch_rows.append(row)

    write_tagged_obj(out_dir / "pieces_template.obj", cx, cur, transformed=False)
    write_tagged_obj(out_dir / "assembly_A.obj", cx, cur, transformed=False)
    write_tagged_obj(out_dir / "assembly_B.obj", cx, cur, transformed=True)
    write_tagged_obj(out_dir / "initial_template.obj", cx, init_positions, transformed=False)
    write_csv(out_dir / "pieces.csv", piece_rows)
    write_csv(out_dir / "patches.csv", patch_rows)

    if args.validate:
        cert, overlap_pair_rows = validation_certificate(
            cx, cur, la, lb, source_proj, target_proj, dilate=args.cert_dilate
        )
        metrics.update(cert)
        write_csv(out_dir / "overlap_pairs.csv", overlap_pair_rows or [{"assembly": "", "label_i": "", "label_k": "", "overlap_voxels": 0, "overlap_fraction_of_shape": 0.0}])
        # Audit trail: discrete mismatch -> certificate failure. M counts
        # uncovered cells on both sides, so M/2 is the rigid-copy lower bound
        # on per-assembly uncoverage; conflict-patch area localizes it.
        m_total = float(match_diag["total_mismatch"])
        metrics["audit_mismatch_total"] = m_total
        metrics["audit_mismatch_half"] = 0.5 * m_total
        metrics["audit_conflict_patch_area"] = float(
            sum(float(r["area_final"]) for r in patch_rows if r["is_conflict"])
        )
        metrics["audit_patch_area_total"] = float(sum(float(r["area_final"]) for r in patch_rows))
        slack = float(args.cert_slack)
        # "clean" requires zero cross-piece intersections in the source
        # assembly and no inside-out pieces. Cross-B intersections are not part
        # of it: conflict regions do interpenetrate in B and are already charged
        # to the B overlap fraction; their count is reported.
        clean = (
            metrics["self_intersections_total"] == 0.0
            and metrics["open_edges_total"] == 0.0
            and metrics.get("cross_intersections_A_total", 0.0) == 0.0
            and metrics.get("pieces_orientation_flipped", 0.0) == 0.0
        )
        for name in ("A", "B"):
            ok = (
                metrics.get(f"certificate_{name}_uncovered_fraction", 1.0) <= slack
                and metrics.get(f"certificate_{name}_spill_fraction", 1.0) <= slack
                and metrics.get(f"certificate_{name}_overlap_fraction", 1.0) <= slack
            )
            metrics[f"certificate_{name}_pass"] = float(ok and clean)
        metrics["certificate_pass"] = float(
            metrics.get("certificate_A_pass", 0.0) > 0 and metrics.get("certificate_B_pass", 0.0) > 0
        )

    summary = {
        "runtime_s": time.perf_counter() - start,
        "labels": float(len(cx.labels)),
        **{f"source_{k}": v for k, v in align_a_diag.items()},
        **{f"target_{k}": v for k, v in align_b_diag.items()},
        **metrics,
        **solve_diag,
    }
    write_csv(out_dir / "summary.csv", [summary])
    payload = {
        "out_dir": str(out_dir),
        "labels_a": str(Path(args.labels_a)),
        "labels_b": str(Path(args.labels_b)),
        "source_mesh": str(Path(args.source_mesh)),
        "target_mesh": str(Path(args.target_mesh)),
        "matching": match_diag,
        "summary": summary,
        "args": vars(args),
        "notes": (
            "Shared patch complex (non-manifold B-rep) common-boundary optimizer. The complex is built once in the "
            "template frame by multi-material surface extraction on the matched Zhou label grid: every contact face "
            "between two pieces is one shared 2-cell, so A-assembly closure is structural. Each face side is dually "
            "classified by its role in B (push through the piece's lattice transform and read the B grid); because "
            "the transforms are exact lattice maps, B-assembly contacts pair into an exact face- and vertex-level "
            "bijection between the two incident pieces, which is merged into shared variables via an affine "
            "union-find (structural B closure; inconsistent relations fall back to strong soft residuals and are "
            "counted). Patches are connected components of the dual (A-role, B-role) classification, seams are the "
            "tag-change curves (1D faired), patch interiors are 2D faired, exterior-A/B patches are pinned to the "
            "original watertight input meshes by iterated closest-point projection, and (exterior,exterior) patches "
            "report their irreconcilable gap as a Zhou feedback signal. The harmonic cross-parameterization step of "
            "the design document is unnecessary here: the lattice itself supplies the material-point bijection. "
            "Validation is a winding-number certificate (piece overlap, coverage, spill vs the input meshes in both "
            "assemblies) plus per-piece watertightness of the extracted closed surfaces. "
            "FIXES (this revision): (1) certificate fractions are normalized by shape volume, matching the stated "
            "definition ('fraction of the input solid'); legacy query-normalized values kept as "
            "*_fraction_of_queries; (2) signed winding numbers with an explicit orientation convention replace "
            "abs(), and inside-out pieces are flagged (pieces_orientation_flipped) and fail 'clean'; (3) mesh-to-"
            "grid ICP is rigid by default (--align-scale re-enables similarity) and the applied scale is reported; "
            "(4) cross-piece intersections are now detected in BOTH assemblies (index-sharing exclusion in A; "
            "B-coincidence classes from the closure relations in B) and relaxed by the untangle pass in root space; "
            "cross-A intersections and orientation flips are part of the pass criterion; (5) per-pair overlap "
            "voxels are exported to overlap_pairs.csv as a discrete-stage feedback signal; (6) certificate query "
            "dilation is configurable (--cert-dilate, default 2); (7) --outer-iters 0 no longer crashes."
        ),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
