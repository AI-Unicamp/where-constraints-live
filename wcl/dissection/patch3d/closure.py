"""B-assembly contact closure as shared variables (affine union-find) and sub-voxel rigid refinement."""
from __future__ import annotations

import numpy as np

from .alignment import MeshProjector
from .complex import Complex3, fwd
EPS = 1e-9
EYE3 = np.eye(3, dtype=np.float64)


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


