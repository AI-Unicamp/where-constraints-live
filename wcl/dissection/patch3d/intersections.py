"""Triangle-triangle intersection tests, non-penetration springs and the untangling pass."""
from __future__ import annotations

import numpy as np

from .alignment import MeshProjector
from .certificate import inside_mesh
from .complex import Complex3, fwd
from .system import piece_triangles, vertex_positions


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


