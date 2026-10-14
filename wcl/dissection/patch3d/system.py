"""Least-squares system over the complex vertices (union-find roots) and per-piece triangle access."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .complex import Complex3


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


