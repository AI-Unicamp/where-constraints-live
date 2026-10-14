"""BACO: boundary-aligned common-boundary optimisation of a two-figure dissection (2D).

Starting from a lattice dissection (piece labels in A and B and one lattice
isometry per piece, :mod:`wcl.dissection.lattice`), every piece becomes ONE
closed polygon (its *shared loop*) living in A's frame and placed in B by its
isometry. The loops are found by robust non-linear least squares so that

* pieces touching in A share their boundary (A-contact residuals),
* pieces touching in B share their boundary after their motions (B-contact residuals),
* boundary samples on A's / B's silhouette lie on that silhouette (exterior residuals),
* loops stay close to the lattice pieces and smooth (shape, fairness, edge-length terms).

Because each piece is a single geometry used in both assemblies, the two
assemblies are congruent piece by piece by construction; contact gaps are what
the optimisation reduces.

Examples
--------
>>> from wcl.dissection import lattice, baco
>>> lb, isos = lattice.match_labels(la, lb_raw)
>>> labels = sorted(set(isos) & set(np.unique(la).tolist()) - {0})
>>> prob = baco.problem(la, lb, isos, labels)
>>> x = baco.solve(prob)
>>> loops = prob.unpack(x)          # dict label -> (n, 2) polygon in A's pixel frame
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage import measure

from .lattice import LatticeIsometry

EPS = 1e-9


# ----------------------------------------------------------------------------- loops

def signed_area(p: np.ndarray) -> float:
    """Signed area of a closed polygon (n, 2); positive for counter-clockwise (x right, y up)."""
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def contour_loop(mask: np.ndarray) -> np.ndarray:
    """Outer boundary of a binary mask as a closed polygon (x, y) with positive signed area.

    The 0.5 level set is traced with marching squares; the longest contour is kept.
    """
    contours = measure.find_contours(mask.astype(np.float32), 0.5, fully_connected="high")
    if not contours:
        raise ValueError("empty contour")
    c = max(contours, key=len)
    p = np.column_stack([c[:, 1], c[:, 0]]).astype(np.float64)
    if len(p) > 1 and np.linalg.norm(p[0] - p[-1]) < 1e-9:
        p = p[:-1]
    return p[::-1] if signed_area(p) < 0 else p


def resample_closed(p: np.ndarray, n: int) -> np.ndarray:
    """``n`` points spaced uniformly by arc length along the closed polygon ``p``."""
    if len(p) < 3:
        raise ValueError("cannot resample fewer than 3 points")
    q = np.vstack([p, p[0]])
    seg = np.linalg.norm(q[1:] - q[:-1], axis=1)
    total = float(seg.sum())
    if total <= EPS:
        raise ValueError("zero-length contour")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.zeros((n, 2))
    j = 0
    for i, t in enumerate(np.linspace(0.0, total, n, endpoint=False)):
        while j + 1 < len(cum) and cum[j + 1] <= t:
            j += 1
        a = (t - cum[j]) / max(cum[j + 1] - cum[j], EPS)
        out[i] = (1.0 - a) * q[j] + a * q[j + 1]
    return out


def align_loop(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Cyclic shift (and possibly reversal) of ``candidate`` closest to ``reference`` in mean squared distance."""
    best, best_cost = candidate, float("inf")
    for c in (candidate, candidate[::-1].copy()):
        for s in range(len(reference)):
            shifted = np.roll(c, s, axis=0)
            cost = float(np.mean(np.sum((reference - shifted) ** 2, axis=1)))
            if cost < best_cost:
                best, best_cost = shifted, cost
    if signed_area(best) * signed_area(reference) < 0:
        best = best[::-1]
    return best.copy()


def boundary_samples(labels: np.ndarray, max_per_pair: int, rng: np.random.Generator) -> dict[tuple[int, int], np.ndarray]:
    """Midpoints of the pixel edges between different labels, grouped by (owner, neighbour).

    Each edge appears under both orderings. At most ``max_per_pair`` random
    samples are kept per pair. Label 0 is the exterior.
    """
    rows: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for (u, v), (ox, oy) in (((labels[:, :-1], labels[:, 1:]), (1.0, 0.5)), ((labels[:-1, :], labels[1:, :]), (0.5, 1.0))):
        ys, xs = np.nonzero(u != v)
        for y, x in zip(ys.tolist(), xs.tolist()):
            a, b = int(u[y, x]), int(v[y, x])
            rows.setdefault((a, b), []).append((x + ox, y + oy))
            rows.setdefault((b, a), []).append((x + ox, y + oy))
    out = {}
    for key, pts in rows.items():
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) > max_per_pair:
            arr = arr[rng.permutation(len(arr))[:max_per_pair]]
        out[key] = arr
    return out


# ----------------------------------------------------------------------------- problem

@dataclass
class Weights:
    """Residual weights (defaults are those of the dissertation runs)."""
    a_contact: float = 8.0
    b_contact: float = 14.0
    a_exterior: float = 3.5
    b_exterior: float = 1.4
    shape: float = 0.9
    a_shape: float = 0.8
    b_shape: float = 0.35
    fair: float = 8.0
    edge: float = 0.8


@dataclass
class Problem:
    """A BACO instance: initial loops, contact pairs and exterior targets.

    Loop ``l`` occupies ``x[offset[l] : offset[l] + 2 n]`` of the unknown vector
    (points stored as interleaved x, y). Contacts are index arrays into the
    stacked samples ``x.reshape(-1, 2)``.
    """
    labels: list[int]
    isos: dict[int, LatticeIsometry]
    init: dict[int, np.ndarray]            # initial loops (blend of A and pulled-back B contours)
    loops_a: dict[int, np.ndarray]
    loops_b: dict[int, np.ndarray]
    a_contacts: np.ndarray                 # (m, 2) sample index pairs touching in A
    b_contacts: np.ndarray                 # (m, 4) (label_i, sample_i, label_j, sample_j) touching in B
    a_exterior: tuple[np.ndarray, np.ndarray]   # sample indices, (m, 2) target points in A
    b_exterior: tuple[np.ndarray, np.ndarray, np.ndarray]  # labels, sample indices, targets in B
    offset: dict[int, int] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.init[self.labels[0]])

    def pack(self, loops: dict[int, np.ndarray]) -> np.ndarray:
        """dict label -> (n, 2) loop  ->  unknown vector."""
        return np.concatenate([loops[l].reshape(-1) for l in self.labels]).astype(np.float64)

    def unpack(self, x: np.ndarray) -> dict[int, np.ndarray]:
        """Unknown vector  ->  dict label -> (n, 2) loop."""
        return {l: x[self.offset[l]: self.offset[l] + 2 * self.n_samples].reshape(-1, 2) for l in self.labels}


def _nearest(loop: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cKDTree(loop).query(pts)[1].astype(np.int32) if len(pts) else np.zeros(0, np.int32)


def problem(la: np.ndarray, lb: np.ndarray, isos: dict[int, LatticeIsometry], labels: list[int],
            samples: int = 96, max_pair_samples: int = 220, init_weights: tuple[float, float] = (0.65, 0.35),
            seed: int = 2012) -> Problem:
    """Build a BACO problem from matched lattice labels.

    Parameters
    ----------
    la, lb : (H, W) int arrays
        Piece labels of A and of B, with B relabelled to A's labels
        (:func:`wcl.dissection.lattice.match_labels`).
    isos : dict label -> LatticeIsometry
    labels : list of int
        Pieces to optimise.
    samples : int
        Samples per loop.
    max_pair_samples : int
        Maximum boundary samples per (piece, neighbour) pair.
    init_weights : (float, float)
        Blend of the A contour and the pulled-back B contour for the initial loop.
    seed : int
        Seed of the boundary subsampling.

    Returns
    -------
    Problem
    """
    rng = np.random.default_rng(seed)
    wa, wb = init_weights
    loops_a, loops_b, init = {}, {}, {}
    for l in labels:
        a = resample_closed(contour_loop(la == l), samples)
        b = align_loop(a, isos[l].inverse_points(resample_closed(contour_loop(lb == l), samples)))
        loops_a[l], loops_b[l], init[l] = a, b, (wa * a + wb * b) / max(wa + wb, EPS)
    offset = {l: 2 * samples * i for i, l in enumerate(labels)}
    base = {l: samples * i for i, l in enumerate(labels)}      # sample index of a loop's first point
    S = set(labels)

    a_pairs, a_ext_idx, a_ext_tgt = [], [], []
    for (o, nb), pts in boundary_samples(la, max_pair_samples, rng).items():
        if o not in S:
            continue
        oi = _nearest(init[o], pts)
        if nb == 0:
            a_ext_idx += (base[o] + oi).tolist(); a_ext_tgt += pts.tolist()
        elif nb in S and o < nb:
            ni = _nearest(init[nb], pts)
            a_pairs += [(base[o] + i, base[nb] + j) for i, j in zip(oi.tolist(), ni.tolist())]

    b_pairs, b_ext_lab, b_ext_idx, b_ext_tgt = [], [], [], []
    for (o, nb), pts in boundary_samples(lb, max_pair_samples, rng).items():
        if o not in S:
            continue
        oi = _nearest(init[o], isos[o].inverse_points(pts))
        if nb == 0:
            b_ext_lab += [o] * len(oi); b_ext_idx += (base[o] + oi).tolist(); b_ext_tgt += pts.tolist()
        elif nb in S and o < nb:
            ni = _nearest(init[nb], isos[nb].inverse_points(pts))
            b_pairs += [(o, base[o] + i, nb, base[nb] + j) for i, j in zip(oi.tolist(), ni.tolist())]

    arr = lambda v, k: np.asarray(v, dtype=np.int64).reshape(-1, k)  # noqa: E731
    return Problem(labels, isos, init, loops_a, loops_b, arr(a_pairs, 2), arr(b_pairs, 4),
                   (np.asarray(a_ext_idx, np.int64), np.asarray(a_ext_tgt, np.float64).reshape(-1, 2)),
                   (np.asarray(b_ext_lab, np.int64), np.asarray(b_ext_idx, np.int64), np.asarray(b_ext_tgt, np.float64).reshape(-1, 2)),
                   offset)


# ----------------------------------------------------------------------------- residuals and solve

def _forward(prob: Problem, labels: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Move each point by the isometry of its label (vectorised per label)."""
    out = np.empty_like(pts)
    for l in np.unique(labels):
        m = labels == l
        out[m] = prob.isos[int(l)].forward_points(pts[m])
    return out


def residuals(x: np.ndarray, prob: Problem, w: Weights = Weights()) -> np.ndarray:
    """Stacked weighted residuals (contact, exterior, shape, fairness, edge length)."""
    P = x.reshape(-1, 2)
    res = []
    if len(prob.a_contacts):
        res.append((w.a_contact * (P[prob.a_contacts[:, 0]] - P[prob.a_contacts[:, 1]])).reshape(-1))
    if len(prob.b_contacts):
        bc = prob.b_contacts
        p = _forward(prob, bc[:, 0], P[bc[:, 1]])
        q = _forward(prob, bc[:, 2], P[bc[:, 3]])
        res.append((w.b_contact * (p - q)).reshape(-1))
    idx, tgt = prob.a_exterior
    if len(idx):
        res.append((w.a_exterior * (P[idx] - tgt)).reshape(-1))
    lab, idx, tgt = prob.b_exterior
    if len(idx):
        res.append((w.b_exterior * (_forward(prob, lab, P[idx]) - tgt)).reshape(-1))
    for l in prob.labels:
        pts = x[prob.offset[l]: prob.offset[l] + 2 * prob.n_samples].reshape(-1, 2)
        init = prob.init[l]
        res.append(w.shape * (pts - init).reshape(-1))
        res.append(w.a_shape * (pts - prob.loops_a[l]).reshape(-1))
        res.append(w.b_shape * (pts - prob.loops_b[l]).reshape(-1))
        res.append(w.fair * (np.roll(pts, 1, axis=0) - 2.0 * pts + np.roll(pts, -1, axis=0)).reshape(-1))
        edge = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        res.append(w.edge * (edge - np.linalg.norm(np.roll(init, -1, axis=0) - init, axis=1)))
    return np.concatenate(res).astype(np.float64)


def solve(prob: Problem, w: Weights = Weights(), max_nfev: int = 120, loss: str = "soft_l1",
          f_scale: float = 2.0) -> np.ndarray:
    """Robust least-squares solve from the initial loops; returns the optimal unknown vector."""
    x0 = prob.pack(prob.init)
    return least_squares(lambda z: residuals(z, prob, w), x0, max_nfev=max_nfev, loss=loss,
                         f_scale=f_scale, x_scale="jac").x


def contact_rms(x: np.ndarray, prob: Problem) -> tuple[float, float]:
    """RMS distance (pixels) between paired contact samples in A and in B."""
    P = x.reshape(-1, 2)
    rms = lambda d: float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0  # noqa: E731
    a = np.linalg.norm(P[prob.a_contacts[:, 0]] - P[prob.a_contacts[:, 1]], axis=1) if len(prob.a_contacts) else np.zeros(0)
    bc = prob.b_contacts
    b = (np.linalg.norm(_forward(prob, bc[:, 0], P[bc[:, 1]]) - _forward(prob, bc[:, 2], P[bc[:, 3]]), axis=1)
         if len(bc) else np.zeros(0))
    return rms(a), rms(b)
