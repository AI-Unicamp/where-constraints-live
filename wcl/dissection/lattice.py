"""Lattice isometries between pieces of two pixel label maps (Zhou and Wang 2012).

A lattice dissection assigns every foreground pixel of figure A and of figure B
to a piece label; each piece moves from A to B by one of the 8 lattice
isometries (4 rotations, optional mirror) plus an integer translation.
:func:`best_isometry` finds that motion for one pair of pixel sets and
:func:`match_labels` pairs the pieces of A and B (Hungarian assignment).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class LatticeIsometry:
    """Rotation by ``rot`` degrees (multiple of 90), optional mirror, integer shift.

    The rotation acts on the piece's bounding box in A (origin ``(y0, x0)``,
    size ``h x w``); ``(dy, dx)`` is the shift in pixels. ``mismatch`` counts the
    pixels of A and B not covered by the moved piece (the Zhou-Wang distance).
    """
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

    def forward(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map continuous pixel coordinates from figure A to figure B.

        Parameters
        ----------
        x, y : arrays of the same shape
            Pixel coordinates (x right, y down) in A.

        Returns
        -------
        x', y' : arrays
            Coordinates in B.
        """
        x = x - self.x0
        y = y - self.y0
        k = (self.rot // 90) % 4
        if k == 0:
            xr, yr, ww = x, y, self.w
        elif k == 1:
            xr, yr, ww = self.h - y, x, self.h
        elif k == 2:
            xr, yr, ww = self.w - x, self.h - y, self.w
        else:
            xr, yr, ww = y, self.w - x, self.h
        if self.flip:
            xr = ww - xr
        return xr + self.x0 + self.dx, yr + self.y0 + self.dy

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map continuous pixel coordinates from figure B back to figure A (inverse of :meth:`forward`)."""
        xr = x - self.x0 - self.dx
        yr = y - self.y0 - self.dy
        k = (self.rot // 90) % 4
        ww = self.w if k in (0, 2) else self.h
        if self.flip:
            xr = ww - xr
        if k == 0:
            x, y = xr, yr
        elif k == 1:
            x, y = yr, self.h - xr
        elif k == 2:
            x, y = self.w - xr, self.h - yr
        else:
            x, y = self.w - yr, xr
        return x + self.x0, y + self.y0

    def forward_points(self, p: np.ndarray) -> np.ndarray:
        """(n, 2) points (x, y) in A -> (n, 2) points in B."""
        x, y = self.forward(p[:, 0], p[:, 1])
        return np.column_stack([x, y]).astype(np.float64)

    def inverse_points(self, p: np.ndarray) -> np.ndarray:
        """(n, 2) points (x, y) in B -> (n, 2) points in A."""
        x, y = self.inverse(p[:, 0], p[:, 1])
        return np.column_stack([x, y]).astype(np.float64)


def label_color(label: int) -> tuple[int, int, int]:
    """RGB colour of a piece label in the PNGs written by the lattice search (an integer hash)."""
    v = (label * 0x9E3779B1) & 0xFFFFFFFF
    v ^= v >> 16
    v = (v * 0x85EBCA6B) & 0xFFFFFFFF
    v ^= v >> 13
    v = (v * 0xC2B2AE35) & 0xFFFFFFFF
    v ^= v >> 16
    r, g, b = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
    if r < 24 and g < 24 and b < 24:
        r, g, b = r ^ 0x55, g ^ 0xAA, b ^ 0xCC
    return int(r), int(g), int(b)


def read_labels(path_a, path_b, max_label: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Decode the piece-label PNGs of figures A and B written by the lattice search.

    Colours are mapped back to labels through :func:`label_color`; colours that are
    not in the palette get fresh labels above ``max_label``. Black is background (0).

    Parameters
    ----------
    path_a, path_b : str or Path
        PNG files of the same size.

    Returns
    -------
    la, lb : (H, W) int32 arrays
    """
    from PIL import Image
    a = np.asarray(Image.open(path_a).convert("RGB"))
    b = np.asarray(Image.open(path_b).convert("RGB"))
    if a.shape != b.shape:
        raise ValueError(f"label images differ in size: {a.shape} vs {b.shape}")
    colors = sorted((set(map(tuple, a.reshape(-1, 3).tolist())) | set(map(tuple, b.reshape(-1, 3).tolist()))) - {(0, 0, 0)})
    palette = {label_color(i): i for i in range(1, max_label + 1)}
    la, lb = np.zeros(a.shape[:2], np.int32), np.zeros(b.shape[:2], np.int32)
    fresh = max_label + 1
    for c in colors:
        if c in palette:
            lab = palette[c]
        else:
            lab, fresh = fresh, fresh + 1
        col = np.asarray(c, dtype=np.uint8)
        la[np.all(a == col, axis=2)] = lab
        lb[np.all(b == col, axis=2)] = lab
    return la, lb


def _rotflip(local: np.ndarray, h: int, w: int, rot: int, flip: bool) -> np.ndarray:
    """Rotate/mirror integer (row, col) coordinates inside an h x w box."""
    y, x = local[:, 0], local[:, 1]
    k = (rot // 90) % 4
    if k == 0:
        yy, xx, ww = y.copy(), x.copy(), w
    elif k == 1:
        yy, xx, ww = x.copy(), (h - 1) - y, h
    elif k == 2:
        yy, xx, ww = (h - 1) - y, (w - 1) - x, w
    else:
        yy, xx, ww = (w - 1) - x, y.copy(), h
    if flip:
        xx = (ww - 1) - xx
    return np.column_stack([yy, xx]).astype(np.int32)


def best_isometry(label: int, a_mask: np.ndarray, b_mask: np.ndarray) -> LatticeIsometry | None:
    """Lattice isometry moving the pixels ``a_mask`` onto ``b_mask`` with the fewest mismatched pixels.

    For each of the 8 isometries the piece's centroid is aligned with the target
    centroid and the 4 integer roundings of the shift are tried (Zhou and Wang's
    "4-snap"). The search stops early at the lower bound ``|#A - #B|``.

    Parameters
    ----------
    label : int
        Piece label (stored in the result).
    a_mask, b_mask : (H, W) bool arrays

    Returns
    -------
    iso : LatticeIsometry, or None if either mask is empty
    """
    ayx = np.column_stack(np.nonzero(a_mask)).astype(np.int32)
    byx = np.column_stack(np.nonzero(b_mask)).astype(np.int32)
    if len(ayx) == 0 or len(byx) == 0:
        return None
    y0, x0 = ayx.min(axis=0).tolist()
    y1, x1 = ayx.max(axis=0).tolist()
    h, w = int(y1 - y0 + 1), int(x1 - x0 + 1)
    local = ayx - np.asarray([y0, x0], dtype=np.int32)
    cy_b, cx_b = byx.mean(axis=0)
    target = b_mask.astype(bool)
    H, W = b_mask.shape
    lower = abs(len(ayx) - len(byx))
    best = None
    for flip in (False, True):
        for rot in (0, 90, 180, 270):
            r = _rotflip(local, h, w, rot, flip)
            cy_a, cx_a = (r + np.asarray([y0, x0])).mean(axis=0)
            dyf, dxf = cy_b - cy_a, cx_b - cx_a
            snaps = {(math.floor(dyf), math.floor(dxf)), (math.floor(dyf), math.ceil(dxf)),
                     (math.ceil(dyf), math.floor(dxf)), (math.ceil(dyf), math.ceil(dxf))}
            for dy, dx in sorted((int(a), int(b)) for a, b in snaps):
                gy, gx = r[:, 0] + y0 + dy, r[:, 1] + x0 + dx
                inside = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
                overlap = int(np.count_nonzero(target[gy[inside], gx[inside]]))
                mismatch = int(len(ayx) + len(byx) - 2 * overlap)
                if best is None or mismatch < best.mismatch:
                    best = LatticeIsometry(label, rot, flip, dy, dx, int(y0), int(x0), h, w, mismatch, len(ayx), len(byx))
                    if mismatch <= lower:
                        return best
    return best


def _perimeter(mask: np.ndarray) -> int:
    """Perimeter in pixel edges: foreground/background 4-neighbour pairs."""
    p = np.pad(mask.astype(bool), 1)
    c = p[1:-1, 1:-1]
    return int(sum(np.count_nonzero(c & ~n) for n in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:])))


def _components(mask: np.ndarray) -> int:
    """Number of 4-connected components."""
    from scipy import ndimage
    return int(ndimage.label(mask.astype(bool))[1])


def match_labels(la: np.ndarray, lb: np.ndarray) -> tuple[np.ndarray, dict[int, LatticeIsometry]]:
    """Pair the pieces of two label maps and find each piece's isometry.

    The cost of pairing source piece ``s`` with target piece ``t`` is
    ``mismatch/(|s|+|t|) + 0.30 area term + 0.08 perimeter term + 0.05 component term``;
    the pairing minimises the total cost (Hungarian algorithm). Saved label
    colours of the lattice search are therefore not trusted as a correspondence.

    Parameters
    ----------
    la, lb : (H, W) int arrays
        Piece labels of figures A and B (0 = background).

    Returns
    -------
    lb_matched : (H, W) int array
        ``lb`` relabelled with the matching source labels.
    isos : dict label -> LatticeIsometry
    """
    sids = sorted(int(v) for v in np.unique(la) if int(v) > 0)
    tids = sorted(int(v) for v in np.unique(lb) if int(v) > 0)
    if not sids or not tids:
        return lb.copy(), {}
    stats = lambda m: (int(np.count_nonzero(m)), _perimeter(m), _components(m))  # noqa: E731
    ss = {s: stats(la == s) for s in sids}
    ts = {t: stats(lb == t) for t in tids}
    table, costs = {}, np.full((len(sids), len(tids)), 1e6)
    for i, s in enumerate(sids):
        for j, t in enumerate(tids):
            iso = best_isometry(s, la == s, lb == t)
            if iso is None:
                continue
            table[(s, t)] = iso
            (sa, sp, sc), (ta, tp, tc) = ss[s], ts[t]
            costs[i, j] = (iso.mismatch / max(sa + ta, 1) + 0.30 * abs(sa - ta) / max(sa + ta, 1)
                           + 0.08 * abs(sp - tp) / max(sp + tp, 1) + 0.05 * abs(sc - tc))
    out, isos = np.zeros_like(lb, dtype=np.int32), {}
    for i, j in zip(*linear_sum_assignment(costs)):
        iso = table.get((sids[i], tids[j]))
        if iso is not None:
            out[lb == tids[j]] = sids[i]
            isos[sids[i]] = iso
    return out, isos
