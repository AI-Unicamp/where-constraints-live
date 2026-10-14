"""Primitive volumes and the volume-equivalent radius used to calibrate the fields.

The dissertation normalises each primitive's signed distance by its
volume-equivalent radius ``r = (3V / 4pi)^(1/3)`` before turning it into an
occupancy (Section 3, Eq. "volume radius"), so that primitives of different
types compete on comparable scales.

For the wedge and corner wedge the dissertation's runs used the volume proxies
``4abc`` and ``2abc`` instead of the exact volumes ``7abc`` and ``19/3 abc`` of
the solids defined in :mod:`wcl.csg.primitives`. :func:`volume_radius` keeps the
proxies by default (``exact=False``) so that results reproduce; pass
``exact=True`` for the true volumes.
"""
from __future__ import annotations

import math

import torch

_PROXY_FACTOR = {1: 8.0, 3: 4.0, 4: 2.0}            # V / (a b c) used in the dissertation runs
_EXACT_FACTOR = {1: 8.0, 3: 7.0, 4: 19.0 / 3.0}     # exact V / (a b c) of the box / wedge / corner wedge


def primitive_volume(kind: int, size: torch.Tensor, exact: bool = True) -> torch.Tensor:
    """Volume of a primitive.

    Parameters
    ----------
    kind : int
        Index into :data:`wcl.csg.primitives.TYPES`.
    size : (..., 3) tensor
        Positive size parameters (see :func:`wcl.csg.primitives.primitive_sdf`).
    exact : bool
        If False, use the proxies of the dissertation runs for the wedge types.

    Returns
    -------
    V : (...) tensor

    Examples
    --------
    >>> primitive_volume(3, torch.tensor([1.0, 1.0, 1.0]))
    tensor(7.)
    """
    if kind == 0:
        return 4.0 / 3.0 * math.pi * size[..., 0] ** 3
    if kind == 2:
        return 2.0 * math.pi * size[..., 0] ** 2 * size[..., 1]
    factor = (_EXACT_FACTOR if exact else _PROXY_FACTOR)[kind]
    return factor * size[..., 0] * size[..., 1] * size[..., 2]


def volume_radius(kind: int, size: torch.Tensor, exact: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """Volume-equivalent radius ``(3V / 4pi)^(1/3)`` of a primitive.

    Parameters
    ----------
    kind : int
        Index into :data:`wcl.csg.primitives.TYPES`.
    size : (..., 3) tensor
        Positive size parameters.
    exact : bool
        Default False reproduces the dissertation (wedge proxies, see module notes).
    eps : float
        Added inside the cube root, as in the dissertation code.

    Returns
    -------
    r : (...) tensor
    """
    if kind == 0:
        return size[..., 0]
    # (3V/4pi) written per type exactly as in the reference implementation
    if kind == 2:
        r, h = size[..., 0], size[..., 1]
        return torch.pow(1.5 * r * r * h + eps, 1.0 / 3.0)
    c = 0.75 * (_EXACT_FACTOR if exact else _PROXY_FACTOR)[kind]   # 3/(4pi) * V = c * abc / pi
    a, b, cc = size[..., 0], size[..., 1], size[..., 2]
    return torch.pow(c * a * b * cc / math.pi + eps, 1.0 / 3.0)
