"""Signed distance functions of the five primitive types.

All functions take query points already expressed in the primitive's local frame
(see :func:`wcl.csg.rotation.to_local`) and positive size parameters. They are
differentiable with respect to both points and sizes.

The square roots carry a small ``eps`` so that the gradient stays finite at the
centre (sphere) and on the box/cylinder surfaces; with the default ``eps=1e-8``
the values match the implementation used for the dissertation exactly.
"""
from __future__ import annotations

import torch

TYPES = ("sphere", "box", "cylinder", "wedge", "corner_wedge")
"""Primitive types, in the index order used everywhere in :mod:`wcl.csg`."""


def sphere_sdf(p: torch.Tensor, r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signed distance to a sphere centred at the origin.

    Parameters
    ----------
    p : (..., 3) tensor
        Query points in the local frame.
    r : (...) tensor
        Radius, broadcastable to ``p.shape[:-1]``.
    eps : float
        Smoothing added under the square root.

    Returns
    -------
    d : (...) tensor
        Signed distance, negative inside.

    Examples
    --------
    >>> p = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> sphere_sdf(p, torch.tensor(1.0))
    tensor([-0.9999,  1.0000])
    """
    return torch.sqrt((p * p).sum(-1) + eps) - r


def box_sdf(p: torch.Tensor, h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signed distance to an axis-aligned box ``[-h, h]`` (exact outside and inside).

    Parameters
    ----------
    p : (..., 3) tensor
        Query points in the local frame.
    h : (..., 3) tensor
        Half extents along x, y, z.
    eps : float
        Smoothing added under the square root.

    Returns
    -------
    d : (...) tensor
        Signed distance, negative inside.
    """
    d = torch.abs(p) - h
    outside = torch.sqrt(torch.clamp(d, min=0).pow(2).sum(-1) + eps)
    inside = torch.clamp(d.max(-1).values, max=0)
    return outside + inside


def cylinder_sdf(p: torch.Tensor, r: torch.Tensor, hh: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signed distance to a capped cylinder with axis y.

    Parameters
    ----------
    p : (..., 3) tensor
        Query points in the local frame.
    r : (...) tensor
        Radius.
    hh : (...) tensor
        Half height along y.
    eps : float
        Smoothing added under the square roots.

    Returns
    -------
    d : (...) tensor
        Signed distance, negative inside.
    """
    dr = torch.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2 + eps) - r
    dy = torch.abs(p[..., 1]) - hh
    outside = torch.sqrt(torch.clamp(dr, min=0) ** 2 + torch.clamp(dy, min=0) ** 2 + eps)
    inside = torch.clamp(torch.maximum(dr, dy), max=0)
    return outside + inside


def _plane(p: torch.Tensor, h: torch.Tensor, i: int, j: int, eps: float) -> torch.Tensor:
    """Signed distance to the plane x_i/h_i + x_j/h_j = 1 (positive on the far side)."""
    inv = 1.0 / torch.sqrt(1.0 / (h[..., i] ** 2) + 1.0 / (h[..., j] ** 2) + eps)
    return (p[..., i] / h[..., i] + p[..., j] / h[..., j] - 1.0) * inv


def wedge_sdf(p: torch.Tensor, h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signed distance bound for a box with one chamfered edge.

    The solid is the box ``[-h, h]`` cut by the half-space
    ``y/h_y + z/h_z <= 1``. The value is ``max(box, plane)``, which is exact
    inside and a lower bound outside near the chamfer (standard for
    intersections of SDFs).

    Notes
    -----
    The exact volume of this solid is ``7 h_x h_y h_z``
    (see :func:`wcl.csg.volume.primitive_volume`).

    Parameters
    ----------
    p : (..., 3) tensor
        Query points in the local frame.
    h : (..., 3) tensor
        Half extents of the bounding box.
    eps : float
        Smoothing used by the box term and the plane normalisation.

    Returns
    -------
    d : (...) tensor
        Signed distance (bound), negative inside.
    """
    return torch.maximum(box_sdf(p, h, eps), _plane(p, h, 1, 2, eps))


def corner_wedge_sdf(p: torch.Tensor, h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signed distance bound for a box with two chamfered edges meeting at a corner.

    The solid is the box ``[-h, h]`` cut by ``x/h_x + y/h_y <= 1`` and
    ``z/h_z + y/h_y <= 1``; its exact volume is ``19/3 h_x h_y h_z``.

    Parameters
    ----------
    p : (..., 3) tensor
        Query points in the local frame.
    h : (..., 3) tensor
        Half extents of the bounding box.
    eps : float
        Smoothing used by the box term and the plane normalisations.

    Returns
    -------
    d : (...) tensor
        Signed distance (bound), negative inside.
    """
    d = torch.maximum(box_sdf(p, h, eps), _plane(p, h, 0, 1, eps))
    return torch.maximum(d, _plane(p, h, 2, 1, eps))


def primitive_sdf(kind: int, p: torch.Tensor, size: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Signed distance of primitive type ``kind`` with packed size parameters.

    Parameters
    ----------
    kind : int
        Index into :data:`TYPES`.
    p : (..., 3) tensor
        Query points in the local frame.
    size : (..., 3) tensor
        Positive size parameters. Sphere: ``size[..., 0]`` is the radius;
        box, wedge, corner wedge: half extents; cylinder: ``(radius, half height, -)``.
    eps : float
        Square-root smoothing.

    Returns
    -------
    d : (...) tensor
        Signed distance, negative inside.
    """
    if kind == 0:
        return sphere_sdf(p, size[..., 0], eps)
    if kind == 1:
        return box_sdf(p, size, eps)
    if kind == 2:
        return cylinder_sdf(p, size[..., 0], size[..., 1], eps)
    if kind == 3:
        return wedge_sdf(p, size, eps)
    if kind == 4:
        return corner_wedge_sdf(p, size, eps)
    raise ValueError(f"unknown primitive type {kind}")
