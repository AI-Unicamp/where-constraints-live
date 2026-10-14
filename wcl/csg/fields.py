"""From signed distance to occupancy, and the local frame of a primitive."""
from __future__ import annotations

from typing import Optional

import torch


def euler_to_rotation(angles: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from Euler angles (x, y, z), row-vector convention.

    Parameters
    ----------
    angles : (..., 3) tensor
        Rotation angles in radians about x, y and z.

    Returns
    -------
    R : (..., 3, 3) tensor
        Matrices such that ``local = (x - c) @ R``.
    """
    cx, sx = torch.cos(angles[..., 0]), torch.sin(angles[..., 0])
    cy, sy = torch.cos(angles[..., 1]), torch.sin(angles[..., 1])
    cz, sz = torch.cos(angles[..., 2]), torch.sin(angles[..., 2])
    R = torch.zeros(*angles.shape[:-1], 3, 3, device=angles.device, dtype=angles.dtype)
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = cz * sx * sy - cx * sz
    R[..., 0, 2] = cx * cz * sy + sx * sz
    R[..., 1, 0] = cy * sz
    R[..., 1, 1] = cx * cz + sx * sy * sz
    R[..., 1, 2] = cx * sy * sz - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cx * cy
    return R


def to_local(points: torch.Tensor, centers: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Express points in the local frame of each of N primitives.

    Parameters
    ----------
    points : (B, 3) tensor
        Query points.
    centers : (N, 3) tensor
        Primitive centres.
    angles : (N, 3) tensor
        Primitive Euler angles (see :func:`euler_to_rotation`).

    Returns
    -------
    local : (N, B, 3) tensor
    """
    R = euler_to_rotation(angles)
    return torch.einsum("nbd,nde->nbe", points.unsqueeze(0) - centers.unsqueeze(1), R)


def sdf_to_occupancy(sdf: torch.Tensor, sharpness: torch.Tensor, scale: Optional[torch.Tensor] = None,
                     min_scale: float = 0.035, min_sharpness: float = 1.0) -> torch.Tensor:
    """Soft occupancy ``sigmoid(-s * d / r)`` of a signed distance field.

    Parameters
    ----------
    sdf : (...) tensor
        Signed distance, negative inside.
    sharpness : tensor broadcastable to ``sdf``
        Sigmoid slope ``s`` (clamped below by ``min_sharpness``).
    scale : tensor broadcastable to ``sdf``, optional
        Calibration radius ``r``, e.g. :func:`wcl.csg.volume.volume_radius`
        (clamped below by ``min_scale``). None leaves the distance unscaled.
    min_scale, min_sharpness : float
        Lower bounds that keep the occupancy well defined for tiny primitives.

    Returns
    -------
    o : (...) tensor in (0, 1)
    """
    if scale is not None:
        sdf = sdf / scale.clamp_min(min_scale)
    return torch.sigmoid(-sdf * sharpness.clamp_min(min_sharpness))
