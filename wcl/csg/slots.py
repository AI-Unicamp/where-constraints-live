"""Learnable typed slots: N slots, each with one candidate primitive per type."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fields import sdf_to_occupancy, to_local
from .primitives import TYPES, primitive_sdf
from .types import combine_types, multihot_mask, onehot_occupancy, type_weights
from .volume import volume_radius

MIN_SIZE = 0.02
"""Smallest primitive size; sizes are ``softplus(raw) + MIN_SIZE``."""


class TypedSlots(nn.Module):
    """N typed slots with shared pose and per-type sizes.

    Every slot has a centre, Euler angles and a sharpness shared by its types,
    plus three size parameters and one type logit per type. Initial values follow
    the dissertation code (random, seeded by the caller).

    Parameters
    ----------
    n : int
        Number of slots.
    selector : str
        Type selector, see :func:`wcl.csg.types.type_weights`.
    aggregate : str
        ``"product"`` (product union over types) or ``"sum"``.
    volume_norm : bool
        Calibrate each candidate by its volume-equivalent radius.
    omega : float
        Sine frequency of the type logits.
    init_scale : float
        Standard deviation of the initial centres.

    Examples
    --------
    >>> slots = TypedSlots(8, selector="sine_softmax")
    >>> x = torch.rand(100, 3) * 2 - 1
    >>> slots(x, beta=1.0).shape          # soft slot occupancies
    torch.Size([8, 100])
    """

    def __init__(self, n: int, selector: str = "sine_softmax", aggregate: str = "product",
                 volume_norm: bool = True, omega: float = 10.0, init_scale: float = 0.3):
        super().__init__()
        self.n, self.selector, self.aggregate = n, selector, aggregate
        self.volume_norm, self.omega = volume_norm, omega
        T = len(TYPES)
        self.register_buffer("fixed_types", torch.tensor([i % T for i in range(n)], dtype=torch.long))
        self.centers = nn.Parameter(torch.randn(n, 3) * init_scale)
        self.euler = nn.Parameter(torch.randn(n, 3) * 0.2)
        self.raw_params = nn.Parameter(torch.randn(n, T, 3) * 0.15 - 1.2)
        self.raw_sharp = nn.Parameter(torch.ones(n) * 2.5)
        self.type_logits = nn.Parameter(torch.randn(n, T) * 0.05)

    def sizes(self) -> torch.Tensor:
        """Effective primitive sizes, (N, T, 3).

        Equal to ``softplus(raw) + 2 MIN_SIZE``: the dissertation code applied the
        size floor twice (once in the parameterisation, once in the SDFs), and it
        is kept so that results reproduce exactly.
        """
        return (F.softplus(self.raw_params) + MIN_SIZE).abs() + MIN_SIZE

    def candidate_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        """Occupancy of every slot/type candidate at points ``x`` (B, 3) -> (N, B, T)."""
        local = to_local(x, self.centers, self.euler)
        sharp = (F.softplus(self.raw_sharp) + 2.0).reshape(-1, 1)
        size = self.sizes()
        out = []
        for t in range(len(TYPES)):
            s = size[:, t, :].reshape(self.n, 1, 3)
            sdf = primitive_sdf(t, local, s)
            scale = volume_radius(t, s).reshape(self.n, 1) if self.volume_norm else None
            out.append(sdf_to_occupancy(sdf, sharp, scale))
        return torch.stack(out, dim=-1)

    def weights(self, beta: float, hard: bool = False) -> torch.Tensor:
        """Type weights (N, T), see :func:`wcl.csg.types.type_weights`."""
        return type_weights(self.type_logits, beta, self.selector, self.omega, hard, self.fixed_types)

    def forward(self, x: torch.Tensor, beta: float, extraction: str = "soft", threshold: float = 0.5) -> torch.Tensor:
        """Slot occupancies (N, B).

        Parameters
        ----------
        x : (B, 3) tensor
        beta : float
            Inverse temperature of the type selector.
        extraction : str
            ``"soft"`` (training), ``"onehot"`` (best type per slot) or
            ``"multihot"`` (all types above ``threshold``).
        threshold : float
        """
        occ = self.candidate_occupancy(x)
        if extraction == "soft":
            return combine_types(occ, self.weights(beta), self.aggregate)
        if extraction == "onehot":
            return onehot_occupancy(occ, self.weights(beta, hard=True))
        if extraction == "multihot":
            mask = multihot_mask(self.type_logits, beta, self.selector, threshold, self.omega, self.fixed_types)
            return combine_types(occ, mask, self.aggregate)
        raise ValueError(extraction)
