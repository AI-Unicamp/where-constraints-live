"""Typed slots: every slot carries all primitive types, which compete for it.

A slot ``i`` holds one candidate occupancy ``o_it`` per type ``t``. Training mixes
them with weights ``w_it`` (:func:`type_weights`) through a product union
(:func:`combine_types`); extraction keeps either the single best type
(one-hot) or every type above a threshold (multi-hot, :func:`multihot_mask`).
"""
from __future__ import annotations

import torch

SELECTORS = ("fixed_cycle", "sine_softmax", "plain_softmax", "sine_sigmoid_multi")


def type_weights(logits: torch.Tensor, beta: float, selector: str, omega: float = 10.0,
                 hard: bool = False, fixed_types: torch.Tensor | None = None) -> torch.Tensor:
    """Per-slot type weights.

    Parameters
    ----------
    logits : (N, T) tensor
        Learned type logits ``alpha``.
    beta : float
        Inverse temperature (annealed during training).
    selector : str
        ``"sine_softmax"``: ``softmax_t(beta sin(omega alpha))`` (one-hot training);
        ``"plain_softmax"``: ``softmax_t(beta alpha)``;
        ``"sine_sigmoid_multi"``: ``sigmoid(beta sin(omega alpha))`` (multi-hot training);
        ``"fixed_cycle"``: one fixed type per slot, ``fixed_types``.
    omega : float
        Frequency of the sine re-parameterisation.
    hard : bool
        Return the one-hot arg-max instead of the soft weights.
    fixed_types : (N,) long tensor
        Required for ``"fixed_cycle"``.

    Returns
    -------
    w : (N, T) tensor
    """
    if selector == "fixed_cycle":
        w = torch.zeros_like(logits)
        return w.scatter_(1, fixed_types.reshape(-1, 1), 1.0)
    z = beta * (logits if selector == "plain_softmax" else torch.sin(omega * logits))
    if hard:
        w = torch.zeros_like(z)
        return w.scatter_(1, torch.argmax(z, dim=-1, keepdim=True), 1.0)
    if selector == "sine_sigmoid_multi":
        return torch.sigmoid(z)
    return torch.softmax(z, dim=-1)


def combine_types(occ: torch.Tensor, w: torch.Tensor, aggregate: str = "product") -> torch.Tensor:
    """Slot occupancy from its per-type candidates.

    Parameters
    ----------
    occ : (N, B, T) tensor
        Candidate occupancies ``o_it(x_b)``.
    w : (N, T) tensor
        Type weights or a 0/1 mask.
    aggregate : str
        ``"product"``: ``1 - prod_t (1 - w_it o_it)`` (product union, used in the
        dissertation); ``"sum"``: ``clamp(sum_t w_it o_it, 0, 1)``.

    Returns
    -------
    q : (N, B) tensor
    """
    wo = occ * w.unsqueeze(1)
    if aggregate == "sum":
        return wo.sum(-1).clamp(0, 1)
    return 1.0 - torch.prod((1.0 - wo).clamp_min(1e-6), dim=-1)


def onehot_occupancy(occ: torch.Tensor, w_hard: torch.Tensor) -> torch.Tensor:
    """Slot occupancy of the single selected type (one-hot extraction).

    Parameters
    ----------
    occ : (N, B, T) tensor
    w_hard : (N, T) one-hot tensor, e.g. ``type_weights(..., hard=True)``

    Returns
    -------
    q : (N, B) tensor
    """
    return (occ * w_hard.unsqueeze(1)).sum(-1)


def multihot_mask(logits: torch.Tensor, beta: float, selector: str, threshold: float, omega: float = 10.0,
                  fixed_types: torch.Tensor | None = None) -> torch.Tensor:
    """Types kept by multi-hot extraction: every type whose weight exceeds ``threshold``.

    Slots with no type above the threshold keep their arg-max type, so every
    slot keeps at least one type.

    Parameters
    ----------
    logits : (N, T) tensor
    beta, selector, omega, fixed_types
        As in :func:`type_weights`.
    threshold : float

    Returns
    -------
    mask : (N, T) 0/1 tensor
    """
    if selector == "fixed_cycle":
        return type_weights(logits, beta, selector, omega, hard=True, fixed_types=fixed_types)
    if selector == "sine_sigmoid_multi":
        mask = (torch.sigmoid(beta * torch.sin(omega * logits)) > threshold).float()
    else:
        mask = (type_weights(logits, beta, selector, omega) > threshold).float()
    none = mask.sum(-1, keepdim=True) <= 0
    if none.any():
        arg = torch.argmax(type_weights(logits, beta, selector, omega), dim=-1, keepdim=True)
        one = torch.zeros_like(mask).scatter_(1, arg, 1.0)
        mask = torch.where(none, one, mask)
    return mask
