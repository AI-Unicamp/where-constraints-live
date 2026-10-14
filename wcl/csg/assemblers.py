"""Boolean assemblers: combine N slot occupancies into one shape occupancy.

Four assemblers are compared in the dissertation (Chapter 3):

* :func:`nary_dnf` - a union of intersections of literals (disjunctive normal
  form), with per-clause positive and negative literal selections;
* :func:`csg_stump` - CSG-Stump normal form with product t-norm/t-conorm
  (complement layer, intersection layer, union layer);
* :func:`csg_stump_minmax` - the same normal form with the relaxed min/max of
  Ren et al. (2021) during training and exact min/max for hard evaluation
  ("paper stump");
* :func:`binary_tree` - a balanced binary tree whose internal nodes select a
  Boolean operation (union, intersection, difference).

Each takes slot occupancies ``q`` of shape (N, B) and returns (B,). With
``hard=True`` every selection is thresholded (or arg-maxed) to give the
extracted CSG program.
"""
from __future__ import annotations

import torch


def _select(logits: torch.Tensor, hard: bool, threshold: float) -> torch.Tensor:
    s = torch.sigmoid(logits)
    return (s > threshold).float() if hard else s


def _soft_or(a: torch.Tensor, dim: int) -> torch.Tensor:
    """Probabilistic union ``1 - prod(1 - a)`` along ``dim``."""
    return 1.0 - torch.prod((1.0 - a).clamp_min(1e-6), dim=dim)


def nary_dnf(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, active: torch.Tensor,
             hard: bool = False, threshold: float = 0.5) -> torch.Tensor:
    """N-ary disjunctive normal form ``OR_c AND_{i in c} (q_i or not q_i)``.

    Parameters
    ----------
    q : (N, B) tensor
        Slot occupancies.
    pos, neg : (N, C) tensors
        Logits selecting slot ``i`` as a positive / negated literal of clause ``c``.
    active : (C,) tensor
        Logits switching clause ``c`` on.
    hard : bool
        Threshold all selections; active clauses with no literal are dropped
        (an empty intersection is the whole space).
    threshold : float

    Returns
    -------
    o : (B,) tensor
    """
    mp, mn, a = _select(pos, hard, threshold), _select(neg, hard, threshold), _select(active, hard, threshold)
    if hard:
        a = a * ((mp + mn).sum(dim=0) > 0).float()
    o = q.unsqueeze(-1)                                    # (N, B, 1)
    lit_pos = 1.0 - mp.unsqueeze(1) * (1.0 - o)            # unselected literals are neutral (1)
    lit_neg = 1.0 - mn.unsqueeze(1) * o
    clause = torch.exp(torch.log(lit_pos.clamp_min(1e-8)).sum(0) + torch.log(lit_neg.clamp_min(1e-8)).sum(0))
    return _soft_or((a.unsqueeze(0) * clause).clamp(0, 1), dim=-1).clamp(0, 1)


def csg_stump(q: torch.Tensor, complement: torch.Tensor, intersect: torch.Tensor, union: torch.Tensor,
              hard: bool = False, threshold: float = 0.5) -> torch.Tensor:
    """CSG-Stump normal form with product t-norm and t-conorm.

    ``F_i = c_i (1 - q_i) + (1 - c_i) q_i``,
    ``S_k = prod_i (1 - w^I_ik (1 - F_i))``,
    ``T = 1 - prod_k (1 - w^U_k S_k)``.

    Parameters
    ----------
    q : (N, B) tensor
    complement : (N,) tensor
        Logits ``c`` choosing a slot or its complement (one choice per slot).
    intersect : (N, K) tensor
        Logits ``w^I`` choosing the literals of each intersection node.
    union : (K,) tensor
        Logits ``w^U`` choosing the intersection nodes of the final union.
    hard : bool
        Threshold the selections; empty intersection nodes are dropped.
    threshold : float

    Returns
    -------
    o : (B,) tensor
    """
    c, wi, wu = _select(complement, hard, threshold), _select(intersect, hard, threshold), _select(union, hard, threshold)
    if hard:
        wu = wu * (wi.sum(dim=0) > 0).float()
    F = (1.0 - c).reshape(-1, 1) * q + c.reshape(-1, 1) * (1.0 - q)
    lit = 1.0 - wi.unsqueeze(1) * (1.0 - F.unsqueeze(-1))
    S = torch.exp(torch.log(lit.clamp_min(1e-8)).sum(0))
    return _soft_or((wu.unsqueeze(0) * S).clamp(0, 1), dim=-1).clamp(0, 1)


def _softmin(x: torch.Tensor, dim: int, psi: float) -> torch.Tensor:
    return (torch.softmax(-psi * x, dim=dim) * x).sum(dim=dim)


def _softmax(x: torch.Tensor, dim: int, psi: float) -> torch.Tensor:
    return (torch.softmax(psi * x, dim=dim) * x).sum(dim=dim)


def csg_stump_minmax(q: torch.Tensor, complement: torch.Tensor, intersect: torch.Tensor, union: torch.Tensor,
                     hard: bool = False, psi: float = 20.0, threshold: float = 0.5) -> torch.Tensor:
    """CSG-Stump with min/max operators (Ren et al. 2021), "paper stump".

    Training uses the relaxed ``min*``/``max*`` (softmax-weighted averages with
    sharpness ``psi``); ``hard=True`` thresholds the connection matrices and
    uses exact min/max. As in the evaluation of the dissertation's budget sweep,
    empty intersection nodes are not removed in the hard program.

    Parameters
    ----------
    q : (N, B) tensor
    complement, intersect, union
        Logits of the WC, WI and WU connection matrices (shapes (N,), (N, K), (K,)).
    hard : bool
    psi : float
        Sharpness of the relaxed min/max.
    threshold : float

    Returns
    -------
    o : (B,) tensor
    """
    wc, wi, wu = _select(complement, hard, threshold), _select(intersect, hard, threshold), _select(union, hard, threshold)
    F = wc.reshape(-1, 1) * (1.0 - q) + (1.0 - wc.reshape(-1, 1)) * q
    X = wi.unsqueeze(1) * F.unsqueeze(-1) + (1.0 - wi.unsqueeze(1))
    if hard:
        Y = wu.reshape(1, -1) * torch.min(X, dim=0).values
        return torch.max(Y, dim=-1).values.clamp(0, 1)
    Y = wu.reshape(1, -1) * _softmin(X, 0, psi)
    return _softmax(Y, -1, psi).clamp(0, 1)


def binary_tree(q: torch.Tensor, gates: torch.Tensor, ops: torch.Tensor, hard: bool = False,
                threshold: float = 0.5) -> torch.Tensor:
    """Balanced binary CSG tree with learned operations.

    Leaves are gated slots ``g_i q_i``. Pairs are merged left to right, level by
    level (an odd leftover is carried up unchanged); internal node ``j`` mixes
    union, intersection and difference with ``softmax(ops[j])`` (or takes the
    arg-max when ``hard``). With 5 operation logits the extra operations are the
    reversed difference and "pass left".

    Parameters
    ----------
    q : (N, B) tensor
    gates : (N,) tensor
        Leaf gate logits.
    ops : (N - 1, 3) or (N - 1, 5) tensor
        Operation logits of the internal nodes, in merge order.
    hard : bool
    threshold : float
        Gate threshold when ``hard``.

    Returns
    -------
    o : (B,) tensor
    """
    g = _select(gates, hard, threshold)
    vals = [(g[i] * q[i]).clamp(0, 1) for i in range(q.shape[0])]
    node = 0
    while len(vals) > 1:
        merged = []
        for i in range(0, len(vals), 2):
            if i + 1 == len(vals):
                merged.append(vals[i])
                continue
            logits = ops[node]
            if hard:
                w = torch.zeros_like(logits); w[torch.argmax(logits)] = 1.0
            else:
                w = torch.softmax(logits, dim=-1)
            a, b = vals[i], vals[i + 1]
            out = w[0] * (1.0 - (1.0 - a) * (1.0 - b)) + w[1] * (a * b) + w[2] * (a * (1.0 - b))
            if logits.shape[0] == 5:
                out = out + w[3] * (b * (1.0 - a)) + w[4] * a
            merged.append(out.clamp(0, 1))
            node += 1
        vals = merged
    return vals[0]
