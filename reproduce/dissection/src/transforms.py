from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable

from .lattice import Cell, CellSet, add


@dataclass(frozen=True)
class Transform:
    id: int
    name: str
    dimension: int
    orientation_preserving: bool
    fn: Callable[[Cell], Cell]


def square_transforms(allow_reflection: bool = True) -> list[Transform]:
    raw: list[tuple[str, bool, Callable[[Cell], Cell]]] = [
        ("r0", True, lambda p: (p[0], p[1])),
        ("r90", True, lambda p: (-p[1], p[0])),
        ("r180", True, lambda p: (-p[0], -p[1])),
        ("r270", True, lambda p: (p[1], -p[0])),
        ("ref_x", False, lambda p: (p[0], -p[1])),
        ("ref_y", False, lambda p: (-p[0], p[1])),
        ("ref_diag", False, lambda p: (p[1], p[0])),
        ("ref_antidiag", False, lambda p: (-p[1], -p[0])),
    ]
    if not allow_reflection:
        raw = [r for r in raw if r[1]]
    return [Transform(i, name, 2, orient, fn) for i, (name, orient, fn) in enumerate(raw)]


def _perm_sign_det(perm: tuple[int, int, int], signs: tuple[int, int, int]) -> int:
    inv = 0
    for i in range(3):
        for j in range(i + 1, 3):
            inv += int(perm[i] > perm[j])
    perm_det = -1 if inv % 2 else 1
    return perm_det * signs[0] * signs[1] * signs[2]


def cube_rotations_3d() -> list[Transform]:
    transforms: list[Transform] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            if _perm_sign_det(perm, signs) != 1:
                continue

            def make_fn(p=perm, s=signs):
                return lambda cell: (s[0] * cell[p[0]], s[1] * cell[p[1]], s[2] * cell[p[2]])

            transforms.append(
                Transform(len(transforms), f"rot3_{perm}_{signs}", 3, True, make_fn())
            )
    return transforms


def all_transforms(dimension: int, allow_reflection: bool = True) -> list[Transform]:
    if dimension == 2:
        return square_transforms(allow_reflection)
    if dimension == 3:
        return cube_rotations_3d()
    raise ValueError(f"unsupported dimension {dimension}")


def transform_then_translate(cells: Iterable[Cell], transform: Transform, offset: Cell) -> CellSet:
    return frozenset(add(transform.fn(c), offset) for c in cells)
