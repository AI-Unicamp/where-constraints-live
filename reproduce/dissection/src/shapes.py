from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .lattice import Cell, Cell2, CellSet


@dataclass(frozen=True)
class TargetShape:
    id: int
    name: str
    cells: CellSet
    dimension: int

    @property
    def volume(self) -> int:
        return len(self.cells)


def target(id: int, name: str, cells: Iterable[Cell]) -> TargetShape:
    cells_fs = frozenset(cells)
    if not cells_fs:
        raise ValueError("target shape cannot be empty")
    dim = len(next(iter(cells_fs)))
    return TargetShape(id=id, name=name, cells=cells_fs, dimension=dim)


def rectangle(x0: int, y0: int, w: int, h: int) -> frozenset[Cell2]:
    return frozenset((x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h))


def l_shape() -> frozenset[Cell2]:
    return frozenset({(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 2)})


def polyomino_from_ascii(text: str, filled: str = "#") -> frozenset[Cell2]:
    rows = [row.rstrip() for row in text.strip("\n").splitlines()]
    cells: set[Cell2] = set()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == filled:
                cells.add((x, y))
    return frozenset(cells)


def normalize_targets(cell_sets: list[CellSet], names: list[str] | None = None) -> list[TargetShape]:
    if names is None:
        names = [f"S{i}" for i in range(len(cell_sets))]
    return [target(i, names[i], cells) for i, cells in enumerate(cell_sets)]
