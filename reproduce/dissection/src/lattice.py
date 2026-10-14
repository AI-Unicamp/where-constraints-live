from __future__ import annotations

from collections import deque
from typing import Iterable, TypeAlias

Cell2: TypeAlias = tuple[int, int]
Cell3: TypeAlias = tuple[int, int, int]
Cell: TypeAlias = Cell2 | Cell3
CellSet: TypeAlias = frozenset[Cell]
Atom: TypeAlias = tuple[int, Cell]


def add(a: Cell, b: Cell) -> Cell:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


