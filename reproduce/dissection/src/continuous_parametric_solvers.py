from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import time
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

FPoint = tuple[float, float]
FPoly = tuple[FPoint, ...]

EPS = 1e-9


def polygon_area(poly: FPoly) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def ensure_ccw(poly: FPoly) -> FPoly:
    if polygon_area(poly) < 0:
        return tuple(reversed(poly))
    return poly


@dataclass(frozen=True)
class Rigid2:
    name: str
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    @property
    def matrix(self) -> tuple[tuple[float, float], tuple[float, float]]:
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return ((c, -s), (s, c))

    def apply_point(self, p: FPoint) -> FPoint:
        (a, b), (c, d) = self.matrix
        return (a * p[0] + b * p[1] + self.tx, c * p[0] + d * p[1] + self.ty)

    def inverse_point(self, p: FPoint) -> FPoint:
        x = p[0] - self.tx
        y = p[1] - self.ty
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return (c * x + s * y, -s * x + c * y)

    def apply_poly(self, poly: FPoly) -> FPoly:
        return ensure_ccw(tuple(self.apply_point(p) for p in poly))

    def inverse_poly(self, poly: FPoly) -> FPoly:
        return ensure_ccw(tuple(self.inverse_point(p) for p in poly))


