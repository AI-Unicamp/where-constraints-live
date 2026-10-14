from __future__ import annotations

import itertools
import heapq
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scipy.optimize import minimize
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from _common import ROOT
from exp100_augmented_silhouette_vector import augmented_pairs

from src.shapely_quantile_solvers import clean_geom


RESULTS = ROOT / "outputs" / "motion_arrangement_search"
EPS = 1e-8


def tab_benchmark() -> tuple[BaseGeometry, BaseGeometry]:
    source = unary_union(
        [
            box(0, 0, 10, 10),
            box(-2, 4, 0, 6),
            box(4, 10, 6, 12),
        ]
    )
    target = unary_union(
        [
            box(0, 0, 10, 10),
            box(10, 1, 12, 3),
        ]
    )
    return clean_geom(source), clean_geom(target)


def load_benchmark(pair_name: str | None, simplify: float) -> tuple[str, BaseGeometry, BaseGeometry]:
    if not pair_name:
        source, target = tab_benchmark()
        return "rectilinear_two_tab_to_one_tab", source, target
    pairs = augmented_pairs(simplify)
    if pair_name not in pairs:
        raise ValueError(f"unknown --pair {pair_name!r}; available pairs: {sorted(pairs)}")
    source, target = pairs[pair_name]
    return pair_name, clean_geom(source), clean_geom(target)


