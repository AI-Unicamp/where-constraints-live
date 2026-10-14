from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from .image_vectorization import vectorize_image
from .shapely_quantile_solvers import clean_geom, normalize_area_and_center


def image_to_shapely_polygon(
    path: Path,
    *,
    name: str,
    simplify: float = 1.5,
    threshold: int = 128,
    target_area: float = 4.0,
) -> BaseGeometry:
    shape = vectorize_image(
        path,
        threshold=threshold,
        largest_only=True,
        min_component_area=32,
        simplify_tolerance=simplify,
        corner_angle_degrees=30.0,
        max_edges_per_interval=12,
    )
    outers: list[Polygon] = []
    holes: list[Polygon] = []
    for loop in shape.loops:
        if len(loop.vertices) < 3:
            continue
        poly = clean_geom(Polygon(loop.vertices))
        if poly.is_empty or poly.area <= 1e-9:
            continue
        if loop.is_hole:
            holes.append(poly)
        else:
            outers.append(poly)
    if not outers:
        raise ValueError(f"could not vectorize silhouette: {path}")
    geom: BaseGeometry = clean_geom(outers[0])
    for poly in outers[1:]:
        geom = clean_geom(geom.union(poly))
    for hole in holes:
        geom = clean_geom(geom.difference(hole))
    geom = normalize_area_and_center(geom, target_area)
    return geom


