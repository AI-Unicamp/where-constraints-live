from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from .shapely_quantile_solvers import SPiece, polygon_components, residual_geometries, validate_shapely_solution


@dataclass(frozen=True)
class SlackCertificate:
    valid_containment: bool
    valid_nonoverlap: bool
    valid_budget: bool
    shared_pieces: int
    source_slack_components: int
    target_slack_components: int
    total_components: int
    source_slack_area: float
    target_slack_area: float
    source_outside_area: float
    target_outside_area: float
    source_overlap_area: float
    target_overlap_area: float
    piece_areas: tuple[float, ...]
    source_slack_areas: tuple[float, ...]
    target_slack_areas: tuple[float, ...]


def slack_certificate(
    source: BaseGeometry,
    target: BaseGeometry,
    pieces: list[SPiece],
    *,
    k_total: int,
    min_component_area: float = 1e-10,
) -> SlackCertificate:
    valid, diag = validate_shapely_solution(source, target, pieces)
    source_residual, target_residual = residual_geometries(source, target, pieces)
    source_slack = polygon_components(source_residual, min_area=min_component_area)
    target_slack = polygon_components(target_residual, min_area=min_component_area)
    total = len(pieces) + len(source_slack) + len(target_slack)
    return SlackCertificate(
        valid_containment=diag["source_outside_area"] <= 1e-7 and diag["target_outside_area"] <= 1e-7,
        valid_nonoverlap=valid,
        valid_budget=total <= k_total,
        shared_pieces=len(pieces),
        source_slack_components=len(source_slack),
        target_slack_components=len(target_slack),
        total_components=total,
        source_slack_area=float(sum(g.area for g in source_slack)),
        target_slack_area=float(sum(g.area for g in target_slack)),
        source_outside_area=float(diag["source_outside_area"]),
        target_outside_area=float(diag["target_outside_area"]),
        source_overlap_area=float(diag["source_overlap_area"]),
        target_overlap_area=float(diag["target_overlap_area"]),
        piece_areas=tuple(float(p.area) for p in pieces),
        source_slack_areas=tuple(float(g.area) for g in source_slack),
        target_slack_areas=tuple(float(g.area) for g in target_slack),
    )


