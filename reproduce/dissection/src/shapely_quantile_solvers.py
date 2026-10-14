from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from typing import Iterable

import numpy as np
from scipy.optimize import differential_evolution
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .continuous_parametric_solvers import Rigid2


def clean_geom(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return geom
    out = geom.buffer(0)
    if out.is_empty:
        return out
    return out


def polygon_components(geom: BaseGeometry, min_area: float = 1e-10) -> list[Polygon]:
    geom = clean_geom(geom)
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > min_area else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.area > min_area]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(polygon_components(g, min_area=min_area))
        return out
    return []


def apply_rigid(geom: BaseGeometry, transform: Rigid2) -> BaseGeometry:
    rotated = affinity.rotate(geom, math.degrees(transform.theta), origin=(0.0, 0.0))
    return clean_geom(affinity.translate(rotated, xoff=transform.tx, yoff=transform.ty))


@dataclass
class SPiece:
    domain_index: int
    transform_name: str
    geom: Polygon
    target_geom: Polygon
    generator: str
    params: tuple[float, ...] = ()

    @property
    def area(self) -> float:
        return float(self.geom.area)

    @property
    def perimeter(self) -> float:
        return float(self.geom.length)


@dataclass
class SSolution:
    method: str
    pieces: list[SPiece]
    runtime_ms: float
    notes: str = ""
    diagnostics: dict[str, float | int | str] | None = None

    @property
    def shared_area(self) -> float:
        return float(sum(p.area for p in self.pieces))


def _piece_overlap(pieces: list[SPiece], *, target: bool) -> float:
    total = 0.0
    for i, a in enumerate(pieces):
        ag = a.target_geom if target else a.geom
        for b in pieces[i + 1 :]:
            bg = b.target_geom if target else b.geom
            total += float(ag.intersection(bg).area)
    return total


def validate_shapely_solution(source: BaseGeometry, target: BaseGeometry, pieces: list[SPiece], tol: float = 1e-7) -> tuple[bool, dict[str, float]]:
    source_out = 0.0
    target_out = 0.0
    for p in pieces:
        source_out += float(p.geom.difference(source).area)
        target_out += float(p.target_geom.difference(target).area)
    source_overlap = _piece_overlap(pieces, target=False)
    target_overlap = _piece_overlap(pieces, target=True)
    valid = source_out <= tol and target_out <= tol and source_overlap <= tol and target_overlap <= tol
    return valid, {
        "source_outside_area": source_out,
        "target_outside_area": target_out,
        "source_overlap_area": source_overlap,
        "target_overlap_area": target_overlap,
    }


def normalize_area_and_center(geom: BaseGeometry, target_area: float = 4.0) -> BaseGeometry:
    scale = math.sqrt(target_area / float(geom.area))
    out = affinity.scale(geom, xfact=scale, yfact=scale, origin=(0.0, 0.0))
    c = out.centroid
    return clean_geom(affinity.translate(out, xoff=-c.x, yoff=-c.y))


def residual_geometries(source: BaseGeometry, target: BaseGeometry, pieces: list[SPiece]) -> tuple[BaseGeometry, BaseGeometry]:
    if not pieces:
        return source, target
    source_used = unary_union([p.geom for p in pieces])
    target_used = unary_union([p.target_geom for p in pieces])
    return clean_geom(source.difference(source_used)), clean_geom(target.difference(target_used))


def _svg_polygon_points(poly: Polygon, xoff: float, scale: float, pad: float, height: float) -> str:
    pts = []
    for x, y in poly.exterior.coords:
        sx = pad + xoff + x * scale
        sy = height - (pad + y * scale)
        pts.append(f"{sx:.3f},{sy:.3f}")
    return " ".join(pts)


def export_shapely_solution_svg(source: BaseGeometry, target: BaseGeometry, solution: SSolution, path) -> None:
    source_parts = polygon_components(source, min_area=0.0)
    target_parts = polygon_components(target, min_area=0.0)
    all_parts = source_parts + target_parts
    if not all_parts:
        return
    minx = min(g.bounds[0] for g in all_parts)
    miny = min(g.bounds[1] for g in all_parts)
    maxx = max(g.bounds[2] for g in all_parts)
    maxy = max(g.bounds[3] for g in all_parts)
    width_world = max(maxx - minx, 1e-6)
    height_world = max(maxy - miny, 1e-6)
    scale = 180.0 / max(width_world, height_world)
    pad = 40.0
    panel_w = width_world * scale + 2.0 * pad
    height = height_world * scale + 2.0 * pad
    gap = 80.0
    total_w = panel_w * 2.0 + gap
    x_shift = -minx
    y_shift = -miny

    def shifted(poly: Polygon) -> Polygon:
        return affinity.translate(poly, xoff=x_shift, yoff=y_shift)

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}" height="{height:.1f}" viewBox="0 0 {total_w:.1f} {height:.1f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for poly in source_parts:
        lines.append(
            f'<polygon points="{_svg_polygon_points(shifted(poly), 0.0, scale, pad, height)}" fill="#f7f7f7" stroke="#333" stroke-width="1.5"/>'
        )
    for poly in target_parts:
        lines.append(
            f'<polygon points="{_svg_polygon_points(shifted(poly), panel_w + gap, scale, pad, height)}" fill="#f7f7f7" stroke="#333" stroke-width="1.5"/>'
        )
    for i, piece in enumerate(solution.pieces):
        color = colors[i % len(colors)]
        for poly in polygon_components(piece.geom, min_area=0.0):
            lines.append(
                f'<polygon points="{_svg_polygon_points(shifted(poly), 0.0, scale, pad, height)}" fill="{color}" fill-opacity="0.65" stroke="#111" stroke-width="1"/>'
            )
        for poly in polygon_components(piece.target_geom, min_area=0.0):
            lines.append(
                f'<polygon points="{_svg_polygon_points(shifted(poly), panel_w + gap, scale, pad, height)}" fill="{color}" fill-opacity="0.65" stroke="#111" stroke-width="1"/>'
            )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
