from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon

from _common import ROOT

from src.shapely_prt_solvers import image_to_shapely_polygon
from src.shapely_quantile_solvers import normalize_area_and_center


RESULTS = ROOT / "outputs" / "augmented_silhouette_vector"


def circle_geom(area: float = 4.0, segments: int = 240) -> Polygon:
    radius = math.sqrt(area / math.pi)
    return normalize_area_and_center(
        Polygon(
            [
                (radius * math.cos(2.0 * math.pi * i / segments), radius * math.sin(2.0 * math.pi * i / segments))
                for i in range(segments)
            ]
        ),
        area,
    )


def heart_geom(area: float = 4.0, segments: int = 360) -> Polygon:
    pts: list[tuple[float, float]] = []
    for i in range(segments):
        t = 2.0 * math.pi * i / segments
        x = 16.0 * math.sin(t) ** 3
        y = 13.0 * math.cos(t) - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t)
        pts.append((x, y))
    return normalize_area_and_center(Polygon(pts), area)


def base_silhouettes(simplify: float) -> dict[str, object]:
    paths: dict[str, Path] = {
        "nike": ROOT / "outputs" / "duncan_paper_inputs" / "masks" / "nike_normalized_area.png",
        "shoe": ROOT / "outputs" / "duncan_paper_inputs" / "masks" / "shoe_normalized_area.png",
        "swoosh_like": ROOT / "siluetteResults" / "continuous" / "inputs" / "swoosh_like_logo.png",
        "shoe_like": ROOT / "siluetteResults" / "continuous" / "inputs" / "shoe_like_silhouette.png",
        "fish": ROOT / "siluetteResults" / "continuous" / "inputs" / "fish.png",
        "bone": ROOT / "siluetteResults" / "continuous" / "inputs" / "bone.png",
    }
    out: dict[str, object] = {}
    for name, path in paths.items():
        if path.exists():
            out[name] = image_to_shapely_polygon(path, name=name, simplify=simplify)
    out["circle"] = circle_geom()
    out["heart"] = heart_geom()
    return out


def augmented_pairs(simplify: float) -> dict[str, tuple[object, object]]:
    shapes = base_silhouettes(simplify)
    wanted = [
        ("nike_to_shoe", "nike", "shoe"),
        ("nike_to_circle", "nike", "circle"),
        ("nike_to_heart", "nike", "heart"),
        ("shoe_to_circle", "shoe", "circle"),
        ("shoe_to_heart", "shoe", "heart"),
        ("circle_to_nike", "circle", "nike"),
        ("heart_to_shoe", "heart", "shoe"),
        ("swoosh_like_to_circle", "swoosh_like", "circle"),
        ("shoe_like_to_heart", "shoe_like", "heart"),
        ("fish_to_bone", "fish", "bone"),
    ]
    return {name: (shapes[a], shapes[b]) for name, a, b in wanted if a in shapes and b in shapes}


