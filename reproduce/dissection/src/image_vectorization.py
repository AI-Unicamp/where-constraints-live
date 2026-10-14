from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure, morphology


PointF = tuple[float, float]
VectorF = tuple[float, float]


@dataclass(frozen=True)
class BoundaryInterval:
    id: int
    loop_id: int
    start_vertex: int
    end_vertex: int
    vertices: list[PointF]
    edge_vectors: list[VectorF]
    length: float
    external: bool = True


@dataclass(frozen=True)
class VectorLoop:
    id: int
    vertices: list[PointF]
    signed_area: float
    edge_vectors: list[VectorF]
    intervals: list[BoundaryInterval]
    is_hole: bool


@dataclass(frozen=True)
class VectorizedShape:
    name: str
    image_path: str
    width: int
    height: int
    foreground_pixels: int
    loops: list[VectorLoop]
    simplify_tolerance: float
    corner_angle_degrees: float


def load_binary_mask(path: Path, threshold: int = 128, invert: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image)
    mask = arr < threshold
    if invert:
        mask = ~mask
    return mask.astype(bool)


def clean_mask(mask: np.ndarray, min_component_area: int = 32, close_radius: int = 1) -> np.ndarray:
    labels, n = ndimage.label(mask)
    cleaned = mask.copy()
    if n:
        counts = np.bincount(labels.ravel())
        remove = {idx for idx, count in enumerate(counts) if idx != 0 and count < min_component_area}
        if remove:
            cleaned[np.isin(labels, list(remove))] = False
    if close_radius > 0:
        cleaned = morphology.closing(cleaned, morphology.disk(close_radius))
    cleaned = ndimage.binary_fill_holes(cleaned)
    return cleaned.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labels == keep


def signed_area(vertices: list[PointF]) -> float:
    s = 0.0
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def edge_vectors(vertices: list[PointF]) -> list[VectorF]:
    return [(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1])]


def vector_length(v: VectorF) -> float:
    return float(np.hypot(v[0], v[1]))


def angle_degrees(a: VectorF, b: VectorF) -> float:
    la = vector_length(a)
    lb = vector_length(b)
    if la == 0 or lb == 0:
        return 0.0
    dot = (a[0] * b[0] + a[1] * b[1]) / (la * lb)
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def remove_duplicate_vertices(vertices: list[PointF], eps: float = 1e-9) -> list[PointF]:
    out: list[PointF] = []
    for p in vertices:
        if not out or abs(out[-1][0] - p[0]) > eps or abs(out[-1][1] - p[1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def contour_to_vertices(contour: np.ndarray, tolerance: float) -> list[PointF]:
    simplified = measure.approximate_polygon(contour, tolerance=tolerance)
    vertices = [(float(col), float(row)) for row, col in simplified]
    vertices = remove_duplicate_vertices(vertices)
    if len(vertices) < 3:
        return []
    return vertices


def corner_indices(vertices: list[PointF], corner_angle_degrees: float, max_edges_per_interval: int) -> list[int]:
    n = len(vertices)
    corners = {0}
    vecs = edge_vectors(vertices)
    for i in range(n):
        prev_v = vecs[(i - 1) % n]
        next_v = vecs[i]
        turn = angle_degrees(prev_v, next_v)
        if turn >= corner_angle_degrees:
            corners.add(i)
    if max_edges_per_interval > 0:
        for i in range(0, n, max_edges_per_interval):
            corners.add(i)
    return sorted(corners)


def slice_cyclic(vertices: list[PointF], start: int, end: int) -> list[PointF]:
    n = len(vertices)
    if start <= end:
        return vertices[start : end + 1]
    return vertices[start:] + vertices[: end + 1]


def build_intervals(
    loop_id: int,
    vertices: list[PointF],
    corner_angle_degrees: float,
    max_edges_per_interval: int,
) -> list[BoundaryInterval]:
    cuts = corner_indices(vertices, corner_angle_degrees, max_edges_per_interval)
    if len(cuts) == 1:
        cuts = [0, len(vertices) // 2]
    intervals: list[BoundaryInterval] = []
    for local_id, start in enumerate(cuts):
        end = cuts[(local_id + 1) % len(cuts)]
        interval_vertices = slice_cyclic(vertices, start, end)
        if len(interval_vertices) < 2:
            continue
        ev = [
            (x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in zip(interval_vertices, interval_vertices[1:])
        ]
        intervals.append(
            BoundaryInterval(
                id=local_id,
                loop_id=loop_id,
                start_vertex=start,
                end_vertex=end,
                vertices=interval_vertices,
                edge_vectors=ev,
                length=sum(vector_length(v) for v in ev),
            )
        )
    return intervals


def vectorize_mask(
    mask: np.ndarray,
    name: str,
    image_path: Path,
    simplify_tolerance: float = 2.0,
    corner_angle_degrees: float = 35.0,
    max_edges_per_interval: int = 12,
) -> VectorizedShape:
    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, level=0.5)
    loops: list[VectorLoop] = []
    for contour in contours:
        vertices = contour_to_vertices(contour - 1.0, simplify_tolerance)
        if len(vertices) < 3:
            continue
        area = signed_area(vertices)
        if abs(area) < 1e-6:
            continue
        intervals = build_intervals(
            len(loops),
            vertices,
            corner_angle_degrees=corner_angle_degrees,
            max_edges_per_interval=max_edges_per_interval,
        )
        loops.append(
            VectorLoop(
                id=len(loops),
                vertices=vertices,
                signed_area=area,
                edge_vectors=edge_vectors(vertices),
                intervals=intervals,
                is_hole=area < 0,
            )
        )
    loops.sort(key=lambda loop: abs(loop.signed_area), reverse=True)
    loops = [
        VectorLoop(
            id=i,
            vertices=loop.vertices,
            signed_area=loop.signed_area,
            edge_vectors=loop.edge_vectors,
            intervals=build_intervals(
                i,
                loop.vertices,
                corner_angle_degrees=corner_angle_degrees,
                max_edges_per_interval=max_edges_per_interval,
            ),
            is_hole=loop.is_hole,
        )
        for i, loop in enumerate(loops)
    ]
    return VectorizedShape(
        name=name,
        image_path=str(image_path),
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        foreground_pixels=int(mask.sum()),
        loops=loops,
        simplify_tolerance=simplify_tolerance,
        corner_angle_degrees=corner_angle_degrees,
    )


def vectorize_image(
    path: Path,
    threshold: int = 128,
    invert: bool = False,
    largest_only: bool = True,
    min_component_area: int = 32,
    simplify_tolerance: float = 2.0,
    corner_angle_degrees: float = 35.0,
    max_edges_per_interval: int = 12,
) -> VectorizedShape:
    mask = load_binary_mask(path, threshold=threshold, invert=invert)
    mask = clean_mask(mask, min_component_area=min_component_area)
    if largest_only:
        mask = largest_component(mask)
    return vectorize_mask(
        mask,
        name=path.stem,
        image_path=path,
        simplify_tolerance=simplify_tolerance,
        corner_angle_degrees=corner_angle_degrees,
        max_edges_per_interval=max_edges_per_interval,
    )


