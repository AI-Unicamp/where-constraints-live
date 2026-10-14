"""Numerical validation: winding-number coverage/spill/overlap certificate and mesh statistics."""
from __future__ import annotations

import math
import igl
import numpy as np
from scipy import ndimage

from .alignment import MeshProjector
from .complex import Complex3, fwd
from .system import piece_triangles


def piece_watertight_stats(cx: Complex3, label: int) -> dict[str, int]:
    uses: dict[tuple[int, int], int] = {}
    for s in cx.sides:
        if s.label != label:
            continue
        c = cx.faces[s.face].corners
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])):
            key = (a, b) if a < b else (b, a)
            uses[key] = uses.get(key, 0) + 1
    vals = list(uses.values())
    return {
        "edges": len(vals),
        "open_edges": sum(1 for v in vals if v == 1),
        "overused_edges": sum(1 for v in vals if v > 2),
    }


def tri_area_sum(verts: np.ndarray, tris: np.ndarray) -> float:
    if len(tris) == 0:
        return 0.0
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Signed volume of a closed triangle surface (divergence theorem)."""
    if len(tris) == 0:
        return 0.0
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def inside_mesh(
    verts_zyx: np.ndarray, faces: np.ndarray, queries_zyx: np.ndarray, orient_sign: float = 0.0
) -> np.ndarray:
    verts = np.ascontiguousarray(verts_zyx, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    queries = np.ascontiguousarray(queries_zyx, dtype=np.float64)
    if hasattr(igl, "fast_winding_number_for_meshes"):
        w = igl.fast_winding_number_for_meshes(verts, tris, queries)
    else:
        w = igl.fast_winding_number(verts, tris, queries)
    w = np.asarray(w)
    # abs(w) > 0.5 cannot distinguish a piece that covers a region from a
    # piece that is inside-out over it. The zyx storage order mirrors
    # handedness relative to the OBJ xyz frame, so the convention is signed
    # winding normalized by the mesh's global orientation (the sign of its
    # signed volume). orient_sign = 0 asks this function to derive it; a piece
    # whose orientation disagrees with the complex convention reads as not
    # covering, and is flagged separately.
    if orient_sign == 0.0:
        orient_sign = math.copysign(1.0, signed_volume(verts_zyx, faces)) if len(faces) else 1.0
    return (w * orient_sign) > 0.5


def validation_certificate(
    cx: Complex3,
    verts: np.ndarray,
    la: np.ndarray,
    lb: np.ndarray,
    source_proj: MeshProjector,
    target_proj: MeshProjector,
    dilate: int = 2,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Winding-number certificate.

    - Fractions are normalized by the shape volume (winding-inside count of
      the input mesh over the query set), i.e. fractions of the input solid.
      Values normalized by the dilated query count are also emitted, as
      *_fraction_of_queries, for comparison; they are smaller (on a 64^3
      bunny by roughly a factor of 1.2).
    - Signed winding with an explicit orientation convention: the complex's
      canonical piece orientation is the majority sign of the per-piece
      signed volumes; a piece whose sign disagrees is counted in
      pieces_orientation_flipped and does not count as covering.
    - Query dilation is configurable (default 2), so spill farther than one
      voxel outside the voxel hull is measured.
    - Per-pair overlap counts are returned as rows (assembly, label_i,
      label_k, voxels), localizing B-interpenetration by piece pair for
      feedback to the discrete stage.

    This is a sampled diagnostic, not a proof: voxel-center queries alias
    sub-voxel features. Exact volumes would require mesh Booleans, which the
    closed piece surfaces allow.
    """
    out: dict[str, float] = {}
    pair_rows: list[dict[str, object]] = []
    piece_meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    piece_signs: dict[int, float] = {}
    for label in cx.labels:
        tris, _tags = piece_triangles(cx, label)
        piece_meshes[label] = (verts, tris)
        v = signed_volume(verts, tris)  # orientation is rigid-invariant
        piece_signs[label] = math.copysign(1.0, v) if v != 0.0 else 0.0
    sign_sum = sum(piece_signs.values())
    canonical = math.copysign(1.0, sign_sum) if sign_sum != 0.0 else 1.0
    flipped = [lab for lab, s in piece_signs.items() if s != canonical]
    out["pieces_orientation_flipped"] = float(len(flipped))
    out["piece_orientation_sign"] = float(canonical)

    for name, vol, proj, transformed in (
        ("A", la, source_proj, False),
        ("B", lb, target_proj, True),
    ):
        fg = ndimage.binary_dilation(vol > 0, iterations=max(1, int(dilate)))
        queries = np.argwhere(fg).astype(np.float64)
        if len(queries) == 0:
            continue
        counts = np.zeros(len(queries), dtype=np.int32)
        inside_by_label: dict[int, np.ndarray] = {}
        for label in cx.labels:
            pv, tris = piece_meshes[label]
            pv2 = fwd(cx, label, pv) if transformed else pv
            ins = inside_mesh(pv2, tris, queries, orient_sign=canonical)
            inside_by_label[label] = ins
            counts += ins.astype(np.int32)
        shape_inside = inside_mesh(proj.vertices_zyx, proj.faces, queries)
        union = counts >= 1
        nq = float(len(queries))
        shape_n = float(np.count_nonzero(shape_inside))
        denom = max(shape_n, 1.0)
        unc = float(np.count_nonzero(shape_inside & ~union))
        spl = float(np.count_nonzero(union & ~shape_inside))
        ovl = float(np.count_nonzero(counts >= 2))
        out[f"certificate_{name}_queries"] = nq
        out[f"certificate_{name}_dilate"] = float(dilate)
        out[f"certificate_{name}_shape_voxels"] = shape_n
        out[f"certificate_{name}_uncovered_voxels"] = unc
        out[f"certificate_{name}_spill_voxels"] = spl
        out[f"certificate_{name}_overlap_voxels"] = ovl
        # Shape-volume-normalized fractions (the stated definition).
        out[f"certificate_{name}_uncovered_fraction"] = unc / denom
        out[f"certificate_{name}_spill_fraction"] = spl / denom
        out[f"certificate_{name}_overlap_fraction"] = ovl / denom
        # Legacy query-normalized values, for continuity with older runs.
        out[f"certificate_{name}_uncovered_fraction_of_queries"] = unc / nq
        out[f"certificate_{name}_spill_fraction_of_queries"] = spl / nq
        out[f"certificate_{name}_overlap_fraction_of_queries"] = ovl / nq
        # Per-pair overlap localization.
        labs = sorted(inside_by_label)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                n_ij = int(np.count_nonzero(inside_by_label[labs[i]] & inside_by_label[labs[j]]))
                if n_ij:
                    pair_rows.append(
                        {
                            "assembly": name,
                            "label_i": labs[i],
                            "label_k": labs[j],
                            "overlap_voxels": n_ij,
                            "overlap_fraction_of_shape": n_ij / denom,
                        }
                    )
    return out, pair_rows


