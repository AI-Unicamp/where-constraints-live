"""Rigid alignment of the input meshes to the voxel label grid, and projection onto them."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import igl
import numpy as np

from .util import rms
from .voxel import write_obj
EPS = 1e-9


@dataclass
class MeshProjector:
    vertices_zyx: np.ndarray
    faces: np.ndarray
    chunk_size: int = 20000

    def project(self, pts_zyx: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_zyx, dtype=np.float64)
        if len(pts) == 0:
            return pts.copy()
        chunks: list[np.ndarray] = []
        for start in range(0, len(pts), max(1, int(self.chunk_size))):
            q = pts[start : start + max(1, int(self.chunk_size))]
            _sqr, _idx, closest = igl.point_mesh_squared_distance(q, self.vertices_zyx, self.faces)
            out = np.asarray(closest, dtype=np.float64)
            chunks.append(out.reshape((1, 3)) if out.ndim == 1 else out)
        return np.vstack(chunks)


def load_aligned_projector(path: Path) -> MeshProjector:
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a mesh")
    # Meshes prepared by exp450 are written with write_obj: OBJ xyz is
    # solver x/y/z, so internal z/y/x is the reversed coordinate order.
    verts_zyx = np.asarray(mesh.vertices, dtype=np.float64)[:, [2, 1, 0]]
    return MeshProjector(verts_zyx, np.asarray(mesh.faces, dtype=np.int32))


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Centers of the boundary faces of a voxel mask (z/y/x coordinates)."""
    inside = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    pts: list[np.ndarray] = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, inside.shape[axis] - 1)
        hi[axis] = slice(1, inside.shape[axis])
        a = inside[tuple(lo)]
        b = inside[tuple(hi)]
        zz, yy, xx = np.nonzero(a != b)
        if len(zz):
            p = np.column_stack([zz, yy, xx]).astype(np.float64) - 1.0
            p[:, axis] += 0.5
            pts.append(p)
    return np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with s R src + t ~= dst.

    With allow_scale=False this is a rigid Kabsch fit (s = 1). The
    certificate compares against the input meshes moved rigidly, not against
    meshes rescaled to fit the voxelization: independent per-mesh scales would
    break the equal-volume premise of the problem."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = dc.T @ sc / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_fix[2, 2] = -1.0
    rot = u @ s_fix @ vt
    if allow_scale:
        var_s = float(np.mean(np.sum(sc * sc, axis=1)))
        scale = float(np.trace(np.diag(d) @ s_fix) / max(var_s, EPS))
    else:
        scale = 1.0
    t = mu_d - scale * rot @ mu_s
    return scale, rot, t


def refine_projector_alignment(
    proj: MeshProjector, mask: np.ndarray, iters: int, allow_scale: bool = False
) -> tuple[MeshProjector, dict[str, float]]:
    """ICP: refine the mesh's fit to the voxel grid boundary.

    The prepared meshes were aligned by a uniform bbox fit, which can be off
    when the voxelization (an SDF-upsampled legacy mask) filled its bounding
    box differently. The label grid is the frame everything else lives in, so
    the mesh is moved, not the grid.

    Rigid by default: a similarity ICP would rescale the source and target
    meshes independently, so the certificate would compare against rescaled
    inputs at possibly inconsistent scales. Scale can be enabled with
    allow_scale=True; either way the total applied scale is reported.
    """
    bnd = mask_boundary_points(mask)
    verts = proj.vertices_zyx.copy()
    diag: dict[str, float] = {}
    if len(bnd) == 0 or iters <= 0:
        return proj, {"align_iters": 0.0, "align_scale_total": 1.0}
    scale_total = 1.0
    for it in range(int(iters)):
        cur = MeshProjector(verts, proj.faces)
        closest = cur.project(bnd)
        resid = np.linalg.norm(closest - bnd, axis=1)
        if it == 0:
            diag["align_rms_before"] = rms(resid)
        scale, rot, t = umeyama_similarity(closest, bnd, allow_scale=allow_scale)
        scale_total *= scale
        verts = (scale * (rot @ verts.T)).T + t
    cur = MeshProjector(verts, proj.faces)
    resid = np.linalg.norm(cur.project(bnd) - bnd, axis=1)
    diag["align_rms_after"] = rms(resid)
    diag["align_iters"] = float(iters)
    diag["align_scale_total"] = float(scale_total)
    return cur, diag


