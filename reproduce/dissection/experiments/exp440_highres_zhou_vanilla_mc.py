from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, zoom


ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT = ROOT / "outputs" / "zhou3d_common_boundary_optimizer" / "test_inputs" / "bunny_to_sphere"
ZHOU_EXE = ROOT / "outputs" / "zhou3d_build" / "exp383_cpp_zhou3d_from_zhou2" / "zhou3d.exe"
RESULTS = ROOT / "outputs" / "zhou3d_highres_vanilla_mc"


def read_binary_stack(path: Path) -> np.ndarray:
    files = sorted(path.glob("*.png"))
    if not files:
        raise ValueError(f"no PNG slices in {path}")
    slices = []
    for f in files:
        im = np.asarray(Image.open(f).convert("L"))
        # Zhou3D's C++ loader treats dark pixels as foreground.
        slices.append(im < 128)
    return np.stack(slices, axis=0)


def write_binary_stack(path: Path, mask: np.ndarray) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    for z in range(mask.shape[0]):
        arr = np.where(mask[z], 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path / f"z_{z:04d}.png")


def upsample_sdf(mask: np.ndarray, resolution: int, sigma_threshold: float = 0.0) -> np.ndarray:
    if mask.ndim != 3 or len(set(mask.shape)) != 1:
        raise ValueError(f"expected cubic 3D mask, got {mask.shape}")
    factor = float(resolution) / float(mask.shape[0])
    signed = distance_transform_edt(~mask) - distance_transform_edt(mask)
    field = zoom(signed.astype(np.float32), zoom=factor, order=1, mode="nearest")
    if field.shape != (resolution, resolution, resolution):
        fixed = np.zeros((resolution, resolution, resolution), dtype=field.dtype)
        z = min(resolution, field.shape[0])
        y = min(resolution, field.shape[1])
        x = min(resolution, field.shape[2])
        fixed[:z, :y, :x] = field[:z, :y, :x]
        if z < resolution:
            fixed[z:, :, :] = fixed[z - 1 : z, :, :]
        if y < resolution:
            fixed[:, y:, :] = fixed[:, y - 1 : y, :]
        if x < resolution:
            fixed[:, :, x:] = fixed[:, :, x - 1 : x]
        field = fixed
    return field < float(sigma_threshold)


def foreground_boundary_points(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    out = []
    d, h, w = mask.shape
    for z, y, x in pts.tolist():
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            zz, yy, xx = z + dz, y + dy, x + dx
            if zz < 0 or yy < 0 or xx < 0 or zz >= d or yy >= h or xx >= w or not mask[zz, yy, xx]:
                out.append((z, y, x))
                break
    return np.asarray(out, dtype=np.float64)


def sphere_like_old_target(old_sphere: np.ndarray, resolution: int, target_voxels: int) -> np.ndarray:
    pts = np.argwhere(old_sphere)
    if len(pts) == 0:
        raise ValueError("old sphere mask is empty")
    center_old = pts.mean(axis=0).astype(np.float64)
    boundary = foreground_boundary_points(old_sphere)
    radius_old = float(np.mean(np.linalg.norm(boundary - center_old[None, :], axis=1)))
    scale = float(resolution) / float(old_sphere.shape[0])
    center = (center_old + 0.5) * scale - 0.5
    radius0 = radius_old * scale

    zz, yy, xx = np.indices((resolution, resolution, resolution), dtype=np.float64)
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)

    lo = max(0.1, radius0 * 0.75)
    hi = max(lo + 1.0, radius0 * 1.25)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        count = int(np.count_nonzero(dist <= mid))
        if count < target_voxels:
            lo = mid
        else:
            hi = mid
    mask = dist <= hi
    # Exact count adjustment keeps the source/target foreground volumes equal,
    # matching the original Zhou setup.
    flat = dist.ravel()
    order = np.argsort(flat)
    exact = np.zeros(flat.shape, dtype=bool)
    exact[order[:target_voxels]] = True
    return exact.reshape((resolution, resolution, resolution))


def run_command(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, timeout: int | None = None) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=so, stderr=se, text=True, timeout=timeout)
    return int(proc.returncode)


