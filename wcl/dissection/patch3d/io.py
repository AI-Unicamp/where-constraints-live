"""Writing the tagged piece meshes."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .complex import Complex3, fwd
from .system import piece_triangles


def write_tagged_obj(path: Path, cx: Complex3, verts: np.ndarray, *, transformed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    offset = 1
    for label in cx.labels:
        tris, tags = piece_triangles(cx, label)
        if len(tris) == 0:
            continue
        used = sorted(set(int(v) for tri in tris for v in tri))
        remap = {v: i for i, v in enumerate(used)}
        pv = verts[used]
        if transformed:
            pv = fwd(cx, label, pv)
        lines.append(f"o piece_{label}")
        for v in pv:
            lines.append(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}")
        current = None
        for tri, tag in zip(tris.tolist(), tags):
            if tag != current:
                current = tag
                lines.append(f"g piece_{label}_{tag}")
            a, b, c = (remap[int(t)] for t in tri)
            lines.append(f"f {a + offset} {b + offset} {c + offset}")
        offset += len(pv)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


