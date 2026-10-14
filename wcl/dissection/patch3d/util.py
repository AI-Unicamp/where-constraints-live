"""Small numeric and CSV helpers."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import numpy as np



def rms(vals: np.ndarray) -> float:
    if len(vals) == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(vals)))))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


