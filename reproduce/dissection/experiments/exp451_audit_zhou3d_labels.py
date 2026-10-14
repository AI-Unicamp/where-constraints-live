from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from exp384_zhou3d_common_boundary_optimizer import decode_label_stack, geometry_match_labels_3d
from exp441_highres_bunny_sphere_piece_sweep import available_label_pairs, connected_audit


BEST_RE = re.compile(
    r"Best L(?P<level>\d+): Obj=(?P<objective>[^\s]+)"
    r"(?: Score=(?P<score>[^\s]+))?"
    r" M=(?P<mismatch>[^\s]+)"
    r"(?: labelsA=(?P<labels_a>\d+))?"
    r"(?: labelsB=(?P<labels_b>\d+))?"
    r"(?: target=(?P<target>\d+))?"
    r"(?: extraA=(?P<extra_a>\d+))?"
    r"(?: extraB=(?P<extra_b>\d+))?"
    r"(?: connected=(?P<connected>\d+))?"
)


def stdout_rows(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: dict[int, dict[str, object]] = {}
    for match in BEST_RE.finditer(text):
        level = int(match.group("level"))
        pieces = int(match.group("target") or level + 1)
        row: dict[str, object] = {
            "stdout_level": level,
            "stdout_pieces": pieces,
            "stdout_objective": float(match.group("objective")),
            "stdout_mismatch": float(match.group("mismatch")),
        }
        for key in ("score", "labels_a", "labels_b", "extra_a", "extra_b", "connected"):
            value = match.group(key)
            if value is None:
                continue
            out_key = "stdout_" + key
            row[out_key] = float(value) if key == "score" else int(value)
        rows[pieces] = row
    return rows


def label_volumes(labels: np.ndarray) -> dict[int, int]:
    return {int(v): int(np.count_nonzero(labels == v)) for v in np.unique(labels) if int(v) > 0}


def audit_pair(pieces: int, kind: str, a_dir: Path, b_dir: Path, stdout: dict[int, dict[str, object]]) -> dict[str, object]:
    la = decode_label_stack(a_dir)
    lb_raw = decode_label_stack(b_dir)
    _lb, transforms, match_diag = geometry_match_labels_3d(la, lb_raw)
    aa = connected_audit(a_dir)
    bb = connected_audit(b_dir)
    va = label_volumes(la)
    vb = label_volumes(lb_raw)
    total_mismatch = float(sum(tf.mismatch for tf in transforms.values()))
    total_abs_volume_delta = 0
    max_abs_volume_delta = 0
    for label, tf in transforms.items():
        delta = abs(va.get(label, 0) - vb.get(tf.target_label, 0))
        total_abs_volume_delta += delta
        max_abs_volume_delta = max(max_abs_volume_delta, delta)
    row: dict[str, object] = {
        "pieces": pieces,
        "kind": kind,
        "labels_a": str(a_dir),
        "labels_b": str(b_dir),
        "labels_a_count": aa["label_count"],
        "labels_b_count": bb["label_count"],
        "a_extra_components": aa["extra_components"],
        "b_extra_components": bb["extra_components"],
        "a_disconnected_label_count": aa["disconnected_label_count"],
        "b_disconnected_label_count": bb["disconnected_label_count"],
        "connected_pair": int(
            aa["extra_components"] == 0
            and bb["extra_components"] == 0
            and aa["label_count"] == pieces
            and bb["label_count"] == pieces
        ),
        "recomputed_mismatch": total_mismatch,
        "match_total_mismatch": float(match_diag.get("total_mismatch", total_mismatch)),
        "total_abs_volume_delta": total_abs_volume_delta,
        "max_abs_volume_delta": max_abs_volume_delta,
        "a_components_by_label": aa["components_by_label"],
        "b_components_by_label": bb["components_by_label"],
    }
    row.update(stdout.get(pieces, {}))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zhou-dir", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stdout_path = args.stdout or (args.zhou_dir.parent / "zhou_stdout.txt")
    stdout = stdout_rows(stdout_path)
    rows = [
        audit_pair(pieces, kind, a_dir, b_dir, stdout)
        for pieces, (a_dir, b_dir, kind) in available_label_pairs(args.zhou_dir).items()
    ]
    rows.sort(key=lambda r: (int(r["pieces"]), str(r["kind"])))

    out = args.out or (args.zhou_dir.parent / "label_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out}")
    for row in rows:
        print(
            f"k={row['pieces']:>2} {row['kind']:<6} "
            f"M={row['recomputed_mismatch']:>8.0f} "
            f"conn={row['connected_pair']} "
            f"extra=({row['a_extra_components']},{row['b_extra_components']}) "
            f"vol_delta={row['total_abs_volume_delta']}"
        )


if __name__ == "__main__":
    main()
