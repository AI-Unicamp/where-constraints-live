from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from exp440_highres_zhou_vanilla_mc import BASE_INPUT, ROOT, ZHOU_EXE, read_binary_stack, run_command, sphere_like_old_target, upsample_sdf, write_binary_stack


OUT_ROOT = ROOT / "outputs" / "zhou3d_highres_piece_sweep"
MC_ROOT = ROOT / "outputs" / "zhou3d_common_boundary_optimizer"


BEST_RE = re.compile(
    r"Best L(?P<level>\d+): Obj=(?P<objective>[^\s]+)"
    r"(?: Score=(?P<score>[^\s]+))?"
    r" M=(?P<mismatch>[^\s]+)"
    r"(?: pieces=(?P<pieces>\d+))?"
    r"(?: labelsA=(?P<labels_a>\d+))?"
    r"(?: labelsB=(?P<labels_b>\d+))?"
    r"(?: target=(?P<target>\d+))?"
)
FINAL_RE = re.compile(r"Pieces:\s*(?P<pieces>\d+).*?Mismatch:\s*(?P<mismatch>[^\s]+)", re.S)


def read_csv_first(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_label_stack_rgb(path: Path) -> np.ndarray:
    files = sorted(path.glob("*.png"))
    if not files:
        raise ValueError(f"no PNG slices in {path}")
    color_to_id: dict[tuple[int, int, int], int] = {}
    slices: list[np.ndarray] = []
    for f in files:
        rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        out = np.zeros(rgb.shape[:2], dtype=np.int32)
        flat = rgb.reshape((-1, 3))
        labels = out.reshape((-1,))
        for idx, c in enumerate(flat):
            key = (int(c[0]), int(c[1]), int(c[2]))
            if key == (255, 255, 255):
                continue
            lid = color_to_id.setdefault(key, len(color_to_id) + 1)
            labels[idx] = lid
        slices.append(out)
    return np.stack(slices, axis=0)


def connected_audit(path: Path) -> dict[str, object]:
    labels = read_label_stack_rgb(path)
    label_ids = [int(x) for x in np.unique(labels) if int(x) > 0]
    structure = ndi.generate_binary_structure(3, 1)
    comps_by_label: dict[int, int] = {}
    total_components = 0
    extra_components = 0
    for lid in label_ids:
        _cc, n = ndi.label(labels == lid, structure=structure)
        comps_by_label[lid] = int(n)
        total_components += int(n)
        extra_components += max(0, int(n) - 1)
    return {
        "label_count": len(label_ids),
        "total_components": total_components,
        "extra_components": extra_components,
        "disconnected_label_count": sum(1 for v in comps_by_label.values() if v != 1),
        "components_by_label": ";".join(f"{k}:{v}" for k, v in sorted(comps_by_label.items())),
    }


def parse_zhou_stdout(text: str) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for m in BEST_RE.finditer(text):
        level = int(m.group("level"))
        pieces = int(m.group("target") or m.group("pieces") or m.group("labels_a") or (level + 1))
        mismatch = float(m.group("mismatch"))
        obj = float(m.group("objective"))
        row = {
            "zhou_level": level,
            "zhou_pieces_reported": pieces,
            "zhou_stdout_mismatch": mismatch,
            "zhou_stdout_objective": obj,
        }
        if m.group("score") is not None:
            row["zhou_stdout_score"] = float(m.group("score"))
        if m.group("labels_a") is not None:
            row["zhou_stdout_labels_a"] = int(m.group("labels_a"))
        if m.group("labels_b") is not None:
            row["zhou_stdout_labels_b"] = int(m.group("labels_b"))
        out[pieces] = row
    final = FINAL_RE.search(text)
    if final:
        pieces = int(final.group("pieces"))
        out.setdefault(pieces, {})
        out[pieces]["zhou_final_pieces"] = pieces
        out[pieces]["zhou_final_mismatch"] = float(final.group("mismatch"))
    return out


def available_label_pairs(zhou_dir: Path) -> dict[int, tuple[Path, Path, str]]:
    pairs: dict[int, tuple[Path, Path, str]] = {}
    for prefix in ("result", "level"):
        for a_dir in zhou_dir.glob(f"{prefix}_*pcs_A"):
            try:
                pieces = int(a_dir.name.split("_")[1].replace("pcs", ""))
            except Exception:
                continue
            b_dir = zhou_dir / f"{prefix}_{pieces}pcs_B"
            if b_dir.exists():
                # Prefer result over level at the same piece count.
                if pieces not in pairs or prefix == "result":
                    pairs[pieces] = (a_dir, b_dir, prefix)
    return dict(sorted(pairs.items()))


def sum_piece_mismatch(piece_csv: Path) -> float | None:
    if not piece_csv.exists():
        return None
    total = 0.0
    found = False
    with piece_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                total += float(row["zhou_mismatch"])
                found = True
            except Exception:
                pass
    return total if found else None


def run_mc_for_pair(args: argparse.Namespace, labels_a: Path, labels_b: Path, tag: str, log_dir: Path) -> tuple[int, Path]:
    out_dir = MC_ROOT / tag
    if out_dir.exists() and not args.force_mc:
        return 0, out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "exp384_zhou3d_common_boundary_optimizer.py"),
        "--labels-a",
        str(labels_a),
        "--labels-b",
        str(labels_b),
        "--tag",
        tag,
        "--max-contact-samples",
        str(args.mc_max_contact_samples),
        "--max-exterior-samples",
        str(args.mc_max_exterior_samples),
        "--lsqr-iter",
        str(args.mc_lsqr_iter),
    ]
    rc = run_command(cmd, ROOT, log_dir / f"{tag}_mc_stdout.txt", log_dir / f"{tag}_mc_stderr.txt", timeout=None)
    return rc, out_dir


def run_one_config(args: argparse.Namespace, resolution: int, zhou_n: int, zhou_r: int, max_level: int, phase_tag: str) -> list[dict[str, object]]:
    phase_start = time.perf_counter()
    out_dir = OUT_ROOT / phase_tag
    input_dir = out_dir / "inputs"
    zhou_dir = out_dir / "zhou3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    old_bunny = read_binary_stack(Path(args.base_input) / "bunny")
    old_sphere = read_binary_stack(Path(args.base_input) / "sphere")
    bunny_hi = upsample_sdf(old_bunny, resolution)
    sphere_hi = sphere_like_old_target(old_sphere, resolution, int(np.count_nonzero(bunny_hi)))
    write_binary_stack(input_dir / "bunny", bunny_hi)
    write_binary_stack(input_dir / "sphere", sphere_hi)

    zhou_cmd = [
        str(Path(args.zhou_exe)),
        str(input_dir / "bunny"),
        str(input_dir / "sphere"),
        "-o",
        str(zhou_dir),
        "-N",
        str(zhou_n),
        "-R",
        str(zhou_r),
        "-Nb",
        str(args.zhou_nb),
        "--max-level",
        str(max_level),
        "--parent-split-N",
        str(args.parent_split_n),
        "--parent-split-R",
        str(args.parent_split_r),
        "--seed",
        str(args.seed),
        "-j",
        str(args.jobs),
    ]
    if not args.allow_disconnected_zhou_search:
        zhou_cmd.append("--require-connected")
    zhou_rc = 0
    if args.force_zhou and zhou_dir.exists():
        shutil.rmtree(zhou_dir)
    if not zhou_dir.exists():
        zhou_rc = run_command(zhou_cmd, ROOT, out_dir / "zhou_stdout.txt", out_dir / "zhou_stderr.txt", timeout=None)

    zhou_stdout = (out_dir / "zhou_stdout.txt").read_text(encoding="utf-8", errors="replace") if (out_dir / "zhou_stdout.txt").exists() else ""
    zhou_by_pieces = parse_zhou_stdout(zhou_stdout)
    pairs = available_label_pairs(zhou_dir)

    rows: list[dict[str, object]] = []
    for pieces, (labels_a, labels_b, source_kind) in pairs.items():
        if pieces < args.min_pieces or pieces > args.max_pieces:
            continue
        tag = f"{phase_tag}_k{pieces}_{source_kind}_exp384"
        audit_a = connected_audit(labels_a)
        audit_b = connected_audit(labels_b)
        invalid_topology = (
            int(audit_a["extra_components"]) > 0
            or int(audit_b["extra_components"]) > 0
            or int(audit_a["label_count"]) != int(audit_b["label_count"])
            or int(audit_a["label_count"]) != pieces
        )
        if invalid_topology and not args.allow_disconnected_labels:
            mc_rc = -99
            mc_out = MC_ROOT / tag
            summary: dict[str, str] = {}
        else:
            mc_rc, mc_out = run_mc_for_pair(args, labels_a, labels_b, tag, out_dir)
            summary = read_csv_first(mc_out / "summary.csv")
        scale = float(resolution) / float(old_bunny.shape[0])
        row: dict[str, object] = {
            "phase_tag": phase_tag,
            "resolution": resolution,
            "scale_vs_old32": scale,
            "zhou_N": zhou_n,
            "zhou_R": zhou_r,
            "zhou_nb": args.zhou_nb,
            "zhou_max_level": max_level,
            "seed": args.seed,
            "label_pair_kind": source_kind,
            "pieces": pieces,
            "labels_a": str(labels_a),
            "labels_b": str(labels_b),
            "topology_invalid": int(invalid_topology),
            "a_label_count": int(audit_a["label_count"]),
            "b_label_count": int(audit_b["label_count"]),
            "a_total_components": int(audit_a["total_components"]),
            "b_total_components": int(audit_b["total_components"]),
            "a_extra_components": int(audit_a["extra_components"]),
            "b_extra_components": int(audit_b["extra_components"]),
            "a_disconnected_label_count": int(audit_a["disconnected_label_count"]),
            "b_disconnected_label_count": int(audit_b["disconnected_label_count"]),
            "a_components_by_label": audit_a["components_by_label"],
            "b_components_by_label": audit_b["components_by_label"],
            "zhou_returncode": zhou_rc,
            "mc_returncode": mc_rc,
            "mc_out": str(mc_out),
            "piece_mismatch_total": sum_piece_mismatch(mc_out / "pieces.csv"),
            "phase_runtime_s_so_far": time.perf_counter() - phase_start,
        }
        row.update(zhou_by_pieces.get(pieces, {}))
        for k, v in summary.items():
            try:
                fv = float(v)
                row[f"mc_{k}"] = fv
                if k.startswith("final_"):
                    row[f"mc_{k}_old32_units"] = fv / scale
            except Exception:
                row[f"mc_{k}"] = v
        rows.append(row)
        write_rows(out_dir / "piece_sweep_partial.csv", rows)
        write_rows(OUT_ROOT / "piece_sweep_all_partial.csv", collect_existing_rows())

    diag = {
        "phase_tag": phase_tag,
        "out_dir": str(out_dir),
        "resolution": resolution,
        "bunny_voxels": int(np.count_nonzero(bunny_hi)),
        "sphere_voxels": int(np.count_nonzero(sphere_hi)),
        "zhou_cmd": zhou_cmd,
        "zhou_returncode": zhou_rc,
        "available_pieces": sorted(pairs),
        "evaluated_pieces": [int(r["pieces"]) for r in rows],
        "runtime_s": time.perf_counter() - phase_start,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2, sort_keys=True), encoding="utf-8")
    write_rows(out_dir / "piece_sweep.csv", rows)
    return rows


def collect_existing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p in OUT_ROOT.glob("*/piece_sweep.csv"):
        with p.open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    for p in OUT_ROOT.glob("*/piece_sweep_partial.csv"):
        with p.open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    dedup: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        dedup[(str(row.get("phase_tag")), str(row.get("pieces")))] = row
    return list(dedup.values())


def parse_phase(s: str) -> tuple[int, int, int, int, str]:
    # resolution:zhouN:zhouR:maxLevel[:tag]
    parts = s.split(":")
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError("phase must be resolution:zhouN:zhouR:maxLevel[:tag]")
    resolution, zhou_n, zhou_r, max_level = (int(parts[i]) for i in range(4))
    tag = parts[4] if len(parts) == 5 else f"r{resolution}_N{zhou_n}_R{zhou_r}_L{max_level}"
    return resolution, zhou_n, zhou_r, max_level, tag


def main() -> None:
    p = argparse.ArgumentParser(description="Overnight high-resolution bunny->sphere Zhou piece-count sweep with vanilla MC evaluation.")
    p.add_argument("--base-input", default=str(BASE_INPUT))
    p.add_argument("--zhou-exe", default=str(ZHOU_EXE))
    p.add_argument("--phase", action="append", type=parse_phase, help="resolution:zhouN:zhouR:maxLevel[:tag]. Can be repeated.")
    p.add_argument("--min-pieces", type=int, default=2)
    p.add_argument("--max-pieces", type=int, default=24)
    p.add_argument("--zhou-nb", type=int, default=10)
    p.add_argument("--parent-split-n", type=int, default=35)
    p.add_argument("--parent-split-r", type=int, default=12)
    p.add_argument("--seed", type=int, default=2012)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--mc-max-contact-samples", type=int, default=1400)
    p.add_argument("--mc-max-exterior-samples", type=int, default=1800)
    p.add_argument("--mc-lsqr-iter", type=int, default=1800)
    p.add_argument("--force-zhou", action="store_true")
    p.add_argument("--force-mc", action="store_true")
    p.add_argument("--allow-disconnected-zhou-search", action="store_true", help="do not ask the C++ Zhou3D search to filter final candidates by connected labels")
    p.add_argument("--allow-disconnected-labels", action="store_true", help="evaluate invalid disconnected label outputs anyway")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    phases = args.phase or [
        (48, 80, 25, 16, "r48_N80_R25_L16"),
        (64, 80, 25, 16, "r64_N80_R25_L16"),
        (64, 120, 40, 18, "r64_N120_R40_L18"),
    ]

    all_rows: list[dict[str, object]] = []
    sweep_start = time.perf_counter()
    for resolution, zhou_n, zhou_r, max_level, tag in phases:
        rows = run_one_config(args, resolution, zhou_n, zhou_r, max_level, tag)
        all_rows.extend(rows)
        write_rows(OUT_ROOT / "piece_sweep_all.csv", collect_existing_rows())

    final_rows = collect_existing_rows()
    write_rows(OUT_ROOT / "piece_sweep_all.csv", final_rows)
    payload = {
        "out_root": str(OUT_ROOT),
        "phases": [
            {"resolution": r, "zhou_N": n, "zhou_R": rr, "max_level": l, "tag": tag}
            for r, n, rr, l, tag in phases
        ],
        "row_count": len(final_rows),
        "runtime_s": time.perf_counter() - sweep_start,
    }
    (OUT_ROOT / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
