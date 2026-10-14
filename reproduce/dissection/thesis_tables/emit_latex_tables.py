r"""Emit the LaTeX table bodies of Chapter 5 from the deposited run outputs.

    python thesis_tables/emit_latex_tables.py --data <data>/dissection [--out tables]

Writes:
  table_remake2d.tex     -- 2D BACO diagnostics on the Duncan et al. pairs (Table "BACO diagnostics")
  table_multitarget.tex  -- bear/cat/fish three-shape dissection (Table "multitarget")
  table_geo3d.tex        -- bunny-to-sphere piece-count progression, K = 4, 5, 6 (Table "geo3d")

`<data>/dissection` is the `dissection/` folder of the data deposit. Each run folder holds the `summary.csv`
written by the experiment script (and `pieces.csv` for the three-shape runs).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# (run folder under 2d_duncan_remake_runs, table label, K, contact RMS comparable?)
# The Nike--Shoe run used an earlier contact normalisation, so its RMS is not reported.
REMAKE_PAIRS = [
    ("exp358_cat_fish_k6_baco_N400_R100_directmask_001", "Cat--Fish", 6, True),
    ("exp358_dove_bomb_k6_baco_N400_R100_directmask_001", "Dove--Bomb", 6, True),
    ("exp358_bear_cat_k6_baco_N400_R100_directmask_001", "Bear--Cat", 6, True),
    ("duncan_serpent_apple_k5_baco_N400_R100", "Serpent--Apple", 5, True),
    ("exp358_catterpillar_butterfly_k6_baco_N400_R100_directmask_001", "Caterpillar--Butterfly", 6, True),
    ("exp358_x_island_k6_baco_N400_R100_directmask_001", "X--Island", 6, True),
    ("exp358_nike_shoe_k6_common_boundary_direct_long_001", "Nike--Shoe", 6, False),
    ("duncan_usa_trump_k5_baco_N400_R100", "USA--Trump", 5, True),
    ("duncan_dog_bone_k6_baco_N400_R100", "Dog--Bone", 6, True),
    ("duncan_egg_bunny_k6_baco_N400_R100", "Egg--Bunny", 6, True),
]

# (run folder under 2d_bearcatfish_multitarget, table label). The third run re-exports the default-lattice
# solution with the clearance seam and proximity guards (--clearance 0.1 --prox-weight 40 --grow-rounds 0).
MULTI_RUNS = [
    ("exp462_bear_cat_fish_cycle__baco_k9", "Decaying (default)"),
    ("exp462_bear_cat_fish_cycle_area__baco_k9", "Constant (strengthened)"),
    ("exp462_bear_cat_fish_cycle__baco_k9_vari01", "Decaying, clearance seam"),
]

# (run folder under 3d_certificate_runs, K)
GEO_RUNS = [
    ("thesis_k4_corrected_cert", 4),
    ("thesis_k5_corrected_cert", 5),
    ("thesis_k6_corrected_cert", 6),
]


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def remake2d(data: Path) -> list[str]:
    lines = []
    for run, label, k, rms_ok in REMAKE_PAIRS:
        s = read_summary(data / "2d_duncan_remake_runs" / run / "summary.csv")
        shared = float(s["shared_area"])
        pct = lambda key: 100.0 * float(s[key]) / shared
        if rms_ok:
            a_rms = f"{float(s['final_a_contact_rms_px']):.2f}"
            b_rms = f"{float(s['final_b_contact_rms_px']):.2f}"
        else:
            a_rms = b_rms = "--"
        lines.append(
            f"    {label} & {k} & {a_rms} & {b_rms} & {pct('source_outside_area'):.2f} & "
            f"{pct('target_outside_area'):.2f} & {pct('source_overlap_area'):.2f} & "
            f"{pct('target_overlap_area'):.2f} & {float(s['runtime_s']):.0f} \\\\"
        )
    return lines


def multitarget(data: Path) -> list[str]:
    lines = []
    for run, label in MULTI_RUNS:
        base = data / "2d_bearcatfish_multitarget" / run
        s = read_summary(base / "summary.csv")
        with (base / "pieces.csv").open(newline="", encoding="utf-8") as f:
            areas = [float(r["area"]) for r in csv.DictReader(f)]
        mean = sum(areas) / len(areas)
        min_frac = 100.0 * min(areas) / float(s["s0_shape_area"])
        cv = math.sqrt(sum((a - mean) ** 2 for a in areas) / len(areas)) / mean
        for t, shape in enumerate(("Bear", "Cat", "Fish")):
            head = f"\\multirow{{3}}{{*}}{{{label}}} & " if t == 0 else " & "
            tail = (f"\\multirow{{3}}{{*}}{{{min_frac:.1f}}} & \\multirow{{3}}{{*}}{{{cv:.2f}}}"
                    if t == 0 else " & ")
            lines.append(
                f"    {head}{shape} & {float(s[f's{t}_outside_pct']):.2f} & "
                f"{float(s[f's{t}_uncovered_pct']):.2f} & {float(s[f's{t}_overlap_pct']):.2f} & "
                f"{float(s[f'final_s{t}_contact_rms_px']):.2f} & {tail} \\\\"
            )
        lines.append("    \\hline")
    return lines[:-1]


def geo3d(data: Path) -> list[str]:
    lines = []
    for run, k in GEO_RUNS:
        s = read_summary(data / "3d_certificate_runs" / run / "summary.csv")
        pct = lambda key: 100.0 * float(s[key])
        mismatch = f"{float(s['audit_mismatch_total']):,.0f}".replace(",", "{,}")
        lines.append(
            f"    {k} & {mismatch} & "
            f"{pct('certificate_A_uncovered_fraction'):.2f} & {pct('certificate_A_spill_fraction'):.2f} & "
            f"{pct('certificate_B_uncovered_fraction'):.2f} & {pct('certificate_B_spill_fraction'):.2f} & "
            f"${float(s['self_intersections_total']):.0f} + {float(s['cross_intersections_A_total']):.0f}$ \\\\"
        )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True, help="the dissection/ folder of the data deposit")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "tables")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    # The first line of each body must be a plain row: \input arrives inside an open alignment row.
    for name, rows in (("table_remake2d.tex", remake2d(a.data)),
                       ("table_multitarget.tex", multitarget(a.data)),
                       ("table_geo3d.tex", geo3d(a.data))):
        (a.out / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
