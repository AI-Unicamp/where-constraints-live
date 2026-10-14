r"""Emit LaTeX table bodies from experiment outputs.

Writes:
  tables/table_remake2d.tex   -- 2D Duncan-remake diagnostics (one row per
                                 finished pair; reruns pick up new pairs)
  tables/table_geo3d.tex      -- 3D certificate rows for bunny+geometric runs

Chapter 5 \input{}s these files, so numbers always match the latest runs.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "trash_stock_dissection"
BACO_OUT = REPO / "outputs" / "zhou_duncan_common_boundary_optimizer"
SP_OUT = REPO / "outputs" / "zhou3d_shared_patch_complex_baco"
TABLES = HERE / "tables"

# (run dir under BACO_OUT, table label, K, contact-RMS comparable?)
# Pairs whose existing runs already used Duncan-derived silhouettes at
# Duncan's piece counts reuse those runs; egg_bunny / serpent_apple /
# dog_bone come from the remake batch (extracted inputs / corrected K).
REMAKE_PAIRS = [
    ("exp358_cat_fish_k6_baco_N400_R100_directmask_001", "Cat--Fish", 6, True),
    ("exp358_dove_bomb_k6_baco_N400_R100_directmask_001", "Dove--Bomb", 6, True),
    ("exp358_bear_cat_k6_baco_N400_R100_directmask_001", "Bear--Cat", 6, True),
    ("duncan_serpent_apple_k5_baco_N400_R100", "Serpent--Apple", 5, True),
    ("exp358_catterpillar_butterfly_k6_baco_N400_R100_directmask_001", "Caterpillar--Butterfly", 6, True),
    ("exp358_x_island_k6_baco_N400_R100_directmask_001", "X--Island", 6, True),
    ("exp358_nike_shoe_k6_common_boundary_direct_long_001", "Nike--Shoe", 6, False),
    # remake preferred; falls back to the earlier trump->usa run (equivalent
    # dissection, same K) until the remake lands.
    (("duncan_usa_trump_k5_baco_N400_R100", "exp358_trump_usa_k6_baco_N400_R100_directmask_001"), "USA--Trump", 5, True),
    ("duncan_dog_bone_k6_baco_N400_R100", "Dog--Bone", 6, True),
    ("duncan_egg_bunny_k6_baco_N400_R100", "Egg--Bunny", 6, True),
]

GEO_RUNS = [
    ("exp453_r64_k4", "Bunny$\\to$Sphere", 4),
    ("exp453_r64_k5", "Bunny$\\to$Sphere", 5),
    ("exp453_r64_k6_v2", "Bunny$\\to$Sphere", 6),
    ("exp453_cube2sphere_k4", "Cube$\\to$Sphere", 4),
    ("exp453_cube2sphere_k6", "Cube$\\to$Sphere", 6),
    ("exp453_cube2sphere_k8", "Cube$\\to$Sphere", 8),
    ("exp453_sphere2pyramid_k4", "Sphere$\\to$Pyramid", 4),
    ("exp453_sphere2pyramid_k6", "Sphere$\\to$Pyramid", 6),
    ("exp453_sphere2pyramid_k8", "Sphere$\\to$Pyramid", 8),
    ("exp453_torus2cube_k4", "Torus$\\to$Cube", 4),
    ("exp453_torus2cube_k6", "Torus$\\to$Cube", 6),
    ("exp453_torus2cube_k8", "Torus$\\to$Cube", 8),
]


def read_summary(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def emit_remake2d() -> None:
    lines = []
    for run, label, k, rms_ok in REMAKE_PAIRS:
        candidates = (run,) if isinstance(run, str) else run
        s = None
        for cand in candidates:
            s = read_summary(BACO_OUT / cand / "summary.csv")
            if s is not None:
                break
        if s is None:
            continue
        shared = float(s["shared_area"])
        a_out = 100.0 * float(s["source_outside_area"]) / shared
        b_out = 100.0 * float(s["target_outside_area"]) / shared
        a_ovl = 100.0 * float(s["source_overlap_area"]) / shared
        b_ovl = 100.0 * float(s["target_overlap_area"]) / shared
        t = float(s["runtime_s"])
        if rms_ok:
            a_rms = f"{float(s['final_a_contact_rms_px']):.2f}"
            b_rms = f"{float(s['final_b_contact_rms_px']):.2f}"
        else:
            a_rms = b_rms = "--"  # legacy normalization, units not comparable
        lines.append(
            f"    {label} & {k} & {a_rms} & {b_rms} & {a_out:.2f} & "
            f"{b_out:.2f} & {a_ovl:.2f} & {b_ovl:.2f} & {t:.0f} \\\\"
        )
    TABLES.mkdir(exist_ok=True)
    # NOTE: the first line must be a plain row (no \multicolumn/\hline);
    # \input is unexpandable and the alignment row is already open when the
    # file content arrives.
    body = "\n".join(lines) if lines else ("    \\emph{runs pending}" + " &" * 8 + " \\\\")
    (TABLES / "table_remake2d.tex").write_text(body + "\n", encoding="utf-8")
    print(f"table_remake2d.tex: {len(lines)} rows")


def emit_geo3d() -> None:
    lines = []
    last_pair = None
    for run, label, k in GEO_RUNS:
        s = read_summary(SP_OUT / run / "summary.csv")
        if s is None:
            continue
        if last_pair is not None and label != last_pair:
            lines.append("    \\hline")
        last_pair = label
        m = float(s.get("audit_mismatch_total", "nan"))
        au = 100.0 * float(s["certificate_A_uncovered_fraction"])
        bu = 100.0 * float(s["certificate_B_uncovered_fraction"])
        asp = 100.0 * float(s["certificate_A_spill_fraction"])
        bsp = 100.0 * float(s["certificate_B_spill_fraction"])
        si = int(float(s.get("self_intersections_total", 0)))
        a_pass = "\\checkmark" if float(s.get("certificate_A_pass", 0)) > 0 else "--"
        b_pass = "\\checkmark" if float(s.get("certificate_B_pass", 0)) > 0 else "--"
        lines.append(
            f"    {label} & {k} & {m:.0f} & {au:.2f} & {asp:.2f} & {a_pass} & "
            f"{bu:.2f} & {bsp:.2f} & {b_pass} & {si} \\\\"
        )
    TABLES.mkdir(exist_ok=True)
    body = "\n".join(lines) if lines else ("    \\emph{runs pending}" + " &" * 9 + " \\\\")
    (TABLES / "table_geo3d.tex").write_text(body + "\n", encoding="utf-8")
    print(f"table_geo3d.tex: {len(lines)} rows")


# (run dir, baco subdir suffix, table label); the third block re-exports the
# default-lattice solution under the clearance seam + proximity guards
# (--clearance 0.1 --prox-weight 40 --grow-rounds 0), overlap exactly zero.
MULTI_RUNS = [
    ("exp462_bear_cat_fish_cycle", "", "Decaying (default)"),
    ("exp462_bear_cat_fish_cycle_area", "", "Constant (strengthened)"),
    ("exp462_bear_cat_fish_cycle", "_vari01", "Decaying, clearance seam"),
]
MULTI_OUT = REPO / "outputs" / "zhou_multitarget"


def emit_multitarget(k: int = 9) -> None:
    """tables/table_multitarget.tex: one row per (lattice run, assembly)."""
    import math

    lines = []
    for run, suffix, label in MULTI_RUNS:
        base = MULTI_OUT / run / f"baco_k{k}{suffix}"
        s = read_summary(base / "summary.csv")
        if s is None:
            continue
        with (base / "pieces.csv").open(newline="", encoding="utf-8") as f:
            areas = [float(r["area"]) for r in csv.DictReader(f)]
        mean_a = sum(areas) / max(len(areas), 1)
        min_frac = 100.0 * min(areas) / float(s["s0_shape_area"])
        cv = math.sqrt(sum((a - mean_a) ** 2 for a in areas) / len(areas)) / mean_a
        for t, shape in enumerate(("Bear", "Cat", "Fish")):
            first = (
                f"\\multirow{{3}}{{*}}{{{label}}} & "
                if t == 0 else " & "
            )
            tail = (
                f"\\multirow{{3}}{{*}}{{{min_frac:.1f}}} & \\multirow{{3}}{{*}}{{{cv:.2f}}}"
                if t == 0 else " & "
            )
            lines.append(
                f"    {first}{shape} & "
                f"{float(s[f's{t}_outside_pct']):.2f} & "
                f"{float(s[f's{t}_uncovered_pct']):.2f} & "
                f"{float(s[f's{t}_overlap_pct']):.2f} & "
                f"{float(s[f'final_s{t}_contact_rms_px']):.2f} & "
                f"{tail} \\\\"
            )
        lines.append("    \\hline")
    if lines and lines[-1] == "    \\hline":
        lines.pop()
    TABLES.mkdir(exist_ok=True)
    body = "\n".join(lines) if lines else ("    \\emph{runs pending}" + " &" * 7 + " \\\\")
    (TABLES / "table_multitarget.tex").write_text(body + "\n", encoding="utf-8")
    print(f"table_multitarget.tex: {len(lines)} rows")


if __name__ == "__main__":
    emit_remake2d()
    emit_geo3d()
    emit_multitarget()
