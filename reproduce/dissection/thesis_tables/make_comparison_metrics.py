"""Quantitative method comparison: reconstruction fidelity and piece quality
for Duncan et al.'s published results versus our BACO runs, measured with the
same pipeline on both sides.

For each pair and each assembly:
  - assembly geometry: Duncan's comes from the digitized published panels
    (color segmentation + vectorization, as in make_duncan_redraws); ours
    comes from the exported result SVG polygons.
  - the same reference silhouette (extracted from Duncan's paper) is
    registered to the assembly by the same scale+translation ICP.
  - reconstruction error = uncovered% + spill% of the reference shape area
    (symmetric difference, reported separately and combined).
  - piece quality = mean isoperimetric compactness 4*pi*A/P^2 over the
    assembly's pieces (1 = circle; lower = thinner/sprawling pieces).

Contact quality is not compared: printed figures draw pieces flush, so
contact gaps are not measurable from the published rasters.

Emits tables/table_method_comparison.tex and method_comparison.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

import make_duncan_input_masks as mk
import make_duncan_redraws as rd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # reproduce/dissection; the experiment scripts write to outputs/
BACO_OUT = REPO / "outputs" / "zhou_duncan_common_boundary_optimizer"
TABLES = HERE / "tables"

POLY_RE = re.compile(r'<polygon points="([^"]+)"([^/]*)/>')
ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')

# pair -> dict with duncan panel config and ours svg config.
#   duncan: panel key in rd.FIG20_PANELS (or fig22 special), assembly order
#           as list of shape names matching the panel's left/right assemblies
#   ours:   svg path; ours assembly order (left/right) as shape names
#   ref:    duncan_* input dir and file names for the reference silhouettes
PAIRS = {
    "cat_fish": {
        "duncan_panel": "cat_fish", "duncan_order": ["cat", "fish"],
        "ours_svg": "exp358_cat_fish_k6_baco_N400_R100_directmask_001/cat_to_fish_common_boundary.svg",
        "ours_order": ["cat", "fish"],
        "ref": ("duncan_cat_fish", {"cat": "cat.png", "fish": "fish.png"}),
        "k": 6,
    },
    "dove_bomb": {
        "duncan_panel": "dove_bomb", "duncan_order": ["dove", "bomb"],
        "ours_svg": "exp358_dove_bomb_k6_baco_N400_R100_directmask_001/dove_to_bomb_common_boundary.svg",
        "ours_order": ["dove", "bomb"],
        "ref": ("duncan_dove_bomb", {"dove": "dove.png", "bomb": "bomb.png"}),
        "k": 6,
    },
    "bear_cat": {
        "duncan_panel": "bear_cat", "duncan_order": ["bear", "cat"],
        "ours_svg": "exp358_bear_cat_k6_baco_N400_R100_directmask_001/bear_to_cat_common_boundary.svg",
        "ours_order": ["bear", "cat"],
        "ref": ("duncan_bear_cat", {"bear": "bear.png", "cat": "cat.png"}),
        "k": 6,
    },
    "serpent_apple": {
        "duncan_panel": "serpent_apple", "duncan_order": ["serpent", "apple"],
        "ours_svg": "duncan_serpent_apple_k5_baco_N400_R100/serpent_to_apple_common_boundary.svg",
        "ours_order": ["serpent", "apple"],
        "ref": ("duncan_serpent_apple", {"serpent": "serpent.png", "apple": "apple.png"}),
        "k": 5,
    },
    "caterpillar_butterfly": {
        "duncan_panel": "caterpillar_butterfly", "duncan_order": ["caterpillar", "butterfly"],
        "ours_svg": "exp358_catterpillar_butterfly_k6_baco_N400_R100_directmask_001/catterpillar_to_butterfly_common_boundary.svg",
        "ours_order": ["caterpillar", "butterfly"],
        "ref": ("duncan_caterpillar_butterfly", {"caterpillar": "caterpillar.png", "butterfly": "butterfly.png"}),
        "k": 6,
    },
    "island_x": {
        "duncan_panel": "island_x", "duncan_order": ["island", "x"],
        "ours_svg": "exp358_x_island_k6_baco_N400_R100_directmask_001/x_to_island_common_boundary.svg",
        "ours_order": ["x", "island"],
        "ref": ("duncan_island_x", {"island": "island.png", "x": "x.png"}),
        "k": 6,
    },
    "nike_shoe": {
        "duncan_panel": "nike_shoe", "duncan_order": ["shoe", "nike"],  # panel order BA
        "ours_svg": "exp358_nike_shoe_k6_common_boundary_direct_long_001/nike_to_shoe_common_boundary.svg",
        "ours_order": ["nike", "shoe"],
        "ref": ("duncan_nike_shoe", {"nike": "nike.png", "shoe": "shoe.png"}),
        "k": 6,
    },
    # remake preferred; falls back to the earlier trump->usa run until the
    # remake lands (fallback assemblies are ordered trump, usa).
    "usa_trump": {
        "duncan_panel": "usa_trump", "duncan_order": ["usa", "trump"],
        "ours_svg": "duncan_usa_trump_k5_baco_N400_R100/usa_to_trump_common_boundary.svg",
        "ours_order": ["usa", "trump"],
        "ours_svg_fallback": "exp358_trump_usa_k6_baco_N400_R100_directmask_001/trump_to_usa_common_boundary.svg",
        "ours_order_fallback": ["trump", "usa"],
        "ref": ("duncan_usa_trump", {"usa": "usa.png", "trump": "trump.png"}),
        "k": 5,
        "note": "textured",
    },
    "egg_bunny": {
        "duncan_panel": "FIG22_B", "duncan_order": ["bunny", "egg"],
        "ours_svg": "duncan_egg_bunny_k6_baco_N400_R100/egg_to_bunny_common_boundary.svg",
        "ours_order": ["egg", "bunny"],
        "ref": ("duncan_egg_bunny", {"egg": "egg.png", "bunny": "bunny.png"}),
        "k": 6,
    },
}

LABELS = {
    "cat_fish": "Cat--Fish", "dove_bomb": "Dove--Bomb", "bear_cat": "Bear--Cat",
    "serpent_apple": "Serpent--Apple", "caterpillar_butterfly": "Caterpillar--Butterfly",
    "island_x": "Island--X", "nike_shoe": "Nike--Shoe", "usa_trump": "USA--Trump",
    "egg_bunny": "Bunny--Egg",
}


def load_ref_mask(pair_dir: str, fname: str) -> np.ndarray:
    p = REPO / "outputs" / "bi_assembly_inputs" / pair_dir / fname
    return np.asarray(Image.open(p)) == 0


def polygons_to_masks(polys: list[np.ndarray], shape_hint: float = 4.0) -> list[np.ndarray]:
    """Rasterize polygons (shared canvas, scale x4 for accuracy)."""
    pts = np.vstack(polys)
    mn = pts.min(axis=0) - 2
    scale = shape_hint
    size = np.ceil((pts.max(axis=0) - mn + 4) * scale).astype(int)
    masks = []
    for poly in polys:
        img = Image.new("L", (int(size[0]), int(size[1])), 0)
        d = ImageDraw.Draw(img)
        d.polygon([tuple((p - mn) * scale) for p in poly], fill=1)
        masks.append(np.asarray(img, dtype=bool).T)  # (x, y) -> transpose to (row=y?) keep consistent
    return [m.T for m in masks]


def compactness(poly: np.ndarray) -> float:
    from shapely.geometry import Polygon

    p = Polygon(poly)
    if not p.is_valid:
        p = p.buffer(0)
    if p.area <= 0 or p.length <= 0:
        return 0.0
    return float(4.0 * np.pi * p.area / (p.length**2))


def resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    w = max(1, int(round(crop.shape[1] * scale)))
    h = max(1, int(round(crop.shape[0] * scale)))
    img = Image.fromarray(np.where(crop, 255, 0).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=bool)


def min_symdiff_pct(ref_mask: np.ndarray, union: np.ndarray) -> float:
    """Minimum symmetric difference between the reference silhouette and the
    assembly union, over all translations (exact, via FFT cross-correlation)
    and a bounded scale sweep, as % of the reference area. This grants every
    assembly its most favorable alignment, so the metric cannot be inflated
    by registration error."""
    from scipy.signal import fftconvolve

    ua = int(union.sum())
    best = np.inf
    s0 = float(np.sqrt(ua / max(int(ref_mask.sum()), 1)))
    for f in (0.94, 0.97, 1.0, 1.03, 1.06, 1.10):
        ref = resize_mask(ref_mask, s0 * f)
        ra = int(ref.sum())
        inter = fftconvolve(union.astype(np.float32), ref[::-1, ::-1].astype(np.float32), mode="full")
        best_inter = float(inter.max())
        sd = 100.0 * (ra + ua - 2.0 * best_inter) / ra
        best = min(best, sd)
    return float(best)


def assembly_metrics(piece_polys: list[np.ndarray], ref_mask: np.ndarray) -> dict[str, float]:
    """Identical metric pipeline for both methods: rasterize pieces, take the
    best-aligned symmetric difference against the reference silhouette;
    compactness on the vector pieces."""
    masks = polygons_to_masks(piece_polys, shape_hint=1.5)
    union = binary_fill_holes(np.logical_or.reduce(masks))
    return {
        "symdiff_pct": min_symdiff_pct(ref_mask, union),
        "compactness_mean": float(np.mean([compactness(p) for p in piece_polys])),
        "pieces": len(piece_polys),
    }


def duncan_assembly_polys(rgb: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pmask = rd.piece_pixel_mask(rgb)
    from scipy.ndimage import binary_dilation

    pmask &= ~binary_dilation(mk.input_color_mask(rgb), iterations=4)
    left, right = rd.split_assemblies(pmask)
    out = []
    for amask in (left, right):
        total = int(amask.sum())
        polys = []
        for piece, _color in rd.color_pieces(rgb, amask):
            if piece.sum() < 0.015 * total:
                continue
            polys.append(rd.simplify(rd.outer_contour(piece)))
        out.append(polys)
    return out[0], out[1]


def ours_assembly_polys(svg_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    text = svg_path.read_text(encoding="utf-8")
    polys = []
    for pm in POLY_RE.finditer(text):
        attrs = dict(ATTR_RE.findall(pm.group(2)))
        if attrs.get("fill", "") in ("#f7f7f7", "none"):
            continue  # input silhouettes / background
        pts = np.asarray([tuple(map(float, p.split(","))) for p in pm.group(1).split()])
        polys.append(pts)
    cx = [p[:, 0].mean() for p in polys]
    order = np.argsort(cx)
    gaps = np.diff(np.asarray(cx)[order])
    split = int(np.argmax(gaps)) + 1
    left = [polys[i] for i in order[:split]]
    right = [polys[i] for i in order[split:]]
    return left, right


def main() -> None:
    doc = fitz.open(mk.PDF)
    rows = []
    report = {}
    for pair, cfg in PAIRS.items():
        # ---- Duncan side ----
        if cfg["duncan_panel"] == "FIG22_B":
            page, box = rd.FIG22_PANELS["egg_bunny_design_b"]
            rgb = mk.render_box(doc, page, box)
        else:
            page, box, _order = rd.FIG20_PANELS[cfg["duncan_panel"]]
            rgb = mk.render_box(doc, page, box)
        d_left, d_right = duncan_assembly_polys(rgb)
        d_assemblies = dict(zip(cfg["duncan_order"], [d_left, d_right]))

        # ---- ours side ----
        svg = BACO_OUT / cfg["ours_svg"]
        order = cfg["ours_order"]
        if not svg.exists() and "ours_svg_fallback" in cfg:
            svg = BACO_OUT / cfg["ours_svg_fallback"]
            order = cfg["ours_order_fallback"]
        if not svg.exists():
            report[pair] = "ours pending"
            continue
        o_left, o_right = ours_assembly_polys(svg)
        o_assemblies = dict(zip(order, [o_left, o_right]))

        ref_dir, ref_files = cfg["ref"]
        shapes = list(ref_files)
        entry = {}
        for shape in shapes:
            ref = load_ref_mask(ref_dir, ref_files[shape])
            entry[shape] = {
                "duncan": assembly_metrics(d_assemblies[shape], ref),
                "baco": assembly_metrics(o_assemblies[shape], ref),
            }
        report[pair] = entry

        sA, sB = shapes
        def sd(e, m):  # best-aligned symmetric difference %
            return e[m]["symdiff_pct"]

        if pair == "nike_shoe":
            # The swoosh is only a few pixels wide at drawing scale; for both
            # methods, area-relative reconstruction percentages on this pair
            # are dominated by digitization noise. Recorded in the JSON but
            # excluded from the table.
            continue
        textured = cfg.get("note") == "textured"
        note = "$^\\dagger$" if textured else ""
        d_comp = (
            "--"
            if textured  # textured pieces merge under color segmentation
            else f"{(entry[sA]['duncan']['compactness_mean'] + entry[sB]['duncan']['compactness_mean']) / 2:.2f}"
        )
        rows.append(
            f"    {LABELS[pair]}{note} & {cfg['k']} & "
            f"{sd(entry[sA], 'duncan'):.1f} & {sd(entry[sA], 'baco'):.1f} & "
            f"{sd(entry[sB], 'duncan'):.1f} & {sd(entry[sB], 'baco'):.1f} & "
            f"{d_comp} & "
            f"{(entry[sA]['baco']['compactness_mean'] + entry[sB]['baco']['compactness_mean']) / 2:.2f} \\\\"
        )
    doc.close()

    TABLES.mkdir(exist_ok=True)
    body = "\n".join(rows) if rows else ("    \\emph{runs pending}" + " &" * 7 + " \\\\")
    (TABLES / "table_method_comparison.tex").write_text(body + "\n", encoding="utf-8")
    (HERE / "method_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"table_method_comparison.tex: {len(rows)} rows")
    for pair, v in report.items():
        if isinstance(v, str):
            print(f"  {pair}: {v}")


if __name__ == "__main__":
    main()
