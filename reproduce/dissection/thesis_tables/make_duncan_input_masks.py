"""Extract Duncan et al.'s exact input silhouettes from the paper PDF and
build equal-area mask pairs for the 2D pipeline (exp335 + exp358).

Each shape is cropped from a page region containing only that input shape,
thresholded by distance-from-white (the inputs are pale yellow with a red
outline), hole-filled, and reduced to its largest connected component. Pair
masks follow the exp361 recipe: fit into 215x215 at 4800 foreground pixels,
center, then equalize counts exactly. Masks are written black-on-white
(foreground = 0), the convention the lattice-search binary reads.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label as cc_label

HERE = Path(__file__).resolve().parent
PDF = HERE / "dt-final-paper-dissections (1).pdf"
OUT = HERE.parent / "trash_stock_dissection" / "outputs" / "bi_assembly_inputs"
SIZE = 215
AREA = 4800
DPI = 300

# Pair-level crops: each box contains BOTH input shapes of a panel (and may
# contain bleed from the colored result pieces, removed by the color filter).
# pair -> (page 0-idx, box x0,y0,x1,y1, sort axis, (slot of A, slot of B),
#          (A name, B name))
PAIR_CROPS = {
    "duncan_cat_fish": (10, (0.05, 0.075, 0.20, 0.25), "y", (0, 1), ("cat", "fish")),
    "duncan_dove_bomb": (10, (0.50, 0.075, 0.65, 0.25), "y", (0, 1), ("dove", "bomb")),
    "duncan_bear_cat": (10, (0.05, 0.25, 0.20, 0.425), "y", (0, 1), ("bear", "cat")),
    "duncan_serpent_apple": (10, (0.50, 0.25, 0.65, 0.425), "y", (0, 1), ("serpent", "apple")),
    "duncan_caterpillar_butterfly": (10, (0.05, 0.43, 0.20, 0.60), "y", (0, 1), ("caterpillar", "butterfly")),
    "duncan_island_x": (10, (0.50, 0.43, 0.65, 0.60), "y", (0, 1), ("island", "x")),
    # Last gallery row is shorter; shoe sits ABOVE the swoosh in the panel,
    # but the pair direction is Nike logo -> shoe.
    "duncan_nike_shoe": (10, (0.05, 0.585, 0.20, 0.71), "y", (1, 0), ("nike", "shoe")),
    "duncan_usa_trump": (10, (0.50, 0.585, 0.65, 0.71), "y", (0, 1), ("usa", "trump")),
    "duncan_dog_bone": (1, (0.50, 0.265, 0.665, 0.36), "x", (0, 1), ("dog", "bone")),
}

# Single-shape crops (egg and bunny live in different sub-panels of Fig. 22).
SINGLE_CROPS = {
    "egg": (11, (0.50, 0.315, 0.60, 0.41)),
    "bunny": (11, (0.50, 0.495, 0.60, 0.58)),
}

SINGLE_PAIRS = {
    "duncan_egg_bunny": ("egg", "bunny"),
}


def render_box(doc: fitz.Document, page_idx: int, box: tuple[float, float, float, float]) -> np.ndarray:
    page = doc[page_idx]
    r = page.rect
    clip = fitz.Rect(
        r.x0 + box[0] * r.width, r.y0 + box[1] * r.height,
        r.x0 + box[2] * r.width, r.y0 + box[3] * r.height,
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72.0, DPI / 72.0), clip=clip, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return arr.copy()


def input_color_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels belonging to Duncan input shapes: pale yellow fill or red
    outline. Excludes the saturated piece colors (magenta, cyan, purple,
    orange, gold) that can bleed into a panel crop, and caption text. A small
    closing bridges hairline gaps inside one drawn shape (e.g. the separately
    outlined collar sliver of the Trump silhouette)."""
    from scipy.ndimage import binary_closing

    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    pale = (r > 200) & (g > 200) & (b > 120) & (b < 225) & (np.abs(r - g) < 30)
    red = (r > 140) & (g < 110) & (b < 110)
    return binary_fill_holes(binary_closing(pale | red, iterations=3))


def components_by_size(mask: np.ndarray, count: int) -> list[np.ndarray]:
    cc, n = cc_label(mask)
    if n < count:
        raise ValueError(f"expected at least {count} components, found {n}")
    sizes = np.bincount(cc.reshape(-1))
    order = np.argsort(sizes[1:])[::-1] + 1
    return [cc == int(order[i]) for i in range(count)]


def shape_mask(rgb: np.ndarray) -> np.ndarray:
    """Largest input-colored component (single-shape crops)."""
    return components_by_size(input_color_mask(rgb), 1)[0]


def pair_shapes(rgb: np.ndarray, axis: str) -> list[np.ndarray]:
    """Two largest input-colored components, ordered along the given axis."""
    comps = components_by_size(input_color_mask(rgb), 2)
    def centroid(m: np.ndarray) -> float:
        ys, xs = np.nonzero(m)
        return float(ys.mean()) if axis == "y" else float(xs.mean())
    comps.sort(key=centroid)
    return comps


def normalize_mask_area(mask: np.ndarray, target_area: int = AREA, size: int = SIZE) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    scale = math.sqrt(float(target_area) / max(float(mask.sum()), 1.0))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    if max(new_w, new_h) > size - 6:
        s2 = (size - 6) / float(max(new_w, new_h))
        new_w = max(1, int(round(new_w * s2)))
        new_h = max(1, int(round(new_h * s2)))
    img = Image.fromarray(np.where(crop, 255, 0).astype(np.uint8), mode="L")
    img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
    out = np.zeros((size, size), dtype=bool)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    out[y0: y0 + new_h, x0: x0 + new_w] = np.asarray(img, dtype=np.uint8) > 0
    return out


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return mask & ~binary_erosion(mask)


def equalize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa, bb = a.copy(), b.copy()
    while int(aa.sum()) != int(bb.sum()):
        if aa.sum() > bb.sum():
            pts = np.argwhere(boundary_pixels(aa))
            y, x = pts[np.lexsort((pts[:, 1], pts[:, 0]))][0]
            aa[y, x] = False
        else:
            pts = np.argwhere(boundary_pixels(bb))
            y, x = pts[np.lexsort((pts[:, 1], pts[:, 0]))][0]
            bb[y, x] = False
    return aa, bb


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L").save(path)


def main() -> None:
    doc = fitz.open(PDF)
    raw_pairs: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, str]]] = {}
    for pair, (page, box, axis, (slot_a, slot_b), names) in PAIR_CROPS.items():
        comps = pair_shapes(render_box(doc, page, box), axis)
        raw_pairs[pair] = (comps[slot_a], comps[slot_b], names)
    for pair, (sa, sb) in SINGLE_PAIRS.items():
        pa, ba = SINGLE_CROPS[sa], SINGLE_CROPS[sb]
        a = shape_mask(render_box(doc, pa[0], pa[1]))
        b = shape_mask(render_box(doc, ba[0], ba[1]))
        raw_pairs[pair] = (a, b, (sa, sb))
    doc.close()

    diag = {}
    tiles = []
    for pair, (ra, rb, (name_a, name_b)) in raw_pairs.items():
        a = normalize_mask_area(ra)
        b = normalize_mask_area(rb)
        a, b = equalize_pair(a, b)
        pair_dir = OUT / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_mask(pair_dir / f"{name_a}.png", a)
        save_mask(pair_dir / f"{name_b}.png", b)
        diag[pair] = {"a": int(a.sum()), "b": int(b.sum())}
        combo = np.ones((SIZE, 2 * SIZE + 8), dtype=np.uint8) * 255
        combo[:, :SIZE][a] = 0
        combo[:, SIZE + 8:][b] = 0
        tiles.append((pair, combo))

    rows = []
    for pair, combo in tiles:
        header = np.ones((16, combo.shape[1]), dtype=np.uint8) * 255
        rows.append(np.vstack([header, combo]))
    sheet = np.vstack(rows)
    Image.fromarray(sheet, mode="L").save(HERE / "figures" / "debug_duncan_masks.png")
    print(json.dumps(diag, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
