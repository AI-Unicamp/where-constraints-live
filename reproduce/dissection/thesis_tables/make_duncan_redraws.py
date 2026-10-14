"""Redraw Duncan et al.'s published result figures in this thesis's style,
from data extracted off the published figures (no image reproduction).

For each result panel: pieces are segmented by color clustering, vectorized
to polygons, and re-rendered with this thesis's styling (translucent piece
fills over the input silhouette, which is registered to each assembly by a
similarity ICP). The input outline behind each assembly makes the deviation
of the assembled silhouette from the input directly visible -- the same
presentation used for our results.

Outputs: figures/duncanredraw_<pair>.png and a QA sheet
figures/qa_duncan_redraws.png (original panel next to the redraw).
"""
from __future__ import annotations

from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_fill_holes, binary_opening, label as cc_label
from skimage import measure

import make_duncan_input_masks as mk

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"

# Fig. 20 panels (page 10, 0-indexed): generous boxes; captions/text are
# ignored by the color filters. (x0, y0, x1, y1) page fractions, plus the
# order of the two assemblies in the panel ("AB": pair-A assembly on the
# left, "BA": pair-B assembly on the left -- the nike/shoe panel shows the
# shoe assembly first).
FIG20_PANELS = {
    "cat_fish": (10, (0.05, 0.075, 0.50, 0.25), "AB"),
    "dove_bomb": (10, (0.50, 0.075, 0.97, 0.25), "AB"),
    "bear_cat": (10, (0.05, 0.25, 0.50, 0.425), "AB"),
    "serpent_apple": (10, (0.50, 0.25, 0.97, 0.425), "AB"),
    "caterpillar_butterfly": (10, (0.05, 0.43, 0.50, 0.583), "AB"),
    "island_x": (10, (0.50, 0.43, 0.97, 0.583), "AB"),
    # Row 4 assemblies rise above the row's nominal top (Trump's hair peaks
    # at ~0.582); the band above contains only caption text, which the color
    # filters ignore.
    "nike_shoe": (10, (0.05, 0.565, 0.50, 0.72), "BA"),
    "usa_trump": (10, (0.50, 0.565, 0.97, 0.72), "AB"),
}

# Fig. 22 sub-panels (page 11): each shows [one input | assembly1 | assembly2]
# where assembly1 reconstructs the bunny and assembly2 the egg.
FIG22_PANELS = {
    "egg_bunny_design_a": (11, (0.50, 0.30, 0.97, 0.47)),
    "egg_bunny_design_b": (11, (0.50, 0.47, 0.97, 0.64)),
}


def piece_pixel_mask(rgb: np.ndarray) -> np.ndarray:
    """Saturated piece colors; excludes white, black text, pale-yellow inputs
    and their dark-red outline."""
    arr = rgb.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturated = (mx - mn) > 45
    bright = mx > 90
    dark_red_outline = (r > 120) & (g < 100) & (b < 100) & ((r - g) > 80) & ((r - b) > 80) & (b < 90) & (g < 90)
    pale = (r > 200) & (g > 200) & (b > 120)
    return saturated & bright & ~dark_red_outline & ~pale


def split_assemblies(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split piece pixels into two assemblies at the largest x gap."""
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        raise ValueError("no piece pixels")
    gaps = np.diff(cols)
    gi = int(np.argmax(gaps))
    split_x = (cols[gi] + cols[gi + 1]) // 2
    left = mask.copy()
    left[:, split_x:] = False
    right = mask.copy()
    right[:, :split_x] = False
    return left, right


def color_pieces(rgb: np.ndarray, mask: np.ndarray, *, min_area: int = 350) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    """Cluster piece pixels by color; return (component mask, mean color 0-1)
    per piece."""
    pts = rgb[mask].astype(np.float64)
    if len(pts) < min_area:
        return []
    k = min(8, max(2, len(pts) // 2000 + 2))
    centroids, labels = kmeans2(pts, k, minit="++", seed=7)
    # Merge near-identical clusters.
    merged = list(range(k))
    for i in range(k):
        for j in range(i):
            if np.linalg.norm(centroids[i] - centroids[j]) < 55:
                merged[i] = merged[j]
                break
    lab_img = np.full(mask.shape, -1, dtype=np.int32)
    lab_img[mask] = [merged[l] for l in labels]

    # Inpaint thin strokes drawn OVER the pieces in the original figures
    # (dark piece outlines, and the dashed highlight circles of their
    # Fig. 22): every non-white pixel near the assembly that was excluded by
    # the color filter is assigned to the nearest piece. This repairs bites
    # where a stroke crosses a piece's outer boundary, which morphological
    # closing cannot.
    from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt

    arr = rgb.astype(np.int16)
    not_white = arr.min(axis=2) < 200
    near = binary_dilation(mask, iterations=4)
    stroke = not_white & near & ~mask
    dist, (iy, ix) = distance_transform_edt(lab_img < 0, return_indices=True)
    fill = stroke & (dist <= 6)
    lab_img[fill] = lab_img[iy[fill], ix[fill]]

    pieces = []
    for cl in sorted(set(merged)):
        cmask = lab_img == cl
        if cmask.sum() < min_area:
            continue
        cmask = binary_fill_holes(binary_closing(binary_fill_holes(cmask), iterations=2))
        cmask = binary_opening(cmask, iterations=1)
        comps, n = cc_label(cmask)
        for ci in range(1, n + 1):
            piece = comps == ci
            if piece.sum() < min_area:
                continue
            color = rgb[piece].mean(axis=0) / 255.0
            pieces.append((piece, tuple(color)))
    return pieces


def line_pieces(rgb: np.ndarray, mask: np.ndarray, *, min_area: int = 350) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    """Segment pieces by the grey boundary strokes drawn between fills.

    For panels where fill color does NOT identify pieces (the usa_trump
    panel reuses orange and yellow for adjacent pieces), only the thin grey
    seam lines define piece boundaries. Pieces are the connected components
    of the assembly support minus low-chroma stroke pixels; stroke pixels
    are then absorbed into the nearest region. Decorative open strokes (the
    facial-feature lines) cannot disconnect a region, so they are absorbed
    harmlessly."""
    from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt

    arr = rgb.astype(np.int16)
    support = binary_fill_holes(binary_closing(mask, iterations=2))
    chroma = arr.max(axis=2) - arr.min(axis=2)
    lum = arr.mean(axis=2)
    # The panel uses GREY strokes for piece seams and BLACK strokes for
    # decoration (facial features, sketch lines). Only grey cuts pieces;
    # black is absorbed below so closed decorative loops cannot fabricate
    # phantom pieces.
    grey_cut = support & (chroma < 45) & (lum >= 90) & (lum < 200)
    black_dec = support & (chroma < 45) & (lum < 90)
    stroke = grey_cut | black_dec
    interior = support & ~binary_dilation(grey_cut, iterations=1) & ~black_dec

    # Cut on fill-color classes as well: anti-aliased gaps in the thin seam
    # lines can connect differently colored fills, and color classes cannot
    # separate same-colored neighbors -- each cut covers the other's blind
    # spot.
    pts = rgb[interior].astype(np.float64)
    if len(pts) < min_area:
        return []
    k = min(4, max(2, len(pts) // 4000 + 2))
    centroids, labels = kmeans2(pts, k, minit="++", seed=7)
    merged = list(range(k))
    for i in range(k):
        for j in range(i):
            if np.linalg.norm(centroids[i] - centroids[j]) < 55:
                merged[i] = merged[j]
                break
    class_img = np.full(mask.shape, -1, dtype=np.int32)
    class_img[interior] = [merged[l] for l in labels]

    # Absorb black decorative strokes into the surrounding color class so a
    # closed black loop reconnects the region it would otherwise split.
    absorb = black_dec & ~binary_dilation(grey_cut, iterations=1)
    if absorb.any():
        _d, (ay, ax_) = distance_transform_edt(class_img < 0, return_indices=True)
        class_img[absorb] = class_img[ay[absorb], ax_[absorb]]

    lab_img = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for cl in sorted(set(merged)):
        comps, n = cc_label(class_img == cl)
        for ci in range(1, n + 1):
            comp = comps == ci
            if comp.sum() < min_area:
                continue
            lab_img[comp] = next_id
            next_id += 1

    _dist, (iy, ix) = distance_transform_edt(lab_img == 0, return_indices=True)
    unassigned = support & (lab_img == 0)
    lab_img[unassigned] = lab_img[iy[unassigned], ix[unassigned]]

    pieces = []
    for ci in range(1, next_id):
        piece = lab_img == ci
        if piece.sum() < min_area:
            continue
        piece = binary_fill_holes(binary_opening(piece, iterations=1))
        fill_px = piece & interior
        sel = fill_px if int(fill_px.sum()) > 50 else piece
        color = rgb[sel].mean(axis=0) / 255.0
        pieces.append((piece, tuple(color)))
    return pieces


def palette_pieces(rgb: np.ndarray, mask: np.ndarray, *, tol: int = 60, min_area: int = 350) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    """Segment a hand-annotated panel whose pieces are filled with exact flat
    palette colors (one distinct color per piece; grey outlines and black
    decoration on top). Pixels are matched to the dominant exact colors, and
    stroke/anti-aliasing pixels are absorbed into the nearest piece."""
    from collections import Counter

    from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt

    counts = Counter(map(tuple, rgb[mask].tolist()))
    dominant = [np.asarray(c, dtype=np.float64) for c, n in counts.most_common(12) if n >= 2000]
    if not dominant:
        return []
    arr = rgb.astype(np.float64)
    lab_img = np.full(mask.shape, -1, dtype=np.int32)
    dist_best = np.full(mask.shape, float(tol))
    for ci, col in enumerate(dominant):
        d = np.linalg.norm(arr - col, axis=2)
        better = mask & (d < dist_best)
        lab_img[better] = ci
        dist_best[better] = d[better]

    # Absorb strokes and AA near the assembly into the nearest piece (same
    # inpainting as color_pieces).
    not_white = rgb.astype(np.int16).min(axis=2) < 200
    near = binary_dilation(mask, iterations=4)
    stroke = not_white & near & (lab_img < 0)
    dist, (iy, ix) = distance_transform_edt(lab_img < 0, return_indices=True)
    fill = stroke & (dist <= 8)
    lab_img[fill] = lab_img[iy[fill], ix[fill]]

    pieces = []
    for ci, col in enumerate(dominant):
        cmask = lab_img == ci
        if cmask.sum() < min_area:
            continue
        cmask = binary_fill_holes(binary_closing(binary_fill_holes(cmask), iterations=2))
        cmask = binary_opening(cmask, iterations=1)
        comps, n = cc_label(cmask)
        for k in range(1, n + 1):
            piece = comps == k
            if piece.sum() < min_area:
                continue
            pieces.append((piece, tuple(col / 255.0)))
    return pieces


def _crop_mask(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)
    return m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def free_rotation_mismatch(a: np.ndarray, b: np.ndarray, *, step: int = 3) -> float:
    """Min symmetric mismatch (% of larger area) between two masks over free
    rotation (step-degree sweep), flip, and all translations (FFT)."""
    from scipy.ndimage import rotate
    from scipy.signal import fftconvolve

    A = _crop_mask(a).astype(np.float32)
    B = _crop_mask(b).astype(np.float32)
    sa, sb = float(A.sum()), float(B.sum())
    best = np.inf
    for flip in (False, True):
        Bf = B[:, ::-1] if flip else B
        for ang in range(0, 360, step):
            R = rotate(Bf, ang, order=0, reshape=True).astype(np.float32)
            corr = fftconvolve(A, R[::-1, ::-1], mode="full")
            mis = sa + sb - 2.0 * float(corr.max())
            if mis < best:
                best = mis
    return 100.0 * best / max(sa, sb)


def reconcile_textured_pieces(sides: list[list[tuple[np.ndarray, tuple[float, float, float]]]],
                              *, merge_below: float = 12.0, min_px: int = 1500):
    """Merge decorative splits in textured panels using both assemblies.

    A textured piece can carry an internal color boundary or a grey texture
    line, which the per-assembly segmentation reads as a seam. The
    discriminator: texture travels rigidly with its piece, so for a
    decorative split the UNION of the two sub-regions is congruent across
    the two assemblies; for a real seam the two pieces are rearranged and
    the union is not. Adjacent-in-both pairs whose merged free-rotation
    mismatch falls below merge_below percent are merged (verified for
    usa_trump: the orange+blue textured piece merges at 9%, all real seams
    measure 19--69%)."""
    from scipy.ndimage import binary_dilation
    from scipy.optimize import linear_sum_assignment

    L = [(p, c) for p, c in sides[0] if int(p.sum()) >= min_px]
    R = [(p, c) for p, c in sides[1] if int(p.sum()) >= min_px]
    if not L or len(L) != len(R):
        return sides
    cost = np.zeros((len(L), len(R)))
    for i, (pl, cl) in enumerate(L):
        for j, (pr, cr) in enumerate(R):
            area = abs(int(pl.sum()) - int(pr.sum())) / max(int(pl.sum()), int(pr.sum()))
            col = float(np.linalg.norm(np.asarray(cl) - np.asarray(cr)))
            cost[i, j] = area + 0.5 * col
    ri, ci = linear_sum_assignment(cost)
    pairs = list(zip(ri.tolist(), ci.tolist()))

    def adjacent(a, b):
        return bool((binary_dilation(a, iterations=3) & b).any())

    candidates = []
    for (i1, j1), (i2, j2) in __import__("itertools").combinations(pairs, 2):
        if not (adjacent(L[i1][0], L[i2][0]) and adjacent(R[j1][0], R[j2][0])):
            continue
        mis = free_rotation_mismatch(L[i1][0] | L[i2][0], R[j1][0] | R[j2][0])
        print(f"  reconcile: merge candidate {int(L[i1][0].sum())}+{int(L[i2][0].sum())} px -> {mis:.1f}%")
        if mis < merge_below:
            candidates.append((mis, (i1, j1), (i2, j2)))

    merged_l: set[int] = set()
    merged_r: set[int] = set()
    out_l = {i: piece for i, piece in enumerate(L)}
    out_r = {j: piece for j, piece in enumerate(R)}
    for _mis, (i1, j1), (i2, j2) in sorted(candidates):
        if {i1, i2} & merged_l or {j1, j2} & merged_r:
            continue
        for out, a, b in ((out_l, i1, i2), (out_r, j1, j2)):
            pa, ca = out[a]
            pb, cb = out[b]
            color = ca if pa.sum() >= pb.sum() else cb
            out[a] = (pa | pb, color)
            del out[b]
        merged_l.update({i2})
        merged_r.update({j2})
    return [list(out_l.values()), list(out_r.values())]


def outer_contour(mask: np.ndarray) -> np.ndarray:
    cs = measure.find_contours(mask.astype(float), 0.5)
    longest = max(cs, key=len)
    return longest[:, ::-1]  # (x, y)


def simplify(poly: np.ndarray, tol: float = 1.3) -> np.ndarray:
    try:
        from shapely.geometry import Polygon

        p = Polygon(poly).simplify(tol, preserve_topology=True)
        return np.asarray(p.exterior.coords)
    except Exception:  # noqa: BLE001
        return poly


def umeyama2(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        s_fix[1, 1] = -1.0
    rot = u @ s_fix @ vt
    var = float(np.mean(np.sum(sc * sc, axis=1)))
    scale = float(np.trace(np.diag(d) @ s_fix) / max(var, 1e-12))
    t = mu_d - scale * rot @ mu_s
    return scale, rot, t


def register_input(input_mask: np.ndarray, assembly_mask: np.ndarray) -> np.ndarray:
    """Register the input silhouette contour onto an assembly.

    Scale is fixed by the area ratio (dissections preserve material), and
    the translation is the global optimum: the placement maximizing mask
    overlap, found exactly by FFT cross-correlation. No iterative fitting,
    so it cannot drift or collapse on strongly deviating assemblies, and it
    matches the best-case alignment used by the quantitative comparison.
    The published figures draw assemblies in the input orientation, so no
    rotation is fitted."""
    from scipy.signal import fftconvolve

    union = binary_fill_holes(assembly_mask)
    s = float(np.sqrt(union.sum() / max(input_mask.sum(), 1)))
    ys, xs = np.nonzero(input_mask)
    y0, x0 = int(ys.min()), int(xs.min())
    crop = input_mask[y0: ys.max() + 1, x0: xs.max() + 1]
    w = max(1, int(round(crop.shape[1] * s)))
    h = max(1, int(round(crop.shape[0] * s)))
    ref = np.asarray(
        Image.fromarray(np.where(crop, 255, 0).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST),
        dtype=bool,
    )
    corr = fftconvolve(union.astype(np.float32), ref[::-1, ::-1].astype(np.float32), mode="full")
    k, l = np.unravel_index(int(np.argmax(corr)), corr.shape)
    ty = k - ref.shape[0] + 1
    tx = l - ref.shape[1] + 1
    ic = outer_contour(input_mask)  # (x, y)
    return (ic - np.asarray([x0, y0])) * s + np.asarray([tx, ty])


def redraw_panel(rgb: np.ndarray, inputs: list[np.ndarray], out_path: Path,
                 *, seg: str = "color") -> None:
    """inputs: input masks in panel coordinates, one per assembly (A, B).
    seg: "color" clusters fills (pieces have unique colors); "lines" splits
    on the grey seam strokes (adjacent pieces may share a fill color)."""
    from scipy.ndimage import binary_dilation

    if seg == "palette":
        # Hand-annotated panel: any saturated bright fill is a piece pixel
        # (no inputs/red outlines present, and piece_pixel_mask's dark-red
        # filter would eat a pure-red palette piece).
        segment = palette_pieces
        arr = rgb.astype(np.int16)
        pmask = ((arr.max(axis=2) - arr.min(axis=2)) > 45) & (arr.max(axis=2) > 90)
    else:
        segment = line_pieces if seg == "lines" else color_pieces
        pmask = piece_pixel_mask(rgb)
        # Remove anti-aliasing specks around the input shapes' red outlines.
        pmask &= ~binary_dilation(mk.input_color_mask(rgb), iterations=4)
    left, right = split_assemblies(pmask)
    assemblies = [left, right]
    per_asm = [segment(rgb, amask) for amask in assemblies]
    if seg == "lines" and len(per_asm) == 2:
        per_asm = reconcile_textured_pieces(per_asm)
    all_pts = []
    outlines = []
    piece_polys = []
    for amask, imask, pieces in zip(assemblies, inputs, per_asm):
        total = int(amask.sum())
        if imask is not None and imask.sum() > 0 and total > 0:
            outline = register_input(imask, amask)
            outlines.append(outline)
            all_pts.append(outline)
        for piece, color in pieces:
            # Drop speck artifacts (anti-aliasing remnants near inputs/text).
            if piece.sum() < 0.015 * total:
                continue
            poly = simplify(outer_contour(piece))
            piece_polys.append((poly, color))
            all_pts.append(poly)

    pts = np.vstack(all_pts)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    pad = 0.03 * float((mx - mn).max())
    fig_h = 4.2
    w = (mx[0] - mn[0] + 2 * pad) / (mx[1] - mn[1] + 2 * pad)
    fig, ax = plt.subplots(figsize=(w * fig_h, fig_h), dpi=230)
    ax.set_xlim(mn[0] - pad, mx[0] + pad)
    ax.set_ylim(mx[1] + pad, mn[1] - pad)
    ax.set_aspect("equal")
    ax.axis("off")
    for outline in outlines:
        ax.add_patch(MplPolygon(outline, closed=True, facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.4, zorder=1))
    for poly, color in piece_polys:
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=color, alpha=0.75, edgecolor="#111111", linewidth=0.9, zorder=2))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(out_path.name)


def input_masks_from_panel(rgb: np.ndarray, axis: str, slots: tuple[int, int]):
    comps = mk.pair_shapes(rgb, axis)
    return comps[slots[0]], comps[slots[1]]


def main() -> None:
    doc = fitz.open(mk.PDF)
    qa_rows = []

    for pair, (page, box, order) in FIG20_PANELS.items():
        rgb = mk.render_box(doc, page, box)
        crop_key = f"duncan_{pair}"
        _p, _b, axis, slots, _names = mk.PAIR_CROPS[crop_key]
        a_mask, b_mask = input_masks_from_panel(rgb, axis, slots)
        inputs = [a_mask, b_mask] if order == "AB" else [b_mask, a_mask]
        out = FIGS / f"duncanredraw_{pair}.png"
        if pair == "usa_trump":
            # Textured panel: fill colors do not identify pieces. Segment a
            # hand-annotated copy with one flat color per piece (verified
            # against the published panel by cross-assembly free-rotation
            # congruence: phantom texture splits merge at ~10%, real seams
            # measure 19--69%).
            user = HERE / "color_trump_usa.png"  # placed next to the paper PDF
            rgb_seg = np.asarray(Image.open(user).convert("RGB"))
            redraw_panel(rgb_seg, inputs, out, seg="palette")
            qa_rows.append((pair, rgb_seg, out))
            continue
        redraw_panel(rgb, inputs, out, seg="color")
        qa_rows.append((pair, rgb, out))

    # Fig. 22: assembly1 = bunny arrangement, assembly2 = egg arrangement.
    egg = mk.shape_mask(mk.render_box(doc, *mk.SINGLE_CROPS["egg"]))
    bunny = mk.shape_mask(mk.render_box(doc, *mk.SINGLE_CROPS["bunny"]))

    def lift(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        h, w = mask.shape
        out[:h, :w] = mask[: shape[0], : shape[1]]
        return out

    for name, (page, box) in FIG22_PANELS.items():
        rgb = mk.render_box(doc, page, box)
        shape = rgb.shape[:2]
        out = FIGS / f"duncanredraw_{name}.png"
        redraw_panel(rgb, [lift(bunny, shape), lift(egg, shape)], out)
        qa_rows.append((name, rgb, out))
    doc.close()

    # QA sheet: original panel next to redraw.
    tiles = []
    for name, rgb, out in qa_rows:
        orig = Image.fromarray(rgb)
        red = Image.open(out)
        h = 240
        o = orig.resize((int(orig.width * h / orig.height), h))
        r = red.resize((int(red.width * h / red.height), h))
        tile = Image.new("RGB", (o.width + r.width + 30, h + 30), "white")
        tile.paste(o, (5, 25))
        tile.paste(r, (o.width + 20, 25))
        from PIL import ImageDraw

        ImageDraw.Draw(tile).text((5, 5), f"{name}  [original | redraw]", fill=(0, 0, 0))
        tiles.append(tile)
    w = max(t.width for t in tiles)
    sheet = Image.new("RGB", (w, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(FIGS / "qa_duncan_redraws.png")
    print("qa_duncan_redraws.png")


if __name__ == "__main__":
    main()
