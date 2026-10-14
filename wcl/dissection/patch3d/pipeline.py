"""Command-line pipeline (exp. 453 in the dissertation): labels + meshes -> shared patch complex -> certificate."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsqr

from .alignment import load_aligned_projector, refine_projector_alignment
from .certificate import piece_watertight_stats, signed_volume, tri_area_sum, validation_certificate
from .closure import build_transform_clusters, merge_relations, refine_transform_sweep, transform_refinement_diag
from .complex import build_complex, classify_vertices, decompose_patches, fwd, inv, pair_b_contacts
from .intersections import b_coincidence_groups, count_self_intersections, find_cross_intersections, find_self_intersections, penetration_springs, piece_vertex_ids, untangle_pass
from .io import write_tagged_obj
from .system import System, piece_triangles, vertex_positions
from .util import rms, write_csv
from .voxel import decode_label_stack, geometry_match_labels_3d, write_label_stack, write_obj

ROOT = Path.cwd()
RESULTS = ROOT / "outputs" / "zhou3d_shared_patch_complex_baco"
EYE3 = np.eye(3, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Shared patch complex 3D BACO: non-manifold B-rep of the Zhou partition with dual "
            "(A-role, B-role) classified patches, structural contact closure, seam-aware fairing, "
            "exterior pinning to the original input meshes, and a winding-number certificate."
        )
    )
    p.add_argument("--labels-a", required=True)
    p.add_argument("--labels-b", required=True)
    p.add_argument("--source-mesh", required=True)
    p.add_argument("--target-mesh", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="exp453_shared_patch_complex")
    p.add_argument("--outer-iters", type=int, default=4)
    p.add_argument("--align-iters", type=int, default=15)
    p.add_argument("--w-exterior-a", type=float, default=4.0)
    p.add_argument("--w-exterior-b", type=float, default=2.5)
    p.add_argument("--w-closure", type=float, default=25.0)
    p.add_argument("--w-fair", type=float, default=1.0)
    p.add_argument("--w-seam", type=float, default=1.0)
    p.add_argument("--w-anchor", type=float, default=0.12)
    p.add_argument("--w-nopen", type=float, default=0.0, help="One-sided non-penetration springs between every piece pair in both assemblies (0 disables; the dissertation's runs used 0). Vertices strictly inside another piece are pulled toward the outside of its surface; targets refresh every outer iteration.")
    p.add_argument("--nopen-clearance", type=float, default=0.05, help="Push-out distance beyond the penetrated surface (voxel units).")
    p.add_argument("--nopen-min-depth", type=float, default=0.02, help="Ignore penetrations shallower than this (guards machine-exact shared-contact vertices, which sit at depth ~0).")
    p.add_argument("--nopen-max-step", type=float, default=0.75, help="Cap on the outward pull per outer iteration (voxel units); deep penetrations walk out over several rounds instead of folding the surface in one solve.")
    p.add_argument("--nopen-start-iter", type=int, default=2, help="First outer iteration that adds penetration springs (the initial blend geometry gives unreliable winding numbers).")
    p.add_argument("--nopen-exempt-a-pinned", action=argparse.BooleanOptionalAction, default=True, help="Do not push A-exterior-pinned vertices in assembly B: those pushes dent the source surface (same template variable), trading A fidelity for B overlap.")
    p.add_argument("--lsqr-tol", type=float, default=1e-8)
    p.add_argument("--lsqr-iter", type=int, default=1500)
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refine-transforms", action=argparse.BooleanOptionalAction, default=False, help="Sub-voxel rigid refinement of the piece transforms (per persistent-contact cluster) alternated with the vertex solve.")
    p.add_argument("--cert-slack", type=float, default=0.01, help="Pass threshold on certificate fractions (uncovered/spill/overlap) per assembly.")
    p.add_argument("--untangle-iters", type=int, default=12, help="Post-solve local relaxation rounds to remove residual self- and cross-piece intersections (root-space, closure-preserving).")
    p.add_argument("--cross-untangle", action=argparse.BooleanOptionalAction, default=True, help="Also detect and relax cross-piece foldovers in both assemblies (FIX: previously only per-piece self-intersections were checked).")
    p.add_argument("--align-scale", action=argparse.BooleanOptionalAction, default=False, help="Allow uniform scale in the mesh-to-grid ICP. Default OFF (FIX): rigid alignment, so the certificate compares against the actual input meshes.")
    p.add_argument("--cert-dilate", type=int, default=2, help="Dilation radius (voxels) of the certificate query grid (was hardwired 1).")
    args = p.parse_args()

    start = time.perf_counter()
    out_dir = Path(args.out) if args.out else RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    la = decode_label_stack(Path(args.labels_a))
    lb_raw = decode_label_stack(Path(args.labels_b))
    lb, transforms, match_diag = geometry_match_labels_3d(la, lb_raw)
    write_label_stack(out_dir / "matched_labels_A", la)
    write_label_stack(out_dir / "matched_labels_B", lb)

    source_proj = load_aligned_projector(Path(args.source_mesh))
    target_proj = load_aligned_projector(Path(args.target_mesh))
    source_proj, align_a_diag = refine_projector_alignment(source_proj, la > 0, args.align_iters, allow_scale=args.align_scale)
    target_proj, align_b_diag = refine_projector_alignment(target_proj, lb > 0, args.align_iters, allow_scale=args.align_scale)
    write_obj(out_dir / "source_mesh_refined.obj", {1: (source_proj.vertices_zyx, source_proj.faces)})
    write_obj(out_dir / "target_mesh_refined.obj", {1: (target_proj.vertices_zyx, target_proj.faces)})

    cx = build_complex(la, lb, transforms)
    relations, paired_sides, unpaired_sides = pair_b_contacts(cx)
    uf, soft_relations, merged_count, trivial_count = merge_relations(cx, relations)
    patch, patch_tags, edge_faces = decompose_patches(cx)
    cls, umbrellas, seams = classify_vertices(cx, patch, edge_faces)
    clusters = build_transform_clusters(cx, relations)

    def rebuild_merge() -> tuple:
        u, s, m, t = merge_relations(cx, relations)
        vm = [u.find(v) for v in range(len(cx.vert_keys))]
        rt = sorted({r for r, _m, _t in vm})
        return u, s, m, t, vm, {r: i for i, r in enumerate(rt)}

    vmap = [uf.find(v) for v in range(len(cx.vert_keys))]
    roots = sorted({r for r, _m, _t in vmap})
    root_col = {r: i for i, r in enumerate(roots)}

    # Exterior pin sets.
    ea_verts: set[int] = set()
    eb_pairs: set[tuple[int, int]] = set()
    for s in cx.sides:
        corners = cx.faces[s.face].corners
        if s.a_role == 0:
            ea_verts.update(corners)
        if s.b_role == 0 and s.own_ok:
            eb_pairs.update((v, s.label) for v in corners)
    ea_list = sorted(ea_verts)
    eb_list = sorted(eb_pairs)
    eb_by_vert: dict[int, list[int]] = {}
    for v, lab in eb_list:
        eb_by_vert.setdefault(v, []).append(lab)

    piece_tris_all = {label: piece_triangles(cx, label)[0] for label in cx.labels}
    piece_vids = piece_vertex_ids(cx)
    nopen_a: list[tuple[int, int, np.ndarray, float]] = []
    nopen_b: list[tuple[int, int, np.ndarray, float]] = []

    # Initialization: pull exterior vertices onto the input surfaces.
    cur = cx.vert_pos.copy()
    last_x = None
    solve_diag: dict[str, float] = {}
    for it in range(max(0, int(args.outer_iters)) + 1):
        # Projection targets from current geometry.
        targets_a = {}
        if ea_list:
            idx = np.asarray(ea_list, dtype=np.int64)
            proj_pts = source_proj.project(cur[idx])
            targets_a = {v: proj_pts[i] for i, v in enumerate(ea_list)}
        targets_b = {}
        if eb_list:
            pts = np.asarray([fwd(cx, lab, cur[v][None, :])[0] for v, lab in eb_list])
            proj_pts = target_proj.project(pts)
            targets_b = {pair: proj_pts[i] for i, pair in enumerate(eb_list)}
        if it == 0:
            # Blend the raw lattice positions toward their pins, then solve.
            for v, t in targets_a.items():
                cur[v] = t
            accum: dict[int, list[np.ndarray]] = {}
            for (v, lab), tb in targets_b.items():
                accum.setdefault(v, []).append(inv(cx, lab, tb[None, :])[0])
            for v, pulls in accum.items():
                pulled = np.mean(np.asarray(pulls), axis=0)
                cur[v] = 0.5 * (cur[v] + pulled) if v in targets_a else pulled
            init_positions = cur.copy()
            continue

        if args.w_nopen > 0 and it >= args.nopen_start_iter:
            b_exempt = ea_verts if args.nopen_exempt_a_pinned else None
            nopen_a = penetration_springs(cx, cur, piece_vids, piece_tris_all, False, args.nopen_min_depth, args.nopen_clearance, args.nopen_max_step)
            nopen_b = penetration_springs(cx, cur, piece_vids, piece_tris_all, True, args.nopen_min_depth, args.nopen_clearance, args.nopen_max_step, b_exempt)

        sysm = System(vmap, root_col)
        for v, t in targets_a.items():
            sysm.add([(v, EYE3)], t, args.w_exterior_a)
        for (v, lab), tb in targets_b.items():
            sysm.add([(v, cx.r_mats[lab])], tb - cx.r_offs[lab], args.w_exterior_b)
        for v, _lab, tgt, _d in nopen_a:
            sysm.add([(v, EYE3)], tgt, args.w_nopen)
        for v, lab, tgt, _d in nopen_b:
            sysm.add([(v, cx.r_mats[lab])], tgt - cx.r_offs[lab], args.w_nopen)
        # Only relations that could NOT be merged structurally need residual rows.
        for v, li, w, lk in soft_relations:
            sysm.add(
                [(v, cx.r_mats[li]), (w, -cx.r_mats[lk])],
                cx.r_offs[lk] - cx.r_offs[li],
                args.w_closure,
            )
        for v in range(len(cx.vert_keys)):
            if cls[v] == 0 and len(umbrellas[v]) > 0:
                n = float(len(umbrellas[v]))
                terms = [(v, EYE3)] + [(int(u), -EYE3 / n) for u in umbrellas[v]]
                sysm.add(terms, np.zeros(3), args.w_fair)
            elif cls[v] == 1:
                a, b = seams[v]
                sysm.add([(v, EYE3), (a, -0.5 * EYE3), (b, -0.5 * EYE3)], np.zeros(3), args.w_seam)
            sysm.add([(v, EYE3)], init_positions[v], args.w_anchor)

        x, solve_diag = sysm.solve(args.lsqr_tol, args.lsqr_iter)
        cur = vertex_positions(vmap, root_col, x)
        last_x = x

        if args.refine_transforms and it < max(0, int(args.outer_iters)):
            refine_transform_sweep(
                cx, cur, relations, clusters, eb_by_vert, target_proj, args.w_exterior_b, args.w_closure
            )
            uf, soft_relations, merged_count, trivial_count, vmap, root_col = rebuild_merge()

    if last_x is None:
        # --outer-iters 0: no linear solve ran. Build a root vector from the
        # initialized positions so the untangle pass still has variables.
        last_x = np.zeros(3 * len(root_col), dtype=np.float64)
        for root, col in root_col.items():
            last_x[col * 3 : col * 3 + 3] = cur[root]
        cur = vertex_positions(vmap, root_col, last_x)

    # B-coincidence classes: which (label, vertex) occurrences are the same
    # geometric point in assembly B (needed to distinguish legitimate contact
    # from interpenetration in the cross-piece checks).
    coincidence = b_coincidence_groups(cx, relations)

    def intersection_snapshot() -> dict[str, int]:
        n_self = sum(len(find_self_intersections(cur, t)) for t in piece_tris_all.values())
        n_ca = n_cb = 0
        if args.cross_untangle:
            n_ca = len(find_cross_intersections(cx, cur, piece_tris_all, transformed=False))
            n_cb = len(
                find_cross_intersections(
                    cx, cur, piece_tris_all, transformed=True, coincidence=coincidence
                )
            )
        return {"self": n_self, "cross_a": n_ca, "cross_b": n_cb}

    before = intersection_snapshot()
    after = dict(before)
    if args.untangle_iters > 0:
        last_x, cur, after = untangle_pass(
            cx,
            cur,
            last_x,
            vmap,
            root_col,
            umbrellas,
            args.untangle_iters,
            coincidence=coincidence,
            check_cross=bool(args.cross_untangle),
        )
    untangled_resolved = float(
        (before["self"] + before["cross_a"] + before["cross_b"])
        - (after["self"] + after["cross_a"] + after["cross_b"])
    )

    # Metrics.
    metrics: dict[str, float] = {
        "untangle_resolved": untangled_resolved,
        "nopen_springs_a": float(len(nopen_a)),
        "nopen_springs_b": float(len(nopen_b)),
        "self_intersections_before_untangle": float(before["self"]),
        "cross_intersections_A_before_untangle": float(before["cross_a"]),
        "cross_intersections_B_before_untangle": float(before["cross_b"]),
        "cross_intersections_A_total": float(after["cross_a"]),
        "cross_intersections_B_total": float(after["cross_b"]),
    }
    if ea_list:
        idx = np.asarray(ea_list, dtype=np.int64)
        d = np.linalg.norm(source_proj.project(cur[idx]) - cur[idx], axis=1)
        metrics["exterior_a_rms"] = rms(d)
        metrics["exterior_a_max"] = float(d.max())
        metrics["exterior_a_vertices"] = float(len(idx))
    if eb_list:
        pts = np.asarray([fwd(cx, lab, cur[v][None, :])[0] for v, lab in eb_list])
        d = np.linalg.norm(target_proj.project(pts) - pts, axis=1)
        metrics["exterior_b_rms"] = rms(d)
        metrics["exterior_b_max"] = float(d.max())
        metrics["exterior_b_vertices"] = float(len(eb_list))
    closure_res = []
    for v, li, w, lk in relations:
        pa = fwd(cx, li, cur[v][None, :])[0]
        pb = fwd(cx, lk, cur[w][None, :])[0]
        closure_res.append(float(np.linalg.norm(pa - pb)))
    closure_res = np.asarray(closure_res)
    metrics["b_closure_relations"] = float(len(relations))
    metrics["b_closure_merged"] = float(merged_count)
    metrics["b_closure_trivial"] = float(trivial_count)
    metrics["b_closure_soft"] = float(len(soft_relations))
    metrics["b_closure_rms"] = rms(closure_res)
    metrics["b_closure_max"] = float(closure_res.max()) if len(closure_res) else 0.0
    # Exterior-in-both conflict: same template point pinned to both inputs.
    extext = [v for v in ea_verts if v in eb_by_vert]
    if extext:
        idx = np.asarray(sorted(extext), dtype=np.int64)
        pa = source_proj.project(cur[idx])
        gaps = []
        for i, v in enumerate(idx.tolist()):
            for lab in eb_by_vert[v]:
                tb = fwd(cx, lab, cur[v][None, :])
                pb = inv(cx, lab, target_proj.project(tb))[0]
                gaps.append(float(np.linalg.norm(pa[i] - pb)))
        metrics["extext_vertices"] = float(len(extext))
        metrics["extext_gap_rms"] = rms(np.asarray(gaps))
        metrics["extext_gap_max"] = float(max(gaps))
    conflict_sides = sum(1 for s in cx.sides if not s.own_ok)
    binterior_sides = sum(1 for s in cx.sides if s.b_role == s.label)
    metrics.update(
        {
            "faces": float(len(cx.faces)),
            "sides": float(len(cx.sides)),
            "vertices": float(len(cx.vert_keys)),
            "variables_after_merge": float(3 * len(roots)),
            "patches": float(len(patch_tags)),
            "sides_paired_b": float(paired_sides),
            "sides_unpaired_b": float(unpaired_sides),
            "sides_conflict_own": float(conflict_sides),
            "sides_b_interior": float(binterior_sides),
            "verts_interior": float(int(np.sum(cls == 0))),
            "verts_seam": float(int(np.sum(cls == 1))),
            "verts_corner": float(int(np.sum(cls == 2))),
        }
    )

    # Per-piece watertightness and self-intersections (rigid-invariant, so the
    # template check covers both assemblies).
    piece_rows = []
    for label in cx.labels:
        stats = piece_watertight_stats(cx, label)
        tris, _ = piece_triangles(cx, label)
        piece_rows.append(
            {
                "label": label,
                "faces": int(np.sum([1 for s in cx.sides if s.label == label])),
                "area_final": tri_area_sum(cur, tris),
                "signed_volume": signed_volume(cur, tris),
                "self_intersections": count_self_intersections(cur, tris),
                "zhou_mismatch": transforms[label].mismatch,
                **stats,
            }
        )
    metrics["open_edges_total"] = float(sum(r["open_edges"] for r in piece_rows))
    metrics["self_intersections_total"] = float(sum(r["self_intersections"] for r in piece_rows))
    metrics["transform_clusters"] = float(len(clusters))
    metrics.update(transform_refinement_diag(cx))

    # Per-patch reporting: localizes where the discrete solution is wrong, the
    # continuous analogue of Zhou's mismatch-cluster signal.
    dist_a: dict[int, float] = {}
    if ea_list:
        idx = np.asarray(ea_list, dtype=np.int64)
        d = np.linalg.norm(source_proj.project(cur[idx]) - cur[idx], axis=1)
        dist_a = {v: float(d[i]) for i, v in enumerate(ea_list)}
    dist_b: dict[tuple[int, int], float] = {}
    if eb_list:
        pts = np.asarray([fwd(cx, lab, cur[v][None, :])[0] for v, lab in eb_list])
        d = np.linalg.norm(target_proj.project(pts) - pts, axis=1)
        dist_b = {pair: float(d[i]) for i, pair in enumerate(eb_list)}

    patch_faces: dict[int, list[int]] = {}
    for fi in range(len(cx.faces)):
        patch_faces.setdefault(int(patch[fi]), []).append(fi)
    patch_rows = []
    for pid, tag in sorted(patch_tags.items()):
        ln, lp, sl_n, br_n, ok_n, sl_p, br_p, ok_p = tag
        fis = patch_faces.get(pid, [])
        verts_p = sorted({v for fi in fis for v in cx.faces[fi].corners})
        tris = np.asarray(
            [(c[0], c[1], c[2]) for fi in fis for c in (cx.faces[fi].corners,)]
            + [(c[0], c[2], c[3]) for fi in fis for c in (cx.faces[fi].corners,)],
            dtype=np.int64,
        )
        row: dict[str, object] = {
            "patch": pid,
            "label_neg": ln,
            "label_pos": lp,
            "b_role_neg_side": br_n,
            "b_role_pos_side": br_p,
            "own_ok_neg": ok_n,
            "own_ok_pos": ok_p,
            "faces": len(fis),
            "area_final": tri_area_sum(cur, tris) if len(tris) else 0.0,
            "is_conflict": int(
                (not ok_n and sl_n >= 0)
                or (not ok_p and sl_p >= 0)
                or (sl_n > 0 and br_n == sl_n)
                or (sl_p > 0 and br_p == sl_p)
            ),
        }
        da = [dist_a[v] for v in verts_p if v in dist_a]
        if da:
            row["exterior_a_mean"] = float(np.mean(da))
            row["exterior_a_max"] = float(np.max(da))
        db = []
        for v in verts_p:
            for lab in (sl_n, sl_p):
                if lab > 0 and (v, lab) in dist_b:
                    db.append(dist_b[(v, lab)])
        if db:
            row["exterior_b_mean"] = float(np.mean(db))
            row["exterior_b_max"] = float(np.max(db))
        if da and db:
            row["extext_patch"] = 1
        patch_rows.append(row)

    write_tagged_obj(out_dir / "pieces_template.obj", cx, cur, transformed=False)
    write_tagged_obj(out_dir / "assembly_A.obj", cx, cur, transformed=False)
    write_tagged_obj(out_dir / "assembly_B.obj", cx, cur, transformed=True)
    write_tagged_obj(out_dir / "initial_template.obj", cx, init_positions, transformed=False)
    write_csv(out_dir / "pieces.csv", piece_rows)
    write_csv(out_dir / "patches.csv", patch_rows)

    if args.validate:
        cert, overlap_pair_rows = validation_certificate(
            cx, cur, la, lb, source_proj, target_proj, dilate=args.cert_dilate
        )
        metrics.update(cert)
        write_csv(out_dir / "overlap_pairs.csv", overlap_pair_rows or [{"assembly": "", "label_i": "", "label_k": "", "overlap_voxels": 0, "overlap_fraction_of_shape": 0.0}])
        # Audit trail: discrete mismatch -> certificate failure. M counts
        # uncovered cells on both sides, so M/2 is the rigid-copy lower bound
        # on per-assembly uncoverage; conflict-patch area localizes it.
        m_total = float(match_diag["total_mismatch"])
        metrics["audit_mismatch_total"] = m_total
        metrics["audit_mismatch_half"] = 0.5 * m_total
        metrics["audit_conflict_patch_area"] = float(
            sum(float(r["area_final"]) for r in patch_rows if r["is_conflict"])
        )
        metrics["audit_patch_area_total"] = float(sum(float(r["area_final"]) for r in patch_rows))
        slack = float(args.cert_slack)
        # "clean" requires zero cross-piece intersections in the source
        # assembly and no inside-out pieces. Cross-B intersections are not part
        # of it: conflict regions do interpenetrate in B and are already charged
        # to the B overlap fraction; their count is reported.
        clean = (
            metrics["self_intersections_total"] == 0.0
            and metrics["open_edges_total"] == 0.0
            and metrics.get("cross_intersections_A_total", 0.0) == 0.0
            and metrics.get("pieces_orientation_flipped", 0.0) == 0.0
        )
        for name in ("A", "B"):
            ok = (
                metrics.get(f"certificate_{name}_uncovered_fraction", 1.0) <= slack
                and metrics.get(f"certificate_{name}_spill_fraction", 1.0) <= slack
                and metrics.get(f"certificate_{name}_overlap_fraction", 1.0) <= slack
            )
            metrics[f"certificate_{name}_pass"] = float(ok and clean)
        metrics["certificate_pass"] = float(
            metrics.get("certificate_A_pass", 0.0) > 0 and metrics.get("certificate_B_pass", 0.0) > 0
        )

    summary = {
        "runtime_s": time.perf_counter() - start,
        "labels": float(len(cx.labels)),
        **{f"source_{k}": v for k, v in align_a_diag.items()},
        **{f"target_{k}": v for k, v in align_b_diag.items()},
        **metrics,
        **solve_diag,
    }
    write_csv(out_dir / "summary.csv", [summary])
    payload = {
        "out_dir": str(out_dir),
        "labels_a": str(Path(args.labels_a)),
        "labels_b": str(Path(args.labels_b)),
        "source_mesh": str(Path(args.source_mesh)),
        "target_mesh": str(Path(args.target_mesh)),
        "matching": match_diag,
        "summary": summary,
        "args": vars(args),
        "notes": (
            "Shared patch complex (non-manifold B-rep) common-boundary optimizer. The complex is built once in the "
            "template frame by multi-material surface extraction on the matched Zhou label grid: every contact face "
            "between two pieces is one shared 2-cell, so A-assembly closure is structural. Each face side is dually "
            "classified by its role in B (push through the piece's lattice transform and read the B grid); because "
            "the transforms are exact lattice maps, B-assembly contacts pair into an exact face- and vertex-level "
            "bijection between the two incident pieces, which is merged into shared variables via an affine "
            "union-find (structural B closure; inconsistent relations fall back to strong soft residuals and are "
            "counted). Patches are connected components of the dual (A-role, B-role) classification, seams are the "
            "tag-change curves (1D faired), patch interiors are 2D faired, exterior-A/B patches are pinned to the "
            "original watertight input meshes by iterated closest-point projection, and (exterior,exterior) patches "
            "report their irreconcilable gap as a Zhou feedback signal. The harmonic cross-parameterization step of "
            "the design document is unnecessary here: the lattice itself supplies the material-point bijection. "
            "Validation is a winding-number certificate (piece overlap, coverage, spill vs the input meshes in both "
            "assemblies) plus per-piece watertightness of the extracted closed surfaces. "
            "FIXES (this revision): (1) certificate fractions are normalized by shape volume, matching the stated "
            "definition ('fraction of the input solid'); legacy query-normalized values kept as "
            "*_fraction_of_queries; (2) signed winding numbers with an explicit orientation convention replace "
            "abs(), and inside-out pieces are flagged (pieces_orientation_flipped) and fail 'clean'; (3) mesh-to-"
            "grid ICP is rigid by default (--align-scale re-enables similarity) and the applied scale is reported; "
            "(4) cross-piece intersections are now detected in BOTH assemblies (index-sharing exclusion in A; "
            "B-coincidence classes from the closure relations in B) and relaxed by the untangle pass in root space; "
            "cross-A intersections and orientation flips are part of the pass criterion; (5) per-pair overlap "
            "voxels are exported to overlap_pairs.csv as a discrete-stage feedback signal; (6) certificate query "
            "dilation is configurable (--cert-dilate, default 2); (7) --outer-iters 0 no longer crashes."
        ),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
