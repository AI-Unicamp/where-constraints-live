r"""Continuous relaxation of multi-target Zhou: piecewise-rigid area transport.

The continuous problem Zhou quantizes: find regions O_1..O_k partitioning
shape 0 and transforms T_{i,t} in SE(2) such that {T_{i,t}(O_i)} partitions
shape t for every t, minimizing total cut length. This script solves the
"relax Zhou's output" instantiation:

  * regions  -> soft indicator fields u = softmax(z / tau) on the frame-0
    grid, masked to silhouette 0 (so assembly 0 is watertight by
    construction; this subsumes the frame-0 seam weld, and turns the
    uncovered-minus-outside invariant into uncovered == outside);
  * lattice transforms -> the discrete Zhou isometry composed with a free
    SE(2) delta (theta, dx, dy) about the mapped piece centroid (flip stays
    inside the discrete affine, det = -1);
  * mismatch count -> differentiable coverage integral per assembly t >= 1:
    mean[ m_t (c_t - 1)^2 + (1 - m_t) c_t^2 ],  c_t(y) = sum_i u_i(T_it^-1 y)
    (the second term is the spill/outside penalty);
  * plus a Modica-Mortola term  eps |grad u|^2 + u^2(1-u)^2 / eps  that
    Gamma-converges to cut length (Zhou has no such prior; it drives short,
    smooth seams) and a min-area hinge that stops pieces from being deleted
    (vanishing a piece is otherwise a legal "fewer pieces" solution).

Optimization: Adam on logits + poses jointly, annealing tau so the fields
harden. Initialized from a finished zhou_multi level (same data the exp462
BACO consumes). Hardened argmax labels are evaluated with the same
uncovered / outside / overlap percentages exp462 reports, before (pure Zhou)
and after, so the lattice-artifact share of the residual is measured
directly.

Run on a finished search level (from dissection/):
  python experiments\exp463_zhou_continuous_relax_2d.py ^
      --tag exp462_bear_cat_fish_cycle_area --names bear,cat,fish --pieces 9
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import box
from shapely.ops import unary_union

from lattice2d import (PixelIsometry, _label_color,
                                             _pixel_forward_xy,
                                             geometry_match_labels)
from exp462_zhou_multitarget_baco_2d import (PALETTE, RESULTS,
                                             assembly_metrics_geoms,
                                             decode_label_png,
                                             label_files_for_pieces,
                                             latest_label_files, load_fg)


def iso_affine(iso: PixelIsometry) -> tuple[np.ndarray, np.ndarray]:
    """Exact 2x2 + offset of the discrete pixel isometry (it is affine in
    (x, y); any half-pixel convention offset is absorbed by the learned
    translation delta)."""
    px = np.array([0.0, 1.0, 0.0])
    py = np.array([0.0, 0.0, 1.0])
    qx, qy = _pixel_forward_xy(iso, px, py)
    b = np.array([qx[0], qy[0]], dtype=np.float64)
    A = np.array([[qx[1] - qx[0], qx[2] - qx[0]],
                  [qy[1] - qy[0], qy[2] - qy[0]]], dtype=np.float64)
    return A, b


def build_data(args):
    out_dir = RESULTS / args.tag
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    masks_dir, zhou_dir = out_dir / "masks", out_dir / "zhou"
    fg = [load_fg(masks_dir / f"{n}.png") for n in names]

    n_shapes = len(names)
    if args.pieces > 0:
        files = label_files_for_pieces(zhou_dir, args.pieces, n_shapes)
        if files is None:
            raise FileNotFoundError(f"no level files for K={args.pieces} in {zhou_dir}")
        k = args.pieces
    else:
        found = latest_label_files(zhou_dir, n_shapes)
        if found is None:
            raise FileNotFoundError(f"no Zhou level outputs in {zhou_dir}")
        k, files = found

    grids_raw = [decode_label_png(f) for f in files]
    grids = [grids_raw[0]]
    isos_list: list[dict[int, PixelIsometry] | None] = [None]
    common = set(int(v) for v in np.unique(grids_raw[0])) - {0}
    for t in range(1, n_shapes):
        lt, isos, _ = geometry_match_labels(grids_raw[0], grids_raw[t])
        grids.append(lt)
        isos_list.append(isos)
        common &= set(int(v) for v in np.unique(lt)) - {0}
        common &= set(isos)
    labels = sorted(common)
    print(f"data: K={k}, {len(labels)} matched pieces, {n_shapes} assemblies")
    return out_dir, names, k, labels, grids, isos_list, fg


class Relax:
    """Phase fields + SE(2) pose deltas; all torch, CPU.

    The map of piece i into assembly t is  y = R(theta)(A p + b - c) + c + d
    with (A, b, c) constant per (t, i) -- for the Zhou-initialized variant A/b
    are the discrete isometry and theta/d are deltas; a from-scratch search
    (exp464) passes its own affines and absolute poses."""

    def __init__(self, z0: np.ndarray, aff_np, fg,
                 theta0: np.ndarray | None = None,
                 trans0: np.ndarray | None = None):
        self.k = z0.shape[0]
        self.T = len(fg)
        self.H, self.W = fg[0].shape
        self.masks = [torch.from_numpy(m.astype(np.float32)) for m in fg]
        self.mask0 = self.masks[0]
        self.z = torch.tensor(z0, dtype=torch.float32, requires_grad=True)

        self.Ainv, self.b, self.c = [], [], []
        self.aff_np = aff_np  # list over t>=1 of list over i of (A, b, c)
        for aff_t in aff_np:
            self.Ainv.append(torch.tensor(
                np.stack([np.linalg.inv(A) for A, _, _ in aff_t]),
                dtype=torch.float32))
            self.b.append(torch.tensor(np.stack([b for _, b, _ in aff_t]),
                                       dtype=torch.float32))
            self.c.append(torch.tensor(np.stack([c for _, _, c in aff_t]),
                                       dtype=torch.float32))
        th0 = np.zeros((self.T - 1, self.k)) if theta0 is None else theta0
        tr0 = np.zeros((self.T - 1, self.k, 2)) if trans0 is None else trans0
        self.theta = torch.tensor(th0, dtype=torch.float32, requires_grad=True)
        self.trans = torch.tensor(tr0, dtype=torch.float32, requires_grad=True)

        yy, xx = torch.meshgrid(torch.arange(self.H, dtype=torch.float32),
                                torch.arange(self.W, dtype=torch.float32),
                                indexing="ij")
        self.xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (HW, 2)

    @classmethod
    def from_zhou(cls, labels, grids, isos_list, fg, args) -> "Relax":
        H, W = fg[0].shape
        z0 = np.zeros((len(labels), H, W), dtype=np.float32)
        for i, lab in enumerate(labels):
            z0[i][grids[0] == lab] = args.init_logit
        aff_np = []
        for t in range(1, len(fg)):
            aff_t = []
            for lab in labels:
                A, b = iso_affine(isos_list[t][lab])
                ys, xs = np.nonzero(grids[t] == lab)
                aff_t.append((A, b, np.array([xs.mean(), ys.mean()])))
            aff_np.append(aff_t)
        return cls(z0, aff_np, fg)

    def fields(self, tau: float) -> torch.Tensor:
        return torch.softmax(self.z / tau, dim=0) * self.mask0

    def inverse_grids(self, t: int) -> torch.Tensor:
        """Normalized grid_sample grids (k, H, W, 2) mapping assembly-t pixels
        back to frame 0 through T_it^-1."""
        j = t - 1
        cos, sin = torch.cos(self.theta[j]), torch.sin(self.theta[j])
        # R(-theta) rows: [cos, sin; -sin, cos]
        Rm = torch.stack([torch.stack([cos, sin], -1),
                          torch.stack([-sin, cos], -1)], -2)        # (k,2,2)
        M = torch.bmm(self.Ainv[j], Rm)                              # (k,2,2)
        cd = self.c[j] + self.trans[j]                               # (k,2)
        off = torch.bmm(self.Ainv[j],
                        (self.c[j] - self.b[j]
                         - torch.bmm(Rm, cd.unsqueeze(-1)).squeeze(-1)
                         ).unsqueeze(-1)).squeeze(-1)                # (k,2)
        q = torch.einsum("kab,nb->kna", M, self.xy) + off[:, None, :]
        gx = (q[..., 0] + 0.5) * (2.0 / self.W) - 1.0
        gy = (q[..., 1] + 0.5) * (2.0 / self.H) - 1.0
        return torch.stack([gx, gy], dim=-1).reshape(self.k, self.H, self.W, 2)

    def coverage(self, u: torch.Tensor, t: int) -> torch.Tensor:
        grid = self.inverse_grids(t)
        samp = F.grid_sample(u.unsqueeze(1), grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
        return samp.squeeze(1).sum(dim=0)                            # (H, W)

    def losses(self, tau: float, args) -> dict[str, torch.Tensor]:
        u = self.fields(tau)
        cover = u.new_zeros(())
        for t in range(1, self.T):
            c = self.coverage(u, t)
            m = self.masks[t]
            # asymmetric: overlap (c > 1) is unbuildable, a gap is just a gap
            gap = torch.relu(1.0 - c) ** 2
            over = torch.relu(c - 1.0) ** 2
            cover = cover + (m * (gap + args.over_weight * over)
                             + args.spill_weight * (1.0 - m) * c ** 2).mean()
        cover = cover / (self.T - 1)

        gx = u[:, :, 1:] - u[:, :, :-1]
        gy = u[:, 1:, :] - u[:, :-1, :]
        perim = args.eps * ((gx ** 2).sum(0).mean() + (gy ** 2).sum(0).mean())
        # double-well, weighted separately and annealed UP by the caller:
        # blending (u=0.5/0.5) lets soft coverage hide hard overlap, so the
        # commitment pressure must eventually rival the coverage term
        well = ((u ** 2) * (1.0 - u) ** 2).sum(0)[self.mask0 > 0.5].mean()

        areas = u.sum(dim=(1, 2))
        avg = self.mask0.sum() / self.k
        minarea = ((torch.relu(args.min_area_frac * avg - areas) / avg) ** 2).mean()

        reg = (self.theta ** 2).mean() + ((self.trans / 10.0) ** 2).mean()
        return {"cover": cover, "perim": perim, "well": well,
                "minarea": minarea, "reg": reg}

    # --------------------------------------------------------------- eval --
    def hard_index(self) -> np.ndarray:
        """argmax piece index inside silhouette 0, -1 outside."""
        idx = torch.argmax(self.z, dim=0).numpy()
        idx[self.mask0.numpy() < 0.5] = -1
        return idx

    def memberships(self, idx0: np.ndarray, t: int) -> np.ndarray:
        """(k, H, W) bool: pixel of assembly t is covered by mapped piece i
        (nearest-neighbor pullback of the hardened frame-0 labels)."""
        with torch.no_grad():
            grid = self.inverse_grids(t)
        q = grid.numpy()
        qx = np.round((q[..., 0] + 1.0) * self.W / 2.0 - 0.5).astype(np.int64)
        qy = np.round((q[..., 1] + 1.0) * self.H / 2.0 - 0.5).astype(np.int64)
        inb = (qx >= 0) & (qx < self.W) & (qy >= 0) & (qy < self.H)
        out = np.zeros((self.k, self.H, self.W), dtype=bool)
        for i in range(self.k):
            ii = inb[i]
            out[i][ii] = idx0[qy[i][ii], qx[i][ii]] == i
        return out

    def clean_index(self, idx0: np.ndarray) -> np.ndarray:
        """Keep the largest connected component of each piece; reassign the
        orphaned crumbs (and nothing else) to the nearest surviving label.
        The annealed double-well commits contested pixels independently, so
        the raw argmax is speckled; pieces must be simply connected for the
        downstream loop-based BACO."""
        out = idx0.copy()
        orphan = np.zeros(idx0.shape, dtype=bool)
        for i in range(self.k):
            m = idx0 == i
            lab, n = cc_label(m)
            if n <= 1:
                continue
            sizes = np.bincount(lab.ravel())
            keep = int(np.argmax(sizes[1:])) + 1
            orphan |= m & (lab != keep)
        if orphan.any():
            valid = (out >= 0) & ~orphan
            _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
            out[orphan] = out[iy, ix][orphan]
        return out

    def assembly_label_map(self, idx0: np.ndarray, t: int) -> np.ndarray:
        """Partition of silhouette t implied by the relaxed solution: nearest
        pullback claims, ties to the piece whose frame-0 claim is deepest
        inside, uncovered silhouette pixels filled from the nearest claim
        (Zhou's per-assembly label-map semantics: mismatch lives in the
        comparison against rigid transport, not in the map itself)."""
        if t == 0:
            return idx0.copy()
        with torch.no_grad():
            grid = self.inverse_grids(t)
        q = grid.numpy()
        qx = np.round((q[..., 0] + 1.0) * self.W / 2.0 - 0.5).astype(np.int64)
        qy = np.round((q[..., 1] + 1.0) * self.H / 2.0 - 0.5).astype(np.int64)
        inb = (qx >= 0) & (qx < self.W) & (qy >= 0) & (qy < self.H)
        best = np.full((self.H, self.W), -1, dtype=np.int32)
        bestd = np.zeros((self.H, self.W), dtype=np.float64)
        for i in range(self.k):
            depth = distance_transform_edt(idx0 == i)
            claim = np.zeros((self.H, self.W), dtype=bool)
            d = np.zeros((self.H, self.W), dtype=np.float64)
            ii = inb[i]
            claim[ii] = idx0[qy[i][ii], qx[i][ii]] == i
            d[ii] = depth[qy[i][ii], qx[i][ii]]
            upd = claim & (d > bestd)
            best[upd] = i
            bestd[upd] = d[upd]
        m = self.masks[t].numpy() > 0.5
        valid = (best >= 0) & m
        if (~valid & m).any():
            _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
            fill = best[iy, ix]
            best = np.where(~valid & m, fill, best)
        best[~m] = -1
        return best

    def export_zhou_level(self, idx0: np.ndarray, labels, names, k,
                          src_dir: Path, export_tag: str) -> Path:
        """Write the relaxed solution as a zhou_multi level (color PNGs that
        round-trip through decode_label_png) under a new tag, ready for
        exp462 --stage baco."""
        dst = RESULTS / export_tag
        (dst / "masks").mkdir(parents=True, exist_ok=True)
        (dst / "zhou").mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copyfile(src_dir / "masks" / f"{n}.png",
                            dst / "masks" / f"{n}.png")
        for t in range(self.T):
            lt = self.assembly_label_map(idx0, t)
            img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            for i, lab in enumerate(labels):
                img[lt == i] = _label_color(int(lab))
            Image.fromarray(img).save(dst / "zhou" / f"level_{k}pcs_S{t}.png")
        return dst

    def piece_geoms(self, idx0: np.ndarray):
        """Frame-0 piece polygons: union of unit pixel squares of the
        hardened label map (pixel centers at integers, edges at +/-0.5 --
        the same convention as contour_loop's 0.5-level silhouette)."""
        geoms = []
        for i in range(self.k):
            ys, xs = np.nonzero(idx0 == i)
            geoms.append(unary_union([box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
                                      for x, y in zip(xs, ys)]))
        return geoms

    def forward_pts(self, t: int, i: int):
        """Numpy point-array forward map of piece i into assembly t with the
        current pose delta: y = R(dth)(A p + b - c) + c + d."""
        if t == 0:
            return lambda pts: pts
        A, b, c = self.aff_np[t - 1][i]
        with torch.no_grad():
            th = float(self.theta[t - 1, i])
            d = self.trans[t - 1, i].numpy().astype(np.float64)
        ca, sa = np.cos(th), np.sin(th)
        R = np.array([[ca, -sa], [sa, ca]])
        return lambda pts: (pts @ A.T + b - c) @ R.T + c + d

    def vec_metrics(self, idx0: np.ndarray, fg) -> dict[str, float]:
        """exp462's polygon-based assembly metrics on the hardened pieces
        mapped with the continuous poses (apples-to-apples with baco_k*)."""
        from exp462_zhou_multitarget_baco_2d import _transform_geom
        geoms = self.piece_geoms(idx0)
        mapped = {}
        for t in range(self.T):
            for i in range(self.k):
                mapped[(t, i)] = (geoms[i] if t == 0 else
                                  _transform_geom(geoms[i], self.forward_pts(t, i)))
        return assembly_metrics_geoms(mapped, list(range(self.k)), self.T, fg)

    def metrics(self, idx0: np.ndarray) -> list[dict[str, float]]:
        rows = [{"uncovered_pct": 0.0, "outside_pct": 0.0, "overlap_pct": 0.0}]
        for t in range(1, self.T):
            mem = self.memberships(idx0, t)
            n = mem.sum(axis=0)
            m = self.masks[t].numpy() > 0.5
            area = float(m.sum())
            rows.append({
                "uncovered_pct": 100.0 * float(((n == 0) & m).sum()) / area,
                "outside_pct": 100.0 * float(((n >= 1) & ~m).sum()) / area,
                "overlap_pct": 100.0 * float((n >= 2).sum()) / area,
            })
        return rows


def optimize(model: Relax, args, iters: int, *, log_every: int = 100,
             quiet: bool = False) -> list[dict[str, float]]:
    """The GNC schedule: anneal tau down and the double-well weight up while
    Adam descends logits + poses jointly."""
    opt = torch.optim.Adam([
        {"params": [model.z], "lr": args.lr_z},
        {"params": [model.theta, model.trans], "lr": args.lr_pose},
    ])
    history = []
    for it in range(iters):
        prog = it / max(1, iters - 1)
        tau = args.tau0 * (args.tau1 / args.tau0) ** prog
        w_well = args.w_well0 * (args.w_well1 / args.w_well0) ** prog
        opt.zero_grad()
        ls = model.losses(tau, args)
        total = (args.w_cover * ls["cover"] + args.w_perim * ls["perim"]
                 + w_well * ls["well"]
                 + args.w_minarea * ls["minarea"] + args.w_pose * ls["reg"])
        total.backward()
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            row = {"it": it, "tau": tau, "w_well": w_well,
                   **{n: float(v.detach()) for n, v in ls.items()},
                   "total": float(total.detach())}
            history.append(row)
            if not quiet:
                print(f"  it {it:5d} tau {tau:.3f} w_well {w_well:.3f} "
                      f"total {row['total']:.5f} cover {row['cover']:.5f} "
                      f"perim {row['perim']:.4f} well {row['well']:.4f}")
    return history


def render(model: Relax, idx0: np.ndarray, names, path: Path) -> None:
    colors = np.array([tuple(int(PALETTE[i % len(PALETTE)][j:j + 2], 16)
                             for j in (1, 3, 5)) for i in range(model.k)],
                      dtype=np.uint8)
    fig, axes = plt.subplots(1, model.T, figsize=(6 * model.T, 6))
    for t, ax in enumerate(np.atleast_1d(axes)):
        img = np.full((model.H, model.W, 3), 255, dtype=np.uint8)
        if t == 0:
            for i in range(model.k):
                img[idx0 == i] = colors[i]
        else:
            mem = model.memberships(idx0, t)
            for i in range(model.k):
                img[mem[i]] = colors[i]
            img[mem.sum(axis=0) >= 2] = (127, 0, 0)
        ax.imshow(img)
        ax.contour(model.masks[t].numpy(), levels=[0.5], colors="k", linewidths=1.0)
        ax.set_title(names[t])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--names", required=True, help="comma list, search order")
    p.add_argument("--pieces", type=int, default=0)
    p.add_argument("--suffix", default="")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--tau0", type=float, default=1.0)
    p.add_argument("--tau1", type=float, default=0.25)
    p.add_argument("--eps", type=float, default=1.5, help="Modica-Mortola width (px)")
    p.add_argument("--w-cover", type=float, default=1.0)
    p.add_argument("--w-perim", type=float, default=0.03)
    p.add_argument("--w-minarea", type=float, default=1.0)
    p.add_argument("--w-pose", type=float, default=1e-3)
    p.add_argument("--spill-weight", type=float, default=1.0)
    p.add_argument("--over-weight", type=float, default=3.0,
                   help="extra penalty on coverage > 1 (overlap) vs < 1 (gap)")
    p.add_argument("--w-well0", type=float, default=0.02,
                   help="double-well weight at iter 0 (graduated non-convexity)")
    p.add_argument("--w-well1", type=float, default=10.0,
                   help="double-well weight at the last iter; must rival the "
                        "coverage term so contested pixels commit to one piece")
    p.add_argument("--min-area-frac", type=float, default=0.3)
    p.add_argument("--init-logit", type=float, default=3.0)
    p.add_argument("--export-tag", default="",
                   help="write the relaxed labels as a zhou level under this "
                        "new tag (then run exp462 --stage baco on it)")
    p.add_argument("--lr-z", type=float, default=0.08)
    p.add_argument("--lr-pose", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir, names, k, labels, grids, isos_list, fg = build_data(args)
    model = Relax.from_zhou(labels, grids, isos_list, fg, args)
    run_dir = out_dir / f"cont_k{k}{args.suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    idx_init = model.hard_index()
    init_vec = model.vec_metrics(idx_init, fg)
    print("init (pure Zhou, discrete poses; polygon metrics):")
    for t, n in enumerate(names):
        print(f"  {n}: uncovered {init_vec[f's{t}_uncovered_pct']:.2f}%  outside "
              f"{init_vec[f's{t}_outside_pct']:.2f}%  overlap "
              f"{init_vec[f's{t}_overlap_pct']:.2f}%")
    render(model, idx_init, names, run_dir / "init_assemblies.png")

    started = time.perf_counter()
    history = optimize(model, args, args.iters)
    runtime = time.perf_counter() - started

    idx_fin = model.clean_index(model.hard_index())
    fin_vec = model.vec_metrics(idx_fin, fg)
    print(f"final ({runtime:.0f}s; polygon metrics):")
    for t, n in enumerate(names):
        print(f"  {n}: uncovered {fin_vec[f's{t}_uncovered_pct']:.2f}%  outside "
              f"{fin_vec[f's{t}_outside_pct']:.2f}%  overlap "
              f"{fin_vec[f's{t}_overlap_pct']:.2f}%")
    render(model, idx_fin, names, run_dir / "final_assemblies.png")

    with torch.no_grad():
        max_rot = float(torch.rad2deg(model.theta.abs().max()))
        max_tr = float(model.trans.abs().max())
    comps = [int(cc_label(idx_fin == i)[1]) for i in range(model.k)]
    changed = int(((idx_fin != idx_init) & (idx_init >= 0)).sum())
    print(f"pose deltas: max |rot| {max_rot:.1f} deg, max |trans| {max_tr:.2f} px; "
          f"{changed} px relabeled; components/piece {comps}")

    summary = {
        "args": vars(args), "k": k, "labels": labels, "runtime_s": runtime,
        "init_vec": init_vec, "final_vec": fin_vec,
        "max_rot_deg": max_rot, "max_trans_px": max_tr,
        "relabeled_px": changed, "components_per_piece": comps,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    if args.export_tag:
        dst = model.export_zhou_level(idx_fin, labels, names, k, out_dir,
                                      args.export_tag)
        print(f"exported zhou level for BACO polish: {dst}")
    print(f"out: {run_dir}")


if __name__ == "__main__":
    main()
