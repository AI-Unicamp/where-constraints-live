r"""Multi-target dissection search entirely in continuous space (no lattice).

exp463 relaxed Zhou's *output*; this runs Zhou's *search* without zhou_multi:
the level hierarchy, piece splitting, and pose discovery all operate on phase
fields and free SE(2) poses. Structure per level (k pieces -> k+1):

  propose   pose discovery = the continuous analog of Zhou's exhaustive
            lattice enumeration: FFT cross-correlation of the piece mask
            against the uncovered residual over --angles sampled rotations
            x both chiralities (self-calibrated against the autocorrelation
            peak, so no index-convention errors), top peaks re-scored
            exactly; greedy placement largest-piece-first
  descend   exp463's GNC schedule (Relax.optimize): coverage + perimeter +
            annealed double-well, Adam on logits + poses jointly
  split     H1 analog: every error pixel (uncovered / outside / overlap) is
            blamed on the nearest claiming piece; the worst-blamed piece is
            split by farthest-pair EDT seeds; children inherit the parent
            pose but each is re-proposed against the residual and keeps
            whichever placement scores better

Pieces live in frame 0 (assembly 0 is watertight by construction). The map
into assembly t is Relax's  y = R(theta)(A p + b - c) + c + d  with
A = S_flip, b = 0, c = S_flip @ centroid: theta/d are ABSOLUTE pose values
(so --w-pose defaults to 0 here), flip is discrete from the proposal.

Run (masks come from a prepared exp462 tag):
  python experiments\exp464_zhou_continuous_search_2d.py ^
      --tag exp462_bear_cat_fish_cycle_area --names bear,cat,fish --k-target 9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import affine_transform, distance_transform_edt
from scipy.ndimage import label as cc_label
from scipy.signal import fftconvolve

from exp462_zhou_multitarget_baco_2d import RESULTS, load_fg
from exp463_zhou_continuous_relax_2d import Relax, optimize, render

FLIP = np.array([[-1.0, 0.0], [0.0, 1.0]])
IDENT = np.eye(2)


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def map_piece_image(P: np.ndarray, S: np.ndarray, theta: float, c: np.ndarray,
                    d: np.ndarray) -> np.ndarray:
    """Rasterize piece mask P (frame 0) into the assembly frame under
    y = R(theta)(S p - c) + c + d, c = S @ centroid. affine_transform needs
    the INVERSE map in (row, col) order: p = S(R(-theta)(y - c - d) + c)."""
    N = S @ rot(-theta)              # xy linear part of the inverse
    m = S @ (-rot(-theta) @ (c + d) + c)
    M_rc = np.array([[N[1, 1], N[1, 0]], [N[0, 1], N[0, 0]]])
    off_rc = np.array([m[1], m[0]])
    return affine_transform(P, M_rc, offset=off_rc, order=1, mode="constant")


def propose_poses(P: np.ndarray, c0: np.ndarray, residual: np.ndarray,
                  mask_t: np.ndarray, placed: np.ndarray, args,
                  inherited: tuple[bool, float, np.ndarray] | None = None,
                  ) -> tuple[bool, float, np.ndarray, float]:
    """Best (flip, theta, d) placing piece P onto the residual of assembly t.
    Candidates from FFT correlation peaks over sampled angles x chirality,
    re-scored exactly; the inherited pose (if any) competes on equal terms."""
    H, W = P.shape
    cands: list[tuple[float, bool, float, np.ndarray]] = []
    for flip in (False, True):
        S = FLIP if flip else IDENT
        c = S @ c0
        for ang in np.linspace(0.0, 2 * np.pi, args.angles, endpoint=False):
            # piece rotated in place (centroid stays at c0): d = c0 - c
            Pr = map_piece_image(P, S, ang, c, c0 - c)
            corr = fftconvolve(residual, Pr[::-1, ::-1], mode="same")
            auto = fftconvolve(Pr, Pr[::-1, ::-1], mode="same")
            p0 = np.unravel_index(np.argmax(auto), auto.shape)
            work = corr.copy()
            for _ in range(args.peaks_per_angle):
                pk = np.unravel_index(np.argmax(work), work.shape)
                val = work[pk]
                if val <= 0:
                    break
                yy, xx = np.ogrid[:H, :W]
                work[(yy - pk[0]) ** 2 + (xx - pk[1]) ** 2 < 64] = -1
                shift_xy = np.array([pk[1] - p0[1], pk[0] - p0[0]], float)
                d = (c0 - c) + shift_xy
                cands.append((float(val), flip, float(ang), d))
    cands.sort(key=lambda r: -r[0])
    if inherited is not None:
        f, th, d = inherited
        cands.insert(0, (np.inf, f, th, d))  # always exactly scored
    best, best_score = None, -np.inf
    for _, flip, ang, d in cands[: args.exact_evals + (inherited is not None)]:
        S = FLIP if flip else IDENT
        Pm = map_piece_image(P, S, ang, S @ c0, d)
        score = (float((Pm * residual).sum())
                 - 0.7 * float((Pm * (1.0 - mask_t)).sum())
                 - 0.7 * float((Pm * placed).sum()))
        if score > best_score:
            best_score, best = score, (flip, ang, d)
    return best[0], best[1], best[2], best_score


class Search:
    """Continuous-space hierarchical search state (frame-0 labels + poses)."""

    def __init__(self, fg, args):
        self.fg = fg
        self.T = len(fg)
        self.H, self.W = fg[0].shape
        self.args = args
        self.mask0 = fg[0].astype(bool)
        # level 1: two farthest-pair seeded pieces in frame 0
        self.labels0 = self._seed_partition(self.mask0, 2)
        self.k = 2
        self.flips = np.zeros((self.T - 1, 2), dtype=bool)
        self.thetas = np.zeros((self.T - 1, 2))
        self.ds = np.zeros((self.T - 1, 2, 2))
        for t in range(1, self.T):
            self._place_all(t)

    @staticmethod
    def _seed_partition(mask: np.ndarray, n: int) -> np.ndarray:
        """Farthest-pair seeds, nearest-seed labels (level-1 flood analog)."""
        edt = distance_transform_edt(mask)
        s1 = np.unravel_index(np.argmax(edt), edt.shape)
        ys, xs = np.nonzero(mask)
        d1 = (ys - s1[0]) ** 2 + (xs - s1[1]) ** 2
        s2 = (ys[np.argmax(d1)], xs[np.argmax(d1)])
        lab = np.full(mask.shape, -1, dtype=np.int32)
        d2 = (ys - s2[0]) ** 2 + (xs - s2[1]) ** 2
        lab[ys, xs] = (d2 < d1).astype(np.int32)
        return lab

    def piece_mask(self, i: int) -> np.ndarray:
        return (self.labels0 == i).astype(np.float64)

    def centroid(self, i: int) -> np.ndarray:
        ys, xs = np.nonzero(self.labels0 == i)
        return np.array([xs.mean(), ys.mean()])

    def _place_all(self, t: int) -> None:
        """Greedy initial placement of every piece onto assembly t."""
        m = self.fg[t].astype(np.float64)
        residual, placed = m.copy(), np.zeros_like(m)
        order = sorted(range(self.k), key=lambda i: -self.piece_mask(i).sum())
        for i in order:
            P, c0 = self.piece_mask(i), self.centroid(i)
            f, th, d, _ = propose_poses(P, c0, residual, m, placed, self.args)
            self.flips[t - 1, i], self.thetas[t - 1, i], self.ds[t - 1, i] = f, th, d
            Pm = map_piece_image(P, FLIP if f else IDENT, th,
                                 (FLIP if f else IDENT) @ c0, d)
            placed += Pm
            residual = np.clip(m - placed, 0.0, 1.0)

    # ------------------------------------------------------------ descent --
    def build_relax(self) -> Relax:
        z0 = np.full((self.k, self.H, self.W), 0.0, dtype=np.float32)
        for i in range(self.k):
            z0[i][self.labels0 == i] = self.args.init_logit
        aff, th0, tr0 = [], [], []
        for t in range(1, self.T):
            aff_t = []
            for i in range(self.k):
                S = FLIP if self.flips[t - 1, i] else IDENT
                aff_t.append((S.copy(), np.zeros(2), S @ self.centroid(i)))
            aff.append(aff_t)
            th0.append(self.thetas[t - 1].copy())
            tr0.append(self.ds[t - 1].copy())
        return Relax(z0, aff, [m.astype(np.float64) for m in self.fg],
                     theta0=np.stack(th0), trans0=np.stack(tr0))

    def absorb(self, model: Relax) -> None:
        """Pull the optimized labels + poses back, re-anchoring d to the new
        centroids so the map is unchanged: d' = d + (R - I) S (c0' - c0)."""
        idx = model.clean_index(model.hard_index())
        old_c0 = [self.centroid(i) for i in range(self.k)]
        self.labels0 = idx
        with torch.no_grad():
            th = model.theta.numpy().astype(np.float64)
            tr = model.trans.numpy().astype(np.float64)
        for t in range(1, self.T):
            for i in range(self.k):
                S = FLIP if self.flips[t - 1, i] else IDENT
                dc = self.centroid(i) - old_c0[i]
                self.thetas[t - 1, i] = th[t - 1, i]
                self.ds[t - 1, i] = (tr[t - 1, i]
                                     + (rot(th[t - 1, i]) - IDENT) @ (S @ dc))

    # -------------------------------------------------------------- split --
    def blame(self, model: Relax) -> np.ndarray:
        """Charge every uncovered / outside / overlap pixel to the nearest
        claiming piece, summed over assemblies."""
        idx = model.clean_index(model.hard_index())
        out = np.zeros(self.k)
        for t in range(1, self.T):
            mem = model.memberships(idx, t)
            n = mem.sum(axis=0)
            m = self.fg[t].astype(bool)
            claimed = n > 0
            owner = np.argmax(mem, axis=0)
            for i in range(self.k):
                out[i] += float((mem[i] & ~m).sum())          # outside
                out[i] += float((mem[i] & (n >= 2) & m).sum())  # overlap
            uncov = m & ~claimed
            if uncov.any() and claimed.any():
                _, (iy, ix) = distance_transform_edt(~claimed,
                                                     return_indices=True)
                near = owner[iy, ix][uncov]
                for i in range(self.k):
                    out[i] += float((near == i).sum())          # gaps
        return out

    def split(self, model: Relax) -> int:
        worst = int(np.argmax(self.blame(model)))
        pm = self.labels0 == worst
        parent_c0 = self.centroid(worst)
        sub = self._seed_partition(pm, 2)
        new = self.k
        self.labels0 = self.labels0.copy()
        self.labels0[pm & (sub == 1)] = new
        # children inherit the parent pose (re-anchored to their own
        # centroids), then each is re-proposed against the residual and
        # keeps the better placement
        self.flips = np.concatenate([self.flips,
                                     self.flips[:, worst:worst + 1]], axis=1)
        self.thetas = np.concatenate([self.thetas,
                                      self.thetas[:, worst:worst + 1]], axis=1)
        self.ds = np.concatenate([self.ds, self.ds[:, worst:worst + 1]], axis=1)
        self.k += 1
        for t in range(1, self.T):
            for i in (worst, new):
                S = FLIP if self.flips[t - 1, i] else IDENT
                dc = self.centroid(i) - parent_c0
                self.ds[t - 1, i] = (self.ds[t - 1, i]
                                     + (rot(self.thetas[t - 1, i]) - IDENT)
                                     @ (S @ dc))
        for t in range(1, self.T):
            m = self.fg[t].astype(np.float64)
            others = np.zeros_like(m)
            for i in range(self.k):
                if i in (worst, new):
                    continue
                S = FLIP if self.flips[t - 1, i] else IDENT
                others += map_piece_image(self.piece_mask(i), S,
                                          self.thetas[t - 1, i],
                                          S @ self.centroid(i),
                                          self.ds[t - 1, i])
            placed = others.copy()
            residual = np.clip(m - placed, 0.0, 1.0)
            for i in (worst, new):
                P, c0 = self.piece_mask(i), self.centroid(i)
                f, th, d, _ = propose_poses(
                    P, c0, residual, m, placed, self.args,
                    inherited=(bool(self.flips[t - 1, i]),
                               float(self.thetas[t - 1, i]),
                               self.ds[t - 1, i].copy()))
                self.flips[t - 1, i], self.thetas[t - 1, i] = f, th
                self.ds[t - 1, i] = d
                Pm = map_piece_image(P, FLIP if f else IDENT, th,
                                     (FLIP if f else IDENT) @ c0, d)
                placed += Pm
                residual = np.clip(m - placed, 0.0, 1.0)
        return worst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="exp462_bear_cat_fish_cycle_area",
                   help="source tag holding masks/ (prepared by exp462)")
    p.add_argument("--names", required=True)
    p.add_argument("--k-target", type=int, default=9)
    p.add_argument("--iters-level", type=int, default=600)
    p.add_argument("--iters-final", type=int, default=2500)
    p.add_argument("--angles", type=int, default=16)
    p.add_argument("--peaks-per-angle", type=int, default=2)
    p.add_argument("--exact-evals", type=int, default=10)
    p.add_argument("--suffix", default="")
    p.add_argument("--export-tag", default="")
    # GNC schedule (same knobs as exp463; theta/d are absolute -> w_pose 0)
    p.add_argument("--tau0", type=float, default=3.0)
    p.add_argument("--tau1", type=float, default=0.15)
    p.add_argument("--eps", type=float, default=1.5)
    p.add_argument("--w-cover", type=float, default=1.0)
    p.add_argument("--w-perim", type=float, default=0.03)
    p.add_argument("--w-minarea", type=float, default=1.0)
    p.add_argument("--w-pose", type=float, default=0.0)
    p.add_argument("--spill-weight", type=float, default=1.0)
    p.add_argument("--over-weight", type=float, default=3.0)
    p.add_argument("--w-well0", type=float, default=0.02)
    p.add_argument("--w-well1", type=float, default=10.0)
    p.add_argument("--min-area-frac", type=float, default=0.3)
    p.add_argument("--init-logit", type=float, default=1.0)
    p.add_argument("--lr-z", type=float, default=0.08)
    p.add_argument("--lr-pose", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    src = RESULTS / args.tag
    fg = [load_fg(src / "masks" / f"{n}.png") for n in names]
    run_dir = RESULTS / f"exp464_continuous_search{args.suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    search = Search(fg, args)
    level_log = []
    model = search.build_relax()
    while True:
        print(f"level k={search.k}: descend {args.iters_level} iters")
        optimize(model, args, args.iters_level, log_every=args.iters_level - 1)
        search.absorb(model)
        model = search.build_relax()  # re-anchored
        vec = model.vec_metrics(search.labels0, fg)
        row = {"k": search.k,
               **{f"s{t}_{w}": vec[f"s{t}_{w}_pct"] for t in range(search.T)
                  for w in ("uncovered", "outside", "overlap")}}
        level_log.append(row)
        print("  " + "  ".join(
            f"{names[t]} un {vec[f's{t}_uncovered_pct']:.2f} out "
            f"{vec[f's{t}_outside_pct']:.2f} ov {vec[f's{t}_overlap_pct']:.2f}"
            for t in range(search.T)))
        render(model, search.labels0, names, run_dir / f"level_k{search.k}.png")
        if search.k >= args.k_target:
            break
        worst = search.split(model)
        print(f"  split piece {worst} -> k={search.k}")
        model = search.build_relax()

    print(f"final polish: {args.iters_final} iters")
    optimize(model, args, args.iters_final, log_every=500)
    search.absorb(model)
    model = search.build_relax()
    vec = model.vec_metrics(search.labels0, fg)
    runtime = time.perf_counter() - started
    print(f"final ({runtime:.0f}s; polygon metrics):")
    for t, n in enumerate(names):
        print(f"  {n}: uncovered {vec[f's{t}_uncovered_pct']:.2f}%  outside "
              f"{vec[f's{t}_outside_pct']:.2f}%  overlap "
              f"{vec[f's{t}_overlap_pct']:.2f}%")
    render(model, search.labels0, names, run_dir / "final_assemblies.png")
    summary = {"args": vars(args), "runtime_s": runtime,
               "levels": level_log, "final_vec": vec}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    if args.export_tag:
        labels = list(range(1, search.k + 1))
        idx = search.labels0
        dst = model.export_zhou_level(idx, labels, names, search.k, src,
                                      args.export_tag)
        print(f"exported zhou level for BACO polish: {dst}")
    print(f"out: {run_dir}")


if __name__ == "__main__":
    main()
