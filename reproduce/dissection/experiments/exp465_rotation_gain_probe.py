r"""Reachability probe: does any piece prefer a non-90-degree pose?

For the best discrete topology (cycle K=9), each piece in each assembly is
re-proposed over the full continuous rotation circle (5-degree steps, both
chiralities, FFT translation proposals, exact rescoring) against the residual
left by the other pieces at their lattice poses. Compare the best continuous
pose against the piece's own lattice pose under the same score:

  gain ~ 0 and |dtheta| ~ 0 (mod 90)  -> lattice poses are native; a hybrid
                                         search would not help here
  gain large at |dtheta| ~ 25-60 deg  -> native non-90 matchings exist that
                                         lattice Zhou cannot express, so the
                                         lattice limits reachability

Score units are px^2 of residual covered (minus 0.7x spill/overlap), so gains
below ~5 px are noise and above ~20 px are substantial.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

from exp462_zhou_multitarget_baco_2d import RESULTS
from exp463_zhou_continuous_relax_2d import Relax, build_data
from exp464_zhou_continuous_search_2d import FLIP, IDENT, propose_poses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="exp462_bear_cat_fish_cycle")
    p.add_argument("--names", default="bear,cat,fish")
    p.add_argument("--pieces", type=int, default=9)
    p.add_argument("--angles", type=int, default=72)
    args = p.parse_args()
    args.init_logit = 3.0  # only used for z0 scale; hardened immediately

    out_dir, names, k, labels, grids, isos_list, fg = build_data(args)
    model = Relax.from_zhou(labels, grids, isos_list, fg, args)
    idx = model.clean_index(model.hard_index())
    prop_args = SimpleNamespace(angles=args.angles, peaks_per_angle=2,
                                exact_evals=12)

    rows = []
    for t in range(1, model.T):
        mem = model.memberships(idx, t).astype(np.float64)
        m = fg[t].astype(np.float64)
        for i in range(model.k):
            placed = mem.sum(axis=0) - mem[i]
            residual = np.clip(m - placed, 0.0, 1.0)
            Pm = mem[i]
            s_cur = (float((Pm * residual).sum())
                     - 0.7 * float((Pm * (1.0 - m)).sum())
                     - 0.7 * float((Pm * placed).sum()))
            P = (idx == i).astype(np.float64)
            ys, xs = np.nonzero(idx == i)
            c0 = np.array([xs.mean(), ys.mean()])
            f, ang, d, s_best = propose_poses(P, c0, residual, m, placed,
                                              prop_args)
            # lattice pose angle: linear part A = R(theta_disc) @ S_flip
            A, _, _ = model.aff_np[t - 1][i]
            flip_disc = np.linalg.det(A) < 0
            Rd = A @ (FLIP if flip_disc else IDENT)
            th_disc = float(np.arctan2(Rd[1, 0], Rd[0, 0]))
            dth = np.degrees(ang - th_disc)
            dth = (dth + 180.0) % 360.0 - 180.0
            dth90 = (dth + 45.0) % 90.0 - 45.0  # offset from nearest lattice pose
            rows.append({
                "t": t, "piece": i, "area_px": float(P.sum()),
                "score_lattice": round(s_cur, 1),
                "score_best": round(s_best, 1),
                "gain": round(s_best - s_cur, 1),
                "dtheta_deg": round(dth, 1),
                "dtheta_mod90_deg": round(dth90, 1),
                "flip_changed": bool(f != flip_disc),
            })
            print(f"  {names[t]:>4s} piece {i} (area {P.sum():4.0f}): "
                  f"lattice {s_cur:7.1f}  best {s_best:7.1f}  "
                  f"gain {s_best - s_cur:6.1f}  dtheta {dth:7.1f} "
                  f"(mod90 {dth90:6.1f}){'  FLIP' if f != flip_disc else ''}")

    gains = np.array([r["gain"] for r in rows])
    off = np.array([abs(r["dtheta_mod90_deg"]) for r in rows])
    big = [(r["t"], r["piece"], r["gain"], r["dtheta_mod90_deg"])
           for r in rows if r["gain"] > 20 and abs(r["dtheta_mod90_deg"]) > 20]
    print(f"\nsummary: median gain {np.median(gains):.1f} px, "
          f"median |dtheta mod 90| {np.median(off):.1f} deg")
    print(f"native non-90 wins (gain>20px AND offset>20deg): {len(big)}  {big}")
    out = RESULTS / args.tag / "rotation_gain_probe.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
