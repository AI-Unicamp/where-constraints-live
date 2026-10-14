#!/usr/bin/env python3
"""Descriptive factor model of Chapter 3 (Table "Descriptive factor model for hard multi-hot IoU over all 432 runs").

Reads the 432 per-run result.json files of the main budget sweep (data folder csg_budget_siggraph_fixed/) and fits an
ordinary least-squares model of best hard multi-hot IoU on: centred log2(N), extraction regime (multi-hot vs one-hot),
target (Fandisk, Rocker-arm vs Bunny), architecture (paper stump, stump, tree vs n-ary), and the interactions
multi-hot x log2 N, Fandisk x log2 N, Rocker-arm x log2 N. The reference condition is Bunny, n-ary, one-hot, at the
mean of log2 N. Prints coefficients, standard errors, two-sided p-values, R^2 and the residual standard error.

Usage:
    python make_factor_model.py --data path/to/csg_budget_siggraph_fixed
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def load_runs(root: Path):
    rows = []
    for f in sorted(root.glob('*/runs/*/result.json')):
        d = json.loads(f.read_text())
        m = d['method']
        arch = 'paper_stump' if m.startswith('paper_stump') else m.split('_')[0]
        rows.append(dict(target=d['mesh'], arch=arch, multihot='multihot' in m,
                         N=d['budget']['bc_prims'], y=d['best_hard_multihot_iou']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder csg_budget_siggraph_fixed (contains bunny/, fandisk/, rockerarm/)')
    a = ap.parse_args()
    rows = load_runs(Path(a.data))
    n = len(rows)
    l2 = np.array([math.log2(r['N']) for r in rows])
    c = l2 - l2.mean()
    mh = np.array([r['multihot'] for r in rows], float)
    fan = np.array([r['target'] == 'fandisk' for r in rows], float)
    rock = np.array([r['target'] in ('rocker-arm', 'rockerarm') for r in rows], float)
    ps = np.array([r['arch'] == 'paper_stump' for r in rows], float)
    st = np.array([r['arch'] == 'stump' for r in rows], float)
    tr = np.array([r['arch'] == 'tree' for r in rows], float)
    y = np.array([r['y'] for r in rows])
    names = ['Intercept', 'Centred log2 N', 'Multi-hot regime', 'Fandisk target', 'Rocker-arm target',
             'Paper-stump architecture', 'Stump architecture', 'Tree architecture',
             'Multi-hot x log2 N', 'Fandisk x log2 N', 'Rocker-arm x log2 N']
    X = np.column_stack([np.ones(n), c, mh, fan, rock, ps, st, tr, mh * c, fan * c, rock * c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = n - X.shape[1]
    s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    print(f'runs: {n}   R^2 = {r2:.3f}   residual standard error = {math.sqrt(s2):.4f}')
    for nm, b, s, pv in zip(names, beta, se, p):
        print(f'{nm:28s} {b:+.4f}  se {s:.4f}  p {pv:.3g}')


if __name__ == '__main__':
    main()
