#!/usr/bin/env python3
"""Recompute the main Chapter 3 budget-sweep tables from the 432 per-run result.json files.

Tables reproduced:
  * top-budget comparison at (b, N, K) = (10, 50, 16), mean +- sample sd over 3 targets x 3 seeds;
  * best top-budget methods by target;
  * budget sweep: hard multi-hot IoU by target and regime, averaged over the four architectures and three seeds;
  * matched multi-hot versus one-hot effect, paired by target, budget, architecture and seed
    (mean difference, two-sided 95% Student-t CI, paired t-test p-value).

Note: every IoU column is the maximum over evaluated epochs, taken separately for soft, one-hot and multi-hot
evaluation; active counts are measured at the final epoch (see the thesis text).

Usage:
    python make_sweep_tables.py --data path/to/csg_budget_siggraph_fixed
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARCHS = ['nary', 'tree', 'stump', 'paper_stump']


def load(root):
    runs = []
    for f in sorted(root.glob('*/runs/*/result.json')):
        d = json.loads(f.read_text())
        m = d['method']
        arch = 'paper_stump' if m.startswith('paper_stump') else m.split('_')[0]
        regime = 'multihot' if 'multihot' in m else 'onehot'
        seed = int(re.search(r'seed(\d+)', f.parent.name).group(1))
        runs.append(dict(target=d['mesh'], arch=arch, regime=regime, seed=seed, N=d['budget']['bc_prims'],
                         hard_multi=d['best_hard_multihot_iou'], soft=d['best_soft_iou'],
                         hard_one=d['best_hard_onehot_iou'],
                         active=d['stats']['active_physical_primitives'],
                         avail=d['stats']['available_physical_primitives']))
    return runs


def ms(x):
    x = np.asarray(x, float)
    return f'{x.mean():.4f} +- {x.std(ddof=1):.4f}' if len(x) > 1 else f'{x.mean():.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    runs = load(Path(ap.parse_args().data))
    print(f'{len(runs)} runs\n')

    print('Top-budget (N = 50): method | hard multi | soft | hard one | active | available')
    top = [r for r in runs if r['N'] == 50]
    by = defaultdict(list)
    for r in top:
        by[(r['arch'], r['regime'])].append(r)
    for k, v in sorted(by.items(), key=lambda kv: -np.mean([r['hard_multi'] for r in kv[1]])):
        print(f"  {k[0]}_{k[1]:9s} {ms([r['hard_multi'] for r in v])} | {ms([r['soft'] for r in v])} | "
              f"{ms([r['hard_one'] for r in v])} | {np.mean([r['active'] for r in v]):.1f} +- "
              f"{np.std([r['active'] for r in v], ddof=1):.1f} | {v[0]['avail']}")

    print('\nBest top-budget method per target and regime (by mean hard multi-hot IoU):')
    for t in sorted({r['target'] for r in runs}):
        for reg in ('onehot', 'multihot'):
            cand = defaultdict(list)
            for r in top:
                if r['target'] == t and r['regime'] == reg:
                    cand[r['arch']].append(r['hard_multi'])
            a, v = max(cand.items(), key=lambda kv: np.mean(kv[1]))
            print(f'  {t:10s} {reg:9s} {a}: {ms(v)}')

    print('\nBudget sweep (hard multi-hot IoU, mean over 4 architectures x 3 seeds):')
    Ns = sorted({r['N'] for r in runs})
    print('  target/regime        ' + '  '.join(f'N={n:<5d}' for n in Ns))
    for t in sorted({r['target'] for r in runs}):
        for reg in ('multihot', 'onehot'):
            vals = [np.mean([r['hard_multi'] for r in runs if r['target'] == t and r['regime'] == reg and r['N'] == n]) for n in Ns]
            print(f'  {t:10s} {reg:9s} ' + '  '.join(f'{v:.4f}' for v in vals))

    print('\nMatched multi-hot minus one-hot (pairs by target, budget, architecture, seed):')
    key = lambda r: (r['target'], r['N'], r['arch'], r['seed'])
    one = {key(r): r for r in runs if r['regime'] == 'onehot'}
    pairs = [(r, one[key(r)]) for r in runs if r['regime'] == 'multihot' and key(r) in one]
    for grp in [None] + ARCHS:
        P = [(m, o) for m, o in pairs if grp is None or m['arch'] == grp]
        d = np.array([m['hard_multi'] - o['hard_multi'] for m, o in P])
        n = len(d); se = d.std(ddof=1) / np.sqrt(n); tcrit = stats.t.ppf(0.975, n - 1)
        p = stats.ttest_rel([m['hard_multi'] for m, o in P], [o['hard_multi'] for m, o in P]).pvalue
        dsoft = np.mean([m['soft'] - o['soft'] for m, o in P])
        done = np.mean([m['hard_one'] - o['hard_one'] for m, o in P])
        dact = np.mean([m['active'] - o['active'] for m, o in P])
        print(f"  {grp or 'all':11s} n={n:3d}  d={d.mean():+.4f}  CI=[{d.mean() - tcrit * se:+.4f},{d.mean() + tcrit * se:+.4f}]"
              f"  p={p:.2g}  d_soft={dsoft:+.4f}  d_hard_one={done:+.4f}  d_active={dact:+.1f}")


if __name__ == '__main__':
    main()
