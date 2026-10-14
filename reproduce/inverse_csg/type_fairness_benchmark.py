#!/usr/bin/env python3
"""
Type-fairness benchmark: does Bc+vol select the right primitive type?

Six targets that isolate type selection from the other factors.

SINGLE-PRIMITIVE (type should match target):
  1. sphere          -> should select sphere
  2. box             -> should select block
  3. cylinder        -> should select cylinder
  4. wedge           -> should select wedge

MULTI-PRIMITIVE (type + boolean):
  5. sphere - cyl    -> should find {sphere:1, cylinder:1}
  6. (box+sph) - 2c  -> should find {block:1, sphere:1, cylinder:2}

Usage:
  python type_fairness_benchmark.py --epochs 12000
  python type_fairness_benchmark.py --epochs 8000 --seeds 1
"""

import os, sys, time, math, json, copy, argparse
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib.util
import types as _types
if importlib.util.find_spec('mcubes') is None:
    from skimage import measure
    mcubes_stub = _types.ModuleType('mcubes')
    def _marching_cubes(vol, iso):
        v, f, _, _ = measure.marching_cubes(vol, level=iso)
        return v, f.astype(np.int32)
    def _export_obj(vertices, triangles, filepath):
        with open(filepath, 'w') as fh:
            for x, y, z in vertices:
                fh.write(f'v {x} {y} {z}\n')
            for a, b, c in triangles:
                fh.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')
    mcubes_stub.marching_cubes = _marching_cubes
    mcubes_stub.export_obj = _export_obj
    sys.modules['mcubes'] = mcubes_stub

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nary_benchmark_v7 import train, device

from controlled_nary_tree_benchmark import ControlledNaryDiscrete, ControlledNaryBcVol, controlled_target_occ, controlled_eval, num_trainable_params, CONTROLLED_TARGETS

MIN_SIZE = 0.02

# TARGET SUITE
def _leaf(type_name, center, size, rot=(0.0, 0.0, 0.0)):
    return {'type': type_name, 'center': list(center),
            'size': list(size), 'rot': list(rot)}

SUITE = {
    'sphere': {
        'desc': 'single sphere',
        'expected_winner': 'sphere',
        'leaves': [_leaf('sphere', (0.0, 0.0, 0.0), (0.60, 0.60, 0.60))],
        'tree_ops': [],
        'dnf_groups': [{'pos': [0], 'neg': []}],
    },
    'box': {
        'desc': 'single box (anisotropic)',
        'expected_winner': 'block',
        'leaves': [_leaf('block', (0.0, 0.0, 0.0), (0.50, 0.40, 0.45))],
        'tree_ops': [],
        'dnf_groups': [{'pos': [0], 'neg': []}],
    },
    'cylinder': {
        'desc': 'single tall cylinder',
        'expected_winner': 'cylinder',
        'leaves': [_leaf('cylinder', (0.0, 0.0, 0.0), (0.20, 0.70, 0.20))],
        'tree_ops': [],
        'dnf_groups': [{'pos': [0], 'neg': []}],
    },
    'wedge': {
        'desc': 'single wedge (roof/ramp)',
        'expected_winner': 'wedge',
        'leaves': [_leaf('wedge', (0.0, 0.0, 0.0), (0.55, 0.35, 0.50))],
        'tree_ops': [],
        'dnf_groups': [{'pos': [0], 'neg': []}],
    },
    'sphere_minus_cyl': {
        'desc': 'sphere - cylinder',
        'expected_winner': None,
        'expected_types': {'sphere': 1, 'cylinder': 1},
        'leaves': [
            _leaf('sphere', (0.0, 0.0, 0.0), (0.70, 0.70, 0.70)),
            _leaf('cylinder', (0.0, 0.0, 0.0), (0.20, 1.10, 0.20)),
        ],
        'tree_ops': ['left_diff_right'],
        'dnf_groups': [{'pos': [0], 'neg': [1]}],
    },
    'box_sph_minus_2cyl': {
        'desc': '(box + sphere) - 2 cyl',
        'expected_winner': None,
        'expected_types': {'block': 1, 'sphere': 1, 'cylinder': 2},
        'leaves': [
            _leaf('block',    (0.10, 0.10, 0.00), (0.55, 0.42, 0.42)),
            _leaf('sphere',   (-0.15, -0.05, -0.05), (0.45, 0.45, 0.45)),
            _leaf('cylinder', (0.00, 0.00, 0.00), (0.18, 1.10, 0.18), (0.0, 0.0, 0.0)),
            _leaf('cylinder', (0.00, 0.00, 0.00), (0.18, 1.10, 0.18),
                  (0.0, 0.0, math.pi/2.0)),
        ],
        'tree_ops': ['union', 'union', 'left_diff_right'],
        'dnf_groups': [
            {'pos': [0], 'neg': [2, 3]},
            {'pos': [1], 'neg': [2, 3]},
        ],
    },
}

# Register all targets so controlled_target_occ can find them
for name, spec in SUITE.items():
    CONTROLLED_TARGETS[name] = spec


# TRAINING HELPER
def train_controlled(model, target_name, out_dir, epochs, samples,
                     resolution, desc, use_temp=True, resume=True):
    if num_trainable_params(model) == 0:
        return
    target_func = lambda p: controlled_target_occ(target_name, p)
    reg = dict(
        sparsity_warmup=max(epochs//20, 200),
        sparsity_rampup=max(epochs//12, 400),
        ls=1e-5, lb=1e-5, lc=1e-5, lgs=1e-5,
        ba_start=epochs//4, ba_end=int(epochs*0.75), ba_max=5e-5,
    )
    train(model, target_func, None, epochs, out_dir,
          num_samples=samples, resolution=resolution,
          resume=resume, desc=desc,
          use_temperature=use_temp, temp_start=1.0, temp_end=20.0, **reg)


# PER-TARGET RUNNER
def run_target(target_name, spec, args, seed):
    is_single = len(spec['leaves']) == 1
    OUT = os.path.join(args.output, target_name, f'seed{seed}')
    os.makedirs(OUT, exist_ok=True)

    if is_single:
        conditions = [
            ('perfect',
             lambda: ControlledNaryDiscrete(spec, train_membership=False),
             False, False),
            ('find_type',
             lambda: ControlledNaryBcVol(spec, train_type=True,
                         train_membership=False),
             True, True),
            ('wrong_type',
             lambda: ControlledNaryBcVol(spec, train_type=True,
                         train_membership=False, wrong_type_init=True),
             True, True),
        ]
    else:
        conditions = [
            ('perfect',
             lambda: ControlledNaryDiscrete(spec, train_membership=False),
             False, False),
            ('find_membership',
             lambda: ControlledNaryDiscrete(spec, train_membership=True),
             True, False),
            ('find_type',
             lambda: ControlledNaryBcVol(spec, train_type=True,
                         train_membership=False),
             True, True),
            ('find_both',
             lambda: ControlledNaryBcVol(spec, train_type=True,
                         train_membership=True),
             True, True),
            ('wrong_type',
             lambda: ControlledNaryBcVol(spec, train_type=True,
                         train_membership=True, wrong_type_init=True),
             True, True),
        ]

    results = {}
    for label, make_model, do_train, use_temp in conditions:
        torch.manual_seed(seed); np.random.seed(seed)
        model = make_model().to(device)
        t0 = time.time()
        if do_train:
            train_controlled(model, target_name,
                             os.path.join(OUT, label),
                             args.epochs, args.samples, args.resolution,
                             desc=f"{target_name}/{label}/s{seed}",
                             use_temp=use_temp,
                             resume=(not args.no_resume))
        res = controlled_eval(model, target_name,
                              os.path.join(OUT, label),
                              args.resolution, f"{target_name}/{label}",
                              eval_temperature=20.0 if use_temp else 1.0)
        res['time_s'] = time.time() - t0
        results[label] = res
    return results


# MAIN
def main():
    parser = argparse.ArgumentParser(description='Type-fairness benchmark')
    parser.add_argument('--epochs', type=int, default=12000)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--output', type=str,
                        default='./benchmark_type_fairness/')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--targets', type=str, nargs='+',
                        default=list(SUITE.keys()),
                        choices=list(SUITE.keys()))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*74}")
    print(f"  TYPE-FAIRNESS BENCHMARK")
    print(f"{'='*74}")
    print(f"  Targets: {args.targets}")
    print(f"  Epochs: {args.epochs}, Seeds: {args.seeds}, Device: {device}")
    print(f"  Key question: does the winning type change with the target?")
    print(f"{'='*74}\n")

    # Run everything
    all_results = {}
    for tname in args.targets:
        spec = SUITE[tname]
        print(f"\n{'_'*74}")
        print(f"  TARGET: {tname} -- {spec['desc']}")
        if spec.get('expected_winner'):
            print(f"  Expected winner: {spec['expected_winner']}")
        else:
            print(f"  Expected types: {spec.get('expected_types', '?')}")
        print(f"{'_'*74}")
        all_results[tname] = {}
        for seed in range(args.seeds):
            print(f"\n  -- seed {seed} --")
            all_results[tname][seed] = run_target(tname, spec, args, seed)

    # TABLE 1: Hard IoU per target x condition
    print(f"\n\n{'='*74}")
    print(f"  HARD IoU (averaged over {args.seeds} seeds)")
    print(f"{'='*74}")

    all_conds = set()
    for t in args.targets:
        for s in range(args.seeds):
            all_conds.update(all_results[t][s].keys())
    cond_order = [c for c in ['perfect','find_membership','find_type',
                               'find_both','wrong_type'] if c in all_conds]

    header = f"  {'Target':<20}" + "".join(f"{c:>14}" for c in cond_order)
    print(header)
    print(f"  {'-'*(20+14*len(cond_order))}")
    for tname in args.targets:
        row = f"  {tname:<20}"
        for cond in cond_order:
            vals = [all_results[tname][s].get(cond, {}).get('dnf_iou', float('nan'))
                    for s in range(args.seeds)]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                row += f"  {np.mean(vals):.4f}+{np.std(vals):.3f}"
            else:
                row += f"  {'--':>12}"
        print(row)

    # TABLE 2: Type selected per target
    print(f"\n\n{'='*74}")
    print(f"  TYPE SELECTION (find_type / find_both)")
    print(f"  Does the dominant type match the target?")
    print(f"{'='*74}")

    print(f"\n  {'Target':<20} {'Expected':<14} {'Recovered':<40} {'Match'}")
    print(f"  {'-'*80}")

    match_count = 0; total_count = 0

    for tname in args.targets:
        spec = SUITE[tname]
        is_single = len(spec['leaves']) == 1
        check_cond = 'find_type' if is_single else 'find_both'
        expected = spec.get('expected_winner') or str(spec.get('expected_types','?'))

        recovered_all = []
        for seed in range(args.seeds):
            td = all_results[tname][seed].get(check_cond,{}).get('type_dist',{})
            recovered_all.append(td)

        match = True
        for td in recovered_all:
            if is_single:
                if td:
                    dominant = max(td, key=td.get)
                    if dominant != spec['expected_winner']:
                        match = False
                else:
                    match = False
            else:
                for gt_type in spec.get('expected_types', {}).keys():
                    if gt_type not in td:
                        match = False

        total_count += 1
        if match: match_count += 1
        sym = 'Y' if match else 'X'
        print(f"  {tname:<20} {expected:<14} {str(recovered_all[0]):<40} {sym}")

    print(f"  {'-'*80}")
    print(f"  Match: {match_count}/{total_count}")

    # TABLE 3: Wrong-init recovery
    print(f"\n\n{'='*74}")
    print(f"  WRONG-INIT RECOVERY")
    print(f"{'='*74}")

    print(f"\n  {'Target':<20} {'ref':>10} {'wrong':>10} {'gap':>10} {'Status'}")
    print(f"  {'-'*60}")
    for tname in args.targets:
        is_single = len(SUITE[tname]['leaves']) == 1
        ref_cond = 'find_type' if is_single else 'find_both'
        ref = np.mean([all_results[tname][s].get(ref_cond,{}).get('dnf_iou',0)
                       for s in range(args.seeds)])
        wrg = np.mean([all_results[tname][s].get('wrong_type',{}).get('dnf_iou',0)
                       for s in range(args.seeds)])
        gap = wrg - ref
        st = 'recovered' if abs(gap)<0.01 else ('lost' if gap<-0.01 else 'better')
        print(f"  {tname:<20} {ref:>10.4f} {wrg:>10.4f} {gap:>+10.4f} {st}")

    # TABLE 4: Bc blend vs perfect discrete
    print(f"\n\n{'='*74}")
    print(f"  Bc BLEND vs PERFECT DISCRETE (Hard IoU)")
    print(f"{'='*74}")

    print(f"\n  {'Target':<20} {'perfect':>10} {'Bc+vol':>10} {'delta':>10}")
    print(f"  {'-'*55}")
    for tname in args.targets:
        is_single = len(SUITE[tname]['leaves']) == 1
        bc_cond = 'find_type' if is_single else 'find_both'
        perf = np.mean([all_results[tname][s].get('perfect',{}).get('dnf_iou',0)
                        for s in range(args.seeds)])
        bc = np.mean([all_results[tname][s].get(bc_cond,{}).get('dnf_iou',0)
                      for s in range(args.seeds)])
        print(f"  {tname:<20} {perf:>10.4f} {bc:>10.4f} {bc-perf:>+10.4f}")

    # VERDICT
    print(f"\n\n{'='*74}")
    print(f"  VERDICT")
    print(f"{'='*74}")

    single_ok = 0; single_n = 0
    for tname in args.targets:
        if len(SUITE[tname]['leaves']) != 1: continue
        single_n += 1
        spec = SUITE[tname]
        all_seeds_ok = True
        for seed in range(args.seeds):
            td = all_results[tname][seed].get('find_type',{}).get('type_dist',{})
            if not td or max(td, key=td.get) != spec['expected_winner']:
                all_seeds_ok = False
        if all_seeds_ok:
            single_ok += 1

    print(f"\n  Single-prim type accuracy: {single_ok}/{single_n} targets correct across all seeds")
    if single_ok == single_n:
        print(f"  >>> Bc+vol selects types FAIRLY -- it does NOT always pick blocks")
    elif single_ok >= single_n - 1:
        print(f"  >>> Bc+vol mostly fair, one target missed")
    else:
        print(f"  >>> Bc+vol has type bias -- check which targets fail")

    print(f"\n  Overall type match: {match_count}/{total_count}")
    print(f"\n  Output: {args.output}")

    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        ser = {}
        for t in all_results:
            ser[t] = {str(s): all_results[t][s] for s in all_results[t]}
        json.dump(ser, f, indent=2, default=str)

if __name__ == '__main__':
    main()
