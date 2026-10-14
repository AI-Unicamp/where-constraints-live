#!/usr/bin/env python3
"""
Convergence benchmark: IoU vs epoch for the n-ary DNF and the pairwise tree.

Runs 4 methods x K seeds on a mesh target (default: bunny).
Each (method, seed) is an independent run with its own checkpoint.
Can be stopped and resumed at any time.

Outputs:
  convergence_log.json   -  master log of all completed runs
  curves/                -  per-run IoU curves (CSV)
  plots/                 -  convergence plots (updated after each seed)

Usage:
  python convergence_benchmark.py --mesh ./bunny.obj
  python convergence_benchmark.py --mesh ./bunny.obj --seeds 5 --epochs 60000
  python convergence_benchmark.py --mesh ./bunny.obj --methods N_nary_bc_vol TN_tree_bc_vol
  python convergence_benchmark.py --mesh ./bunny.obj --plot-only
"""

import os, sys, time, json, argparse, math, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# mcubes fallback
import importlib.util, types as _types
if importlib.util.find_spec('mcubes') is None:
    from skimage import measure
    mcubes_stub = _types.ModuleType('mcubes')
    def _mc(vol, iso):
        v, f, _, _ = measure.marching_cubes(vol, level=iso)
        return v, f.astype(np.int32)
    def _export(vertices, triangles, filepath):
        with open(filepath, 'w') as fh:
            for x, y, z in vertices: fh.write(f'v {x} {y} {z}\n')
            for a, b, c in triangles: fh.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')
    mcubes_stub.marching_cubes = _mc; mcubes_stub.export_obj = _export
    sys.modules['mcubes'] = mcubes_stub

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)

from nary_benchmark_v7 import MeshSDF, sample_3D_grid, compute_iou, batched_forward, BaselineCSG, surface_init, spatial_group_init, device, TState, CkptMgr
from full_benchmark_with_levels import (
    NaryBcVolCSG, TreeDiscreteCSG, TreeBcVolCSG,
    tree_surface_init, tree_spatial_init,
)


# TRAINING WITH DENSE LOGGING
def train_with_logging(
    model, target_func, mesh_sdf, epochs, output_folder,
    csv_path, eval_every=200, save_every=2000,
    lr=1e-3, num_samples=100000, resample_every=10,
    sparsity_warmup=3000, sparsity_rampup=5000,
    ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
    ba_start=15000, ba_end=40000, ba_max=5e-4,
    resolution=128, use_temperature=False,
    temp_start=1.0, temp_end=20.0, desc="Training",
):
    """Train with IoU logged to CSV every eval_every epochs. Fully resumable."""
    os.makedirs(output_folder, exist_ok=True)
    ckpt = CkptMgr(output_folder)
    loss_fn = nn.MSELoss()
    eval_grid = sample_3D_grid(resolution)
    eval_gt = target_func(eval_grid)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    state = None
    state = ckpt.load(model, optimizer)
    if state and state.epoch >= epochs:
        print(f"  Already done ({state.epoch} ep, best={state.best_iou:.4f})")
        ckpt.load_best(model)
        return state

    start_epoch = 0
    if state:
        start_epoch = state.epoch
        print(f"  Resuming from epoch {start_epoch}, best={state.best_iou:.4f}")
    else:
        state = TState()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01, last_epoch=start_epoch - 1)

    # Load existing CSV rows to avoid duplicating
    existing_epochs = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row: existing_epochs.add(int(row[0]))

    # Open CSV in append mode
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not existing_epochs:
        csv_writer.writerow(['epoch', 'soft_iou', 'hard_iou', 'mse', 'loss',
                             'prims_used', 'groups_active', 'best_iou', 'wall_s'])

    t0 = time.time()
    P, gt = None, None

    from tqdm import tqdm
    pbar = tqdm(range(state.epoch, epochs), initial=state.epoch, total=epochs,
                desc=f"  {desc}", leave=False)

    for epoch in pbar:
        progress = epoch / max(epochs - 1, 1)
        temperature = temp_start + (temp_end - temp_start) * progress if use_temperature else 1.0

        if epoch % resample_every == 0 or P is None:
            if mesh_sdf is not None:
                pts, occ = mesh_sdf.sample_points(num_samples)
                P = torch.from_numpy(pts).to(device)
                gt = torch.from_numpy(occ).to(device)
            else:
                P = torch.rand(num_samples, 3, device=device) * 2 - 1
                gt = target_func(P)

        pred = model(P, temperature=temperature)
        recon = loss_fn(pred, gt)

        if epoch < sparsity_warmup: sw = 0.0
        elif epoch < sparsity_warmup + sparsity_rampup:
            sw = (epoch - sparsity_warmup) / sparsity_rampup
        else: sw = 1.0

        if epoch < ba_start: bm = 1.0
        elif epoch < ba_end:
            bm = 1.0 + (epoch - ba_start) / (ba_end - ba_start) * (ba_max / max(lb, 1e-10) - 1.0)
        else: bm = ba_max / max(lb, 1e-10)

        struct = model.structure_loss(ls=ls*sw, lb=lb*bm*sw, lc=lc*sw, lgs=lgs*sw,
                                       temperature=temperature)
        loss = recon + struct

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        state.epoch = epoch + 1
        state.losses.append(loss.item())

        if (epoch + 1) % eval_every == 0:
            eval_temp = temp_end if use_temperature else 1.0
            with torch.no_grad():
                pf = batched_forward(model, eval_grid, temperature=eval_temp)
                soft_iou = compute_iou(pf, eval_gt)
                mse = ((pf - eval_gt) ** 2).mean().item()
                # Hard IoU
                if hasattr(model, 'forward_hard_dnf'):
                    dp = torch.cat([model.forward_hard_dnf(b)
                                    for b in torch.split(eval_grid, 32**3)], 0)
                    hard_iou = compute_iou(dp, eval_gt)
                else:
                    hard_iou = soft_iou

            state.ious.append(soft_iou)
            if soft_iou > state.best_iou:
                state.best_iou = soft_iou
                state.best_epoch = epoch + 1
                torch.save(model.state_dict(), ckpt.best)

            s = model.get_stats()
            pbar.set_postfix(
                IoU=f'{soft_iou:.4f}', hard=f'{hard_iou:.4f}',
                best=f'{state.best_iou:.4f}',
                p=s.get('prims_used', '?'), g=s.get('groups_active', '?'))

            # Write to CSV (skip if already logged from previous run)
            ep = epoch + 1
            if ep not in existing_epochs:
                wall = time.time() - t0
                csv_writer.writerow([
                    ep, f'{soft_iou:.6f}', f'{hard_iou:.6f}', f'{mse:.8f}',
                    f'{loss.item():.6f}',
                    s.get('prims_used', 0), s.get('groups_active', 0),
                    f'{state.best_iou:.6f}', f'{wall:.1f}'
                ])
                csv_file.flush()

        if (epoch + 1) % save_every == 0:
            ckpt.save(model, optimizer, state)

    ckpt.save(model, optimizer, state)
    ckpt.load_best(model)
    csv_file.close()
    return state


# METHOD DEFINITIONS
def get_methods(args):
    """Return dict of method_name -> (factory, use_temp, init_fn)."""
    nd = args.baseline_prims * 5
    nb = args.bc_prims

    methods = {}

    methods['A_nary_discrete'] = dict(
        factory=lambda: BaselineCSG(
            prims_per_type=args.baseline_prims,
            num_groups=args.groups).to(device),
        use_temp=False,
        init=lambda m, ms: (surface_init(m, ms), spatial_group_init(m, ms)) if ms else None,
    )

    methods['N_nary_bc_vol'] = dict(
        factory=lambda: NaryBcVolCSG(
            num_primitives=nb, num_groups=args.groups,
            omega=args.omega).to(device),
        use_temp=True,
        init=lambda m, ms: (surface_init(m, ms), spatial_group_init(m, ms)) if ms else None,
    )

    methods['TA_tree_discrete'] = dict(
        factory=lambda: TreeDiscreteCSG(
            prims_per_type=args.baseline_prims,
            omega=args.omega).to(device),
        use_temp=False,
        init=lambda m, ms: (tree_surface_init(m, ms), tree_spatial_init(m, ms)) if ms else None,
    )

    methods['TN_tree_bc_vol'] = dict(
        factory=lambda: TreeBcVolCSG(
            num_primitives=nb, omega=args.omega).to(device),
        use_temp=True,
        init=lambda m, ms: (tree_surface_init(m, ms), tree_spatial_init(m, ms)) if ms else None,
    )

    return methods


# PLOTTING
COLORS = {
    'A_nary_discrete': '#FF9800',
    'N_nary_bc_vol': '#E65100',
    'TA_tree_discrete': '#42A5F5',
    'TN_tree_bc_vol': '#1565C0',
}
LABELS = {
    'A_nary_discrete': 'N-ary Discrete',
    'N_nary_bc_vol': 'N-ary Bc+vol',
    'TA_tree_discrete': 'Tree Discrete',
    'TN_tree_bc_vol': 'Tree Bc+vol',
}


def load_curve(csv_path):
    """Load a CSV curve -> (epochs, soft_iou, hard_iou, best_iou)."""
    epochs, soft, hard, best = [], [], [], []
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            soft.append(float(row['soft_iou']))
            hard.append(float(row['hard_iou']))
            best.append(float(row['best_iou']))
    if not epochs:
        return None
    return np.array(epochs), np.array(soft), np.array(hard), np.array(best)


def generate_plots(out_dir, methods_to_plot, seeds, eval_every):
    """Generate convergence plots from all available CSV data."""
    curves_dir = os.path.join(out_dir, 'curves')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Collect all data
    all_data = {}  # method -> list of (epochs, soft, hard, best) per seed
    for method in methods_to_plot:
        all_data[method] = []
        for seed in range(seeds):
            csv_path = os.path.join(curves_dir, f'{method}_seed{seed}.csv')
            curve = load_curve(csv_path)
            if curve is not None:
                all_data[method].append(curve)

    # Plot 1: Soft IoU mean ± std across seeds
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric_idx, (metric_name, metric_label) in enumerate([
        ('soft', 'Soft IoU'), ('hard', 'Hard IoU')
    ]):
        ax = axes[metric_idx]
        for method in methods_to_plot:
            if not all_data[method]:
                continue
            # Align curves to common epoch grid
            min_len = min(len(c[0]) for c in all_data[method])
            epochs = all_data[method][0][0][:min_len]

            if metric_name == 'soft':
                vals = np.array([c[1][:min_len] for c in all_data[method]])
            else:
                vals = np.array([c[2][:min_len] for c in all_data[method]])

            mean = vals.mean(axis=0)
            color = COLORS.get(method, '#999999')
            label = LABELS.get(method, method)

            if len(all_data[method]) > 1:
                std = vals.std(axis=0)
                ax.fill_between(epochs, mean - std, mean + std,
                                alpha=0.15, color=color)
                ax.plot(epochs, mean, color=color, label=f'{label} (n={len(all_data[method])})',
                        linewidth=2)
            else:
                ax.plot(epochs, mean, color=color, label=f'{label} (n=1)',
                        linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} Convergence')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        # Set y-axis to show detail
        all_vals = []
        for method in methods_to_plot:
            for c in all_data[method]:
                idx = 1 if metric_name == 'soft' else 2
                all_vals.extend(c[idx].tolist())
        if all_vals:
            lo = max(min(all_vals) - 0.02, 0)
            hi = min(max(all_vals) + 0.01, 1.0)
            ax.set_ylim(lo, hi)

    plt.suptitle('Convergence Benchmark: N-ary DNF vs Pairwise Tree (Bunny)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'convergence_mean.png'), dpi=150)
    plt.close()

    # Plot 2: All individual seed curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for metric_idx, (mi, metric_label) in enumerate([(1, 'Soft IoU'), (2, 'Hard IoU')]):
        ax = axes[metric_idx]
        for method in methods_to_plot:
            color = COLORS.get(method, '#999999')
            label_base = LABELS.get(method, method)
            for si, curve in enumerate(all_data[method]):
                lbl = f'{label_base}' if si == 0 else None
                ax.plot(curve[0], curve[mi], color=color, alpha=0.4,
                        linewidth=1, label=lbl)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} — All Seeds')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'convergence_all_seeds.png'), dpi=150)
    plt.close()

    # Plot 3: Best IoU envelope per method
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods_to_plot:
        if not all_data[method]:
            continue
        min_len = min(len(c[0]) for c in all_data[method])
        epochs = all_data[method][0][0][:min_len]
        bests = np.array([c[3][:min_len] for c in all_data[method]])
        mean_best = bests.mean(axis=0)
        color = COLORS.get(method, '#999999')
        label = LABELS.get(method, method)
        ax.plot(epochs, mean_best, color=color, label=label, linewidth=2)
        if len(all_data[method]) > 1:
            ax.fill_between(epochs, bests.min(axis=0), bests.max(axis=0),
                            alpha=0.1, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best IoU (running max)')
    ax.set_title('Best IoU Envelope — When does each method peak?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_iou_envelope.png'), dpi=150)
    plt.close()

    # Summary table
    summary = {}
    for method in methods_to_plot:
        if not all_data[method]:
            continue
        final_soft = [c[1][-1] for c in all_data[method]]
        final_hard = [c[2][-1] for c in all_data[method]]
        best_soft = [c[3][-1] for c in all_data[method]]
        summary[method] = {
            'n_seeds': len(all_data[method]),
            'final_soft_mean': float(np.mean(final_soft)),
            'final_soft_std': float(np.std(final_soft)),
            'final_hard_mean': float(np.mean(final_hard)),
            'final_hard_std': float(np.std(final_hard)),
            'best_soft_mean': float(np.mean(best_soft)),
            'best_soft_std': float(np.std(best_soft)),
            'max_epoch': int(all_data[method][0][0][-1]),
        }

    with open(os.path.join(plots_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  CONVERGENCE SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Method':<22} {'Seeds':>5} {'Final Soft':>12} {'Final Hard':>12} {'Best Soft':>12}")
    print(f"  {'-'*65}")
    for m, s in summary.items():
        label = LABELS.get(m, m)
        print(f"  {label:<22} {s['n_seeds']:>5} "
              f"{s['final_soft_mean']:.4f}±{s['final_soft_std']:.4f} "
              f"{s['final_hard_mean']:.4f}±{s['final_hard_std']:.4f} "
              f"{s['best_soft_mean']:.4f}±{s['best_soft_std']:.4f}")
    print(f"  {'-'*65}")
    print(f"  Plots saved to: {plots_dir}/")

    return summary


# MASTER LOG
def load_master_log(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'completed': [], 'in_progress': None}


def save_master_log(path, log):
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)


# MAIN
def main():
    parser = argparse.ArgumentParser(description='Convergence benchmark')

    parser.add_argument('--mesh', type=str, default='./bunny.obj')
    parser.add_argument('--epochs', type=int, default=60000,
                        help='Epochs per run (default: 60k for long convergence)')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--eval-every', type=int, default=200,
                        help='Evaluate IoU every N epochs')
    parser.add_argument('--save-every', type=int, default=2000)

    parser.add_argument('--baseline-prims', type=int, default=20,
                        help='Prims per type for discrete (x5 total)')
    parser.add_argument('--bc-prims', type=int, default=100)
    parser.add_argument('--groups', type=int, default=20)
    parser.add_argument('--omega', type=float, default=10.0)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--resolution', type=int, default=128)

    parser.add_argument('--methods', nargs='+', default=None,
                        help='Which methods to run (default: all 4)')
    parser.add_argument('--output', type=str, default='./benchmark_convergence/')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate plots from existing CSV data')

    args = parser.parse_args()

    OUT = args.output
    os.makedirs(OUT, exist_ok=True)
    curves_dir = os.path.join(OUT, 'curves')
    os.makedirs(curves_dir, exist_ok=True)

    all_methods = get_methods(args)
    methods_to_run = args.methods if args.methods else list(all_methods.keys())

    # Validate method names
    for m in methods_to_run:
        if m not in all_methods:
            print(f"Unknown method: {m}. Available: {list(all_methods.keys())}")
            return

    # Plot only mode
    if args.plot_only:
        generate_plots(OUT, methods_to_run, args.seeds, args.eval_every)
        return

    # Load mesh
    mesh_sdf = MeshSDF(args.mesh)
    target_func = lambda p: torch.from_numpy(
        (mesh_sdf.query(p.cpu().numpy()) < 0).astype(np.float32)).to(device)

    master_log_path = os.path.join(OUT, 'convergence_log.json')
    master_log = load_master_log(master_log_path)

    # Regularization
    reg = dict(
        sparsity_warmup=max(args.epochs // 20, 1500),
        sparsity_rampup=max(args.epochs // 12, 2000),
        ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
        ba_start=args.epochs // 4,
        ba_end=int(args.epochs * 0.75),
        ba_max=5e-4,
    )

    print(f"\n{'='*74}")
    print(f"  CONVERGENCE BENCHMARK")
    print(f"{'='*74}")
    print(f"  Mesh: {args.mesh}")
    print(f"  Methods: {methods_to_run}")
    print(f"  Seeds: {args.seeds}, Epochs: {args.epochs}")
    print(f"  Discrete: {args.baseline_prims}x5={args.baseline_prims*5} prims, "
          f"Bc: {args.bc_prims} prims, Groups: {args.groups}")
    print(f"  Eval every: {args.eval_every}, Save every: {args.save_every}")
    print(f"  Output: {OUT}")
    print(f"  Device: {device}")
    print(f"{'='*74}\n")

    total_runs = len(methods_to_run) * args.seeds
    completed = 0

    # Run seed by seed
    for seed in range(args.seeds):
        for method_name in methods_to_run:
            run_key = f"{method_name}_seed{seed}"

            # Skip if already completed
            if run_key in master_log['completed']:
                print(f"\n  ✓ {run_key} already completed, skipping")
                completed += 1
                continue

            mdef = all_methods[method_name]
            run_dir = os.path.join(OUT, 'runs', run_key)
            csv_path = os.path.join(curves_dir, f'{run_key}.csv')

            print(f"\n{'─'*74}")
            print(f"  {run_key}  ({completed+1}/{total_runs})")
            print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─'*74}")

            # Mark in-progress
            master_log['in_progress'] = run_key
            save_master_log(master_log_path, master_log)

            # Seed
            torch.manual_seed(seed * 1000 + 42)
            np.random.seed(seed * 1000 + 42)

            # Create model
            model = mdef['factory']()
            mdef['init'](model, mesh_sdf)

            # Train
            t0 = time.time()
            state = train_with_logging(
                model, target_func, mesh_sdf, args.epochs, run_dir,
                csv_path=csv_path,
                eval_every=args.eval_every,
                save_every=args.save_every,
                num_samples=args.samples,
                resolution=args.resolution,
                use_temperature=mdef['use_temp'],
                temp_start=1.0, temp_end=20.0,
                desc=run_key,
                **reg,
            )
            elapsed = time.time() - t0

            # Final eval
            final_stats = model.get_stats() if hasattr(model, 'get_stats') else {}
            print(f"\n  {run_key}: best_iou={state.best_iou:.4f} @ ep {state.best_epoch}, "
                  f"time={elapsed:.0f}s ({elapsed/3600:.1f}h)")
            if 'type_dist' in final_stats:
                print(f"  Types: {final_stats['type_dist']}")

            # Mark completed
            master_log['completed'].append(run_key)
            master_log['in_progress'] = None
            save_master_log(master_log_path, master_log)
            completed += 1

            # Regenerate plots after each seed completes
            print(f"\n  Updating plots...")
            generate_plots(OUT, methods_to_run, args.seeds, args.eval_every)

    # Final summary
    print(f"\n\n{'='*74}")
    print(f"  ALL RUNS COMPLETE ({completed}/{total_runs})")
    print(f"{'='*74}")
    generate_plots(OUT, methods_to_run, args.seeds, args.eval_every)
    print("\nDone!")


if __name__ == '__main__':
    main()
