#!/usr/bin/env python3
"""
TreeBcVolActiveCSG: active-set training variant of TreeBcVolCSG.

Motivation (from responsibility field analysis):
  The gradient invariance proposition shows weighted sums cannot provide
  spatial shielding. Product union (Bc+vol) provides it, but the responsibility
  analysis shows residual global contamination: at any point x, many primitives
  receive nonzero gradient even though only 2-3 control the
  surface there.

  This class masks the leaf tensor before the tree reduction so that only the
  top-k most responsible primitives at each point receive gradient. The
  forward value is preserved exactly via a straight-through estimator, so at
  inference the model is numerically identical to plain Bc+vol. Only training
  gradients are modified.

Drop-in replacement for TreeBcVolCSG: same __init__ signature plus `active_k`.
Inherits structure_loss, get_stats, forward_hard_dnf, etc. from the parent.

Usage:
    from tree_bc_vol_active import TreeBcVolActiveCSG
    model = TreeBcVolActiveCSG(num_primitives=100, omega=10.0, active_k=4)
    # ... train as usual ...

    # Optional schedule:
    model.set_active_k(k_schedule(epoch, total_epochs, N=100))
"""

import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from full_benchmark_with_levels import TreeBcVolCSG


# Model
class TreeBcVolActiveCSG(TreeBcVolCSG):
    """TreeBcVolCSG with top-k active-set gradient masking at the leaves.

    Args (additional):
        active_k: int. Number of most-responsible primitives allowed to
            receive gradient at each point. k >= num_primitives disables
            masking and reduces to plain Bc+vol.
        eval_masked: if True, apply the active-set mask during eval too.
            Default False, so evaluation matches plain Bc+vol exactly.
    """

    def __init__(self, *args, active_k: int = 4, eval_masked: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_k = int(active_k)
        self._eval_masked = bool(eval_masked)
        self._last_active_frac = None
        self._last_rho_mean_top1 = None

    # Config
    def set_active_k(self, k: int):
        """Update the active-set size. Clamped to [1, num_primitives]."""
        self._current_k = max(1, min(int(k), self.num_primitives))

    @property
    def active_k(self) -> int:
        return self._current_k

    # Forward with active-set masking
    def forward(self, points, temperature: float = 1.0):
        # Leaves: [B, N]
        leaves = self.pool.get_all_occs(
            points, mode='bc_vol_norm', temperature=temperature
        )

        # Short-circuit: k>=N or eval (unless explicitly masked) -> baseline path.
        # Guarantees numerical identity to plain Bc+vol, so k=N is a clean sanity
        # check and inference is unchanged.
        if self._current_k >= leaves.shape[1] or (
            not self.training and not self._eval_masked
        ):
            self._last_active_frac = 1.0
            return self._reduce_tree(leaves, temperature=temperature)

        # Responsibility probe
        # ρ[b, i] = |∂S[b] / ∂leaves[b, i]|, computed on a detached copy so
        # the main computation graph is untouched. No create_graph: this is
        # a one-shot first-order probe.
        with torch.enable_grad():
            leaves_probe = leaves.detach().requires_grad_(True)
            pred_probe = self._reduce_tree(leaves_probe, temperature=temperature)
            (rho,) = torch.autograd.grad(
                pred_probe.sum(),
                leaves_probe,
                create_graph=False,
                retain_graph=False,
            )
            rho = rho.abs()

        # Top-k active set per point
        k = self._current_k
        _, topk_idx = rho.topk(k=k, dim=1)  # [B, k]
        active_mask = torch.zeros_like(rho)
        active_mask.scatter_(1, topk_idx, 1.0)  # [B, N]

        # Diagnostics (cheap, detached)
        with torch.no_grad():
            self._last_active_frac = float(active_mask.mean().item())
            top1 = rho.max(dim=1).values
            self._last_rho_mean_top1 = float(top1.mean().item())

        # Straight-through leaf mask
        # Forward value: leaves (unchanged). Gradient: flows only through
        # leaves where active_mask == 1.
        leaves_masked = leaves * active_mask + leaves.detach() * (1.0 - active_mask)

        return self._reduce_tree(leaves_masked, temperature=temperature)

    # Stats passthrough
    def get_stats(self):
        try:
            stats = super().get_stats()
        except AttributeError:
            stats = {}
        stats = dict(stats) if stats else {}
        stats['active_k'] = self._current_k
        if self._last_active_frac is not None:
            stats['active_frac'] = round(self._last_active_frac, 4)
        if self._last_rho_mean_top1 is not None:
            stats['rho_top1'] = round(self._last_rho_mean_top1, 4)
        return stats


# k schedules
def constant_k(k_final: int):
    """Constant k for the whole run."""
    return lambda epoch, total_epochs, N: k_final


def linear_decay_k(k_final: int = 4):
    """Start at N, decay linearly to k_final:
        k = max(k_final, N - epoch * (N - k_final) / total_epochs)
    """
    def sched(epoch, total_epochs, N):
        if total_epochs <= 0:
            return k_final
        return int(max(k_final, N - epoch * (N - k_final) / total_epochs))
    return sched


def warmup_then_active(warmup_epochs: int, k_final: int = 4):
    """Plain Bc+vol (k=N) for warmup_epochs, then drop to k_final."""
    def sched(epoch, total_epochs, N):
        return N if epoch < warmup_epochs else k_final
    return sched


# Sweep runner (uses convergence_benchmark.train_with_logging)
def _run_sweep():
    """Run a k-sweep on a single mesh across multiple seeds, reusing the
    train_with_logging / CSV / plotting infrastructure from
    convergence_benchmark.py.

    Each (k, seed) pair is an independent resumable run. Results land under
    --output/runs and --output/curves, and plots regenerate after each seed.
    """
    import argparse
    import time
    import json
    from datetime import datetime
    import numpy as np

    from nary_benchmark_v7 import MeshSDF, device
    from full_benchmark_with_levels import tree_surface_init, tree_spatial_init
    from convergence_benchmark import (
        train_with_logging, generate_plots,
        load_master_log, save_master_log,
        COLORS, LABELS,
    )

    p = argparse.ArgumentParser(description='Active-set k-sweep')
    p.add_argument('--mesh', type=str, default='./bunny.obj')
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--bc-prims', type=int, default=100)
    p.add_argument('--omega', type=float, default=10.0)
    p.add_argument('--samples', type=int, default=100000)
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--ks', type=int, nargs='+', default=[2, 4, 8, 16, 32, 100],
                   help='Active-set sizes to sweep. Include N (e.g. 100) as baseline.')
    p.add_argument('--schedule', type=str, default='constant',
                   choices=['constant', 'linear_decay', 'warmup'],
                   help='k schedule type. For sweep, "constant" is standard.')
    p.add_argument('--warmup', type=int, default=3000,
                   help='Warmup epochs for the "warmup" schedule.')
    p.add_argument('--output', type=str, default='./benchmark_active_sweep/')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()

    OUT = args.output
    os.makedirs(OUT, exist_ok=True)
    curves_dir = os.path.join(OUT, 'curves')
    os.makedirs(curves_dir, exist_ok=True)

    # Register per-k "methods" so generate_plots works unchanged
    method_names = [f'active_k{k}' for k in args.ks]
    # Inject labels / colors so plots are readable
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    for i, (k, name) in enumerate(zip(args.ks, method_names)):
        LABELS[name] = f'Active k={k}' + (' (baseline)' if k >= args.bc_prims else '')
        COLORS[name] = '#%02x%02x%02x' % tuple(
            int(255 * c) for c in cmap(i / max(len(args.ks) - 1, 1))[:3]
        )

    if args.plot_only:
        generate_plots(OUT, method_names, args.seeds, args.eval_every)
        return

    mesh_sdf = MeshSDF(args.mesh)
    target_func = lambda pts: torch.from_numpy(
        (mesh_sdf.query(pts.cpu().numpy()) < 0).astype(np.float32)
    ).to(device)

    reg = dict(
        sparsity_warmup=max(args.epochs // 20, 1500),
        sparsity_rampup=max(args.epochs // 12, 2000),
        ls=2e-4, lb=5e-5, lc=2e-4, lgs=1e-5,
        ba_start=args.epochs // 4,
        ba_end=int(args.epochs * 0.75),
        ba_max=5e-4,
    )

    master_log_path = os.path.join(OUT, 'convergence_log.json')
    master_log = load_master_log(master_log_path)

    print(f"\n{'='*74}")
    print(f"  ACTIVE-SET K-SWEEP")
    print(f"{'='*74}")
    print(f"  Mesh: {args.mesh}")
    print(f"  k values: {args.ks}  (N={args.bc_prims})")
    print(f"  Schedule: {args.schedule}")
    print(f"  Seeds: {args.seeds}, Epochs: {args.epochs}")
    print(f"  Output: {OUT}")
    print(f"  Device: {device}")
    print(f"{'='*74}\n")

    total_runs = len(args.ks) * args.seeds
    done = 0

    for seed in range(args.seeds):
        for k in args.ks:
            run_key = f'active_k{k}_seed{seed}'
            if run_key in master_log['completed']:
                print(f"  ✓ {run_key} already done, skipping")
                done += 1
                continue

            run_dir = os.path.join(OUT, 'runs', run_key)
            csv_path = os.path.join(curves_dir, f'active_k{k}_seed{seed}.csv')

            print(f"\n{'─'*74}")
            print(f"  {run_key}  ({done+1}/{total_runs})")
            print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─'*74}")

            master_log['in_progress'] = run_key
            save_master_log(master_log_path, master_log)

            torch.manual_seed(seed * 1000 + 42)
            np.random.seed(seed * 1000 + 42)

            model = TreeBcVolActiveCSG(
                num_primitives=args.bc_prims,
                omega=args.omega,
                active_k=k,
            ).to(device)
            tree_surface_init(model, mesh_sdf)
            tree_spatial_init(model, mesh_sdf)

            # The schedule is applied by a forward pre-hook on the model, which updates
            # _current_k based on the training epoch reported by the loop.
            # train_with_logging does not pass the epoch to the model, so
            # non-constant schedules patch forward.
            if args.schedule != 'constant':
                if args.schedule == 'linear_decay':
                    sched = linear_decay_k(k_final=k)
                else:
                    sched = warmup_then_active(args.warmup, k_final=k)
                # Monkey-patch: track epoch via a counter on the model.
                model._epoch_counter = 0
                model._total_epochs = args.epochs
                orig_forward = model.forward
                def _scheduled_forward(points, temperature=1.0,
                                       _m=model, _sched=sched, _of=orig_forward):
                    _m.set_active_k(_sched(_m._epoch_counter, _m._total_epochs,
                                           _m.num_primitives))
                    if _m.training:
                        _m._epoch_counter += 1
                    return _of(points, temperature=temperature)
                model.forward = _scheduled_forward

            t0 = time.time()
            state = train_with_logging(
                model, target_func, mesh_sdf, args.epochs, run_dir,
                csv_path=csv_path,
                eval_every=args.eval_every,
                save_every=args.save_every,
                num_samples=args.samples,
                resolution=args.resolution,
                use_temperature=True,
                temp_start=1.0, temp_end=20.0,
                desc=run_key,
                **reg,
            )
            elapsed = time.time() - t0
            print(f"\n  {run_key}: best_iou={state.best_iou:.4f} @ ep {state.best_epoch}, "
                  f"time={elapsed:.0f}s ({elapsed/3600:.1f}h)")

            master_log['completed'].append(run_key)
            master_log['in_progress'] = None
            save_master_log(master_log_path, master_log)
            done += 1

            print("\n  Updating plots...")
            generate_plots(OUT, method_names, args.seeds, args.eval_every)

    print(f"\n\n{'='*74}")
    print(f"  SWEEP COMPLETE ({done}/{total_runs})")
    print(f"{'='*74}")
    generate_plots(OUT, method_names, args.seeds, args.eval_every)


if __name__ == '__main__':
    _run_sweep()
