# Reproducing the dissertation

The scripts here are the ones that produced the dissertation's results, reduced to the code those runs execute
(unused experimental code removed; the remaining code is unchanged). The library in `../wcl` is a cleaned version of
the same methods and is tested to give identical results (`../tests`).

Run the commands from the folder named in each section. "Data" refers to the deposited data set (see the
dissertation for its DOI); `<data>` is where you unpacked it.

## Chapter 3: inverse CSG (`reproduce/inverse_csg`)

### Main budget sweep (432 runs; Tables "top budget", "per-target winners", "budget sweep", "multi-hot vs one-hot", "factor model")

From the deposited per-run results (seconds):

```bash
python analysis/make_sweep_tables.py --data <data>/inverse_csg/csg_budget_siggraph_fixed
python analysis/make_factor_model.py --data <data>/inverse_csg/csg_budget_siggraph_fixed
```

Re-running the sweep itself (GPU recommended; one top-budget run of 30,000 epochs took about 50 minutes):

```bash
python csg_leaf_budget_ablation_stable_fixed.py --suite budget_siggraph \
    --meshes bunny fandisk rocker-arm --fetch-meshes --seeds 3 --epochs 30000 \
    --baseline-prims-list 2,3,4,6,8,10 --bc-prims-list 8,12,16,24,32,50 --groups-list 8,10,12,16,16,16 \
    --output csg_budget_siggraph_fixed
```

All other settings are the script defaults (evaluation every 200 epochs, inverse temperature annealed from 1 to 20,
multi-hot threshold 0.5). The solver core is `siggraph_csg_benchmark_v3.py`.

### Other Chapter 3 experiments

| Dissertation item | Script | Data folder |
|---|---|---|
| Primitive-type fairness (controlled targets) | `type_fairness_benchmark.py` | `benchmark_type_fairness/` |
| Controlled Boolean recovery (tree vs n-ary) | `controlled_nary_tree_benchmark.py` | `benchmark_controlled_nary_tree/`, `benchmark_controlled/` |
| Product-volume vs weighted sum, direct comparison | `experiment_product_vs_weightedsum.py` | `output_product_vs_weightedsum/` |
| Product-volume vs weighted sum, convergence (Bunny, Fandisk; n-ary, tree) | `convergence_benchmark.py`, `convergence_benchmark_download_wsum.py` | `benchmark_convergence/` |
| Gating sweep | `nary_benchmark_v3.py`, `nary_benchmark_v4.py` | `benchmark_v3/`, `benchmark_v4/` |
| Normalisation and type-aggregation ablation | `nary_benchmark_v7.py` | `benchmark_v7/` |
| Calibration diagnostics | `calibrated_field_ablation.py` | `calib_diag/`, `calib_bunny_smoke24/` |
| Type / operator / structural recovery | `recovery_benchmark.py` | `type_recovery_results.json` |
| Active-primitive capacity runs | `tree_bc_vol_active.py` | `benchmark_active_sweep/` |
| Responsibility, ownership, pair competition | `responsibility_field_fast.py`, `internal_node_responsibility_correct_full.py`, `internal_node_pair_competition.py`, `ownership_experiment.py`, `ownership_projection_experiment.py` | `responsibility_results_fast/`, `internal_node_*`, `ownership_*` |
| Auxiliary metric experiments | `metric_experiment_v2.py` | `metric_v2_results.json` |

`full_benchmark_with_levels.py` is the shared tree/n-ary benchmark module these scripts import. Run any script with
`--help` for its options. The two-dimensional GitHub-logo capacity experiment was run in an earlier workspace whose
code was not kept; only its reported numbers appear in the dissertation.

## Chapter 5: geometric dissection (`reproduce/dissection`)

Run from `reproduce/dissection`. Outputs go to `outputs/`.

### Two shapes (Duncan et al. remakes; Table "BACO diagnostics", comparison table)

1. **Lattice search** (Zhou and Wang 2012; about one hour per pair). The source is `experiments/zhou2d.cpp`; the
   driver compiles it (MSVC via `vcvars64.bat`, or pass a binary built with any C++17 compiler as `--zhou-bin`):

   ```bash
   python experiments/lattice2d.py --input-mode png --shape-a A.png --shape-b B.png --build-zhou \
       --zhou-N 400 --zhou-R 100 --zhou-Nb 30 --zhou-max-level 5 --seed 2012 --tag <pair>
   ```

   The per-level labels are written to `outputs/zhou_seeded_vector_solve/<pair>/zhou_run/level_<K>pcs_{A,B}.png`.
   These are the settings recorded in the dissertation runs' logs.

2. **BACO** on the K-piece level (defaults are the dissertation's settings):

   ```bash
   python experiments/exp358_zhou_duncan_common_boundary_optimizer.py --input-mode png \
       --shape-a A.png --shape-b B.png --labels-a level_6pcs_A.png --labels-b level_6pcs_B.png --tag <run>
   ```

3. **Tables**: `python thesis_tables/emit_latex_tables.py` (diagnostics table, reads the BACO run folders listed at
   the top of the script) and `python thesis_tables/make_comparison_metrics.py` (comparison with Duncan et al.;
   needs their paper's PDF, see `make_duncan_input_masks.py`).

The input silhouettes were extracted from the figures of Duncan et al. (2017) and are not redistributed.

### Three shapes and the FEEC-to-heart progression

`experiments/exp462_zhou_multitarget_baco_2d.py` (stages `prepare`, `build`, `search`, `baco`, `render`; the lattice
search is `experiments/zhou_multi.cpp`). The dissertation rows used K = 9 with the cyclic sweep for bear/cat/fish and
K = 12...20 for FEEC/heart; the clearance-seam variant used `--clearance 0.1 --prox-weight 40 --grow-rounds 0`.
These runs did not record their full command lines; their outputs are in the data deposit
(`dissection/2d_bearcatfish_multitarget/`, `dissection/2d_feec_heart_progression/`). The continuous alternatives and
the learned split policy are `exp463`...`exp466` (`--help` for options).

### Three dimensions (bunny to sphere; Tables "3D certificate" and "piece-count progression")

1. Lattice search on the 64^3 grid: `zhou3d/zhou3d.cpp` driven by `experiments/exp441_highres_bunny_sphere_piece_sweep.py`
   (N = 400, R = 100); `experiments/exp451_audit_zhou3d_labels.py` audits the labels.
2. Shared patch complex and certificate (exp. 453). For K = 4, 5, 6 use the lattice levels `level_4pcs`,
   `level_5pcs` and `result_6pcs` (in the data deposit) and:

   ```bash
   python experiments/exp453_shared_patch_complex_baco_3d.py \
       --labels-a <labels>_A --labels-b <labels>_B \
       --source-mesh bunny_watertight_aligned.obj --target-mesh sphere_aligned.obj \
       --align-scale --align-iters 15 --cert-dilate 2 --cert-slack 0.01 --cross-untangle --validate \
       --outer-iters 4 --untangle-iters 12 --lsqr-iter 1500 --lsqr-tol 1e-8 \
       --w-anchor 0.12 --w-closure 25 --w-exterior-a 4 --w-exterior-b 2.5 --w-fair 1 --w-seam 1 \
       --w-nopen 0 --out runs/thesis_k6
   ```

   `--w-nopen 0` matters: the non-penetration springs were added after the dissertation's runs and are on by default
   in this script (they are off by default in `python -m wcl.dissection.patch3d.pipeline`, which takes the same
   arguments). With these settings the six-piece run reproduces every number of the dissertation's certificate table.
3. Marching-cubes baseline (exp. 384): `experiments/exp384_zhou3d_common_boundary_optimizer.py` on the same labels.
