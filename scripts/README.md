# Reproduction scripts

Orchestrators that drive `run.py` / `eval.py` / `tools/*.py` over several GPUs. All of them are
restart-safe (finished runs or existing result files are skipped), take `--gpus` (default: every
visible GPU) and `--lanes` (jobs per GPU), and accept `--dry-run` to print the commands only.
Run them from anywhere; they `chdir` to the repository root, so checkpoints go to `models/`,
logs to `logs/` and results to `results/`.

| Script | Paper | Replaces (internal repo) |
|---|---|---|
| `toy2d_suite.py` | Tab. main-results (2D columns), Fig. viz-toy / viz-sas | `run_corner_heavy_v4*.py` |
| `cifar10lt_train.py` | Tab. main-results (CIFAR column), Tab. cifar10lt-main-results | `run_fp13.py`, `finish_baselines.py`, `cap_baselines_1300.py` |
| `cifar10lt_eval.py` + `tools/eval_fid_12k.py` | same tables + Tab. nfe-ablation / solver-ablation | `run_fp13_eval.py`, `run_baseline_eval.py`, `tools/eval_{fp10_12k,baseline_12k,gmflow_12k,fp12_dlpm,table7}.py` |
| `hrrr_tail_sweep.py` | Tab. main-results (HRRR columns) | `run_hrrr_batch3_sweep.py`, `run_htfm_hrrr_sweep.py` |

## `toy2d_suite.py` — 2D imbalanced α-stable mixture

Trains DLPM, HTFM-α (with and without clock conditioning) and HTFM-t on the two 9-component
S-α-S grid mixtures (`htfm/configs/toy2d`, α_D = 1.8 and 1.5), 300 epochs, seed 42, then evaluates
24,000 samples per checkpoint at several NFE budgets and collects `f_1_pr` (the paper metric) and
the 1-D Wasserstein distance into `results/toy2d/metrics.csv`. α = 2.0 of `fm_no_pc` is HTFM-G.

```bash
python scripts/toy2d_suite.py --gpus 0 1 2 3 --yes
python scripts/toy2d_suite.py --only-variant st_fm_no_pc --only-nu 1.7 --eval_epochs 100 200 300 --yes
```

## `cifar10lt_train.py` — CIFAR10-LT training

Queues the 8 HTFM variants × 5 seeds (`FP13_<variant>_s<seed>`) or, with `--baselines`, the LIM /
DLPM / GMFlow baselines, over `gpus × lanes` slots with a same-GPU launch stagger. Every reported
CIFAR10-LT model was trained to epoch 1300 (`--train_to`, see the docstring for why the YAMLs say
2000); GMFlow runs to its config budget.

```bash
python scripts/cifar10lt_train.py --gpus 0 1 2 3 4 5 6 7 --lanes 2
python scripts/cifar10lt_train.py --baselines --gpus 0 1 2 3
```

## `cifar10lt_eval.py` — CIFAR10-LT FID grid

Calls `tools/eval_fid_12k.py` for every (run, epoch, NFE) cell: 12,406 EMA samples vs the full
12,406-image long-tailed split, Inception FID + precision / recall / density / coverage. HTFM uses
the Euler ODE (NFE = reverse steps); LIM / DLPM / GMFlow their stochastic samplers; `--dlim` adds the
deterministic (η = 0) DLPM pass. Results: `results/fid/<name>_ep<E>_rs<R>[...].json`.

```bash
python scripts/cifar10lt_eval.py --gpus 0 1 2 3                 # HTFM variants, all seeds
python scripts/cifar10lt_eval.py --baselines --dlim --gpus 4 5
# single cells / other solvers (solver ablation):
python tools/eval_fid_12k.py --config cifar10lt_ablation/FP8_fm_a17_la025_lbn025_lev1 \
    --name FP8_a17_lbn025_lev1 --epoch 700 --solver heun --reverse_steps 10 25 50 --gpu 0
```

## `hrrr_tail_sweep.py` — HRRR-VIL tail statistics

For each HRRR run and checkpoint epoch, generates 20,000 VIL fields (Heun, 25 steps) and computes
KR / SR / KS (q = 0.999) against the 2021 test year with `tools/eval_hrrr_tails.py`, then picks each
run's reporting epoch by the KR+SR+KS rank-sum (ties → KS), as in `tools/hrrr_pick_best.py`.
Per-cell JSONs live next to the checkpoints; the summary is `results/hrrr/tail_sweep_summary.json`.

```bash
python scripts/hrrr_tail_sweep.py --gpus 0 1 2 3 --epochs 500 600 700 800 900 1000 1100 1200
python scripts/hrrr_tail_sweep.py --summary-only
```
