# Heavy-Tailed Flow Matching via Random Clocks (HTFM)

Official implementation of **Heavy-Tailed Flow Matching via Random Clocks**
(NeurIPS 2026). <!-- TODO: add paper / arXiv / OpenReview links -->

HTFM represents heavy-tailed source distributions as mixtures of
clock-conditioned Gaussian sources. Conditioned on a realised clock path the
source and the flow are Gaussian; marginalising over the clock yields a
Gaussian scale mixture that covers the Gaussian, α-stable and Student-t
families. The velocity field observes the realised clock through truncated
logsignature features, so one architecture and one training objective serve
all three families, and the tail heaviness of the generated samples is
calibrated by changing only the clock law (α for the α/2-stable subordinator,
ν for the inverse-Gamma Student-t clock).

The repository contains the model, the three benchmarks of the paper
(2D imbalanced α-stable mixtures, CIFAR10-LT, HRRR VIL weather fields), the
heavy-tailed baselines trained under the same backbones (LIM, DLPM / DLIM,
Gaussian flow matching) and the evaluation code (FID, precision–recall
$f_1^{\mathrm{pr}}$, kurtosis / skewness ratios and tail KS).

## Repository layout

```
run.py / eval.py / visualize_2d.py   training, evaluation and 2D plotting entry points
script_utils.py                     command-line flags and config overrides
htfm/
  methods/                          fm (HTFM), dlpm, lim, gmflow method builders
    flow_matching.py                clock-conditioned flow matching (training loss + sampling)
    clocks/                         alpha_stable / student_t / gaussian random clocks
    path/                           clock-path features (logsignature), path sampling, conditioning module
    fm_solvers/                     euler / heun / midpoint / rk4 / ab2 / dopri5 ODE solvers
    noise_schedule/, predictors/, loss_weighting/, loss_aggregation/
    LIM/                            vendored pieces of LIM (SDE, sampler, torchlevy, CUDA ops)
  models/                           UNet (CIFAR), NCSN++ (HRRR), MLP (2D)
  configs/
    toy2d/                          2D alpha-stable mixtures (Table 1, 2D columns)
    cifar10lt/                      CIFAR10-LT main results and baselines (Tables 1-2)
    cifar10lt_ablation/             signature-order, flow-type, NFE and solver ablations
    hrrr_vil/                       HRRR VIL (Table 1, HRRR columns)
    extra/                          GMFlow baseline configs (not in the paper)
bem/                                experiment framework (training / eval managers, datasets, metrics)
scripts/                            reproduction drivers, see scripts/README.md
tools/                              HRRR cache builder, FID and tail-metric evaluators
tests/                              unit tests (python -m pytest)
docs/                               config reference, logsignature notes
data/                               see data/README.md
```

## Installation

Python 3.11, PyTorch 2.5.1 with CUDA 12.1 is what the paper used.

```bash
conda create -n htfm python=3.11 -y && conda activate htfm
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest            # CPU unit tests, ~1 min
```

Notes

- The NCSN++ backbone used for HRRR JIT-compiles two small CUDA kernels on
  first use (`htfm/methods/LIM/op`). This needs `nvcc` (a CUDA toolkit matching
  your PyTorch build) and `ninja`. Without them the code falls back to a
  pure-PyTorch implementation with a warning; results are identical, training
  is slower.
- `iisignature` is an optional logsignature backend; the default `numeric`
  backend has no extra dependency.
- `--log` streams metrics to Weights & Biases (`wandb` is optional otherwise).

## Data

See [data/README.md](data/README.md). In short: the 2D mixtures are generated
on the fly, CIFAR-10 is downloaded by torchvision and the fixed CIFAR10-LT
subset (12,406 images, imbalance ratio 0.01) is read from
`data/splits/`, and the HRRR VIL cache is built once with
`tools/build_hrrr_cache.py`.

## Quick start

Every run is `run.py --config <family>/<name> --name <run_name> --method <fm|dlpm|lim|gmflow>`.
Checkpoints, evaluation JSONs and generated samples go to `models/<run_name>/`.

```bash
# 2D: HTFM-alpha (alpha = 1.8) on the alpha_D = 1.8 mixture, 300 epochs, checkpoints every 25 epochs from 100
python run.py --config toy2d/ch_sas18_fm_pc --name htfm_a18_sas18 --method fm \
    --epochs 300 --check 25 --check_start 100
# evaluate the final checkpoint: 24k samples, Euler ODE, NFE = 20
python eval.py --config toy2d/ch_sas18_fm_pc --name htfm_a18_sas18 --method fm \
    --epochs 300 --eval 300 --generate 24000 --reverse_steps 20 --solver euler
python visualize_2d.py --config toy2d/ch_sas18_fm_pc --name htfm_a18_sas18 --method fm --epochs 300

# CIFAR10-LT: HTFM-alpha, alpha = 1.7, logsignature level 1 (paper Table 1 / 2 HTFM row)
python run.py --config cifar10lt/FP13_a17_lev1 --name FP13_a17_lev1_s42 --method fm \
    --epochs 1300 --check 100 --set_seed 42
python tools/eval_fid_12k.py --config cifar10lt/FP13_a17_lev1 --name FP13_a17_lev1_s42 \
    --epoch 700 --reverse_steps 20 50 100

# HRRR VIL: HTFM-alpha, alpha = 1.7 (NCSN++, 1500 epochs)
python run.py --config hrrr_vil/HRRR_VIL_alpha17 --name HTFM_HRRR_a17 --method fm --epochs 1500 --check 50
```

Useful flags (all optional, they override the YAML): `--epochs`, `--check`
(checkpoint frequency), `--check_start`, `--set_seed`, `--resume`,
`--n_max_batch N` (smoke test), `--reverse_steps`, `--solver`, `--alpha`,
`--nu`, `--lr`, `--log --wandb_project …`. `script_utils.py` lists them all.

## Method names used in the paper

| Paper | `--method` | Config knobs |
|---|---|---|
| HTFM-G (Gaussian limit, $p=\infty$) | `fm` | `fm.clock.type: gaussian` |
| HTFM-α, tail degree $p_\alpha$ | `fm` | `fm.clock.type: alpha_stable`, `fm.alpha: 1.6 … 1.9` |
| HTFM-t, tail degree $p_\nu$ | `fm` | `fm.clock.type: student_t`, `fm.clock.nu: 1.7 / 2.0 / 3.0` |
| signature order $m$ | `fm` | `path_conditioning.path_signature.level: 1 / 2`; $m=0$ is `path_conditioning.enabled: false` (the `*_no_pc` configs) |
| source type (terminal vs area) | `fm` | `fm.noise_type: terminal` vs `forward_process` |
| flow type (straight-line, signature-linear VE, $\bar t$-linear) | `fm` | `fm.flow_type` (+ `fm.signature_linear_ve.lambda_*`) |
| LIM | `lim` | `lim.alpha` |
| DLPM / DLIM | `dlpm` | `dlpm.alpha`; DLIM = `eval.dlpm.deterministic: true, dlim_eta: 0.0` (`--deterministic`) |
| t-Flow | — | external re-implementation of Pandey et al. (2024), not part of this repository |
| GMFlow (not in the paper) | `gmflow` | `htfm/configs/extra/` |

The full list of YAML fields is in [docs/config_reference.md](docs/config_reference.md).

## Reproducing the paper

Each table maps to one config family and one driver script; the drivers only
orchestrate `run.py` / `eval.py` / `tools/*.py` over GPUs, seeds and
checkpoints, so every number can also be reproduced by hand with the commands
above. Protocol details are in `scripts/README.md`.

| Paper | Configs (`htfm/configs/…`) | Driver |
|---|---|---|
| Table 1, 2D columns ($f_1^{\mathrm{pr}}$, $\alpha_D\in\{1.5,1.8\}$) | `toy2d/ch_sas{15,18}_*` | `scripts/toy2d_suite.py` |
| Table 1 CIFAR column and Table 2 HTFM rows (FID vs NFE, 5 seeds) | `cifar10lt/FP13_*` | `scripts/cifar10lt_train.py`, `scripts/cifar10lt_eval.py` |
| Table 2 baselines LIM / DLPM / DLIM | `cifar10lt/FP9_lim_a*_clip`, `cifar10lt/FP13_*` with `--method dlpm`, `cifar10lt/FP12_dlpm_a2` | `scripts/cifar10lt_train.py --baselines`, `scripts/cifar10lt_eval.py` |
| Table 1 HRRR columns (KR / SR / KS) | `hrrr_vil/HRRR_VIL_{alpha16,alpha17,alpha18,vanilla,nu2_pc_concat,nu3_pc_concat}` | `scripts/hrrr_tail_sweep.py` |
| Signature-order ablation ($m=0,1,2$) | `cifar10lt_ablation/FP11_*` | `scripts/cifar10lt_eval.py` |
| Flow-type, NFE and ODE-solver ablations | `cifar10lt_ablation/FP8_*` | `scripts/cifar10lt_eval.py`, `tools/eval_fid_12k.py --solver …` |

Protocol notes

- CIFAR10-LT runs use the schedule written in the configs (`run.epochs: 2000`,
  step-LR counted in optimiser steps) but were stopped at epoch 1300 for all
  methods, so pass `--epochs 1300`. FID is 12,406 generated vs the 12,406-image
  LT training split, EMA weights, Euler sampler unless the solver ablation says
  otherwise. Tables report the mean over seeds `42 1234 0 4321 100`.
- HRRR tail statistics use 20,000 generated fields vs the 2021 test year,
  Heun with 25 reverse steps (NFE 50). The reporting epoch per model is chosen
  by the rank sum of (KR, SR, KS) over the checkpoint sweep
  (`tools/hrrr_pick_best.py`).
- 2D runs train for 300 epochs and report the best checkpoint over
  epochs 100–300 and the NFE grid listed in `scripts/toy2d_suite.py`
  (24,000 samples vs a 24,000-sample held-out reference set).

## Pretrained checkpoints

<!-- TODO: decide whether to publish the FP13 / HRRR checkpoints (e.g. Hugging Face Hub) and link them here -->

## Citation

```bibtex
@inproceedings{htfm2026,
  title     = {Heavy-Tailed Flow Matching via Random Clocks},
  author    = {TODO},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements and license

The code grew out of the [DLPM](https://github.com/darioShar/DLPM) /
[BEM](https://github.com/darioShar/bem) code base and vendors components of
LIM, score_sde, improved-diffusion and GMFlow; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact files and
their licenses. <!-- TODO: LICENSE for this repository is still to be decided (bem/ is AGPL-3.0 upstream) -->
