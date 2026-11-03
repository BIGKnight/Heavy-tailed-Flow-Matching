# Heavy-Tailed Flow Matching via Random Clocks (HTFM)

Official implementation of **Heavy-Tailed Flow Matching via Random Clocks**
(NeurIPS 2026) by Zhouhao Yang, Yezhen Wang, Kenji Kawaguchi, Vladimir Braverman
and Haoyang Cao.

HTFM represents heavy-tailed source distributions as mixtures of
clock-conditioned Gaussian sources. Conditioned on a realised clock path the
source and the flow are Gaussian; marginalising over the clock yields a
Gaussian scale mixture that covers the Gaussian, α-stable and Student-t
families. The velocity field observes the realised clock through truncated
logsignature features, so one architecture and one training objective serve
all three families, and the tail heaviness of the generated samples is
calibrated by changing only the clock law (α for the α/2-stable subordinator,
ν for the inverse-Gamma Student-t clock).

This repository contains the model, the training / evaluation entry points,
the configs of the three benchmarks used in the paper (2D imbalanced α-stable
mixtures, CIFAR10-LT, HRRR VIL weather fields) and the heavy-tailed baselines
trained under the same backbones (LIM, DLPM / DLIM, Gaussian flow matching).

## Repository layout

```
run.py / eval.py / visualize_2d.py   training, evaluation and 2D plotting entry points
script_utils.py                     command-line flags and config overrides
htfm/
  methods/                          fm (HTFM), dlpm, lim method builders
    flow_matching.py                clock-conditioned flow matching (training loss + sampling)
    clocks/                         alpha_stable / student_t / gaussian random clocks
    path/                           clock-path features (logsignature), path sampling, conditioning module
    fm_solvers/                     euler / heun / midpoint / rk4 / ab2 / dopri5 ODE solvers
    noise_schedule/, predictors/, loss_weighting/, loss_aggregation/
    LIM/                            vendored pieces of LIM (SDE, sampler, torchlevy, CUDA ops)
  models/                           UNet (CIFAR), NCSN++ (HRRR), MLP (2D)
  configs/
    toy2d/                          2D alpha-stable mixtures and the classic 2D toys
    cifar10lt/                      CIFAR10-LT: HTFM variants and the LIM / DLPM baselines
    hrrr_vil/                       HRRR VIL: HTFM variants and the LIM / DLPM baselines
bem/                                experiment framework (training / eval managers, datasets, metrics)
tools/                              HRRR cache builder and HRRR tail-metric evaluator
tests/                              unit tests (python -m pytest)
docs/config_reference.md            every YAML field explained
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

Every run is `run.py --config <family>/<name> --name <run_name> --method <fm|dlpm|lim>`.
Checkpoints, evaluation results and generated samples go to `models/<run_name>/`.

```bash
# 2D: HTFM-alpha (alpha = 1.8) on the alpha_D = 1.8 mixture, 300 epochs, checkpoints every 25 epochs from 100
python run.py --config toy2d/ch_sas18_fm_pc --name htfm_a18_sas18 --method fm \
    --epochs 300 --check 25 --check_start 100
# evaluate the final checkpoint: 24k samples, Euler ODE, NFE = 20 (precision-recall f1, Wasserstein)
python eval.py --config toy2d/ch_sas18_fm_pc --name htfm_a18_sas18 --method fm \
    --epochs 300 --eval 300 --generate 24000 --reverse_steps 20 --solver euler
python visualize_2d.py --config toy2d/ch_sas18_fm_pc --name htfm_a18_sas18 --method fm --epochs 300
# classic 2D toys (checkerboard, two_moons, swiss_roll, rings, olympic_rings) with a chosen clock
python run.py --config toy2d/toys_fm --name htfm_checkerboard --method fm --dataset checkerboard --alpha 1.8 --epochs 100 --check 100

# CIFAR10-LT: HTFM-alpha, alpha = 1.7, logsignature level 1
python run.py --config cifar10lt/FP13_a17_lev1 --name FP13_a17_lev1_s42 --method fm \
    --epochs 1300 --check 100 --set_seed 42
# FID / precision / recall of the EMA model at epoch 700, 12,406 samples, Euler with 100 steps
python eval.py --config cifar10lt/FP13_a17_lev1 --name FP13_a17_lev1_s42 --method fm \
    --epochs 700 --eval 700 --generate 12406 --reverse_steps 100
# baselines on the same data and backbone
python run.py --config cifar10lt/FP13_a17_lev1 --name dlpm_a17 --method dlpm --epochs 1300 --check 100   # DLPM (DLIM: eval with --deterministic)
python run.py --config cifar10lt/FP9_lim_a17_clip --name lim_a17 --method lim --epochs 1300 --check 100   # LIM

# HRRR VIL: HTFM-alpha, alpha = 1.7 (NCSN++, 1500 epochs)
python run.py --config hrrr_vil/HRRR_VIL_alpha17 --name HTFM_HRRR_a17 --method fm --epochs 1500 --check 50
# tail statistics (kurtosis ratio, skewness ratio, tail KS) of 20k generated fields vs the 2021 test year, Heun with 25 steps
python tools/eval_hrrr_tails.py --config hrrr_vil/HRRR_VIL_alpha17 --name HTFM_HRRR_a17 \
    --epoch 1100 --generate 20000 --reverse_steps 25
```

`eval.py` computes the generic metrics (FID / precision–recall on images, PRD /
Wasserstein on 2D data); HRRR fields are evaluated with `tools/eval_hrrr_tails.py`.

Useful flags (all optional, they override the YAML): `--epochs`, `--check`
(checkpoint frequency), `--check_start`, `--set_seed`, `--resume`,
`--n_max_batch N` (smoke test), `--reverse_steps`, `--solver`, `--alpha`,
`--nu`, `--dataset`, `--lr`, `--log --wandb_project …`. `script_utils.py` lists them all.

## Method names used in the paper

| Paper | `--method` | Config knobs |
|---|---|---|
| HTFM-G (Gaussian limit, $p=\infty$) | `fm` | `fm.clock.type: gaussian` |
| HTFM-α, tail degree $p_\alpha$ | `fm` | `fm.clock.type: alpha_stable`, `fm.alpha: 1.6 … 1.9` |
| HTFM-t, tail degree $p_\nu$ | `fm` | `fm.clock.type: student_t`, `fm.clock.nu: 1.7 / 2.0 / 3.0` |
| signature order $m$ | `fm` | `path_conditioning.path_signature.level: 1 / 2`; $m=0$ is `path_conditioning.enabled: false` (the `*_no_pc` configs) |
| source type (terminal vs area) | `fm` | `fm.noise_type: terminal` vs `forward_process` |
| flow type | `fm` | `fm.flow_type: straight_line` (paper default), `tbar_linear`, `signature_linear_ve` |
| LIM | `lim` | `lim.alpha` |
| DLPM / DLIM | `dlpm` | `dlpm.alpha`; DLIM = `eval.dlpm.deterministic: true, dlim_eta: 0.0` (`--deterministic`) |

Config families: `toy2d/ch_sas{15,18}_*` are the 2D mixtures with
$\alpha_D\in\{1.5,1.8\}$ (HTFM-α with / without clock conditioning, HTFM-t,
DLPM); `cifar10lt/FP13_*` are the eight HTFM variants of the main CIFAR10-LT
table and double as the DLPM configs (`--method dlpm`), `FP9_lim_*` the LIM
baselines and `FP12_dlpm_a2` the Gaussian DDPM / DDIM limit; `hrrr_vil/` holds
the HRRR HTFM-α / HTFM-t / HTFM-G runs and the DLPM / LIM baselines. All YAML
fields are described in [docs/config_reference.md](docs/config_reference.md).

Protocol notes

- CIFAR10-LT configs say `run.epochs: 2000` (the step-LR schedule counts
  optimiser steps), but all reported CIFAR10-LT models were stopped at epoch
  1300, so pass `--epochs 1300`. FID is 12,406 generated vs the 12,406-image
  long-tailed training split, EMA weights, Euler sampler; the paper averages
  seeds `42 1234 0 4321 100`.
- HRRR tail statistics use 20,000 generated fields vs the 2021 test year with
  Heun and 25 reverse steps (NFE 50); the reporting epoch is chosen by the rank
  sum of (KR, SR, KS) over the checkpoint sweep.
- 2D runs train for 300 epochs, report the best checkpoint over epochs 100–300
  and compare 24,000 samples with a 24,000-sample held-out reference set
  ($k=10$ nearest neighbours for $f_1^{\mathrm{pr}}$).

## Citation

```bibtex
@inproceedings{yang2026htfm,
  title     = {Heavy-Tailed Flow Matching via Random Clocks},
  author    = {Yang, Zhouhao and Wang, Yezhen and Kawaguchi, Kenji and Braverman, Vladimir and Cao, Haoyang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements and license

The code grew out of the [DLPM](https://github.com/darioShar/DLPM) /
[BEM](https://github.com/darioShar/bem) code base and vendors components of
LIM, score_sde and improved-diffusion; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact files and
their licenses. <!-- TODO: LICENSE for this repository is still to be decided (bem/ is AGPL-3.0 upstream) -->
