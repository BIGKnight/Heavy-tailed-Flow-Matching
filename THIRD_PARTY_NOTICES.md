# Third-party code

This repository builds on, and vendors modified copies of, code from the
projects below. Each original copyright and license continues to apply to the
corresponding files; see the linked repositories for the full texts.

| Project | Used for | Files in this repo | Upstream license |
|---|---|---|---|
| [BEM — Better Experimentation Manager](https://github.com/darioShar/bem) (Dario Shariatian) | experiment framework: training / evaluation / generation managers, checkpointing, FID and precision–recall evaluation | `bem/` (modified: CIFAR10-LT subset builder, HRRR loader, tail metrics, EMA handling) | GNU AGPL-3.0 |
| [DLPM — Denoising Lévy Probabilistic Models](https://github.com/darioShar/DLPM) (Shariatian, Simsekli, Durmus; ICLR 2025) | the original package layout, the `dlpm` method (α-stable DDPM), model builders, 2D α-stable mixture generators | `htfm/methods/dlpm*.py`, `htfm/methods/noise_schedule/`, `htfm/models/Model.py`, `htfm/models/DiffusionBlocks.py`, `htfm/models/Embeddings.py`, `bem/datasets/Data.py`, `bem/datasets/Distributions.py`, `run.py`, `eval.py`, `script_utils.py` | no license file in the upstream repository; used with the authors' knowledge |
| [LIM — Lévy–Itô Models](https://github.com/UNIST-LIM-Lab/LIM) and [torchlevy](https://github.com/UNIST-LIM-Lab/torchlevy) (Yoon et al., ICLR 2023) | the `lim` baseline (fractional-score SDE), α-stable sampling utilities | `htfm/methods/LIM/functions/`, `htfm/methods/LIM/torchlevy/`, `htfm/methods/lim_trainer.py`, `bem/datasets/torchlevy/` | no explicit license in the upstream repositories |
| [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) (Song et al.) | NCSN++ backbone used for HRRR, StyleGAN2-style up/down-sampling ops | `htfm/models/ncsnpp.py`, `htfm/models/layerspp.py`, `htfm/models/layers.py`, `htfm/models/normalization.py`, `htfm/models/ddpm.py`, `htfm/models/up_or_down_sampling.py`, `htfm/methods/LIM/op/` | Apache-2.0 |
| [improved-diffusion](https://github.com/openai/improved-diffusion) (OpenAI) | UNet backbone used for CIFAR10-LT | `htfm/models/unet.py`, `htfm/models/nn.py`, `htfm/models/fp16_util.py` | MIT |
| [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [prdc](https://github.com/clovaai/generative-evaluation-prdc) | Inception features, FID, density / coverage | `bem/evaluate/fid_score.py`, `bem/evaluate/inception.py` (via the `prdc` package) | Apache-2.0 / MIT |
| [precision-recall-distributions](https://github.com/msmsajjadi/precision-recall-distributions) (Sajjadi et al.) | PRD curves and the $f_1^{\mathrm{pr}}$ score used on the 2D benchmarks | `bem/evaluate/prd_score.py`, `bem/evaluate/prd_legacy.py` | Apache-2.0 |

The HTFM-specific code (random clocks, path/logsignature features, clock-conditioned
flow matching, solvers, HRRR tail metrics, configs) is new to this repository.
