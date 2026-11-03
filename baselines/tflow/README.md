# t-Flow

Standalone re-implementation of the t-Flow construction from
**Pandey et al. 2024, "Heavy-Tailed Diffusion Models"** (arXiv:2410.14171,
Appendix B), applied to image generation on CIFAR10-LT for an
apples-to-apples comparison with the HTFM project.

The original paper only evaluates t-Flow on weather data (HRRR VIL/W20).
This project extends the same construction to images. **Do not interpret
results from this project as Pandey et al.'s reported numbers** — the
authors did not run CIFAR-10 / CIFAR10-LT in their paper.

## Project layout

```
tFlow/
├── README.md
├── requirements.txt
├── train.py                 # Training entry point (YAML-driven)
├── eval.py                  # FID evaluation entry point
├── configs/
│   ├── cifar10lt_nu1p7.yml
│   ├── cifar10lt_nu2p0.yml
│   ├── cifar10lt_nu3p0.yml
│   └── cifar10lt_gaussian.yml   # nu = inf sanity baseline
└── tflow/
    ├── source.py            # Multivariate Student-t source (the t-Flow novelty)
    ├── flow.py              # FM loss + Euler ODE sampler
    ├── model.py             # Adapter wrapping the DDPM U-Net
    ├── model_unet.py        # DDPM-style U-Net (copy of htfm/models/ddpm.py)
    ├── data_cifar10lt.py    # CIFAR10-LT loader (deterministic seed-42 split)
    ├── ema.py               # EMA helper
    ├── trainer.py           # Training loop
    ├── sample.py            # Image sampling utility
    └── eval_fid.py          # FID via `torch-fidelity`
```

The DDPM U-Net (`tflow/model_unet.py`) is a verbatim copy of
`htfm/models/ddpm.py` from the main repository so we can compare under
**the exact same backbone**. Everything else is fresh code that does NOT
import from the `htfm` package.

## Method recap

For each batch, t-Flow draws a heavy-tailed source

$$ X_0 = \sqrt{V}\, Z, \quad V \sim \mathrm{InvGamma}(\nu/2, \nu/2), \quad Z \sim \mathcal{N}(0, I) $$

(equivalently $X_0 \sim \mathrm{Student\text{-}}t_\nu(0, I)$, with one
scalar $V$ per sample shared across all pixels — exactly the construction
of Pandey et al. Eq. 164) and pairs it with a data sample $X_1$ via the
straight-line interpolant

$$ X_t = (1-t) X_0 + t X_1, \quad t = 0 \to \text{noise},\ t = 1 \to \text{data}. $$

The network is trained with one of two parameterizations (selectable via
`param: noise` / `param: velocity` in the YAML):

* **Noise-prediction** (default; matches Pandey App. B Eq. 166):
  $\epsilon_\theta(X_t, t)$ regresses $X_0$, and the sampling vector field
  is $v(X_t, t) = (X_t - \epsilon_\theta) / t$.
* **Velocity-prediction** (standard CFM): $v_\theta$ regresses
  $X_1 - X_0$ directly.

The two parameterizations are equivalent at the optimum; we provide both so
the noise-prediction default tracks the paper while velocity-prediction is
useful for direct comparison with HTFM's velocity-prediction setup.

Sampling integrates $X_t' = v(X_t, t)$ from a small $t_{\min}$ up to $t=1$
with one of two solvers:

* **Euler** (default for our table): fixed-step Euler at user-specified
  NFE budget. Use this for the matched-NFE comparison with HTFM.
* **Heun** (`--solver heun`): 2nd-order Heun on the Karras-EDM polynomial
  time grid ($\sigma \in [0.01, 1.0]$, $\rho=7$, default NFE=25). This
  reproduces Pandey App. B / Algorithm 4.

Set $\nu = \infty$ (`.inf` in YAML) to recover Gaussian flow matching as a
sanity-check baseline.

The only t-Flow-specific files are `tflow/source.py` (the heavy-tailed
sampler) and the `flow_matching_loss` / `euler_sample` / `heun_sample`
functions in `tflow/flow.py` — everything else is standard FM scaffolding.

## Match status vs. Pandey App. B

| aspect | paper | this repo | match |
| ------ | ----- | --------- | ----- |
| Source | $\epsilon/\sqrt{\kappa}$, $\kappa\sim\chi^2(\nu)/\nu$, scalar $\kappa$ per sample | $\sqrt V Z$, $V\sim\mathrm{InvGamma}(\nu/2,\nu/2)$, scalar $V$ per sample | ✓ (equivalent) |
| Path | $X_t=tX_1+(1-t)\epsilon/\sqrt\kappa$ | $X_t=(1-t)X_0+tX_1$ | ✓ |
| Boundaries | $t{=}0$ noise / $t{=}1$ data | same | ✓ |
| Regression target | $X_0$ (noise pred) | `param='noise'` regresses $X_0$ (default) | ✓ |
| Time | $t\sim U[0,1]$, no reweighting | same | ✓ |
| Default sampler | Heun + Karras grid, NFE=25 | `--solver heun` | ✓ when chosen |
| Default sampler in CIFAR sweep | (n/a; paper has no CIFAR) | Euler at matched NFE | n/a |

## Quickstart

```bash
# 1. Install (CUDA-enabled torch, etc.)
pip install -r requirements.txt

# 2. Train one decay degree (nu = 1.7) on CIFAR10-LT.
python train.py --config configs/cifar10lt_nu1p7.yml

# 3. Evaluate FID at NFE=100 against the LT train set.
python eval.py --ckpt runs/tflow_cifar10lt_nu1p7/ckpt_ep2000.pt \
               --nu 1.7 --nfe 100
```

## Setup that matches the HTFM CIFAR10-LT runs

Pulled from `dlpm/configs/cifar10lt_fp11/*.yml`:

- Backbone: DDPM U-Net, base channels 128, multipliers (1,2,2,2),
  2 ResBlocks/scale, 4 attention heads, attention at resolution 16,
  dropout 0.
- Long-tail split: imbalance ratio 0.01, head count 5000, 10 classes,
  seed 42 (~12.4k images).
- Optimizer: AdamW, lr 2e-4, warmup 500 steps, step-LR decay 0.99 every
  1000 steps, batch 100, grad clip 5.0, EMA 0.9999.
- Training: 2000 epochs, ckpt every 100 epochs.
- Sampling: deterministic Euler ODE solver, default NFE=100 for the
  eval script (override via `--nfe`).
- FID: 12,406 generated samples vs the full LT train set, computed by
  `torch-fidelity`.

## Caveats

- `t_eps` defaults to $10^{-3}$ at both training and sampling. The
  $t \to 0$ regime is degenerate (every sample is its own source draw),
  so we never actually evaluate the velocity at $t=0$. Pandey et al.'s
  recipe uses a similar logit-normal time sampler; we use uniform $t$
  for simplicity.
- The heavy-tailed scale $V$ is drawn iid \emph{per sample}; we do not
  share $V$ across the batch. This matches the standard interpretation
  of an isotropic Student-$t$ source.
- For very heavy tails ($\nu \le 2$) the variance is infinite; gradient
  clipping at $5.0$ is load-bearing. If training diverges, lower the
  clip threshold (e.g. $1.0$) before reducing the learning rate.
