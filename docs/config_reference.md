# Configuration reference

Every experiment is a YAML file under `htfm/configs/<family>/`, selected with
`run.py --config <family>/<name>`. Command-line flags from `script_utils.py`
override individual fields (`--epochs`, `--alpha`, `--nu`, `--reverse_steps`,
`--solver`, `--lr`, …). A config may contain blocks for several methods
(`fm:`, `dlpm:`, `lim:` and the matching `training.*` / `eval.*` blocks);
`method:` or `--method` picks which one is used, so one file can define a
model and all its baselines on the same data.

The experiment hash that names the checkpoint directory (`models/<run>/<hash>/`)
is computed from `data` (dataset, channels, image size, long-tail block), the
active method block and `model`. `optim`, `training`, `eval` and `run` are not
hashed, so they can be changed without invalidating checkpoints.

## `data`

| key | meaning |
|---|---|
| `dataset` | `sas_grid` (2D α-stable mixture), `cifar10_lt`, `cifar10`, `hrrr_vil`, … |
| 2D: `dim`, `n_mixture`, `grid_spacing`, `std`, `data_alpha`, `weights`, `nsamples`, `isotropic` | mixture geometry: 9 components on a 3×3 grid of spacing 2.0, isotropic α-stable components with scale `std` and stability index `data_alpha`, imbalanced `weights` |
| CIFAR: `image_size`, `channels`, `random_flip`, `long_tail.{enabled, imbalance_ratio, max_count, num_classes, seed, metadata_dir}` | long-tailed subset; the index file lives in `data/splits/` |
| HRRR: `cache_dir`, `channels: 1`, `image_size: 128`, `train_years`, `test_years` | cache built by `tools/build_hrrr_cache.py`; z-scored per channel from the train years |

## `method` and the method blocks

### `fm` (HTFM and Gaussian flow matching)

| key | values | meaning |
|---|---|---|
| `clock.type` | `alpha_stable`, `student_t`, `gaussian` | random clock family: α/2-stable subordinator (HTFM-α), inverse-Gamma slope `T_t = t V_ν` (HTFM-t), deterministic `T_t = t` (HTFM-G) |
| `alpha` (or `clock.alpha`) | 1 < α ≤ 2 | tail index of the α-stable source; equals the paper's `p_α` |
| `clock.nu` | ν > 0 | Student-t degrees of freedom (`p_ν`); `clock.feature_mode` (`log_scalar` default) chooses how `V_ν` enters the conditioning, `clock.clamp_V_max/min` optionally clamp extreme draws |
| `clock.path_sampling.{group_size, resample_interval}` | ints ≥ 1 | share one clock path across a group of samples / reuse it for several steps (1, 1 in all paper configs) |
| `noise_type` | `terminal`, `forward_process` | which clock functional defines the source covariance: terminal value `T_1` or the integrated path (the paper's "Area" source, Σ = 2·A(T)) |
| `flow_type` | `straight_line` (default), `tbar_linear`, `signature_linear_ve`, `ot_geodesic` | interpolation between source and data; `signature_linear_ve` reads `signature_linear_ve.{lambda_alpha, lambda_beta}` |
| `prediction_type` | `velocity` (paper), `epsilon` | network parameterisation |
| `reverse_steps` | int | default number of ODE steps (overridden by `eval.fm.reverse_steps` / `--reverse_steps`) |
| `scale`, `isotropic`, `input_scaling` | `scale_preserving`; bools | noise-schedule family inherited from DLPM; all paper configs use `scale_preserving`, `isotropic: true`, `input_scaling: false` |

### `dlpm` (α-stable DDPM, Shariatian et al.)

`alpha`, `reverse_steps` (training-time discretisation, 4000 on CIFAR),
`mean_predict: EPSILON`, `var_predict: FIXED`, `rescale_timesteps`,
`isotropic`, `scale`, `input_scaling`. DLIM is the same model sampled with
`eval.dlpm.deterministic: true` and `dlim_eta: 0.0`.

### `lim` (Lévy–Itô SDE, Yoon et al.)

`alpha`, `reverse_steps` (1000), `rescale_timesteps`, `isotropic`, `clamp_a`,
`clamp_eps`.

### `gmflow` (optional baseline, not in the paper)

`K` mixture components, `transition_lambda`, `loss`, `sampler`,
`spectral_weight`; model types `gmflow_ddpm` / `gmflow_ncsnpp`.

## `path_conditioning` (clock features fed to the network)

```yaml
path_conditioning:
  enabled: true                 # false = m = 0, no clock features (the *_no_pc configs)
  path_signature:
    level: 1                    # logsignature truncation order m (1 or 2 in the paper)
    input_channels: [t, T]      # path channels: time and (effective) clock
    discretization_steps: 200   # grid on which the clock path is discretised
    backend: numeric            # or iisignature
    normalization: standardize
    detach: false
  injection:
    global:   {enabled: true, mode: concat}   # concat to / add to the time embedding
    modulation: {enabled: false, target_layers: [middle]}
  projection: {dim: 128, hidden_dim: 128, dropout: 0.0}
```

For the Student-t clock the feature is the scalar `log V_ν` (set by
`fm.clock.feature_mode`), so `path_signature` is ignored. Older configs use a
flat layout (`path_feature_type`, `logsignature_level`, `proj_dim`, …); it is
migrated to the block above when the config is loaded.

## `model`

| `model_type` | backbone | main keys |
|---|---|---|
| `ddpm` | UNet (improved-diffusion), CIFAR10-LT | `model_channels: 128`, `channel_mult: [1,2,2,2]`, `num_res_blocks: 2`, `attn_resolutions: [16]`, `dropout`, `num_heads` |
| `lim_ncsnpp` | NCSN++ (score_sde), HRRR 128×128 | `ch: 32`, `ch_mult: [1,2,2,4,4]`, `num_res_blocks: 4`, `attn_resolutions: [16]`, `progressive*`, `fir`, `embedding_type: positional` |
| `mlp_fm` / MLP | residual MLP, 2D data | `nblocks: 4`, `nunits: 64`, `time_emb_size`, `time_emb_type`, `act: silu`, `group_norm`, `skip_connection` |

`ema_rate` / `training.*.ema_rates` set the EMA decay used for evaluation
(0.9999 on images, 0.99 on 2D data).

## `training`, `optim`, `run`

| block | keys |
|---|---|
| `training` | `batch_size`, `num_workers`, then per method: `ema_rates`, `grad_clip`, `lploss` (2 = squared L2), `loss_monte_carlo` (`mean`; `median` / `robust` aggregate over `monte_carlo_outer × monte_carlo_inner` replicates), `clamp_a` / `clamp_eps` (clamp extreme subordinator / Gaussian draws for stability), `fm_loss_weighting.{enabled, mode, time_fn, norm_fn, lambda_weight, …}` (`weight = base + λ·g(t)·h(‖x_t‖)`; CIFAR configs use `time_norm`) |
| `optim` | `optimizer: adamw`, `lr`, `warmup` (steps), `schedule` (`steplr` or `null`), `lr_step_size`, `lr_gamma`, `weight_decay` |
| `run` | `epochs`, `checkpoint_freq`, `checkpoint_start`, `eval_freq` (`null` = evaluate only through `eval.py`), `progress` |

## `eval`

`data_to_generate` (samples per evaluation), `real_data` (reference set
size), `batch_size`, then per method: `reverse_steps` (NFE for Euler),
`solver` (`euler`, `heun`, `midpoint`, `rk4`, `ab2`, `dopri5`),
`deterministic` (ODE sampling for `fm`, DLIM for `dlpm`), `dlim_eta`,
`clip_denoised`, `clamp_a`, `clamp_eps`. `eval.py --generate N
--reverse_steps R --solver S` override these at evaluation time; results
are stored next to the checkpoint and summarised by the drivers in `scripts/`.
