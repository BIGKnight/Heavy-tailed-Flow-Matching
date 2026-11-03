"""
Heavy-tailed Flow Matching under the DLPM Continuous-Time Framework.

Implements the flow matching model from Sections 3-4 of the paper:
  - Straight-line flow matching (Section 4.2, eq 4.9)
  - Clock-conditional OT geodesic flow matching (Section 5)
  - Conditional flow matching with probability-flow velocity (Section 4.2, eq 4.7)
  - Noise-prediction parameterization (Section 4.2, eq 4.8)
  - ODE-based sampling via Euler integration
  - Robust aggregation with power r < alpha/2 for heavy-tailed stability

Key idea: the DLPM Gaussian-mixture representation (alpha-stable = sqrt(A) * Gaussian)
allows flow matching to remain well-posed under infinite-variance noise. The inner
expectation (over Gaussian randomness given A) is always finite, while the outer
expectation over the heavy-tailed mixing variable A uses a robust power aggregation.

Two flow types are supported:
  - 'straight_line': X1 = barsigma_T * sqrt(A) * G (pure noise endpoint)
  - 'ot_geodesic': X1 = bargamma_T * X0 + barsigma_T * sqrt(A) * G (full forward terminal)
    The OT geodesic minimizes the conditional W2 transport cost between the data point
    and its terminal conditional Gaussian law, pathwise in the clock realization.
"""

from tqdm import tqdm
import torch
import torch.nn as nn
# v12.1: FM imports tensor utils from _utils (no longer reaches into dlpm.py
# or GenerativeLevyProcess.py for them — completes the v11 decoupling).
from ._utils import match_last_dims, compute_loss_terms
from .noise_schedule import build_schedule
from .generators import build_skewed_levy, build_sas
from .fm_loss_weighting import (
    get_fm_loss_weighting_config,
    validate_fm_loss_weighting_config,
    compute_fm_loss_weights,
    materialize_weight_diagnostics,
)


class FlowMatchingLevyProcess:
    """
    Heavy-tailed Flow Matching using the DLPM continuous-time framework.

    Supports two training modes (prediction_type):
      1. 'velocity' (default): The model directly predicts the velocity field v(t, x).
      2. 'epsilon': Noise-prediction parameterization. The model predicts the
         standardized noise epsilon, which is mapped to a velocity field via
         the score (eq 4.8). Compatible with DLPM-trained models.

    Supports two flow types (flow_type):
      1. 'straight_line' (default): Straight-line flow matching.
         X1 = barsigma_T * sqrt(A) * G, Xt = (1-t)*X0 + t*X1, v* = X1 - X0.
      2. 'ot_geodesic': Clock-conditional OT geodesic flow matching.
         X1 = bargamma_T * X0 + barsigma_T * sqrt(A) * G (full forward terminal),
         Xt = (1-t)*X0 + t*X1, v* = X1 - X0.
         Minimizes conditional W2 transport cost between data point and terminal law.

    Sampling uses Euler ODE integration from t=1 (noise) to t=0 (data).
    """

    def __init__(
        self,
        alpha,
        device,
        reverse_steps,
        time_spacing='linear',
        isotropic=True,
        scale='scale_preserving',
        input_scaling=False,
        prediction_type='velocity',  # 'velocity' or 'epsilon'
        flow_type='straight_line',   # 'straight_line' | 'ot_geodesic' | 'signature_linear_ve' | 'tbar_linear'
        noise_type='terminal',       # 'terminal' | 'forward_process' | 'area' (paper §3.4)
        path_sampling_cfg=None,      # {'group_size': G, 'resample_interval': M}
        path_pool_cfg=None,          # {'enabled': bool, 'N', 'seed', 'clamp_a'}
        clock_type='alpha_stable',   # 'alpha_stable' (legacy) | 'student_t' | 'gaussian' (vanilla FM)
        clock_kwargs=None,           # e.g. {'nu': 3.0, 'feature_mode': 'log_scalar'}
        ve_schedule=None,            # SignatureLinearVESchedule | None (only used when flow_type='signature_linear_ve')
    ):
        self.alpha = alpha
        self.device = device
        self.reverse_steps = reverse_steps
        self.time_spacing = time_spacing
        self.isotropic = isotropic
        self.input_scaling = input_scaling
        self.prediction_type = prediction_type
        self.flow_type = flow_type
        self.noise_type = noise_type
        self.ve_schedule = ve_schedule  # None for old flow_types — branches that use it gate on flow_type
        self._weight_log_counter = 0
        self.logger = None
        self.path_encoder = None

        # v15.5: dropped host-side _path_sampling_cfg / _path_pool_cfg.
        # AlphaStableClock owns these; capability asserts read directly
        # from the incoming kwargs (no need to mirror them on FM).

        # v11: FM holds schedule + generators directly (no DLPM class).
        self.schedule = build_schedule(
            scale, alpha=alpha, device=device,
            diffusion_steps=reverse_steps, time_spacing=time_spacing,
        )
        self.gen_a = build_skewed_levy(alpha=alpha, device=device, isotropic=isotropic)
        self.gen_eps = build_sas(alpha=alpha, device=device, isotropic=isotropic)

        # B4 fix: epsilon's eps→velocity mapping (barsigma_T * eps) is derived
        # for the straight-line FM target. With ot_geodesic the target also has
        # a bargamma_T·X0 term and the closed-form mapping breaks. Fail fast.
        if flow_type == 'ot_geodesic' and prediction_type == 'epsilon':
            raise NotImplementedError(
                "prediction_type='epsilon' is only valid for "
                "flow_type='straight_line'; the eps→velocity mapping in "
                "EpsilonPredictor.to_velocity does not handle the bargamma_T·X0 "
                "term present in ot_geodesic. Use prediction_type='velocity' "
                "instead, or implement the ot_geodesic eps mapping.")

        # Predictor: encapsulates how the model output is interpreted.
        # 'velocity' -> direct v(t, X_t); 'epsilon' -> standardized noise eps.
        # Adding a new parameterization = one file in htfm/methods/predictors/.
        from .predictors import build_predictor
        self.predictor = build_predictor(prediction_type, schedule=self.schedule)

        # Clock: alpha_stable or student_t. Both implement the Clock interface
        # (clocks/base.py). training_losses_fm / sample() dispatch uniformly via
        # self.clock.sample_X1_and_features / self.clock.sample_initial.
        self.clock_type = clock_type
        from .clocks import build_clock
        if clock_type == 'alpha_stable':
            ckw = {
                'alpha': alpha, 'device': device,
                'schedule': self.schedule, 'gen_a': self.gen_a,
                'flow_type': flow_type, 'noise_type': noise_type,
                'path_sampling_cfg': path_sampling_cfg,
                'path_pool_cfg': path_pool_cfg,
            }
        else:
            ckw = dict(clock_kwargs or {})
            ckw.setdefault('device', device)
        self.clock = build_clock(clock_type, **ckw)

        # v15.2: Clock-capability-driven config validation. Each clock
        # declares what it supports via supports_path_pool /
        # supports_path_sampling; we surface a clean error if the host
        # config asks for something the clock doesn't accept.
        psg = path_sampling_cfg or {}
        ppg = path_pool_cfg or {}
        if not self.clock.supports_path_sampling:
            G = psg.get('group_size', 1)
            M = psg.get('resample_interval', 1)
            assert G == 1 and M == 1, (
                f'{type(self.clock).__name__} does not support path_sampling '
                f'(got G={G}, M={M}; both must be 1).')
        if not self.clock.supports_path_pool:
            assert not ppg.get('enabled', False), (
                f'{type(self.clock).__name__} does not support '
                f'path_pool.enabled=true.')
        # Flow-type / clock-type compatibility stays clock-specific because
        # clocks don't take flow_type in their __init__ today.
        if clock_type == 'student_t':
            # tbar_linear is allowed because for Student-t it degenerates
            # to straight_line (bar_T_t = t exactly); the flow_type label
            # stays for experiment-matrix uniformity.
            assert flow_type in ('straight_line', 'tbar_linear'), (
                f'student_t clock supports flow_type in '
                f'{{straight_line, tbar_linear}}, got {flow_type}')
            print(f'[clock] using {self.clock}', flush=True)

        if clock_type == 'gaussian':
            # Vanilla FM. straight_line is canonical; tbar_linear degenerates
            # to it. ot_geodesic / signature_linear_ve don't make sense
            # without sigma/gamma scaling or T-path.
            assert flow_type in ('straight_line', 'tbar_linear'), (
                f'gaussian clock supports flow_type in '
                f'{{straight_line, tbar_linear}}, got {flow_type}')
            print(f'[clock] using {self.clock}', flush=True)

        # signature_linear_ve preconditions: needs T-path (forward_process),
        # alpha_stable clock (only it produces A sequences), velocity head,
        # and an actual ve_schedule instance to query.
        if flow_type == 'signature_linear_ve':
            assert clock_type == 'alpha_stable', (
                f"flow_type='signature_linear_ve' requires "
                f"clock_type='alpha_stable', got {clock_type!r}")
            assert noise_type in ('forward_process', 'area'), (
                f"flow_type='signature_linear_ve' requires per-step T-path; "
                f"noise_type must be forward_process or area, got {noise_type!r}")
            assert prediction_type == 'velocity', (
                f"flow_type='signature_linear_ve' requires "
                f"prediction_type='velocity', got {prediction_type!r}")
            assert self.ve_schedule is not None, (
                "flow_type='signature_linear_ve' requires a SignatureLinearVESchedule "
                "instance via ve_schedule=...")
            print(f'[fm] signature_linear_ve schedule: {self.ve_schedule}', flush=True)

        # tbar_linear preconditions: alpha_stable + forward_process (T_path
        # required), or any student_t (degenerates). Always velocity head.
        if flow_type == 'tbar_linear':
            assert prediction_type == 'velocity', (
                f"flow_type='tbar_linear' requires prediction_type='velocity', "
                f"got {prediction_type!r}")
            if clock_type == 'alpha_stable':
                assert noise_type in ('forward_process', 'area'), (
                    f"flow_type='tbar_linear' with alpha_stable clock requires "
                    f"noise_type='forward_process' or 'area' (per-step T_path needed), "
                    f"got {noise_type!r}")
            elif clock_type in ('student_t', 'gaussian'):
                pass  # bar_T_t = t exactly; degenerates to straight_line
            else:
                raise AssertionError(
                    f"flow_type='tbar_linear' supports clock_type in "
                    f"{{alpha_stable, student_t, gaussian}}, got {clock_type!r}")
            print(f'[fm] tbar_linear flow active (clock={clock_type})', flush=True)

    def set_path_encoder(self, path_encoder):
        self.path_encoder = path_encoder
        # Clock decides what to do with the encoder; AlphaStableClock builds
        # path_pool + samplers, StudentTClock inherits the no-op base default.
        self.clock.set_path_encoder(path_encoder)

    def get_ckpt_extra(self):
        """Return extras to embed in ckpt (e.g. path_pool config + sha256).
        Called by TrainingManager.save() via hasattr() hook.

        On-disk dict layout (preserved across the v1 clock-abstraction
        refactor): {'path_pool': {...}|None, 'clock': {'type', 'kwargs'}}.
        """
        extra = {'path_pool': None}            # default for non-alpha-stable
        extra.update(self.clock.get_ckpt_extra())  # alpha-stable contributes path_pool entry
        extra['clock'] = {
            'type': self.clock_type,
            'kwargs': dict(self.clock.ckpt_metadata_kwargs),
        }
        return extra

    def validate_ckpt_extra(self, extra):
        """Check pool/clock consistency between ckpt-time and now. Prints WARN on mismatch.
        Called by TrainingManager.load() via hasattr() hook.
        """
        if not extra:
            return
        # Path-pool consistency (alpha-stable only; Student-t no-op).
        self.clock.validate_ckpt_extra(extra)

        # Clock metadata consistency (host-level since it knows clock_type).
        ckpt_clock = extra.get('clock')
        if ckpt_clock is not None:
            ckpt_type = ckpt_clock.get('type', 'alpha_stable')
            ckpt_kwargs = ckpt_clock.get('kwargs', {})
            if ckpt_type != self.clock_type:
                print(f'[clock] WARN: ckpt trained with clock_type={ckpt_type!r} '
                      f'but current config uses clock_type={self.clock_type!r}.', flush=True)
            elif self.clock.ckpt_metadata_kwargs:  # has metadata to compare
                cur_kwargs = dict(self.clock.ckpt_metadata_kwargs)
                if ckpt_kwargs != cur_kwargs:
                    print(f'[clock] WARN: clock kwargs mismatch:\n'
                          f'  ckpt   : {ckpt_kwargs}\n'
                          f'  current: {cur_kwargs}', flush=True)
                else:
                    print(f'[clock] OK: type={ckpt_type} kwargs={ckpt_kwargs} match ckpt', flush=True)

    def get_timesteps(self, N, **kwargs):
        return self.schedule.get_timesteps(N)

    ###########################################################################
    # SAMPLING: ODE-based Euler integration
    ###########################################################################

    def sample(
        self,
        models,
        shape,
        reverse_steps,
        time_spacing=None,
        initial_data=None,
        clip_denoised=False,
        deterministic=True,
        print_progression=False,
        get_sample_history=False,
        clamp_a=None,
        clamp_eps=None,
        solver='euler',
        solver_kwargs=None,
        n_velocity_paths=1,
        x1_strategy='direct_alpha',
        **kwargs,
    ):
        """
        Generate samples by integrating the learned velocity field from t=1 to t=0.

        The probability-flow ODE is:
            dX_t / dt = v_theta(t, X_t)

        We integrate backward from t=1 (heavy-tailed noise) to t=0 (data) using
        the ODE solver specified by `solver` (default 'euler'; see fm_solvers/ for others).
        """
        self.gen_a.setParams(clamp_a=clamp_a)
        self.gen_eps.setParams(clamp_eps=clamp_eps)

        model = models['default']
        model.eval()

        N = reverse_steps
        dt = 1.0 / N

        x_hist = []

        with torch.inference_mode():
            A_sample = None
            # path_features_sample is set by self.clock.sample_initial in the
            # default branch; explicit initialization covers the initial_data
            # and K_vel>1 paths that don't go through the clock.
            path_features_sample = None
            # For inference-time velocity averaging (n_velocity_paths > 1):
            #   phi_stack: [B*K, feat_dim] — K path features per sample.
            #   When set, v_fn does K-way model forward + mean.
            phi_stack = None
            K_vel = int(n_velocity_paths) if n_velocity_paths is not None else 1

            if initial_data is not None:
                x = initial_data
            elif K_vel > 1:
                # --- Inference-time velocity averaging ---
                # At each ODE step: v̄ = mean over K path features.
                # x₁ sampling: direct_alpha (A for x₁ independent of K paths)
                #           or k_mean (x₁ = K^(-1/α) · Σ_k sqrt(Σ(A_k))·G_k).
                assert self.noise_type in ('forward_process', 'area'), (
                    'velocity averaging requires noise_type ∈ {forward_process, area}, '
                    f'got {self.noise_type}')
                assert self.path_encoder is not None and self.path_encoder.enabled, (
                    'velocity averaging requires path_encoder enabled')
                if self.clock.eval_sampler is not None:
                    print('[velocity_avg] WARN: ignoring eval PathSampler '
                          '(group_size/M) because n_velocity_paths>1', flush=True)

                T_total = len(self.schedule.sigmas)
                B = shape[0]

                # K A-sequences per sample, packed into one tensor [T, B*K].
                # Index convention: A_stack[:, b*K + k] = A of sample b, path k.
                A_flat = self.gen_a.generate(size=[T_total * B * K_vel])
                A_stack = A_flat.reshape(T_total, B * K_vel).to(self.device)

                sigmas = self.schedule.sigmas.to(self.device)
                gammas = self.schedule.gammas.to(self.device)
                if self.noise_type == 'forward_process':
                    Sigma_stack = sigmas[0] ** 2 * A_stack[0]
                    for s in range(1, T_total):
                        Sigma_stack = sigmas[s] ** 2 * A_stack[s] + gammas[s] ** 2 * Sigma_stack
                else:  # 'area' — paper §3.4 area integral 2·∫T_s ds (cached coef)
                    area_coef = self.clock.get_area_noise_coef(T_total)  # [T]
                    Sigma_stack = (area_coef.unsqueeze(-1) * A_stack).sum(dim=0)
                # Sigma_stack shape: [B*K]

                if x1_strategy == 'direct_alpha':
                    # Auxiliary A_tmp used ONLY for x_1; discarded, not used for path.
                    A_tmp = self.gen_a.generate(size=shape).to(self.device)
                    G = torch.randn(shape, device=self.device)
                    x = self.schedule.barsigmas[-1] * torch.sqrt(A_tmp) * G
                    A_sample = A_tmp   # for fingerprint log only
                elif x1_strategy == 'k_mean':
                    # x_1 = (1/K^(1/α)) · Σ_k sqrt(Σ(A_k)) · G_k
                    # α-stable CLT normalization: preserves the correct marginal scale.
                    Sigma_bc = Sigma_stack.view(-1, *([1] * (len(shape) - 1)))
                    G_stack = torch.randn([B * K_vel, *shape[1:]], device=self.device)
                    x1_per_k = torch.sqrt(Sigma_bc) * G_stack     # [B*K, ...]
                    x1_grouped = x1_per_k.reshape(B, K_vel, *shape[1:])
                    x = x1_grouped.sum(dim=1) / (K_vel ** (1.0 / self.alpha))
                    G = G_stack.reshape(B, K_vel, *shape[1:])[:, 0]
                    A_sample = A_stack  # shape [T, B*K] — used for fingerprint only
                else:
                    raise ValueError(f"Unknown x1_strategy: {x1_strategy!r}")

                # Precompute K path features for all samples: [B*K, feat_dim]
                t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
                    sigmas, gammas, A_stack)
                phi_stack = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)

                print(f'[velocity_avg] K={K_vel} strategy={x1_strategy} '
                      f'x1.shape={tuple(x.shape)} phi.shape={tuple(phi_stack.shape)}',
                      flush=True)
            else:
                # Uniform Clock dispatch as of Step E.2 — both alpha-stable and
                # student_t go here. Returns (x, A_sample, G, path_features).
                x, A_sample, G, path_features_sample = self.clock.sample_initial(shape)

            # [multipath eval] log fingerprint of (A, G) for first batch item.
            # Used to verify that K independent sampling rounds see different paths.
            try:
                if A_sample is not None:
                    a_fp = A_sample.detach().cpu().float().flatten()[:3].tolist()
                    g_fp = G.detach().cpu().float().flatten()[:3].tolist()
                    print(f"[path_fingerprint] A_head={a_fp} G_head={g_fp}", flush=True)
            except Exception as _e:
                print(f"[path_fingerprint] ERR: {_e}", flush=True)

            if get_sample_history:
                x_hist.append(x.clone())

            # K_vel>1 still inline (moves in Step F): when phi_stack is set,
            # v_fn uses it directly so path_features_sample must be None.
            if phi_stack is not None:
                path_features_sample = None
            # else: path_features_sample was set by clock.sample_initial OR
            #       remains its initial-data-path / unset value.

            if print_progression:
                tqdm._instances.clear()
                pbar = tqdm(total=N)

            # --- Build pure velocity function for the solver ---
            if phi_stack is not None:
                # K-way velocity averaging: forward K paths in one batched call, take mean.
                K = K_vel
                _phi = phi_stack
                def v_fn(x_in, t_scalar):
                    B_loc = x_in.shape[0]
                    x_exp = x_in.repeat_interleave(K, dim=0)      # [B*K, ...]
                    t_batch = torch.full([x_exp.shape[0]], t_scalar, device=self.device)
                    v_flat = self._get_velocity(model, x_exp, t_batch, path_features=_phi)
                    return v_flat.reshape(B_loc, K, *x_in.shape[1:]).mean(dim=1)
            else:
                def v_fn(x_in, t_scalar):
                    t_batch = torch.full([x_in.shape[0]], t_scalar, device=self.device)
                    return self._get_velocity(model, x_in, t_batch,
                                              path_features=path_features_sample)

            # --- Post-step hook: clip / history / progress (orthogonal to solver).
            # Pass None when everything's inactive so solvers (esp. dopri5) can take a
            # single-call fast path.
            need_hook = clip_denoised or get_sample_history or print_progression
            step_hook = None
            if need_hook:
                def step_hook(x_new, i):
                    if clip_denoised:
                        x_new = x_new.clamp(-1, 1)
                    if get_sample_history:
                        x_hist.append(x_new.clone())
                    if print_progression:
                        pbar.update(1)
                    return x_new

            # --- Build solver and integrate ---
            from htfm.methods.fm_solvers import build_solver
            s = build_solver(solver)
            x = s.integrate(v_fn, x, N,
                            t_start=1.0, t_end=0.0,
                            step_hook=step_hook,
                            **(solver_kwargs or {}))

            if print_progression:
                pbar.close()
                tqdm._instances.clear()

            if get_sample_history:
                return x, torch.stack(x_hist)
            return x

    def _get_velocity(self, model, x, t, path_features=None):
        """Forward through the model and let the Predictor map output to velocity."""
        model_output = model(x, t, path_features=path_features)
        return self.predictor.to_velocity(model_output, x, t)

    ###########################################################################
    # TRAINING
    ###########################################################################

    def training_losses(
        self,
        models,
        x_start,
        model_kwargs=None,
        **kwargs,
    ):
        """Standard BEM interface for computing training losses."""
        model = models['default']
        x_start = x_start.to(self.device)
        if model_kwargs is None:
            model_kwargs = {}

        loss, raw_diag, w_cfg = self.training_losses_fm(model, x_start, **model_kwargs, **kwargs)

        result = {'loss': loss}
        if raw_diag:
            self._weight_log_counter += 1
            log_interval = w_cfg.get('log_interval', 500)
            if self._weight_log_counter % log_interval == 1:
                materialized = materialize_weight_diagnostics(raw_diag)
                result.update(materialized)
                if self.logger is not None:
                    for k, v in materialized.items():
                        self.logger.log(k, v)
                else:
                    parts = ['[fm_weight]'] + ['{}: {:.4f}'.format(k, v) for k, v in materialized.items()]
                    print(' '.join(parts))

        return result

    def training_losses_fm(
        self,
        model,
        x_start,
        lploss=2.0,
        loss_monte_carlo='mean',
        monte_carlo_outer=1,
        monte_carlo_inner=1,
        monte_carlo_t=1,
        robust_exponent=0.5,
        clamp_a=None,
        clamp_eps=None,
        fm_loss_weighting=None,
        **kwargs,
    ):
        """
        Flow matching loss with support for two flow types:

        1. Straight-line (flow_type='straight_line', Section 4.2, eq 4.9):
            X1 = barsigma_T * sqrt(A) * G
            Xt = (1-t)*X0 + t*X1,  v* = X1 - X0

        2. OT geodesic (flow_type='ot_geodesic', Section 5, eq 5.12):
            X1 = bargamma_T * X0 + barsigma_T * sqrt(A) * G  (full forward terminal)
            Xt = (1-t)*X0 + t*X1,  v* = X1 - X0

        Both use robust aggregation: L(theta) = int E[(E[||v_theta - v*||^2 | T])^r] dt

        Monte Carlo structure:
            - monte_carlo_outer: independent A draws per sample (for median/robust)
            - monte_carlo_inner: independent G draws per A (for E[loss | A])
            - monte_carlo_t: independent t draws per (X0, X1) pair (for E_t[loss])
              Amortizes the cost of A sequence sampling across multiple t values.
        """
        self.gen_a.setParams(clamp_a=clamp_a)
        self.gen_eps.setParams(clamp_eps=clamp_eps)

        B = x_start.shape[0]

        # Setup median-of-means / robust aggregation
        # (same structure as DLPM Proposition 9 loss)
        total_mc = monte_carlo_outer * monte_carlo_inner
        x0_ext = x_start.repeat(total_mc, *([1] * len(x_start.shape[1:])))

        # --- Sample terminal noise X1 ---
        outer_shape = list(x_start.shape)
        outer_shape[0] *= monte_carlo_outer
        B_outer = outer_shape[0]
        G = torch.randn_like(x0_ext, device=self.device)

        # --- Sample X1 + path_features (uniform Clock dispatch as of Step E.2) ---
        X1, path_features = self.clock.sample_X1_and_features(
            x_start, x0_ext, G, monte_carlo_inner, monte_carlo_outer)

        # --- Expand for multiple t samples per (X0, X1) pair ---
        # X1, x0_ext, path_features are all [B*total_mc, ...] at this point.
        # Repeat them monte_carlo_t times; sample fresh t for each copy.
        if monte_carlo_t > 1:
            x0_ext = x0_ext.repeat(monte_carlo_t, *([1] * (x0_ext.dim() - 1)))
            X1 = X1.repeat(monte_carlo_t, *([1] * (X1.dim() - 1)))
            G = G.repeat(monte_carlo_t, *([1] * (G.dim() - 1)))
            if path_features is not None:
                path_features = path_features.repeat(monte_carlo_t, 1)

        B_full = x0_ext.shape[0]  # B * total_mc * monte_carlo_t
        # tbar_linear samples t exactly on the clock grid (k/T_total) so
        # bar_T_t can be looked up without interpolation; all other flows
        # use continuous t in [0.001, 0.999].
        if self.flow_type == 'tbar_linear':
            t = self.clock.sample_t_for_tbar_linear(B_full)
        else:
            t = torch.rand(B_full, device=self.device) * 0.998 + 0.001
        t_expanded = t.view(B_full, *([1] * (len(x_start.shape) - 1)))

        # --- Interpolation: Xt = (1-t)*X0 + t*X1 ---
        # signature_linear_ve overrides Xt below using the affine schedule:
        #   X_t  = alpha_t(t, R_m) * X1 + beta_t(t, R_m) * X0
        # where R_m is the level-2 logsignature Lévy-area projection of the
        # normalized clock path (clock._last_T_path, cached in sample_X1_and_features).
        # tbar_linear uses the normalized clock path directly:
        #   X_t = bar_T_t * X1 + (1 - bar_T_t) * X0,   v* = bar_T_dot_t * (X1 - X0).
        if self.flow_type == 'signature_linear_ve':
            T_path = self.clock._last_T_path
            assert T_path is not None, (
                "signature_linear_ve expected clock._last_T_path to be cached "
                "by sample_X1_and_features (forward_process branch).")
            R_m = self.clock.compute_R_m(T_path)            # [B_outer]
            # Match R_m's batch dim to B_full = B_outer * MC_inner * MC_t.
            # Order matches X1/x0_ext expansion: first MC_inner replicate, then MC_t.
            if monte_carlo_inner > 1:
                R_m = R_m.repeat(monte_carlo_inner)
            if monte_carlo_t > 1:
                R_m = R_m.repeat(monte_carlo_t)
            Xt = self.ve_schedule.X_t_from_endpoints(t, R_m, x0_ext, X1)
        elif self.flow_type == 'tbar_linear':
            # MC replication is handled inside compute_bar_T_at via bar_T_path.repeat.
            bar_T, bar_T_dot = self.clock.compute_bar_T_at(t)   # both [B_full]
            spatial = (1,) * (x0_ext.dim() - 1)
            bar_T_b = bar_T.view(-1, *spatial)
            Xt = bar_T_b * X1 + (1.0 - bar_T_b) * x0_ext
        else:
            Xt = (1 - t_expanded) * x0_ext + t_expanded * X1

        # --- Model prediction (parameterization-agnostic dispatch) ---
        prediction = model(Xt, t, path_features=path_features)
        if self.flow_type == 'signature_linear_ve':
            # Affine target v*(t, X_t, X_1) — must use the same R_m as Xt above.
            target = self.ve_schedule.affine_target(t, R_m, Xt, X1)
        elif self.flow_type == 'tbar_linear':
            spatial = (1,) * (x0_ext.dim() - 1)
            bar_T_dot_b = bar_T_dot.view(-1, *spatial)
            target = bar_T_dot_b * (X1 - x0_ext)
        else:
            target = self.predictor.compute_target(X1, x0_ext, G)

        # --- Compute per-sample loss ---
        losses = compute_loss_terms(prediction, target, lploss)
        assert not torch.isnan(losses).any(), 'NaN in FM losses'

        w_cfg = get_fm_loss_weighting_config(fm_loss_weighting)
        if w_cfg['enabled'] and w_cfg['mode'] != 'none':
            validate_fm_loss_weighting_config(w_cfg)
            weights, raw_diag = compute_fm_loss_weights(t, Xt, w_cfg)
            losses = losses * weights.to(dtype=losses.dtype, device=losses.device)
        else:
            raw_diag = {}

        # --- Aggregate per-sample losses via the registry-backed aggregator ---
        from .loss_aggregation import build_aggregator
        aggregator = build_aggregator(loss_monte_carlo)
        loss = aggregator.aggregate(
            losses, monte_carlo_outer, monte_carlo_inner, monte_carlo_t, B,
            robust_exponent=robust_exponent,
        )

        return loss, raw_diag, w_cfg
