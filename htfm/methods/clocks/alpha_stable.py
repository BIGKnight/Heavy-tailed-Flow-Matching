"""AlphaStableClock: wrap the alpha-stable code path of flow_matching.py.

Step C: alongside the setup helpers from Step B, this class now owns
sample_X1_and_features() — the training-time alpha-stable noise/feature
production (formerly two inline branches in training_losses_fm).
"""

import os
import torch

from .base import Clock


class AlphaStableClock(Clock):
    """alpha-stable random clock. Owns A-sequence sampling, Sigma recurrence,
    logsignature feature extraction, (G, M) path sampling, and path_pool.
    """

    feature_dim = 0  # depends on path_encoder, set after set_path_encoder

    # v15.1: alpha-stable accepts path_pool + (G, M) path_sampling configs.
    # Student-t (single-scalar V) does not.
    supports_path_pool = True
    supports_path_sampling = True

    def feature_dim_for_conditioning(self):
        """Conditioning feature comes from the attached path_encoder
        (logsignature output dim). Returns 0 if no encoder is attached
        (i.e., conditioning is disabled at the YAML level).
        """
        enc = getattr(self, 'path_encoder', None)
        if enc is None or not getattr(enc, 'enabled', False):
            return 0
        return int(getattr(enc, 'output_dim', 0))

    def __init__(
        self,
        alpha,
        device,
        schedule,            # NoiseSchedule with sigmas/gammas/barsigmas/bargammas
        gen_a,               # skewed-Lévy subordinator generator
        flow_type,
        noise_type,
        path_sampling_cfg=None,
        path_pool_cfg=None,
    ):
        # Pure reference storage; NO RNG consumption, NO tensor allocation.
        self.alpha = alpha
        self.device = device
        self.schedule = schedule
        self.gen_a = gen_a
        self.flow_type = flow_type
        self.noise_type = noise_type
        self._path_sampling_cfg = dict(path_sampling_cfg or {})
        self._path_pool_cfg = dict(path_pool_cfg or {})

        # Populated by set_path_encoder / build helpers:
        self.path_encoder = None
        self.path_pool = None
        self.train_sampler = None
        self.eval_sampler = None
        self._init_noise_coef_cache = None

        # Cached per-batch T-path for signature_linear_ve schedule.
        # Refreshed by sample_X1_and_features (forward_process branch).
        # Shape [B, T_total]; None if not in forward_process mode.
        self._last_T_path = None

    # ---- tbar_linear flow support ----------------------------------------
    # X_t = bar_T_t * X_1 + (1 - bar_T_t) * X_0  with bar_T_t = T_t / T_1.
    # bar_T_t is exact only at the discrete clock grid; for tbar_linear we
    # therefore sample t on the grid {1/T_total, ..., (T_total-1)/T_total}
    # (avoids interpolation entirely).

    def sample_t_for_tbar_linear(self, B_full):
        """Sample t exactly on the clock grid (no interpolation needed).

        Args:
            B_full: int, full batch size (B * MC_outer * MC_inner * MC_t).
        Returns:
            t: [B_full] float tensor with values k/T_total for k in [1, T_total-1].
        """
        T_total = self._last_T_path.shape[-1]
        k = torch.randint(1, T_total, (int(B_full),), device=self.device)
        return k.float() / float(T_total)

    def compute_bar_T_at(self, t):
        """Look up bar_T and bar_T_dot at grid-aligned t.

        Internal grid (after padding T(0)=0):
            T_padded[k] for k in 0..T_total, with T_padded[0]=0, T_padded[T_total]=T_1.
            grid time s_k = k / T_total.
        Linear interp would be:
            bar_T(t) = (1-frac) * bar_T_padded[k] + frac * bar_T_padded[k+1]
        With t = k/T_total exactly (sample_t_for_tbar_linear), frac=0 and the
        formula collapses to bar_T_padded[k]. bar_T_dot is the right-segment
        difference, which is constant within (s_k, s_{k+1}).

        Args:
            t: [B_full] in (0, 1). Expected to be a multiple of 1/T_total.
        Returns:
            bar_T:    [B_full]
            bar_T_dot: [B_full]
        """
        T_path = self._last_T_path
        assert T_path is not None, (
            "tbar_linear expected clock._last_T_path to be cached "
            "(forward_process branch only).")
        B_outer, T_total = T_path.shape
        B_full = t.shape[0]
        assert B_full % B_outer == 0, (
            f"B_full={B_full} not a multiple of B_outer={B_outer}; "
            f"MC replication misaligned.")
        rep = B_full // B_outer

        # Pad a leading 0 so bar_T(s=0)=0 holds; padded length = T_total+1.
        T_padded = torch.cat(
            [torch.zeros(B_outer, 1, device=T_path.device, dtype=T_path.dtype),
             T_path],
            dim=-1,
        )                                                       # [B_outer, T_total+1]
        T1 = T_padded[:, -1:].clamp(min=1e-12)
        bar_T_padded = T_padded / T1                            # [B_outer, T_total+1]

        # Replicate to match B_full (same order as X1's MC_inner/MC_t expansion:
        # both use .repeat which stacks copies, not interleave).
        bar_T_padded_rep = bar_T_padded.repeat(rep, 1)          # [B_full, T_total+1]

        # Grid index: k = round(t * T_total). Clamp into [0, T_total-1] so
        # k+1 stays a valid index into the padded array.
        k = (t * T_total).round().long().clamp(0, T_total - 1)  # [B_full]
        batch_idx = torch.arange(B_full, device=t.device)
        bt_k = bar_T_padded_rep[batch_idx, k]
        bt_k1 = bar_T_padded_rep[batch_idx, k + 1]
        bar_T = bt_k                                            # exact at grid t
        bar_T_dot = (bt_k1 - bt_k) * float(T_total)             # right-segment slope
        return bar_T, bar_T_dot

    @staticmethod
    def compute_R_m(T_path):
        """Signed clock-shape score from level-2 logsignature Lévy area projection.

        After Fubini + integration-by-parts on
            <e_ct - e_tc, phi_2(bar_T)>,
        this collapses to a 1-D integral:
            R_m = 2 * int_0^1 (T_s / T_1) ds - 1   in [-1, 1]

        R_m > 0: front-loaded clock (jumps occur early).
        R_m < 0: back-loaded clock (jumps occur late).
        R_m = 0: linear clock.

        Args:
          T_path: [batch, n_steps] subordinator trajectory T_s = cumsum(A_s).
                  Assumed T_path[:, 0] >= 0 and T_path[:, -1] = T_1 > 0 per sample.

        Returns:
          [batch] scalar R_m per sample, clamped to [-1, 1].
        """
        T_1 = T_path[:, -1:].clamp(min=1e-12)
        bar_T = T_path / T_1
        n_steps = T_path.shape[-1]
        s_grid = torch.linspace(
            0, 1, n_steps,
            device=T_path.device, dtype=T_path.dtype,
        )
        integral = torch.trapz(bar_T, s_grid, dim=-1)
        return (2.0 * integral - 1.0).clamp(-1.0, 1.0)

    # ---- Setup helpers (moved from flow_matching.py in Step B) ----------

    def set_path_encoder(self, path_encoder):
        """Attach a path-feature encoder; build pool and samplers if configured."""
        self.path_encoder = path_encoder
        # Pool first, so samplers can use it.
        self._maybe_build_path_pool()
        self._maybe_build_samplers()

    def _maybe_build_path_pool(self):
        cfg = self._path_pool_cfg
        if not cfg.get('enabled', False):
            return
        assert self.noise_type in ('forward_process', 'area'), (
            'path_pool only supports noise_type ∈ {forward_process, area} '
            '(both consume per-step A sequences)')
        from ..path import PathPool
        T_total = len(self.schedule.sigmas)
        self.path_pool = PathPool(
            N=cfg['N'], alpha=self.alpha, T_total=T_total,
            seed=cfg.get('seed', 42),
            clamp_a=cfg.get('clamp_a', None),
            cache_dir=cfg.get('cache_dir', 'data/path_pools'),
            device=self.device,
        ).load_or_build()

    def current_pool_sha256(self):
        if self.path_pool is None:
            return None
        sha_path = self.path_pool.cache_path + '.sha256'
        if not os.path.exists(sha_path):
            return 'no-sidecar'
        with open(sha_path) as f:
            return f.read().strip()

    def _maybe_build_samplers(self):
        """Build train/eval PathSamplers if (G, M) != (1, 1)."""
        cfg = self._path_sampling_cfg
        G = int(cfg.get('group_size', 1))
        M = int(cfg.get('resample_interval', 1))
        if G == 1 and M == 1:
            return  # default path, no sampler needed (zero-change behavior)
        from ..path import PathSampler
        self.train_sampler = PathSampler(
            group_size=G, resample_interval=M,
            gen_a=self.gen_a, path_encoder=self.path_encoder,
            schedule=self.schedule, noise_type=self.noise_type,
            device=self.device, context='train',
            path_pool=self.path_pool)
        self.eval_sampler = PathSampler(
            group_size=G, resample_interval=M,
            gen_a=self.gen_a, path_encoder=self.path_encoder,
            schedule=self.schedule, noise_type=self.noise_type,
            device=self.device, context='eval',
            path_pool=self.path_pool)

    def get_ckpt_extra(self):
        """Contribute the path_pool sub-dict; host wraps it with clock metadata."""
        if self.path_pool is None:
            return {'path_pool': None}
        return {'path_pool': {
            'cfg': dict(self._path_pool_cfg),
            'cache_path': self.path_pool.cache_path,
            'sha256': self.current_pool_sha256(),
        }}

    def validate_ckpt_extra(self, extra):
        """Path-pool consistency check (host handles clock metadata)."""
        if not extra:
            return
        ckpt_pool = extra.get('path_pool')
        current_enabled = self.path_pool is not None
        ckpt_enabled = ckpt_pool is not None

        if not ckpt_enabled and not current_enabled:
            return
        if ckpt_enabled and not current_enabled:
            print(f'[path_pool] WARN: ckpt was TRAINED WITH pool '
                  f'(N={ckpt_pool["cfg"].get("N")}, seed={ckpt_pool["cfg"].get("seed")}) '
                  f'but current run has pool DISABLED '
                  f'→ inference will see OOD paths (defeats pool purpose).', flush=True)
            return
        if current_enabled and not ckpt_enabled:
            print(f'[path_pool] WARN: ckpt was trained WITHOUT pool, '
                  f'but current run has pool ENABLED '
                  f'(N={self._path_pool_cfg.get("N")}). '
                  f'Eval uses pool paths ~ same distribution as training; expect no major effect.',
                  flush=True)
            return
        # Both enabled — compare configs
        if ckpt_pool['cfg'] != dict(self._path_pool_cfg):
            print(f'[path_pool] WARN: pool cfg MISMATCH:\n'
                  f'  ckpt   : {ckpt_pool["cfg"]}\n'
                  f'  current: {dict(self._path_pool_cfg)}', flush=True)
        sha_ckpt = ckpt_pool.get('sha256', 'unknown')
        sha_now = self.current_pool_sha256() or 'unknown'
        if sha_ckpt != sha_now:
            print(f'[path_pool] WARN: pool file sha256 differs:\n'
                  f'  ckpt   : {sha_ckpt}\n'
                  f'  current: {sha_now}', flush=True)
        else:
            print(f'[path_pool] OK: pool cfg + sha256 match ckpt ({sha_ckpt[:16]}...)', flush=True)

    def get_area_noise_coef(self, T_total):
        """Coefficients for area-source initial-noise Sigma (paper §3.4 eq 3.21).

        Σ_A(T) = 2 · ∫₀¹ T_s ds · I_d, where T_s = Σ_{r ≤ s} A_r.

        Discretizing [0, 1] into T_total uniform steps Δ = 1/T_total and
        swapping summation order:

            Σ_A ≈ 2 · Σ_s T_s · Δ
                = 2 · Σ_s (Σ_{r ≤ s} A_r) · Δ
                = 2 · Σ_r A_r · (T_total − r) · Δ
                = Σ_r A_r · area_coef[r],
            area_coef[r] = 2 · (T_total − r) / T_total.

        Linear decay (early increments contribute most because they live
        inside T_s for the longest time). Cached per (T, device).

        Returns: 1-D tensor of length T_total on self.device.
        """
        cache = getattr(self, '_area_noise_coef_cache', None)
        if (cache is not None
                and cache.shape[0] == T_total
                and cache.device == torch.device(self.device)):
            return cache
        r = torch.arange(T_total, device=self.device, dtype=torch.float32)
        coef = 2.0 * (T_total - r) / float(T_total)
        self._area_noise_coef_cache = coef
        return coef

    def get_init_noise_coef(self, T_total):
        """Precomputed coefficients for forward_process initial-noise Sigma.

        The recurrence
            Sigma_s = sigmas[s]^2 * A[s] + gammas[s]^2 * Sigma_{s-1},  s=1..T-1
        with Sigma_0 = sigmas[0]^2 * A[0] unrolls exactly to
            Sigma_{T-1} = sum_{s=0}^{T-1} coef[s] * A[s],
            coef[s] = sigmas[s]^2 * prod_{r=s+1..T-1} gammas[r]^2.
        Cache the result per (T, device).
        """
        cache = self._init_noise_coef_cache
        if (cache is not None
                and cache.shape[0] == T_total
                and cache.device == torch.device(self.device)):
            return cache
        sigmas = self.schedule.sigmas.to(self.device)
        gammas = self.schedule.gammas.to(self.device)
        g2 = gammas ** 2
        # sp[s] = prod_{r=s+1..T-1} gammas[r]^2; sp[T-1] = 1 (empty product).
        sp = torch.cat([
            g2[1:].flip(0).cumprod(0).flip(0),
            torch.ones(1, device=self.device, dtype=gammas.dtype),
        ])
        coef = sigmas ** 2 * sp
        self._init_noise_coef_cache = coef
        return coef

    # ---- Step C: training-time X1 + path_features production ------------

    def sample_X1_and_features(
        self, x_start, x0_ext, G,
        monte_carlo_inner, monte_carlo_outer,
    ):
        """Produce (X1, path_features) for the alpha-stable training step.

        Verbatim port of the inline alpha-stable branches from
        FlowMatchingLevyProcess.training_losses_fm. Preserves exact RNG
        order and tensor ops so bit-identity holds.

        Args:
            x_start: [B, ...] data batch
            x0_ext:  [B*total_mc, ...] = x_start.repeat(monte_carlo_outer*monte_carlo_inner, 1, 1, 1)
            G:       [B*total_mc, ...] = pre-sampled standard normal
            monte_carlo_inner / monte_carlo_outer: MC counts
        Returns:
            X1:            [B*total_mc, ...]
            path_features: [B*total_mc, fdim] or None
        """
        outer_shape = list(x_start.shape)
        outer_shape[0] *= monte_carlo_outer
        B_outer = outer_shape[0]

        # If train PathSampler is configured, delegate A (+phi) sampling to it.
        # Otherwise inline-sample as before (exact legacy behavior).
        use_train_sampler = self.train_sampler is not None
        if use_train_sampler and monte_carlo_outer > 1:
            # Safety: current sampler groups indices [0..G-1], [G..2G-1], ...
            # across B_outer. With monte_carlo_outer>1 this can mix MC replicates
            # (violating the "MC 仍独立" spec). Not supported until we add
            # per-replicate independent draws.
            raise NotImplementedError(
                f'path_sampling + monte_carlo_outer={monte_carlo_outer} > 1 '
                'is not supported yet; grouping would span MC replicates. '
                'Set monte_carlo_outer=1 or disable path_sampling.')

        if self.noise_type == 'terminal':
            # One-shot terminal A: X1 = barsigma_T * sqrt(A) * G
            if use_train_sampler:
                A, pre_phi = self.train_sampler.sample(
                    B_outer, data_shape_tail=tuple(x_start.shape[1:]))
            else:
                A = self.gen_a.generate(size=outer_shape)
                pre_phi = None
            A_ext = A.repeat(monte_carlo_inner, *([1] * len(A.shape[1:])))
            barsigma_T = self.schedule.barsigmas[-1]
            noise = barsigma_T * torch.sqrt(A_ext) * G

            if self.flow_type == 'straight_line':
                X1 = noise
            elif self.flow_type == 'ot_geodesic':
                bargamma_T = self.schedule.bargammas[-1]
                X1 = bargamma_T * x0_ext + noise
            elif self.flow_type == 'signature_linear_ve':
                # Needs full per-step T-path to compute R_m; terminal noise_type
                # only sees a single A draw, so refuse early with a clear message.
                raise ValueError(
                    "flow_type='signature_linear_ve' requires "
                    "noise_type='forward_process' (per-step A sequence to build "
                    "T_path); got noise_type='terminal'.")
            elif self.flow_type == 'tbar_linear':
                # Same reason: tbar_linear walks the cumulative T_path, so the
                # forward_process per-step A sequence is required.
                raise ValueError(
                    "flow_type='tbar_linear' requires "
                    "noise_type='forward_process' (per-step A sequence to "
                    "build T_path); got noise_type='terminal'.")
            else:
                raise ValueError(f"Unknown flow_type: {self.flow_type}")

            # Path features from terminal A
            path_features = None
            if pre_phi is not None:
                path_features = pre_phi
                if monte_carlo_inner > 1:
                    path_features = path_features.repeat(monte_carlo_inner, 1)
            elif self.path_encoder is not None and self.path_encoder.enabled:
                A_scalar = A.reshape(A.shape[0], -1).mean(dim=-1)
                t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_terminal_A(
                    self.schedule.barsigmas, A_scalar)
                path_features = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)
                if monte_carlo_inner > 1:
                    path_features = path_features.repeat(monte_carlo_inner, 1)

        elif self.noise_type == 'forward_process':
            # Per-step A sequence: sample A_0,...,A_{T-1}, compute Sigma_T via recurrence
            T_total = len(self.schedule.sigmas)
            sigmas = self.schedule.sigmas.to(self.device)
            gammas = self.schedule.gammas.to(self.device)

            if use_train_sampler:
                A_seq_dev, pre_phi = self.train_sampler.sample(B_outer)
            elif self.path_pool is not None:
                # Draw B_outer indices from the fixed pool.
                A_seq_dev = self.path_pool.sample_forward_process(B_outer, device=self.device)
                pre_phi = None
            else:
                # Single batched scipy call is ~13x faster than T_total sequential calls.
                A_flat = self.gen_a.generate(size=[T_total * B_outer])
                A_sequence = A_flat.reshape(T_total, B_outer)  # [T_total, B_outer]
                A_seq_dev = A_sequence.to(self.device)
                pre_phi = None

            # Closed-form equivalent of the scalar recurrence
            # Sigma_t = sigma_t^2 * A_t + gamma_t^2 * Sigma_{t-1}
            # unrolls to Sigma = sum_s coef[s] * A[s]. Same numerics as sample().
            coef = self.get_init_noise_coef(T_total)  # [T]
            Sigma = (coef.unsqueeze(-1) * A_seq_dev).sum(dim=0)  # [B_outer]

            # Cache cumulative subordinator T-path for downstream consumers
            # (e.g. SignatureLinearVESchedule via clock.compute_R_m).
            # A_seq_dev: [T_total, B_outer] -> T_path: [B_outer, T_total].
            self._last_T_path = torch.cumsum(A_seq_dev, dim=0).t().contiguous()

            Sigma_ext = Sigma.repeat(monte_carlo_inner)
            Sigma_bc = Sigma_ext.view(-1, *([1] * (len(x_start.shape) - 1)))

            if self.flow_type in ('straight_line', 'signature_linear_ve', 'tbar_linear'):
                # straight_line / signature_linear_ve / tbar_linear all share
                # the same noise endpoint X_1 = sqrt(Sigma) * G; the trajectory
                # difference lives in how X_t and v* are built downstream.
                X1 = torch.sqrt(Sigma_bc) * G
            elif self.flow_type == 'ot_geodesic':
                bargamma_T = self.schedule.bargammas[-1]
                X1 = bargamma_T * x0_ext + torch.sqrt(Sigma_bc) * G
            else:
                raise ValueError(f"Unknown flow_type: {self.flow_type}")

            # Path features from actual per-step A sequence
            path_features = None
            if pre_phi is not None:
                path_features = pre_phi
                if monte_carlo_inner > 1:
                    path_features = path_features.repeat(monte_carlo_inner, 1)
            elif self.path_encoder is not None and self.path_encoder.enabled:
                t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
                    sigmas, gammas, A_seq_dev)
                path_features = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)
                if monte_carlo_inner > 1:
                    path_features = path_features.repeat(monte_carlo_inner, 1)

        elif self.noise_type == 'area':
            # Area-source covariance Σ_A = 2 ∫T_s ds (paper §3.4). Same
            # A-sequence sampling as forward_process; only the per-step
            # weighting differs (uniform-time area integral instead of the
            # DLPM schedule recurrence).
            T_total = len(self.schedule.sigmas)

            if use_train_sampler:
                A_seq_dev, pre_phi = self.train_sampler.sample(B_outer)
            elif self.path_pool is not None:
                A_seq_dev = self.path_pool.sample_forward_process(B_outer, device=self.device)
                pre_phi = None
            else:
                A_flat = self.gen_a.generate(size=[T_total * B_outer])
                A_sequence = A_flat.reshape(T_total, B_outer)
                A_seq_dev = A_sequence.to(self.device)
                pre_phi = None

            area_coef = self.get_area_noise_coef(T_total)
            Sigma = (area_coef.unsqueeze(-1) * A_seq_dev).sum(dim=0)  # [B_outer]

            # Same T-path cache as forward_process (cumsum of A); features
            # downstream stay compatible.
            self._last_T_path = torch.cumsum(A_seq_dev, dim=0).t().contiguous()

            Sigma_ext = Sigma.repeat(monte_carlo_inner)
            Sigma_bc = Sigma_ext.view(-1, *([1] * (len(x_start.shape) - 1)))

            if self.flow_type in ('straight_line', 'signature_linear_ve', 'tbar_linear'):
                X1 = torch.sqrt(Sigma_bc) * G
            elif self.flow_type == 'ot_geodesic':
                bargamma_T = self.schedule.bargammas[-1]
                X1 = bargamma_T * x0_ext + torch.sqrt(Sigma_bc) * G
            else:
                raise ValueError(f"Unknown flow_type: {self.flow_type}")

            # Path features: reuse compute_clock_path_from_sigmas for shape
            # compatibility (T_raw = cumsum(A) is the right signal regardless
            # of how Σ is summed downstream).
            path_features = None
            if pre_phi is not None:
                path_features = pre_phi
                if monte_carlo_inner > 1:
                    path_features = path_features.repeat(monte_carlo_inner, 1)
            elif self.path_encoder is not None and self.path_encoder.enabled:
                sigmas = self.schedule.sigmas.to(self.device)
                gammas = self.schedule.gammas.to(self.device)
                t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
                    sigmas, gammas, A_seq_dev)
                path_features = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)
                if monte_carlo_inner > 1:
                    path_features = path_features.repeat(monte_carlo_inner, 1)

        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        return X1, path_features

    # ---- Step D: inference-time initial state + path features ----------

    def sample_initial(self, shape):
        """Produce (x, A_sample, G, path_features) for the alpha-stable
        inference-time initial state (K_vel=1 case; velocity averaging
        moves here in Step F).

        Verbatim port of the forward_process / terminal branches in
        FlowMatchingLevyProcess.sample(), plus the path_features_sample
        resolution that follows them. G is returned so the host can keep
        printing the [path_fingerprint] line unchanged.

        Args:
            shape: list/tuple. Output tensor shape.
        Returns:
            x:               [B, ...] initial noise endpoint
            A_sample:        the A draws used (for fingerprint log)
            G:               the standard-normal draw used (for fingerprint log)
            path_features:   [B, fdim] or None
        """
        if self.noise_type in ('forward_process', 'area'):
            T_total = len(self.schedule.sigmas)
            B = shape[0]
            if self.eval_sampler is not None:
                A_sample, pre_phi = self.eval_sampler.sample(B)
            elif self.path_pool is not None:
                A_sample = self.path_pool.sample_forward_process(B, device=self.device)
                pre_phi = None
            else:
                A_flat = self.gen_a.generate(size=[T_total * B])
                A_sample = A_flat.reshape(T_total, B).to(self.device)
                pre_phi = None

            # Pick the per-step weighting from noise_type:
            #   forward_process: DLPM schedule recurrence (single-sum form)
            #   area:           paper §3.4 area integral 2·∫T_s ds
            if self.noise_type == 'forward_process':
                coef = self.get_init_noise_coef(T_total)
            else:
                coef = self.get_area_noise_coef(T_total)
            Sigma = (coef.unsqueeze(-1) * A_sample).sum(dim=0)

            G = torch.randn(shape, device=self.device)
            Sigma_bc = Sigma.view(-1, *([1] * (len(shape) - 1)))
            x = torch.sqrt(Sigma_bc) * G
        else:
            # 'terminal' (or anything non-forward-process)
            if self.eval_sampler is not None:
                A_sample, pre_phi = self.eval_sampler.sample(
                    shape[0], data_shape_tail=tuple(shape[1:]))
            else:
                A_sample = self.gen_a.generate(size=shape)
                pre_phi = None
            G = torch.randn(shape, device=self.device)
            x = self.schedule.barsigmas[-1] * torch.sqrt(A_sample) * G

        # Resolve path_features_sample (formerly inline after the branch).
        if pre_phi is not None:
            path_features = pre_phi
        else:
            path_features = None
            if self.path_encoder is not None and self.path_encoder.enabled:
                if self.noise_type in ('forward_process', 'area'):
                    t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
                        self.schedule.sigmas.to(self.device),
                        self.schedule.gammas.to(self.device),
                        A_sample)
                    path_features = self.path_encoder(
                        t_grid, T_raw=T_raw, T_effective=T_eff)
                elif A_sample is not None:
                    A_scalar = A_sample.reshape(A_sample.shape[0], -1).mean(dim=-1)
                    t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_terminal_A(
                        self.schedule.barsigmas, A_scalar)
                    path_features = self.path_encoder(
                        t_grid, T_raw=T_raw, T_effective=T_eff)

        return x, A_sample, G, path_features
