"""Policy-driven path (A, φ) sampler for heavy-tailed flow matching.

Decouples two orthogonal variance-reduction knobs:

  - group_size G: within a call of batch_size B, samples are partitioned
    into B/G groups; each group shares a single A realization (hence
    the same path feature φ). G=1 is the default (per-sample paths);
    G=B means the whole batch shares one path.

  - resample_interval M: A/φ are cached across M consecutive calls
    and reused. On the M-th reuse the cache is invalidated and a
    fresh draw replaces it. M=1 is the default (resample every call).

Shared quantity: A (the heavy-tailed stochastic clock). The caller
still samples its own Gaussian G, so the marginal distribution of
x_1 = sqrt(Σ(A)) · G is correct; within a group the x_1 samples
are correlated via common A (common random numbers for variance
reduction).

Cache invalidation rules (conservative):
  - batch_size changed between calls → invalidate (fresh draw for new B).
  - iter_since_resample reached M → invalidate.
  - cache is None (first call) → draw.

The sampler also produces the matching path_features φ by calling
the provided path_encoder; grouping naturally carries into φ since φ
is a deterministic function of A per sample.
"""

import torch


class PathSampler:

    def __init__(
        self,
        group_size,
        resample_interval,
        gen_a,
        path_encoder,
        schedule,
        noise_type,
        device,
        context='sampler',
        path_pool=None,
    ):
        """
        Args:
          group_size:        int ≥ 1. Samples per group sharing the same A.
          resample_interval: int ≥ 1. Max calls a cached (A, φ) is reused.
          gen_a:             heavy-tailed subordinator generator.
          path_encoder:      PathFeatureEncoder instance, or None.
                             If None/disabled, φ returned as None.
          schedule:          NoiseSchedule (sigmas / gammas / barsigmas).
          noise_type:        'forward_process' or 'terminal'.
          device:            torch device for tensor construction.
          context:           short label for log messages, e.g. 'train'/'eval'.
          path_pool:         optional PathPool to draw A sequences from. When
                             provided and noise_type=='forward_process',
                             grouped A samples come from the pool instead of
                             fresh gen_a.generate calls. Ignored for terminal.
        """
        assert isinstance(group_size, int) and group_size >= 1, \
            f'group_size must be int ≥ 1, got {group_size}'
        assert isinstance(resample_interval, int) and resample_interval >= 1, \
            f'resample_interval must be int ≥ 1, got {resample_interval}'
        assert noise_type in ('forward_process', 'terminal', 'area'), \
            f'noise_type must be forward_process|terminal|area, got {noise_type}'
        self.group_size = group_size
        self.resample_interval = resample_interval
        self.gen_a = gen_a
        self.path_encoder = path_encoder
        self.schedule = schedule
        self.noise_type = noise_type
        self.device = device
        self.context = context
        self.path_pool = path_pool
        self._cache = None        # (A, phi) or None
        self._cached_B = None
        self._iter_since_resample = 0

    def sample(self, batch_size, data_shape_tail=()):
        """Return (A, phi) with leading batch_size of grouped realizations.

        For noise_type='forward_process', A has shape [T, batch_size]
        where T = len(dlpm.sigmas), and columns i, i+1, ..., i+G-1 are
        identical within each group.

        For noise_type='terminal', A has shape [batch_size, *data_shape_tail]
        and rows within each group are identical.

        phi has shape [batch_size, feat_dim] (or None if path_encoder off).
        """
        # Handle partial last batch: when B < group_size (dataloader tail with
        # drop_last=False), collapse to a single group of size B for this call.
        # This preserves "all-samples-share-a-path" semantics and avoids crashing.
        effective_group = self.group_size
        if batch_size < self.group_size:
            if not getattr(self, '_warned_partial_batch', False):
                print(f'[path_sampler:{self.context}] INFO: partial batch '
                      f'B={batch_size} < group_size={self.group_size}; '
                      f'collapsing to one group of size B for such batches '
                      f'(this message printed once).', flush=True)
                self._warned_partial_batch = True
            effective_group = batch_size

        assert batch_size % effective_group == 0, (
            f'[path_sampler:{self.context}] batch_size={batch_size} not '
            f'divisible by group_size={self.group_size} (effective_group='
            f'{effective_group}). Either pick B such that B % G == 0, or set '
            f'drop_last=True on the DataLoader.')

        if self._cache is None:
            need_redraw = True
        elif self._cached_B != batch_size:
            print(f'[path_sampler:{self.context}] WARN: batch_size changed '
                  f'{self._cached_B} -> {batch_size}, cache invalidated.',
                  flush=True)
            need_redraw = True
        elif self._iter_since_resample >= self.resample_interval:
            need_redraw = True
        else:
            need_redraw = False

        if need_redraw:
            self._cache = self._draw(batch_size, data_shape_tail, effective_group)
            self._cached_B = batch_size
            self._iter_since_resample = 0

        self._iter_since_resample += 1
        return self._cache

    def reset(self):
        """Force a fresh draw on the next call. Useful between train/eval."""
        self._cache = None
        self._cached_B = None
        self._iter_since_resample = 0

    # --- internals ---------------------------------------------------

    def _draw(self, B, shape_tail, effective_group=None):
        if effective_group is None:
            effective_group = self.group_size
        n_groups = B // effective_group
        if self.noise_type in ('forward_process', 'area'):
            # Both noise types consume per-step A sequences; only the Σ
            # weighting in the clock differs (handled in alpha_stable.py).
            if self.path_pool is not None:
                A_group = self.path_pool.sample_forward_process(n_groups, device=self.device)
            else:
                T = len(self.schedule.sigmas)
                A_flat = self.gen_a.generate(size=[T * n_groups])
                A_group = A_flat.reshape(T, n_groups).to(self.device)
            A = A_group.repeat_interleave(effective_group, dim=1)  # [T, B]
            phi = self._compute_phi_forward(A)
        else:
            # terminal: A shape [n_groups, *shape_tail]
            A_group = self.gen_a.generate(size=[n_groups, *shape_tail])
            if not isinstance(A_group, torch.Tensor):
                A_group = torch.as_tensor(A_group)
            A_group = A_group.to(self.device)
            A = A_group.repeat_interleave(effective_group, dim=0)  # [B, *tail]
            phi = self._compute_phi_terminal(A)
        return (A, phi)

    def _compute_phi_forward(self, A_sequence):
        if self.path_encoder is None or not self.path_encoder.enabled:
            return None
        sigmas = self.schedule.sigmas.to(self.device)
        gammas = self.schedule.gammas.to(self.device)
        t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
            sigmas, gammas, A_sequence)
        return self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)

    def _compute_phi_terminal(self, A_sample):
        if self.path_encoder is None or not self.path_encoder.enabled:
            return None
        A_scalar = A_sample.reshape(A_sample.shape[0], -1).mean(dim=-1)
        t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_terminal_A(
            self.schedule.barsigmas, A_scalar)
        return self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)
