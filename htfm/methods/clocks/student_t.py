"""Student-t random clock for heavy-tailed flow matching.

Paper §3.3 (Heavy_tailed_Flow_Matching_Model):

    V_nu ~ InvGamma(nu/2, nu/2)          # single latent scalar per sample
    T_t  = t * V_nu                       # deterministic path given V_nu
    Sigma(T_[0,1]) = V_nu * I_d           # terminal variance is just V_nu
    X_t | X_1 = x_1, V_nu ~ N(alpha_t * x_1, beta_t^2 * V_nu * I_d)

After marginalizing over V_nu, X_t | X_1 is multivariate Student-t with
location alpha_t x_1, scale beta_t^2 I_d, and nu degrees of freedom.

Code convention (reversed from paper): t=0 is data, t=1 is noise.
  * x_1  = sqrt(V_nu) * G    (noise endpoint, paper's X_0)
  * x_0 ~ p_data             (data endpoint, paper's X_1)
  * Straight-line: X_t = (1-t) x_0 + t x_1,  v* = x_1 - x_0.

Because the entire clock path is determined by a single scalar V_nu, path
conditioning only needs V_nu (or a simple transform of it) as the feature —
no logsignature over a T-step sequence.
"""

import torch
import torch.distributions as D

from .base import Clock


class StudentTClock(Clock):

    # Path-pool / sampler attrs are NOT applicable to Student-t (single-scalar
    # clock); inherit None from Clock base. Asserts in sample_X1_and_features
    # / sample_initial enforce config sanity.

    # v15.1: Student-t does NOT accept path_pool or (G, M) path_sampling
    # configs. Explicit overrides (== False inheritance) for documentation.
    supports_path_pool = False
    supports_path_sampling = False

    def __init__(self, nu, device, feature_mode='log_scalar',
                 clamp_V_max=None, clamp_V_min=None):
        """
        Args:
          nu: float > 0. Degrees of freedom. Smaller nu = heavier tail.
          device: torch device for generated tensors.
          feature_mode: how to encode V_nu as a path_feature vector.
            'scalar'     -> [V]          shape [B, 1]
            'log_scalar' -> [log V]      shape [B, 1]  (recommended)
            'both'       -> [V, log V]   shape [B, 2]
          clamp_V_max: optional upper clamp on V_nu draws (training stability
            for small nu where InvGamma right tail is extremely heavy).
          clamp_V_min: optional lower clamp on V_nu draws.
        """
        assert nu > 0, f'nu must be > 0, got {nu}'
        self.nu = float(nu)
        self.device = device
        assert feature_mode in ('scalar', 'log_scalar', 'both'), feature_mode
        self.feature_mode = feature_mode
        self.clamp_V_max = float(clamp_V_max) if clamp_V_max is not None else None
        self.clamp_V_min = float(clamp_V_min) if clamp_V_min is not None else None
        # Gamma(shape, rate); InvGamma(a, b) = 1 / Gamma(a, 1/b).
        # For V ~ InvGamma(nu/2, nu/2), V = 1 / G where G ~ Gamma(nu/2, nu/2).
        # Using rate parameterization: Gamma(concentration=nu/2, rate=nu/2).
        self._gamma_dist = D.Gamma(torch.tensor(self.nu / 2.0),
                                   torch.tensor(self.nu / 2.0))

    @property
    def feature_dim(self):
        return 2 if self.feature_mode == 'both' else 1

    # -- Sampling ------------------------------------------------------------

    def sample_V(self, batch_size):
        """Sample V_nu ~ InvGamma(nu/2, nu/2). Returns [batch_size] tensor on device.
        Optionally clamps V to [clamp_V_min, clamp_V_max] before returning.
        """
        g = self._gamma_dist.sample((int(batch_size),))
        V = 1.0 / g
        if self.clamp_V_max is not None or self.clamp_V_min is not None:
            V = V.clamp(min=self.clamp_V_min, max=self.clamp_V_max)
        return V.to(self.device)

    def compute_x1(self, V, shape, G):
        """x_1 = sqrt(V) * G, broadcasting V across non-batch dims.

        Args:
          V:     [B] scalar-per-sample
          shape: full tensor shape like [B, C, H, W]
          G:     [B, C, H, W] Gaussian ~ N(0, I)
        Returns:
          x_1:   same shape as G
        """
        assert V.shape == (shape[0],), f'V shape {V.shape} != ({shape[0]},)'
        assert G.shape == tuple(shape), f'G shape {G.shape} != {tuple(shape)}'
        V_bc = V.view(-1, *([1] * (len(shape) - 1)))
        return torch.sqrt(V_bc) * G

    # -- Path feature --------------------------------------------------------

    def encode_feature(self, V):
        """Map V_nu [B] to path_feature [B, feature_dim]."""
        if self.feature_mode == 'scalar':
            return V.view(-1, 1)
        if self.feature_mode == 'log_scalar':
            return torch.log(V).view(-1, 1)
        # 'both'
        return torch.stack([V, torch.log(V)], dim=-1)

    def __repr__(self):
        return (f'StudentTClock(nu={self.nu}, feature_mode={self.feature_mode!r}, '
                f'feat_dim={self.feature_dim})')

    @staticmethod
    def compute_R_m(T_path):
        """Student-t has no native subordinator path (only a single scalar V).
        The signature_linear_ve schedule requires alpha_stable clock.
        """
        raise NotImplementedError(
            "Student-t clock has no native clock path; signature_linear_ve "
            "schedule requires fm.clock.type=alpha_stable.")

    # ---- tbar_linear flow support ---------------------------------------
    # For Student-t: T_t = t * V_nu, so bar_T_t = T_t / T_1 = t exactly
    # (independent of V_nu). The flow X_t = bar_T_t*X_1 + (1-bar_T_t)*X_0
    # therefore degenerates to plain straight-line flow. We still expose the
    # interface so YAMLs can use flow_type='tbar_linear' uniformly across
    # clock_types for cleaner experiment matrices.

    def sample_t_for_tbar_linear(self, B_full):
        """Continuous t in (0, 1); no grid since bar_T_t = t is exact for any t."""
        return torch.rand(int(B_full), device=self.device) * 0.998 + 0.001

    def compute_bar_T_at(self, t):
        """For Student-t, bar_T_t = t and bar_T_dot_t = 1 (degenerate to straight_line)."""
        return t, torch.ones_like(t)

    # ---- Step E: uniform Clock interface --------------------------------

    @property
    def ckpt_metadata_kwargs(self):
        """Hyperparams to embed in ckpt 'extra' for type/kwargs validation."""
        return {'nu': self.nu, 'feature_mode': self.feature_mode}

    def sample_X1_and_features(
        self, x_start, x0_ext, G,
        monte_carlo_inner, monte_carlo_outer,
    ):
        """Training-time. Verbatim port of the inline Student-t branch in
        FlowMatchingLevyProcess.training_losses_fm. Returns (X1, path_features).
        """
        # Student-t requires straight_line + (G=M=1) + no pool. This is a
        # property of the math (single scalar V; no A sequence to subgroup).
        # The host previously asserted these in __init__; replicated here so
        # routing through the Clock interface stays loud on misconfiguration.
        # (Host's __init__ may also still assert; harmless.)
        B_outer = monte_carlo_outer * x_start.shape[0]
        V = self.sample_V(B_outer)                                 # [B_outer]
        # Expand V to match G's full MC_outer*MC_inner leading dim.
        V_ext = V.repeat(monte_carlo_inner)                         # [total_mc * B]
        X1 = self.compute_x1(V_ext, list(G.shape), G)              # [total_mc*B, *]
        # Path feature from the clock; V_ext expansion handles MC_inner replication
        # already via stacking logic (repeat_interleave preserves sample order).
        path_features = self.encode_feature(V)                      # [B_outer, feat_dim]
        if monte_carlo_inner > 1:
            path_features = path_features.repeat(monte_carlo_inner, 1)
        return X1, path_features

    def sample_initial(self, shape):
        """Inference-time. Verbatim port of the inline Student-t branch in
        FlowMatchingLevyProcess.sample.

        Returns (x, A_sample, G, path_features). A_sample is V here (used as
        a placeholder for the [path_fingerprint] log; it is NOT an alpha-stable
        increment sequence).
        """
        B = shape[0]
        V = self.sample_V(B)                                  # [B]
        G = torch.randn(shape, device=self.device)
        x = self.compute_x1(V, list(shape), G)
        # Pre-resolved path_features (V→encode); host won't recompute.
        pre_phi = self.encode_feature(V)                      # [B, feat_dim]
        # Use V as A_sample placeholder for fingerprint logging.
        A_sample = V
        return x, A_sample, G, pre_phi
