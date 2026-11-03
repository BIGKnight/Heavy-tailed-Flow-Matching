"""GaussianClock: vanilla flow matching (no random clock, no heavy-tailed scale).

This clock degenerates X_1 to plain standard Gaussian noise:
    X_1 = G   with   G ~ N(0, I)

i.e. T_t = t deterministically (no subordinator), bar_T_t = t, and the flow
reduces to the standard linear flow matching baseline:
    X_t = (1 - t) * X_0 + t * X_1
    v*  = X_1 - X_0

There is no path conditioning and no clock-derived feature — the model
sees only (X_t, t) at training and inference time.

Compatible flow_types:
  - 'straight_line' (canonical vanilla FM)
  - 'tbar_linear' (degenerates to straight_line because bar_T_t = t exactly)

Incompatible:
  - 'ot_geodesic' — needs sigma/gamma scaling that doesn't exist for vanilla
  - 'signature_linear_ve' — needs Lévy-area R_m which has no source here
"""
import torch

from .base import Clock


class GaussianClock(Clock):
    """Vanilla Gaussian noise endpoint, no clock state, no path features.

    No __init__ arguments other than device. Stateless across calls.
    """

    feature_dim = 0
    supports_path_pool = False
    supports_path_sampling = False

    def __init__(self, device):
        self.device = device

    def __repr__(self):
        return 'GaussianClock(vanilla, no clock state)'

    # ---- Training-time -------------------------------------------------

    def sample_X1_and_features(
        self, x_start, x0_ext, G,
        monte_carlo_inner, monte_carlo_outer,
    ):
        """Vanilla noise endpoint.

        X_1 = G  (G is the [B*total_mc, ...] standard normal already drawn
        by FlowMatchingLevyProcess.training_losses_fm; we just return it
        unchanged so the host's MC bookkeeping stays valid).

        No path features.
        """
        return G, None

    # ---- Inference-time ------------------------------------------------

    def sample_initial(self, shape):
        """Vanilla initial state at t=1: X_1 = G ~ N(0, I).

        Returns (x, A_sample, G, path_features) following the Clock
        interface contract. A_sample is None (no subordinator); G is
        returned for the [path_fingerprint] log; path_features is None.
        """
        G = torch.randn(shape, device=self.device)
        return G, None, G, None

    # ---- tbar_linear flow support --------------------------------------
    # bar_T_t = t exactly (no random clock), bar_T_dot_t = 1.
    # Mirrors the StudentTClock degenerate path so flow_type='tbar_linear'
    # works uniformly across clock types.

    def sample_t_for_tbar_linear(self, B_full):
        """Continuous t in (0, 1); no grid alignment needed since bar_T = t."""
        return torch.rand(int(B_full), device=self.device) * 0.998 + 0.001

    def compute_bar_T_at(self, t):
        """bar_T_t = t,  bar_T_dot_t = 1.  Degenerate (no warp)."""
        return t, torch.ones_like(t)

    # ---- Disallowed APIs (loud errors instead of silent wrong answers) -

    @staticmethod
    def compute_R_m(T_path):
        """Vanilla clock has no T_path; signature_linear_ve schedule must
        not be used with this clock_type.
        """
        raise NotImplementedError(
            "GaussianClock has no T_path; flow_type='signature_linear_ve' "
            "requires clock_type='alpha_stable'.")

    # No ckpt_metadata_kwargs override — base class default `{}` is correct
    # (no hyperparams to embed). No set_path_encoder override — base class
    # no-op default is correct (no encoder needed for vanilla FM).
