"""VE-type signature-linear schedule (paper §4 trajectory ablation).

This is a *flow-path interpolation schedule*, NOT a sigma/gamma noise schedule.
It controls how X_t is interpolated between X_0 (noise) and X_1 (clock-noisy
data) along an affine trajectory whose coefficients carry a small clock-shape
correction:

    alpha_t = t      + lambda_alpha * t * (1 - t) * R_m
    beta_t  = (1-t)  * (1 + lambda_beta * t * R_m)

with R_m in [-1, 1] the signed Lévy-area projection of the normalized clock
path (see AlphaStableClock.compute_R_m).

When lambda_alpha = lambda_beta = 0, this degenerates to the standard linear
VE path alpha_t = t, beta_t = 1 - t. The endpoint conditions
alpha_0 = 0, alpha_1 = 1, beta_0 = 1, beta_1 = 0 hold for any R_m and
any lambda values.

Admissibility constraint: |lambda_beta| < 1 ensures
beta_t > (1-t)(1-|lambda_beta|) > 0 for t < 1.
"""

import torch


class SignatureLinearVESchedule:
    """VE-type signature-linear schedule.

    Stateless w.r.t. (t, R_m). Constructed once with the two lambda hyperparameters
    and queried per training step.
    """

    def __init__(self, lambda_alpha=0.25, lambda_beta=0.25):
        self.la = float(lambda_alpha)
        self.lb = float(lambda_beta)
        assert abs(self.lb) < 1.0, (
            f"|lambda_beta| = {abs(self.lb)} must be < 1 for admissibility "
            f"(beta_t > (1-t)(1-|lambda_beta|) > 0 for t<1)"
        )

    def __repr__(self):
        return (f"SignatureLinearVESchedule("
                f"lambda_alpha={self.la}, lambda_beta={self.lb})")

    # ---- Coefficient curves -------------------------------------------------

    def alpha_t(self, t, R_m):
        """alpha_t = t + lambda_alpha * t * (1-t) * R_m."""
        return t + self.la * t * (1.0 - t) * R_m

    def beta_t(self, t, R_m):
        """beta_t = (1-t) * (1 + lambda_beta * t * R_m)."""
        return (1.0 - t) * (1.0 + self.lb * t * R_m)

    def alpha_dot(self, t, R_m):
        """d alpha_t / dt = 1 + lambda_alpha * (1 - 2t) * R_m."""
        return 1.0 + self.la * (1.0 - 2.0 * t) * R_m

    def beta_dot(self, t, R_m):
        """d beta_t / dt = -1 + lambda_beta * (1 - 2t) * R_m."""
        return -1.0 + self.lb * (1.0 - 2.0 * t) * R_m

    # ---- Affine target velocity --------------------------------------------

    def affine_target(self, t, R_m, x_t, x_1):
        """v*(t, X_t, X_1) = (beta_dot/beta) * X_t + (alpha_dot - (beta_dot/beta) * alpha) * X_1.

        Args:
          t:    [B] timesteps in [0, 1]
          R_m:  [B] clock-shape scores in [-1, 1]
          x_t:  [B, ...] interpolated state
          x_1:  [B, ...] data endpoint (clock-noisy data)

        Returns:
          [B, ...] target velocity v*.
        """
        spatial = (1,) * (x_t.dim() - 1)
        t_b = t.view(-1, *spatial)
        R_b = R_m.view(-1, *spatial)
        a = self.alpha_t(t_b, R_b)
        b = self.beta_t(t_b, R_b)
        a_dot = self.alpha_dot(t_b, R_b)
        b_dot = self.beta_dot(t_b, R_b)
        # Guard against beta -> 0 at t -> 1 (division). Since training samples
        # t in (0, 1), this should never trigger in practice, but the clamp
        # makes the function safe for any t in [0, 1].
        L = b_dot / b.clamp(min=1e-8)
        return L * x_t + (a_dot - L * a) * x_1

    # ---- Convenience: build X_t given X_0, X_1, t, R_m ----------------------

    def X_t_from_endpoints(self, t, R_m, x_0, x_1):
        """X_t = alpha_t * X_1 + beta_t * X_0 (affine flow path)."""
        spatial = (1,) * (x_0.dim() - 1)
        t_b = t.view(-1, *spatial)
        R_b = R_m.view(-1, *spatial)
        a = self.alpha_t(t_b, R_b)
        b = self.beta_t(t_b, R_b)
        return a * x_1 + b * x_0
