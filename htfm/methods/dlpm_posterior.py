"""DLPM reverse-process posterior calculations.

Holds the 5 methods that together describe the discrete-time anterior
posterior  q(x_{t-1} | x_t, x_0, A_{1:T})  used during DLPM-style sampling
(both the standard DLPM path and the DDIM/DLIM variant):

  - compute_Gamma_t         — shrinkage factor 1 - g_t^2 * Sigma_{t-1}/Sigma_t
  - compute_Sigma_tilde_t_1 — anterior variance Gamma * Sigma_{t-1}
  - compute_m_tilde_t_1     — anterior mean (x_t - bs[t]*Gamma*eps) / g[t]
  - anterior_mean_variance_dlpm — full DLPM anterior (uses self.Sigmas[t-1, t])
  - anterior_mean_variance_dlim — DDIM-style anterior with η stochasticity

These are mixed into DLPM as a separate concern from forward dynamics
(predict_xstart / sample_x_t_*) and noise-schedule arithmetic. Each method
references attributes that live on the DLPM instance: `update_constants`,
`get_t_to_batch_size`, `Sigmas`, `A`, `alpha`.
"""


class DLPMPosteriorMixin:
    """Mixin providing the 5 posterior methods. DLPM inherits this."""

    def compute_Gamma_t(self, t, Sigma_t_1, Sigma_t):
        g, bg, s, bs = self.update_constants(Sigma_t_1.shape)
        t = self.get_t_to_batch_size(Sigma_t_1, t)
        Gamma_t = 1 - (g[t]**2 * Sigma_t_1) / Sigma_t
        return Gamma_t

    def compute_Sigma_tilde_t_1(self, Gamma_t, Sigma_t_1):
        return Gamma_t * Sigma_t_1

    def compute_m_tilde_t_1(self, x_t, t, Gamma_t, eps_t):
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        m_tilde_t_1 = (x_t - bs[t] * Gamma_t * eps_t) / g[t]
        return m_tilde_t_1

    def anterior_mean_variance_dlpm(self, x_t, t, eps):
        """Standard DLPM anterior posterior (DDPM-style with alpha-stable noise)."""
        g, bg, s, bs = self.update_constants(x_t.shape)
        Gamma_t = self.compute_Gamma_t(t, self.Sigmas[t-1], self.Sigmas[t])
        t = self.get_t_to_batch_size(x_t, t)
        x_t_1 = (x_t - bs[t] * Gamma_t * eps) / g[t]
        Sigma_t_1 = self.compute_Sigma_tilde_t_1(Gamma_t, self.Sigmas[t-1])
        return x_t_1, Sigma_t_1

    def anterior_mean_variance_dlim(self, x_t, t, eps, eta=0.0):
        """DLIM (DDIM-style) anterior with eta-controlled stochasticity."""
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        # No noise when t == 1.
        nonzero_mask = (t != 1).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        if eta == 0.0:
            sample = (x_t - bs[t] * eps) / g[t] + bs[t-1] * eps
            return sample, 0

        sigma_t = eta * bs[t-1]
        sample = (x_t - bs[t] * eps) / g[t]
        sample += (bs[t-1]**(self.alpha) - sigma_t**(self.alpha))**(1 / self.alpha) * eps
        mean = sample
        variance = nonzero_mask * sigma_t**2 * self.A[t]
        return sample, variance
