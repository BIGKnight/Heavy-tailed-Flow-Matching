"""DLPM forward-process dynamics.

Holds the methods that describe the discrete-time forward process and its
inverses (predict_xstart from eps, etc.):

  - predict_xstart            inverse of x_t = bg[t]*x_0 + bs[t]*eps
  - predict_eps               same equation solved for eps
  - predict_eps_from_m_tilde  recovers eps from anterior mean (uses Gamma_t,
                              hence depends on the posterior mixin)
  - sample_x_t_from_xstart    forward q(x_t | x_0) sampler
  - sample_A                  draw a full A_1..A_T sequence, store on self.A
  - compute_Sigmas            populate self.Sigmas from self.A via recurrence
  - sample_x_t_from_xstart_given_Sigma   q(x_t | x_0, Sigma_t) sampler

These are mixed into DLPM as a separate concern from posterior calcs and
schedule arithmetic. Each method references attributes living on the
DLPM instance: update_constants, get_t_to_batch_size, gen_eps, gen_a,
compute_Gamma_t (from posterior mixin), A, Sigmas.
"""

import torch


class DLPMDynamicsMixin:
    """Mixin providing forward-dynamics methods. DLPM inherits this."""

    # ---- Forward q(x_t | x_0) without conditioning on A_{1:T} ----------

    def predict_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        xstart = (x_t - eps * bs[t]) / bg[t]
        return xstart

    def predict_eps(self, x_t, t, xstart):
        g, bg, s, bs = self.update_constants(x_t.shape)
        t = self.get_t_to_batch_size(x_t, t)
        eps = (x_t - xstart * bg[t]) / bs[t]
        return eps

    def predict_eps_from_m_tilde(self, x_t, t, m_tilde_t_1):
        g, bg, s, bs = self.update_constants(m_tilde_t_1.shape)
        t = self.get_t_to_batch_size(m_tilde_t_1, t)
        # Note: needs compute_Gamma_t from DLPMPosteriorMixin (also on DLPM).
        Gamma_t = self.compute_Gamma_t(t, self.Sigmas[t-1], self.Sigmas[t])
        # m_tilde_t_1 = (x_t - bs[t]*Gamma_t*eps) / g[t]  =>  eps = ...
        eps = (x_t - m_tilde_t_1 * g[t]) / (bs[t] * Gamma_t)
        return eps

    def sample_x_t_from_xstart(self, xstart, t, eps=None):
        g, bg, s, bs = self.update_constants(xstart.shape)
        t = self.get_t_to_batch_size(xstart, t)
        if eps is None:
            eps = self.gen_eps.generate(size=xstart.size())
        x_t = bg[t] * xstart + bs[t] * eps
        return x_t, eps

    # ---- Dynamics conditioned on the A_{1:T} sequence ------------------

    def sample_A(self, shape, diffusion_steps):
        """Sample a full per-step A_1..A_T sequence; store on self.A."""
        self.A = torch.stack([
            self.gen_a.generate(size=shape) for i in range(diffusion_steps)
        ])

    def compute_Sigmas(self):
        """Populate self.Sigmas from self.A via the recurrence
        Sigma_t = s_t^2 * A_t + g_t^2 * Sigma_{t-1}.
        """
        g, bg, s, bs = self.update_constants(self.A[0].shape)
        t = self.get_t_to_batch_size(self.A[0], 0)
        Sigmas = [s[0]**2 * self.A[0]]
        for t in range(1, self.A.shape[0]):
            A_t = self.A[t]
            t = self.get_t_to_batch_size(self.A[0], t)
            Sigmas.append(s[t]**2 * A_t + g[t]**2 * Sigmas[-1])
        self.Sigmas = torch.stack(Sigmas)

    def sample_x_t_from_xstart_given_Sigma(self, xstart, t, Sigma_t, z_t=None):
        """Forward q(x_t | x_0, Sigma_t) using a given Sigma_t (rather than
        querying the schedule). Used by the one-rv training loss helper."""
        g, bg, s, bs = self.update_constants(xstart.shape)
        t = self.get_t_to_batch_size(xstart, t)
        if z_t is None:
            z_t = torch.randn_like(xstart)
        x_t = bg[t] * xstart + Sigma_t**(1/2) * z_t
        return x_t
