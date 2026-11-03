"""Dormand-Prince 5(4) adaptive via torchdiffeq. Embedded 4-5 order with step control.

Semantics
---------
- If step_hook is None: collapse to a single adaptive integration from t_start to t_end
  regardless of N. The adaptive stepper controls its own substeps; the user-facing
  `reverse_steps` would only force artificial restarts (each adaptive call re-bootstraps
  with a conservative initial step, wasting NFEs). When no hook is active there is
  nothing for N to gate, so we ignore it.
- If step_hook is set: do N adaptive sub-integrations of length dt each, applying
  step_hook between. This is how we support per-step clipping/history while keeping
  internal adaptivity within each sub-interval.

Kwargs (via yaml eval.fm.solver_kwargs or defaults):
    rtol: relative tolerance (default 1e-5)
    atol: absolute tolerance (default 1e-5)
"""
from .base import ODESolver


class DopriSolver(ODESolver):
    name = 'dopri5'
    nfe_per_step = 6  # approximate; actual NFE is adaptive

    def _adaptive_integrate(self, v_fn, x, t0, t1, rtol, atol):
        """One adaptive ODE integration from t0 to t1 (t0 may be > or < t1)."""
        import torch
        from torchdiffeq import odeint

        def f(s, y):
            # torchdiffeq feeds scalar tensor s; our v_fn expects Python float.
            return v_fn(y, float(s.item()))

        ts = torch.tensor([t0, t1], dtype=x.dtype, device=x.device)
        y_path = odeint(f, x, ts, method='dopri5', rtol=rtol, atol=atol)
        return y_path[-1]

    def step(self, v_fn, x, t, dt, rtol=1e-5, atol=1e-5):
        return self._adaptive_integrate(v_fn, x, t, t - dt, rtol, atol)

    def integrate(self, v_fn, x0, N, t_start=1.0, t_end=0.0, step_hook=None,
                  rtol=1e-5, atol=1e-5, **_ignored):
        # No hook -> collapse to one adaptive call; N is meaningless here.
        if step_hook is None:
            return self._adaptive_integrate(v_fn, x0, t_start, t_end, rtol, atol)

        dt = (t_start - t_end) / N
        x = x0
        for i in range(N):
            t = t_start - i * dt
            x = self._adaptive_integrate(v_fn, x, t, t - dt, rtol, atol)
            x = step_hook(x, i)
        return x
