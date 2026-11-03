"""Adams-Bashforth-2 (AB2) multistep. 2nd order, NFE=1/step (after Euler bootstrap).

Rationale: the original plan called this 'dpm2'. Strict DPM-Solver-2 assumes a noise-
parameterization and schedule-aware log-SNR midpoints that don't directly apply to
velocity-FM. AB2 is the canonical 2nd-order multistep method that fits FM-velocity
cleanly and is distinct in structure from heun/midpoint (which are single-step).

Update:
    step 0: Euler bootstrap  x_1 = x_0 - dt * v(x_0, t_0)
    step i>=1:
        v_curr = v(x_i, t_i)
        x_{i+1} = x_i - dt * (1.5 * v_curr - 0.5 * v_prev)
        v_prev <- v_curr

v_prev carries between steps, so this solver overrides integrate() (step() alone is
insufficient to hold multistep state cleanly).
"""
from .base import ODESolver


class AB2Solver(ODESolver):
    name = 'ab2'
    nfe_per_step = 1  # amortized after bootstrap

    def step(self, v_fn, x, t, dt):
        # AB2 is multistep; single-step call falls back to Euler (used by default
        # integrate()s of other solvers only, not ours).
        return x - dt * v_fn(x, t)

    def integrate(self, v_fn, x0, N, t_start=1.0, t_end=0.0, step_hook=None,
                  **solver_kwargs):
        dt = (t_start - t_end) / N
        x = x0

        # --- bootstrap: one Euler step ---
        t = t_start
        v_prev = v_fn(x, t)
        x = x - dt * v_prev
        if step_hook is not None:
            x = step_hook(x, 0)

        # --- multistep ---
        for i in range(1, N):
            t = t_start - i * dt
            v_curr = v_fn(x, t)
            x = x - dt * (1.5 * v_curr - 0.5 * v_prev)
            v_prev = v_curr
            if step_hook is not None:
                x = step_hook(x, i)

        return x
