"""1st order explicit Euler. Baseline; matches the original inline loop bit-for-bit."""
from .base import ODESolver


class EulerSolver(ODESolver):
    name = 'euler'
    nfe_per_step = 1

    def step(self, v_fn, x, t, dt):
        v = v_fn(x, t)
        return x - dt * v
