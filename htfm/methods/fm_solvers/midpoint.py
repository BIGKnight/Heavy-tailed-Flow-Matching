"""2nd order midpoint (modified Euler). NFE=2 per step.

Update (going t -> t-dt):
    v1 = v(x, t)
    x_mid = x - 0.5*dt * v1
    v2 = v(x_mid, t - 0.5*dt)
    x_new = x - dt * v2
"""
from .base import ODESolver


class MidpointSolver(ODESolver):
    name = 'midpoint'
    nfe_per_step = 2

    def step(self, v_fn, x, t, dt):
        v1 = v_fn(x, t)
        x_mid = x - 0.5 * dt * v1
        v2 = v_fn(x_mid, t - 0.5 * dt)
        return x - dt * v2
