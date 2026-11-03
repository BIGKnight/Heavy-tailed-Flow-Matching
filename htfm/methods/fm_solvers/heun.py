"""2nd order Heun (explicit trapezoidal / improved Euler). NFE=2 per step.

Update:
    v1 = v(x, t)
    x_pred = x - dt * v1                # Euler predictor
    v2 = v(x_pred, t - dt)              # evaluate at end of interval
    x_new = x - 0.5 * dt * (v1 + v2)    # trapezoidal correction
"""
from .base import ODESolver


class HeunSolver(ODESolver):
    name = 'heun'
    nfe_per_step = 2

    def step(self, v_fn, x, t, dt):
        v1 = v_fn(x, t)
        x_pred = x - dt * v1
        v2 = v_fn(x_pred, t - dt)
        return x - 0.5 * dt * (v1 + v2)
