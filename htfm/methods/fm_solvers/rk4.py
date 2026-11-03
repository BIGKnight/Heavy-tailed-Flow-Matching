"""Classical 4th order Runge-Kutta. NFE=4 per step.

Adapted for backward integration (t -> t-dt). The sign convention follows
x_{t-dt} = x_t - dt * v; we express the standard RK4 of the ODE dx/dtau = -v(x, t=1-tau)
and unfold back to t-time.

Equivalent explicit form in t-time (k_i are velocities evaluated at intermediate x,t):
    k1 = v(x, t)
    k2 = v(x - 0.5*dt * k1, t - 0.5*dt)
    k3 = v(x - 0.5*dt * k2, t - 0.5*dt)
    k4 = v(x - dt * k3,     t - dt)
    x_new = x - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
"""
from .base import ODESolver


class RK4Solver(ODESolver):
    name = 'rk4'
    nfe_per_step = 4

    def step(self, v_fn, x, t, dt):
        k1 = v_fn(x, t)
        k2 = v_fn(x - 0.5 * dt * k1, t - 0.5 * dt)
        k3 = v_fn(x - 0.5 * dt * k2, t - 0.5 * dt)
        k4 = v_fn(x - dt * k3,       t - dt)
        return x - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
