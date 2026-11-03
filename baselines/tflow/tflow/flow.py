"""Flow-matching loss and ODE sampler for t-Flow.

Two network parameterizations are supported, both consistent with the
straight-line interpolant
    X_t = (1 - t) X_0 + t X_1,    t = 0 -> noise, t = 1 -> data:

* ``param='noise'`` (default; matches Pandey et al. 2024 App. B Eq. 166):
  the network predicts X_0 itself. Loss is
      L = E || eps_theta(X_t, t) - X_0 ||^2
  and the sampling vector field is
      v(X_t, t) = (X_t - eps_theta) / t.

* ``param='velocity'`` (standard CFM): the network directly predicts
  v = X_1 - X_0, with loss
      L = E || v_theta(X_t, t) - (X_1 - X_0) ||^2.

The two are mathematically equivalent at the optimum but train differently;
``noise`` reproduces the Pandey paper.

Two ODE solvers are provided:

* ``solver='euler'`` (default): fixed-step Euler from ``t_start`` to 1.0
  with ``nfe`` steps. Useful for matched-NFE comparison against HTFM.
* ``solver='heun'``: 2nd-order Heun (matches Pandey App. B / Algorithm 4).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .source import sample_student_t


def _to_view(t: torch.Tensor, ndim: int) -> torch.Tensor:
    return t.view(-1, *([1] * (ndim - 1)))


def flow_matching_loss(
    model: Callable,
    x1: torch.Tensor,
    nu: float,
    *,
    param: str = 'noise',
    t_eps: float = 1e-3,
):
    """Single batch t-Flow training loss.

    Args:
        model:  ``model(x, t) -> tensor`` predicting either the source
                draw X_0 (``param='noise'``) or the velocity X_1 - X_0
                (``param='velocity'``).
        x1:     data tensor (B, ...).
        nu:     Student-t degrees-of-freedom; ``float('inf')`` for Gaussian.
        param:  ``'noise'`` (Pandey default) or ``'velocity'``.
        t_eps:  lower clip on the sampled t to avoid the degenerate t=0
                regime when ``param='noise'`` (where the implied velocity
                (X_t - X_0)/t is 0/0).

    Returns:
        Scalar loss.
    """
    B = x1.shape[0]
    device = x1.device
    dtype = x1.dtype

    x0 = sample_student_t(x1.shape, nu, device=device, dtype=dtype)
    t = torch.rand(B, device=device, dtype=dtype).clamp_min(t_eps)
    t_view = _to_view(t, x1.dim())
    xt = (1.0 - t_view) * x0 + t_view * x1

    pred = model(xt, t)
    if param == 'noise':
        target = x0
    elif param == 'velocity':
        target = x1 - x0
    else:
        raise ValueError(f"unknown param='{param}', expected 'noise' or 'velocity'")
    return F.mse_loss(pred, target)


def _velocity_from_pred(pred: torch.Tensor, x_t: torch.Tensor,
                       t_view: torch.Tensor, param: str) -> torch.Tensor:
    """Convert the raw network output into a flow velocity, depending on the
    chosen parameterization.

    For ``param='noise'``, ``pred`` is an estimate of X_0 and the velocity
    is computed via the closed-form ``(X_t - X_0)/t``. For
    ``param='velocity'``, ``pred`` is the velocity directly.
    """
    if param == 'noise':
        return (x_t - pred) / t_view
    return pred


@torch.no_grad()
def euler_sample(
    model: Callable,
    shape,
    nu: float,
    *,
    nfe: int = 100,
    param: str = 'noise',
    device=None,
    dtype: torch.dtype = torch.float32,
    t_start: float = 1e-3,
    t_end: float = 1.0,
    return_trajectory: bool = False,
):
    """Generate samples by Euler integration of the learned velocity.

    Integrates ``X_t' = v(X_t, t)`` from ``t_start`` to ``t_end`` with
    ``nfe`` equal Euler steps. ``v`` is recovered from the network output
    using the chosen ``param`` (see ``_velocity_from_pred``).
    """
    x = sample_student_t(shape, nu, device=device, dtype=dtype)
    ts = torch.linspace(t_start, t_end, nfe + 1, device=device, dtype=dtype)
    if return_trajectory:
        traj = [x.clone()]
    for k in range(nfe):
        t_now = ts[k].expand(shape[0])
        t_view = _to_view(t_now, x.dim())
        dt = ts[k + 1] - ts[k]
        v = _velocity_from_pred(model(x, t_now), x, t_view, param)
        x = x + dt * v
        if return_trajectory:
            traj.append(x.clone())
    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x


def _karras_t_grid(*, nfe: int, sigma_min: float, sigma_max: float,
                   rho: float, device=None, dtype=torch.float32):
    """Build the Karras-EDM time grid (mapped to t = 1 - sigma).

    Used by the Heun sampler to match the Pandey App. B / Algorithm 4
    discretization.
    """
    i = torch.arange(nfe + 1, device=device, dtype=dtype)
    sigma = (sigma_max ** (1 / rho) +
             i / nfe * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # Pandey's convention: t = 1 - sigma (sigma=1 is noise, sigma=0 is data).
    t = 1.0 - sigma
    return t


@torch.no_grad()
def heun_sample(
    model: Callable,
    shape,
    nu: float,
    *,
    nfe: int = 25,
    param: str = 'noise',
    device=None,
    dtype: torch.dtype = torch.float32,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    rho: float = 7.0,
    return_trajectory: bool = False,
):
    """2nd-order Heun sampler with Karras-EDM time grid (Pandey defaults).

    Each Heun step evaluates the velocity twice; total network calls = 2*nfe.
    The grid endpoints follow Pandey App. B (sigma in [0.01, 1.0], rho=7).
    The ``t = 1 - sigma`` mapping is used so we still march from t=0
    (noise) towards t=1 (data).
    """
    x = sample_student_t(shape, nu, device=device, dtype=dtype)
    ts = _karras_t_grid(nfe=nfe, sigma_min=sigma_min, sigma_max=sigma_max,
                        rho=rho, device=device, dtype=dtype)
    if return_trajectory:
        traj = [x.clone()]
    for k in range(nfe):
        t0 = ts[k].expand(shape[0])
        t1 = ts[k + 1].expand(shape[0])
        dt = (ts[k + 1] - ts[k])
        if dt.item() <= 0:  # numerically degenerate step (final t=1 boundary)
            continue
        # Predictor (Euler).
        t0v = _to_view(t0, x.dim())
        v0 = _velocity_from_pred(model(x, t0), x, t0v, param)
        x_pred = x + dt * v0
        # Corrector. Skip if we hit the t=1 boundary exactly (avoid /0).
        if ts[k + 1].item() < 1.0:
            t1v = _to_view(t1, x.dim())
            v1 = _velocity_from_pred(model(x_pred, t1), x_pred, t1v, param)
            x = x + 0.5 * dt * (v0 + v1)
        else:
            x = x_pred
        if return_trajectory:
            traj.append(x.clone())
    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x
