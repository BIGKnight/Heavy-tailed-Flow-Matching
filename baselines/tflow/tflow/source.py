"""Multivariate Student-t source distribution for t-Flow.

The standard $d$-dimensional zero-mean isotropic Student-$t_\\nu(0, I_d)$
admits the Gaussian scale-mixture representation
    $X = \\sqrt{V}\\, Z, \\qquad V \\sim \\mathrm{InvGamma}(\\nu/2, \\nu/2),
    \\quad Z \\sim \\mathcal{N}(0, I_d), \\quad V \\perp Z$.
Marginalizing over $V$ recovers a heavy-tailed density with polynomial tail
decay rate $r^{-\\nu-d}$ (here $d=$ number of pixels), and the Gaussian
limit is recovered as $\\nu \\to \\infty$.

Pandey et al. 2024 (App.\\ B) train flow matching with this source by drawing
$X_0 \\sim \\mathrm{Student\\text{-}}t_\\nu(0, I_d)$ at each step and pairing it
with the data sample $X_1$ via the straight-line interpolant
    $X_t = (1-t) X_0 + t X_1$.
The velocity target is the constant $X_1 - X_0$.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributions as D


def sample_student_t(shape, nu: float, device=None, dtype=torch.float32):
    """Sample from isotropic Student-t with degrees-of-freedom ``nu``.

    Uses the Gaussian scale-mixture representation: V ~ InverseGamma(nu/2, nu/2),
    Z ~ N(0, I), X = sqrt(V) * Z. The first dimension of ``shape`` is treated
    as the batch dimension and we draw one V per sample.

    Args:
        shape: tensor shape, e.g. (B, C, H, W).
        nu:    degrees-of-freedom; pass ``float('inf')`` for Gaussian limit.
        device, dtype: standard torch options.

    Returns:
        Tensor of shape ``shape``.
    """
    if not torch.is_tensor(nu):
        nu_val = float(nu)
    else:
        nu_val = float(nu.item())

    z = torch.randn(shape, device=device, dtype=dtype)
    if math.isinf(nu_val):
        return z  # Gaussian limit

    # InverseGamma(a, b) sample: V = b / Gamma(a, 1).
    a = nu_val / 2.0
    b = nu_val / 2.0
    batch = shape[0]
    gamma = D.Gamma(concentration=a, rate=1.0).sample((batch,)).to(device=device, dtype=dtype)
    V = b / gamma  # shape (B,)

    # Broadcast V to z's shape.
    V = V.view(batch, *([1] * (z.dim() - 1)))
    return torch.sqrt(V) * z


@dataclass
class StudentTSource:
    """Wrapper around ``sample_student_t`` for a fixed ``nu``."""

    nu: float

    def sample(self, shape, device=None, dtype=torch.float32):
        return sample_student_t(shape, self.nu, device=device, dtype=dtype)
