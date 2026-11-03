"""Scale-exploding (EDM-style) noise schedule.

Mirrors the ``scale='scale_exploding'`` branch of the legacy
DLPM.gen_noise_schedule method (Karras et al. EDM-style sigma_min -> sigma_max
sweep with rho=7).
"""

import torch

from .base import NoiseSchedule


class ScaleExplodingSchedule(NoiseSchedule):
    def _compute(self, timesteps, diffusion_steps, alpha):
        sigma_min = 0.002
        sigma_max = 80
        rho = 7

        gammas = torch.ones_like(timesteps)
        bargammas = torch.ones_like(timesteps)
        barsigmas = (
            sigma_min ** (1 / rho)
            + (timesteps / (diffusion_steps - 1))
              * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))
        ) ** rho

        # Per-step sigmas via cumulative-sum constraint:
        # barsigmas[i]^alpha = sum_{j<=i} sigmas[j]^alpha
        barsigmas_alpha = barsigmas ** alpha
        sigmas_alpha = torch.ones_like(barsigmas) * barsigmas_alpha[0]
        for i in range(1, len(barsigmas)):
            sigmas_alpha[i] = barsigmas_alpha[i] - torch.sum(sigmas_alpha[:i])
        sigmas = sigmas_alpha ** (1 / alpha)
        return gammas, bargammas, sigmas, barsigmas
