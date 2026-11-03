"""Scale-preserving (cosine) noise schedule.

Mirrors the ``scale='scale_preserving'`` branch of the legacy
DLPM.gen_noise_schedule method (Nichol & Dhariwal cosine schedule, with the
alpha-stable generalization gamma = alpha**(1/alpha)).
"""

import torch

from .base import NoiseSchedule


class ScalePreservingSchedule(NoiseSchedule):
    def _compute(self, timesteps, diffusion_steps, alpha):
        s = 0.008
        schedule = torch.cos(
            (timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2) ** 2
        baralphas = schedule / schedule[0]
        betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
        alphas_t = 1 - betas

        # alpha-stable generalization of the Gaussian schedule.
        gammas = alphas_t ** (1 / alpha)
        bargammas = torch.cumprod(gammas, dim=0)
        sigmas = (1 - gammas ** alpha) ** (1 / alpha)
        barsigmas = (1 - bargammas ** alpha) ** (1 / alpha)
        return gammas, bargammas, sigmas, barsigmas
