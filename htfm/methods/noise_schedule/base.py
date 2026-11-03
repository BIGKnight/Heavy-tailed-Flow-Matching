"""Abstract NoiseSchedule base class.

A NoiseSchedule provides the four 1-D tensor arrays that determine the
forward / reverse dynamics of a heavy-tailed diffusion / flow-matching
model: gammas / bargammas / sigmas / barsigmas. Subclasses override
_compute() with the schedule-specific math.

Currently shipped subclasses:
  - ScalePreservingSchedule: cosine-style scale-preserving schedule
                             (DDPM-with-cosine, alpha-stable generalization).
  - ScaleExplodingSchedule:  EDM-style sigma_min -> sigma_max sweep.
"""

import torch


class NoiseSchedule:
    """Holds the (gammas, bargammas, sigmas, barsigmas) tuple as device tensors.

    Subclasses must override _compute(timesteps, diffusion_steps, alpha) and
    return a 4-tuple of 1-D float tensors.
    """

    def __init__(self, alpha, device, diffusion_steps, time_spacing='linear'):
        self.alpha = alpha
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.time_spacing = time_spacing

        timesteps = self._get_timesteps(time_spacing, diffusion_steps)
        gammas, bargammas, sigmas, barsigmas = self._compute(
            timesteps, diffusion_steps, alpha)
        self.gammas = gammas.to(device)
        self.bargammas = bargammas.to(device)
        self.sigmas = sigmas.to(device)
        self.barsigmas = barsigmas.to(device)

    @staticmethod
    def _get_timesteps(time_spacing, diffusion_steps):
        """Construct the time grid. Mirrors the DLPM.get_timesteps utility."""
        if time_spacing == 'linear':
            return torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
        if time_spacing == 'quadratic':
            t = torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
            return diffusion_steps * (t / diffusion_steps) ** 2
        raise NotImplementedError(time_spacing)

    def get_timesteps(self, steps):
        """Public method: time grid for arbitrary `steps` count using this
        schedule's `time_spacing`. Used by FM `sample` to override
        reverse_steps at eval time independently of the trained schedule."""
        return NoiseSchedule._get_timesteps(self.time_spacing, steps)

    def _compute(self, timesteps, diffusion_steps, alpha):
        """Return (gammas, bargammas, sigmas, barsigmas) as 1-D float tensors."""
        raise NotImplementedError
