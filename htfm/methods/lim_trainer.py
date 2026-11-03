"""LIMTrainer — continuous-time alpha-stable SDE (LIM), training + sampling.

LIM (Levy It-os/Imura method) lives in the continuous-time regime, so it
does not share the discrete-time schedule machinery with DLPM. Instead it
holds its own VPSDE + LevyStable + heavy-tailed noise generators directly.

V13 split: this is the LIM-side of what used to be `GenerativeLevyProcess`
(behind a `LIM=True` flag). After V13.3 / V13.4 the flag dispatch is gone.

Public surface (matches DLPMTrainer / FlowMatchingLevyProcess):
  - training_losses(models, x_start, model_kwargs, **kwargs) -> {'loss': Tensor[N]}
  - sample(models, shape, reverse_steps, ...) -> Tensor
  - get_timesteps(N, **kwargs) -> Tensor   (uses linear N-step grid)
  - set_path_encoder(path_encoder)
"""

import torch

from .LIM.functions.sde import VPSDE
from .LIM.functions.sampler import LIM_sampler
from .LIM.torchlevy import LevyStable
from .LIM.functions.loss import loss_fn
from .generators import build_skewed_levy, build_sas


class LIMTrainer:
    """Continuous-time alpha-stable SDE (LIM) — train + sample interface."""

    def __init__(
            self,
            alpha,
            device,
            reverse_steps,
            rescale_timesteps=True,
            isotropic=True,
    ):
        # LIM only supports epsilon-prediction + fixed variance + rescaled
        # timesteps. The original GLP enforced this via assert; preserved.
        assert rescale_timesteps is True, \
            "LIM only supports rescale_timesteps=True"

        self.alpha = alpha
        self.device = device
        self.reverse_steps = reverse_steps
        self.rescale_timesteps = rescale_timesteps
        self.isotropic = isotropic

        self.sde = VPSDE(alpha, 'cosine')
        self.levy = LevyStable()

        # LIM doesn't need a discrete schedule; it just needs generators
        # for noise sampling at training and at sampler init.
        self.gen_a = build_skewed_levy(alpha=alpha, device=device, isotropic=isotropic)
        self.gen_eps = build_sas(alpha=alpha, device=device, isotropic=isotropic)

        self.path_encoder = None

    def set_path_encoder(self, path_encoder):
        # LIM doesn't currently consume path features, but the BEM glue
        # may still call this; accept and stash to mirror DLPMTrainer.
        self.path_encoder = path_encoder

    def get_timesteps(self, N, **kwargs):
        """Linear N-step grid in [0, N). Used by GenerationManager animation."""
        return torch.tensor(range(0, N), dtype=torch.float32)

    #####################################################################################
    ''' SAMPLING '''
    #####################################################################################

    def lim_sample(self,
                   model,
                   shape,
                   ddim=False,
                   get_sample_history=False,
                   clip_denoised=False):
        """Sample via the vendored LIM_sampler (SDE or ODE)."""
        x = self.gen_eps.generate(size=shape)

        # Conditional class branch is disabled in current code (was always
        # `if False:` in GLP); kept simple here.
        y = None

        samples = LIM_sampler(
            ddim=ddim,
            x=x,
            y=y,
            model=model,
            sde=self.sde,
            levy=self.levy,
            isotropic=self.isotropic,
            steps=self.reverse_steps,
            gen_a=self.gen_a,
            gen_eps=self.gen_eps,
            device=self.device,
            get_sample_history=get_sample_history,
        )
        return samples

    def sample(self,
               models,
               shape,
               reverse_steps,
               time_spacing=None,
               initial_data=None,
               clip_denoised=False,
               deterministic=False,
               dlim_eta=1.0,
               print_progression=False,
               get_sample_history=False,
               clamp_a=None,
               clamp_eps=None):
        """BEM-facing sampler. `deterministic=True` -> ODE; otherwise SDE."""
        self.gen_a.setParams(clamp_a=clamp_a)
        self.gen_eps.setParams(clamp_eps=clamp_eps)

        model = models['default']
        assert time_spacing is None, "Specific time spacing is not yet supported for LIM sampling"

        # GLP supported reverse_steps overrides via DLPM.rescale_diffusion;
        # for LIM, the sampler takes `steps=` directly so we just swap the
        # attribute around the call.
        # NOTE: the restore-after block intentionally mirrors GLP's
        # `if self.reverse_steps != reverse_steps:` post-call check, which is
        # dead code after the assignment above (always False). Preserved for
        # byte-identity with GLP — fixing the latent leak (sample() with a
        # different reverse_steps mutates self.reverse_steps for the rest of
        # the instance lifetime) is out of scope for V13.
        default_reverse_steps = self.reverse_steps
        if self.reverse_steps != reverse_steps:
            self.reverse_steps = reverse_steps

        x = self.lim_sample(
            model,
            shape=shape,
            ddim=deterministic,
            get_sample_history=get_sample_history,
            clip_denoised=clip_denoised,
        )

        if self.reverse_steps != reverse_steps:
            self.reverse_steps = default_reverse_steps

        return x

    #####################################################################################
    ''' TRAINING '''
    #####################################################################################

    def training_losses(self,
                        models,
                        x_start,
                        model_kwargs=None,
                        **kwargs):
        """Compute LIM training losses for a single timestep.

        Returns {'loss': Tensor[N]}.
        """
        model = models['default']
        x_start = x_start.to(self.device)
        if model_kwargs is None:
            model_kwargs = {}

        loss = self.training_losses_lim(model, x_start, **model_kwargs, **kwargs)
        return {'loss': loss}

    def training_losses_lim(self,
                            model,
                            x_start,
                            y=None,
                            clamp_a=None,
                            clamp_eps=None,
                            **_unused):
        # _unused: silently absorb dlpm/fm-style kwargs (loss_monte_carlo,
        # monte_carlo_*, lploss, ema_rates, grad_clip, ...) that the YAML
        # training.lim block carries from the shared template. LIM is a
        # single-timestep loss so the MC machinery doesn't apply.
        n = x_start.size(0)
        self.gen_a.setParams(clamp_a=clamp_a)
        self.gen_eps.setParams(clamp_eps=clamp_eps)

        # Noise: Gaussian if alpha=2, otherwise heavy-tailed via gen_eps.
        if self.sde.alpha == 2.0:
            e = torch.randn_like(x_start).to(self.device)
        else:
            e = self.gen_eps.generate(size=x_start.shape)

        start_eps = 1e-5
        t = torch.rand(n).to(self.device) * (self.sde.T - start_eps) + start_eps
        losses = loss_fn(model, self.sde, x_start, t, e, config=None, y=None)
        assert not torch.isnan(losses).any(), 'Nan in losses'
        return losses
