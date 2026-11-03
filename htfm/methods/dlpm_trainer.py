"""DLPMTrainer — discrete-time alpha-stable DDPM, training + sampling layer.

Inherits the math from `dlpm.DLPM` (forward dynamics via DLPMDynamicsMixin
plus posterior calcs via DLPMPosteriorMixin) and adds the BEM-facing
`training_losses` / `sample` entry points.

V13 split: this file holds the DLPM-side of what used to be
`GenerativeLevyProcess`. The LIM branch lives in `lim_trainer.py` — neither
class now carries an `LIM=…` flag dispatch.

Public surface (called by bem.TrainingManager / GenerationManager):
  - training_losses(models, x_start, model_kwargs, **kwargs) -> {'loss': Tensor[N]}
  - sample(models, shape, reverse_steps, ...) -> Tensor
  - get_timesteps(N, **kwargs) -> Tensor
  - set_path_encoder(path_encoder)

Internal layers (re-used by EvaluationManager / external scripts):
  - p_sample_loop / p_sample_loop_progressive / p_sample / p_mean_variance
  - ddim_sample_loop / ddim_sample_loop_progressive / ddim_sample
  - q_sample (forward q(x_t | x_0))
"""

from tqdm import tqdm
import torch
import torch as th

from .dlpm import DLPM, ModelMeanType, ModelVarType, LossType
from ._utils import match_last_dims, compute_loss_terms


class DLPMTrainer(DLPM):
    """Train + sample interface for the discrete-time DLPM (alpha-stable DDPM)."""

    def __init__(
            self,
            alpha,
            device,
            reverse_steps,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED,
            time_spacing='linear',
            rescale_timesteps=False,
            isotropic=True,
            scale='scale_preserving',
            input_scaling=False,
    ):
        # DLPM.__init__ owns: schedule (gammas, bargammas, sigmas, barsigmas),
        # gen_a, gen_eps, A, Sigmas, constants. We just layer training/sampling
        # on top.
        super().__init__(
            alpha=alpha,
            device=device,
            diffusion_steps=reverse_steps,
            time_spacing=time_spacing,
            isotropic=isotropic,
            scale=scale,
        )
        self.reverse_steps = reverse_steps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps
        self.input_scaling = input_scaling

        assert (self.model_mean_type == ModelMeanType.EPSILON) \
            and (self.model_var_type == ModelVarType.FIXED), \
            'Only epsilon prediction and fixed variance are supported for the moment'

        self.path_encoder = None

    def set_path_encoder(self, path_encoder):
        self.path_encoder = path_encoder

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1.0 / self.reverse_steps)
        return t

    def get_timesteps(self, N, **kwargs):
        # DLPM.get_timesteps doesn't accept kwargs; absorb extras silently
        # (BEM's GenerationManager forwards self.kwargs).
        return DLPM.get_timesteps(self, N)

    #####################################################################################
    ''' SAMPLING'''
    #####################################################################################

    def q_sample(self, x_start, t, eps=None):
        """Sample from q(x_t | x_0) — forward diffusion."""
        return self.sample_x_t_from_xstart(x_start, t, eps)

    def p_mean_variance(self,
                        model,
                        x,
                        t,
                        clip_denoised=False,
                        denoised_fn=None,
                        model_kwargs=None,
                        path_features=None):
        """Apply the model to get p(x_{t-1} | x_t) and a prediction of x_0.

        Returns dict with 'eps', 'mean', 'variance'.
        """
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        input_scaling = 1.0
        if self.input_scaling and (self.scale == 'scale_exploding'):
            input_scaling = match_last_dims(1 / (1 + self.barsigmas[t]), x.shape)
        pf_kwargs = {'path_features': path_features} if path_features is not None else {}
        model_output = model(x * input_scaling, self._scale_timesteps(t), **pf_kwargs, **model_kwargs)

        if (self.model_mean_type == ModelMeanType.EPSILON) and (not clip_denoised):
            model_eps = model_output
        else:
            if self.model_mean_type == ModelMeanType.EPSILON:
                model_xstart = process_xstart(
                    self.predict_xstart(x_t=x, t=t, eps=model_output)
                )
            elif self.model_mean_type == ModelMeanType.START_X:
                model_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.Z:
                eps = torch.sqrt(self.A[t]) * model_output
                model_xstart = process_xstart(
                    self.predict_xstart(x_t=x, t=t, eps=eps)
                )
            elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                eps = self.predict_eps_from_m_tilde(x, t, model_output)
                model_xstart = process_xstart(
                    self.predict_xstart(x_t=x, t=t, eps=eps)
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            model_eps = self.predict_eps(x_t=x, t=t, xstart=model_xstart)

        model_mean, model_variance = self.anterior_mean_variance_dlpm(x, t[0], model_eps)

        assert model_mean.shape == x.shape, 'model mean:{}, x:{}'.format(model_mean.shape, x.shape)

        return {
            'eps': model_eps,
            'mean': model_mean,
            'variance': model_variance,
        }

    def p_sample(
            self, model, x, t, clip_denoised=False, denoised_fn=None,
            model_kwargs=None, path_features=None,
    ):
        """Sample x_{t-1} from the model at the given timestep."""
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            path_features=path_features,
        )
        noise = th.randn_like(x)
        nonzero_mask = ((t != 1).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.sqrt(out["variance"]) * noise
        return {"sample": sample}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            progress=False,
            get_sample_history=False,
    ):
        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total=self.reverse_steps)
        model.eval()
        x_hist = []

        with th.inference_mode():
            final = None
            for sample in self.p_sample_loop_progressive(
                    model, shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)

            if progress:
                pbar.close()
                tqdm._instances.clear()

            if get_sample_history:
                return final['sample'], torch.stack(x_hist)
            return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
    ):
        assert self.device is not None
        assert isinstance(shape, (tuple, list))

        self.sample_A(shape, self.reverse_steps)
        self.compute_Sigmas()

        path_features = None
        if self.path_encoder is not None and self.path_encoder.enabled:
            t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
                self.sigmas, self.gammas, self.A)
            path_features = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)

        if noise is not None:
            img = noise
        else:
            img = self.barsigmas[-1] * self.gen_eps.generate(size=shape)
        yield {'sample': img}

        indices = list(range(self.reverse_steps - 1, 0, -1))

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            out = self.p_sample(
                model, img, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                path_features=path_features,
            )
            yield out
            img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            path_features=None,
    ):
        """Sample x_{t-1} via DDIM (deterministic when eta=0.0)."""
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            path_features=path_features,
        )
        eps = out['eps']
        model_mean, model_variance = self.anterior_mean_variance_dlim(x, t, eps, eta=eta)
        assert not torch.isnan(model_mean).any()

        if eta == 0.0:
            return {"sample": model_mean}

        sample = model_mean + torch.sqrt(model_variance) * torch.randn_like(x)
        return {"sample": sample}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            progress=False,
            eta=0.0,
            get_sample_history=False,
    ):
        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total=self.reverse_steps)

        model.eval()
        x_hist = []
        with th.inference_mode():
            final = None
            for sample in self.ddim_sample_loop_progressive(
                    model, shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)
            if progress:
                pbar.close()
                tqdm._instances.clear()
            if get_sample_history:
                return final['sample'], torch.stack(x_hist)
            return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        assert self.device is not None
        assert isinstance(shape, (tuple, list))

        self.sample_A(shape, self.reverse_steps)
        self.compute_Sigmas()

        path_features = None
        if self.path_encoder is not None and self.path_encoder.enabled:
            t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_sigmas(
                self.sigmas, self.gammas, self.A)
            path_features = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)

        if noise is not None:
            img = noise
        else:
            img = self.barsigmas[-1] * self.gen_eps.generate(size=shape)
        yield {'sample': img}
        indices = list(range(self.reverse_steps - 1, 0, -1))
        for i in indices:
            t = th.tensor([i] * shape[0], device=self.device)
            out = self.ddim_sample(
                model, img, t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
                path_features=path_features,
            )
            yield out
            img = out["sample"]

    #####################################################################################
    ''' BEM: SAMPLING '''
    #####################################################################################

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
        """BEM-facing sampler. Dispatches to p_sample_loop / ddim_sample_loop."""
        self.gen_a.setParams(clamp_a=clamp_a)
        self.gen_eps.setParams(clamp_eps=clamp_eps)

        model = models['default']

        assert time_spacing is None, "Specific time spacing is not yet supported for diffusion reverse sampling"

        default_reverse_steps = self.reverse_steps
        default_time_spacing = self.time_spacing
        if self.reverse_steps != reverse_steps:
            assert self.rescale_timesteps, "Rescaling only works when rescale_timesteps is True"
            self.rescale_diffusion(reverse_steps, time_spacing=time_spacing)
            self.reverse_steps = reverse_steps

        if deterministic:
            x = self.ddim_sample_loop(
                model,
                shape=initial_data.shape if initial_data is not None else shape,
                noise=initial_data if initial_data is not None else None,
                eta=dlim_eta,
                progress=print_progression,
                get_sample_history=get_sample_history,
                clip_denoised=clip_denoised,
            )
        else:
            x = self.p_sample_loop(
                model,
                shape=shape,
                progress=print_progression,
                get_sample_history=get_sample_history,
                clip_denoised=clip_denoised,
            )

        if self.reverse_steps != reverse_steps:
            self.rescale_diffusion(default_reverse_steps, default_time_spacing)
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
        """Compute training losses for a single timestep.

        Returns {'loss': Tensor[N]}.
        """
        model = models['default']
        x_start = x_start.to(self.device)
        if model_kwargs is None:
            model_kwargs = {}

        loss = self.training_losses_dlpm(model, x_start, **model_kwargs, **kwargs)
        return {'loss': loss}

    def training_losses_dlpm(self,
                             model,
                             x_start,
                             loss_type='EPSILON',
                             lploss=2.0,
                             loss_monte_carlo='mean',
                             monte_carlo_outer=1,
                             monte_carlo_inner=1,
                             monte_carlo_t=1,
                             robust_exponent=0.5,
                             model_kwargs=None,
                             clamp_a=None,
                             clamp_eps=None):
        """Proposition (9) one-rv loss with median-of-means / robust aggregation."""
        assert self.model_mean_type == ModelMeanType.EPSILON, \
            'only epsilon model output is supported for the moment'
        assert loss_type == LossType.EPS_LOSS, 'only epsilon loss is supported for the moment'
        if model_kwargs is None:
            model_kwargs = {}

        self.gen_a.setParams(clamp_a=clamp_a)
        self.gen_eps.setParams(clamp_eps=clamp_eps)

        # median-of-means estimator setup
        total_monte_carlo = monte_carlo_outer * monte_carlo_inner
        x_start_extended = x_start.repeat(total_monte_carlo, *([1] * len(x_start.shape[1:])))
        outer_shape = torch.tensor(x_start.shape)
        outer_shape[0] *= monte_carlo_outer
        A = self.get_one_rv_faster_sampling(list(outer_shape))
        A_extended = A.repeat(monte_carlo_inner, *([1] * len(A.shape[1:])))
        z_t_extended = torch.randn_like(x_start_extended, device=self.device)
        if monte_carlo_t > 1:
            x_start_extended = x_start_extended.repeat(monte_carlo_t, *([1] * len(x_start.shape[1:])))
            A_extended = A_extended.repeat(monte_carlo_t, *([1] * len(A.shape[1:])))
            z_t_extended = z_t_extended.repeat(monte_carlo_t, *([1] * len(z_t_extended.shape[1:])))

        t_extended = torch.randint(1, self.reverse_steps, size=[x_start_extended.shape[0]]).to(self.device)

        x_t, eps_t = self.get_one_rv_loss_elements(t_extended, x_start_extended, A_extended, z_t_extended)

        path_features = None
        if self.path_encoder is not None and self.path_encoder.enabled:
            A_scalar = A.reshape(A.shape[0], -1).mean(dim=-1)
            t_grid, T_raw, T_eff = self.path_encoder.compute_clock_path_from_terminal_A(
                self.barsigmas, A_scalar)
            path_features = self.path_encoder(t_grid, T_raw=T_raw, T_effective=T_eff)
            if monte_carlo_inner > 1:
                path_features = path_features.repeat(monte_carlo_inner, 1)
            if monte_carlo_t > 1:
                path_features = path_features.repeat(monte_carlo_t, 1)

        input_scaling = 1.0
        if self.input_scaling and (self.scale == 'scale_exploding'):
            input_scaling = match_last_dims(1 / (1 + self.barsigmas[t_extended]), x_t.shape)
        pf_kwargs = {'path_features': path_features} if path_features is not None else {}
        model_eps = model(x_t * input_scaling, self._scale_timesteps(t_extended), **pf_kwargs, **model_kwargs)

        losses = compute_loss_terms(model_eps, eps_t, lploss)
        assert not torch.isnan(losses).any(), 'Nan in losses'

        from .loss_aggregation import build_aggregator
        aggregator = build_aggregator(loss_monte_carlo)
        loss = aggregator.aggregate(
            losses, monte_carlo_outer, monte_carlo_inner, monte_carlo_t,
            x_start.shape[0],
            robust_exponent=robust_exponent,
        )
        return loss
