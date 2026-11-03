import torch
import bem.datasets.Data as Data
from bem.datasets.Distributions import *




### TO REIMPLEMENT
### For the moment, only implementing 'EPSILON' prediction, 'FIXED' variance, 'EPS_LOSS'
class ModelMeanType():
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = 'PREVIOUS_X'  # the model predicts x_{t-1}
    START_X = 'START_X'  # the model predicts x_0
    EPSILON = 'EPSILON'  # the model predicts epsilon
    Z = 'Z' # the model predicts z_t
    SQRT_GAMMA_EPSILON = 'SQRT_GAMMA_EPSILON' # the model predicts (1 - sqrt(1 - gamma))eps

class ModelVarType():
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED = 'FIXED' # fixed variance
    # GAMMA = 'GAMMA' # can learn gamma factor, with the true var being tilde_sigma = gamma * sigma
    # SIGMA_T_1_OVER_A = enum.auto() # learn directly Sigma_t_1 / a_t
    # TRUE_VAR = enum.auto()

class LossType():
    # can replace different means by normal loss and lambda loss, with some lambda to
    # pass as argument in training. should be better.
    LP_LOSS = 'LP_LOSS' # L_p loss between model output and target value
    MEAN_LOSS = 'MEAN_LOSS' # LP loss between model mean and true mean
    EPS_LOSS = 'EPS_LOSS' # lp loss between noise and model noise
    LAMBDA_LOSS = 'LAMBDA_LOSS' #  L_p loss between model output and added noise, times lambda factor (=eps * lambda factor)
    VAR_KL = 'VAR_KL' # when mixing mean and var.
    VAR_LP_SUM = 'VAR_LP_SUM' # when mixing mean and var.


# match_last_dims moved to htfm/methods/_utils.py (V12.1).
# Re-exported here for any external code still using the old import path.
from ._utils import match_last_dims


# Compute the right noise schedules, with arbitrary gamma and sigma, so no more bg (bargammas)

from .dlpm_dynamics import DLPMDynamicsMixin
from .dlpm_posterior import DLPMPosteriorMixin


class DLPM(DLPMDynamicsMixin, DLPMPosteriorMixin):
    """DLPM forward/reverse process.

    Methods grouped by concern across mixin modules:
      * DLPMDynamicsMixin (dlpm_dynamics.py): predict_xstart, predict_eps,
        predict_eps_from_m_tilde, sample_x_t_from_xstart, sample_A,
        compute_Sigmas, sample_x_t_from_xstart_given_Sigma.
      * DLPMPosteriorMixin (dlpm_posterior.py): compute_Gamma_t,
        compute_Sigma_tilde_t_1, compute_m_tilde_t_1,
        anterior_mean_variance_dlpm, anterior_mean_variance_dlim.

    Method signatures unchanged: callers using `self.dlpm.X(...)` continue
    to work via Python's normal MRO mixin dispatch.
    """

    def __init__(
        self,
        alpha,
        device,
        diffusion_steps,
        time_spacing = 'linear',
        isotropic = True, # isotropic levy noise
        clamp_a = None,
        clamp_eps = None,
        scale = 'scale_preserving'
    ):
        self.alpha = alpha
        self.device = device
        self.time_spacing = time_spacing
        self.isotropic = isotropic
        # use_single_a_chain attr removed in v6.1 (was set, never read).
        self.scale = scale

        # 1d noising schedules — math lives in htfm/methods/noise_schedule/.
        # Adding a new schedule = one file + one entry in SCHEDULE_REGISTRY.
        from .noise_schedule import build_schedule
        self.schedule = build_schedule(
            scale, alpha=alpha, device=device,
            diffusion_steps=diffusion_steps, time_spacing=time_spacing,
        )
        # Backward-compat: expose schedule arrays as DLPM attrs (many readers
        # access self.dlpm.sigmas etc.). Refs, not copies — same tensor objects.
        self.gammas = self.schedule.gammas
        self.bargammas = self.schedule.bargammas
        self.sigmas = self.schedule.sigmas
        self.barsigmas = self.schedule.barsigmas

        # noising schedules, adapted to batch size, to be potentially updated for each incoming batch
        self.constants = None

        # self.bargammas = torch.cumprod(self.gammas, dim = 0)
        self.gen_a = Data.Generator('skewed_levy', 
                                    alpha = self.alpha, 
                                    device=self.device,
                                    isotropic=isotropic,
                                    clamp_a = clamp_a)
        
        self.gen_eps = Data.Generator('sas',
                                      alpha = self.alpha, 
                                      device=self.device,
                                      isotropic=isotropic,
                                      clamp_eps = clamp_eps)
        
        self.A = None
        self.Sigmas = None


#######################################################################################
    ''' Noise schedules '''
#######################################################################################
    
    def get_timesteps(self, steps):
        if self.time_spacing == 'linear':
            timesteps = torch.tensor(range(0, steps), dtype=torch.float32)
        elif self.time_spacing == 'quadratic':
            timesteps = steps* (torch.tensor(range(0, steps), dtype=torch.float32) / steps)**2
        else:
            raise NotImplementedError(self.time_spacing)
        return timesteps


    # gen_noise_schedule(...) was removed in v6.1 — the backward-compat shim
    # had no callers. Use noise_schedule.build_schedule() directly if you
    # need to construct a schedule outside DLPM.

    def get_schedule(self, shape):
        g = match_last_dims(self.gammas, shape)
        bg = match_last_dims(self.bargammas, shape)
        s = match_last_dims(self.sigmas, shape)
        bs = match_last_dims(self.barsigmas, shape)
        return g, bg, s, bs
    
    def get_t_to_batch_size(self, x_t, t):
        if isinstance(t, int):
            return torch.full([x_t.shape[0]], t).to(self.device)
        return t
    
    # in order not to compute the gammas, bargammas constants each time.
    def update_constants(self, shape):
        if (self.constants is None) or (self.constants[0].shape != shape):
            self.constants = self.get_schedule(shape)
        return self.constants
    
    def rescale_diffusion(self, diffusion_steps, time_spacing=None):
        assert isinstance(diffusion_steps, int), "Diffusion steps must be an integer"
        if time_spacing is not None:
            self.time_spacing = time_spacing
        # Rebuild the schedule object, then refresh the backward-compat refs.
        # NOTE: legacy code defaulted to scale='scale_preserving' here regardless
        # of self.scale; preserved verbatim to avoid silent behavior change.
        from .noise_schedule import build_schedule
        self.schedule = build_schedule(
            'scale_preserving',
            alpha=self.alpha, device=self.device,
            diffusion_steps=diffusion_steps, time_spacing=self.time_spacing,
        )
        self.gammas = self.schedule.gammas
        self.bargammas = self.schedule.bargammas
        self.sigmas = self.schedule.sigmas
        self.barsigmas = self.schedule.barsigmas
        self.constants = None

    # Forward dynamics (predict_xstart / predict_eps / predict_eps_from_m_tilde /
    # sample_x_t_from_xstart / sample_A / compute_Sigmas /
    # sample_x_t_from_xstart_given_Sigma) moved to DLPMDynamicsMixin in
    # dlpm_dynamics.py (v6.3). Inherited via class declaration above.

    # Posterior calcs (compute_Gamma_t / compute_Sigma_tilde_t_1 /
    # compute_m_tilde_t_1 / anterior_mean_variance_dlpm /
    # anterior_mean_variance_dlim) moved to DLPMPosteriorMixin in
    # dlpm_posterior.py (v6.2). Inherited via class declaration above.

#######################################################################################
    ''' Training: we use Proposition (9), and only sample one heavy-tailed r.v a_t '''
#######################################################################################

    # NOTE: the Proposition (8) two-rv variant (get_two_rv_*, compute_two_rv_*)
    # was removed in v6.1 — it had no callers (the active training path goes
    # through Proposition (9) one-rv only, used by GenerativeLevyProcess.
    # training_losses_dlpm via get_one_rv_loss_elements below).
    
    ''' 
        Here we must satisfy design choices D1, D2, D3, so 
        we do not learn variance (set to Sigma_tilde)
        we do not compute lambda (set to 1)
        we only compute eps_t
    '''

    # returns a_t as in Propostion (9)
    def get_one_rv_faster_sampling(self, shape):
        return self.gen_a.generate(size = shape)

    # compute Sigma_t from a_t, as in Proposition (9)
    def compute_one_rv_Sigma_prime_t(self, t, a_t):
        g, bg, s, bs = self.update_constants(a_t.shape)
        t = self.get_t_to_batch_size(a_t, t)
        # only returns Sigmas[t]
        Sigma_prime_t = a_t * bs[t]**2 
        return Sigma_prime_t

    def get_one_rv_loss_elements(self, t, x_0, a_t = None, z_t = None):
        if a_t is None:
            a_t = self.get_one_rv_faster_sampling(x_0.shape)
        Sigma_prime_t = self.compute_one_rv_Sigma_prime_t(t, a_t)
        x_t = self.sample_x_t_from_xstart_given_Sigma(x_0, t, Sigma_prime_t, z_t=z_t)
        eps_t = self.predict_eps(x_t, t, x_0)
        return x_t, eps_t


    # def _eps_from_previous_x(self, t, xt, previous_x, Gamma, constants = None):
    #     g, bg = self.get_constants(constants, xt.size())
    #     return (xt - g[t]**(1/self.alpha)*previous_x) / (Gamma * (1 - bg[t])**(1/self.alpha))



    # def get_constants_to_incoming_batch_dims(self, Xbatch, t):
    #     g, bg, s, bs = self.get_schedule(Xbatch)
    #     t = self.get_t_to_batch_size(Xbatch, t)
    #     a_t_1, a_t_prime, a_t = self.get_noises_to_incoming_batch_dims(Xbatch, t)
    #     return g, bg, t, a_t_1, a_t_prime, a_t

