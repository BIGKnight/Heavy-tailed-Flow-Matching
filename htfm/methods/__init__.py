"""Method registry: FM / DLPM / LIM.

Adding a new generative method = one builder + one registry entry. Each
builder takes the full param dict `p` and returns a method instance.

Consumed by htfm.experiment.init_method_by_parameter, which keeps
path-conditioning module setup separate (path_encoder is wired AFTER the
method is built; that wiring is method-agnostic).
"""


def _build_fm(p):
    """FlowMatchingLevyProcess + clock + predictor.

    v15.2: path_sampling / path_pool may live in either fm.clock.* (new,
    canonical) or path_conditioning.* (legacy). We read from the canonical
    location with fallback. The shim in dlpm_experiment._shim_v15_path_keys
    already migrates legacy → canonical pre-call, so when the shim has run
    the second branch is dead.
    """
    from .flow_matching import FlowMatchingLevyProcess
    fm_cfg = p['fm']
    clock_cfg = fm_cfg.get('clock', {})
    pc_cfg = p.get('path_conditioning', {})
    # Canonical location (post-shim) wins; fall back to legacy location.
    path_sampling_cfg = clock_cfg.get('path_sampling',
                                      pc_cfg.get('path_sampling', {}))
    path_pool_cfg = clock_cfg.get('path_pool',
                                  pc_cfg.get('path_pool', {}))
    # Canonical schema for clock-tail-index alpha: fm.clock.alpha (alongside
    # fm.clock.nu for student_t). Legacy fallback: fm.alpha. For non-alpha-
    # stable clocks the schedule still gets built with this value but is
    # never queried (default 2.0 keeps schedule numerically Gaussian-limit).
    alpha = clock_cfg.get('alpha', fm_cfg.get('alpha', 2.0))

    # Strip path_sampling/path_pool/type/alpha from clock_kwargs:
    #   - alpha-stable receives alpha via FlowMatchingLevyProcess's dedicated
    #     `alpha=` arg (not via clock_kwargs)
    #   - student_t / gaussian don't accept alpha and would TypeError
    clock_kwargs = {k: v for k, v in clock_cfg.items()
                    if k not in ('type', 'path_sampling', 'path_pool', 'alpha')}

    # signature_linear_ve trajectory schedule. Only built when flow_type
    # selects it; otherwise None (existing flow_types ignore ve_schedule).
    flow_type = fm_cfg.get('flow_type', 'straight_line')
    ve_schedule = None
    if flow_type == 'signature_linear_ve':
        from .noise_schedule import SignatureLinearVESchedule
        sve_cfg = fm_cfg.get('signature_linear_ve', {}) or {}
        ve_schedule = SignatureLinearVESchedule(
            lambda_alpha=sve_cfg.get('lambda_alpha', 0.25),
            lambda_beta=sve_cfg.get('lambda_beta', 0.25),
        )

    return FlowMatchingLevyProcess(
        alpha=alpha,
        device=p['device'],
        reverse_steps=fm_cfg['reverse_steps'],
        isotropic=fm_cfg['isotropic'],
        scale=fm_cfg['scale'],
        input_scaling=fm_cfg.get('input_scaling', False),
        prediction_type=fm_cfg.get('prediction_type', 'velocity'),
        flow_type=flow_type,
        noise_type=fm_cfg.get('noise_type', 'terminal'),
        path_sampling_cfg=path_sampling_cfg,
        path_pool_cfg=path_pool_cfg,
        clock_type=clock_cfg.get('type', 'alpha_stable'),
        clock_kwargs=clock_kwargs,
        ve_schedule=ve_schedule,
    )


def _build_dlpm(p):
    """Discrete-time DLPM (alpha-stable DDPM). V13.3: DLPMTrainer (was GLP)."""
    from .dlpm_trainer import DLPMTrainer
    cfg = p['dlpm']
    return DLPMTrainer(
        alpha=cfg['alpha'],
        device=p['device'],
        reverse_steps=cfg['reverse_steps'],
        model_mean_type=cfg['mean_predict'],
        model_var_type=cfg['var_predict'],
        rescale_timesteps=cfg['rescale_timesteps'],
        isotropic=cfg['isotropic'],
        scale=cfg['scale'],
        input_scaling=cfg['input_scaling'],
    )


def _build_lim(p):
    """Continuous-time LIM (alpha-stable SDE). V13.3: LIMTrainer (was GLP)."""
    from .lim_trainer import LIMTrainer
    cfg = p['lim']
    return LIMTrainer(
        alpha=cfg['alpha'],
        device=p['device'],
        reverse_steps=cfg['reverse_steps'],
        rescale_timesteps=cfg['rescale_timesteps'],
        isotropic=cfg['isotropic'],
    )


METHOD_BUILDERS = {
    'fm':     _build_fm,
    'dlpm':   _build_dlpm,
    'lim':    _build_lim,
}


def build_method(p):
    """Build the method instance dispatched by p['method']."""
    name = p['method']
    if name not in METHOD_BUILDERS:
        raise ValueError(
            f"Unknown method={name!r}. Known: {list(METHOD_BUILDERS.keys())}")
    return METHOD_BUILDERS[name](p)


def available_methods():
    return list(METHOD_BUILDERS.keys())


__all__ = ['METHOD_BUILDERS', 'build_method', 'available_methods']
