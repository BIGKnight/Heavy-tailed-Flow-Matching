"""Random clock implementations for heavy-tailed flow matching.

A "clock" determines how the heavy-tailed noise x_1 is generated and what
latent conditioning signal the model sees.

Currently implemented:
  - alpha_stable: A_1..A_T ~ skewed_levy alpha/2; cumulative Sigma recurrence.
    Owns path_pool, (G, M) path-sampling, logsignature feature extraction.
  - student_t: V_nu ~ InvGamma(nu/2, nu/2), T_t = t * V_nu; single scalar
    latent per sample. Marginal X_t is Student-t distributed (paper §3.3).
  - gaussian: vanilla FM baseline, X_1 = G (no clock, no path features).

All implement the Clock interface in clocks/base.py.
"""

from .alpha_stable import AlphaStableClock
from .student_t import StudentTClock
from .gaussian import GaussianClock


_REGISTRY = {
    'alpha_stable': AlphaStableClock,
    'student_t': StudentTClock,
    'gaussian': GaussianClock,
}


def build_clock(clock_type, **kwargs):
    """Factory. Returns a Clock instance."""
    if clock_type is None:
        return None
    if clock_type not in _REGISTRY:
        raise ValueError(
            f"Unknown clock_type={clock_type!r}. "
            f"Known: {list(_REGISTRY.keys())}")
    return _REGISTRY[clock_type](**kwargs)


def available_clocks():
    return list(_REGISTRY.keys())


def clock_feature_dim_from_cfg(p):
    """v15.4: resolve the conditioning feature dim purely from p (no
    clock instance required). Used by build_path_conditioning_module
    in init_model_by_parameter, which builds the model before the
    method (and thus before the clock instance exists).

    Centralizes the clock-type → feature_dim mapping that was
    previously inlined in dlpm_experiment.py.
    """
    pc = p.get('path_conditioning', {}) or {}
    fm_clock = (p.get('fm', {}) or {}).get('clock', {}) or {}
    clock_type = fm_clock.get('type', 'alpha_stable')
    if clock_type == 'student_t':
        feature_mode = fm_clock.get('feature_mode', 'log_scalar')
        return 2 if feature_mode == 'both' else 1
    if clock_type == 'alpha_stable':
        ft = pc.get('path_feature_type', 'logsignature')
        if ft == 'area':            # A(T) scalar summary of the clock path
            return 1
        if ft == 'area_terminal':   # (A(T), T_1) scalars
            return 2
        if ft == 'none':
            return 0
        from ..path import logsig_dim
        n_ch = len(pc.get('path_feature_input_channels', ['t', 'effective_T']))
        depth = pc.get('logsignature_level', 3)
        return logsig_dim(n_ch, depth)
    if clock_type == 'gaussian':
        return 0  # vanilla: no clock-derived feature
    raise ValueError(f'Unknown clock_type={clock_type!r}; '
                     f'extend clock_feature_dim_from_cfg.')


def clock_has_feature_source(p):
    """Whether the configured clock can produce a conditioning feature.

    Single source of truth for the path-conditioning enable derivation
    (dlpm_experiment._derive_path_conditioning_enabled) and the model-side
    flag computation (Model.py:MLPModel.__init__). Adding a new clock_type
    means extending this function in one place, instead of patching the
    `has_feature_source` predicate in multiple call sites.

    Behavior:
      - alpha_stable: True only if path_feature_type != 'none' (logsig
        encoder must be configured).
      - student_t: always True (V_nu provides an intrinsic scalar feature).
      - gaussian: always False (no random clock, no feature exists).
    """
    fm_clock = (p.get('fm', {}) or {}).get('clock', {}) or {}
    clock_type = fm_clock.get('type', 'alpha_stable')
    if clock_type == 'gaussian':
        return False
    if clock_type == 'student_t':
        return True
    if clock_type == 'alpha_stable':
        pc = p.get('path_conditioning', {}) or {}
        return pc.get('path_feature_type', 'none') != 'none'
    raise ValueError(f'Unknown clock_type={clock_type!r}; '
                     f'extend clock_has_feature_source.')
