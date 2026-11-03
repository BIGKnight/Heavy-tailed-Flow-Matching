"""Loss-weighting core: config defaults, validation, weight computation, diagnostics.

Functional form:
    weight(t, x_t) = base_weight + lambda_weight * g(t) * h(||x_t||)

Modes:
    - 'none':      weight = 1  (no-op)
    - 'time':      weight = base_weight + lambda_weight * g(t)
    - 'norm':      weight = base_weight + lambda_weight * h(||x_t||)
    - 'time_norm': weight = base_weight + lambda_weight * g(t) * h(||x_t||)

`g` and `h` come from the time_fns / norm_fns registries (siblings).

Note on Monte Carlo interaction:
    When monte_carlo_outer/inner > 1, weights are computed per replicate
    (each has its own Xt → its own ||Xt||). 'norm' / 'time_norm' modes
    therefore modify the quantity being robust-aggregated, not just the
    final loss. 'time' is unaffected (t is shared across replicates).
"""

import torch

from .time_fns import TIME_FNS
from .norm_fns import NORM_FNS


DEFAULTS = {
    'enabled': False,
    'mode': 'none',
    'time_fn': 'power',
    'norm_fn': 'power',
    'base_weight': 1.0,
    'lambda_weight': 0.0,
    'time_power': 1.0,
    'norm_power': 1.0,
    'norm_ref': 1.0,
    'log_scale': 1.0,
    'sigmoid_k': 10.0,
    'sigmoid_center': 0.5,
    'max_weight': None,
    'min_weight': None,
    'detach_norm': True,
    'eps': 1e-8,
    'log_interval': 500,
}

VALID_MODES = ('none', 'time', 'norm', 'time_norm')

_FLOAT_KEYS = ('base_weight', 'lambda_weight', 'time_power', 'norm_power',
               'norm_ref', 'log_scale', 'sigmoid_k', 'sigmoid_center',
               'max_weight', 'min_weight', 'eps')


def get_config(user_cfg):
    """Merge user config with defaults, returning a complete config dict.

    If user_cfg is None or empty, returns defaults (weighting disabled).
    """
    cfg = dict(DEFAULTS)
    if user_cfg is not None:
        cfg.update(user_cfg)
    for k in _FLOAT_KEYS:
        if cfg[k] is not None:
            cfg[k] = float(cfg[k])
    return cfg


def validate_config(cfg):
    """Lightweight validation. Raises AssertionError on invalid values."""
    assert cfg['mode'] in VALID_MODES, (
        f'loss_weighting mode must be one of {VALID_MODES}, got {cfg["mode"]}')

    assert cfg['time_power'] is not None and cfg['time_power'] >= 0, (
        f'time_power must be non-negative, got {cfg["time_power"]}')
    assert cfg['norm_power'] is not None and cfg['norm_power'] >= 0, (
        f'norm_power must be non-negative, got {cfg["norm_power"]}')
    assert cfg['norm_ref'] is not None and cfg['norm_ref'] > 0, (
        f'norm_ref must be positive (not None), got {cfg["norm_ref"]}')
    assert cfg['eps'] is not None and cfg['eps'] > 0, (
        f'eps must be positive, got {cfg["eps"]}')

    assert cfg['base_weight'] is not None and cfg['base_weight'] >= 0, (
        f'base_weight must be non-negative, got {cfg["base_weight"]}')
    assert cfg['lambda_weight'] is not None and cfg['lambda_weight'] >= 0, (
        f'lambda_weight must be non-negative, got {cfg["lambda_weight"]}')

    if cfg['max_weight'] is not None and cfg['min_weight'] is not None:
        assert cfg['max_weight'] >= cfg['min_weight'], (
            f'max_weight ({cfg["max_weight"]}) must be >= '
            f'min_weight ({cfg["min_weight"]})')

    if cfg['min_weight'] is not None:
        assert cfg['min_weight'] >= 0, (
            f'min_weight must be non-negative, got {cfg["min_weight"]}')

    if cfg['mode'] in ('norm', 'time_norm'):
        assert cfg['max_weight'] is not None, (
            f'max_weight is required for mode={cfg["mode"]} to prevent '
            f'overflow under heavy-tailed norms')
        assert cfg['norm_fn'] in NORM_FNS, (
            f'Unknown norm_fn: {cfg["norm_fn"]}. Available: {list(NORM_FNS.keys())}')

    if cfg['mode'] in ('time', 'time_norm'):
        assert cfg['time_fn'] in TIME_FNS, (
            f'Unknown time_fn: {cfg["time_fn"]}. Available: {list(TIME_FNS.keys())}')


def _compute_per_sample_norm(x_t, detach):
    """Per-sample L2 norm over all non-batch dims."""
    x = x_t.detach() if detach else x_t
    flat = x.reshape(x.shape[0], -1)
    return torch.norm(flat, p=2, dim=1)


def compute_weights(t, x_t, cfg):
    """Compute per-sample loss weights.

    Returns (weights, raw_diag). raw_diag is a dict of GPU tensors;
    use materialize_diagnostics() to convert to Python floats for logging.
    """
    mode = cfg['mode']

    if not cfg['enabled'] or mode == 'none':
        return torch.ones(t.shape[0], device=t.device), {}

    g_t = TIME_FNS[cfg['time_fn']](t, cfg) if mode in ('time', 'time_norm') else None
    if mode in ('norm', 'time_norm'):
        r = _compute_per_sample_norm(x_t, cfg['detach_norm'])
        h_r = NORM_FNS[cfg['norm_fn']](r, cfg)
    else:
        h_r = None

    if mode == 'time':
        factor = g_t
    elif mode == 'norm':
        factor = h_r
    elif mode == 'time_norm':
        factor = g_t * h_r

    weights = cfg['base_weight'] + cfg['lambda_weight'] * factor

    if cfg['min_weight'] is not None:
        weights = torch.clamp(weights, min=cfg['min_weight'])
    if cfg['max_weight'] is not None:
        weights = torch.clamp(weights, max=cfg['max_weight'])

    assert torch.isfinite(weights).all(), (
        'Non-finite loss weights detected (NaN or Inf). Check config values.')

    raw_diag = {'weights': weights}
    if g_t is not None:
        raw_diag['g_t'] = g_t
    if h_r is not None:
        raw_diag['h_r'] = h_r
    return weights, raw_diag


def materialize_diagnostics(raw_diag):
    """Convert raw tensor diagnostics to Python floats (triggers GPU sync).

    Call only when actually logging, not every step.
    """
    if not raw_diag:
        return {}

    w = raw_diag['weights']
    out = {
        'fm_weight/mean': w.mean().item(),
        'fm_weight/max': w.max().item(),
        'fm_weight/min': w.min().item(),
    }
    if 'g_t' in raw_diag:
        g = raw_diag['g_t']
        out['fm_weight/g_t_mean'] = g.mean().item()
        out['fm_weight/g_t_max'] = g.max().item()
        out['fm_weight/g_t_min'] = g.min().item()
    if 'h_r' in raw_diag:
        h = raw_diag['h_r']
        out['fm_weight/h_r_mean'] = h.mean().item()
        out['fm_weight/h_r_max'] = h.max().item()
        out['fm_weight/h_r_min'] = h.min().item()
    return out
