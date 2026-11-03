"""Loss-weighting subpackage.

Configurable per-sample loss weighting:
    weight(t, x_t) = base_weight + lambda_weight * g(t) * h(||x_t||)

Modes (cfg['mode']): 'none' | 'time' | 'norm' | 'time_norm'.
Time / norm functions live in their own registries (time_fns.py, norm_fns.py)
so adding a new variant is one file + one register decorator.

Public API:
    get_config(user_cfg)             merge user yml with defaults
    validate_config(cfg)             assertions for sanity
    compute_weights(t, x_t, cfg)     -> (weights, raw_diag)
    materialize_diagnostics(raw_diag) -> dict[str, float] for logging

Currently consumed only by FlowMatchingLevyProcess.training_losses_fm; the
schema is generic enough to apply to DLPM training too if ever wired up.
"""

from .core import (
    DEFAULTS,
    VALID_MODES,
    get_config,
    validate_config,
    compute_weights,
    materialize_diagnostics,
)
from .time_fns import TIME_FNS, register_time_fn
from .norm_fns import NORM_FNS, register_norm_fn


__all__ = [
    'DEFAULTS', 'VALID_MODES',
    'get_config', 'validate_config',
    'compute_weights', 'materialize_diagnostics',
    'TIME_FNS', 'register_time_fn',
    'NORM_FNS', 'register_norm_fn',
]
