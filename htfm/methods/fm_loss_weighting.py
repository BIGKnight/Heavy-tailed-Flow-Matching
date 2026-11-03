"""Backward-compat shim for v10 refactor.

The implementation moved to htfm/methods/loss_weighting/ (a subpackage with
time_fns / norm_fns / core split out for extension). New code should import
from there directly:

    from htfm.methods.loss_weighting import (
        get_config, validate_config, compute_weights, materialize_diagnostics,
    )

This shim preserves the old function names so callers using
    from .fm_loss_weighting import (
        get_fm_loss_weighting_config,
        validate_fm_loss_weighting_config,
        compute_fm_loss_weights,
        materialize_weight_diagnostics,
    )
continue to work without code changes.
"""

from .loss_weighting.core import (
    DEFAULTS as FM_LOSS_WEIGHTING_DEFAULTS,
    VALID_MODES,
    get_config as get_fm_loss_weighting_config,
    validate_config as validate_fm_loss_weighting_config,
    compute_weights as compute_fm_loss_weights,
    materialize_diagnostics as materialize_weight_diagnostics,
)
from .loss_weighting.time_fns import TIME_FNS as _TIME_FNS
from .loss_weighting.norm_fns import NORM_FNS as _NORM_FNS


__all__ = [
    'FM_LOSS_WEIGHTING_DEFAULTS',
    'VALID_MODES',
    'get_fm_loss_weighting_config',
    'validate_fm_loss_weighting_config',
    'compute_fm_loss_weights',
    'materialize_weight_diagnostics',
    '_TIME_FNS',
    '_NORM_FNS',
]
