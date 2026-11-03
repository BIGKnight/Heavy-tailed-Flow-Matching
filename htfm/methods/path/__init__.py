"""Path-conditioning subsystem for heavy-tailed flow matching.

Three layers of "path" abstraction:

  features.py   — PathFeatureEncoder (logsignature/signature of clock path),
                  helpers (logsig_dim, witt_dimension, get_lyndon_words,
                  compute_signature, compute_logsignature) and three
                  compute_clock_path_from_* utilities used by the encoder.
  pool.py       — PathPool: fixed pre-sampled bank of A sequences (discrete
                  prior) for variance reduction at the cost of OOD risk.
  sampling.py   — PathSampler: (group_size G, resample_interval M) variance
                  reduction over independent A draws; can read from a Pool.
  conditioning.py — PathConditioningModule (PCM): turns a path feature
                    vector into model-side conditioning (additive on time
                    embedding and/or FiLM modulation).

Public API is re-exported here so callers can write
    from htfm.methods.path import PathFeatureEncoder, PathPool, PathSampler, PathConditioningModule
"""

from .features import (
    PathFeatureEncoder,
    logsig_dim,
    sig_dim,
    witt_dimension,
    get_lyndon_words,
    compute_signature,
    compute_logsignature,
)
from .pool import PathPool
from .sampling import PathSampler
from .conditioning import PathConditioningModule, build_path_conditioning_module


__all__ = [
    'PathFeatureEncoder',
    'PathPool',
    'PathSampler',
    'PathConditioningModule',
    'build_path_conditioning_module',
    'logsig_dim',
    'sig_dim',
    'witt_dimension',
    'get_lyndon_words',
    'compute_signature',
    'compute_logsignature',
]
