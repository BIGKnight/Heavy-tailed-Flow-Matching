"""Abstract Clock base class for heavy-tailed flow matching.

A "clock" determines how the heavy-tailed noise endpoint x_1 is generated and
what latent conditioning signal (if any) the model sees. Subclasses provide:

  - sample_X1_and_features(...) for training-time x_1 + path_feature production
  - sample_initial(...) for inference-time initial state + path_feature

This base class is intentionally minimal: it pins the I/O contract but does
not force a uniform "Path object" abstraction, because alpha-stable produces
a sequence A_1..A_T while Student-t produces a single scalar V.

This module is part of the v1 clock-abstraction refactor. Step A introduces
this base class but does NOT route any code through it yet.
"""


class Clock:
    """Abstract base. Subclasses set feature_dim and override the methods below."""

    feature_dim = 0

    # Class-level None defaults for alpha-stable-only attrs (path_pool /
    # samplers). Subclasses that don't use them inherit these as-is; the
    # AlphaStableClock subclass shadows them via instance attrs in __init__.
    path_pool = None
    train_sampler = None
    eval_sampler = None

    # Hyperparams that should be embedded in ckpt 'extra' under
    # extra['clock']['kwargs']. Override on subclasses with non-empty
    # metadata (e.g. Student-t exposes nu, feature_mode).
    ckpt_metadata_kwargs = {}

    # Capability flags (v15.1): clocks declare what config blocks they
    # accept. Host code (flow_matching.py / methods/__init__.py) and the
    # YAML shim consult these instead of hard-coding clock-type checks.
    # Defaults are False; AlphaStableClock overrides both to True.
    supports_path_pool = False
    supports_path_sampling = False

    def feature_dim_for_conditioning(self):
        """Dim of the raw clock feature passed to the PCM (PathConditioning
        Module). Default returns self.feature_dim; subclasses can override
        when the conditioning feature comes from an external encoder
        (alpha-stable: logsig dim from path_encoder).
        """
        return getattr(self, 'feature_dim', 0)

    def set_path_encoder(self, path_encoder):
        """Optional: receive a path-feature encoder. Default is a no-op."""
        pass

    def get_ckpt_extra(self):
        """Optional: contribute to the ckpt 'extra' dict. Default returns {}.
        Host merges over default {'path_pool': None}; clocks that own a path
        pool (AlphaStableClock) override this to provide the path_pool entry.
        """
        return {}

    def validate_ckpt_extra(self, extra):
        """Optional: validate clock-related fields in a loaded ckpt 'extra' dict."""
        pass

    def sample_X1_and_features(
        self, x_start, x0_ext, G,
        monte_carlo_inner, monte_carlo_outer,
    ):
        """Training-time. Returns (X1, path_features|None)."""
        raise NotImplementedError

    def sample_initial(self, shape):
        """Inference-time. Returns (x_init, A_sample, G, path_features|None)."""
        raise NotImplementedError
