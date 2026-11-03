"""VelocityPredictor: model outputs the velocity field v(t, X_t) directly.

Training target = v* = X1 - X0 (straight-line conditional flow-matching target).
Sampling: identity — model output is already the velocity.
"""

from .base import Predictor


class VelocityPredictor(Predictor):
    def __init__(self, **kwargs):
        # Accept and discard kwargs (e.g. dlpm) so the factory can pass a
        # uniform set of args; VelocityPredictor doesn't need any state.
        pass

    def compute_target(self, X1, x0_ext, G):
        return X1 - x0_ext

    def to_velocity(self, model_output, x_t, t):
        return model_output
