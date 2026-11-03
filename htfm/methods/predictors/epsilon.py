"""EpsilonPredictor: model outputs the standardized Gaussian noise eps.

Under the DLPM Gaussian-mixture representation X1 = barsigma_T * sqrt(A) * G,
G is the standardized noise and is the natural target for an eps-prediction
parameterization.

Sampling: under the straight-line schedule, the velocity is approximately
v ~ barsigma_T * eps (see FM paper §4.2 eq 4.8). The exact mapping depends
on the schedule; this simplified form matches the legacy code path.
"""

from .base import Predictor


class EpsilonPredictor(Predictor):
    def __init__(self, schedule, **kwargs):
        # Need barsigma_T from the schedule for the eps -> v mapping.
        self.schedule = schedule

    def compute_target(self, X1, x0_ext, G):
        # The standardized noise itself is the target (X1 = barsigma_T * sqrt(A) * G).
        return G

    def to_velocity(self, model_output, x_t, t):
        barsigma_T = self.schedule.barsigmas[-1]
        return barsigma_T * model_output
