"""Prediction-parameterization registry.

Adding a new parameterization (score, x0, consistency, ...):
  1. Create htfm/methods/predictors/<name>.py with a class subclassing Predictor.
  2. Register it in PREDICTOR_REGISTRY below with the string used in YAML's
     `<method>.prediction_type`.

Consumed by FlowMatchingLevyProcess.__init__ via build_predictor().
"""

from .base import Predictor
from .velocity import VelocityPredictor
from .epsilon import EpsilonPredictor


PREDICTOR_REGISTRY = {
    'velocity': VelocityPredictor,
    'epsilon': EpsilonPredictor,
}


def build_predictor(prediction_type, **kwargs):
    if prediction_type not in PREDICTOR_REGISTRY:
        raise ValueError(
            f'Unknown prediction_type={prediction_type!r}. '
            f'Known: {list(PREDICTOR_REGISTRY.keys())}')
    return PREDICTOR_REGISTRY[prediction_type](**kwargs)


def available_predictors():
    return list(PREDICTOR_REGISTRY.keys())
