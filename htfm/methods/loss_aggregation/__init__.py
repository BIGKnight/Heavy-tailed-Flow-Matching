"""Loss-aggregation registry.

Adding a new aggregator (e.g. trimmed mean, geometric mean, Catoni):
  1. Create htfm/methods/loss_aggregation/<name>.py with a class subclassing
     LossAggregator.
  2. Register it in AGGREGATOR_REGISTRY below with the string used in YAML's
     `training.<method>.loss_monte_carlo`.

Consumed by FlowMatchingLevyProcess.training_losses_fm and
DLPMTrainer.training_losses_dlpm (eps loss).
"""

from .base import LossAggregator
from .mean import MeanAggregator
from .median_of_means import MedianOfMeansAggregator
from .robust import RobustPowerAggregator


AGGREGATOR_REGISTRY = {
    'mean':   MeanAggregator,
    'median': MedianOfMeansAggregator,
    'robust': RobustPowerAggregator,
}


def build_aggregator(name):
    if name not in AGGREGATOR_REGISTRY:
        raise ValueError(
            f'Unknown loss_monte_carlo={name!r}. '
            f'Known: {list(AGGREGATOR_REGISTRY.keys())}')
    return AGGREGATOR_REGISTRY[name]()


def available_aggregators():
    return list(AGGREGATOR_REGISTRY.keys())
