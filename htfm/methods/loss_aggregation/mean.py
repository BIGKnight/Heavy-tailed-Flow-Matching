"""Plain mean over all per-sample losses."""

from .base import LossAggregator


class MeanAggregator(LossAggregator):
    def aggregate(self, losses, monte_carlo_outer, monte_carlo_inner,
                  monte_carlo_t, B, **kwargs):
        return losses.mean()
