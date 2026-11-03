"""Median-of-means aggregator (robust to per-A heavy-tailed outliers).

Inner mean over (G, t) draws conditional on A; outer median over A draws.
"""

from .base import LossAggregator


class MedianOfMeansAggregator(LossAggregator):
    def aggregate(self, losses, monte_carlo_outer, monte_carlo_inner,
                  monte_carlo_t, B, **kwargs):
        inner_size = monte_carlo_inner * monte_carlo_t
        losses = losses.reshape(monte_carlo_outer, inner_size, B)
        losses = losses.mean(dim=1)            # inner mean over (G, t)
        losses, _ = losses.median(dim=0)       # outer median over A
        return losses.mean()
