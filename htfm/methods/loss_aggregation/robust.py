"""Robust power aggregator: E[(E[loss|A])^r] with r < alpha/2 (DLPM Prop 9)."""

from .base import LossAggregator


class RobustPowerAggregator(LossAggregator):
    def aggregate(self, losses, monte_carlo_outer, monte_carlo_inner,
                  monte_carlo_t, B, robust_exponent=0.5, **kwargs):
        inner_size = monte_carlo_inner * monte_carlo_t
        losses = losses.reshape(monte_carlo_outer, inner_size, B)
        inner_mean = losses.mean(dim=1)                 # inner expectation
        robust_losses = inner_mean ** robust_exponent   # power r < alpha/2
        return robust_losses.mean()
