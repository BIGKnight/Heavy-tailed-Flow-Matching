"""Abstract LossAggregator base class.

Aggregates a [B * MC_outer * MC_inner * MC_t]-shaped per-sample loss tensor
into a scalar training loss. Different aggregators correspond to different
estimators of the heavy-tailed loss expectation:

  - mean:      simple flat mean (Gaussian-style; biased low under heavy tails).
  - median:    median-of-means (Levy & Lerasle); robust to a few outliers in A.
  - robust:    E[(E[loss|A])^r] power aggregation with r < alpha/2 (DLPM Prop 9).

The reshape contract: per-sample losses are laid out as
    losses.shape == [B * MC_outer * MC_inner * MC_t]
where the leading axis is x0_ext.repeat(MC_outer*MC_inner) followed by .repeat(MC_t).
"""


class LossAggregator:
    """Abstract base. Subclasses override aggregate(...)."""

    def aggregate(self, losses, monte_carlo_outer, monte_carlo_inner,
                  monte_carlo_t, B, **kwargs):
        """Reduce per-sample `losses` to a scalar training loss.

        Args:
            losses: [B * MC_outer * MC_inner * MC_t] flat per-sample losses.
            monte_carlo_outer: # independent A draws per data point.
            monte_carlo_inner: # G draws per A.
            monte_carlo_t:     # t draws per (X0, X1) pair.
            B:                 # data points per batch.
            **kwargs:          # aggregator-specific (e.g. robust_exponent).

        Returns:
            scalar tensor (0-d) suitable for backward().
        """
        raise NotImplementedError
