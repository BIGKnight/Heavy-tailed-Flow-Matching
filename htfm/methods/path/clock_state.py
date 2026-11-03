"""ClockState: dataclasses describing how a clock-path is materialized.

The PathFeatureEncoder consumes a `ClockState` and produces (t_grid, T_raw,
T_effective) tensors.  Two concrete states are currently supported:

  - TerminalA   — single scalar A per sample (constant-A approximation;
                  used in the alpha-stable terminal forward branch and the
                  velocity-averaging k_mean variant).
  - ASequence   — full per-step A_1..A_T sequence (forward_process branch);
                  T_effective comes from the DLPM Sigma recurrence.

Adding a new state (e.g. for Student-t's V) is one dataclass + one branch
in PathFeatureEncoder.compute_clock_path.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TerminalA:
    """Constant-A approximation of the clock path.

    Attributes:
        A_scalar:  [batch] tensor — one scalar A per sample.
        barsigmas: [T] tensor — full schedule (only the indexed values are read).
    """
    A_scalar: torch.Tensor
    barsigmas: torch.Tensor


@dataclass(frozen=True)
class ASequence:
    """Full per-step A sequence + DLPM recurrence for T_effective.

    Attributes:
        sigmas:     [T] per-step noise scales.
        gammas:     [T] per-step decay factors.
        A_sequence: [T, batch, ...] subordinator values.
    """
    sigmas: torch.Tensor
    gammas: torch.Tensor
    A_sequence: torch.Tensor
