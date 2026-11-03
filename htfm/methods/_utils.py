"""Cross-method tensor utilities.

Used by FM (flow_matching.py), DLPM (dlpm.py), and GLP / LIM
(GenerativeLevyProcess.py). Lives in its own module so methods don't have
to import from each other for shared helpers (V12.1 refactor — completes
the V11 decoupling: FM no longer imports from htfm.py or GLP).

Contents:
  - match_last_dims(data, size)       broadcast-friendly reshape util
  - compute_loss(tens, lploss)        Lp norm helper (general lploss)
  - compute_loss_terms(x, y, lploss)  per-sample {L1, L2, sqL2, Lp} loss
"""

import torch
import torch.nn as nn


def match_last_dims(data, size):
    """Repeat a 1-D tensor so its trailing dims match size[1:].

    Used to broadcast a per-batch scalar schedule (sigmas[t], gammas[t], ...)
    across the spatial / channel dims of an image batch.
    """
    assert len(data.size()) == 1, '1-dimensional input expected (one value per batch)'
    for i in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))


def compute_loss(tens, lploss):
    """Return per-sample Lp norm of `tens` raised to 1/lploss.

    norm(tens, p=lploss, dim=non-batch) ** (1/lploss).
    """
    return torch.pow(
        torch.linalg.norm(
            tens,
            ord=lploss,
            dim=list(range(1, len(tens.shape))),
        ),
        1 / lploss,
    )


def compute_loss_terms(x, y, lploss):
    """Per-sample loss term between predictions x and target y.

    Specializes for common cases:
      lploss == 2.   -> sqrt(MSE) per sample (L2)
      lploss == 1.   -> smooth-L1 mean per sample (L1)
      lploss == -1   -> raw MSE per sample (squared L2, no sqrt)
      else           -> general Lp via compute_loss(x - y, lploss).
    """
    if lploss == 2.:
        tmp = nn.functional.mse_loss(x, y, reduction='none')
        tmp = torch.sqrt(tmp.mean(dim=list(range(1, len(x.shape)))))
        return tmp
    if lploss == 1.:
        tmp = nn.functional.smooth_l1_loss(x, y, beta=1, reduction='none')
        return tmp.mean(dim=list(range(1, len(x.shape))))
    if lploss == -1:
        return nn.functional.mse_loss(x, y, reduction='none').mean(
            dim=list(range(1, len(x.shape))))
    return compute_loss(x - y, lploss)
