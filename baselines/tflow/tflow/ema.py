"""Exponential moving average of model parameters."""

from __future__ import annotations

import copy
from typing import Iterable

import torch


class EMA:
    """Maintains an EMA copy of a model's parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)
        # Sync buffers (e.g., for any future norm layers).
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

    def state_dict(self):
        return {'shadow': self.shadow.state_dict(), 'decay': self.decay}

    def load_state_dict(self, sd):
        self.decay = sd['decay']
        self.shadow.load_state_dict(sd['shadow'])
