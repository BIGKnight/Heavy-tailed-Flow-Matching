"""Norm-factor functions h(||x_t||) used in loss weighting.

Adding a new h(r):
  1. Define a function h(r, cfg) returning a tensor.
  2. Register it via @register_norm_fn('<name>').

Then YAML's `<method>.fm_loss_weighting.norm_fn: <name>` picks it up.
"""

import torch


NORM_FNS = {}


def register_norm_fn(name):
    def decorator(fn):
        NORM_FNS[name] = fn
        return fn
    return decorator


@register_norm_fn('power')
def _norm_power(r, cfg):
    """h(r) = (r / norm_ref) ^ norm_power"""
    norm_ref = max(cfg['norm_ref'], cfg['eps'])
    return (r / norm_ref) ** cfg['norm_power']


@register_norm_fn('log')
def _norm_log(r, cfg):
    """h(r) = log(1 + r / norm_ref). Heavy-tail safe: grows slowly for large r."""
    norm_ref = max(cfg['norm_ref'], cfg['eps'])
    return torch.log1p(r / norm_ref)


@register_norm_fn('tanh')
def _norm_tanh(r, cfg):
    """h(r) = tanh(r / norm_ref). Bounded in [0, 1)."""
    norm_ref = max(cfg['norm_ref'], cfg['eps'])
    return torch.tanh(r / norm_ref)
