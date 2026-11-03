"""Time-factor functions g(t) used in loss weighting.

Adding a new g(t):
  1. Define a function g(t, cfg) returning a tensor.
  2. Register it via @register_time_fn('<name>').

Then YAML's `<method>.fm_loss_weighting.time_fn: <name>` picks it up.
"""

import torch


TIME_FNS = {}


def register_time_fn(name):
    def decorator(fn):
        TIME_FNS[name] = fn
        return fn
    return decorator


@register_time_fn('power')
def _time_power(t, cfg):
    """g(t) = t ^ time_power"""
    return t ** cfg['time_power']


@register_time_fn('log')
def _time_log(t, cfg):
    """g(t) = log(1 + log_scale * t)"""
    return torch.log1p(cfg['log_scale'] * t)


@register_time_fn('sigmoid')
def _time_sigmoid(t, cfg):
    """g(t) = sigmoid(sigmoid_k * (t - sigmoid_center)).
    Sharp transition around t = sigmoid_center."""
    return torch.sigmoid(cfg['sigmoid_k'] * (t - cfg['sigmoid_center']))
