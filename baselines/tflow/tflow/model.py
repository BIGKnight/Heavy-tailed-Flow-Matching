"""Thin adapter around the (copied) DDPM-style U-Net.

Wraps the ``Model`` class from :mod:`tflow.model_unet` so that the t-Flow
trainer can call it as ``model(x, t)`` without having to thread the YAML
config object through every call site.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_unet import Model as _UNet


def _config_dict(*, image_size, channels, model_channels, channel_mult,
                 num_res_blocks, attn_resolutions, dropout, resamp_with_conv,
                 diffusion_steps=1000):
    """Build the ``p`` dict that ``model_unet.Model`` consumes."""
    return {
        'data': {
            'image_size': image_size,
            'channels': channels,
        },
        'model': {
            'model_channels': model_channels,
            'channel_mult': list(channel_mult),
            'num_res_blocks': num_res_blocks,
            'attn_resolutions': list(attn_resolutions),
            'dropout': dropout,
            'resamp_with_conv': resamp_with_conv,
        },
        'diffusion': {
            'diffusion_steps': diffusion_steps,
        },
    }


class TFlowModel(nn.Module):
    """Velocity network for t-Flow.

    Identical backbone to the HTFM CIFAR10-LT setup (htfm/models/ddpm.py)
    (DDPM U-Net, 128 base channels, mults (1,2,2,2), 2 ResBlocks/scale,
    4 attention heads, attention at resolution 16).
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        channels: int = 3,
        model_channels: int = 128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ):
        super().__init__()
        self.cfg = _config_dict(
            image_size=image_size,
            channels=channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
        )
        self.net = _UNet(self.cfg)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # ``model_unet.Model`` expects t with shape (B,).
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(x, t)
