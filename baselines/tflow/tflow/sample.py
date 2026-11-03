"""Sample images from a trained t-Flow checkpoint."""

from __future__ import annotations

import os

import torch
from torchvision.utils import save_image

from .flow import euler_sample, heun_sample
from .model import TFlowModel


@torch.no_grad()
def sample_from_ckpt(
    ckpt_path: str,
    *,
    n_samples: int = 64,
    nu: float | None = None,
    nfe: int = 100,
    image_size: int = 32,
    out_path: str | None = None,
    device: str | None = None,
    solver: str = 'euler',
    param: str | None = None,
):
    """Load the EMA model from ``ckpt_path`` and generate ``n_samples`` images."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('cfg', {})
    if nu is None:
        nu = cfg.get('nu', 3.0)
    if param is None:
        param = cfg.get('param', 'noise')

    model = TFlowModel(image_size=image_size).to(device)
    ema_sd = ckpt.get('ema', {}).get('shadow') if isinstance(ckpt.get('ema'), dict) else None
    if ema_sd is not None:
        model.load_state_dict(ema_sd)
    else:
        model.load_state_dict(ckpt['model'])
    model.eval()

    shape = (n_samples, 3, image_size, image_size)
    if solver == 'euler':
        x = euler_sample(model, shape, nu, nfe=nfe, param=param, device=device)
    elif solver == 'heun':
        x = heun_sample(model, shape, nu, nfe=nfe, param=param, device=device)
    else:
        raise ValueError(f"unknown solver={solver!r}")
    x = (x.clamp(-1, 1) + 1) / 2

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        save_image(x, out_path, nrow=int(n_samples ** 0.5))
    return x
