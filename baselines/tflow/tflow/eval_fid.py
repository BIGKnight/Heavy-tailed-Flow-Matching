"""FID evaluation of a t-Flow checkpoint against the CIFAR10-LT train set.

Uses ``torch-fidelity`` (``pip install torch-fidelity``). The reference set
is the long-tailed split selected by :mod:`tflow.data_cifar10lt`; generated
samples are saved as PNGs to a (temp) directory and FID is computed
against the same number of real images.
"""

from __future__ import annotations

import os
import tempfile

import torch
from torchvision.utils import save_image

from .data_cifar10lt import get_cifar10lt
from .flow import euler_sample, heun_sample
from .model import TFlowModel


def _save_pngs(images, out_dir, prefix='img', start_idx=0):
    """Save a batch of [0, 1] images to PNGs with a global start index.

    Pulling ``start_idx`` from the caller avoids the index collision that
    occurs if every batch reuses local indices ``0..b-1``.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, x in enumerate(images):
        save_image(x, os.path.join(out_dir, f'{prefix}_{start_idx + i:06d}.png'))


@torch.no_grad()
def generate_to_dir(model, *, n_samples, nu, nfe, image_size, batch_size,
                    out_dir, device, solver='euler', param='noise'):
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    while written < n_samples:
        b = min(batch_size, n_samples - written)
        shape = (b, 3, image_size, image_size)
        if solver == 'euler':
            x = euler_sample(model, shape, nu, nfe=nfe, param=param,
                             device=device)
        elif solver == 'heun':
            x = heun_sample(model, shape, nu, nfe=nfe, param=param,
                            device=device)
        else:
            raise ValueError(f"unknown solver={solver!r}")
        x = (x.clamp(-1, 1) + 1) / 2
        _save_pngs(x, out_dir, prefix='gen', start_idx=written)
        written += b


def dump_real_to_dir(dataset, out_dir, n_real, image_size=32):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_real):
        x, _ = dataset[i % len(dataset)]
        save_image(x, os.path.join(out_dir, f'real_{i:06d}.png'))


def compute_fid(ckpt_path, *, nu, nfe, n_samples=12406, batch_size=100,
                image_size=32, cifar_root='./data/cifar10',
                cifar_split_dir='./data/splits', work_dir=None,
                solver='euler', param='noise', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    try:
        from torch_fidelity import calculate_metrics
    except ImportError as e:
        raise ImportError(
            'torch-fidelity is required for FID. Run `pip install torch-fidelity`.'
        ) from e

    ckpt = torch.load(ckpt_path, map_location=device)
    # Use checkpoint's recorded parameterization if available; CLI override
    # via ``param`` wins.
    ckpt_param = ckpt.get('cfg', {}).get('param', param)
    if param is None:
        param = ckpt_param

    model = TFlowModel(image_size=image_size).to(device)
    ema_sd = ckpt.get('ema', {}).get('shadow') if isinstance(ckpt.get('ema'), dict) else None
    model.load_state_dict(ema_sd if ema_sd is not None else ckpt['model'])
    model.eval()

    work_dir = work_dir or tempfile.mkdtemp(prefix='tflow_fid_')
    real_dir = os.path.join(work_dir, 'real')
    gen_dir = os.path.join(work_dir, 'gen')

    dataset = get_cifar10lt(root=cifar_root, cache_dir=cifar_split_dir,
                            image_size=image_size, random_flip=False)
    n_real = min(n_samples, len(dataset))
    dump_real_to_dir(dataset, real_dir, n_real=n_real, image_size=image_size)
    generate_to_dir(model, n_samples=n_samples, nu=nu, nfe=nfe,
                    image_size=image_size, batch_size=batch_size,
                    out_dir=gen_dir, device=device,
                    solver=solver, param=param)

    metrics = calculate_metrics(
        input1=gen_dir, input2=real_dir, fid=True, isc=False, kid=False,
        cuda=device.type == 'cuda',
    )
    return metrics['frechet_inception_distance']
