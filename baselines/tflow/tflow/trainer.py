"""t-Flow training loop on CIFAR10-LT."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch.optim import AdamW

from .data_cifar10lt import get_cifar10lt, make_dataloader
from .data_hrrr import get_hrrr_vil
from .ema import EMA
from .flow import flow_matching_loss
from .model import TFlowModel


@dataclass
class TrainConfig:
    nu: float = 3.0
    epochs: int = 2000
    batch_size: int = 100
    lr: float = 2e-4
    lr_gamma: float = 0.99
    lr_step: int = 1000
    warmup: int = 500
    grad_clip: float = 5.0
    ema_decay: float = 0.9999
    ckpt_freq: int = 100
    out_dir: str = './runs/tflow_default'
    seed: int | None = None
    num_workers: int = 2
    image_size: int = 32
    cifar_root: str = './data/cifar10'
    cifar_split_dir: str = './data/splits'
    # Pandey App. B regresses the source X_0 ('noise'); 'velocity' regresses
    # X_1 - X_0 directly (standard CFM). The two are mathematically
    # equivalent at the optimum but train differently.
    param: str = 'noise'
    # Dataset selection + backbone shape. 'cifar10lt' (default) keeps the
    # original 32x32x3 path; 'hrrr_vil' loads z-scored 128x128x1 VIL fields
    # (already model-scale, so the [0,1]->[-1,1] shift is skipped).
    dataset: str = 'cifar10lt'
    channels: int = 3
    channel_mult: tuple = (1, 2, 2, 2)
    hrrr_cache: str = ''
    hrrr_train_years: tuple = (2019, 2020)


def _lr_for_step(step: int, base_lr: float, warmup: int, step_size: int,
                 gamma: float) -> float:
    """Linear warmup * step decay, computed deterministically from ``step``.

    Replaces the redundant ``StepLR + manual override`` combination from the
    earlier draft (which was hard to reason about).
    """
    warm = 1.0 if warmup <= 0 else min(1.0, step / float(warmup))
    decay = gamma ** (step // step_size)
    return base_lr * warm * decay


def train(cfg: TrainConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.out_dir, exist_ok=True)

    if cfg.dataset == 'hrrr_vil':
        dataset = get_hrrr_vil(cfg.hrrr_cache, cfg.hrrr_train_years, image_size=cfg.image_size)
        shift_to_pm1 = False   # HRRR fields are already z-scored (model-scale)
    else:
        dataset = get_cifar10lt(
            root=cfg.cifar_root,
            cache_dir=cfg.cifar_split_dir,
            image_size=cfg.image_size,
        )
        shift_to_pm1 = True    # CIFAR pixels [0,1] -> [-1,1]
    loader = make_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )

    model = TFlowModel(image_size=cfg.image_size, channels=cfg.channels,
                       channel_mult=tuple(cfg.channel_mult)).to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    ema.shadow.to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss_sum, loss_n = 0.0, 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if shift_to_pm1:
                x = x * 2.0 - 1.0  # CIFAR [0,1] -> [-1, 1]

            lr_now = _lr_for_step(global_step, cfg.lr, cfg.warmup,
                                  cfg.lr_step, cfg.lr_gamma)
            for g in opt.param_groups:
                g['lr'] = lr_now

            opt.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, x, cfg.nu, param=cfg.param)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)

            loss_sum += loss.item() * x.shape[0]
            loss_n += x.shape[0]
            global_step += 1

        avg = loss_sum / max(1, loss_n)
        dt = time.time() - t0
        print(f'epoch={epoch:04d} step={global_step:07d} '
              f'loss={avg:.5f} dt={dt:.1f}s lr={lr_now:.2e}')

        if epoch % cfg.ckpt_freq == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(cfg.out_dir, f'ckpt_ep{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'opt': opt.state_dict(),
                'cfg': vars(cfg),
            }, ckpt_path)
            print(f'  -> saved {ckpt_path}')
