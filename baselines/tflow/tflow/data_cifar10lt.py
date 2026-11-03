"""CIFAR10-LT dataset loader.

Self-contained reproduction of the long-tailed CIFAR-10 split used by the
companion HTFM project, deterministic under the same (max_count,
imbalance_ratio, seed) triple. Default settings match
``dlpm/configs/cifar10lt_*`` (max_count=5000, imbalance_ratio=0.01,
num_classes=10, seed=42; ~12.4k images at 32x32).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def _compute_lt_counts(max_count, imbalance_ratio, num_classes):
    denom = num_classes - 1
    return [
        int(np.floor(max_count * (imbalance_ratio ** (i / denom))))
        for i in range(num_classes)
    ]


def _build_or_load_indices(targets, *, max_count, imbalance_ratio, num_classes, seed,
                           cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)['selected_indices']

    counts = _compute_lt_counts(max_count, imbalance_ratio, num_classes)
    rng = np.random.default_rng(seed)
    selected = []
    targets_np = np.asarray(targets)
    for c in range(num_classes):
        idxs_c = np.where(targets_np == c)[0]
        rng.shuffle(idxs_c)
        selected.extend(idxs_c[:counts[c]].tolist())
    selected = sorted(int(i) for i in selected)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'selected_indices': selected, 'counts': counts,
                       'class_names': CIFAR10_CLASSES}, f)
    return selected


def get_cifar10lt(
    root: str = './data/cifar10',
    *,
    max_count: int = 5000,
    imbalance_ratio: float = 0.01,
    num_classes: int = 10,
    seed: int = 42,
    image_size: int = 32,
    random_flip: bool = True,
    cache_dir: str = './data/splits',
) -> Dataset:
    """Return the deterministic long-tailed CIFAR-10 train Subset."""
    train_tfms = [transforms.Resize(image_size), transforms.ToTensor()]
    if random_flip:
        train_tfms.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    transform = transforms.Compose(train_tfms)

    base = tv_datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    cache_path = os.path.join(
        cache_dir,
        f'cifar10_lt_max{max_count}_ratio{imbalance_ratio}_'
        f'classes{num_classes}_seed{seed}.json',
    )
    indices = _build_or_load_indices(
        base.targets,
        max_count=max_count,
        imbalance_ratio=imbalance_ratio,
        num_classes=num_classes,
        seed=seed,
        cache_path=cache_path,
    )
    return Subset(base, indices)


def make_dataloader(dataset, *, batch_size: int = 100, num_workers: int = 2,
                    drop_last: bool = True, shuffle: bool = True, pin_memory: bool = True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
