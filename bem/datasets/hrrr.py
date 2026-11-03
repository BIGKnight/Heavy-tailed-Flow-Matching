"""
HRRR (High-Resolution Rapid Refresh) PyTorch Dataset for heavy-tailed
flow-matching experiments.

Loads VIL (Vertically Integrated Liquid) hourly analysis fields cached locally
by `tools/build_hrrr_cache.py`, applies per-channel z-score normalization using
training-year statistics, and returns float tensors of shape (1, 128, 128).

This setup mirrors Pandey et al. 2024 (arXiv 2410.14171) Sec C.1.1:
  - train years: 2019, 2020 (~17.4k samples)
  - test year:   2021       (~ 8.7k samples)
  - region: 128x128 patch over the Central US (REGEN-compat Oklahoma window)
  - normalization: z-score with stats precomputed from train years only

Config keys consumed (under `data:` in YAML):
    dataset       : 'hrrr_vil'
    cache_dir     : default './data/hrrr/vil'
    image_size    : must be 128 (we assert)
    channels      : must be 1   (we assert)
    train_years   : default [2019, 2020]
    test_years    : default [2021]
    drop_invalid  : default True (skip NaN-filled slots from missing HRRR hours)
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_int_list(x, default):
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if hasattr(x, '__iter__'):
        return [int(v) for v in x]
    return [int(x)]


def _load_year_arrays(cache_dir, years):
    """Load per-year `.npy` files, drop NaN-rows, return (concat array, mask, year_idx).

    Returns:
      data   : float32 array (N_valid, H, W)
      times_year : int32 (N_valid,) — origin year for QA
      times_idx  : int64 (N_valid,) — hour-of-year index for QA
    """
    chunks = []
    yyears = []
    yidx = []
    for y in years:
        path = os.path.join(cache_dir, f'{y}.npy')
        assert os.path.exists(path), (
            f'HRRR cache file missing: {path}. Run tools/build_hrrr_cache.py first.')
        arr = np.load(path, mmap_mode='r')
        valid = np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=-1)
        idxs = np.where(valid)[0].astype(np.int64)
        if len(idxs) == 0:
            continue
        chunks.append(np.array(arr[idxs], dtype=np.float32))
        yyears.append(np.full(len(idxs), y, dtype=np.int32))
        yidx.append(idxs)
    if not chunks:
        raise RuntimeError(f'No valid HRRR samples found in {cache_dir} for years {years}.')
    data = np.concatenate(chunks, axis=0)
    yyears = np.concatenate(yyears)
    yidx = np.concatenate(yidx)
    return data, yyears, yidx


class HRRRDataset(Dataset):
    """In-memory HRRR dataset returning (x, dummy_label) tuples.

    The full Phase-1 cache (3 years of 128x128 hourly fields) is ~1.7 GB
    fp32; we just hold it in RAM.
    """

    def __init__(self, cache_dir, years, mean, std, image_size=128):
        self.cache_dir = cache_dir
        self.years = list(years)
        self.image_size = int(image_size)
        self.mean = float(mean)
        self.std = float(std)
        self.data, self.years_per_sample, self.hour_idx = _load_year_arrays(cache_dir, self.years)
        assert self.data.shape[1:] == (self.image_size, self.image_size), (
            f'cached HRRR has spatial shape {self.data.shape[1:]} '
            f'but image_size={self.image_size}')
        # Pre-normalize once (cheap, avoids recomputing in __getitem__).
        self._normalized = (self.data - self.mean) / self.std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self._normalized[idx]).unsqueeze(0)  # (1, H, W)
        return x, torch.tensor(0.)


def get_hrrr_datasets(config, default_channel='vil') -> Tuple[HRRRDataset, HRRRDataset]:
    """Entry point for `bem.datasets.get_dataset` dispatcher.

    Returns (train_ds, test_ds). Both share the same train-year mean/std
    (Pandey: stats are computed on the train years and applied to test).

    `default_channel` decides the cache subdirectory if `data.cache_dir`
    is not explicitly set in the config; pass 'vil' for HRRR_VIL or 'w20'
    for HRRR_W20.
    """
    d = config.data
    cache_dir = getattr(d, 'cache_dir', f'./data/hrrr/{default_channel}')
    image_size = int(getattr(d, 'image_size', 128))
    train_years = _as_int_list(getattr(d, 'train_years', None), default=[2019, 2020])
    test_years = _as_int_list(getattr(d, 'test_years', None), default=[2021])

    channels = int(getattr(d, 'channels', 1))
    assert channels == 1, f'hrrr_vil expects channels=1, got {channels}'
    assert image_size == 128, f'hrrr_vil expects image_size=128, got {image_size}'

    stats_path = os.path.join(cache_dir, 'stats.npz')
    assert os.path.exists(stats_path), (
        f'HRRR stats file missing: {stats_path}. '
        f'Run `python tools/build_hrrr_cache.py --stats_only` after the cache is built.')
    s = np.load(stats_path)
    mean, std = float(s['mean']), float(s['std'])

    train_ds = HRRRDataset(cache_dir, train_years, mean, std, image_size=image_size)
    test_ds = HRRRDataset(cache_dir, test_years, mean, std, image_size=image_size)

    print(f'[{default_channel}] train: {len(train_ds)} samples (years={train_years}); '
          f'test: {len(test_ds)} samples (years={test_years}); '
          f'mean={mean:.4f} std={std:.4f}; cache={cache_dir}', flush=True)

    # Stash mean/std on the dataset objects so eval / denormalization can find them.
    for ds in (train_ds, test_ds):
        ds.norm_mean = mean
        ds.norm_std = std

    return train_ds, test_ds
