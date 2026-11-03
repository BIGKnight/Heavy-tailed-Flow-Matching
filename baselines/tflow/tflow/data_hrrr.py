"""HRRR-VIL dataset for t-Flow (ported from bem/datasets/hrrr.py of the main repository).

Loads cached per-year VIL fields (N, 128, 128) float32, drops NaN-filled slots, z-score
normalizes with the train-year stats in stats.npz, and returns (1, 128, 128) tensors that
are ALREADY in model-ready scale — the trainer must NOT apply the CIFAR [0,1]->[-1,1] shift
to these.

Self-contained (no dependency on the htfm package): reads the cache produced by
tools/build_hrrr_cache.py in the main repository.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_years(cache_dir, years):
    chunks = []
    for y in years:
        path = os.path.join(cache_dir, f'{int(y)}.npy')
        assert os.path.exists(path), f'HRRR cache file missing: {path}'
        arr = np.load(path, mmap_mode='r')
        valid = np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=-1)
        idxs = np.where(valid)[0]
        if len(idxs):
            chunks.append(np.array(arr[idxs], dtype=np.float32))
    assert chunks, f'no valid HRRR samples in {cache_dir} for years {years}'
    return np.concatenate(chunks, axis=0)


class HRRRVILDataset(Dataset):
    def __init__(self, cache_dir, years, mean, std, image_size=128):
        self.data = _load_years(cache_dir, years)
        assert self.data.shape[1:] == (image_size, image_size), (
            f'cached HRRR spatial shape {self.data.shape[1:]} != ({image_size},{image_size})')
        self.mean, self.std = float(mean), float(std)
        self._norm = (self.data - self.mean) / self.std       # z-score once

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self._norm[idx]).unsqueeze(0), torch.tensor(0.)  # (1,H,W)


def get_hrrr_vil(cache_dir, train_years, image_size=128):
    """Return the HRRR-VIL train dataset (z-score normalized with train-year stats)."""
    stats_path = os.path.join(cache_dir, 'stats.npz')
    assert os.path.exists(stats_path), (
        f'HRRR stats file missing: {stats_path}. Build it with tools/build_hrrr_cache.py first.')
    s = np.load(stats_path)
    mean, std = float(s['mean']), float(s['std'])
    ds = HRRRVILDataset(cache_dir, train_years, mean, std, image_size=image_size)
    ds.norm_mean, ds.norm_std = mean, std
    print(f'[hrrr_vil] {len(ds)} train samples (years={list(train_years)}); '
          f'mean={mean:.4f} std={std:.4f}; cache={cache_dir}', flush=True)
    return ds
