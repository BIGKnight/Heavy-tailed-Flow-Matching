"""Fixed pool of pre-sampled path (A) sequences for heavy-tailed FM.

Replaces continuous α-stable sampling with a finite, deterministic bank of N
paths. Training iterations sample batch_size A-sequences by drawing random
indices into the pool. Inference does the same from the *same* pool, so test-
time paths are guaranteed in-distribution.

Pool contents: shape [N, T_total] (isotropic skewed Lévy with optional clamp).
Stored at:
    data/path_pools/levy_a{alpha}_N{N}_T{T_total}_seed{seed}_clamp{clamp_a}.pt

Deterministic build (for bit-exact reproducibility across train/eval):
    rng = np.random.default_rng(seed)
    scipy.stats.levy_stable.rvs(alpha/2, 1, ..., size=N*T, random_state=rng)

On subsequent runs with the same (N, alpha, T, seed, clamp_a), the pool is
loaded from disk — bits-identical regardless of surrounding RNG state.

Note: only `noise_type == 'forward_process'` is supported (FP3_02/04/05, FP4).
Terminal-mode noise is a scalar per sample and doesn't benefit from pooling.
"""

import os
import hashlib

import numpy as np
import torch


class PathPool:

    def __init__(
        self,
        N,
        alpha,
        T_total,
        seed,
        clamp_a=None,
        cache_dir='data/path_pools',
        device='cpu',
    ):
        assert N > 0 and T_total > 0
        self.N = int(N)
        self.alpha = float(alpha)
        self.T = int(T_total)
        self.seed = int(seed)
        self.clamp_a = clamp_a
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        clamp_tag = 'none' if clamp_a is None else str(clamp_a)
        fname = (f'levy_a{self.alpha}_N{self.N}_T{self.T}_'
                 f'seed{self.seed}_clamp{clamp_tag}.pt')
        self.cache_path = os.path.join(cache_dir, fname)
        self.pool = None   # [N, T] tensor (kept on CPU; indexed per-batch)
        # Note: B11 (dedicated CPU Generator for index sampling) was reverted
        # to preserve byte-level identity with the pre-refactor sequence.
        # The reproducibility concern (pool indices share global CPU RNG with
        # dataloader) is real but should be revisited after paper data is
        # collected, since switching the generator changes drawn indices.

    def load_or_build(self, verbose=True):
        # File-locked build to avoid races with concurrent processes.
        # Common case (file already exists): no lock overhead, just load.
        if os.path.exists(self.cache_path):
            if verbose:
                print(f'[path_pool] loading {self.cache_path}', flush=True)
            self.pool = torch.load(self.cache_path, map_location='cpu')
            assert self.pool.shape == (self.N, self.T), (
                f'cached pool shape {self.pool.shape} != expected ({self.N}, {self.T})')
            return self

        # File missing: acquire exclusive lock, recheck, build if still missing.
        import fcntl
        lock_path = self.cache_path + '.lock'
        with open(lock_path, 'w') as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            if os.path.exists(self.cache_path):
                # Another process built it while we waited.
                if verbose:
                    print(f'[path_pool] lock: other process built it, loading '
                          f'{self.cache_path}', flush=True)
                self.pool = torch.load(self.cache_path, map_location='cpu')
            else:
                self._build(verbose=verbose)
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        # Best-effort cleanup
        try:
            os.unlink(lock_path)
        except OSError:
            pass
        return self

    def _build(self, verbose=True):
        import scipy.stats
        rng = np.random.default_rng(self.seed)
        scale = 2 * np.cos(np.pi * self.alpha / 4) ** (2 / self.alpha)
        if verbose:
            print(f'[path_pool] building {self.N}x{self.T} α={self.alpha} '
                  f'seed={self.seed} clamp_a={self.clamp_a}  '
                  f'-> {self.cache_path}', flush=True)
        samples = scipy.stats.levy_stable.rvs(
            self.alpha / 2, 1,
            loc=0, scale=scale,
            size=self.N * self.T,
            random_state=rng)
        pool = torch.tensor(samples, dtype=torch.float32).reshape(self.N, self.T)
        if self.clamp_a is not None:
            pool = pool.clamp_(0.0, float(self.clamp_a))
        self.pool = pool
        # Atomic write: save to tmp then rename.
        tmp_path = self.cache_path + f'.tmp.{os.getpid()}'
        torch.save(self.pool, tmp_path)
        os.rename(tmp_path, self.cache_path)
        h = hashlib.sha256(self.pool.numpy().tobytes()).hexdigest()
        with open(self.cache_path + '.sha256', 'w') as f:
            f.write(h + '\n')
        if verbose:
            size_mb = os.path.getsize(self.cache_path) / 1e6
            print(f'[path_pool] saved {size_mb:.1f} MB, sha256={h[:16]}...', flush=True)

    def sample_forward_process(self, batch_size, device=None):
        """Draw batch_size paths from pool.

        Returns A sequence with shape [T, batch_size] on `device` (or
        self.device if None), matching the shape conventions of the
        forward_process branch in flow_matching.
        """
        assert self.pool is not None, 'call load_or_build() first'
        if device is None:
            device = self.device
        idx = torch.randint(0, self.N, (batch_size,))
        chunk = self.pool[idx]            # [B, T] on CPU
        return chunk.to(device).T         # [T, B] on device

    def __repr__(self):
        loaded = 'loaded' if self.pool is not None else 'not loaded'
        return (f'PathPool(N={self.N}, α={self.alpha}, T={self.T}, '
                f'seed={self.seed}, clamp_a={self.clamp_a}, {loaded})')
