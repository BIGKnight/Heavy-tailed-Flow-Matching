"""
Tail-evaluation metrics for heavy-tailed generative model fit on HRRR.

Implements the three metrics from Pandey et al. 2024 (Heavy-Tailed Diffusion
Models, App. C.1.3) used to score VIL / w20 generation:

  - Kurtosis Ratio (KR):    |1 - kurt_sim / kurt_data|       (lower is better)
  - Skewness Ratio (SR):    |1 - skew_sim / skew_data|       (lower is better)
  - Two-sample KS on tails: KS-2 between tail-restricted samples (lower is
    better). Tails are defined relative to the data percentiles
    (>99.9% for the right tail, <0.1% for the left tail). When both
    tails are present we average the two; when only one tail is heavy
    (e.g. VIL right tail, no left tail), we report just that side.

All functions accept generated and data tensors of any shape (they get
flattened internally) and operate in numpy with float64 to keep the
moment computations numerically stable.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kurtosis, skew, ks_2samp


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1).astype(np.float64)


def kurtosis_ratio(samples_sim, samples_data) -> float:
    """KR = |1 - kurt(sim)/kurt(data)|, using Fisher's definition (excess)."""
    s = _to_numpy(samples_sim)
    d = _to_numpy(samples_data)
    k_sim = float(kurtosis(s, fisher=True, bias=False))
    k_dat = float(kurtosis(d, fisher=True, bias=False))
    if k_dat == 0:
        return float('inf')
    return abs(1.0 - k_sim / k_dat)


def skewness_ratio(samples_sim, samples_data) -> float:
    """SR = |1 - skew(sim)/skew(data)|."""
    s = _to_numpy(samples_sim)
    d = _to_numpy(samples_data)
    sk_sim = float(skew(s, bias=False))
    sk_dat = float(skew(d, bias=False))
    if sk_dat == 0:
        return float('inf')
    return abs(1.0 - sk_sim / sk_dat)


def ks2_tails(samples_sim, samples_data, q: float = 0.999,
              tails: str = 'both') -> Tuple[float, Dict[str, float]]:
    """Two-sample Kolmogorov-Smirnov restricted to distribution tails.

    Args:
      samples_sim:  array-like, generated samples (any shape)
      samples_data: array-like, ground-truth samples (any shape)
      q: percentile threshold for tail definition (e.g. 0.999 → top/bottom 0.1%).
      tails: 'right', 'left', or 'both'. For VIL (no left tail) pass 'right';
             for w20 'left'; if symmetric or unknown, 'both' averages.

    Returns:
      (ks_score, breakdown_dict).
        ks_score: average KS over the requested tail(s).
        breakdown_dict: per-side {'right': float, 'left': float, ...} for QA.
    """
    assert 0.0 < q < 1.0, f'q must be in (0, 1), got {q}'
    s = _to_numpy(samples_sim)
    d = _to_numpy(samples_data)

    breakdown: Dict[str, float] = {}
    components = []

    if tails in ('right', 'both'):
        thr = np.quantile(d, q)
        d_tail = d[d >= thr]
        s_tail = s[s >= thr]
        if len(d_tail) > 0 and len(s_tail) > 0:
            stat, _p = ks_2samp(s_tail, d_tail)
            breakdown['right'] = float(stat)
            components.append(stat)
        else:
            # When the model produces zero samples in the tail, the empirical
            # CDFs differ everywhere → assign 1.0 (worst possible KS).
            breakdown['right'] = 1.0
            components.append(1.0)

    if tails in ('left', 'both'):
        thr = np.quantile(d, 1.0 - q)
        d_tail = d[d <= thr]
        s_tail = s[s <= thr]
        if len(d_tail) > 0 and len(s_tail) > 0:
            stat, _p = ks_2samp(s_tail, d_tail)
            breakdown['left'] = float(stat)
            components.append(stat)
        else:
            breakdown['left'] = 1.0
            components.append(1.0)

    return float(np.mean(components)), breakdown


def hrrr_tail_metrics(samples_sim, samples_data,
                      channel: str = 'vil',
                      q: float = 0.999) -> Dict[str, float]:
    """Convenience wrapper: compute Pandey's KR / SR / KS-2 in one call.

    Channel-aware defaults:
      - 'vil': right-tail-only KS (Pandey: "for the VIL channel, we report
        KS scores only for the right tail due to the absence of a left tail").
      - 'w20': both tails (heavy left, mild right).
      - other: both tails.
    """
    tails = 'right' if channel == 'vil' else 'both'
    kr = kurtosis_ratio(samples_sim, samples_data)
    sr = skewness_ratio(samples_sim, samples_data)
    ks, ks_breakdown = ks2_tails(samples_sim, samples_data, q=q, tails=tails)

    out = {
        'kurtosis_ratio': kr,
        'skewness_ratio': sr,
        'ks2_tail': ks,
        f'ks2_tail_q': float(q),
        f'ks2_tail_tails': tails,
    }
    for side, val in ks_breakdown.items():
        out[f'ks2_{side}'] = val
    return out


# Quick sanity test if invoked directly.
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # Heavy-tailed reference: Student-t with low ν.
    data = rng.standard_t(df=3.0, size=200_000)
    # Light-tailed simulator: standard normal.
    sim_bad = rng.standard_normal(size=200_000)
    # Good simulator: Student-t at correct df.
    sim_good = rng.standard_t(df=3.0, size=200_000)

    print('--- bad simulator (Gaussian vs Student-t data) ---')
    for k, v in hrrr_tail_metrics(sim_bad, data, channel='w20').items():
        print(f'  {k}: {v}')

    print('--- good simulator (Student-t vs Student-t) ---')
    for k, v in hrrr_tail_metrics(sim_good, data, channel='w20').items():
        print(f'  {k}: {v}')
