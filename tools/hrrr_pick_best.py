"""Pick each HRRR baseline's best epoch by KR+SR+KS rank-sum (per reverse_steps).

Scans models/<name>/hrrr_vil/tail_metrics_ep*_rs*_n*.json. For each (model, rs), ranks the
epochs by KR, SR, KS separately (ascending — all lower-is-better), sums the three ranks, and
reports the min-rank-sum epoch (ties broken by KS). The chosen (ep) is used for BOTH the
marginal table and the spatial-eval generation, at a fixed rs across models.

Usage: python tools/hrrr_pick_best.py [rs]     # rs default 25
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

MODELS = ['HRRRB_lim_a17_s42', 'HRRRB_dlpm_a17_s42']
TARGET_RS = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def load():
    rows = defaultdict(list)   # (name, rs) -> [(ep, kr, sr, ks)]
    for name in MODELS:
        for jp in glob.glob(os.path.join('models', name, 'hrrr_vil', 'tail_metrics_ep*_rs*_n*.json')):
            m = re.search(r'ep(\d+)_rs(\d+)_', jp)
            if not m:
                continue
            d = json.load(open(jp))['metrics']
            rows[(name, int(m.group(2)))].append(
                (int(m.group(1)), d['kurtosis_ratio'], d['skewness_ratio'], d['ks2_tail']))
    return rows


def rank_sum_best(entries):
    """entries: [(ep, kr, sr, ks)] -> (best_ep, table sorted by rank-sum)."""
    eps = [e[0] for e in entries]
    def ranks(idx):
        order = sorted(range(len(entries)), key=lambda i: entries[i][idx])
        r = [0] * len(entries)
        for rank, i in enumerate(order):
            r[i] = rank + 1
        return r
    rk, rs, rks = ranks(1), ranks(2), ranks(3)
    scored = []
    for i, (ep, kr, sr, ks) in enumerate(entries):
        scored.append((rk[i] + rs[i] + rks[i], ks, ep, kr, sr, ks))
    scored.sort(key=lambda x: (x[0], x[1]))          # rank-sum, then KS
    best_ep = scored[0][2]
    return best_ep, scored


def main():
    rows = load()
    print(f'=== best epoch by KR+SR+KS rank-sum (rs={TARGET_RS}) ===\n')
    for name in MODELS:
        entries = rows.get((name, TARGET_RS), [])
        if not entries:
            print(f'{name} rs{TARGET_RS}: (no results yet)\n')
            continue
        best_ep, scored = rank_sum_best(entries)
        print(f'{name} rs{TARGET_RS}:  BEST ep={best_ep}  ({len(entries)} epochs swept)')
        print(f"  {'ep':>5} {'ranksum':>8} {'KR':>7} {'SR':>7} {'KS':>7}")
        for ranksum, _ks, ep, kr, sr, ks in scored:
            mark = '  <== best' if ep == best_ep else ''
            print(f'  {ep:>5} {ranksum:>8} {kr:>7.3f} {sr:>7.3f} {ks:>7.3f}{mark}')
        print()


if __name__ == '__main__':
    main()
