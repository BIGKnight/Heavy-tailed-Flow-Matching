#!/usr/bin/env python3
"""HRRR-VIL marginal tail sweep: KR / SR / KS per checkpoint, then rank-sum best epoch
(paper Tab. main-results, HRRR columns).

For every run and checkpoint epoch, tools/eval_hrrr_tails.py loads the EMA model, generates
--generate (20,000) VIL fields with the config's sampler at --reverse_steps (25; the HRRR configs
use Heun, so NFE = 50) and scores them against the 2021 test year with Pandey et al.'s
kurtosis ratio (KR), skewness ratio (SR) and right-tail two-sample KS at q = 0.999.
Each cell lands in models/<name>/hrrr_vil/tail_metrics_ep<E>_rs<R>_n<N>.json (written by the
evaluator); cells that already exist are skipped.

Afterwards, per (run, reverse_steps), epochs are ranked by KR, SR and KS separately and the
epoch with the smallest rank-sum (ties -> smaller KS) is reported, using the same selection as
tools/hrrr_pick_best.py. The table is printed and written to results/hrrr/tail_sweep_summary.json.

Default runs (name -> config, method):
    HTFM_HRRR_a16 / a17 / a18 / a19   hrrr_vil/HRRR_VIL_alpha1*      fm    (HTFM-alpha)
    HTFM_HRRR_vanilla                 hrrr_vil/HRRR_VIL_vanilla      fm    (HTFM-G)
    HRRR_dlpm_a2                      hrrr_vil/HRRR_VIL_dlpm_a2      dlpm
    HRRRB_dlpm_a17_s42                hrrr_vil/HRRR_VIL_dlpm_a17     dlpm
    HRRRB_lim_a16_s42 / a17_s42       hrrr_vil/HRRR_VIL_lim_a1*      lim
Add or replace runs with --runs NAME=CONFIG[:METHOD] (method defaults to fm).

GPU gating: a worker only claims a GPU with at least --min_free_gb free (nvidia-smi), so the
sweep can share GPUs with training jobs; set 0 to disable.

Examples:
    python scripts/hrrr_tail_sweep.py --gpus 0 1 2 3 --epochs 500 600 700 800 900 1000 1100 1200
    python scripts/hrrr_tail_sweep.py --runs HTFM_HRRR_a17=hrrr_vil/HRRR_VIL_alpha17 --reverse_steps 25 50 100
    python scripts/hrrr_tail_sweep.py --summary-only
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PY, REPO, add_gpu_args, child_env, free_mb, log, resolve_gpus, run_pool  # noqa: E402

DEFAULT_RUNS = [
    ('HTFM_HRRR_a16', 'hrrr_vil/HRRR_VIL_alpha16', 'fm'),
    ('HTFM_HRRR_a17', 'hrrr_vil/HRRR_VIL_alpha17', 'fm'),
    ('HTFM_HRRR_a18', 'hrrr_vil/HRRR_VIL_alpha18', 'fm'),
    ('HTFM_HRRR_a19', 'hrrr_vil/HRRR_VIL_alpha19', 'fm'),
    ('HTFM_HRRR_vanilla', 'hrrr_vil/HRRR_VIL_vanilla', 'fm'),
    ('HRRR_dlpm_a2', 'hrrr_vil/HRRR_VIL_dlpm_a2', 'dlpm'),
    ('HRRRB_dlpm_a17_s42', 'hrrr_vil/HRRR_VIL_dlpm_a17', 'dlpm'),
    ('HRRRB_lim_a16_s42', 'hrrr_vil/HRRR_VIL_lim_a16', 'lim'),
    ('HRRRB_lim_a17_s42', 'hrrr_vil/HRRR_VIL_lim_a17', 'lim'),
]
LOG_DIR = REPO / 'logs' / 'hrrr_tail_sweep'
SUMMARY = REPO / 'results' / 'hrrr' / 'tail_sweep_summary.json'


def load_rank_sum_best():
    """tools/hrrr_pick_best.py parses sys.argv at import; shield it."""
    spec = importlib.util.spec_from_file_location('hrrr_pick_best', REPO / 'tools' / 'hrrr_pick_best.py')
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod.rank_sum_best


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gpu_args(ap)
    ap.add_argument('--runs', nargs='+', default=None, help='NAME=CONFIG[:METHOD] (replaces the default list)')
    ap.add_argument('--epochs', type=int, nargs='+', default=list(range(400, 1301, 100)))
    ap.add_argument('--reverse_steps', type=int, nargs='+', default=[25])
    ap.add_argument('--generate', type=int, default=20000)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--min_free_gb', type=float, default=30.0, help='GPU memory gate; 0 disables')
    ap.add_argument('--summary-only', action='store_true', help='skip evaluation, only rebuild the summary')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.runs:
        runs = []
        for spec in a.runs:
            name, rest = spec.split('=', 1)
            cfg, _, method = rest.partition(':')
            runs.append((name, cfg, method or 'fm'))
        a.run_list = runs
    else:
        a.run_list = DEFAULT_RUNS
    return a


def dataset_dir(cfg):
    with open(REPO / 'htfm' / 'configs' / (cfg + '.yml')) as f:
        return yaml.safe_load(f)['data']['dataset']


def jpath(name, ds, ep, rs, n):
    return REPO / 'models' / name / ds / f'tail_metrics_ep{ep}_rs{rs}_n{n}.json'


def build_cmd(a, name, cfg, method, ep, rs):
    return [PY, 'tools/eval_hrrr_tails.py', '--config', cfg, '--name', name, '--method', method,
            '--epoch', str(ep), '--generate', str(a.generate), '--reverse_steps', str(rs),
            '--batch_size', str(a.batch_size)]


def summarize(a, rank_sum_best):
    rows = defaultdict(list)                       # (name, rs) -> [(ep, kr, sr, ks)]
    for name, cfg, _m in a.run_list:
        ds = dataset_dir(cfg)
        for jp in glob.glob(str(REPO / 'models' / name / ds / f'tail_metrics_ep*_rs*_n{a.generate}.json')):
            m = re.search(r'ep(\d+)_rs(\d+)_', os.path.basename(jp))
            if not m:
                continue
            d = json.load(open(jp))['metrics']
            rows[(name, int(m.group(2)))].append(
                (int(m.group(1)), d['kurtosis_ratio'], d['skewness_ratio'], d['ks2_tail']))
    summary = {}
    for rs in a.reverse_steps:
        print(f'\n=== best epoch by KR+SR+KS rank-sum (rs={rs}, n={a.generate}) ===')
        for name, _cfg, _m in a.run_list:
            entries = sorted(rows.get((name, rs), []))
            if not entries:
                print(f'{name}: (no results)')
                continue
            best_ep, scored = rank_sum_best(entries)
            print(f'{name}: BEST ep={best_ep} ({len(entries)} epochs)')
            print(f"  {'ep':>5} {'ranksum':>8} {'KR':>7} {'SR':>7} {'KS':>7}")
            for ranksum, _ks, ep, kr, sr, ks in scored:
                print(f'  {ep:>5} {ranksum:>8} {kr:>7.3f} {sr:>7.3f} {ks:>7.3f}{"  <== best" if ep == best_ep else ""}')
            summary[f'{name}@rs{rs}'] = {
                'name': name, 'reverse_steps': rs, 'best_epoch': best_ep,
                'best': dict(zip(('kurtosis_ratio', 'skewness_ratio', 'ks2_tail'),
                                 next(e[1:] for e in entries if e[0] == best_ep))),
                'epochs': [{'epoch': e, 'kurtosis_ratio': kr, 'skewness_ratio': sr, 'ks2_tail': ks}
                           for e, kr, sr, ks in entries],
            }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nsummary -> {SUMMARY}')


def main():
    a = parse_args()
    os.chdir(REPO)
    rank_sum_best = load_rank_sum_best()
    if a.summary_only:
        summarize(a, rank_sum_best)
        return
    gpus = resolve_gpus(a)
    cells = []
    for rs in a.reverse_steps:                     # fast feedback: smallest rs first
        for name, cfg, method in a.run_list:
            ds = dataset_dir(cfg)
            for ep in a.epochs:
                if not jpath(name, ds, ep, rs, a.generate).exists():
                    cells.append((name, cfg, method, ds, ep, rs))
    total = len(a.run_list) * len(a.epochs) * len(a.reverse_steps)
    log(f'{total} target cells, {len(cells)} to run (rest cached), GPUs {gpus} x {a.lanes}')
    if a.dry_run:
        for name, cfg, method, _ds, ep, rs in cells:
            print(' '.join(build_cmd(a, name, cfg, method, ep, rs)))
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = [0]

    def run_one(cell, gpu, lane, _lock):
        name, cfg, method, ds, ep, rs = cell
        out = jpath(name, ds, ep, rs, a.generate)
        if out.exists():
            return 'cached'
        waited = 0
        while a.min_free_gb > 0:
            free = free_mb(gpu)
            if free is None or free >= a.min_free_gb * 1024:
                break
            if waited == 0:
                log(f'gpu{gpu} busy ({free} MB free); waiting to start {name} ep{ep} rs{rs}')
            time.sleep(60)
            waited += 1
        log(f'START gpu{gpu}: {name} ep{ep} rs{rs}')
        with open(LOG_DIR / f'{name}_ep{ep}_rs{rs}.log', 'w') as lf:
            rc = subprocess.call(build_cmd(a, name, cfg, method, ep, rs), cwd=REPO, env=child_env(gpu),
                                 stdout=lf, stderr=subprocess.STDOUT)
        ok = out.exists()
        done[0] += 1
        log(f'{"DONE " if ok else "FAIL "}gpu{gpu}: {name} ep{ep} rs{rs} rc={rc} [{done[0]}/{len(cells)}]')
        return 'ok' if ok else f'FAILED rc={rc}'

    results = run_pool(cells, gpus, a.lanes, run_one)
    failed = [c[0] for c, s in results if str(s).startswith(('FAILED', 'EXCEPTION'))]
    log(f'ALL DONE: {len(results)} cells, {len(failed)} failed')
    summarize(a, rank_sum_best)


if __name__ == '__main__':
    main()
