#!/usr/bin/env python3
"""12K-vs-12K FID grid on CIFAR10-LT (paper Tab. cifar10lt-main-results, Tab. nfe-ablation).

Runs tools/eval_fid_12k.py over (run x epoch x reverse_steps) and skips cells whose JSON already
exists, so it can be relaunched at any time. Runs are either the HTFM variants trained by
scripts/cifar10lt_train.py (FP13_<variant>_s<seed>) or, with --baselines, the LIM / DLPM / GMFlow
baselines (BASE_*_s<seed>), evaluated with the same method mapping as training:

    LIM / GMFlow    stochastic sampler
    DLPM            stochastic sampler (DLPM row); add --dlim for the deterministic eta=0 pass (DLIM row)
    HTFM (fm)       Euler ODE, reverse_steps = NFE

Defaults reproduce the paper grid: epochs 500..1000 step 100, NFE {20, 50, 100}, one evaluation
per GPU slot, cheap-NFE-first ordering. Each cell writes results/fid/<name>_ep<E>_rs<R>[...].json
(see tools/eval_fid_12k.py) and logs to logs/cifar10lt_eval/.

Examples:
    python scripts/cifar10lt_eval.py --gpus 0 1 2 3 --seeds 42
    python scripts/cifar10lt_eval.py --baselines --dlim --gpus 4 5
    python scripts/cifar10lt_eval.py --variants a17_lev1 --epochs 700 --reverse_steps 100 --dry-run
"""
import argparse
import importlib.util
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PY, REPO, add_gpu_args, child_env, log, resolve_gpus, run_pool  # noqa: E402
from cifar10lt_train import BASELINES, SEEDS, VARIANTS  # noqa: E402

_spec = importlib.util.spec_from_file_location('eval_fid_12k', REPO / 'tools' / 'eval_fid_12k.py')
_efid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_efid)                      # torch-free at import time
default_out_path = _efid.default_out_path
cached_fid = _efid.cached_fid

LOG_DIR = REPO / 'logs' / 'cifar10lt_eval'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gpu_args(ap)
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help=f'default {SEEDS} for HTFM variants, [42] for --baselines')
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--baseline_names', nargs='+', default=None)
    ap.add_argument('--dlim', action='store_true', help='also evaluate DLPM baselines deterministically (DLIM)')
    ap.add_argument('--config_dir', default='cifar10lt')
    ap.add_argument('--epochs', type=int, nargs='+', default=[500, 600, 700, 800, 900, 1000])
    ap.add_argument('--reverse_steps', type=int, nargs='+', default=[20, 50, 100])
    ap.add_argument('--n_gen', type=int, default=12406)
    ap.add_argument('--batch_size', type=int, default=500)
    ap.add_argument('--out_dir', default='results/fid')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    a.seeds = a.seeds or ([42] if a.baselines else SEEDS)
    return a


def build_runs(a):
    """[(name, config, method, sampler)]"""
    runs = []
    if a.baselines:
        rows = [b for b in BASELINES if not a.baseline_names or b[0] in a.baseline_names]
        for seed in a.seeds:
            for stem, cfg, method, _capped in rows:
                m = method or ('gmflow' if 'gmflow' in stem else 'lim' if 'lim' in stem else 'dlpm')
                runs.append((f'{stem}_s{seed}', cfg, m, 'stochastic'))
                if a.dlim and m == 'dlpm':
                    runs.append((f'{stem}_s{seed}', cfg, m, 'deterministic'))
    else:
        for seed in a.seeds:
            for v in a.variants:
                runs.append((f'FP13_{v}_s{seed}', f'{a.config_dir}/FP13_{v}', 'fm', 'stochastic'))
    return runs


def build_cells(a):
    cells = []
    for rs in a.reverse_steps:                     # cheap NFE first
        for name, cfg, method, sampler in build_runs(a):
            for ep in a.epochs:
                out = default_out_path(name, ep, rs, method, 'euler', sampler, a.out_dir)
                if cached_fid(out) is None:
                    cells.append((name, cfg, method, sampler, ep, rs, out))
    return cells


def build_cmd(a, cell, gpu=None):
    name, cfg, method, sampler, ep, rs, out = cell
    cmd = [PY, 'tools/eval_fid_12k.py', '--config', cfg, '--name', name, '--epoch', str(ep),
           '--reverse_steps', str(rs), '--n_gen', str(a.n_gen), '--batch_size', str(a.batch_size),
           '--out_dir', a.out_dir]
    if method != 'fm':
        cmd += ['--method', method, '--sampler', sampler]
    return cmd


def main():
    a = parse_args()
    os.chdir(REPO)
    gpus = resolve_gpus(a)
    cells = build_cells(a)
    total = len(build_runs(a)) * len(a.epochs) * len(a.reverse_steps)
    log(f'{total} target cells, {len(cells)} to run (rest cached), GPUs {gpus} x {a.lanes}')
    if a.dry_run:
        for c in cells:
            print(' '.join(build_cmd(a, c)))
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = [0]

    def run_one(cell, gpu, lane, _lock):
        name, cfg, method, sampler, ep, rs, out = cell
        if cached_fid(out) is not None:
            return 'cached'
        tag = f'{name}_ep{ep}_rs{rs}' + ('_dlim' if sampler == 'deterministic' else '')
        log(f'START gpu{gpu}: {tag}')
        with open(LOG_DIR / f'{tag}.log', 'w') as lf:
            rc = subprocess.call(build_cmd(a, cell), cwd=REPO, env=child_env(gpu),
                                 stdout=lf, stderr=subprocess.STDOUT)
        fid = cached_fid(out)
        done[0] += 1
        log(f'{"DONE " if fid is not None else "FAIL "}gpu{gpu}: {tag} rc={rc} fid={fid} [{done[0]}/{len(cells)}]')
        return 'ok' if fid is not None else f'FAILED rc={rc}'

    results = run_pool(cells, gpus, a.lanes, run_one)
    failed = [c[0] for c, s in results if str(s).startswith(('FAILED', 'EXCEPTION'))]
    log(f'ALL DONE: {len(results)} cells, {len(failed)} failed')


if __name__ == '__main__':
    main()
