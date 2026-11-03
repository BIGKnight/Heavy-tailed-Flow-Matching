#!/usr/bin/env python3
"""Train the CIFAR10-LT models of paper Tab. cifar10lt-main-results (and the CIFAR column of
Tab. main-results): the 8 HTFM variants x seeds, or (--baselines) the LIM / DLPM / GMFlow baselines.

HTFM variants (configs htfm/configs/cifar10lt/FP13_<variant>.yml):
    a16_lev1 a17_lev1 a18_lev1 a19_lev1   HTFM-alpha, p_alpha = 1.6 .. 1.9, logsignature level 1
    st_nu1p7 st_nu2 st_nu3                HTFM-t, p_nu = 1.7 / 2.0 / 3.0
    vanilla                               HTFM-G (Gaussian flow matching)
Run name: FP13_<variant>_s<seed> -> models/FP13_<variant>_s<seed>/

Baselines (--baselines; seed 42 unless --seeds is given):
    BASE_lim_a{16,17,18,19}   cifar10lt/FP9_lim_a*_clip      method lim
    BASE_dlpm_a{16,17,18,19}  cifar10lt/FP13_a*_lev1         method dlpm (the FP13 YAMLs carry a
                              dlpm block identical to the internal FP6 baseline configs)
    BASE_dlpm_a2              cifar10lt/FP12_dlpm_a2         method dlpm, p = inf
    BASE_gmflow_k8_fix        extra/GMFLOW_k8_fix            method gmflow, runs to its config epochs

Training budget: the YAMLs say run.epochs = 2000 (1500 for GMFlow) but every reported CIFAR10-LT
model was trained with ``--epochs 1300`` (--train_to). The LR schedule (warm-up + step decay)
counts optimizer steps and never reads run.epochs, so this is a 2000-epoch run stopped at 1300;
raising --train_to and re-running extends finished runs from their last checkpoint (--resume is
added automatically when a checkpoint exists). Finished runs (model_*_<train_to>.pt present) are
skipped, so the script is restart-safe.

Scheduling: FIFO queue (seed-major) over ``len(--gpus) x --lanes`` slots. The second lane on a
GPU waits until the first job on that GPU has logged its first epoch (cuDNN autotune deadlocks
under concurrent same-GPU launches). Per-run logs: logs/cifar10lt_train/<name>.log.

Examples:
    python scripts/cifar10lt_train.py --gpus 0 1 2 3 4 5 6 7 --lanes 2
    python scripts/cifar10lt_train.py --variants a17_lev1 vanilla --seeds 42
    python scripts/cifar10lt_train.py --baselines --gpus 0 1 2 3
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PY, REPO, add_gpu_args, child_env, log, resolve_gpus, run_pool  # noqa: E402

VARIANTS = ['a16_lev1', 'a17_lev1', 'a18_lev1', 'a19_lev1', 'st_nu1p7', 'st_nu2', 'st_nu3', 'vanilla']
SEEDS = [42, 1234, 0, 4321, 100]
# (name stem, config, --method override or None, capped at --train_to?)
BASELINES = [
    ('BASE_lim_a16', 'cifar10lt/FP9_lim_a16_clip', None, True),
    ('BASE_lim_a17', 'cifar10lt/FP9_lim_a17_clip', None, True),
    ('BASE_lim_a18', 'cifar10lt/FP9_lim_a18_clip', None, True),
    ('BASE_lim_a19', 'cifar10lt/FP9_lim_a19_clip', None, True),
    ('BASE_dlpm_a16', 'cifar10lt/FP13_a16_lev1', 'dlpm', True),
    ('BASE_dlpm_a17', 'cifar10lt/FP13_a17_lev1', 'dlpm', True),
    ('BASE_dlpm_a18', 'cifar10lt/FP13_a18_lev1', 'dlpm', True),
    ('BASE_dlpm_a19', 'cifar10lt/FP13_a19_lev1', 'dlpm', True),
    ('BASE_dlpm_a2', 'cifar10lt/FP12_dlpm_a2', None, True),
    ('BASE_gmflow_k8_fix', 'extra/GMFLOW_k8_fix', None, False),
]
LOG_DIR = REPO / 'logs' / 'cifar10lt_train'
WARMUP_TIMEOUT_S = 45 * 60


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gpu_args(ap, lanes_default=2)
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help=f'default {SEEDS} for HTFM variants, [42] for --baselines')
    ap.add_argument('--variants', nargs='+', default=VARIANTS, help='HTFM variants to train')
    ap.add_argument('--baselines', action='store_true', help='train the baselines instead of the HTFM variants')
    ap.add_argument('--baseline_names', nargs='+', default=None,
                    help='subset of baseline stems, e.g. BASE_lim_a17 BASE_dlpm_a2')
    ap.add_argument('--train_to', type=int, default=1300, help='stop (and checkpoint) at this epoch')
    ap.add_argument('--config_dir', default='cifar10lt', help='config folder for the HTFM variants')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    a.seeds = a.seeds or ([42] if a.baselines else SEEDS)
    return a


def config_epochs(config):
    with open(REPO / 'htfm' / 'configs' / (config + '.yml')) as f:
        return int(yaml.safe_load(f)['run']['epochs'])


def build_jobs(a):
    """[(name, config, method, final_epoch)] in launch order (seed-major)."""
    jobs = []
    if a.baselines:
        rows = [b for b in BASELINES if not a.baseline_names or b[0] in a.baseline_names]
        for seed in a.seeds:
            for stem, cfg, method, capped in rows:
                final = a.train_to if capped else config_epochs(cfg)
                jobs.append((f'{stem}_s{seed}', cfg, method, final, seed, capped))
    else:
        for seed in a.seeds:
            for v in a.variants:
                jobs.append((f'FP13_{v}_s{seed}', f'{a.config_dir}/FP13_{v}', None, a.train_to, seed, True))
    return jobs


def final_ckpt_exists(name, epoch):
    return bool(glob.glob(str(REPO / 'models' / name / '*' / f'model_*_{epoch}.pt')))


def any_ckpt_exists(name):
    return bool(glob.glob(str(REPO / 'models' / name / '*' / 'model_*.pt')))


def build_cmd(name, cfg, method, final, seed, capped):
    cmd = [PY, 'run.py', '--config', cfg, '--name', name, '--set_seed', str(seed)]
    if method:
        cmd += ['--method', method]
    if capped:
        cmd += ['--epochs', str(final)]
    if any_ckpt_exists(name):
        cmd.append('--resume')
    return cmd


def wait_for_epoch1(log_path, proc):
    start = time.time()
    while time.time() - start < WARMUP_TIMEOUT_S:
        if proc.poll() is not None:
            return
        try:
            with open(log_path, errors='replace') as f:
                if 'epoch_loss' in f.read():
                    return
        except FileNotFoundError:
            pass
        time.sleep(20)
    log(f'WARN warm-up timeout for {log_path}; releasing the launch lock anyway')


def main():
    a = parse_args()
    os.chdir(REPO)
    gpus = resolve_gpus(a)
    jobs = build_jobs(a)
    if a.dry_run:
        for name, cfg, method, final, seed, capped in jobs:
            state = 'done' if final_ckpt_exists(name, final) else 'todo'
            print(f'{state:4s} {" ".join(build_cmd(name, cfg, method, final, seed, capped))}')
        print(f'\n[summary] {len(jobs)} runs on GPUs {gpus} x {a.lanes} lanes')
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f'queued {len(jobs)} runs on {len(gpus)} GPUs x {a.lanes} lanes')

    def run_one(job, gpu, lane, launch_lock):
        name, cfg, method, final, seed, capped = job
        if final_ckpt_exists(name, final):
            log(f'SKIP  {name} (epoch-{final} checkpoint exists)')
            return 'skipped'
        cmd = build_cmd(name, cfg, method, final, seed, capped)
        log_path = LOG_DIR / f'{name}.log'
        with launch_lock:                      # stagger launches on the same GPU
            with open(log_path, 'a') as lf:
                lf.write(f'\n===== gpu{gpu} lane{lane}: {" ".join(cmd)} =====\n')
                lf.flush()
                proc = subprocess.Popen(cmd, cwd=REPO, env=child_env(gpu), stdout=lf, stderr=subprocess.STDOUT)
            log(f'START {name} on gpu{gpu} lane{lane} (pid {proc.pid}){" [resume]" if "--resume" in cmd else ""}')
            wait_for_epoch1(log_path, proc)
        rc = proc.wait()
        status = 'ok' if rc == 0 else f'FAILED rc={rc}'
        log(f'END   {name} -> {status}')
        return status

    results = run_pool(jobs, gpus, a.lanes, run_one)
    failed = [(j[0], s) for j, s in results if str(s).startswith(('FAILED', 'EXCEPTION'))]
    log(f'ALL DONE: {len(results)} runs, {len(failed)} failed')
    for n, s in failed:
        log(f'  {n}: {s}')


if __name__ == '__main__':
    main()
