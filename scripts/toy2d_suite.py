#!/usr/bin/env python3
"""2D imbalanced alpha-stable-mixture suite (paper Tab. main-results, 2D columns; Fig. viz-toy).

Trains and evaluates every variant on the two 9-component S-alpha-S grid mixtures
(configs ``htfm/configs/toy2d/ch_sas{18,15}_*``, data alpha_D = 1.8 / 1.5):

  dlpm          DLPM baseline, method alpha in --alphas
  fm_no_pc      HTFM-alpha (alpha-stable clock, terminal source), alpha in --alphas
                (alpha = 2.0 is HTFM-G, i.e. Gaussian flow matching)
  fm_pc         HTFM-alpha + logsignature clock conditioning, (group_size G, resample_interval M)
                in --gm_grid
  st_fm_no_pc   HTFM-t (Student-t clock), nu in --nus
  st_fm_pc      HTFM-t + clock conditioning, nu in --nus

Protocol (unchanged from the internal runs): 300 epochs, seed 42, checkpoints every 25 epochs
from epoch 100, 1024 training reverse steps, 24,000 generated samples per evaluation, evaluated
with ``eval.py`` at the epochs in --eval_epochs; DLPM is evaluated with its stochastic sampler at
--dlpm_steps and as DLIM (deterministic) at --dlim_steps, flow variants at --fm_nfes steps of the
Euler ODE solver.  ``f_1_pr`` (the paper metric) and ``wass`` are parsed from each eval log into
<out_dir>/metrics.csv.

Run names follow the internal scheme (chv4_<dataset>_<variant>_a<alpha> / _nu<nu> / _G<G>_M<M>)
so checkpoints under models/ stay comparable with the original runs.  The original suite was
split over three launches (alphas {2.0, 1.8, 1.5} + G/M grid, then alpha 1.7, then alpha 1.6 and
nu 1.7); the defaults here are the union.

Examples:
    python scripts/toy2d_suite.py --dry-run
    python scripts/toy2d_suite.py --gpus 0 1 2 3 --yes
    python scripts/toy2d_suite.py --only-variant fm_no_pc --only-alpha 1.7 --eval_epochs 100 200 300 --yes
    python scripts/toy2d_suite.py --skip-training --yes          # re-evaluate + rebuild metrics.csv
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PY, REPO, add_gpu_args, log, resolve_gpus, run_pool  # noqa: E402

CONFIG_DIR = 'toy2d'
DATASETS = [('ch_sas18', 'sas18'), ('ch_sas15', 'sas15')]
VARIANTS = ['dlpm', 'fm_no_pc', 'fm_pc', 'st_fm_no_pc', 'st_fm_pc']
CKPT_SUBDIR = 'sas_grid'            # = data.dataset of the toy2d configs


def fmt_nu(nu):
    return f'{nu}'.replace('.', 'p')


@dataclass
class TrainRun:
    config: str
    name: str
    method: str
    dataset: str
    variant: str
    alpha: float
    epochs: int
    nu: Optional[float] = None
    group_size: Optional[int] = None
    resample_interval: Optional[int] = None

    def command(self, a):
        cmd = [PY, 'run.py', '--config', self.config, '--name', self.name, '--method', self.method,
               '--alpha', str(self.alpha), '--epochs', str(self.epochs), '--eval', str(self.epochs),
               '--check', str(a.check_freq), '--check_start', str(a.check_start),
               '--set_seed', str(a.seed), '--loss_monte_carlo', 'mean', '--monte_carlo_outer', '1',
               '--train_reverse_steps', str(a.train_reverse_steps)]
        if self.nu is not None:
            cmd += ['--nu', str(self.nu)]
        if self.group_size is not None:
            cmd += ['--group_size', str(self.group_size)]
        if self.resample_interval is not None:
            cmd += ['--resample_interval', str(self.resample_interval)]
        return cmd

    log_suffix = 'train'


@dataclass
class EvalRun:
    train: TrainRun
    eval_mode: str        # 'dlpm' | 'dlim' | 'fm'
    nfe: int
    epoch: int

    def command(self, a):
        t = self.train
        cmd = [PY, 'eval.py', '--config', t.config, '--name', t.name, '--method', t.method,
               '--alpha', str(t.alpha), '--epochs', str(self.epoch), '--eval', str(self.epoch),
               '--generate', str(a.generate), '--reverse_steps', str(self.nfe),
               '--set_seed', str(a.seed), '--train_reverse_steps', str(a.train_reverse_steps)]
        if self.eval_mode == 'dlim':
            cmd.append('--deterministic')
        if t.nu is not None:
            cmd += ['--nu', str(t.nu)]
        if t.group_size is not None:
            cmd += ['--group_size', str(t.group_size)]
        if t.resample_interval is not None:
            cmd += ['--resample_interval', str(t.resample_interval)]
        return cmd

    @property
    def name(self):
        return self.train.name

    @property
    def log_suffix(self):
        return f'eval_{self.eval_mode}{self.nfe}' + ('' if self.epoch == self.train.epochs else f'_e{self.epoch}')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_gpu_args(ap)
    ap.add_argument('--out_dir', default='results/toy2d', help='logs, metrics.csv and config copies')
    ap.add_argument('--alphas', type=float, nargs='+', default=[2.0, 1.8, 1.7, 1.6, 1.5],
                    help='method alpha for dlpm / fm_no_pc / fm_pc (2.0 = Gaussian)')
    ap.add_argument('--nus', type=float, nargs='+', default=[3.0, 2.0, 1.8, 1.7, 1.5],
                    help='Student-t degrees of freedom for st_fm_*')
    ap.add_argument('--gm_grid', nargs='+', default=['1,1', '16,4', '64,8'],
                    help='fm_pc (group_size,resample_interval) pairs')
    ap.add_argument('--pc_alphas', type=float, nargs='+', default=[2.0, 1.8, 1.5],
                    help='alphas for the fm_pc G/M sweep (original launch used these three)')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--eval_epochs', type=int, nargs='+', default=None,
                    help='checkpoint epochs to evaluate (default: only --epochs)')
    ap.add_argument('--check_freq', type=int, default=25)
    ap.add_argument('--check_start', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--generate', type=int, default=24000)
    ap.add_argument('--train_reverse_steps', type=int, default=1024)
    ap.add_argument('--dlpm_steps', type=int, nargs='+', default=[10, 20, 40, 80, 120])
    ap.add_argument('--dlim_steps', type=int, nargs='+', default=[6, 10, 15, 20, 25, 30])
    ap.add_argument('--fm_nfes', type=int, nargs='+', default=[6, 10, 15, 20, 25, 30])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--yes', '-y', action='store_true', help='do not ask for confirmation')
    ap.add_argument('--skip-training', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--resume', action='store_true', help='skip training runs whose final checkpoint exists')
    ap.add_argument('--only-variant', choices=VARIANTS, default=None)
    ap.add_argument('--only-dataset', choices=[d for _, d in DATASETS], default=None)
    ap.add_argument('--only-alpha', type=float, default=None)
    ap.add_argument('--only-nu', type=float, default=None)
    a = ap.parse_args()
    a.eval_epochs = a.eval_epochs or [a.epochs]
    a.gm_grid = [tuple(int(x) for x in g.split(',')) for g in a.gm_grid]
    return a


def build_train_runs(a):
    runs = []
    for cfg_stem, dataset in DATASETS:
        cfg = f'{CONFIG_DIR}/{cfg_stem}'
        for al in a.alphas:
            runs.append(TrainRun(f'{cfg}_dlpm', f'chv4_{dataset}_dlpm_a{al}', 'dlpm', dataset, 'dlpm', al, a.epochs))
        for al in a.alphas:
            runs.append(TrainRun(f'{cfg}_fm_no_pc', f'chv4_{dataset}_fm_no_pc_a{al}', 'fm', dataset, 'fm_no_pc', al, a.epochs))
        for al in a.pc_alphas:
            for G, M in a.gm_grid:
                runs.append(TrainRun(f'{cfg}_fm_pc', f'chv4_{dataset}_fm_pc_a{al}_G{G}_M{M}', 'fm', dataset,
                                     'fm_pc', al, a.epochs, group_size=G, resample_interval=M))
        for nu in a.nus:   # Student-t clock: fm.alpha stays 2.0
            runs.append(TrainRun(f'{cfg}_st_fm_no_pc', f'chv4_{dataset}_st_fm_no_pc_nu{fmt_nu(nu)}', 'fm',
                                 dataset, 'st_fm_no_pc', 2.0, a.epochs, nu=nu))
        for nu in a.nus:
            runs.append(TrainRun(f'{cfg}_st_fm_pc', f'chv4_{dataset}_st_fm_pc_nu{fmt_nu(nu)}', 'fm',
                                 dataset, 'st_fm_pc', 2.0, a.epochs, nu=nu))
    return runs


def build_eval_runs(train_runs, a):
    runs = []
    for tr in train_runs:
        for ep in a.eval_epochs:
            if tr.method == 'dlpm':
                runs += [EvalRun(tr, 'dlpm', n, ep) for n in a.dlpm_steps]
                runs += [EvalRun(tr, 'dlim', n, ep) for n in a.dlim_steps]
            else:
                runs += [EvalRun(tr, 'fm', n, ep) for n in a.fm_nfes]
    return runs


def matches(run, a):
    tr = run.train if isinstance(run, EvalRun) else run
    if a.only_variant and tr.variant != a.only_variant:
        return False
    if a.only_dataset and tr.dataset != a.only_dataset:
        return False
    if a.only_alpha is not None and tr.variant in ('dlpm', 'fm_no_pc', 'fm_pc') and float(tr.alpha) != a.only_alpha:
        return False
    if a.only_nu is not None and tr.nu is not None and float(tr.nu) != a.only_nu:
        return False
    return True


def ckpt_exists(name, epoch):
    return bool(list((REPO / 'models' / name / CKPT_SUBDIR).glob(f'model_*_{epoch}.pt')))


def log_path(a, run):
    return Path(a.out_dir) / 'logs' / f'{run.name}_{run.log_suffix}.log'


def write_log(path, cmd, rc, out, err):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('$ ' + ' '.join(cmd) + f'\n\n[exit] {rc}\n')
        if out:
            f.write('\n[stdout]\n' + out + ('' if out.endswith('\n') else '\n'))
        if err:
            f.write('\n[stderr]\n' + err + ('' if err.endswith('\n') else '\n'))


def extract_metric(text, key):
    m = re.findall(rf'^\t{re.escape(key)}\s*=\s*(.+)$', text, re.MULTILINE)
    if not m:
        return None
    num = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m[-1])
    return num.group(0) if num else None


def collect_metrics(eval_runs, a):
    rows = []
    for r in eval_runs:
        try:
            text = log_path(a, r).read_text(encoding='utf-8')
        except FileNotFoundError:
            text = ''
        t = r.train
        rows.append({'dataset': t.dataset, 'variant': t.variant, 'method_alpha': t.alpha,
                     'nu': t.nu if t.nu is not None else '', 'G': t.group_size if t.group_size is not None else '',
                     'M': t.resample_interval if t.resample_interval is not None else '',
                     'eval_mode': r.eval_mode, 'nfe': r.nfe, 'epochs': r.epoch,
                     'wass': extract_metric(text, 'wass') or 'ERROR',
                     'f_1_pr': extract_metric(text, 'f_1_pr') or 'ERROR',
                     'log_path': str(log_path(a, r))})
    return rows


def run_jobs(jobs, a, gpus):
    def run_one(job, gpu, lane, _lock):
        cmd = job.command(a)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
        write_log(log_path(a, job), cmd, proc.returncode, proc.stdout, proc.stderr)
        if proc.returncode != 0:
            log(f'ERROR {job.name} {job.log_suffix} (exit={proc.returncode}) see {log_path(a, job)}')
        return proc.returncode
    return run_pool(jobs, gpus, a.lanes, run_one)


def main():
    a = parse_args()
    os.chdir(REPO)
    gpus = resolve_gpus(a)
    train_runs = [r for r in build_train_runs(a) if matches(r, a)]
    eval_runs = [r for r in build_eval_runs(train_runs, a) if matches(r, a)]

    if a.dry_run:
        for r in train_runs:
            print('TRAIN:', ' '.join(r.command(a)))
        for r in eval_runs:
            print('EVAL :', ' '.join(r.command(a)))
        print(f'\n[summary] {len(train_runs)} train + {len(eval_runs)} eval jobs on GPUs {gpus}')
        return 0
    if not a.yes:
        ans = input(f'About to run {len(train_runs)} train + {len(eval_runs)} eval jobs on GPUs {gpus}. '
                    f'Proceed? [y/N] ').strip().lower()
        if ans not in ('y', 'yes'):
            print('Aborted.')
            return 0

    for d in ('logs', 'configs'):
        (Path(a.out_dir) / d).mkdir(parents=True, exist_ok=True)
    for src in sorted((REPO / 'htfm' / 'configs' / CONFIG_DIR).glob('*.yml')):
        dst = Path(a.out_dir) / 'configs' / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    failures = 0
    if not a.skip_training:
        todo = train_runs
        if a.resume:
            todo = [r for r in train_runs if not ckpt_exists(r.name, r.epochs)]
            log(f'[resume] skipping {len(train_runs) - len(todo)} already-trained runs')
        failures += sum(1 for _, rc in run_jobs(todo, a, gpus) if rc != 0)
    if not a.skip_eval:
        failures += sum(1 for _, rc in run_jobs(eval_runs, a, gpus) if rc != 0)

    rows = collect_metrics(eval_runs, a)
    out_csv = Path(a.out_dir) / 'metrics.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        fields = ['dataset', 'variant', 'method_alpha', 'nu', 'G', 'M', 'eval_mode', 'nfe', 'epochs',
                  'wass', 'f_1_pr', 'log_path']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f'[done] failures={failures}; metrics -> {out_csv}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
