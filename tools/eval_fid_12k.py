#!/usr/bin/env python3
"""12K-vs-12K FID on CIFAR10-LT for every method in this repo (HTFM / DLPM / DLIM / LIM / GMFlow).

Protocol (the one behind every CIFAR10-LT number in the paper):
  * generate ``--n_gen`` = 12,406 images from the **EMA** weights of checkpoint ``--epoch``;
  * reference set = the full long-tailed training split (12,406 images), dumped once to
    ``eval_files/cifar10_lt/original_data_n12406/`` and reused by every call;
  * Inception FID + precision / recall / density / coverage (prdc) from ``bem.evaluate``.

Sampler per method (``--method`` defaults to the config's ``method:``):
  fm       deterministic ODE, ``--solver`` (euler | heun | midpoint | rk4 | ab2 | dopri5),
           ``--reverse_steps`` = number of solver steps (= NFE for euler / ab2)
  dlpm     ``--sampler stochastic`` (default; the DLPM row) or ``deterministic`` (= DLIM, eta = 0)
  lim      ``--sampler stochastic`` (default; reverse SDE) or ``deterministic``
  gmflow   ``--sampler stochastic`` (default) or ``deterministic``

Examples:
    python tools/eval_fid_12k.py --config cifar10lt/FP13_a17_lev1 --name FP13_a17_lev1_s42 \
        --epoch 700 --reverse_steps 20 50 100 --gpu 0
    python tools/eval_fid_12k.py --config cifar10lt/FP13_a17_lev1 --name BASE_dlpm_a17_s42 \
        --method dlpm --epoch 700 --reverse_steps 100                     # DLPM baseline
    python tools/eval_fid_12k.py --config cifar10lt/FP12_dlpm_a2 --name BASE_dlpm_a2_s42 \
        --epoch 1000 --reverse_steps 100 --sampler deterministic          # DLIM
    python tools/eval_fid_12k.py --config cifar10lt_ablation/FP8_fm_a17_la025_lbn025_lev1 \
        --name FP8_a17_lbn025_lev1 --epoch 700 --solver heun --reverse_steps 10 25 50   # solver ablation

One JSON per (epoch, reverse_steps) cell. Default path:
    results/fid/<name>_ep<E>_rs<R>[_<solver>][_dlim|_det].json
Cells whose JSON already holds a FID are skipped unless ``--force``.

This script replaces the per-experiment evaluators of the internal repo
(eval_fp10_12k / eval_baseline_12k / eval_gmflow_12k / eval_fp12_dlpm / eval_table7).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_ROOT = 'htfm/configs/'
METRIC_KEYS = ['fid', 'precision', 'recall', 'density', 'coverage', 'f_1_pr', 'f_1_dc']
# function evaluations per solver step (fm only); None = adaptive
NFE_PER_STEP = {'euler': 1, 'heun': 2, 'midpoint': 2, 'ab2': 1, 'rk4': 4, 'dopri5': None}


# --------------------------------------------------------------------------------------
# Path helpers (import-safe: no torch here, so orchestrators can reuse them cheaply)
# --------------------------------------------------------------------------------------
def cell_suffix(method, solver, sampler):
    """Filename suffix that disambiguates non-default samplers."""
    if method == 'fm':
        return '' if solver == 'euler' else f'_{solver}'
    if sampler == 'deterministic':
        return '_dlim' if method == 'dlpm' else '_det'
    return ''


def default_out_path(name, epoch, reverse_steps, method='fm', solver='euler',
                     sampler='stochastic', out_dir='results/fid'):
    sfx = cell_suffix(method, solver, sampler)
    return os.path.join(out_dir, f'{name}_ep{epoch}_rs{reverse_steps}{sfx}.json')


def cached_fid(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get('fid')
    except Exception:
        return None


# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='config path relative to htfm/configs, without .yml')
    ap.add_argument('--name', required=True, help='run name: checkpoints are read from models/<name>/')
    ap.add_argument('--epoch', type=int, nargs='+', required=True, help='checkpoint epoch(s)')
    ap.add_argument('--reverse_steps', type=int, nargs='+', required=True, help='sampling steps (one or more)')
    ap.add_argument('--method', default=None, choices=['fm', 'dlpm', 'lim', 'gmflow'],
                    help='override the config method (e.g. evaluate a DLPM baseline trained with --method dlpm)')
    ap.add_argument('--solver', default='euler', help='fm only: ODE solver (default euler)')
    ap.add_argument('--sampler', default='stochastic', choices=['stochastic', 'deterministic', 'config'],
                    help='dlpm/lim/gmflow only: stochastic (default), deterministic (DLIM eta=0 for dlpm), '
                         'or keep the config value')
    ap.add_argument('--n_gen', type=int, default=12406, help='generated and reference sample count')
    ap.add_argument('--batch_size', type=int, default=500, help='generation batch size')
    ap.add_argument('--seed', type=int, default=None, help='torch / numpy seed before sampling')
    ap.add_argument('--no_ema', action='store_true', help='evaluate the raw (non-EMA) weights')
    ap.add_argument('--real_dir', default=None,
                    help='existing folder with exactly n_gen reference PNGs (default: built once under eval_files/)')
    ap.add_argument('--out', default=None, help='JSON path (single cell only)')
    ap.add_argument('--out_dir', default='results/fid', help='folder for the default JSON names')
    ap.add_argument('--no_detect_hash', action='store_true',
                    help='do not read the experiment hash from the checkpoint filename')
    ap.add_argument('--force', action='store_true', help='recompute cells whose JSON already exists')
    ap.add_argument('--gpu', type=int, default=None, help='sets CUDA_VISIBLE_DEVICES')
    args = ap.parse_args()
    if args.out and (len(args.epoch) > 1 or len(args.reverse_steps) > 1):
        ap.error('--out is only valid for a single (epoch, reverse_steps) cell')
    if args.solver not in NFE_PER_STEP:
        ap.error(f'unknown solver {args.solver}; choose from {sorted(NFE_PER_STEP)}')
    return args


def install_real_dir_patch(real_dir):
    """Route the reference set to a per-count folder instead of bem's shared original_data/.

    Mirrors bem.utils_exp.FileHandler._prepare_data_directories for image datasets, except that
    the reference folder is eval_files/<dataset>/original_data_n<N>/ (or --real_dir). It is
    (re)built from the first N training images whenever it is missing or holds a different count,
    exactly as bem does for its own folder.
    """
    import bem.utils_exp as _ue
    import torchvision.utils as tvu
    from bem.datasets import inverse_affine_transform

    def _prepare(self, dataset_name, dataset_files, remove_existing_eval_files, num_real_data, hash_params):
        folder = Path('eval_files') / dataset_name
        gen_path = folder / 'generated_data' / hash_params
        real_path = Path(real_dir) if real_dir else folder / f'original_data_n{num_real_data}'
        if gen_path.exists():
            if remove_existing_eval_files:
                for f in gen_path.iterdir():
                    f.unlink()
        else:
            gen_path.mkdir(parents=True, exist_ok=True)

        n_have = len([f for f in real_path.iterdir() if f.is_file()]) if real_path.is_dir() else 0
        if n_have != num_real_data:
            if real_dir:
                raise RuntimeError(f'--real_dir {real_path} holds {n_have} files, expected {num_real_data}')
            if len(dataset_files) < num_real_data:
                raise RuntimeError(f'dataset has {len(dataset_files)} images < n_gen={num_real_data}')
            print(f'[eval_fid] building reference set: {num_real_data} images -> {real_path}', flush=True)
            if real_path.exists():
                for f in real_path.iterdir():
                    f.unlink()
            real_path.mkdir(parents=True, exist_ok=True)
            for i in range(num_real_data):
                tvu.save_image(inverse_affine_transform(dataset_files[i][0]), real_path / f'{i}.png')
        else:
            print(f'[eval_fid] reference set: {real_path} ({n_have} images)', flush=True)
        return str(gen_path), str(real_path)

    _ue.FileHandler._prepare_data_directories = _prepare


def detect_ckpt_hash(name, epoch):
    """Experiment hash embedded in models/<name>/*/model_<hash>_<epoch>.pt (None if ambiguous)."""
    hashes = set()
    for f in glob.glob(os.path.join('models', name, '*', f'model_*_{epoch}.pt')):
        m = re.match(r'model_([a-f0-9]+)_\d+\.pt$', os.path.basename(f))
        if m:
            hashes.add(m.group(1))
    return hashes.pop() if len(hashes) == 1 else None


def _last(evals, k):
    v = evals.get(k)
    if isinstance(v, list) and v and v[-1] is not None:
        return float(v[-1])
    return None


def evaluate_cell(args, epoch, rs):
    import torch
    import numpy as np
    import bem.Experiments as Exp
    import bem.utils_exp as _ue
    import htfm.experiment as htfm_exp
    from bem.utils_exp import FileHandler

    p = FileHandler.get_param_from_config(CONFIG_ROOT, args.config + '.yml')
    method = args.method or p['method']
    p['method'] = method
    p['eval']['data_to_generate'] = args.n_gen
    p['eval']['real_data'] = args.n_gen
    p['eval']['batch_size'] = args.batch_size
    ev = p['eval'][method]
    ev['reverse_steps'] = rs
    if method == 'fm':
        ev['solver'] = args.solver
        sampler = 'ode'
        deterministic = bool(ev.get('deterministic', True))
    else:
        if args.sampler == 'stochastic':
            ev['deterministic'] = False
        elif args.sampler == 'deterministic':
            ev['deterministic'] = True
            if method == 'dlpm':
                ev['dlim_eta'] = 0.0
        deterministic = bool(ev.get('deterministic', False))
        sampler = 'deterministic' if deterministic else 'stochastic'
        if method == 'dlpm' and not deterministic:
            ev['dlim_eta'] = ev.get('dlim_eta', 0.0)

    # isolate this cell's generated_data folder from concurrent evaluations
    os.environ['EVAL_DIR_SUFFIX'] = f'{args.name}_ep{epoch}_rs{rs}_{method}_{args.solver if method == "fm" else sampler}_n{args.n_gen}'

    if not args.no_detect_hash:
        h = detect_ckpt_hash(args.name, epoch)
        if h is not None:
            print(f'[eval_fid] using checkpoint hash {h} from models/{args.name}/', flush=True)
            _ue.FileHandler.get_exp_hash = lambda self, p, verbose=False, _h=h: _h

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    exp = Exp.Experiment(
        checkpoint_dir=os.path.join('models', args.name), p=p, logger=None,
        exp_hash=htfm_exp.exp_hash, eval_hash=None,
        init_method_by_parameter=htfm_exp.init_method_by_parameter,
        init_models_by_parameter=htfm_exp.init_models_by_parameter,
        reset_models=htfm_exp.reset_models,
    )
    exp.load()
    exp.manager.eval.reset(keep_losses=True, keep_evals=False)
    exp.load(epoch=epoch)
    emas = getattr(exp.manager, 'ema_objects', None)
    use_ema = (not args.no_ema) and bool(emas)
    if use_ema:
        for ema_obj in emas:
            ema_obj['eval'].reset(keep_losses=True, keep_evals=False)

    t0 = time.time()
    exp.manager.evaluate(evaluate_emas=use_ema)
    wall = time.time() - t0

    fid, metrics, ema_mu = None, {}, None
    if use_ema:
        for ema_obj in emas:                     # first EMA object with a FID
            evs = ema_obj['eval'].evals
            if _last(evs, 'fid') is not None:
                fid = _last(evs, 'fid')
                ema_mu = getattr(ema_obj['default'], 'mu', None)
                metrics = {k: _last(evs, k) for k in METRIC_KEYS}
                break
    if fid is None:
        evs = exp.manager.eval.evals
        fid = _last(evs, 'fid')
        metrics = {k: _last(evs, k) for k in METRIC_KEYS}
    exp.terminate()

    per_step = NFE_PER_STEP[args.solver] if method == 'fm' else 1
    return {
        'config': args.config, 'name': args.name, 'method': method,
        'sampler': sampler, 'solver': args.solver if method == 'fm' else None,
        'deterministic': deterministic,
        'epoch': epoch, 'reverse_steps': rs,
        'nfe_per_step': per_step, 'nfe_total': (rs * per_step) if per_step else None,
        'n_gen': args.n_gen, 'n_real': args.n_gen, 'batch_size': args.batch_size,
        'use_ema': use_ema, 'ema_mu': ema_mu, 'seed': args.seed,
        'fid': fid, 'metrics': metrics, 'wall_sec': wall,
    }


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.chdir(REPO)                      # models/, eval_files/, results/ are repo-relative
    sys.path.insert(0, str(REPO))

    # method for naming (config read without torch)
    import yaml
    with open(os.path.join(CONFIG_ROOT, args.config + '.yml')) as f:
        method = args.method or yaml.safe_load(f)['method']
    sampler_for_name = 'stochastic' if args.sampler == 'config' else args.sampler

    cells = [(e, r) for e in args.epoch for r in args.reverse_steps]
    todo = []
    for e, r in cells:
        out = args.out or default_out_path(args.name, e, r, method, args.solver, sampler_for_name, args.out_dir)
        c = cached_fid(out)
        if c is not None and not args.force:
            print(f'[skip] {out} fid={c}')
            continue
        todo.append((e, r, out))
    if not todo:
        return

    install_real_dir_patch(args.real_dir)
    for e, r, out in todo:
        rec = evaluate_cell(args, e, r)
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump(rec, f, indent=2)
        print(f'\n=== {args.name} {rec["method"]}/{rec["sampler"]} ep{e} rs{r} -> FID={rec["fid"]} '
              f'({rec["wall_sec"] / 60:.1f} min) ===\nwrote {out}', flush=True)


if __name__ == '__main__':
    main()
