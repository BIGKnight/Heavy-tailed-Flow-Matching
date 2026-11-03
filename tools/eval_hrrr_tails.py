"""
HRRR tail-metrics evaluator: load a trained checkpoint, draw N samples, and
score against the test set using Pandey et al.'s KR / SR / KS-2 protocol.

Bypasses the standard FID/MMD evaluator because those operate on 2D feature
vectors and our HRRR samples are (1, 128, 128) image tensors.

Example:
    python eval_files/hrrr/eval_tails.py \
        --config hrrr_st_unc/HRRR_VIL_nu3_no_pc \
        --name hrrr_vil_nu3 --method fm --epoch 100 \
        --generate 20000 --reverse_steps 25 --batch_size 64

Saves results to models/<name>/<dataset>/tail_metrics_ep<epoch>_n<N>.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Make repo modules importable when invoked directly.
# tools/eval_hrrr_tails.py → tools/ → repo root  (dirname × 2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bem.Experiments as Exp
from bem.utils_exp import FileHandler
from bem.evaluate.tail_metrics import hrrr_tail_metrics
import htfm.experiment as dlpm_exp


def _build_experiment(args):
    config_path = './htfm/configs/'
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')
    p['method'] = args.method
    if args.eval_batch_size is not None:
        p['eval']['batch_size'] = args.eval_batch_size
    if args.reverse_steps is not None:
        # Only override the EVAL-block reverse_steps. The method-level reverse_steps is a
        # training/model param (diffusion T for dlpm, SDE discretization for lim) and is part
        # of exp_hash — overriding it makes the checkpoint lookup miss. The actual sampling
        # steps are passed via eval_kwargs in _generate_samples, so the eval block is enough.
        p['eval'][args.method]['reverse_steps'] = args.reverse_steps

    checkpoint_dir = os.path.join('models', args.name)
    exp = Exp.Experiment(
        checkpoint_dir=checkpoint_dir,
        p=p,
        logger=None,
        exp_hash=dlpm_exp.exp_hash,
        eval_hash=None,
        init_method_by_parameter=dlpm_exp.init_method_by_parameter,
        init_models_by_parameter=dlpm_exp.init_models_by_parameter,
        reset_models=dlpm_exp.reset_models,
    )
    exp.load(epoch=args.epoch)
    return exp


def _generate_samples(exp, n_samples, batch_size, reverse_steps,
                      use_ema=True, denormalize=False):
    p = exp.utils.p
    method = p['method']
    image_size = int(p['data']['image_size'])
    channels = int(p['data']['channels'])
    shape_per = (channels, image_size, image_size)

    method_obj = exp.manager.method
    ema_objects = getattr(exp.manager, 'ema_objects', None)
    if use_ema and ema_objects:
        ema_dict = ema_objects[0]   # use first EMA rate
        models = {k: ema_dict[k].get_ema_model() for k in exp.manager.models}
    else:
        models = exp.manager.models
    for m in models.values():
        m.eval()

    eval_kwargs = dict(p['eval'][method])
    eval_kwargs['reverse_steps'] = reverse_steps
    for k in ('batch_size', 'data_to_generate'):
        eval_kwargs.pop(k, None)

    out_chunks = []
    drawn = 0
    print(f'[eval_tails] generating {n_samples} samples '
          f'(batch={batch_size}, reverse_steps={reverse_steps}, ema={use_ema})',
          flush=True)
    t0 = time.time()
    while drawn < n_samples:
        bs = min(batch_size, n_samples - drawn)
        shape = (bs,) + shape_per
        x = method_obj.sample(models, shape, **eval_kwargs)
        out_chunks.append(x.detach().cpu().numpy().astype(np.float32))
        drawn += bs
        rate = drawn / max(time.time() - t0, 1e-9)
        eta = (n_samples - drawn) / max(rate, 1e-9)
        if drawn == bs or drawn % (batch_size * 4) == 0 or drawn == n_samples:
            print(f'[eval_tails] {drawn}/{n_samples} rate={rate:.2f}/s '
                  f'eta={eta/60:.1f}min', flush=True)

    samples = np.concatenate(out_chunks, axis=0)
    if denormalize:
        train_ds = exp.data
        mean = float(getattr(train_ds, 'norm_mean', 0.0))
        std = float(getattr(train_ds, 'norm_std', 1.0))
        samples = samples * std + mean
    return samples


def _load_test_pool(exp, denormalize=False):
    """exp.test_data is a torch DataLoader; iterate it to collect all frames."""
    loader = exp.test_data
    chunks = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        chunks.append(x.detach().cpu().numpy())
    arr = np.concatenate(chunks, axis=0).astype(np.float32)
    if denormalize:
        ds = getattr(loader, 'dataset', None)
        mean = float(getattr(ds, 'norm_mean', 0.0))
        std = float(getattr(ds, 'norm_std', 1.0))
        arr = arr * std + mean
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='YAML config path under htfm/configs/')
    p.add_argument('--name', required=True, help='experiment name (must match training run)')
    p.add_argument('--method', default='fm')
    p.add_argument('--epoch', type=int, default=None,
                   help='checkpoint epoch to load (default: latest)')
    p.add_argument('--generate', type=int, default=20000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--reverse_steps', type=int, default=25)
    p.add_argument('--eval_batch_size', type=int, default=None)
    p.add_argument('--channel', default=None)
    p.add_argument('--q', type=float, default=0.999)
    p.add_argument('--denormalize', action='store_true')
    p.add_argument('--no_ema', action='store_true')
    p.add_argument('--save_samples', action='store_true')
    p.add_argument('--seed', type=int, default=None,
                   help='seed torch/cuda/numpy RNG (for reproducible / shardable generation)')
    p.add_argument('--out_tag', default='',
                   help='suffix inserted into the output filename (isolates parallel shards)')
    args = p.parse_args()

    exp = _build_experiment(args)

    if args.channel is None:
        ds_name = exp.utils.p['data']['dataset'].lower()
        args.channel = 'w20' if 'w20' in ds_name else 'vil'
    print(f'[eval_tails] channel={args.channel} q={args.q} '
          f'denormalize={args.denormalize}', flush=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f'[eval_tails] seeded RNG with {args.seed}', flush=True)

    samples = _generate_samples(
        exp, args.generate, args.batch_size, args.reverse_steps,
        use_ema=not args.no_ema, denormalize=args.denormalize)

    test_arr = _load_test_pool(exp, denormalize=args.denormalize)
    print(f'[eval_tails] test set: {test_arr.shape}; samples: {samples.shape}', flush=True)

    metrics = hrrr_tail_metrics(samples, test_arr, channel=args.channel, q=args.q)
    print('[eval_tails] === RESULTS ===')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.6f}')
        else:
            print(f'  {k}: {v}')

    out_dir = os.path.join('models', args.name, exp.utils.p['data']['dataset'])
    os.makedirs(out_dir, exist_ok=True)
    epoch_str = args.epoch if args.epoch is not None else 'latest'
    out_path = os.path.join(out_dir, f'tail_metrics_ep{epoch_str}_rs{args.reverse_steps}_n{args.generate}{args.out_tag}.json')
    record = {
        'config': args.config,
        'name': args.name,
        'epoch': args.epoch,
        'generate': args.generate,
        'reverse_steps': args.reverse_steps,
        'channel': args.channel,
        'q': args.q,
        'denormalize': args.denormalize,
        'use_ema': not args.no_ema,
        'metrics': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in metrics.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'[eval_tails] saved -> {out_path}')

    if args.save_samples:
        np.save(out_path.replace('.json', '_samples.npy'), samples)
        print(f'[eval_tails] saved samples')

    exp.terminate()


if __name__ == '__main__':
    main()
