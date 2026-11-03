"""Train a t-Flow model on CIFAR10-LT.

Usage:
    python train.py --config configs/cifar10lt_nu3p0.yml
    python train.py --config configs/cifar10lt_nu1p7.yml --out-dir runs/tflow_nu17
"""

from __future__ import annotations

import argparse
import dataclasses

import yaml

from tflow.trainer import TrainConfig, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config.')
    parser.add_argument('--out-dir', type=str, default=None, help='Override config out_dir.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Filter to known fields and build dataclass.
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
    if args.out_dir is not None:
        cfg_dict['out_dir'] = args.out_dir
    if args.epochs is not None:
        cfg_dict['epochs'] = args.epochs
    if args.seed is not None:
        cfg_dict['seed'] = args.seed

    cfg = TrainConfig(**cfg_dict)
    print('Training config:')
    for k, v in dataclasses.asdict(cfg).items():
        print(f'  {k}: {v}')
    train(cfg)


if __name__ == '__main__':
    main()
