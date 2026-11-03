"""Run a single (ckpt_ep, NFE) FID eval and append a row to a markdown file.

This is the worker script for the multi-GPU sweep launched by
``run_eval_sweep.sh``. It is fcntl-locked so multiple GPU workers can append
to the same markdown safely.
"""

import argparse
import datetime
import fcntl
import os
import sys

# Run from baselines/tflow so the `tflow` package imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tflow.eval_fid import compute_fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ep', type=int, required=True)
    ap.add_argument('--nu', type=float, default=3.0)
    ap.add_argument('--nfe', type=int, required=True)
    ap.add_argument('--n-samples', type=int, default=12406)
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--cifar-root', default='./data/cifar10')
    ap.add_argument('--cifar-split-dir', default='./data/splits')
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    started = datetime.datetime.now()
    fid = compute_fid(
        args.ckpt,
        nu=args.nu,
        nfe=args.nfe,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        cifar_root=args.cifar_root,
        cifar_split_dir=args.cifar_split_dir,
        work_dir=args.work_dir,
        solver='euler',
    )
    finished = datetime.datetime.now()

    line = (
        f'| {args.ep} | {args.nfe} | {fid:.4f} | '
        f'{started:%m-%d %H:%M} | {(finished - started).total_seconds() / 60:.1f} |\n'
    )
    with open(args.out_md, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)

    print(f'OK: ep={args.ep} nfe={args.nfe} fid={fid:.4f} '
          f'wall={(finished - started).total_seconds() / 60:.1f}min')


if __name__ == '__main__':
    main()
