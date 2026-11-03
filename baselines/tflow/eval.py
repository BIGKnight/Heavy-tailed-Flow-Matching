"""Evaluate a trained t-Flow checkpoint via FID on CIFAR10-LT.

Usage:
    python eval.py --ckpt runs/.../ckpt_ep2000.pt --nu 1.7 --nfe 100
    python eval.py --ckpt ... --nu 3.0 --nfe 25 --solver heun
"""

from __future__ import annotations

import argparse

from tflow.eval_fid import compute_fid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--nu', type=float, required=True)
    p.add_argument('--nfe', type=int, default=100)
    p.add_argument('--n-samples', type=int, default=12406)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--cifar-root', default='./data/cifar10')
    p.add_argument('--cifar-split-dir', default='./data/splits')
    p.add_argument('--work-dir', default=None)
    p.add_argument('--solver', choices=['euler', 'heun'], default='euler',
                   help="'euler' for matched-NFE comparison with HTFM, 'heun'"
                        " to reproduce Pandey et al. App. B Algorithm 4.")
    p.add_argument('--param', choices=['noise', 'velocity'], default=None,
                   help='Network parameterization. Defaults to the value '
                        "stored in the checkpoint's training config.")
    args = p.parse_args()

    fid = compute_fid(
        args.ckpt,
        nu=args.nu,
        nfe=args.nfe,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        cifar_root=args.cifar_root,
        cifar_split_dir=args.cifar_split_dir,
        work_dir=args.work_dir,
        solver=args.solver,
        param=args.param,
    )
    print(f'FID(nu={args.nu}, NFE={args.nfe}, solver={args.solver}): '
          f'{fid:.4f}')


if __name__ == '__main__':
    main()
