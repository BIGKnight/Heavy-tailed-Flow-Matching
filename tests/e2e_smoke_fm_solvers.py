"""E2E smoke test: load real FM ckpt, sample 4 images with every solver, diff vs euler.

Run:
    cd /disk3/yezhen/Experiments/Heavy-tailed-flow-matching
    CUDA_VISIBLE_DEVICES=3 python -m htfm.methods.fm_solvers.tests.e2e_smoke
"""
import os
import sys
import torch

sys.path.insert(0, os.getcwd())
from bem.utils_exp import FileHandler
import htfm.experiment as dlpm_exp
from htfm.methods.fm_solvers import available_solvers


def main():
    cfg_root = 'htfm/configs/'
    cfg_name = 'cifar10_lt_fp3/FP3_01_fm_a15_no_pc.yml'
    ckpt_dir = 'models/fp3_01_fm_a15_no_pc'
    ckpt_epoch = 500
    # hash-matching: fp3_01 training used fm.reverse_steps=4000, current yml says 200.
    # override here to reproduce the training hash (same reason as fp3_01 eval scripts).

    p = FileHandler.get_param_from_config(cfg_root, cfg_name)
    p['fm']['reverse_steps'] = 4000  # restore training-time value
    p['eval']['fm']['clip_denoised'] = False
    p['eval']['fm']['reverse_steps'] = 5  # few steps; we're just validating code paths
    p['data_to_generate'] = 4

    # build models + method
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p['device'] = device
    models, _, _ = dlpm_exp.init_models_by_parameter(p)
    method = dlpm_exp.init_method_by_parameter(p)
    # load ckpt
    model_hash = '663182f8495ad7e5'  # fp3_01 training hash
    ckpt_path = f'{ckpt_dir}/cifar10_lt/model_{model_hash}_{ckpt_epoch}.pt'
    print(f'loading {ckpt_path}')
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    models['default'].load_state_dict(sd['model_parameters'])
    models['default'].to(device).eval()
    # use EMA weights (what eval.py uses with --ema_eval)
    if isinstance(sd.get('ema_models'), list) and sd['ema_models']:
        ema_sd = sd['ema_models'][0]  # first EMA rate
        if hasattr(ema_sd, 'state_dict'):
            ema_sd = ema_sd.state_dict()
        if isinstance(ema_sd, dict) and 'shadow_params' in ema_sd:
            # torch-ema style: copy shadow_params into model
            with torch.no_grad():
                for p, s in zip(models['default'].parameters(), ema_sd['shadow_params']):
                    p.copy_(s.to(device))
            print('  loaded EMA shadow weights into default model')

    shape = [4, 3, 32, 32]
    solvers = available_solvers()
    print(f'solvers to test: {solvers}')

    results = {}
    for sname in solvers:
        torch.manual_seed(42)
        kw = dict(p['eval']['fm'])
        kw['solver'] = sname
        kw.pop('deterministic', None)  # fm.sample() doesn't use this anymore
        if sname == 'dopri5':
            kw['solver_kwargs'] = dict(rtol=1e-4, atol=1e-4)
        print(f'\n--- sampling with {sname} ---')
        try:
            with torch.no_grad():
                x = method.sample(models=models, shape=shape, **kw)
            stats = dict(mean=x.mean().item(), std=x.std().item(),
                         min=x.min().item(), max=x.max().item(),
                         nan=torch.isnan(x).any().item())
            print(f'  stats: {stats}')
            results[sname] = x.detach().cpu()
        except Exception as e:
            print(f'  FAILED: {e}')
            raise

    # Diff each solver vs euler on identical init
    euler_x = results['euler']
    print('\n--- L2 diff vs euler on rs=5 (expected: >0 for nonzero-order diff, all finite) ---')
    for sname, x in results.items():
        if sname == 'euler':
            continue
        diff = (x - euler_x).pow(2).mean().sqrt().item()
        print(f'  {sname:10s}: rms diff = {diff:.4e}')

    print('\nE2E SMOKE PASSED')


if __name__ == '__main__':
    main()
