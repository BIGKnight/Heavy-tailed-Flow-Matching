"""
Visualize 2D flow matching results: load a trained checkpoint and plot
original vs generated samples.
"""
import os
import torch
import matplotlib.pyplot as plt
import bem.Experiments as Exp
from bem.utils_exp import *
import htfm.experiment as dlpm_exp
from script_utils import *


def visualize(config_path):
    args = parse_args()
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')
    update_parameters_before_loading(p, args)

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

    # load checkpoint
    epoch = args.resume_epoch if args.resume_epoch is not None else args.epochs
    exp.load(epoch=epoch)
    update_experiment_after_loading(exp, args)

    method = p['method']
    eval_p = p['eval'][method]
    gen_manager = exp.manager.eval.gen_manager

    # generate samples
    print(f'Generating {p["eval"]["data_to_generate"]} samples with {eval_p["reverse_steps"]} steps...')
    gen_manager.generate(
        exp.manager.models,
        p['eval']['data_to_generate'],
        reverse_steps=eval_p['reverse_steps'],
        deterministic=eval_p.get('deterministic', True),
        clip_denoised=eval_p.get('clip_denoised', False),
        clamp_a=eval_p.get('clamp_a', None),
        clamp_eps=eval_p.get('clamp_eps', None),
    )

    # plot side by side for clarity
    # grid means are at (i,j) for i,j in range(n), so for n=3: 0..2
    # center the view on (1,1) with enough margin
    n_side = int(p['data'].get('n_mixture', 9) ** 0.5)
    fig_center = (n_side - 1) / 2.0  # 1.0 for 3x3
    fig_margin = (n_side - 1) / 2.0 + 0.5  # 1.5 for 3x3
    original_data = gen_manager.load_original_data(10000).squeeze(1)
    gen_data = gen_manager.samples.squeeze(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(original_data[:, 0], original_data[:, 1],
                s=1, alpha=0.3, color='darkorange')
    ax1.set_xlim(fig_center - fig_margin, fig_center + fig_margin)
    ax1.set_ylim(fig_center - fig_margin, fig_center + fig_margin)
    ax1.set_title(f'Original Data (9 Gaussians, 3x3 grid)')
    ax1.set_aspect('equal')

    ax2.scatter(gen_data[:10000, 0], gen_data[:10000, 1],
                s=1, alpha=0.3, color='royalblue')
    ax2.set_xlim(fig_center - fig_margin, fig_center + fig_margin)
    ax2.set_ylim(fig_center - fig_margin, fig_center + fig_margin)
    method_names = {'fm': 'Flow Matching', 'dlpm': 'DLPM', 'lim': 'LIM'}
    method_label = method_names.get(method, method.upper())
    ax2.set_title(f'Generated ({method_label}, α={p[method]["alpha"]}, epoch={epoch})')
    ax2.set_aspect('equal')

    fig.suptitle(f'{method_label} on GMM Grid', fontsize=14)
    fig.tight_layout()

    # save and show
    save_path = os.path.join(checkpoint_dir, 'visualization.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.show()


if __name__ == '__main__':
    config_path = 'htfm/configs/'
    visualize(config_path)
