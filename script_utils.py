import argparse

# These parameters should be changed for this specific run, before objects are loaded
def update_parameters_before_loading(p, args):
    
    if args.method is not None:
        p['method'] = args.method
    
    method = p['method']
    # if alpha is specified, change the parameter, etc.
    if args.alpha is not None:
        p[method]['alpha'] = args.alpha
    
    if args.epochs is not None:
        p['run']['epochs'] = args.epochs
    
    if args.eval is not None:
        p['run']['eval_freq'] = args.eval

    if args.check is not None:
        p['run']['checkpoint_freq'] = args.check

    if args.check_start is not None:
        p['run']['checkpoint_start'] = args.check_start
    
    if args.train_reverse_steps is not None:
        p[method]['reverse_steps'] = args.train_reverse_steps

    if args.non_iso:
        p[method]['isotropic'] = False 

    if args.non_iso_data:
        p['data']['isotropic'] = False

    if args.data_std is not None:
        p['data']['std'] = args.data_std

    if args.data_alpha is not None:
        p['data']['data_alpha'] = args.data_alpha
    
    if args.set_seed is not None:
        p['seed'] = args.set_seed

    if args.random_seed is not None:
        p['seed'] = None

    if args.lploss is not None:
        assert method in ('dlpm', 'fm'), 'lploss is only for dlpm and fm'
        p['training'][method]['lploss'] = args.lploss

    if args.variance:
        assert method != 'fm', 'variance learning is not supported for fm'
        p[method]['var_predict'] = 'GAMMA'
        p['training'][method]['loss_type'] = 'VAR_KL'
        p['model']['learn_variance'] = True

    if args.median is not None and args.loss_monte_carlo is not None:
        raise ValueError('--median and --loss_monte_carlo are mutually exclusive. Use one or the other.')

    if args.median is not None:
        assert len(args.median) == 2
        p['training'][method]['loss_monte_carlo'] = 'median'
        p['training'][method]['monte_carlo_outer'] = int(args.median[0])
        p['training'][method]['monte_carlo_inner'] = int(args.median[1])

    if args.loss_monte_carlo is not None:
        p['training'][method]['loss_monte_carlo'] = args.loss_monte_carlo
    if args.monte_carlo_outer is not None:
        p['training'][method]['monte_carlo_outer'] = args.monte_carlo_outer
    if args.robust_exponent is not None:
        p['training'][method]['robust_exponent'] = args.robust_exponent
    
    if args.deterministic:
        p['eval'][method]['deterministic'] = True
        if method == 'dlpm':
            p['eval'][method]['dlim_eta'] = 0.0
        # fm is always deterministic (ODE-based), no action needed

    if args.clip:
        p['eval'][method]['clip_denoised'] = True

    if args.generate is not None:
        #assert False, 'NYI. eval_files are stored in some folder, and the prdc and fid functions consider all the files in a folder. So if a previous run had generated more data, there is a contamination. To be fixed'
        p['eval']['data_to_generate'] = args.generate
        assert args.generate <= p['eval']['real_data'], 'Must have more real data stored that number of data points to generate'
    
    # will do the neceassary changes after loading
    if args.lr is not None:
        p['optim']['lr'] = args.lr
    
    if args.lr_steps is not None:
        p['optim']['lr_steps'] = args.lr_steps
    
    if args.lr_schedule is not None:
        if args.lr_schedule == 'None':
            p['optim']['schedule'] = None
        else:
            p['optim']['schedule'] = args.lr_schedule

    if args.reverse_steps is not None:
        p['eval'][method]['reverse_steps'] = args.reverse_steps

    if getattr(args, 'solver', None) is not None:
        p['eval'][method]['solver'] = args.solver

    if getattr(args, 'solver_rtol', None) is not None or getattr(args, 'solver_atol', None) is not None:
        sk = dict(p['eval'][method].get('solver_kwargs') or {})
        if args.solver_rtol is not None:
            sk['rtol'] = args.solver_rtol
        if args.solver_atol is not None:
            sk['atol'] = args.solver_atol
        p['eval'][method]['solver_kwargs'] = sk

    if getattr(args, 'n_velocity_paths', None) is not None:
        p['eval'][method]['n_velocity_paths'] = args.n_velocity_paths
    if getattr(args, 'x1_strategy', None) is not None:
        p['eval'][method]['x1_strategy'] = args.x1_strategy

    # FM clock overrides (path-sampling G/M for alpha_stable; nu for student_t)
    if args.group_size is not None or args.resample_interval is not None:
        clock_cfg = p.setdefault('fm', {}).setdefault('clock', {})
        ps_cfg = clock_cfg.setdefault('path_sampling', {})
        if args.group_size is not None:
            ps_cfg['group_size'] = args.group_size
        if args.resample_interval is not None:
            ps_cfg['resample_interval'] = args.resample_interval

    if args.nu is not None:
        clock_cfg = p.setdefault('fm', {}).setdefault('clock', {})
        clock_cfg['nu'] = args.nu

    if getattr(args, 'clamp_V_max', None) is not None:
        clock_cfg = p.setdefault('fm', {}).setdefault('clock', {})
        clock_cfg['clamp_V_max'] = args.clamp_V_max
    if getattr(args, 'clamp_V_min', None) is not None:
        clock_cfg = p.setdefault('fm', {}).setdefault('clock', {})
        clock_cfg['clamp_V_min'] = args.clamp_V_min

    if args.dataset is not None:
        p['data']['dataset'] = args.dataset
    
    if args.nsamples is not None:
        p['data']['nsamples'] = args.nsamples
    
    # model
    if args.blocks is not None:
        p['model']['nblocks'] = args.blocks
    
    if args.units is not None:
        p['model']['nunits'] = args.units

    if args.t_embedding_type is not None:
        p['model']['time_emb_type'] = args.t_embedding_type

    if args.t_embedding_size is not None:
        p['model']['time_emb_size'] = args.t_embedding_size

    if args.scale is not None:
        p[method]['scale'] = args.scale

    if args.input_scaling:
        p[method]['input_scaling'] = True

    return p


# change some parameters for the run.
# These parameters should act on the objects already loaded from the previous runs
def update_experiment_after_loading(exp, args):
    # scheduler
    schedule_reset = False 
    if args.lr is not None:
        schedule_reset = True
        for optim in exp.manager.optimizers.values():
            for param_group in optim.param_groups:
                param_group['lr'] = args.lr
            exp.p['optim']['lr'] = args.lr
    if args.lr_steps is not None:
        schedule_reset = True
        exp.p['optim']['lr_steps'] = args.lr_steps
    if schedule_reset:
        for k, ls in exp.manager.learning_schedules.items():
            ls = exp.utils.init_default_ls(exp.p, exp.manager.optimizers[k])
            exp.manager.learning_schedules[k] = ls # redundant?



# some additional logging 
def additional_logging(exp, args):
    # logging job id
    if (exp.manager.logger is not None) and (args.job_id is not None):
        exp.manager.logger.log('job_id', args.job_id)
    
    # logging hash parameter
    if (exp.manager.logger is not None):
        exp.manager.logger.log('hash_parameter', exp.utils.exp_hash(exp.p))
    
    # logging hash eval
    if (exp.manager.logger is not None):
        exp.manager.logger.log('hash_eval', exp.utils.eval_hash(exp.p))
    
    # starting epoch and batch
    if (exp.manager.logger is not None):
        exp.manager.logger.log('starting_epoch', exp.manager.epochs)
        exp.manager.logger.log('starting_batch', exp.manager.total_steps)


# define and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # processes to choose from. Either diffusion, pdmp, or 'nf' to use a normal normalizing flow.
    parser.add_argument("--method", help='generative method to use', default=None, type=str)

    # EXPERIMENT parameters, specific to TRAINING
    parser.add_argument("--config", help='config file to use', type=str, required=True)
    parser.add_argument("--name", help='name of the experiment. Defines save location: ./models/name/', type=str, required=True)
    parser.add_argument('--epochs', help='epochs', default=None, type = int)
    parser.add_argument('-r', "--resume", help="resume existing experiment", action='store_true', default=False)
    parser.add_argument('--resume_epoch', help='epoch from which to resume', default = None, type=int)
    parser.add_argument('--eval', help='evaluation frequency', default=None, type = int)
    parser.add_argument('--check', help='checkpoint frequency', default=None, type = int)
    parser.add_argument('--check_start', help='only save checkpoints once epoch >= this value (default: from epoch 0)', default=None, type=int)
    parser.add_argument('--n_max_batch', help='max batch per epoch (to speed up testing)', default=None, type = int)
    parser.add_argument('--train_reverse_steps', help='number of diffusion steps used for training', default=None, type = int)

    parser.add_argument('--set_seed', help='set random seed', default = None, type=int)
    parser.add_argument('--random_seed', help='set random seed to a random number', action = 'store_true', default=None)

    parser.add_argument('--log', help='enable wandb logging', action='store_true', default=False)
    parser.add_argument('--wandb_project', help='wandb project name', default='dlpm', type=str)
    parser.add_argument('--wandb_name', help='wandb run name (defaults to --name)', default=None, type=str)
    parser.add_argument('--wandb_entity', help='wandb entity (team or username)', default=None, type=str)
    parser.add_argument('--job_id', help='slurm job id', default=None, type = str)

    # EXPERIMENT parameters, specific to EVALUATION
    parser.add_argument('--ema_eval', help='evaluate all ema models', action='store_true', default = False)
    parser.add_argument('--no_ema_eval', help='dont evaluate ema models', action='store_true', default = False)
    parser.add_argument('--generate', help='how many images/datapoints to generate', default = None, type = int)
    parser.add_argument('--reverse_steps', help='choose number of reverse_steps', default = None, type = int)
    parser.add_argument('--solver', help='FM ODE solver: euler|heun|midpoint|rk4|ab2|dopri5', default = None, type = str)
    parser.add_argument('--n_velocity_paths', help='inference-time velocity averaging K (default 1 = off)', default=None, type=int)
    parser.add_argument('--x1_strategy', help='x1 sampling when K>1', choices=['direct_alpha', 'k_mean'], default=None, type=str)
    parser.add_argument('--solver_rtol', help='adaptive solver rtol (dopri5)', default = None, type = float)
    parser.add_argument('--solver_atol', help='adaptive solver atol (dopri5)', default = None, type = float)
    parser.add_argument('--reset_eval', help='reset evaluation metrics', action='store_true', default = False)

    parser.add_argument('--deterministic', help='use deterministic sampling', default = False, action='store_true')
    parser.add_argument('--clip', help='use clip denoised (diffusion)', default = False, action='store_true')

    # DATA
    parser.add_argument('--dataset', help='choose specific dataset', default = None, type = str)
    parser.add_argument('--nsamples', help='choose the size of the dataset (only 2d datasets)', default = None, type = str)
    parser.add_argument('--data_std', help='choose data std or scale', default = None, type = float)
    parser.add_argument('--data_alpha', help='override data alpha for sas distributions', default = None, type = float)

    # OPTIMIZER
    parser.add_argument('--lr', help='reinitialize learning rate', type=float, default = None)
    parser.add_argument('--lr_steps', help='reinitialize learning rate steps', type=int, default = None)
    parser.add_argument('--lr_schedule', help='set learning rate schedule', type=str, default = None)

    # MODEL
    # only useful for 2d datasets
    parser.add_argument('--blocks', help='choose number of blocks in mlp', default = None, type = int)
    parser.add_argument('--units', help='choose number of units in mlp', default = None, type = int)
    parser.add_argument('--transforms', help='choose number of transforms in neural spline flow', default = None, type = int)
    parser.add_argument('--depth', help='choose depth in neural spline flow', default = None, type = int)
    parser.add_argument('--width', help='choose width in neural spline flow', default = None, type = int)
    parser.add_argument('--t_embedding_type', help='choose time embedding type', default = None, type = str)
    parser.add_argument('--t_embedding_size', help='choose time embedding size', default = None, type = int)

    # DIFFUSION
    parser.add_argument('--alpha', help='alpha value for diffusion', default=None, type = float)
    parser.add_argument('--non_iso', help='use non isotropic noise in the diffusion', action='store_true', default = False)
    parser.add_argument('--non_iso_data', help='use non isotropic data', action='store_true', default = False)
    parser.add_argument('--median', help='use median of mean. Specify (outer, inner).', nargs ='+', default = None)
    parser.add_argument('--loss_monte_carlo', help='loss aggregation mode', choices=['mean', 'median', 'robust'], default=None, type=str)
    parser.add_argument('--monte_carlo_outer', help='number of outer MC samples for clock A', default=None, type=int)
    parser.add_argument('--robust_exponent', help='exponent for robust loss aggregation', default=None, type=float)
    parser.add_argument('--lploss', help='set p in lploss. p=1: L1, p=2 L2, p=-1 squared L2', default = None, type = float)
    parser.add_argument('--variance', help='learn variance', default = False, action='store_true')
    parser.add_argument('--scale', help='choose scale exploding or scale preserving process', default = None, type=str)
    parser.add_argument('--input_scaling', help='input scaling', default = None, action='store_true')

    # FM clock (path-conditioning / random-clock variants)
    parser.add_argument('--group_size', help='FM clock path_sampling group_size G', default=None, type=int)
    parser.add_argument('--resample_interval', help='FM clock path_sampling resample_interval M', default=None, type=int)
    parser.add_argument('--nu', help='Student-t clock degrees of freedom (when fm.clock.type=student_t)', default=None, type=float)
    parser.add_argument('--clamp_V_max', help='upper clamp on V_nu draws (Student-t clock training stabilizer)', default=None, type=float)
    parser.add_argument('--clamp_V_min', help='lower clamp on V_nu draws (Student-t clock training stabilizer)', default=None, type=float)

    # PARSE AND RETURN
    args = parser.parse_args()
    assert (args.no_ema_eval and args.ema_eval) == False, 'No possible evaluation to make'
    return args