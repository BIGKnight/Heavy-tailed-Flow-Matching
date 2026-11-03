from bem.datasets import get_dataset, is_image_dataset
from htfm.models import build_model
from bem.utils_exp import InitUtils
from torch.utils.data import DataLoader


# v15.2: YAML schema migration shim. Copies legacy keys at
# `path_conditioning.path_sampling` / `.path_pool` to the canonical
# location `fm.clock.path_sampling` / `.path_pool`, preserving full
# back-compat for in-tree YAMLs. Idempotent: a no-op once the canonical
# keys exist. Called once near the top of init paths in run.py / eval.py
# (or directly from init_method_by_parameter to keep call sites narrow).
def _shim_v15_path_keys(p):
    pc = p.get('path_conditioning', {}) or {}

    # v15.2: legacy path_conditioning.path_{sampling,pool} -> fm.clock.*
    if p.get('method') == 'fm':
        fm_cfg = p.setdefault('fm', {})
        clock_cfg = fm_cfg.setdefault('clock', {})
        moved = []
        for key in ('path_sampling', 'path_pool'):
            if key in pc and key not in clock_cfg:
                clock_cfg[key] = pc[key]
                moved.append(key)
        if moved:
            print(f'[v15-shim] migrated path_conditioning.{{{",".join(moved)}}}'
                  f' -> fm.clock.{{{",".join(moved)}}}', flush=True)

    # v15.6: new schema -> legacy keys (so existing readers keep working).
    # Direction: NEW (path_signature/projection/injection) -> LEGACY flat keys.
    # Idempotent (uses setdefault). One-way arrow; legacy keys stay around
    # until we flip readers in a later cleanup pass.
    if 'path_signature' in pc:
        sig = pc['path_signature']
        # Presence of path_signature block implies α-stable wants logsignature.
        pc.setdefault('path_feature_type', 'logsignature')
        if 'level' in sig:
            pc.setdefault('logsignature_level', sig['level'])
        if 'input_channels' in sig:
            pc.setdefault('path_feature_input_channels', sig['input_channels'])
        if 'normalization' in sig:
            pc.setdefault('path_feature_normalization', sig['normalization'])
        if 'backend' in sig:
            pc.setdefault('signature_backend', sig['backend'])
        if 'discretization_steps' in sig:
            pc.setdefault('path_discretization_steps', sig['discretization_steps'])
        if 'detach' in sig:
            pc.setdefault('path_feature_detach', sig['detach'])
    if 'projection' in pc:
        proj = pc['projection']
        if 'dim' in proj:
            pc.setdefault('proj_dim', proj['dim'])
        if 'hidden_dim' in proj:
            pc.setdefault('proj_hidden_dim', proj['hidden_dim'])
        if 'dropout' in proj:
            pc.setdefault('proj_dropout', proj['dropout'])
    if 'injection' in pc:
        inj = pc['injection']
        if 'global' in inj:
            pc.setdefault('global', inj['global'])
        if 'modulation' in inj:
            pc.setdefault('modulation', inj['modulation'])


# v15.3: derive `path_conditioning.enabled` from legacy keys, becoming
# the single on/off gate consumed by init_model_by_parameter and downstream.
# Idempotent: respects an explicitly-set value (allows new-schema YAMLs
# to opt in cleanly). Derivation matches the previous fragmented predicate
# exactly so byte-identity holds:
#   enabled = (global.enabled OR modulation.enabled) AND has_feature_source
# where has_feature_source means either α-stable+logsig encoder is set up,
# or any non-α-stable clock (currently student-t supplies its own feature).
def _derive_path_conditioning_enabled(p):
    pc = p.setdefault('path_conditioning', {})
    if 'enabled' in pc:
        return  # explicit user choice — leave alone
    use_global = pc.get('global', {}).get('enabled', False)
    use_mod = pc.get('modulation', {}).get('enabled', False)
    # Single source of truth for "does the clock produce a feature?".
    # alpha_stable: needs path_feature_type != 'none'.
    # student_t: always (V_nu intrinsic scalar).
    # gaussian: never (vanilla FM, no clock state).
    from htfm.methods.clocks import clock_has_feature_source
    pc['enabled'] = bool((use_global or use_mod) and clock_has_feature_source(p))


''''''''''' FILE MANIPULATION '''''''''''
def exp_hash(p):
    data_hash = {k:v for k, v in p['data'].items() if k in ['dataset', 'channels', 'image_size']}
    if 'long_tail' in p['data']:
        data_hash['long_tail'] = p['data']['long_tail']
    model_param = p['model']
    to_hash = {'data': data_hash,
            p['method']: {k:v for k, v in p[p['method']].items()},
            'model':  {k:v for k, v in model_param.items()}, 
            #'optim': p['optim'],
            #'training': p['training']
            }
    return to_hash
    

''' INITIALIZATION UTILS '''
# Model construction is delegated to htfm/models/__init__.py:build_model,
# which dispatches via MODEL_BUILDERS registry. Adding a new architecture
# = adding one entry to that dict + the model file.

def init_model_by_parameter(p):
    # v15.2/v15.3: ensure shim has run before reading path_conditioning.
    _shim_v15_path_keys(p)
    _derive_path_conditioning_enabled(p)

    model_type = p.get('model', {}).get('model_type', None)
    model = build_model(model_type, p)

    # v15.4: PCM construction lives in htfm.methods.path.conditioning;
    # init_model_by_parameter just wires the result onto the backbone.
    from htfm.methods.path import build_path_conditioning_module
    pcm = build_path_conditioning_module(p)
    if pcm is not None:
        # Only some backbones support path conditioning. Currently UNetModel
        # ('ddpm') has set_path_conditioning; lim_ddpm / lim_ncsnpp / mlp_fm
        # do not. Fail loud rather than AttributeError into user space.
        if not hasattr(model, 'set_path_conditioning'):
            raise NotImplementedError(
                f"model_type={p.get('model', {}).get('model_type')!r} does not "
                f"support path conditioning. Either set path_conditioning.enabled=false "
                f"or use model_type='ddpm'.")
        model.set_path_conditioning(pcm)

    return model.to(p['device'])


def init_data_by_parameter(p):
    # get the dataset
    dataset_files, test_dataset_files = get_dataset(p)
    assert dataset_files is not None
    assert test_dataset_files is not None
    
    # implement DDP later on
    data = DataLoader(dataset_files, 
                    batch_size=p['training']['batch_size'], 
                    shuffle=True, 
                    num_workers=p['training']['num_workers'])
    test_data = DataLoader(test_dataset_files,
                            batch_size=p['training']['batch_size'],
                            shuffle=True,
                            num_workers=p['training']['num_workers'])
    return data, test_data, dataset_files, test_dataset_files


def init_method_by_parameter(p):
    """Build method instance via the registry, then attach a path_encoder.

    Registry: htfm.methods.METHOD_BUILDERS (V12.2). Adding a new method =
    one file + one entry; init_method_by_parameter doesn't need editing.
    """
    # v15.2/v15.3: ensure shim + enabled-derivation have run before any
    # downstream code reads path_conditioning.* (idempotent; init_model_*
    # may have run them earlier).
    _shim_v15_path_keys(p)

    # gaussian clock has no clock state to condition on; fail loud BEFORE
    # _derive_path_conditioning_enabled silently demotes user intent to
    # enabled=False. Check the raw user intent (path_feature_type + the
    # injection.global/modulation flags + explicit pc.enabled) — any of
    # them being "on" while clock_type='gaussian' is a misconfig.
    fm_clock = (p.get('fm', {}) or {}).get('clock', {}) or {}
    if fm_clock.get('type', 'alpha_stable') == 'gaussian':
        pc_raw = p.get('path_conditioning', {}) or {}
        assert pc_raw.get('path_feature_type', 'none') == 'none', (
            "clock_type='gaussian' is incompatible with path conditioning: "
            f"path_feature_type={pc_raw.get('path_feature_type')!r} but the "
            "vanilla clock has no T-path / no V_nu and produces no feature.")
        assert not pc_raw.get('enabled', False), (
            "clock_type='gaussian' is incompatible with "
            "path_conditioning.enabled=true (no feature source).")
        assert not pc_raw.get('global', {}).get('enabled', False), (
            "clock_type='gaussian' is incompatible with "
            "path_conditioning.global.enabled=true (no feature to inject).")
        assert not pc_raw.get('modulation', {}).get('enabled', False), (
            "clock_type='gaussian' is incompatible with "
            "path_conditioning.modulation.enabled=true (no feature to inject).")
        # Legacy schema: same checks via the v15.6 → legacy mirror.
        # Already covered by path_feature_type assert above.

    _derive_path_conditioning_enabled(p)

    from htfm.methods import build_method
    method = build_method(p)

    pc = p.get('path_conditioning', {})

    if pc.get('path_feature_type', 'none') != 'none':
        from htfm.methods.path import PathFeatureEncoder
        path_encoder = PathFeatureEncoder(p)
        method.set_path_encoder(path_encoder)

    return method

def init_optimizer_by_parameter(p, model):
    return InitUtils.init_default_optimizer(p, model)

def init_ls_by_parameter(p, optim):
    return InitUtils.init_default_ls(p, optim)

def init_models_by_parameter(p):
    models = {
        'default': init_model_by_parameter(p)
        }
    optimizers = {
        'default': init_optimizer_by_parameter(p, models['default']),
        }
    learning_schedules = {
        'default': init_ls_by_parameter(p, optimizers['default'])
        }

    return models, optimizers, learning_schedules

def reset_models(p):
    return init_models_by_parameter(p)
