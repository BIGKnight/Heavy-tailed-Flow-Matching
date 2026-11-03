"""Tests for GaussianClock (vanilla flow matching, no random clock).

Math:
    X_1 = G  with G ~ N(0, I)
    no path features, no T-path, no R_m

Compatible flow_types: straight_line (canonical), tbar_linear (degenerate).
Incompatible: ot_geodesic, signature_linear_ve.

Run:
    cd <repo root>
    python -m htfm.methods.tests.test_gaussian_clock
"""
import torch

from htfm.methods.clocks import build_clock, available_clocks
from htfm.methods.clocks.gaussian import GaussianClock


# ---- Registry & basic construction --------------------------------------

def test_in_registry():
    assert 'gaussian' in available_clocks()
    c = build_clock('gaussian', device='cpu')
    assert isinstance(c, GaussianClock)
    print('  test_in_registry: OK')


def test_capabilities():
    c = GaussianClock(device='cpu')
    assert c.supports_path_pool is False
    assert c.supports_path_sampling is False
    assert c.feature_dim == 0
    assert c.feature_dim_for_conditioning() == 0
    assert c.ckpt_metadata_kwargs == {}
    assert c.get_ckpt_extra() == {}
    print('  test_capabilities: OK')


# ---- Sampling identity X_1 = G ------------------------------------------

def test_sample_X1_returns_G_unchanged():
    """X_1 must equal the input G (no scaling)."""
    c = GaussianClock(device='cpu')
    B, D = 4, 3
    x_start = torch.randn(B, D)
    x0_ext = x_start.repeat(2, 1)  # mc_outer=2, mc_inner=1 -> total_mc=2
    G = torch.randn(B * 2, D)
    X1, path_features = c.sample_X1_and_features(x_start, x0_ext, G,
                                                  monte_carlo_inner=1,
                                                  monte_carlo_outer=2)
    assert X1.shape == G.shape
    assert torch.equal(X1, G), 'X1 must be exactly G (vanilla)'
    assert path_features is None
    print('  test_sample_X1_returns_G_unchanged: OK')


def test_sample_initial_returns_G():
    """sample_initial returns (G, None, G, None) where G ~ N(0, I)."""
    torch.manual_seed(0)
    c = GaussianClock(device='cpu')
    shape = [16, 3, 8, 8]
    x, A_sample, G, pc = c.sample_initial(shape)
    assert tuple(x.shape) == tuple(shape)
    assert torch.equal(x, G), 'inference x must equal G'
    assert A_sample is None
    assert pc is None
    # Sanity check: roughly mean=0, std=1
    assert abs(x.mean().item()) < 0.1
    assert abs(x.std().item() - 1.0) < 0.1
    print('  test_sample_initial_returns_G: OK')


# ---- tbar_linear degeneration ------------------------------------------

def test_tbar_linear_degenerates():
    """For Gaussian clock, bar_T_t = t and bar_T_dot_t = 1 exactly."""
    c = GaussianClock(device='cpu')
    t = torch.linspace(0.05, 0.95, 19)
    bar_T, bar_T_dot = c.compute_bar_T_at(t)
    assert torch.allclose(bar_T, t)
    assert torch.allclose(bar_T_dot, torch.ones_like(t))
    print('  test_tbar_linear_degenerates: OK')


def test_sample_t_continuous():
    c = GaussianClock(device='cpu')
    t = c.sample_t_for_tbar_linear(2000)
    assert (t > 0).all() and (t < 1).all()
    print(f'  test_sample_t_continuous: OK  (t range '
          f'[{t.min():.4f}, {t.max():.4f}])')


# ---- Disallowed APIs ----------------------------------------------------

def test_compute_R_m_raises():
    """signature_linear_ve schedule must NOT use this clock."""
    try:
        GaussianClock.compute_R_m(torch.zeros(1, 10))
    except NotImplementedError as e:
        assert 'alpha_stable' in str(e), f'unexpected error: {e}'
        print('  test_compute_R_m_raises: OK')
        return
    raise AssertionError('expected NotImplementedError')


# ---- FM construction matrix (which flow_type × clock combos are allowed?) -

def test_fm_gaussian_straight_line_runs():
    """Vanilla FM: straight_line + gaussian clock — full training step."""
    import torch.nn as nn
    from htfm.methods.flow_matching import FlowMatchingLevyProcess

    class TinyVel(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.lin = nn.Linear(D + 1, D)

        def forward(self, x, t, path_features=None):
            z = torch.cat([x, t.view(-1, 1)], dim=-1)
            return self.lin(z)

    D, B = 4, 5
    torch.manual_seed(0)
    model = TinyVel(D)
    x_start = torch.randn(B, D)
    fm = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        clock_type='gaussian',
    )
    out = fm.training_losses({'default': model}, x_start)
    assert torch.isfinite(out['loss']), f'non-finite loss: {out["loss"]}'
    print(f'  test_fm_gaussian_straight_line_runs: OK '
          f'(loss={out["loss"].item():.4f})')


def test_fm_gaussian_tbar_linear_runs():
    """tbar_linear + gaussian clock — degenerates to straight_line."""
    import torch.nn as nn
    from htfm.methods.flow_matching import FlowMatchingLevyProcess

    class TinyVel(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.lin = nn.Linear(D + 1, D)

        def forward(self, x, t, path_features=None):
            z = torch.cat([x, t.view(-1, 1)], dim=-1)
            return self.lin(z)

    D, B = 4, 5
    torch.manual_seed(0)
    model = TinyVel(D)
    x_start = torch.randn(B, D)
    fm = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='tbar_linear', noise_type='terminal',
        clock_type='gaussian',
    )
    out = fm.training_losses({'default': model}, x_start)
    assert torch.isfinite(out['loss']), f'non-finite loss: {out["loss"]}'
    print(f'  test_fm_gaussian_tbar_linear_runs: OK '
          f'(loss={out["loss"].item():.4f})')


def test_fm_gaussian_tbar_linear_equals_straight_line():
    """tbar_linear + gaussian must produce IDENTICAL loss to straight_line +
    gaussian (since bar_T_t = t and bar_T_dot = 1 exactly)."""
    import torch.nn as nn
    from htfm.methods.flow_matching import FlowMatchingLevyProcess

    class TinyVel(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.lin = nn.Linear(D + 1, D)

        def forward(self, x, t, path_features=None):
            z = torch.cat([x, t.view(-1, 1)], dim=-1)
            return self.lin(z)

    D, B = 4, 6
    torch.manual_seed(0)
    model = TinyVel(D)
    x_start = torch.randn(B, D)

    fm_sl = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        clock_type='gaussian',
    )
    fm_tbar = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='tbar_linear', noise_type='terminal',
        clock_type='gaussian',
    )
    torch.manual_seed(42)
    out_sl = fm_sl.training_losses({'default': model}, x_start)
    torch.manual_seed(42)
    out_tbar = fm_tbar.training_losses({'default': model}, x_start)
    diff = abs(out_sl['loss'].item() - out_tbar['loss'].item())
    assert diff < 1e-6, (
        f'gaussian: tbar_linear loss {out_tbar["loss"].item()} != '
        f'straight_line loss {out_sl["loss"].item()} (diff {diff})')
    print(f'  test_fm_gaussian_tbar_linear_equals_straight_line: OK '
          f'(diff={diff:.2e})')


def test_fm_gaussian_rejects_ot_geodesic():
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    try:
        FlowMatchingLevyProcess(
            alpha=1.7, device='cpu', reverse_steps=20,
            isotropic=True, scale='scale_preserving',
            flow_type='ot_geodesic', noise_type='terminal',
            clock_type='gaussian',
        )
    except AssertionError as e:
        assert 'gaussian' in str(e) and 'flow_type' in str(e)
        print('  test_fm_gaussian_rejects_ot_geodesic: OK')
        return
    raise AssertionError('expected AssertionError for gaussian + ot_geodesic')


def test_fm_gaussian_rejects_signature_linear_ve():
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    from htfm.methods.noise_schedule.signature_linear_ve import SignatureLinearVESchedule
    try:
        FlowMatchingLevyProcess(
            alpha=1.7, device='cpu', reverse_steps=20,
            isotropic=True, scale='scale_preserving',
            flow_type='signature_linear_ve', noise_type='forward_process',
            clock_type='gaussian',
            ve_schedule=SignatureLinearVESchedule(0.25, 0.25),
        )
    except AssertionError as e:
        # gaussian-flow_type assert fires first ('not in {straight_line,tbar_linear}'),
        # but signature_linear_ve's own assert (clock_type=='alpha_stable') would also catch.
        print(f'  test_fm_gaussian_rejects_signature_linear_ve: OK (msg: {str(e)[:80]}...)')
        return
    raise AssertionError('expected AssertionError for gaussian + signature_linear_ve')


def test_fm_gaussian_supports_epsilon_predictor():
    """Vanilla FM with epsilon prediction should work."""
    import torch.nn as nn
    from htfm.methods.flow_matching import FlowMatchingLevyProcess

    class TinyEps(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.lin = nn.Linear(D + 1, D)

        def forward(self, x, t, path_features=None):
            z = torch.cat([x, t.view(-1, 1)], dim=-1)
            return self.lin(z)

    D, B = 4, 5
    torch.manual_seed(0)
    model = TinyEps(D)
    x_start = torch.randn(B, D)
    fm = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        prediction_type='epsilon',
        clock_type='gaussian',
    )
    out = fm.training_losses({'default': model}, x_start)
    assert torch.isfinite(out['loss'])
    print(f'  test_fm_gaussian_supports_epsilon_predictor: OK '
          f'(loss={out["loss"].item():.4f})')


# ---- Existing-clock isolation guard -------------------------------------

def test_gaussian_rejects_path_conditioning():
    """clock_type='gaussian' must fail-loud at init_method_by_parameter for
    any path-conditioning misconfig. Covers all 4 user-facing knobs:
      - path_feature_type
      - path_conditioning.enabled
      - path_conditioning.injection.global.enabled
      - path_conditioning.injection.modulation.enabled
    """
    from htfm.experiment import init_method_by_parameter

    # Minimum viable config — vanilla FM with gaussian clock.
    base_p = lambda: {
        'method': 'fm',
        'device': 'cpu',
        'fm': {
            'alpha': 1.7, 'reverse_steps': 20, 'isotropic': True,
            'scale': 'scale_preserving', 'flow_type': 'straight_line',
            'noise_type': 'terminal', 'prediction_type': 'velocity',
            'clock': {'type': 'gaussian'},
        },
    }

    def expect_assert(p, must_contain):
        try:
            init_method_by_parameter(p)
        except AssertionError as e:
            for kw in must_contain:
                assert kw in str(e), f'expected {kw!r} in: {e}'
            return
        raise AssertionError(f'expected AssertionError, got success ({must_contain})')

    # Case 1: path_feature_type explicitly set
    p = base_p(); p['path_conditioning'] = {'path_feature_type': 'logsignature'}
    expect_assert(p, ['gaussian', 'path_feature_type'])

    # Case 2: pc.enabled=true explicitly
    p = base_p(); p['path_conditioning'] = {'enabled': True}
    expect_assert(p, ['gaussian', 'enabled'])

    # Case 3: injection.global.enabled=true (silent failure case before fix)
    p = base_p(); p['path_conditioning'] = {'global': {'enabled': True}}
    expect_assert(p, ['gaussian', 'global'])

    # Case 4: injection.modulation.enabled=true
    p = base_p(); p['path_conditioning'] = {'modulation': {'enabled': True}}
    expect_assert(p, ['gaussian', 'modulation'])

    # Case 5: clean config — should construct fine
    p = base_p()
    method = init_method_by_parameter(p)
    assert method.clock_type == 'gaussian'
    assert method.path_encoder is None
    print('  test_gaussian_rejects_path_conditioning: OK')


def test_existing_clocks_unchanged():
    """alpha_stable and student_t should still be present and constructable."""
    import torch.nn as nn
    from htfm.methods.flow_matching import FlowMatchingLevyProcess

    # alpha_stable still works
    fm_as = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        clock_type='alpha_stable',
    )
    assert fm_as.clock_type == 'alpha_stable'

    # student_t still works
    fm_st = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        clock_type='student_t', clock_kwargs={'nu': 3.0},
    )
    assert fm_st.clock_type == 'student_t'
    print('  test_existing_clocks_unchanged: OK')


# ---- Test runner --------------------------------------------------------

if __name__ == '__main__':
    print('--- GaussianClock tests ---')
    test_in_registry()
    test_capabilities()
    test_sample_X1_returns_G_unchanged()
    test_sample_initial_returns_G()
    test_tbar_linear_degenerates()
    test_sample_t_continuous()
    test_compute_R_m_raises()
    test_fm_gaussian_straight_line_runs()
    test_fm_gaussian_tbar_linear_runs()
    test_fm_gaussian_tbar_linear_equals_straight_line()
    test_fm_gaussian_rejects_ot_geodesic()
    test_fm_gaussian_rejects_signature_linear_ve()
    test_fm_gaussian_supports_epsilon_predictor()
    test_gaussian_rejects_path_conditioning()
    test_existing_clocks_unchanged()
    print('--- ALL OK ---')
