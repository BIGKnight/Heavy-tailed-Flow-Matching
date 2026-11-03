"""Tests for Student-t clock.

Run:
    cd <repo root>
    python -m htfm.methods.tests.test_student_t_clock
"""
import math
import torch

from htfm.methods.clocks import build_clock, available_clocks
from htfm.methods.clocks.student_t import StudentTClock


def test_alpha_stable_in_registry():
    """Post-v1-step-E.2: 'alpha_stable' is a real clock in _REGISTRY (not None).
    Only `clock_type=None` still resolves to None now.
    """
    assert build_clock(None) is None
    assert 'alpha_stable' in available_clocks()
    assert 'student_t' in available_clocks()
    print('  test_alpha_stable_in_registry: OK')


def test_invgamma_moments():
    """E[V] = (nu/2)/(nu/2 - 1) for nu>2.

    For nu=3: E[V] = 1.5/0.5 = 3.  Sample size 100k → std error on mean ~0.01.
    """
    torch.manual_seed(0)
    c = build_clock('student_t', nu=3.0, device='cpu')
    V = c.sample_V(100_000)
    empirical_mean = V.mean().item()
    expected_mean = 3.0
    err = abs(empirical_mean - expected_mean)
    assert err < 0.2, f'empirical E[V]={empirical_mean}, expected {expected_mean}, err={err}'
    assert (V > 0).all()
    print(f'  test_invgamma_moments: OK  (E[V]≈{empirical_mean:.3f}, expected {expected_mean})')


def test_x1_marginal_isotropic():
    """x_1 = sqrt(V)·G marginal is a Student-t isotropic r.v. with scale 1, dof=nu.

    Check: per-coord variance = nu/(nu-2) for nu>2.  For nu=4: 4/2 = 2.
    """
    torch.manual_seed(1)
    c = build_clock('student_t', nu=4.0, device='cpu')
    B = 50_000
    V = c.sample_V(B)
    G = torch.randn(B, 1)
    x1 = c.compute_x1(V, [B, 1], G)
    # per-coord variance
    var = x1.var().item()
    expected_var = 4.0 / (4.0 - 2.0)   # = 2
    err = abs(var - expected_var)
    assert err < 0.3, f'x1 var={var}, expected {expected_var}, err={err}'
    assert x1.mean().abs().item() < 0.1
    print(f'  test_x1_marginal_isotropic: OK  (Var≈{var:.3f}, expected {expected_var})')


def test_feature_modes():
    c_s = StudentTClock(nu=3, device='cpu', feature_mode='scalar')
    c_l = StudentTClock(nu=3, device='cpu', feature_mode='log_scalar')
    c_b = StudentTClock(nu=3, device='cpu', feature_mode='both')
    assert c_s.feature_dim == 1
    assert c_l.feature_dim == 1
    assert c_b.feature_dim == 2
    V = torch.tensor([0.5, 1.0, 2.0])
    torch.testing.assert_close(c_s.encode_feature(V), V.view(-1, 1))
    torch.testing.assert_close(c_l.encode_feature(V), torch.log(V).view(-1, 1))
    both = c_b.encode_feature(V)
    assert both.shape == (3, 2)
    torch.testing.assert_close(both[:, 0], V)
    torch.testing.assert_close(both[:, 1], torch.log(V))
    print('  test_feature_modes: OK')


def test_shape_compute_x1():
    """Broadcasting V across image dims works."""
    c = StudentTClock(nu=3, device='cpu')
    B = 4
    V = c.sample_V(B)
    G = torch.randn(B, 3, 32, 32)
    x1 = c.compute_x1(V, [B, 3, 32, 32], G)
    assert x1.shape == (B, 3, 32, 32)
    # per-sample: x1[i, ...] / G[i, ...] = sqrt(V[i]) constant
    ratio = x1[0] / G[0]
    assert (ratio - math.sqrt(V[0].item())).abs().max().item() < 1e-5
    print('  test_shape_compute_x1: OK')


def test_default_fm_equivalence():
    """Building FlowMatchingLevyProcess with clock_type='alpha_stable' yields
    an AlphaStableClock instance (post v1-step-E.2 unified dispatch).
    """
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    from htfm.methods.clocks.alpha_stable import AlphaStableClock
    fm = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=200,
        noise_type='forward_process',
    )  # all defaults → clock_type='alpha_stable'
    assert isinstance(fm.clock, AlphaStableClock)
    assert fm.clock_type == 'alpha_stable'
    # with student_t
    fm2 = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=200,
        noise_type='forward_process',
        clock_type='student_t',
        clock_kwargs={'nu': 3.0, 'feature_mode': 'log_scalar'},
    )
    assert fm2.clock is not None
    assert isinstance(fm2.clock, StudentTClock)
    assert fm2.clock.nu == 3.0
    print('  test_default_fm_equivalence: OK')


if __name__ == '__main__':
    print('Student-t clock tests:')
    test_alpha_stable_in_registry()
    test_invgamma_moments()
    test_x1_marginal_isotropic()
    test_feature_modes()
    test_shape_compute_x1()
    test_default_fm_equivalence()
    print('ALL PASS')
