"""Tests for the VE-type signature-linear schedule.

Covers:
  - alpha/beta endpoint conditions for any (lambda_alpha, lambda_beta, R_m)
  - lambda=0 degeneration to plain linear VE (alpha_t=t, beta_t=1-t)
  - Affine-target identity: at v* the path differential matches the affine ODE
  - Admissibility assertion: |lambda_beta| >= 1 raises
  - AlphaStableClock.compute_R_m: linear path -> R_m=0; front/back-loaded -> sign
  - FlowMatchingLevyProcess: signature_linear_ve construction asserts
  - Byte-identity: with flow_type='straight_line' & ve_schedule=None, the new
    code path is bypassed (no behavior change).

Run:
    cd /disk3/yezhen/Experiments/HTFM-refactor
    python -m htfm.methods.tests.test_signature_linear_ve
"""

import math
import torch

from htfm.methods.noise_schedule.signature_linear_ve import SignatureLinearVESchedule
from htfm.methods.clocks.alpha_stable import AlphaStableClock


# ---- Schedule curves ----------------------------------------------------

def test_endpoints_invariant_under_lambda_and_Rm():
    """alpha(0)=0, alpha(1)=1, beta(0)=1, beta(1)=0 for ANY (la, lb, R_m)."""
    for la in [-0.5, 0.0, 0.25, 0.9]:
        for lb in [-0.9, 0.0, 0.25, 0.9]:
            sch = SignatureLinearVESchedule(la, lb)
            for R in [-1.0, -0.3, 0.0, 0.7, 1.0]:
                t0 = torch.tensor([0.0])
                t1 = torch.tensor([1.0])
                R_t = torch.tensor([R])
                a0 = sch.alpha_t(t0, R_t).item()
                a1 = sch.alpha_t(t1, R_t).item()
                b0 = sch.beta_t(t0, R_t).item()
                b1 = sch.beta_t(t1, R_t).item()
                assert abs(a0) < 1e-6, f'alpha(0)={a0} != 0 (la={la}, lb={lb}, R={R})'
                assert abs(a1 - 1.0) < 1e-6, f'alpha(1)={a1} != 1'
                assert abs(b0 - 1.0) < 1e-6, f'beta(0)={b0} != 1'
                assert abs(b1) < 1e-6, f'beta(1)={b1} != 0'
    print('  test_endpoints_invariant_under_lambda_and_Rm: OK')


def test_lambda_zero_degenerates_to_linear_ve():
    """la=lb=0 -> alpha_t=t, beta_t=1-t for any t and R_m."""
    sch = SignatureLinearVESchedule(0.0, 0.0)
    t = torch.linspace(0.0, 1.0, 11)
    for R in [-1.0, 0.0, 1.0]:
        R_t = torch.full_like(t, R)
        a = sch.alpha_t(t, R_t)
        b = sch.beta_t(t, R_t)
        assert torch.allclose(a, t, atol=1e-7), f'alpha != t at R={R}'
        assert torch.allclose(b, 1.0 - t, atol=1e-7), f'beta != 1-t at R={R}'
    print('  test_lambda_zero_degenerates_to_linear_ve: OK')


def test_admissibility_assert():
    """|lambda_beta| >= 1 must trip the constructor assert."""
    for bad in [1.0, -1.0, 1.5, -2.7]:
        try:
            SignatureLinearVESchedule(0.0, bad)
        except AssertionError:
            continue
        raise AssertionError(f'lambda_beta={bad} should have asserted, did not')
    # Boundary: |lb|<1 must work.
    SignatureLinearVESchedule(0.0, 0.999)
    SignatureLinearVESchedule(0.0, -0.999)
    print('  test_admissibility_assert: OK')


def test_affine_target_consistency_at_endpoints():
    """v* at t=0 and t=1 should be finite and consistent with d/dt[X_t].

    For X_t = alpha_t * X_1 + beta_t * X_0, the analytical velocity is
    dX/dt = alpha_dot * X_1 + beta_dot * X_0. The closed-form affine_target
    rewrites this in terms of (X_t, X_1):
        v* = (beta_dot/beta) * X_t + (alpha_dot - (beta_dot/beta) * alpha) * X_1
           = (beta_dot/beta) * (alpha*X_1 + beta*X_0) + (alpha_dot - (beta_dot/beta)*alpha)*X_1
           = beta_dot * X_0 + alpha_dot * X_1.
    Equivalence holds wherever beta > 0; t=0 has beta=1 and t<1 has beta>0.
    """
    sch = SignatureLinearVESchedule(0.3, 0.4)
    torch.manual_seed(0)
    B, D = 8, 4
    X_0 = torch.randn(B, D)
    X_1 = torch.randn(B, D)
    R_m = torch.tensor([-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99, -0.99])
    for t_val in [0.0, 0.2, 0.5, 0.8, 0.9]:
        t = torch.full((B,), t_val)
        # Closed-form direct velocity (the "ground truth")
        spatial = (1,) * (X_0.dim() - 1)
        t_b = t.view(-1, *spatial)
        R_b = R_m.view(-1, *spatial)
        a_dot = sch.alpha_dot(t_b, R_b)
        b_dot = sch.beta_dot(t_b, R_b)
        v_direct = a_dot * X_1 + b_dot * X_0
        # affine_target form (uses X_t)
        X_t = sch.X_t_from_endpoints(t, R_m, X_0, X_1)
        v_aff = sch.affine_target(t, R_m, X_t, X_1)
        diff = (v_direct - v_aff).abs().max().item()
        assert diff < 1e-5, f'v_direct vs affine mismatch at t={t_val}: {diff}'
    print('  test_affine_target_consistency_at_endpoints: OK')


# ---- compute_R_m on AlphaStableClock ------------------------------------

def test_R_m_linear_path_is_zero():
    """A perfectly linear T_path ~ s gives R_m = 2*int_0^1 s ds - 1 = 0."""
    n_steps = 200
    s = torch.linspace(0, 1, n_steps).unsqueeze(0).expand(4, -1)
    R = AlphaStableClock.compute_R_m(s)
    assert torch.allclose(R, torch.zeros(4), atol=1e-6), f'R_m={R}'
    print('  test_R_m_linear_path_is_zero: OK')


def test_R_m_front_loaded_positive():
    """Front-loaded clock (T_s rises early, then plateaus) has R_m > 0."""
    n_steps = 200
    s = torch.linspace(0, 1, n_steps)
    front = s ** 0.25  # rises sharply early, plateaus near 1
    front = front.unsqueeze(0).expand(2, -1)
    R = AlphaStableClock.compute_R_m(front)
    assert (R > 0.3).all(), f'expected R_m>0.3 for front-loaded, got {R}'
    print(f'  test_R_m_front_loaded_positive: OK (R_m={R.tolist()})')


def test_R_m_back_loaded_negative():
    """Back-loaded clock (T_s flat then jumps late) has R_m < 0."""
    n_steps = 200
    s = torch.linspace(0, 1, n_steps)
    back = s ** 4.0
    back = back.unsqueeze(0).expand(2, -1)
    R = AlphaStableClock.compute_R_m(back)
    assert (R < -0.3).all(), f'expected R_m<-0.3 for back-loaded, got {R}'
    print(f'  test_R_m_back_loaded_negative: OK (R_m={R.tolist()})')


def test_R_m_clamp_range():
    """R_m must always lie in [-1, 1] (closed-form bound after Fubini)."""
    torch.manual_seed(0)
    # Random monotone-nondecreasing T_path
    inc = torch.rand(8, 100).abs()
    T_path = torch.cumsum(inc, dim=-1)
    R = AlphaStableClock.compute_R_m(T_path)
    assert (R.abs() <= 1.0 + 1e-6).all(), f'|R_m|>1: {R}'
    print(f'  test_R_m_clamp_range: OK (R_m range [{R.min():.3f}, {R.max():.3f}])')


# ---- FM construction asserts --------------------------------------------

def test_fm_signature_linear_ve_requires_alpha_stable():
    """flow_type='signature_linear_ve' must reject student_t clock."""
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    sch = SignatureLinearVESchedule(0.25, 0.25)
    try:
        FlowMatchingLevyProcess(
            alpha=1.7, device='cpu', reverse_steps=20,
            isotropic=True, scale='scale_preserving',
            flow_type='signature_linear_ve', noise_type='forward_process',
            clock_type='student_t', clock_kwargs={'nu': 3.0},
            ve_schedule=sch,
        )
    except AssertionError:
        # student_t-vs-flow_type assert fires first; that's fine — both gates
        # protect the same misconfig.
        print('  test_fm_signature_linear_ve_requires_alpha_stable: OK')
        return
    raise AssertionError('signature_linear_ve + student_t should have asserted')


def test_fm_signature_linear_ve_requires_forward_process():
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    sch = SignatureLinearVESchedule(0.25, 0.25)
    try:
        FlowMatchingLevyProcess(
            alpha=1.7, device='cpu', reverse_steps=20,
            isotropic=True, scale='scale_preserving',
            flow_type='signature_linear_ve', noise_type='terminal',
            clock_type='alpha_stable', ve_schedule=sch,
        )
    except AssertionError as e:
        assert 'forward_process' in str(e), f'unexpected assert: {e}'
        print('  test_fm_signature_linear_ve_requires_forward_process: OK')
        return
    raise AssertionError('signature_linear_ve + terminal should have asserted')


def test_fm_signature_linear_ve_requires_ve_schedule():
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    try:
        FlowMatchingLevyProcess(
            alpha=1.7, device='cpu', reverse_steps=20,
            isotropic=True, scale='scale_preserving',
            flow_type='signature_linear_ve', noise_type='forward_process',
            clock_type='alpha_stable', ve_schedule=None,
        )
    except AssertionError as e:
        assert 've_schedule' in str(e), f'unexpected assert: {e}'
        print('  test_fm_signature_linear_ve_requires_ve_schedule: OK')
        return
    raise AssertionError('signature_linear_ve + ve_schedule=None should have asserted')


# ---- Byte-identity guard for the existing flow_types --------------------

def test_existing_flow_types_unchanged_when_ve_schedule_none():
    """Constructing FM with flow_type='straight_line' and ve_schedule=None
    must NOT touch any signature_linear_ve code path. Smoke check that no
    asserts fire and ve_schedule is stored as None.
    """
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    fm = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        clock_type='alpha_stable', ve_schedule=None,
    )
    assert fm.ve_schedule is None
    assert fm.flow_type == 'straight_line'
    print('  test_existing_flow_types_unchanged_when_ve_schedule_none: OK')


def test_lambda_zero_loss_equals_straight_line_loss():
    """End-to-end equivalence: with lambda_alpha=lambda_beta=0, the
    signature_linear_ve training loss must match the straight_line training
    loss modulo float32 round-off. Exercises the full training_losses_fm
    pipeline (sample X1, compute R_m, build Xt, compute target, model
    forward, lp loss, aggregator).

    Both torch AND numpy seeds must be reset because gen_a uses scipy/numpy.
    """
    import numpy as np
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

    ve0 = SignatureLinearVESchedule(0.0, 0.0)
    fm_ve0 = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='signature_linear_ve', noise_type='forward_process',
        clock_type='alpha_stable', ve_schedule=ve0,
    )
    fm_sl = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='forward_process',
        clock_type='alpha_stable',
    )

    torch.manual_seed(42); np.random.seed(42)
    out_ve0 = fm_ve0.training_losses({'default': model}, x_start)
    torch.manual_seed(42); np.random.seed(42)
    out_sl = fm_sl.training_losses({'default': model}, x_start)

    diff = abs(out_ve0['loss'].item() - out_sl['loss'].item())
    assert diff < 1e-5, (
        f'lambda=0 loss {out_ve0["loss"].item()} vs straight_line loss '
        f'{out_sl["loss"].item()} diverged by {diff} (>1e-5 tolerance).')
    print(f'  test_lambda_zero_loss_equals_straight_line_loss: OK '
          f'(diff={diff:.2e})')


def test_mc_inner_and_mc_t_run():
    """End-to-end signature_linear_ve loss with MC_outer/inner/t > 1.
    Asserts shape-correctness of R_m repeat semantics.
    """
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
    ve = SignatureLinearVESchedule(0.25, 0.25)
    fm = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='signature_linear_ve', noise_type='forward_process',
        clock_type='alpha_stable', ve_schedule=ve,
    )
    out = fm.training_losses({'default': model}, x_start,
                             monte_carlo_outer=2,
                             monte_carlo_inner=3,
                             monte_carlo_t=2)
    assert torch.isfinite(out['loss']), f'non-finite loss: {out["loss"]}'
    # _last_T_path is shaped [B*MC_outer, T_total].
    assert tuple(fm.clock._last_T_path.shape) == (B * 2, 20)
    print(f'  test_mc_inner_and_mc_t_run: OK (loss={out["loss"].item():.4f})')


# ---- Test runner --------------------------------------------------------

if __name__ == '__main__':
    print('--- SignatureLinearVE tests ---')
    test_endpoints_invariant_under_lambda_and_Rm()
    test_lambda_zero_degenerates_to_linear_ve()
    test_admissibility_assert()
    test_affine_target_consistency_at_endpoints()
    test_R_m_linear_path_is_zero()
    test_R_m_front_loaded_positive()
    test_R_m_back_loaded_negative()
    test_R_m_clamp_range()
    test_fm_signature_linear_ve_requires_alpha_stable()
    test_fm_signature_linear_ve_requires_forward_process()
    test_fm_signature_linear_ve_requires_ve_schedule()
    test_existing_flow_types_unchanged_when_ve_schedule_none()
    test_lambda_zero_loss_equals_straight_line_loss()
    test_mc_inner_and_mc_t_run()
    print('--- ALL OK ---')
