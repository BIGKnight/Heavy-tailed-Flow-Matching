"""Tests for the tbar_linear flow type.

Math:
    X_t = bar_T_t * X_1 + (1 - bar_T_t) * X_0
    v*  = bar_T_dot_t * (X_1 - X_0)
where bar_T_t = T_t / T_1 with T_path padded by a leading 0 so bar_T(0)=0,
bar_T(1)=1. Training samples t exactly on the clock grid {1/T_total, ...,
(T_total-1)/T_total} to avoid interpolation.

Run:
    cd /disk3/yezhen/Experiments/HTFM-refactor
    python -m htfm.methods.tests.test_tbar_linear
"""
import torch

from htfm.methods.clocks.alpha_stable import AlphaStableClock
from htfm.methods.clocks.student_t import StudentTClock


# ---- AlphaStableClock.compute_bar_T_at ---------------------------------

def _make_clock_with_T_path(T_path):
    """Bypass __init__ by injecting a pre-built T_path into a fresh clock."""
    c = AlphaStableClock.__new__(AlphaStableClock)
    c.device = T_path.device
    c._last_T_path = T_path
    return c


def test_alpha_stable_endpoint_bar_T():
    """At t=1/T_total (k=1), bar_T = T_padded[1]/T_1 = A_0/T_1.
    At t=(T_total-1)/T_total (k=T_total-1), bar_T = T_padded[T_total-1]/T_1.
    """
    torch.manual_seed(0)
    B, T_total = 3, 50
    A = torch.rand(B, T_total).abs() + 0.1
    T_path = torch.cumsum(A, dim=-1)               # [B, T_total]
    clock = _make_clock_with_T_path(T_path)

    # k=1 → t=1/T_total, bar_T should be A[0]/T_1.
    t1 = torch.full((B,), 1.0 / T_total)
    bar_T, _ = clock.compute_bar_T_at(t1)
    expected = A[:, 0] / T_path[:, -1]
    diff = (bar_T - expected).abs().max().item()
    assert diff < 1e-6, f'bar_T at k=1 mismatch {diff}'
    print('  test_alpha_stable_endpoint_bar_T: OK')


def test_alpha_stable_grid_lookup_exact():
    """Sweep all interior k; bar_T should equal cumsum(A)[k-1]/T_1 = T_padded[k]/T_1.
    (T_padded[0]=0, T_padded[k] for k>=1 is T_path[k-1].)"""
    torch.manual_seed(1)
    B, T_total = 4, 20
    A = torch.rand(B, T_total).abs() + 0.1
    T_path = torch.cumsum(A, dim=-1)
    clock = _make_clock_with_T_path(T_path)

    for k in range(1, T_total):
        t = torch.full((B,), k / float(T_total))
        bar_T, _ = clock.compute_bar_T_at(t)
        expected = T_path[:, k - 1] / T_path[:, -1]
        diff = (bar_T - expected).abs().max().item()
        assert diff < 1e-6, f'k={k}: max diff {diff}'
    print('  test_alpha_stable_grid_lookup_exact: OK')


def test_alpha_stable_bar_T_dot_matches_segment_slope():
    """bar_T_dot at grid t = k/T_total should equal A_k * T_total / T_1
    (right segment slope).
    """
    torch.manual_seed(2)
    B, T_total = 4, 30
    A = torch.rand(B, T_total).abs() + 0.1
    T_path = torch.cumsum(A, dim=-1)
    clock = _make_clock_with_T_path(T_path)

    # at t = k/T_total for k in [1, T_total-1]: right segment [k, k+1]
    # padded T_padded[k+1] - T_padded[k] = (T_path[k] - T_path[k-1]) = A[k]
    # but at k=T_total-1, T_padded[T_total] = T_1 (the original last element);
    # T_padded[T_total-1] = T_path[T_total-2]. Difference = A[T_total-1].
    for k in range(1, T_total):
        t = torch.full((B,), k / float(T_total))
        _, bar_T_dot = clock.compute_bar_T_at(t)
        expected = A[:, k] * T_total / T_path[:, -1]
        diff = (bar_T_dot - expected).abs().max().item()
        assert diff < 1e-5, f'k={k}: bar_T_dot diff {diff}'
    print('  test_alpha_stable_bar_T_dot_matches_segment_slope: OK')


def test_alpha_stable_bar_T_dot_telescopes_to_one():
    """sum over all segments of (bar_T_dot * (1/T_total)) = bar_T(1) - bar_T(0) = 1."""
    torch.manual_seed(3)
    B, T_total = 5, 100
    A = torch.rand(B, T_total).abs() + 0.05
    T_path = torch.cumsum(A, dim=-1)
    clock = _make_clock_with_T_path(T_path)

    seg_sum = torch.zeros(B)
    for k in range(0, T_total):
        # Use t in segment k by sampling its left endpoint k/T_total. (k=0 is
        # excluded by sampler in production, but we cover it here for the sum.)
        if k == 0:
            t = torch.full((B,), 0.5 / T_total)  # any t in (0, 1/T_total)
            # bar_T_dot is the slope of segment 0 -> A[0]/T_1*T_total
            _, dot = clock.compute_bar_T_at(torch.full((B,), 1.0 / T_total))  # k=1 sample
            # Actually for segment 0, we'd need t in [0, 1/T_total). Skip — the
            # sampler excludes this segment.
            continue
        t = torch.full((B,), k / float(T_total))
        _, dot = clock.compute_bar_T_at(t)
        seg_sum = seg_sum + dot / T_total

    # seg_sum covers segments k=1..T_total-1. Segment 0 contributes A[0]/T_1.
    seg0 = A[:, 0] / T_path[:, -1]
    total = seg0 + seg_sum
    diff = (total - 1.0).abs().max().item()
    assert diff < 1e-5, f'bar_T_dot telescope total {total}, expected 1, diff {diff}'
    print('  test_alpha_stable_bar_T_dot_telescopes_to_one: OK')


def test_alpha_stable_mc_replication_alignment():
    """B_full = rep * B_outer; bar_T at sample i should use T_path[i % B_outer]."""
    torch.manual_seed(4)
    B_outer, T_total, rep = 3, 20, 4
    A = torch.rand(B_outer, T_total).abs() + 0.1
    T_path = torch.cumsum(A, dim=-1)
    clock = _make_clock_with_T_path(T_path)

    # Same t for all entries; bar_T should repeat pattern [bar_T_outer] * rep.
    k = 7
    t = torch.full((B_outer * rep,), k / float(T_total))
    bar_T, _ = clock.compute_bar_T_at(t)
    expected_outer = T_path[:, k - 1] / T_path[:, -1]
    expected = expected_outer.repeat(rep)
    diff = (bar_T - expected).abs().max().item()
    assert diff < 1e-6, f'MC replication mismatch {diff}'
    print('  test_alpha_stable_mc_replication_alignment: OK')


def test_alpha_stable_sample_t_grid_aligned():
    """Sampled t should be exactly k/T_total with k integer in [1, T_total-1]."""
    torch.manual_seed(5)
    B, T_total = 1, 40
    T_path = torch.cumsum(torch.rand(B, T_total).abs() + 0.1, dim=-1)
    clock = _make_clock_with_T_path(T_path)

    t = clock.sample_t_for_tbar_linear(1000)
    k = (t * T_total).round()
    diff = (t * T_total - k).abs().max().item()
    assert diff < 1e-5, f'sampled t not on grid: max off-grid {diff}'
    assert k.min().item() >= 1
    assert k.max().item() <= T_total - 1
    print(f'  test_alpha_stable_sample_t_grid_aligned: OK  (k range '
          f'[{int(k.min())}, {int(k.max())}])')


# ---- StudentTClock.compute_bar_T_at -------------------------------------

def test_student_t_degenerates():
    """For Student-t, bar_T_t = t and bar_T_dot_t = 1 exactly."""
    c = StudentTClock(nu=3.0, device='cpu')
    t = torch.linspace(0.05, 0.95, 19)
    bar_T, bar_T_dot = c.compute_bar_T_at(t)
    assert torch.allclose(bar_T, t)
    assert torch.allclose(bar_T_dot, torch.ones_like(t))
    print('  test_student_t_degenerates: OK')


def test_student_t_sample_t_continuous():
    """Student-t t-sampler returns continuous values in (0, 1)."""
    c = StudentTClock(nu=3.0, device='cpu')
    t = c.sample_t_for_tbar_linear(2000)
    assert (t > 0).all() and (t < 1).all()
    print(f'  test_student_t_sample_t_continuous: OK  (t range '
          f'[{t.min():.4f}, {t.max():.4f}])')


# ---- FM end-to-end with tbar_linear -------------------------------------

def test_fm_tbar_linear_alpha_stable_runs():
    """End-to-end training step for alpha_stable + tbar_linear; loss finite,
    _last_T_path populated, MC replication works."""
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
        flow_type='tbar_linear', noise_type='forward_process',
        clock_type='alpha_stable',
    )
    out = fm.training_losses({'default': model}, x_start,
                             monte_carlo_outer=2, monte_carlo_inner=2,
                             monte_carlo_t=2)
    assert torch.isfinite(out['loss']), f'non-finite loss: {out["loss"]}'
    assert tuple(fm.clock._last_T_path.shape) == (B * 2, 20)
    print(f'  test_fm_tbar_linear_alpha_stable_runs: OK '
          f'(loss={out["loss"].item():.4f})')


def test_fm_tbar_linear_student_t_equals_straight_line():
    """For Student-t, tbar_linear should produce the same loss as straight_line
    modulo float32 round-off (since bar_T_t = t and bar_T_dot_t = 1)."""
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

    fm_tbar = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='tbar_linear', noise_type='terminal',
        clock_type='student_t', clock_kwargs={'nu': 3.0},
    )
    fm_sl = FlowMatchingLevyProcess(
        alpha=1.7, device='cpu', reverse_steps=20,
        isotropic=True, scale='scale_preserving',
        flow_type='straight_line', noise_type='terminal',
        clock_type='student_t', clock_kwargs={'nu': 3.0},
    )

    torch.manual_seed(42); np.random.seed(42)
    out_tbar = fm_tbar.training_losses({'default': model}, x_start)
    torch.manual_seed(42); np.random.seed(42)
    out_sl = fm_sl.training_losses({'default': model}, x_start)
    diff = abs(out_tbar['loss'].item() - out_sl['loss'].item())
    assert diff < 1e-5, (
        f'student_t tbar_linear loss {out_tbar["loss"].item()} vs straight_line '
        f'{out_sl["loss"].item()} differ by {diff}')
    print(f'  test_fm_tbar_linear_student_t_equals_straight_line: OK (diff={diff:.2e})')


def test_fm_tbar_linear_rejects_terminal_alpha_stable():
    """alpha_stable + terminal + tbar_linear must assert at construction."""
    from htfm.methods.flow_matching import FlowMatchingLevyProcess
    try:
        FlowMatchingLevyProcess(
            alpha=1.7, device='cpu', reverse_steps=20,
            isotropic=True, scale='scale_preserving',
            flow_type='tbar_linear', noise_type='terminal',
            clock_type='alpha_stable',
        )
    except AssertionError as e:
        assert 'forward_process' in str(e)
        print('  test_fm_tbar_linear_rejects_terminal_alpha_stable: OK')
        return
    raise AssertionError('expected AssertionError for alpha_stable+terminal+tbar_linear')


if __name__ == '__main__':
    print('--- tbar_linear tests ---')
    test_alpha_stable_endpoint_bar_T()
    test_alpha_stable_grid_lookup_exact()
    test_alpha_stable_bar_T_dot_matches_segment_slope()
    test_alpha_stable_bar_T_dot_telescopes_to_one()
    test_alpha_stable_mc_replication_alignment()
    test_alpha_stable_sample_t_grid_aligned()
    test_student_t_degenerates()
    test_student_t_sample_t_continuous()
    test_fm_tbar_linear_alpha_stable_runs()
    test_fm_tbar_linear_student_t_equals_straight_line()
    test_fm_tbar_linear_rejects_terminal_alpha_stable()
    print('--- ALL OK ---')
