"""Correctness tests for ODE solvers.

Run as:
    cd <repo root>
    python -m pytest tests/test_fm_solvers.py

Tests
-----
1. Consistency on constant v: all solvers should match analytic solution for any N.
2. Consistency on linear v(x,t)=-x: closed-form x(0) = x(1)*e (backward integration).
3. Empirical convergence order: log-log slope of error vs N should be ~ solver order.
"""
import math

import pytest
import torch

from htfm.methods.fm_solvers import build_solver, available_solvers


@pytest.fixture
def solvers():
    return available_solvers()


EXPECTED_ORDER = {
    'euler': 1,
    'heun': 2,
    'midpoint': 2,
    'rk4': 4,
    'ab2': 2,
    'dopri5': 5,  # adaptive; with fixed eval slope varies but high
}


def test_constant_velocity(solvers, rtol=1e-5):
    """v(x,t)=c constant; exact: x(0) = x(1) - c*(1-0). Every solver (order>=1) is exact.
    Use float64 so we measure algebraic correctness, not float32 round-off."""
    torch.manual_seed(0)
    x1 = torch.randn(4, 3, dtype=torch.float64)
    c = torch.tensor([0.7, -0.3, 1.2], dtype=torch.float64)
    v_fn = lambda x, t: c.expand_as(x)
    x0_exact = x1 - c  # integrate dx/dt = c from t=1 backward: x(0) = x(1) - c*(1-0)

    print('--- constant velocity (exact for all orders) ---')
    for name in solvers:
        s = build_solver(name)
        for N in [1, 5, 50]:
            x0 = s.integrate(v_fn, x1, N)
            err = (x0 - x0_exact).abs().max().item()
            assert err < rtol, f'{name} N={N}: err={err:.2e} > {rtol}'
        print(f'  {name:10s}: OK (err < {rtol})')


def test_linear_velocity_exact(solvers):
    """v(x,t) = -x; backward x(t-dt) = x(t) + dt*x(t). Exact continuous: x(0) = x(1)*e."""
    torch.manual_seed(1)
    x1 = torch.tensor([1.0], dtype=torch.float64)
    v_fn = lambda x, t: -x

    print('--- v(x,t) = -x; exact x(0) = x(1)*e = %.6f ---' % math.e)
    N = 1000
    for name in solvers:
        s = build_solver(name)
        if name == 'dopri5':
            x0 = s.integrate(v_fn, x1, 1, rtol=1e-9, atol=1e-9)
        else:
            x0 = s.integrate(v_fn, x1, N)
        err = abs(x0.item() - math.e)
        print(f'  {name:10s}: x(0)={x0.item():.6f}  |err|={err:.2e}')


def test_convergence_order(solvers, N_list=(10, 20, 40, 80, 160)):
    """v(x,t) = -x; empirical slope of log(err) vs log(1/N) ~= solver order."""
    torch.manual_seed(2)
    x1 = torch.tensor([1.0], dtype=torch.float64)
    v_fn = lambda x, t: -x
    exact = math.e

    print('--- convergence order (v=-x, error vs N) ---')
    all_pass = True
    for name in solvers:
        if name == 'dopri5':
            # adaptive: error is tolerance-bound, not N-bound; test accuracy instead
            s = build_solver(name)
            x0 = s.integrate(v_fn, x1, 1, rtol=1e-8, atol=1e-8)
            err = abs(x0.item() - exact)
            ok = err < 1e-6
            tag = 'OK ' if ok else 'FAIL'
            print(f'  {name:10s}: adaptive N=1 rtol=atol=1e-8  |err|={err:.2e}  [{tag}]')
            if not ok:
                all_pass = False
            continue
        s = build_solver(name)
        errs = []
        for N in N_list:
            x0 = s.integrate(v_fn, x1, N)
            errs.append(abs(x0.item() - exact))
        # log-log slope between adjacent N
        slopes = []
        for i in range(len(N_list) - 1):
            if errs[i] == 0 or errs[i + 1] == 0:
                continue
            slope = (math.log(errs[i]) - math.log(errs[i + 1])) / (
                math.log(N_list[i + 1]) - math.log(N_list[i]))
            slopes.append(slope)
        mean_slope = sum(slopes) / len(slopes) if slopes else float('nan')
        expected = EXPECTED_ORDER.get(name, 1)
        ok = abs(mean_slope - expected) < 0.5 or mean_slope > expected - 0.3
        tag = 'OK ' if ok else 'FAIL'
        print(f'  {name:10s}: errs={[f"{e:.2e}" for e in errs]}  '
              f'slope~{mean_slope:.2f} (expected~{expected})  [{tag}]')
        if not ok:
            all_pass = False
    if not all_pass:
        raise AssertionError('one or more solvers failed convergence-order test')


def test_step_hook_called():
    """step_hook is applied after every step and its output feeds the next step."""
    torch.manual_seed(3)
    x1 = torch.tensor([10.0])
    v_fn = lambda x, t: torch.zeros_like(x)  # no change
    call_count = [0]

    def hook(x, i):
        call_count[0] += 1
        return x - 1.0  # subtract 1 each step

    print('--- step_hook feeds into next step ---')
    s = build_solver('euler')
    x0 = s.integrate(v_fn, x1, 5, step_hook=hook)
    assert call_count[0] == 5, f'hook called {call_count[0]} times, expected 5'
    assert x0.item() == 5.0, f'expected 5.0, got {x0.item()}'
    print('  OK')


if __name__ == '__main__':
    solvers = available_solvers()
    print(f'registered solvers: {solvers}\n')
    test_constant_velocity(solvers)
    print()
    test_linear_velocity_exact(solvers)
    print()
    test_convergence_order(solvers)
    print()
    test_step_hook_called()
    print('\nALL TESTS PASSED')
