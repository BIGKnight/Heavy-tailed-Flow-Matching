"""Tests that fm.clock.alpha (canonical) works alongside fm.alpha (legacy).

Schema:
  Canonical (new):
    fm:
      clock:
        type: alpha_stable
        alpha: 1.7
  Legacy (existing YAMLs):
    fm:
      alpha: 1.7
      clock:
        type: alpha_stable

Both must yield identical FlowMatchingLevyProcess.

Run:
    cd /disk3/yezhen/Experiments/HTFM-refactor
    python -m htfm.methods.tests.test_alpha_schema_migration
"""
import torch

from htfm.methods import _build_fm


def _common_p(extra_fm=None):
    """Minimum YAML-like dict for FM construction."""
    base = {
        'method': 'fm',
        'device': 'cpu',
        'fm': {
            'reverse_steps': 20, 'isotropic': True,
            'scale': 'scale_preserving', 'flow_type': 'straight_line',
            'noise_type': 'terminal', 'prediction_type': 'velocity',
        },
    }
    if extra_fm:
        base['fm'].update(extra_fm)
    return base


def test_canonical_schema_alpha_stable():
    """fm.clock.alpha is read directly."""
    p = _common_p()
    p['fm']['clock'] = {'type': 'alpha_stable', 'alpha': 1.7}
    fm = _build_fm(p)
    assert fm.alpha == 1.7
    assert fm.clock_type == 'alpha_stable'
    assert fm.clock.alpha == 1.7
    print('  test_canonical_schema_alpha_stable: OK')


def test_legacy_schema_alpha_stable():
    """fm.alpha (legacy) is read as fallback."""
    p = _common_p()
    p['fm']['alpha'] = 1.7
    p['fm']['clock'] = {'type': 'alpha_stable'}
    fm = _build_fm(p)
    assert fm.alpha == 1.7
    assert fm.clock.alpha == 1.7
    print('  test_legacy_schema_alpha_stable: OK')


def test_canonical_overrides_legacy():
    """When both are set, fm.clock.alpha wins."""
    p = _common_p()
    p['fm']['alpha'] = 1.5  # legacy
    p['fm']['clock'] = {'type': 'alpha_stable', 'alpha': 1.9}  # canonical
    fm = _build_fm(p)
    assert fm.alpha == 1.9
    print('  test_canonical_overrides_legacy: OK')


def test_canonical_schema_student_t():
    """Student-t uses fm.clock.{type, nu}; alpha can be omitted entirely."""
    p = _common_p()
    p['fm']['clock'] = {'type': 'student_t', 'nu': 3.0}
    fm = _build_fm(p)
    # alpha defaults to 2.0 (Gaussian limit) for non-alpha-stable clocks
    assert fm.alpha == 2.0
    assert fm.clock_type == 'student_t'
    assert fm.clock.nu == 3.0
    print('  test_canonical_schema_student_t: OK')


def test_canonical_schema_gaussian():
    """Gaussian needs no clock kwargs at all."""
    p = _common_p()
    p['fm']['clock'] = {'type': 'gaussian'}
    fm = _build_fm(p)
    assert fm.alpha == 2.0
    assert fm.clock_type == 'gaussian'
    print('  test_canonical_schema_gaussian: OK')


def test_legacy_alpha_inert_for_student_t():
    """fm.alpha set with student_t still works (won't be passed to StudentTClock
    constructor); the value is recorded on FM but unused at runtime."""
    p = _common_p()
    p['fm']['alpha'] = 1.7  # legacy, inert for student_t
    p['fm']['clock'] = {'type': 'student_t', 'nu': 3.0}
    fm = _build_fm(p)
    assert fm.alpha == 1.7  # stored, but not queried by clock
    assert fm.clock.nu == 3.0
    print('  test_legacy_alpha_inert_for_student_t: OK')


def test_canonical_alpha_with_student_t_does_not_break():
    """If user erroneously sets fm.clock.alpha with student_t, the alpha value
    is still passed to FM (and stored on schedule), but stripped before going
    into StudentTClock kwargs (else TypeError)."""
    p = _common_p()
    p['fm']['clock'] = {'type': 'student_t', 'alpha': 1.7, 'nu': 3.0}
    fm = _build_fm(p)
    assert fm.alpha == 1.7
    # StudentTClock did NOT receive alpha (would have TypeError'd)
    assert fm.clock.nu == 3.0
    print('  test_canonical_alpha_with_student_t_does_not_break: OK')


def test_loss_bit_identity_canonical_vs_legacy():
    """Same alpha via canonical vs legacy schema -> identical training step."""
    import numpy as np
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self, D):
            super().__init__()
            self.lin = nn.Linear(D + 1, D)

        def forward(self, x, t, path_features=None):
            z = torch.cat([x, t.view(-1, 1)], dim=-1)
            return self.lin(z)

    D, B = 4, 6
    torch.manual_seed(0)
    model = Tiny(D)
    x_start = torch.randn(B, D)

    # Canonical
    p_new = _common_p()
    p_new['fm']['clock'] = {'type': 'alpha_stable', 'alpha': 1.7}
    fm_new = _build_fm(p_new)

    # Legacy
    p_old = _common_p()
    p_old['fm']['alpha'] = 1.7
    p_old['fm']['clock'] = {'type': 'alpha_stable'}
    fm_old = _build_fm(p_old)

    torch.manual_seed(42); np.random.seed(42)
    out_new = fm_new.training_losses({'default': model}, x_start)
    torch.manual_seed(42); np.random.seed(42)
    out_old = fm_old.training_losses({'default': model}, x_start)

    diff = abs(out_new['loss'].item() - out_old['loss'].item())
    assert diff == 0.0, (
        f'canonical vs legacy schema diff {diff} (expected 0.0)')
    print(f'  test_loss_bit_identity_canonical_vs_legacy: OK (diff={diff:.2e})')


if __name__ == '__main__':
    print('--- alpha schema migration tests ---')
    test_canonical_schema_alpha_stable()
    test_legacy_schema_alpha_stable()
    test_canonical_overrides_legacy()
    test_canonical_schema_student_t()
    test_canonical_schema_gaussian()
    test_legacy_alpha_inert_for_student_t()
    test_canonical_alpha_with_student_t_does_not_break()
    test_loss_bit_identity_canonical_vs_legacy()
    print('--- ALL OK ---')
