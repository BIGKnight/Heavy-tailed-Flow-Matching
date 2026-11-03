"""Tests for PathSampler (group-size + resample-interval policy).

Run:
    cd /disk3/yezhen/Experiments/Heavy-tailed-flow-matching
    python -m htfm.methods.tests.test_path_sampling
"""
import numpy as np
import torch

from htfm.methods.path import PathSampler


# ---------------------------------------------------------------------------
# Minimal mock objects (don't instantiate real DLPM for speed)
# ---------------------------------------------------------------------------

class MockGenA:
    """Deterministic mock with an internal counter so each .generate() call
    returns distinct values. Shape-aware."""
    def __init__(self, base=0.0):
        self.call_count = 0
        self.base = base

    def generate(self, size):
        # Return a deterministic tensor based on call_count; different shape OK.
        n = int(np.prod(size))
        out = torch.arange(n, dtype=torch.float32) + (self.call_count * 1000.0) + self.base
        self.call_count += 1
        return out.reshape(size)


class MockDLPM:
    """Minimal mock carrying only the attributes PathSampler touches."""
    def __init__(self, T=10, device='cpu'):
        # mock schedules (not used beyond .sigmas length for forward_process)
        self.sigmas = torch.ones(T) * 0.1
        self.gammas = torch.ones(T) * 0.99
        self.barsigmas = torch.ones(T) * 1.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_equivalence_forward():
    """G=1, M=1, forward_process: sampler's A[i] == direct gen_a reshape[i]."""
    gen = MockGenA()
    dlpm = MockDLPM(T=10)
    sampler = PathSampler(
        group_size=1, resample_interval=1,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='forward_process', device='cpu')
    A, phi = sampler.sample(batch_size=8)
    assert A.shape == (10, 8), f'expected [T=10, B=8], got {A.shape}'
    assert phi is None
    # Direct reference: a fresh gen_a.generate should produce same values
    # since sampler uses call_count=0 internally.
    gen_ref = MockGenA()
    A_flat_ref = gen_ref.generate(size=[10 * 8])
    A_ref = A_flat_ref.reshape(10, 8)
    torch.testing.assert_close(A, A_ref)
    print('  test_default_equivalence_forward: OK')


def test_default_equivalence_terminal():
    """G=1, M=1, terminal: sampler's A == direct gen_a.generate(size=[B, ...])."""
    gen = MockGenA()
    dlpm = MockDLPM()
    sampler = PathSampler(
        group_size=1, resample_interval=1,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='terminal', device='cpu')
    A, phi = sampler.sample(batch_size=8, data_shape_tail=(3, 4))
    assert A.shape == (8, 3, 4), f'expected [8,3,4], got {A.shape}'
    # Reference
    gen_ref = MockGenA()
    A_ref = gen_ref.generate(size=[8, 3, 4])
    torch.testing.assert_close(A, A_ref)
    print('  test_default_equivalence_terminal: OK')


def test_within_batch_grouping_forward():
    """G=4, B=16: A[:,0..3] identical, A[:,4..7] identical, etc."""
    gen = MockGenA()
    dlpm = MockDLPM(T=5)
    sampler = PathSampler(
        group_size=4, resample_interval=1,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='forward_process', device='cpu')
    A, _ = sampler.sample(batch_size=16)
    assert A.shape == (5, 16)
    for g in range(4):
        start = g * 4
        cols = [A[:, start + j] for j in range(4)]
        for j in range(1, 4):
            torch.testing.assert_close(cols[0], cols[j])
    # Cross-group should differ
    assert not torch.equal(A[:, 0], A[:, 4])
    print('  test_within_batch_grouping_forward: OK')


def test_within_batch_grouping_terminal():
    gen = MockGenA()
    dlpm = MockDLPM()
    sampler = PathSampler(
        group_size=3, resample_interval=1,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='terminal', device='cpu')
    A, _ = sampler.sample(batch_size=9, data_shape_tail=(2,))
    assert A.shape == (9, 2)
    for g in range(3):
        start = g * 3
        rows = [A[start + j] for j in range(3)]
        for j in range(1, 3):
            torch.testing.assert_close(rows[0], rows[j])
    assert not torch.equal(A[0], A[3])
    print('  test_within_batch_grouping_terminal: OK')


def test_nondivisible_raises():
    gen = MockGenA()
    dlpm = MockDLPM()
    sampler = PathSampler(
        group_size=4, resample_interval=1,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='forward_process', device='cpu')
    try:
        sampler.sample(batch_size=17)
        raise RuntimeError('expected AssertionError, none raised')
    except AssertionError as e:
        assert '17' in str(e) and '4' in str(e), f'message: {e}'
    print('  test_nondivisible_raises: OK')


def test_cache_resample_interval():
    """M=3: the same (A, phi) is returned for the first 3 calls; 4th is new."""
    gen = MockGenA()
    dlpm = MockDLPM()
    sampler = PathSampler(
        group_size=1, resample_interval=3,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='forward_process', device='cpu')
    A1, _ = sampler.sample(batch_size=4)
    A2, _ = sampler.sample(batch_size=4)
    A3, _ = sampler.sample(batch_size=4)
    A4, _ = sampler.sample(batch_size=4)
    torch.testing.assert_close(A1, A2)
    torch.testing.assert_close(A2, A3)
    assert not torch.equal(A3, A4), 'expected 4th call to resample'
    # Call count: 2 draws (first + 4th)
    assert gen.call_count == 2, f'expected 2 gen_a calls, got {gen.call_count}'
    print('  test_cache_resample_interval: OK')


def test_batch_size_change_invalidates_cache():
    """B change forces a fresh draw (and emits WARN to stdout)."""
    gen = MockGenA()
    dlpm = MockDLPM()
    sampler = PathSampler(
        group_size=1, resample_interval=5,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='forward_process', device='cpu')
    sampler.sample(batch_size=8)   # first draw
    sampler.sample(batch_size=8)   # cache hit
    assert gen.call_count == 1
    sampler.sample(batch_size=6)   # B changed → resample (+WARN)
    assert gen.call_count == 2, f'expected resample on B change, got {gen.call_count}'
    print('  test_batch_size_change_invalidates_cache: OK')


def test_reset_invalidates():
    gen = MockGenA()
    dlpm = MockDLPM()
    sampler = PathSampler(
        group_size=1, resample_interval=10,
        gen_a=gen, path_encoder=None,
        dlpm=dlpm, noise_type='forward_process', device='cpu')
    sampler.sample(batch_size=4)
    sampler.reset()
    sampler.sample(batch_size=4)
    assert gen.call_count == 2
    print('  test_reset_invalidates: OK')


if __name__ == '__main__':
    print('PathSampler tests:')
    test_default_equivalence_forward()
    test_default_equivalence_terminal()
    test_within_batch_grouping_forward()
    test_within_batch_grouping_terminal()
    test_nondivisible_raises()
    test_cache_resample_interval()
    test_batch_size_change_invalidates_cache()
    test_reset_invalidates()
    print('ALL PASS')
