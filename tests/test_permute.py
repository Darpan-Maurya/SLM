"""The permutation must be a bijection, stable, and well-mixed."""
import numpy as np
import pytest

from slm.data.permute import epoch_seed, permute, permuted_range


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 9, 100, 1000, 4096, 65537, 999_983])
def test_is_a_bijection(n):
    out = permuted_range(0, n, n, seed=1234)
    assert out.min() >= 0 and out.max() < n
    assert len(np.unique(out)) == n, f"collisions at n={n}"


def test_scalar_and_vector_agree():
    n = 5000
    vec = permuted_range(0, n, n, seed=7)
    for i in (0, 1, 42, 4999):
        assert permute(i, n, 7) == vec[i]


def test_deterministic_across_calls():
    a = permuted_range(0, 1000, 1000, seed=99)
    b = permuted_range(0, 1000, 1000, seed=99)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_orders():
    a = permuted_range(0, 10_000, 10_000, seed=1)
    b = permuted_range(0, 10_000, 10_000, seed=2)
    assert (a != b).mean() > 0.9


def test_slices_are_consistent():
    """A window of the permutation must not depend on how it was sliced."""
    full = permuted_range(0, 1000, 1000, seed=5)
    assert np.array_equal(permuted_range(300, 400, 1000, seed=5), full[300:400])


def test_shuffle_is_not_identity_or_trivial():
    n = 100_000
    out = permuted_range(0, n, n, seed=3)
    idx = np.arange(n)
    assert (out == idx).mean() < 0.001                      # not identity
    assert abs(np.corrcoef(out, idx)[0, 1]) < 0.02          # not monotone-ish
    # uniform spread: each tenth of the domain lands ~uniformly across the range
    hist = np.histogram(out[: n // 10], bins=10, range=(0, n))[0]
    assert hist.min() > 0.8 * hist.mean()


def test_epoch_seeds_are_distinct():
    seeds = {epoch_seed(1337, e) for e in range(200)}
    assert len(seeds) == 200


def test_large_domain_is_cheap():
    """No allocation proportional to n: 1e12 must be as fast as 1e3."""
    out = permuted_range(0, 1024, 1_000_000_000_000, seed=11)
    assert len(np.unique(out)) == 1024
    assert out.max() < 1_000_000_000_000
