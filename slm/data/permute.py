"""Constant-memory shuffling of an index space."""
from __future__ import annotations

import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_ROUNDS = 4
# splitmix64 constants; a strong avalanche in three multiply-xorshift steps
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised over uint64."""
    x = x & _MASK64
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def _half_bits(n: int) -> tuple[int, int]:
    """Smallest even bit width covering ``n``, split into two halves."""
    bits = max(2, int(n - 1).bit_length())
    if bits % 2:
        bits += 1
    return bits, bits // 2


def _feistel(x: np.ndarray, half: int, keys: np.ndarray) -> np.ndarray:
    mask = np.uint64((1 << half) - 1)
    h = np.uint64(half)
    left = (x >> h) & mask
    right = x & mask
    for k in keys:
        new_right = left ^ (_mix(right + k) & mask)
        left, right = right, new_right
    return ((left << h) | right) & _MASK64


def _round_keys(seed: int) -> np.ndarray:
    # Everything runs through 1-element arrays: uint64 arithmetic here is *intended* to wrap...
    s = np.array([seed], dtype=np.uint64) * _GOLDEN
    offsets = (np.arange(1, _ROUNDS + 1, dtype=np.uint64) * _GOLDEN)
    return _mix(s + offsets)


def permute(index: np.ndarray | int, n: int, seed: int) -> np.ndarray | int:
    """Map ``index`` in ``[0, n)`` to a shuffled position in ``[0, n)``."""
    scalar = np.isscalar(index)
    idx = np.atleast_1d(np.asarray(index, dtype=np.uint64))
    if n <= 1:
        return int(idx[0]) if scalar else idx.astype(np.int64)

    _, half = _half_bits(n)
    keys = _round_keys(seed)
    n64 = np.uint64(n)

    out = _feistel(idx, half, keys)
    # cycle walking: anything that fell outside [0, n) gets re-permuted
    for _ in range(96):
        bad = out >= n64
        if not bad.any():
            break
        out[bad] = _feistel(out[bad], half, keys)
    else:  # pragma: no cover - unreachable for a sane domain/range ratio
        raise RuntimeError("cycle walking failed to converge")

    result = out.astype(np.int64)
    return int(result[0]) if scalar else result


def permuted_range(start: int, stop: int, n: int, seed: int) -> np.ndarray:
    """``permute`` over ``[start, stop)`` - the loader's batch-fetch primitive."""
    return permute(np.arange(start, stop, dtype=np.uint64), n, seed)


def epoch_seed(base_seed: int, epoch: int) -> int:
    """Distinct, well-separated seed per epoch (avoids correlated orderings)."""
    base = np.array([base_seed], dtype=np.uint64) * _GOLDEN
    s = _mix(base + np.array([epoch + 1], dtype=np.uint64) * _M1)
    return int(s[0] & np.uint64(0x7FFFFFFFFFFFFFFF))
