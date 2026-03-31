import numpy as np
import time
import pytest
from core.statistical_field_engine import _compute_swing_noise_numba

def swing_noise_original(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for _ni in range(noise_window, n):
        _seg_hi = highs[_ni - noise_window:_ni + 1]
        _seg_lo = lows[_ni - noise_window:_ni + 1]
        # Max drawdown from running high (long-side noise)
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / tick_size
        # Max drawup from running low (short-side noise)
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise

def test_swing_noise_numba_optimization():
    np.random.seed(42)
    n = 10000
    noise_window = 30
    tick_size = 0.25

    # Generate some realistic looking price data
    price_changes = np.random.randn(n) * 2
    prices = 4000 + np.cumsum(price_changes)

    highs = prices + np.abs(np.random.randn(n)) * 5
    lows = prices - np.abs(np.random.randn(n)) * 5

    # Warmup Numba
    _ = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)

    # Time original
    start = time.perf_counter()
    orig_result = swing_noise_original(highs, lows, n, noise_window, tick_size)
    orig_time = time.perf_counter() - start

    # Time numba
    start = time.perf_counter()
    numba_result = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    numba_time = time.perf_counter() - start

    print(f"\nOriginal time: {orig_time:.5f}s")
    print(f"Numba time: {numba_time:.5f}s")
    print(f"Speedup: {orig_time / numba_time:.2f}x")

    # Verify exact parity
    assert np.allclose(orig_result, numba_result, rtol=1e-4)

if __name__ == '__main__':
    test_swing_noise_numba_optimization()
