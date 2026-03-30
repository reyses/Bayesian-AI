import numpy as np
import time
import pytest
from core.statistical_field_engine import _compute_swing_noise_numba

def py_swing_noise(highs, lows, n, noise_window, tick_size):
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

def test_swing_noise_numerical_equivalence_and_speed():
    n = 10000
    noise_window = 30
    tick_size = 0.25

    # Generate realistic-looking price data
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.randn(n) * 0.5
    prices = base_price + np.cumsum(price_changes)

    highs = prices + np.random.rand(n) * 1.5
    lows = prices - np.random.rand(n) * 1.5

    # Warm up numba compilation
    _ = _compute_swing_noise_numba(highs, lows, 100, noise_window, tick_size)

    # Benchmark Python version
    t0 = time.perf_counter()
    py_result = py_swing_noise(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    py_time = t1 - t0

    # Benchmark Numba version
    t2 = time.perf_counter()
    nb_result = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t3 = time.perf_counter()
    nb_time = t3 - t2

    print(f"\\nPython Time: {py_time:.5f}s")
    print(f"Numba Time:  {nb_time:.5f}s")
    print(f"Speedup:     {py_time/nb_time:.2f}x")

    # Assert exact equivalence
    assert np.allclose(py_result, nb_result, rtol=1e-4), "Numba result does not match Python result"

if __name__ == "__main__":
    test_swing_noise_numerical_equivalence_and_speed()
