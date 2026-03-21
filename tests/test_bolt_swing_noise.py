import time
import numpy as np
import pytest
from core.statistical_field_engine import _compute_swing_noise_numba

def _compute_swing_noise_numpy(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)  # default 35 ticks
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

def test_swing_noise_numba():
    n = 20000
    np.random.seed(42)
    highs = np.random.rand(n) * 100 - 50
    lows = highs - np.random.rand(n) * 5
    noise_window = 32
    tick_size = 0.25

    # Warmup Numba
    _ = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)

    t0 = time.perf_counter()
    res_numpy = _compute_swing_noise_numpy(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    res_numba = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t3 = time.perf_counter()

    print(f"NumPy: {t1-t0:.4f}s")
    print(f"Numba: {t3-t2:.4f}s")

    np.testing.assert_allclose(res_numpy, res_numba, rtol=1e-4)
