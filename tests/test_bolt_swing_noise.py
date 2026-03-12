import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def test_compute_swing_noise_numba_correctness_and_speed():
    n = 20000
    np.random.seed(42)
    highs = np.random.rand(n) * 100
    lows = highs - np.random.rand(n) * 2
    _noise_window = 32
    _tick_size = 0.25

    # Original implementation for correctness check
    t0 = time.perf_counter()
    expected = np.full(n, 35.0)
    for _ni in range(_noise_window, n):
        _seg_hi = highs[_ni - _noise_window:_ni + 1]
        _seg_lo = lows[_ni - _noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / _tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / _tick_size
        expected[_ni] = max(_dd, _du)
    t1 = time.perf_counter()
    time_orig = t1 - t0

    # Numba implementation
    _ = _compute_swing_noise_numba(highs, lows, _noise_window, _tick_size, n) # Warmup
    t0 = time.perf_counter()
    actual = _compute_swing_noise_numba(highs, lows, _noise_window, _tick_size, n)
    t1 = time.perf_counter()
    time_numba = t1 - t0

    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    print(f"\nSpeedup: {time_orig / time_numba:.1f}x (Orig: {time_orig:.4f}s -> Numba: {time_numba:.4f}s)")
