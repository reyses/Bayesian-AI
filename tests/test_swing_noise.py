import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def compute_swing_noise_cpu(highs, lows, n, noise_window, tick_size):
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

def test_swing_noise_numba_correctness_and_speed():
    # Setup test data
    n = 100000
    np.random.seed(42)
    # create price trace
    highs = np.random.rand(n) * 100 + 100
    lows = highs - np.random.rand(n) * 5

    tick_size = 0.25
    noise_window = 32

    # 1. CPU implementation (Baseline)
    t0 = time.perf_counter()
    expected = compute_swing_noise_cpu(highs, lows, n, noise_window, tick_size)
    t_cpu = time.perf_counter() - t0

    # Warmup Numba (compile time)
    _ = _compute_swing_noise_numba(highs[:100], lows[:100], 100, noise_window, tick_size)

    # 2. Numba implementation (Optimized)
    t1 = time.perf_counter()
    actual = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t_numba = time.perf_counter() - t1

    print(f"\\nCPU time: {t_cpu:.4f}s")
    print(f"Numba time: {t_numba:.4f}s")
    print(f"Speedup: {t_cpu / t_numba:.2f}x")

    # Assert correctness
    assert np.allclose(expected, actual, rtol=1e-4), "Numba output does not match CPU output"
