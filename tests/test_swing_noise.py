import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba


def compute_swing_noise_numpy(highs, lows, noise_window, tick_size):
    n = len(highs)
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


def test_swing_noise_numba_exact_match():
    # Setup random price data
    np.random.seed(42)
    n = 1000
    highs = np.random.rand(n) * 100 + 1000
    lows = highs - np.random.rand(n) * 10
    noise_window = 30
    tick_size = 0.25

    # Original implementation output
    res_numpy = compute_swing_noise_numpy(highs, lows, noise_window, tick_size)

    # Numba implementation output
    res_numba = _compute_swing_noise_numba(highs, lows, noise_window, tick_size)

    # They should match exactly (floating point safe)
    assert np.allclose(res_numba, res_numpy, rtol=1e-4)


def test_swing_noise_numba_performance():
    # Setup large array
    np.random.seed(42)
    n = 10000
    highs = np.random.rand(n) * 100 + 1000
    lows = highs - np.random.rand(n) * 10
    noise_window = 30
    tick_size = 0.25

    # Warmup Numba JIT compilation
    _compute_swing_noise_numba(highs, lows, noise_window, tick_size)

    # Time NumPy
    t0 = time.perf_counter()
    compute_swing_noise_numpy(highs, lows, noise_window, tick_size)
    t1 = time.perf_counter()
    numpy_time = t1 - t0

    # Time Numba
    t0 = time.perf_counter()
    _compute_swing_noise_numba(highs, lows, noise_window, tick_size)
    t1 = time.perf_counter()
    numba_time = t1 - t0

    print(f"\nNumPy time: {numpy_time:.6f}s")
    print(f"Numba time: {numba_time:.6f}s")
    print(f"Speedup: {numpy_time / numba_time:.2f}x")

    # Assert Numba is significantly faster (at least 10x)
    assert numba_time < numpy_time / 10


def test_swing_noise_empty_short():
    noise_window = 30
    tick_size = 0.25

    # Short array
    highs = np.array([100.0, 101.0])
    lows = np.array([99.0, 100.0])

    res_numba = _compute_swing_noise_numba(highs, lows, noise_window, tick_size)
    res_numpy = compute_swing_noise_numpy(highs, lows, noise_window, tick_size)

    assert np.allclose(res_numba, res_numpy, rtol=1e-4)
    assert len(res_numba) == 2
    assert res_numba[0] == 35.0
    assert res_numba[1] == 35.0
