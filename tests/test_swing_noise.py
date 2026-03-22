import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba


def compute_swing_noise_python(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for _ni in range(noise_window, n):
        _seg_hi = highs[_ni - noise_window:_ni + 1]
        _seg_lo = lows[_ni - noise_window:_ni + 1]

        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / tick_size

        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / tick_size

        swing_noise[_ni] = max(_dd, _du)

    return swing_noise


def test_compute_swing_noise_numba_match():
    n = 1000
    np.random.seed(42)
    highs = np.random.rand(n) * 100 + 100
    lows = highs - np.random.rand(n) * 2
    noise_window = 32
    tick_size = 0.25

    res_py = compute_swing_noise_python(highs, lows, n, noise_window, tick_size)
    res_nb = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)

    assert np.allclose(res_py, res_nb, rtol=1e-4), "Numba output does not match Python output"


def test_compute_swing_noise_numba_performance():
    n = 100000
    np.random.seed(42)
    highs = np.random.rand(n) * 100 + 100
    lows = highs - np.random.rand(n) * 2
    noise_window = 32
    tick_size = 0.25

    # warmup
    _compute_swing_noise_numba(highs[:100], lows[:100], 100, noise_window, tick_size)

    t0 = time.perf_counter()
    compute_swing_noise_python(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t3 = time.perf_counter()

    py_time = t1 - t0
    nb_time = t3 - t2
    speedup = py_time / nb_time

    print(f"Python time: {py_time:.4f}s")
    print(f"Numba time:  {nb_time:.4f}s")
    print(f"Speedup:     {speedup:.2f}x")

    assert speedup > 10.0, f"Expected >10x speedup, got {speedup:.2f}x"
