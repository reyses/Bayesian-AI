import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def swing_noise_python(highs, lows, n, noise_window, tick_size):
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

def test_swing_noise_optimization():
    np.random.seed(42)
    n = 100000
    highs = np.random.rand(n) * 100 + 100
    lows = highs - np.random.rand(n) * 5
    noise_window = 30
    tick_size = 0.25

    # Warmup Numba
    _ = _compute_swing_noise_numba(highs, lows, 100, noise_window, tick_size)

    # Time Python implementation
    t0 = time.perf_counter()
    res1 = swing_noise_python(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()

    # Time Numba implementation
    t2 = time.perf_counter()
    res2 = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t3 = time.perf_counter()

    py_time = t1 - t0
    nb_time = t3 - t2
    speedup = py_time / nb_time

    print(f"Python: {py_time:.4f}s")
    print(f"Numba:  {nb_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")

    assert np.allclose(res1, res2, rtol=1e-4), "Numerical drift detected!"

if __name__ == "__main__":
    test_swing_noise_optimization()
