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

def test_swing_noise_numerical_parity():
    n = 10000
    np.random.seed(42)
    highs = np.cumsum(np.random.randn(n)) + 1000
    lows = highs - np.random.rand(n) * 2

    noise_window = 30
    tick_size = 0.25

    # Warmup
    _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)

    t0 = time.perf_counter()
    res_py = compute_swing_noise_python(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    res_nb = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t2 = time.perf_counter()

    print(f"Python: {t1-t0:.4f}s")
    print(f"Numba:  {t2-t1:.4f}s")
    speedup = (t1-t0)/(t2-t1)
    print(f"Speedup: {speedup:.2f}x")

    assert np.allclose(res_py, res_nb, rtol=1e-4), "Numerical output mismatch between python and numba implementations"
    assert speedup > 1.0, f"Speedup is not > 1.0 (was {speedup:.2f})"
