import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def compute_swing_noise_numpy(highs, lows, n, noise_window, tick_size):
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
    n = 23400 # typical 1-second bars per day
    noise_window = 32
    tick_size = 0.25

    # generate random walk
    prices = np.cumsum(np.random.randn(n) * 0.1) + 100
    highs = prices + np.abs(np.random.randn(n) * 0.05)
    lows = prices - np.abs(np.random.randn(n) * 0.05)

    t0 = time.perf_counter()
    res_np = compute_swing_noise_numpy(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    np_time = t1 - t0

    # trigger JIT compile
    _compute_swing_noise_numba(highs[:100], lows[:100], 100, noise_window, tick_size)

    t0 = time.perf_counter()
    res_numba = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    numba_time = t1 - t0

    print(f"NumPy time: {np_time:.4f}s")
    print(f"Numba time: {numba_time:.4f}s")
    if numba_time > 0:
        print(f"Speedup: {np_time / numba_time:.2f}x")

    assert np.allclose(res_np, res_numba, rtol=1e-4), "Numba optimization output mismatch with pure python numpy output"
    print("Results match exactly.")

if __name__ == "__main__":
    test_swing_noise_optimization()
