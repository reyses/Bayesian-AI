import numpy as np
import time
import pytest
from core.statistical_field_engine import _compute_swing_noise_numba

def compute_swing_noise_python(highs, lows, noise_window, tick_size, n):
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
    # Generate test data
    np.random.seed(42)
    n = 10000
    highs = np.random.rand(n) * 100 + 1000
    lows = highs - np.random.rand(n) * 5
    noise_window = 30
    tick_size = 0.25

    # 1. Warmup Numba JIT
    _compute_swing_noise_numba(highs[:100], lows[:100], noise_window, tick_size)

    # 2. Benchmark Python (NumPy) version
    t0 = time.perf_counter()
    res_python = compute_swing_noise_python(highs, lows, noise_window, tick_size, n)
    t1 = time.perf_counter()
    python_time = t1 - t0

    # 3. Benchmark Numba version
    t0 = time.perf_counter()
    res_numba = _compute_swing_noise_numba(highs, lows, noise_window, tick_size)
    t1 = time.perf_counter()
    numba_time = t1 - t0

    # 4. Verification
    print(f"\nPython Time: {python_time:.4f}s")
    print(f"Numba Time:  {numba_time:.4f}s")
    if numba_time > 0:
        print(f"Speedup:     {python_time / numba_time:.2f}x")

    assert np.allclose(res_python, res_numba, rtol=1e-4), "Numerical output mismatch between Numba and Python implementations"

if __name__ == "__main__":
    test_swing_noise_numba_optimization()
