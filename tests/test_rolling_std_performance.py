import numpy as np
import time
from core.quantum_field_engine import _compute_rolling_std_numba
from numpy.lib.stride_tricks import sliding_window_view
import pytest

def calc_sliding(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    if n >= window:
        windows = sliding_window_view(arr, window_shape=window)
        out[window-1:] = windows.std(axis=1, ddof=1)
        if n > window - 1:
            out[:window - 1] = out[window - 1]
    return out

def test_rolling_std_correctness():
    arr = np.random.randn(1000)
    window = 5
    res1 = calc_sliding(arr, window)
    res2 = _compute_rolling_std_numba(arr, window)
    np.testing.assert_allclose(res1, res2, equal_nan=True)

def test_rolling_std_performance():
    n = 100000
    arr = np.random.randn(n)
    window = 5

    # Warmup
    _compute_rolling_std_numba(arr[:10], window)

    t0 = time.perf_counter()
    calc_sliding(arr, window)
    t1 = time.perf_counter()
    time_sliding = t1 - t0

    t0 = time.perf_counter()
    _compute_rolling_std_numba(arr, window)
    t1 = time.perf_counter()
    time_numba = t1 - t0

    print(f"\nsliding_window_view: {time_sliding:.5f}s")
    print(f"numba: {time_numba:.5f}s")

    assert time_numba < time_sliding, "Numba implementation should be faster"
