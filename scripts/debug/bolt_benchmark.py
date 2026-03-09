import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit, prange

@njit(parallel=True, cache=True)
def _compute_rolling_std_numba(arr, window, ddof=1):
    n = len(arr)
    out = np.full(n, np.nan)
    if n < window:
        return out

    for i in prange(window - 1, n):
        sum_x = 0.0
        for j in range(i - window + 1, i + 1):
            sum_x += arr[j]
        mean = sum_x / window

        sum_sq = 0.0
        for j in range(i - window + 1, i + 1):
            diff = arr[j] - mean
            sum_sq += diff * diff

        out[i] = np.sqrt(sum_sq / (window - ddof))

    first_val = out[window - 1]
    for i in range(window - 1):
        out[i] = first_val

    return out

def compute_rolling_std_old(z_scores, _ow):
    n = len(z_scores)
    osc_std = np.full(n, np.nan)
    if n >= _ow:
         z_windows = sliding_window_view(z_scores, window_shape=_ow)
         osc_std[_ow-1:] = z_windows.std(axis=1, ddof=1)
         if n > _ow - 1:
              osc_std[:_ow - 1] = osc_std[_ow - 1]
    return osc_std

def run_bench():
    # Warm up numba
    z_scores_small = np.random.randn(100)
    _compute_rolling_std_numba(z_scores_small, 5)

    n = 1_000_000
    z_scores = np.random.randn(n)
    _ow = 5

    t0 = time.perf_counter()
    old_res = compute_rolling_std_old(z_scores, _ow)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    new_res = _compute_rolling_std_numba(z_scores, _ow)
    t3 = time.perf_counter()

    print(f"Old: {t1-t0:.4f}s")
    print(f"New: {t3-t2:.4f}s")
    print(f"Speedup: {(t1-t0)/(t3-t2):.2f}x")
    print(f"Allclose: {np.allclose(old_res, new_res, equal_nan=True)}")

run_bench()
