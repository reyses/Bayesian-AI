import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import numba

def _compute_rolling_std_numpy(z_scores, n, window):
    osc_std = np.full(n, np.nan)
    if n >= window:
         z_windows = sliding_window_view(z_scores, window_shape=window)
         osc_std[window-1:] = z_windows.std(axis=1, ddof=1)
         if n > window - 1:
              osc_std[:window - 1] = osc_std[window - 1]
    return osc_std

@numba.njit(parallel=True, cache=True)
def _compute_rolling_std_numba(z_scores, n, window):
    osc_std = np.full(n, np.nan)
    if n >= window:
        for i in numba.prange(window - 1, n):
            # Compute mean
            sum_val = 0.0
            for j in range(i - window + 1, i + 1):
                sum_val += z_scores[j]
            mean_val = sum_val / window

            # Compute variance (ddof=1)
            sum_sq_diff = 0.0
            for j in range(i - window + 1, i + 1):
                diff = z_scores[j] - mean_val
                sum_sq_diff += diff * diff

            osc_std[i] = np.sqrt(sum_sq_diff / (window - 1))

        for i in range(window - 1):
            osc_std[i] = osc_std[window - 1]

    return osc_std

n = 20000
z_scores = np.random.randn(n)
window = 5

# Warmup Numba
_ = _compute_rolling_std_numba(z_scores, n, window)

t0 = time.perf_counter()
res_numpy = _compute_rolling_std_numpy(z_scores, n, window)
t1 = time.perf_counter()

t2 = time.perf_counter()
res_numba = _compute_rolling_std_numba(z_scores, n, window)
t3 = time.perf_counter()

print(f"NumPy: {t1-t0:.4f}s")
print(f"Numba: {t3-t2:.4f}s")
print(f"Equal: {np.allclose(res_numpy, res_numba, equal_nan=True)}")
