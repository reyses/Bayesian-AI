import time
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from core.statistical_field_engine import _compute_rolling_std_numba

def _compute_rolling_std_numpy(z_scores, n, window):
    osc_std = np.full(n, np.nan)
    if n >= window:
         z_windows = sliding_window_view(z_scores, window_shape=window)
         osc_std[window-1:] = z_windows.std(axis=1, ddof=1)
         if n > window - 1:
              osc_std[:window - 1] = osc_std[window - 1]
    return osc_std

def test_rolling_std_numba():
    n = 20000
    np.random.seed(42)
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

    np.testing.assert_allclose(res_numpy, res_numba, equal_nan=True, rtol=1e-4)
