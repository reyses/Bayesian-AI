import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from core.statistical_field_engine import _compute_rolling_std_numba

def test_rolling_std_numba_correctness_and_performance():
    # Setup test data
    np.random.seed(42)
    n = 10000
    z_scores = np.random.randn(n)
    window = 5

    # --- Original Implementation ---
    def original_rolling_std(z, w):
        n_len = len(z)
        osc_std = np.full(n_len, np.nan)
        if n_len >= w:
            z_windows = sliding_window_view(z, window_shape=w)
            osc_std[w-1:] = z_windows.std(axis=1, ddof=1)
            if n_len > w - 1:
                osc_std[:w - 1] = osc_std[w - 1]
        return osc_std

    # --- Warmup ---
    expected = original_rolling_std(z_scores, window)
    actual = _compute_rolling_std_numba(z_scores, window)

    # --- Correctness Check ---
    np.testing.assert_allclose(actual, expected, rtol=1e-4, equal_nan=True)

    # --- Performance Test ---
    iterations = 100

    t0 = time.perf_counter()
    for _ in range(iterations):
        original_rolling_std(z_scores, window)
    t1 = time.perf_counter()
    orig_time = t1 - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _compute_rolling_std_numba(z_scores, window)
    t1 = time.perf_counter()
    numba_time = t1 - t0

    print(f"\nRolling Std Benchmarks (n={n}, window={window}, iterations={iterations}):")
    print(f"Original sliding_window_view: {orig_time:.4f}s")
    print(f"Numba _compute_rolling_std_numba: {numba_time:.4f}s")
    if numba_time > 0:
        print(f"Speedup: {orig_time/numba_time:.1f}x")
