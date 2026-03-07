import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from core.quantum_field_engine import _compute_rolling_std_numba

def test_rolling_std_performance_and_accuracy():
    print("Testing Numba parallel rolling standard deviation vs sliding_window_view...")
    np.random.seed(42)

    # Simulate a day's worth of 15s bar data (~5760 bars)
    n = 200_000
    z_scores = np.random.randn(n)
    _ow = 5

    # Ensure Numba function compiles first before benchmarking
    _ = _compute_rolling_std_numba(z_scores[:100], _ow)

    # Benchmarking sliding_window_view (original approach)
    t0 = time.perf_counter()
    z_windows = sliding_window_view(z_scores, window_shape=_ow)
    expected_std = z_windows.std(axis=1, ddof=1)
    t_old = time.perf_counter() - t0
    print(f"Original sliding_window_view Time: {t_old:.4f}s")

    # Benchmarking Numba parallel
    t1 = time.perf_counter()
    actual_std = _compute_rolling_std_numba(z_scores, _ow)
    t_new = time.perf_counter() - t1
    print(f"New _compute_rolling_std_numba Time: {t_new:.4f}s")

    # Calculate Speedup
    speedup = t_old / t_new if t_new > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")

    # Validate numerical accuracy
    assert np.allclose(actual_std, expected_std, rtol=1e-4), "Mismatch in calculated standard deviation"
    print("Numerical output matches exact values (np.allclose = True).")

if __name__ == '__main__':
    test_rolling_std_performance_and_accuracy()
