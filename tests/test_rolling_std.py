import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from core.quantum_field_engine import _compute_rolling_std_numba

def test_rolling_std_numba_correctness():
    np.random.seed(42)
    n = 1000
    z_scores = np.random.randn(n)
    _ow = 5

    # Original implementation
    osc_std_orig = np.full(n, np.nan)
    if n >= _ow:
        z_windows = sliding_window_view(z_scores, window_shape=_ow)
        osc_std_orig[_ow-1:] = z_windows.std(axis=1, ddof=1)
        if n > _ow - 1:
            osc_std_orig[:_ow - 1] = osc_std_orig[_ow - 1]

    # Numba implementation
    osc_std_numba = _compute_rolling_std_numba(z_scores, _ow)
    if n > _ow - 1:
        osc_std_numba[:_ow - 1] = osc_std_numba[_ow - 1]

    mask = ~np.isnan(osc_std_orig) & ~np.isnan(osc_std_numba)
    assert np.allclose(osc_std_orig[mask], osc_std_numba[mask]), "Numba implementation output differs from sliding_window_view"

def test_rolling_std_numba_performance():
    np.random.seed(42)
    n = 100000
    z_scores = np.random.randn(n)
    _ow = 5

    # Warmup Numba
    _compute_rolling_std_numba(np.random.randn(10), _ow)

    # Time original
    t0 = time.perf_counter()
    osc_std_orig = np.full(n, np.nan)
    if n >= _ow:
        z_windows = sliding_window_view(z_scores, window_shape=_ow)
        osc_std_orig[_ow-1:] = z_windows.std(axis=1, ddof=1)
        if n > _ow - 1:
            osc_std_orig[:_ow - 1] = osc_std_orig[_ow - 1]
    t1 = time.perf_counter()

    # Time Numba
    t2 = time.perf_counter()
    osc_std_numba = _compute_rolling_std_numba(z_scores, _ow)
    if n > _ow - 1:
        osc_std_numba[:_ow - 1] = osc_std_numba[_ow - 1]
    t3 = time.perf_counter()

    orig_time = t1 - t0
    numba_time = t3 - t2

    print(f"\\nsliding_window_view: {orig_time:.5f}s")
    print(f"numba: {numba_time:.5f}s")
    print(f"Speedup: {orig_time / numba_time:.2f}x")
