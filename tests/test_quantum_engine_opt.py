import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from core.quantum_field_engine import _compute_rolling_std_numba

def test_rolling_std_numba_correctness():
    arr = np.random.randn(200000)

    # Test typical window > 1
    window = 5
    z_windows = sliding_window_view(arr, window_shape=window)
    expected = z_windows.std(axis=1, ddof=1)
    actual = _compute_rolling_std_numba(arr, window)
    np.testing.assert_allclose(actual, expected)

    # Test window = 1 (edge case resulting in NaNs)
    window = 1
    z_windows = sliding_window_view(arr, window_shape=window)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected = z_windows.std(axis=1, ddof=1)

    actual = _compute_rolling_std_numba(arr, window)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
