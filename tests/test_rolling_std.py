import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from core.statistical_field_engine import _compute_rolling_std_numba

def old_rolling_std(z_scores, n, _ow):
    osc_std = np.full(n, np.nan)
    if n >= _ow:
        z_windows = sliding_window_view(z_scores, window_shape=_ow)
        osc_std[_ow-1:] = z_windows.std(axis=1, ddof=1)
        if n > _ow - 1:
            osc_std[:_ow - 1] = osc_std[_ow - 1]
    return osc_std

def test_rolling_std():
    n = 1000
    _ow = 5
    np.random.seed(42)
    z_scores = np.random.randn(n) * 2

    out_old = old_rolling_std(z_scores, n, _ow)
    out_new = _compute_rolling_std_numba(z_scores, n, _ow)

    assert np.allclose(out_old, out_new, equal_nan=True)

if __name__ == '__main__':
    test_rolling_std()
