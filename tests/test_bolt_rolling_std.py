import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
from core.physics_utils import _compute_rolling_std_numba

def test_rolling_std_numba_vs_numpy():
    np.random.seed(42)
    n = 10_000
    z_scores = np.random.randn(n)
    _ow = 5

    # Numpy approach
    start = time.perf_counter()
    osc_std_np = np.full(n, np.nan)
    if n >= _ow:
        z_windows = sliding_window_view(z_scores, window_shape=_ow)
        osc_std_np[_ow-1:] = z_windows.std(axis=1, ddof=1)
        if n > _ow - 1:
            osc_std_np[:_ow - 1] = osc_std_np[_ow - 1]
    np_time = time.perf_counter() - start

    # Numba approach
    start = time.perf_counter()
    osc_std_numba = _compute_rolling_std_numba(z_scores, _ow, ddof=1)
    if n > _ow - 1:
        osc_std_numba[:_ow - 1] = osc_std_numba[_ow - 1]
    numba_time = time.perf_counter() - start

    mask = ~np.isnan(osc_std_np)
    assert np.allclose(osc_std_np[mask], osc_std_numba[mask], rtol=1e-4), "Outputs do not match!"
    print(f"NumPy time: {np_time:.5f}s, Numba time: {numba_time:.5f}s. Validation successful.")

if __name__ == "__main__":
    test_rolling_std_numba_vs_numpy()
