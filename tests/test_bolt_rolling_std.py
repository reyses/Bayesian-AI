import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import numba

def original_rolling_std(z_scores, _ow):
    n = len(z_scores)
    osc_std = np.full(n, np.nan)
    if n >= _ow:
         z_windows = sliding_window_view(z_scores, window_shape=_ow)
         osc_std[_ow-1:] = z_windows.std(axis=1, ddof=1)
         if n > _ow - 1:
              osc_std[:_ow - 1] = osc_std[_ow - 1]
    return osc_std


@numba.njit(parallel=True, cache=True)
def _compute_rolling_std_numba(z_scores, _ow):
    n = len(z_scores)
    osc_std = np.full(n, np.nan)
    if n >= _ow:
        # Loop starting from the first fully populated window
        for i in numba.prange(_ow - 1, n):
            sum_x = 0.0
            sum_x2 = 0.0
            for j in range(i - _ow + 1, i + 1):
                val = z_scores[j]
                sum_x += val
                sum_x2 += val * val

            mean = sum_x / _ow
            var = (sum_x2 - _ow * mean * mean) / (_ow - 1)
            # handle negative float accuracy issues around 0
            if var < 0:
                var = 0.0
            osc_std[i] = np.sqrt(var)

        # Fill the front padding
        front_val = osc_std[_ow - 1]
        for i in range(_ow - 1):
            osc_std[i] = front_val

    return osc_std

def test_rolling_std():
    n = 1000000
    z_scores = np.random.randn(n)
    _ow = 5

    # warmup
    _compute_rolling_std_numba(z_scores[:100], _ow)

    t0 = time.perf_counter()
    out1 = original_rolling_std(z_scores, _ow)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    out2 = _compute_rolling_std_numba(z_scores, _ow)
    t3 = time.perf_counter()

    print(f"Original: {t1 - t0:.4f}s")
    print(f"Numba: {t3 - t2:.4f}s")

    np.testing.assert_allclose(out1[4:], out2[4:], rtol=1e-4, atol=1e-6)
    print("Test passed!")

if __name__ == "__main__":
    test_rolling_std()
