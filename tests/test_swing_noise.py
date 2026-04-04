import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def swing_noise_old(highs, lows, n, _noise_window, _tick_size):
    swing_noise = np.full(n, 35.0)  # default 35 ticks
    for _ni in range(_noise_window, n):
        _seg_hi = highs[_ni - _noise_window:_ni + 1]
        _seg_lo = lows[_ni - _noise_window:_ni + 1]
        # Max drawdown from running high (long-side noise)
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / _tick_size
        # Max drawup from running low (short-side noise)
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / _tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise

def test_swing_noise_numba_numerical_parity():
    n = 1000
    np.random.seed(42)
    highs = np.random.rand(n) * 100
    lows = highs - np.random.rand(n) * 10
    _noise_window = 30
    _tick_size = 0.25

    t0 = time.perf_counter()
    res_old = swing_noise_old(highs, lows, n, _noise_window, _tick_size)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    res_new = _compute_swing_noise_numba(highs, lows, _noise_window, _tick_size)
    t3 = time.perf_counter()

    print(f"Old time: {t1-t0:.4f}s")
    print(f"New time: {t3-t2:.4f}s")

    assert np.allclose(res_old, res_new, rtol=1e-4), "Numerical drift detected in Numba swing noise computation"
