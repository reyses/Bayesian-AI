import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def old_swing_noise(highs, lows, n, _noise_window, _tick_size):
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

def test_swing_noise():
    n = 1000
    _noise_window = 32
    _tick_size = 0.25
    np.random.seed(42)
    highs = np.random.rand(n) * 100 + 4000
    lows = highs - np.random.rand(n) * 2

    out_old = old_swing_noise(highs, lows, n, _noise_window, _tick_size)
    out_new = _compute_swing_noise_numba(highs, lows, n, _noise_window, _tick_size)

    assert np.allclose(out_old, out_new)

if __name__ == '__main__':
    test_swing_noise()
