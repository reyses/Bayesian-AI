import pytest
import numpy as np
from core.statistical_field_engine import _compute_swing_noise_numba

def compute_swing_noise_orig(highs, lows, n, _noise_window, _tick_size):
    swing_noise = np.full(n, 35.0)
    for _ni in range(_noise_window, n):
        _seg_hi = highs[_ni - _noise_window:_ni + 1]
        _seg_lo = lows[_ni - _noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / _tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / _tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise

def test_swing_noise_numba():
    np.random.seed(42)
    n = 1000
    highs = np.random.rand(n) * 100 + 4000
    lows = highs - np.random.rand(n) * 10
    _noise_window = 32
    _tick_size = 0.25

    res_orig = compute_swing_noise_orig(highs, lows, n, _noise_window, _tick_size)
    res_numba = _compute_swing_noise_numba(highs, lows, n, _noise_window, _tick_size)

    assert np.allclose(res_orig, res_numba), "Numba implementation does not match original."
