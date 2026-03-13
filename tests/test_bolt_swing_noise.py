import time
import numpy as np
import pytest
from core.statistical_field_engine import _compute_swing_noise_numba

def test_swing_noise_optimization():
    n = 100000
    _noise_window = 32
    _tick_size = 0.25
    np.random.seed(42)
    highs = np.random.rand(n) * 100 + 10000
    lows = highs - np.random.rand(n) * 10

    # Old implementation
    t0 = time.perf_counter()
    swing_noise_old = np.full(n, 35.0)
    for _ni in range(_noise_window, n):
        _seg_hi = highs[_ni - _noise_window:_ni + 1]
        _seg_lo = lows[_ni - _noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / _tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / _tick_size
        swing_noise_old[_ni] = max(_dd, _du)
    t1 = time.perf_counter()

    # Compile run
    _compute_swing_noise_numba(highs, lows, n, _noise_window, _tick_size)

    t2 = time.perf_counter()
    swing_noise_new = _compute_swing_noise_numba(highs, lows, n, _noise_window, _tick_size)
    t3 = time.perf_counter()

    assert np.allclose(swing_noise_old, swing_noise_new)
