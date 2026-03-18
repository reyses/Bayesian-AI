import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from core.statistical_field_engine import _compute_rolling_std_numba, _compute_swing_noise_numba


def original_osc_std(z_scores, _ow):
    n = len(z_scores)
    osc_std = np.full(n, np.nan)
    if n >= _ow:
        z_windows = sliding_window_view(z_scores, window_shape=_ow)
        osc_std[_ow-1:] = z_windows.std(axis=1, ddof=1)
        if n > _ow - 1:
            osc_std[:_ow - 1] = osc_std[_ow - 1]
    return osc_std


def original_swing_noise(highs, lows, _noise_window, _tick_size):
    n = len(highs)
    swing_noise = np.full(n, 35.0)  # default 35 ticks
    for _ni in range(_noise_window, n):
        _seg_hi = highs[_ni - _noise_window:_ni + 1]
        _seg_lo = lows[_ni - _noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / _tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / _tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise


def test_rolling_std_performance():
    n = 100000
    np.random.seed(42)
    z_scores = np.random.randn(n)

    # Warmup
    _compute_rolling_std_numba(z_scores[:100], 5)

    t0 = time.perf_counter()
    std1 = original_osc_std(z_scores, 5)
    t1 = time.perf_counter()
    std2 = _compute_rolling_std_numba(z_scores, 5)
    t2 = time.perf_counter()

    orig_time = t1 - t0
    numba_time = t2 - t1

    assert np.allclose(std1, std2, equal_nan=True), "Outputs do not match for osc_std"
    print(f"\\nRolling std - Original: {orig_time:.4f}s, Numba: {numba_time:.4f}s")
    assert numba_time < orig_time, "Numba optimization did not speed up osc_std"


def test_swing_noise_performance():
    n = 100000
    np.random.seed(42)
    highs = np.random.rand(n) * 100 + 100
    lows = highs - np.random.rand(n) * 5

    # Warmup
    _compute_swing_noise_numba(highs[:100], lows[:100], 32, 0.25)

    t0 = time.perf_counter()
    sn1 = original_swing_noise(highs, lows, 32, 0.25)
    t1 = time.perf_counter()
    sn2 = _compute_swing_noise_numba(highs, lows, 32, 0.25)
    t2 = time.perf_counter()

    orig_time = t1 - t0
    numba_time = t2 - t1

    assert np.allclose(sn1, sn2), "Outputs do not match for swing_noise"
    print(f"\\nSwing noise - Original: {orig_time:.4f}s, Numba: {numba_time:.4f}s")
    assert numba_time < orig_time, "Numba optimization did not speed up swing_noise"
