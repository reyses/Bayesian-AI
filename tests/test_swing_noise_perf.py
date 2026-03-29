import numpy as np
import time
from core.statistical_field_engine import _compute_swing_noise_numba

def compute_swing_noise_cpu(highs, lows, noise_window, tick_size):
    n = len(highs)
    swing_noise = np.full(n, 35.0)
    for _ni in range(noise_window, n):
        _seg_hi = highs[_ni - noise_window:_ni + 1]
        _seg_lo = lows[_ni - noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise

def test_swing_noise_perf_and_parity():
    # 10,000 bars is roughly 1-2 years of 1H data, or a good chunk of 1m data
    n = 10000
    np.random.seed(42)
    # create synthetic realistic data with random walk
    moves = np.random.randn(n) * 10
    closes = 4000.0 + np.cumsum(moves)
    highs = closes + np.random.rand(n) * 5
    lows = closes - np.random.rand(n) * 5

    noise_window = 30
    tick_size = 0.25

    # Warm up numba
    _ = _compute_swing_noise_numba(highs[:100], lows[:100], noise_window, tick_size)

    start = time.perf_counter()
    res_cpu = compute_swing_noise_cpu(highs, lows, noise_window, tick_size)
    cpu_time = time.perf_counter() - start

    start = time.perf_counter()
    res_numba = _compute_swing_noise_numba(highs, lows, noise_window, tick_size)
    numba_time = time.perf_counter() - start

    print(f"\n[Swing Noise Perf Benchmark]")
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"Numba time: {numba_time:.4f}s")
    print(f"Speedup: {cpu_time / numba_time:.2f}x")

    # Assert numerical equivalence
    assert np.allclose(res_cpu, res_numba, rtol=1e-4), "Outputs do not match!"
    print("Numerical output matches exactly.")
