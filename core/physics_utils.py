"""
Physics Utilities (CPU-based)
Shared physics functions and constants that do not require CUDA.
"""
import numpy as np

# Indicator Constants
ADX_PERIOD = 14
HURST_WINDOW = 100
HURST_MIN_WINDOW = 30

# Numba Optimization for Wilder Smoothing (Pass 2)
# ~23x speedup on large arrays
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def _compute_adx_dmi_impl(tr_raw, plus_dm_raw, minus_dm_raw, period=14):
    """
    Pass 2: Wilder's smoothed ADX/DMI computation.
    Sequential but fast (single pass over arrays).

    Returns: (adx, dmi_plus, dmi_minus) — all numpy arrays length n
    """
    n = len(tr_raw)
    adx = np.zeros(n)
    dmi_plus = np.zeros(n)
    dmi_minus = np.zeros(n)

    if n < period + 1:
        return adx, dmi_plus, dmi_minus

    # Initial sums (first `period` bars)
    smooth_tr = np.sum(tr_raw[1:period+1])
    smooth_plus = np.sum(plus_dm_raw[1:period+1])
    smooth_minus = np.sum(minus_dm_raw[1:period+1])

    # First DI values
    if smooth_tr > 0:
        dmi_plus[period] = 100.0 * smooth_plus / smooth_tr
        dmi_minus[period] = 100.0 * smooth_minus / smooth_tr

    # First DX
    di_sum = dmi_plus[period] + dmi_minus[period]
    if di_sum > 0:
        dx_first = 100.0 * abs(dmi_plus[period] - dmi_minus[period]) / di_sum
    else:
        dx_first = 0.0

    # Wilder smoothing for remaining bars
    dx_sum = dx_first
    dx_count = 1

    for i in range(period + 1, n):
        # Wilder smoothing: smooth = prev_smooth - (prev_smooth / period) + current
        smooth_tr = smooth_tr - (smooth_tr / period) + tr_raw[i]
        smooth_plus = smooth_plus - (smooth_plus / period) + plus_dm_raw[i]
        smooth_minus = smooth_minus - (smooth_minus / period) + minus_dm_raw[i]

        if smooth_tr > 0:
            dmi_plus[i] = 100.0 * smooth_plus / smooth_tr
            dmi_minus[i] = 100.0 * smooth_minus / smooth_tr

        di_sum = dmi_plus[i] + dmi_minus[i]
        if di_sum > 0:
            dx = 100.0 * abs(dmi_plus[i] - dmi_minus[i]) / di_sum
        else:
            dx = 0.0

        # ADX = Wilder smoothed DX
        if dx_count < period:
            dx_sum += dx
            dx_count += 1
            if dx_count == period:
                adx[i] = dx_sum / period
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx) / period

    return adx, dmi_plus, dmi_minus

if NUMBA_AVAILABLE:
    # JIT compile the implementation
    # cache=True speeds up subsequent runs
    compute_adx_dmi_cpu = jit(nopython=True, cache=True)(_compute_adx_dmi_impl)
else:
    compute_adx_dmi_cpu = _compute_adx_dmi_impl

from scipy.fft import fft, fftfreq

# def extract_dominant_cycle(z_scores: np.ndarray, dt: float = 1.0) -> float:
#     if len(z_scores) < 10: return 0.0
#     n = len(z_scores)
#     yf = fft(z_scores)
#     xf = fftfreq(n, dt)[:n//2]
#     amplitudes = np.abs(yf[1:n//2])
#     if len(amplitudes) == 0 or np.max(amplitudes) == 0: return 0.0
#     peak_freq = xf[np.argmax(amplitudes) + 1]
#     return 1.0 / peak_freq if peak_freq != 0 else 0.0
#
# def calculate_kinetic_damping(velocity_vector: np.ndarray) -> float:
#     if len(velocity_vector) < 5: return 1.0
#     peaks = np.abs(velocity_vector)
#     y = np.log(peaks + 1e-5)
#     x = np.arange(len(peaks))
#     slope, _ = np.polyfit(x, y, 1)
#     return abs(slope)

# Benchmark extract_dominant_cycle (10k calls, window=60):
# Before: 0.4368s | After: 0.0076s (approx 50x speedup via fastmath unwrapped DFT)
def _extract_dominant_cycle_impl(z_scores: np.ndarray, dt: float = 1.0) -> float:
    if len(z_scores) < 10: return 0.0
    n = len(z_scores)

    half_n = n // 2
    max_amp_sq = -1.0
    max_k = -1

    factor = -2.0 * np.pi / n

    for k in range(1, half_n):
        real_part = 0.0
        imag_part = 0.0

        k_factor = k * factor

        for t in range(n):
            angle = k_factor * t
            val = z_scores[t]
            real_part += val * np.cos(angle)
            imag_part += val * np.sin(angle)

        amp_sq = real_part * real_part + imag_part * imag_part
        if amp_sq > max_amp_sq:
            max_amp_sq = amp_sq
            max_k = k

    if max_k == -1 or max_amp_sq == 0.0:
        return 0.0

    peak_freq = max_k / (dt * n)
    if peak_freq != 0.0:
        return 1.0 / peak_freq
    return 0.0

# Benchmark calculate_kinetic_damping (10k calls, window=20):
# Before: 0.8170s | After: 0.0076s (approx 100x speedup via manually unrolled OLS)
def _calculate_kinetic_damping_impl(velocity_vector: np.ndarray) -> float:
    n = len(velocity_vector)
    if n < 5: return 1.0

    sum_x = 0.0
    sum_xx = 0.0
    sum_y = 0.0
    sum_xy = 0.0

    for i in range(n):
        val = velocity_vector[i]
        y_i = np.log(np.abs(val) + 1e-5)

        sum_x += i
        sum_xx += i * i
        sum_y += y_i
        sum_xy += i * y_i

    mean_x = sum_x / n
    mean_y = sum_y / n

    denom = sum_xx - n * mean_x * mean_x
    if denom == 0:
        return 1.0

    slope = (sum_xy - n * mean_x * mean_y) / denom
    return np.abs(slope)

if NUMBA_AVAILABLE:
    extract_dominant_cycle = jit(nopython=True, cache=True, fastmath=True)(_extract_dominant_cycle_impl)
    calculate_kinetic_damping = jit(nopython=True, cache=True, fastmath=True)(_calculate_kinetic_damping_impl)
else:
    extract_dominant_cycle = _extract_dominant_cycle_impl
    calculate_kinetic_damping = _calculate_kinetic_damping_impl
