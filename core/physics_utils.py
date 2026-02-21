"""
Physics Utilities (CPU-based)
Shared physics functions and constants that do not require CUDA.
"""
import numpy as np

# Indicator Constants
ADX_PERIOD = 14
HURST_WINDOW = 100
HURST_MIN_WINDOW = 30
REL_VOLUME_WINDOW = 20
OSCILLATION_COHERENCE_WINDOW = 5

def compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, period=14):
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
