import numpy as np
import time
from numba import njit, prange
import math
import pandas as pd
from training.orchestrator_worker import _extract_arrays_from_df

@njit(cache=True, fastmath=True)
def _extract_dominant_cycle_numba(z_scores, dt):
    n = len(z_scores)
    if n < 10: return 0.0

    n_half = n // 2
    max_amp2 = 0.0
    best_freq_idx = 0

    base_factor = -2.0 * math.pi / n

    for k in range(1, n_half):
        re = 0.0
        im = 0.0
        factor = base_factor * k
        for t in range(n):
            angle = factor * t
            re += z_scores[t] * math.cos(angle)
            im += z_scores[t] * math.sin(angle)

        amp2 = re*re + im*im
        if amp2 > max_amp2:
            max_amp2 = amp2
            best_freq_idx = k

    if max_amp2 == 0.0: return 0.0

    peak_freq = best_freq_idx / (n * dt)
    return 1.0 / peak_freq if peak_freq != 0 else 0.0

@njit(parallel=True, cache=True, fastmath=True)
def process_periods_dampings(z_scores, velocities, Z_SCORE_CYCLE_WINDOW, VELOCITY_DAMPING_WINDOW, dt):
    n = len(z_scores)
    periods = np.zeros(n)
    dampings = np.zeros(n)
    for i in prange(10, n):
        z_start = i - Z_SCORE_CYCLE_WINDOW
        if z_start < 0: z_start = 0
        w_z = z_scores[z_start:i]
        periods[i] = _extract_dominant_cycle_numba(w_z, dt)

        v_start = i - VELOCITY_DAMPING_WINDOW
        if v_start < 0: v_start = 0
        w_v = velocities[v_start:i]

        n_v = len(w_v)
        if n_v < 5:
            dampings[i] = 1.0
            continue

        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0

        for j in range(n_v):
            x = float(j)
            y = math.log(abs(w_v[j]) + 1e-5)
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_xx += x * x

        denom = n_v * sum_xx - sum_x * sum_x
        if denom == 0:
            dampings[i] = 1.0
        else:
            slope = (n_v * sum_xy - sum_x * sum_y) / denom
            dampings[i] = abs(slope)

    return periods, dampings

def _extract_arrays_from_df_new(df):
    prices = None
    timestamps = None

    if 'price' in df.columns:
        prices = df['price'].values
    elif 'close' in df.columns:
        prices = df['close'].values
    else:
        return None

    if 'timestamp' in df.columns:
        ts_data = df['timestamp'].values
        if ts_data.dtype.type == np.datetime64:
             timestamps = ts_data.astype('int64') / 1e9
        else:
             timestamps = ts_data.astype(np.float64)
    else:
        return None

    n = len(prices)
    periods = np.zeros(n)
    dampings = np.zeros(n)

    if 'z_score' in df.columns and 'velocity' in df.columns:
        z_scores = df['z_score'].values
        velocities = df['velocity'].values

        dt = 1.0
        if timestamps is not None and len(timestamps) > 1:
            diffs = np.diff(timestamps)
            dt = float(np.median(diffs))
            if dt <= 0: dt = 1.0

        periods, dampings = process_periods_dampings(z_scores, velocities, 60, 20, dt)

    return prices, timestamps, periods, dampings

n = 1000
df = pd.DataFrame({
    'price': np.random.randn(n),
    'timestamp': np.arange(n) * 1e9, # ns
    'z_score': np.random.randn(n),
    'velocity': np.random.randn(n)
})

# Warmup new function
_ = _extract_arrays_from_df_new(df)

t0 = time.perf_counter()
for i in range(100):
    _ = _extract_arrays_from_df_new(df)
t1 = time.perf_counter()

print(f"New _extract_arrays_from_df time: {t1-t0:.4f}")
