import numpy as np
import time
from numba import njit, prange
import math
import pandas as pd
from scipy.fft import fft, fftfreq

def extract_dominant_cycle(z_scores: np.ndarray, dt: float = 1.0) -> float:
    if len(z_scores) < 10: return 0.0
    n = len(z_scores)
    yf = fft(z_scores)
    xf = fftfreq(n, dt)[:n//2]
    amplitudes = np.abs(yf[1:n//2])
    if len(amplitudes) == 0 or np.max(amplitudes) == 0: return 0.0
    peak_freq = xf[np.argmax(amplitudes) + 1]
    return 1.0 / peak_freq if peak_freq != 0 else 0.0

def calculate_kinetic_damping(velocity_vector: np.ndarray) -> float:
    if len(velocity_vector) < 5: return 1.0
    peaks = np.abs(velocity_vector)
    y = np.log(peaks + 1e-5)
    x = np.arange(len(peaks))
    slope, _ = np.polyfit(x, y, 1)
    return abs(slope)

def _extract_arrays_from_df_old(df):
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

        for i in range(10, n):
            w_z = z_scores[max(0, i - 60):i]
            w_v = velocities[max(0, i - 20):i]
            periods[i] = extract_dominant_cycle(w_z, dt=dt)
            dampings[i] = calculate_kinetic_damping(w_v)

    return prices, timestamps, periods, dampings

n = 1000
df = pd.DataFrame({
    'price': np.random.randn(n),
    'timestamp': np.arange(n) * 1e9, # ns
    'z_score': np.random.randn(n),
    'velocity': np.random.randn(n)
})

t0 = time.perf_counter()
for i in range(100):
    _ = _extract_arrays_from_df_old(df)
t1 = time.perf_counter()

print(f"Old _extract_arrays_from_df time: {t1-t0:.4f}")
