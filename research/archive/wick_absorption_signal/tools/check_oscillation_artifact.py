import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import random

RAW_DATA_PATH = "DATA/ATLAS/order_flow_delta_5s.parquet"
OUTPUT_DIR = Path("research/wick_absorption_signal/reports/stage_1_profile")

print("Loading raw 5s data...")
df = pd.read_parquet(RAW_DATA_PATH)
df = df.sort_index()

# Let's check the 1-minute timeframe for the spot check
print("Resampling to 1min...")
df_1m = df.resample('1min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
}).dropna()

# Pick a random 300-bar slice (about 5 hours of 1m data)
slice_len = 300
start_idx = random.randint(0, len(df_1m) - slice_len)
df_slice = df_1m.iloc[start_idx : start_idx + slice_len].copy()

# Apply the exact same smoothing from stage 1
window = 11
smoothed = savgol_filter(df_slice['close'].values, window_length=window, polyorder=3)
df_slice['smoothed'] = smoothed

# Find peaks and troughs
peaks, _ = find_peaks(smoothed)
troughs, _ = find_peaks(-smoothed)

plt.figure(figsize=(15, 6))
plt.plot(df_slice.index, df_slice['close'], label='Raw Close', color='lightgray', alpha=0.8)
plt.plot(df_slice.index, df_slice['smoothed'], label=f'Savgol Smoothed (w={window}, poly=3)', color='blue', linewidth=2)

plt.scatter(df_slice.index[peaks], df_slice['smoothed'].iloc[peaks], color='red', s=50, zorder=5, label='Peaks')
plt.scatter(df_slice.index[troughs], df_slice['smoothed'].iloc[troughs], color='green', s=50, zorder=5, label='Troughs')

plt.title(f"Spot Check: Oscillation Artifact? (1m TF, Window={window})")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)

out_path = OUTPUT_DIR / "spot_check_oscillation.png"
plt.tight_layout()
plt.savefig(out_path)
plt.close()

print(f"Spot check plot saved to {out_path}")
