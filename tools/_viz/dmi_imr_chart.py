"""
DMI I-MR Chart — Statistical Process Control for DMI diff.

I chart:  DMI diff per bar + control limits (mean +/- 3*sigma)
MR chart: |DMI_diff[i] - DMI_diff[i-1]| = rate of change of DMI
          When MR spikes = regime change happening

3 panels:
  1. Price
  2. I chart: DMI diff with UCL/LCL (control limits)
  3. MR chart: DMI acceleration with UCL

Usage:
  python tools/dmi_imr_chart.py 2026-03-19
  python tools/dmi_imr_chart.py 2026-03-26
"""
import numpy as np, pandas as pd, gc, matplotlib, sys, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine

date = sys.argv[1] if len(sys.argv) > 1 else '2026-03-19'
start_hm = sys.argv[2] if len(sys.argv) > 2 else None
end_hm = sys.argv[3] if len(sys.argv) > 3 else None
WINDOW = 30  # rolling window for control limits

df = pd.read_parquet('DATA/ATLAS/1m/2026_03.parquet').sort_values('timestamp').reset_index(drop=True)
sfe = StatisticalFieldEngine()
states = sfe.batch_compute_states(df)

dmi_plus = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_plus', 0.0) for s in states])
dmi_minus = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_minus', 0.0) for s in states])
dmi_diff = dmi_plus - dmi_minus
del states, sfe; gc.collect()

prices = df['close'].values
timestamps = pd.to_datetime(df['timestamp'].values, unit='s', utc=True)

# Filter to day + optional time range
date_ts = pd.Timestamp(date).timestamp()
if start_hm and end_hm:
    sh, sm = map(int, start_hm.split(':'))
    eh, em = map(int, end_hm.split(':'))
    t_start = date_ts + sh * 3600 + sm * 60
    t_end = date_ts + eh * 3600 + em * 60
    if t_end <= t_start:
        t_end += 86400
    mask = (df['timestamp'] >= t_start) & (df['timestamp'] < t_end)
else:
    mask = (df['timestamp'] >= date_ts) & (df['timestamp'] < date_ts + 86400)
idx = np.where(mask.values)[0]
time_label = f'{start_hm}-{end_hm}' if start_hm else 'full day'
print(f'{date} ({time_label}): {len(idx)} bars')

ts = timestamps[idx]
pr = prices[idx]
dd = dmi_diff[idx]
n = len(idx)

# I chart: DMI diff with rolling control limits
dd_series = pd.Series(dd)
dd_mean = dd_series.rolling(WINDOW, min_periods=10).mean().values
dd_std = dd_series.rolling(WINDOW, min_periods=10).std().values
ucl_i = dd_mean + 3 * dd_std
lcl_i = dd_mean - 3 * dd_std

# MR chart: moving range = |diff[i] - diff[i-1]|
mr = np.abs(np.diff(dd, prepend=dd[0]))
mr_series = pd.Series(mr)
mr_mean = mr_series.rolling(WINDOW, min_periods=10).mean().values
mr_std = mr_series.rolling(WINDOW, min_periods=10).std().values
ucl_mr = mr_mean + 3 * mr_std

# Detect out-of-control points
ooc_i = (dd > ucl_i) | (dd < lcl_i)  # DMI outside control limits
ooc_mr = mr > ucl_mr                   # MR spike = regime change

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                                      gridspec_kw={'height_ratios': [2, 1.5, 1]})
fig.suptitle(f'DMI I-MR Chart - {date} (1m, {WINDOW}-bar rolling limits)', fontsize=13, fontweight='bold')

# Panel 1: Price + MR spike markers
ax1.set_facecolor('#1a1a2e')
ax1.plot(ts, pr, color='#AAAAAA', linewidth=0.8)

# Mark bars where MR spiked (regime change)
mr_spike_idx = np.where(ooc_mr)[0]
if len(mr_spike_idx) > 0:
    ax1.scatter(ts[mr_spike_idx], pr[mr_spike_idx], color='#FFD700', s=15, zorder=5,
                alpha=0.7, label=f'MR spike ({len(mr_spike_idx)} bars)')

# Mark bars where DMI is out of control limits
ooc_up = np.where(dd > ucl_i)[0]
ooc_dn = np.where(dd < lcl_i)[0]
if len(ooc_up) > 0:
    ax1.scatter(ts[ooc_up], pr[ooc_up], color='#26A69A', s=10, marker='^', zorder=4, alpha=0.5)
if len(ooc_dn) > 0:
    ax1.scatter(ts[ooc_dn], pr[ooc_dn], color='#EF5350', s=10, marker='v', zorder=4, alpha=0.5)

ax1.set_ylabel('Price')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.15)

# Panel 2: I chart — DMI diff with control limits
ax2.set_facecolor('#1a1a2e')
ax2.fill_between(ts, 0, dd, where=dd > 0, color='#26A69A', alpha=0.3)
ax2.fill_between(ts, 0, dd, where=dd < 0, color='#EF5350', alpha=0.3)
ax2.plot(ts, dd, color='#00FFFF', linewidth=0.8, label='DMI diff (I)')
ax2.plot(ts, dd_mean, color='#FFD700', linewidth=1, linestyle='--', alpha=0.7, label='Mean')
ax2.plot(ts, ucl_i, color='#FF00FF', linewidth=0.8, linestyle=':', alpha=0.5, label='UCL/LCL (3-sigma)')
ax2.plot(ts, lcl_i, color='#FF00FF', linewidth=0.8, linestyle=':', alpha=0.5)
ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)

# Highlight out-of-control
if len(mr_spike_idx) > 0:
    ax2.scatter(ts[mr_spike_idx], dd[mr_spike_idx], color='#FFD700', s=20, zorder=5)

ax2.set_ylabel('DMI diff')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.15)

# Panel 3: MR chart — rate of change
ax3.set_facecolor('#1a1a2e')
ax3.fill_between(ts, 0, mr, color='#00FFFF', alpha=0.3)
ax3.plot(ts, mr, color='#00FFFF', linewidth=0.6, label='MR (|delta DMI|)')
ax3.plot(ts, mr_mean, color='#FFD700', linewidth=1, linestyle='--', alpha=0.7, label='MR mean')
ax3.plot(ts, ucl_mr, color='#FF00FF', linewidth=0.8, linestyle=':', alpha=0.5, label='UCL (3-sigma)')

# Highlight spikes
if len(mr_spike_idx) > 0:
    ax3.scatter(ts[mr_spike_idx], mr[mr_spike_idx], color='#FFD700', s=20, zorder=5)

ax3.set_ylabel('Moving Range')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.15)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
plt.xticks(rotation=45)
plt.tight_layout()

out = f'reports/findings/dmi_imr_{date.replace("-","")}.png'
os.makedirs('reports/findings', exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# Stats
print(f'\nI CHART:')
print(f'  Out of control (DMI > UCL or < LCL): {ooc_i.sum()} bars ({ooc_i.mean()*100:.1f}%)')
print(f'  Mean DMI diff: {np.nanmean(dd_mean):.1f}')

print(f'\nMR CHART:')
print(f'  MR spikes (> UCL): {ooc_mr.sum()} bars ({ooc_mr.mean()*100:.1f}%)')
print(f'  Mean MR: {np.nanmean(mr_mean):.2f}')
print(f'  Median MR: {np.median(mr):.2f}')
print(f'  p90 MR: {np.percentile(mr, 90):.2f}')
print(f'  p99 MR: {np.percentile(mr, 99):.2f}')
