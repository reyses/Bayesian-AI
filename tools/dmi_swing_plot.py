"""Quick DMI swing analysis for a single day."""
import numpy as np, pandas as pd, gc, matplotlib, sys, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine

date = sys.argv[1] if len(sys.argv) > 1 else '2026-03-19'

df = pd.read_parquet('DATA/ATLAS/1m/2026_03.parquet').sort_values('timestamp').reset_index(drop=True)
sfe = StatisticalFieldEngine()
states = sfe.batch_compute_states(df)

dmi_plus = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_plus', 0.0) for s in states])
dmi_minus = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_minus', 0.0) for s in states])
dmi_diff = dmi_plus - dmi_minus
del states, sfe; gc.collect()

prices = df['close'].values
timestamps = pd.to_datetime(df['timestamp'].values, unit='s', utc=True)

# Filter to day
date_ts = pd.Timestamp(date).timestamp()
mask = (df['timestamp'] >= date_ts) & (df['timestamp'] < date_ts + 86400)
idx = np.where(mask.values)[0]
print(f'{date}: {len(idx)} bars')

ts_day = timestamps[idx]
pr_day = prices[idx]
dd_day = dmi_diff[idx]
dp_day = dmi_plus[idx]
dm_day = dmi_minus[idx]
n = len(idx)

# Find swings
swings = []
swing_start = 0
swing_max = 0
for i in range(1, n):
    swing_max = max(swing_max, abs(dd_day[i]))
    if (dd_day[i-1] > 0 and dd_day[i] <= 0) or (dd_day[i-1] < 0 and dd_day[i] >= 0):
        if i - swing_start > 1:
            swings.append({
                'start': swing_start, 'end': i,
                'bars': i - swing_start,
                'max_diff': swing_max,
                'dir': 'BULL' if dd_day[swing_start] > 0 else 'BEAR',
                'price_move': pr_day[i] - pr_day[swing_start],
            })
        swing_start = i
        swing_max = abs(dd_day[i])

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                                      gridspec_kw={'height_ratios': [2.5, 1, 1]})
fig.suptitle(f'DMI Swing Analysis - {date} (1m)', fontsize=13, fontweight='bold')

# Panel 1: Price + swing coloring
ax1.set_facecolor('#1a1a2e')
ax1.plot(ts_day, pr_day, color='#AAAAAA', linewidth=0.8)

for s in swings:
    si, ei = s['start'], min(s['end'], n - 1)
    color = '#26A69A' if s['dir'] == 'BULL' else '#EF5350'
    alpha = 0.1 if s['max_diff'] < 10 else 0.35
    ax1.axvspan(ts_day[si], ts_day[ei], color=color, alpha=alpha)
    if s['max_diff'] >= 10:
        mid = (si + ei) // 2
        lbl = f"{s['dir'][0]} d={s['max_diff']:.0f}\n{s['bars']}b ${abs(s['price_move'])*2:.0f}"
        ax1.annotate(lbl, xy=(ts_day[mid], pr_day[mid]), fontsize=7,
                     color='white', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

ax1.set_ylabel('Price')
ax1.grid(True, alpha=0.15)

# Panel 2: DMI+ and DMI-
ax2.set_facecolor('#1a1a2e')
ax2.plot(ts_day, dp_day, color='#26A69A', linewidth=1, label='+DI')
ax2.plot(ts_day, dm_day, color='#EF5350', linewidth=1, label='-DI')
ax2.fill_between(ts_day, dp_day, dm_day, where=dp_day > dm_day, color='#26A69A', alpha=0.15)
ax2.fill_between(ts_day, dp_day, dm_day, where=dp_day < dm_day, color='#EF5350', alpha=0.15)
ax2.set_ylabel('+DI / -DI')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.15)

# Panel 3: DMI diff
ax3.set_facecolor('#1a1a2e')
ax3.fill_between(ts_day, 0, dd_day, where=dd_day > 0, color='#26A69A', alpha=0.4)
ax3.fill_between(ts_day, 0, dd_day, where=dd_day < 0, color='#EF5350', alpha=0.4)
ax3.axhline(y=10, color='#26A69A', linewidth=1, linestyle='--', alpha=0.5, label='Trade threshold')
ax3.axhline(y=-10, color='#EF5350', linewidth=1, linestyle='--', alpha=0.5)
ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
ax3.set_ylabel('DMI diff')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.15)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
plt.xticks(rotation=45)
plt.tight_layout()

out = f'reports/findings/dmi_swing_{date.replace("-","")}.png'
os.makedirs('reports/findings', exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

big = [s for s in swings if s['max_diff'] >= 10]
small = [s for s in swings if s['max_diff'] < 10]
print(f'\nNoise swings (<10): {len(small)}, avg bars={np.mean([s["bars"] for s in small]):.0f}, avg move=${np.mean([abs(s["price_move"]) for s in small])*2:.1f}')
print(f'Trade swings (>10): {len(big)}, avg bars={np.mean([s["bars"] for s in big]):.0f}, avg move=${np.mean([abs(s["price_move"]) for s in big])*2:.1f}')
print(f'If captured all big swings: ${sum(abs(s["price_move"]) * 2 for s in big):.0f}')
