"""Draw horizontal lines at every bar high/low to visualize limit order clusters."""
import numpy as np, pandas as pd, sys, os, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

date = sys.argv[1] if len(sys.argv) > 1 else '2026-03-19'
start_hm = sys.argv[2] if len(sys.argv) > 2 else '06:00'
end_hm = sys.argv[3] if len(sys.argv) > 3 else '10:00'
tf = sys.argv[4] if len(sys.argv) > 4 else '1m'

files = sorted(glob.glob(f'DATA/ATLAS/{tf}/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files[-3:]], ignore_index=True)
df = df.sort_values('timestamp').reset_index(drop=True)

# Filter
date_ts = pd.Timestamp(date).timestamp()
sh, sm = map(int, start_hm.split(':'))
eh, em = map(int, end_hm.split(':'))
t_start = date_ts + sh * 3600 + sm * 60
t_end = date_ts + eh * 3600 + em * 60
if t_end <= t_start:
    t_end += 86400

mask = (df['timestamp'] >= t_start) & (df['timestamp'] < t_end)
d = df[mask].reset_index(drop=True)
n = len(d)
print(f'{date} {start_hm}-{end_hm}: {n} bars')

ts = pd.to_datetime(d['timestamp'].values, unit='s', utc=True)
closes = d['close'].values
opens = d['open'].values
highs = d['high'].values
lows = d['low'].values
green = closes > opens

TICK = 0.25

# Plot: candlestick + horizontal lines at highs of green bars and lows of red bars
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
fig.suptitle(f'Bar Highs/Lows as Levels — {date} {start_hm}-{end_hm} ({tf})', fontsize=13, fontweight='bold')
ax.set_facecolor('#1a1a2e')

from datetime import timedelta

# Draw candlesticks
for i in range(n):
    o, c, h, l = opens[i], closes[i], highs[i], lows[i]
    color = '#26A69A' if c >= o else '#EF5350'
    # Wick
    ax.plot([ts[i], ts[i]], [l, h], color='#666666', linewidth=0.5)
    # Body
    body = max(abs(c - o), TICK)
    width = timedelta(seconds=40)
    ax.bar(ts[i], body, bottom=min(o, c), width=width, color=color,
           edgecolor='none', alpha=0.9)

# Draw horizontal lines at green bar highs (where sellers sit)
for i in range(n):
    if green[i]:
        upper_wick_ratio = (highs[i] - closes[i]) / max(highs[i] - lows[i], TICK)
        if upper_wick_ratio > 0.02:  # has some upper wick
            alpha = min(0.6, upper_wick_ratio * 2)  # bigger wick = stronger line
            ax.axhline(y=highs[i], color='#EF5350', linewidth=0.3, alpha=alpha)

# Draw horizontal lines at red bar lows (where buyers sit)
for i in range(n):
    if not green[i] and closes[i] != opens[i]:
        lower_wick_ratio = (closes[i] - lows[i]) / max(highs[i] - lows[i], TICK)
        if lower_wick_ratio > 0.02:
            alpha = min(0.6, lower_wick_ratio * 2)
            ax.axhline(y=lows[i], color='#26A69A', linewidth=0.3, alpha=alpha)

ax.set_ylabel('Price')
ax.grid(True, alpha=0.1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
plt.xticks(rotation=45)
plt.tight_layout()

out = f'reports/findings/bar_levels_{date.replace("-","")}.png'
os.makedirs('reports/findings', exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# Cluster analysis: where do the highs cluster?
# Round to nearest 2 ticks (0.50) and count
from collections import Counter

high_prices = np.round(highs / (TICK * 2)) * (TICK * 2)  # round to 0.50
low_prices = np.round(lows / (TICK * 2)) * (TICK * 2)

high_counts = Counter(high_prices)
low_counts = Counter(low_prices)

print(f'\nTOP 10 HIGH CLUSTERS (resistance):')
for price, count in high_counts.most_common(10):
    print(f'  {price:.2f}: {count} bars touched')

print(f'\nTOP 10 LOW CLUSTERS (support):')
for price, count in low_counts.most_common(10):
    print(f'  {price:.2f}: {count} bars touched')
