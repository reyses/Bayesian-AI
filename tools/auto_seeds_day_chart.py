"""Visualize auto seeds (ZigZag) on 1m price for one day — compare with physics peaks."""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
TARGET_DATE = '2026-02-05'
MIN_REVERSAL = 30
MIN_BARS = 5

# Load OOS 1m
files = sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))
df_full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
day_1m = df_full[df_full['timestamp'].apply(
    lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') == TARGET_DATE)]

closes = day_1m['close'].values
highs = day_1m['high'].values
lows = day_1m['low'].values
ts_1m = day_1m['timestamp'].values

print(f"Day: {TARGET_DATE}, 1m bars: {len(day_1m)}")

# ZigZag on 1m closes
pivots = [0]
direction = None
last_high = closes[0]
last_low = closes[0]
last_high_i = 0
last_low_i = 0

for i in range(1, len(closes)):
    if closes[i] > last_high:
        last_high = closes[i]
        last_high_i = i
    if closes[i] < last_low:
        last_low = closes[i]
        last_low_i = i

    if direction is None:
        if (last_high - last_low) / TICK >= MIN_REVERSAL:
            if last_high_i > last_low_i:
                direction = 'UP'
                pivots.append(last_low_i)
            else:
                direction = 'DOWN'
                pivots.append(last_high_i)
    elif direction == 'UP':
        drop = (last_high - closes[i]) / TICK
        if drop >= MIN_REVERSAL and (i - last_high_i) >= MIN_BARS:
            pivots.append(last_high_i)
            direction = 'DOWN'
            last_low = closes[last_high_i]
            last_low_i = last_high_i
    elif direction == 'DOWN':
        rise = (closes[i] - last_low) / TICK
        if rise >= MIN_REVERSAL and (i - last_low_i) >= MIN_BARS:
            pivots.append(last_low_i)
            direction = 'UP'
            last_high = closes[last_low_i]
            last_high_i = last_low_i

# Build seeds
seeds = []
for j in range(len(pivots) - 1):
    si = pivots[j]
    ei = pivots[j + 1]
    if ei <= si:
        continue
    d = 'LONG' if closes[ei] > closes[si] else 'SHORT'
    change = abs(closes[ei] - closes[si]) / TICK
    seeds.append({
        'start': si, 'end': ei, 'direction': d,
        'entry_price': closes[si], 'exit_price': closes[ei],
        'change_ticks': change, 'bars': ei - si,
    })

total_pnl = sum(s['change_ticks'] for s in seeds) * 0.50
print(f"Seeds: {len(seeds)} | Oracle PnL: ${total_pnl:+,.0f}")

# Chart
fig, axes = plt.subplots(2, 1, figsize=(48, 16), sharex=True,
                          gridspec_kw={'height_ratios': [4, 1]})
fig.suptitle(
    f"Auto Seeds (ZigZag): {TARGET_DATE}  |  "
    f"Seeds: {len(seeds)}  |  Oracle PnL: ${total_pnl:+,.0f}",
    fontsize=14, fontweight='bold')

# Panel 1: Price colored by seed direction
ax = axes[0]
x = range(len(day_1m))

bar_colors = ['#888'] * len(day_1m)
for s in seeds:
    color = '#2ecc71' if s['direction'] == 'LONG' else '#e74c3c'
    for b in range(s['start'], min(s['end'] + 1, len(bar_colors))):
        bar_colors[b] = color

prev_color = bar_colors[0]
seg_start = 0
for i in range(1, len(bar_colors)):
    if bar_colors[i] != prev_color or i == len(bar_colors) - 1:
        seg_end = i + 1 if i == len(bar_colors) - 1 else i + 1
        ax.plot(range(seg_start, seg_end), closes[seg_start:seg_end],
                color=prev_color, linewidth=1.5)
        seg_start = i
        prev_color = bar_colors[i]

ax.fill_between(x, lows, highs, alpha=0.05, color='#666')

# Seed boundary markers + PnL
for s in seeds:
    marker = '^' if s['direction'] == 'LONG' else 'v'
    color = '#2ecc71' if s['direction'] == 'LONG' else '#e74c3c'
    ax.scatter(s['start'], s['entry_price'], marker=marker, color=color,
              s=200, zorder=5, edgecolors='black', linewidths=1)
    ax.annotate(f"${s['change_ticks']*0.50:+.0f}",
               xy=(s['end'], s['exit_price']),
               fontsize=7, color=color, fontweight='bold', ha='left')

tick_pos = list(range(0, len(day_1m), 5))
tick_labels = [datetime.fromtimestamp(ts_1m[i], tz=timezone.utc).strftime('%H:%M') for i in tick_pos]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
ax.set_ylabel('Price (1m)')
ax.grid(True, alpha=0.3)

# Panel 2: Seed magnitude
ax = axes[1]
for s in seeds:
    mid = (s['start'] + s['end']) // 2
    color = '#2ecc71' if s['direction'] == 'LONG' else '#e74c3c'
    ax.bar(mid, s['change_ticks'], width=s['bars'], color=color, alpha=0.6)
ax.set_ylabel('Seed Size (ticks)')
ax.set_xlabel('1m bar')
ax.grid(True, alpha=0.2)

plt.tight_layout()
output = 'reports/findings/imr_charts/auto_seeds_day.png'
os.makedirs(os.path.dirname(output), exist_ok=True)
fig.savefig(output, dpi=600, bbox_inches='tight')
plt.close()

# Preview
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 500000000
img = PIL.Image.open(output)
img.resize((1800, 600), PIL.Image.LANCZOS).save(
    output.replace('.png', '_preview.png'))

print(f"Saved: {output}")
for s in seeds:
    t = datetime.fromtimestamp(ts_1m[s['start']], tz=timezone.utc).strftime('%H:%M')
    print(f"  {t} {s['direction']:>5} {s['change_ticks']:>+6.0f}t "
          f"{s['bars']:>3} bars ${s['change_ticks']*0.50:>+7.1f}")
