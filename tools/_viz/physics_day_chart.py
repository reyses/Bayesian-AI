"""Visualize pure physics 5m exhaustion peaks overlaid on 1m price for one day."""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine

TICK = 0.25
TARGET_DATE = '2026-02-05'

# Load OOS 1m
files = sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))
df_full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Compute 5m states from resampled 1m
df_full['ts_5m'] = (df_full['timestamp'] // 300) * 300
df_5m = df_full.groupby('ts_5m').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum', 'timestamp': 'first'
}).reset_index(drop=True)

engine = StatisticalFieldEngine()
raw_5m = engine.batch_compute_states(df_5m)
states_5m = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_5m]
closes_5m = df_5m['close'].values
ts_5m = df_5m['timestamp'].values

# Filter to target day
day_1m = df_full[df_full['timestamp'].apply(
    lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') == TARGET_DATE)]
day_5m_mask = [datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') == TARGET_DATE
               for ts in ts_5m]
day_5m_idx = [i for i, m in enumerate(day_5m_mask) if m]

print(f"Day: {TARGET_DATE}")
print(f"1m bars: {len(day_1m)}, 5m bars: {len(day_5m_idx)}")

# Detect 5m FM sign flips for this day
peaks = []
prev_fm = 0
move_start_price = closes_5m[day_5m_idx[0]]

for i in day_5m_idx:
    s = states_5m[i]
    if s is None: continue
    fm = s.F_momentum
    if prev_fm != 0 and fm != 0 and np.sign(prev_fm) != np.sign(fm):
        prior_move = abs(closes_5m[i] - move_start_price) / TICK
        new_dir = 'LONG' if prev_fm < 0 else 'SHORT'
        peaks.append({
            'bar_5m': i, 'ts': ts_5m[i], 'price': closes_5m[i],
            'prior_move': prior_move, 'direction': new_dir, 'fm': fm,
        })
        move_start_price = closes_5m[i]
    prev_fm = fm

pk = pd.DataFrame(peaks)
if len(pk) > 0:
    pk['prior_pctile'] = pk['prior_move'].rank(pct=True)
print(f"5m peaks: {len(pk)}")

# Simulate trades (top 50%, hold 2 bars = 10 min)
trades = []
in_trade = False
entry_price = entry_bar = 0
trade_dir = ''
entry_ts = 0

if len(pk) > 0:
    big = pk[pk.prior_pctile >= 0.50]
    for _, p in big.iterrows():
        bar = int(p.bar_5m)
        if in_trade and bar >= entry_bar + 2:
            eb = min(entry_bar + 2, len(closes_5m) - 1)
            ep = closes_5m[eb]
            pnl = ((ep - entry_price) if trade_dir == 'LONG' else (entry_price - ep)) / TICK
            trades.append({'entry_ts': entry_ts, 'exit_ts': ts_5m[eb],
                          'entry_price': entry_price, 'exit_price': ep,
                          'dir': trade_dir, 'pnl': pnl})
            in_trade = False
        if in_trade: continue
        if bar + 2 < len(closes_5m):
            trade_dir = p.direction
            entry_price = p.price
            entry_bar = bar
            entry_ts = p.ts
            in_trade = True

print(f"Trades: {len(trades)}")

# === BUILD CHART ===
fig, axes = plt.subplots(3, 1, figsize=(48, 24), sharex=True,
                          gridspec_kw={'height_ratios': [4, 1.5, 1]})

total_pnl = sum(t['pnl'] for t in trades) * 0.50
wr = sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1) * 100
fig.suptitle(
    f'Pure Physics: 5m Exhaustion on 1m Price  --  {TARGET_DATE}\n'
    f'Peaks: {len(pk)} | Trades (top 50% magnitude): {len(trades)} | '
    f'WR: {wr:.0f}% | PnL: ${total_pnl:+,.0f}',
    fontsize=13, fontweight='bold')

# Panel 1: 1m price + peak markers + trade shading
ax = axes[0]
closes_1m = day_1m['close'].values
highs_1m = day_1m['high'].values
lows_1m = day_1m['low'].values
ts_1m = day_1m['timestamp'].values
x = range(len(day_1m))

# Draw price line in trade color (green=LONG, red=SHORT, gray=flat)
# Build color per bar
bar_colors = ['#888'] * len(day_1m)  # default gray (flat)
for t in trades:
    ex = int(np.argmin(np.abs(ts_1m - t['entry_ts'])))
    xx = int(np.argmin(np.abs(ts_1m - t['exit_ts'])))
    color = '#2ecc71' if t['dir'] == 'LONG' else '#e74c3c'
    for b in range(ex, min(xx + 1, len(bar_colors))):
        bar_colors[b] = color

# Plot price segments by color
prev_color = bar_colors[0]
seg_start = 0
for i in range(1, len(bar_colors)):
    if bar_colors[i] != prev_color or i == len(bar_colors) - 1:
        seg_end = i + 1 if i == len(bar_colors) - 1 else i + 1
        ax.plot(range(seg_start, seg_end), closes_1m[seg_start:seg_end],
                color=prev_color, linewidth=1.2)
        seg_start = i
        prev_color = bar_colors[i]

ax.fill_between(x, lows_1m, highs_1m, alpha=0.05, color='#666')

# Peak markers
for _, p in pk.iterrows():
    nearest_x = int(np.argmin(np.abs(ts_1m - p.ts)))
    is_big = p.prior_pctile >= 0.50
    color = '#2ecc71' if p.direction == 'LONG' else '#e74c3c'
    if not is_big:
        color = '#bbb'
    marker = '^' if p.direction == 'LONG' else 'v'
    size = 150 if is_big else 40
    ax.scatter(nearest_x, p.price, marker=marker, color=color, s=size, zorder=5,
              edgecolors='black' if is_big else 'none', linewidths=0.8)

# PnL annotations at exit points
for t in trades:
    xx = int(np.argmin(np.abs(ts_1m - t['exit_ts'])))
    color = '#2ecc71' if t['pnl'] > 0 else '#e74c3c'
    ax.annotate(f"${t['pnl']*0.50:+.0f}", xy=(xx, t['exit_price']),
               fontsize=6, color=color, fontweight='bold', ha='left')

# X ticks every 1 min (every bar)
tick_pos = list(range(0, len(day_1m), 5))  # every 5 bars = 5 min labels
tick_labels = [datetime.fromtimestamp(ts_1m[i], tz=timezone.utc).strftime('%H:%M') for i in tick_pos]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
# Minor ticks every 1 bar
ax.set_xticks(range(len(day_1m)), minor=True)
ax.tick_params(axis='x', which='minor', length=2)
ax.set_ylabel('Price (1m)')
ax.grid(True, alpha=0.3, which='major')
ax.grid(True, alpha=0.08, which='minor')

# Panel 2: 5m F_momentum with zero crossings highlighted
ax = axes[1]
day_fm = [getattr(states_5m[i], 'F_momentum', 0) if states_5m[i] else 0 for i in day_5m_idx]
# Map 5m bars to 1m x positions
fm_x = [int(np.argmin(np.abs(ts_1m - ts_5m[i]))) for i in day_5m_idx]
ax.plot(fm_x, day_fm, color='purple', linewidth=1.2)
ax.axhline(0, color='gray', linewidth=0.5)
ax.fill_between(fm_x, day_fm, 0, where=[f > 0 for f in day_fm], alpha=0.15, color='green')
ax.fill_between(fm_x, day_fm, 0, where=[f < 0 for f in day_fm], alpha=0.15, color='red')

# Mark peaks on FM panel
for _, p in pk.iterrows():
    px = int(np.argmin(np.abs(ts_1m - p.ts)))
    is_big = p.prior_pctile >= 0.50
    ax.axvline(px, color='orange' if is_big else '#ddd', alpha=0.6 if is_big else 0.2, linewidth=1)

ax.set_ylabel('5m F_momentum')
ax.grid(True, alpha=0.2)

# Panel 3: Prior move magnitude as bars
ax = axes[2]
if len(pk) > 0:
    bar_x = [int(np.argmin(np.abs(ts_1m - p.ts))) for _, p in pk.iterrows()]
    colors = ['#2ecc71' if p.prior_pctile >= 0.50 else '#ddd' for _, p in pk.iterrows()]
    ax.bar(bar_x, pk['prior_move'].values, color=colors, width=3)
    ax.axhline(pk['prior_move'].median(), color='red', linestyle='--', alpha=0.5)
ax.set_ylabel('Prior Move (ticks)')
ax.set_xlabel('Time (1m bars)')
ax.grid(True, alpha=0.2)

plt.tight_layout()
output = 'reports/findings/imr_charts/physics_5m_day.png'
os.makedirs(os.path.dirname(output), exist_ok=True)
fig.savefig(output, dpi=600, bbox_inches='tight')
plt.close()
print(f"Saved: {output}")

if trades:
    print(f"Day: ${total_pnl:+,.0f} | {len(trades)} trades | {wr:.0f}% WR")
