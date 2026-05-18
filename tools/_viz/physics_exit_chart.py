"""Physics Exit: Enter at big 5m peak, exit at NEXT 5m peak (any size).
No fixed hold — hold until physics says the move is done."""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine

TARGET_DATE = None  # None = run all days
TICK = 0.25

# Load OOS 1m
files = sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))
df_full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Resample to 5m
df_full['ts_5m'] = (df_full['timestamp'] // 300) * 300
df_5m = df_full.groupby('ts_5m').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum', 'timestamp': 'first'
}).reset_index(drop=True)

engine = StatisticalFieldEngine()
raw_5m = engine.batch_compute_states(df_5m)
states_5m = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_5m]
ts_5m = df_5m['timestamp'].values
closes_5m = df_5m['close'].values

# Day filter
day_1m = df_full[df_full['timestamp'].apply(
    lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') == TARGET_DATE)]
closes_1m = day_1m['close'].values
highs_1m = day_1m['high'].values
lows_1m = day_1m['low'].values
ts_1m = day_1m['timestamp'].values

day_5m_mask = [datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') == TARGET_DATE
               for ts in ts_5m]
day_5m_idx = [i for i, m in enumerate(day_5m_mask) if m]

print(f"Day: {TARGET_DATE}, 1m: {len(day_1m)}, 5m: {len(day_5m_idx)}")

# First: collect ALL peaks before target day for prior distribution
all_5m_mask = [datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') < TARGET_DATE
               for ts in ts_5m]
prior_idx = [i for i, m in enumerate(all_5m_mask) if m]

prior_magnitudes = []
prev_fm_prior = 0
move_start_prior = closes_5m[prior_idx[0]] if prior_idx else 0

for i in prior_idx:
    s = states_5m[i]
    if s is None:
        continue
    fm = s.F_momentum
    if prev_fm_prior != 0 and fm != 0 and np.sign(prev_fm_prior) != np.sign(fm):
        prior_magnitudes.append(abs(closes_5m[i] - move_start_prior) / TICK)
        move_start_prior = closes_5m[i]
    prev_fm_prior = fm

print(f"Prior days peaks for ranking: {len(prior_magnitudes)}")

# Detect 5m FM flips for target day
peaks = []
prev_fm = 0
move_start_price = closes_5m[day_5m_idx[0]]

for i in day_5m_idx:
    s = states_5m[i]
    if s is None:
        continue
    fm = s.F_momentum
    if prev_fm != 0 and fm != 0 and np.sign(prev_fm) != np.sign(fm):
        prior_move = abs(closes_5m[i] - move_start_price) / TICK
        new_dir = 'LONG' if prev_fm < 0 else 'SHORT'
        peaks.append({
            'bar_5m': i, 'ts': ts_5m[i], 'price': closes_5m[i],
            'prior_move': prior_move, 'direction': new_dir,
        })
        move_start_price = closes_5m[i]
    prev_fm = fm

# Causal percentile: rank against prior days + peaks seen so far today
pk = pd.DataFrame(peaks)
causal_pctile = []
for i in range(len(pk)):
    # All prior days + today's peaks up to now
    history = prior_magnitudes + list(pk['prior_move'].iloc[:i + 1])
    rank = sum(1 for h in history if h < pk['prior_move'].iloc[i]) / max(len(history), 1)
    causal_pctile.append(rank)
pk['prior_pctile'] = causal_pctile
big = pk[pk.prior_pctile >= 0.50]
print(f"5m peaks: {len(pk)}, big (top50%): {len(big)}")

# Trades: enter at big peak, exit at NEXT BIG peak (magnitude filtered exit too)
trades = []
for _, p in big.iterrows():
    entry_ts = p.ts
    entry_price = p.price
    trade_dir = p.direction

    # Exit only at next BIG peak (top 50% magnitude), not any tiny wobble
    future = big[big.ts > entry_ts]
    if len(future) == 0:
        continue
    nxt = future.iloc[0]
    exit_ts = nxt.ts
    exit_price = nxt.price

    if trade_dir == 'LONG':
        pnl = (exit_price - entry_price) / TICK
    else:
        pnl = (entry_price - exit_price) / TICK

    entry_1m = int(np.argmin(np.abs(ts_1m - entry_ts)))
    exit_1m = int(np.argmin(np.abs(ts_1m - exit_ts)))
    hold_min = (exit_ts - entry_ts) / 60

    trades.append({
        'entry_ts': entry_ts, 'exit_ts': exit_ts,
        'entry_price': entry_price, 'exit_price': exit_price,
        'dir': trade_dir, 'pnl': pnl, 'hold_min': hold_min,
        'entry_1m': entry_1m, 'exit_1m': exit_1m,
    })

total_pnl = sum(t['pnl'] for t in trades) * 0.50
wr = sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1) * 100
print(f"Trades: {len(trades)} | WR: {wr:.0f}% | PnL: ${total_pnl:+,.0f}")

# Chart
fig, axes = plt.subplots(2, 1, figsize=(48, 16), sharex=True,
                          gridspec_kw={'height_ratios': [4, 1]})
fig.suptitle(
    f"Physics Exit: big 5m peak entry, next peak exit  --  {TARGET_DATE}\n"
    f"Trades: {len(trades)} | WR: {wr:.0f}% | PnL: ${total_pnl:+,.0f}  "
    f"(Oracle: $14,438)",
    fontsize=14, fontweight='bold')

ax = axes[0]
x = range(len(day_1m))

bar_colors = ['#888'] * len(day_1m)
for t in trades:
    color = '#2ecc71' if t['dir'] == 'LONG' else '#e74c3c'
    for b in range(t['entry_1m'], min(t['exit_1m'] + 1, len(bar_colors))):
        bar_colors[b] = color

prev_color = bar_colors[0]
seg_start = 0
for i in range(1, len(bar_colors)):
    if bar_colors[i] != prev_color or i == len(bar_colors) - 1:
        ax.plot(range(seg_start, i + 1), closes_1m[seg_start:i + 1],
                color=prev_color, linewidth=1.5)
        seg_start = i
        prev_color = bar_colors[i]

ax.fill_between(x, lows_1m, highs_1m, alpha=0.05, color='#666')

for t in trades:
    color = '#2ecc71' if t['pnl'] > 0 else '#e74c3c'
    ax.annotate(f"${t['pnl']*0.50:+.0f}",
                xy=(t['exit_1m'], t['exit_price']),
                fontsize=6, color=color, fontweight='bold', ha='left')

for _, p in pk.iterrows():
    px = int(np.argmin(np.abs(ts_1m - p.ts)))
    is_big = p.prior_pctile >= 0.50
    marker = '^' if p.direction == 'LONG' else 'v'
    color = '#2ecc71' if p.direction == 'LONG' else '#e74c3c'
    if not is_big:
        color = '#bbb'
    ax.scatter(px, p.price, marker=marker, color=color,
              s=150 if is_big else 40, zorder=5,
              edgecolors='black' if is_big else 'none', linewidths=0.8)

tick_pos = list(range(0, len(day_1m), 5))
tick_labels = [datetime.fromtimestamp(ts_1m[i], tz=timezone.utc).strftime('%H:%M')
               for i in tick_pos]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
ax.set_ylabel('Price (1m)')
ax.grid(True, alpha=0.3)

ax = axes[1]
for t in trades:
    mid = (t['entry_1m'] + t['exit_1m']) // 2
    width = max(t['exit_1m'] - t['entry_1m'], 1)
    color = '#2ecc71' if t['pnl'] > 0 else '#e74c3c'
    ax.bar(mid, t['hold_min'], width=width, color=color, alpha=0.6)
ax.set_ylabel('Hold (min)')
ax.set_xlabel('1m bar')
ax.grid(True, alpha=0.2)

plt.tight_layout()
output = 'reports/findings/imr_charts/physics_exit_day.png'
os.makedirs(os.path.dirname(output), exist_ok=True)
fig.savefig(output, dpi=600, bbox_inches='tight')
plt.close()

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 500000000
img = PIL.Image.open(output)
img.resize((1800, 600), PIL.Image.LANCZOS).save(
    output.replace('.png', '_preview.png'))

print(f"Saved: {output}")
for t in trades:
    ts = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).strftime('%H:%M')
    print(f"  {ts} {t['dir']:>5} {t['pnl']:>+6.0f}t {t['hold_min']:>5.1f}m "
          f"${t['pnl']*0.50:>+7.1f}")
