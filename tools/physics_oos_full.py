"""Full OOS: Pure physics 5m peaks, IS-ranked magnitude, big entry + big exit."""
import os, sys, glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine

TICK = 0.25

# Load IS 1m -> resample to 5m -> compute states -> collect magnitude distribution
print('Loading IS...')
is_files = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
is_1m = pd.concat([pd.read_parquet(f) for f in is_files], ignore_index=True)
is_1m['ts_5m'] = (is_1m['timestamp'] // 300) * 300
is_5m = is_1m.groupby('ts_5m').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum', 'timestamp': 'first'
}).reset_index(drop=True)
del is_1m  # free memory

engine = StatisticalFieldEngine()
print('Computing IS 5m states...')
raw_is = engine.batch_compute_states(is_5m)
states_is = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_is]

prior_mags = []
prev_fm = 0
move_start = is_5m['close'].values[0]
for i in range(1, len(states_is)):
    s = states_is[i]
    if s is None:
        continue
    fm = s.F_momentum
    if prev_fm != 0 and fm != 0 and np.sign(prev_fm) != np.sign(fm):
        prior_mags.append(abs(is_5m['close'].values[i] - move_start) / TICK)
        move_start = is_5m['close'].values[i]
    prev_fm = fm

prior_median = np.median(prior_mags)
print(f"IS magnitudes: {len(prior_mags)}, median: {prior_median:.0f}t")
del is_5m, raw_is, states_is  # free memory

# Rolling window for magnitude ranking (last 200 peaks)
from collections import deque
ROLLING_WINDOW = 140  # ~2 days of peaks (70/day)
mag_window = deque(prior_mags[-ROLLING_WINDOW:], maxlen=ROLLING_WINDOW)
print(f"Rolling window seeded with last {len(mag_window)} IS magnitudes (2-day max)")

# Load OOS
print('Loading OOS...')
oos_files = sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))
oos_1m = pd.concat([pd.read_parquet(f) for f in oos_files], ignore_index=True)
oos_1m['ts_5m'] = (oos_1m['timestamp'] // 300) * 300
oos_5m = oos_1m.groupby('ts_5m').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum', 'timestamp': 'first'
}).reset_index(drop=True)
del oos_1m

print('Computing OOS 5m states...')
raw_oos = engine.batch_compute_states(oos_5m)
states_oos = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_oos]
closes = oos_5m['close'].values
ts = oos_5m['timestamp'].values

# Detect peaks
peaks = []
prev_fm = 0
move_start = closes[0]

for i in range(1, len(states_oos)):
    s = states_oos[i]
    if s is None:
        continue
    fm = s.F_momentum
    if prev_fm != 0 and fm != 0 and np.sign(prev_fm) != np.sign(fm):
        prior_move = abs(closes[i] - move_start) / TICK
        new_dir = 'LONG' if prev_fm < 0 else 'SHORT'
        # Rolling percentile: rank against last 200 peaks (adapts to current conditions)
        pctile = sum(1 for m in mag_window if m < prior_move) / max(len(mag_window), 1)
        adx = s.adx_strength
        peaks.append({
            'bar': i, 'ts': ts[i], 'price': closes[i],
            'prior_move': prior_move, 'direction': new_dir,
            'pctile': pctile, 'is_big': pctile >= 0.50,
            'adx': adx, 'trending': adx > 30,
        })
        mag_window.append(prior_move)  # update rolling window
        move_start = closes[i]
    prev_fm = fm

pk = pd.DataFrame(peaks)
big = pk[pk.is_big]
print(f"OOS peaks: {len(pk)}, big: {len(big)}")

# Trade: enter big, exit next big
# ADX regime switch: trending (ADX>30) = don't flip, ride the trend
trades = []
in_trade = False
entry_price = 0.0
trade_dir = ''
entry_ts_val = 0.0

for _, p in big.iterrows():
    if in_trade:
        # On trending regime: only exit if new peak is SAME direction (trend continues)
        # or if regime switched to oscillating
        if p.trending and p.direction == trade_dir:
            # Trending + same direction = trend continues, don't exit
            continue

        exit_price = p.price
        if trade_dir == 'LONG':
            pnl = (exit_price - entry_price) / TICK
        else:
            pnl = (entry_price - exit_price) / TICK
        hold_min = (p.ts - entry_ts_val) / 60
        date = datetime.fromtimestamp(entry_ts_val, tz=timezone.utc).strftime('%Y-%m-%d')
        trades.append({'pnl': pnl, 'dir': trade_dir, 'hold_min': hold_min, 'date': date,
                       'adx': p.adx, 'trending': p.trending})
        in_trade = False

    # Enter: on oscillating days flip normally, on trending days only enter with trend
    if p.trending:
        # Trending: enter in the trend direction (FM sign = contrarian = against old move)
        # But skip if this would fight the macro direction we're already tracking
        trade_dir = p.direction
    else:
        # Oscillating: flip as usual
        trade_dir = p.direction

    entry_price = p.price
    entry_ts_val = p.ts
    in_trade = True

t = pd.DataFrame(trades)
wins = t[t.pnl > 0]
total = t.pnl.sum() * 0.50
n_days = t.date.nunique()

print()
print('=' * 70)
print('FULL OOS: Pure Physics 5m - Big Entry + Big Exit')
print(f'IS median threshold: {prior_median:.0f}t | No lookahead')
print('=' * 70)
print(f"  Trades: {len(t):,}")
print(f"  WR: {len(wins)/len(t)*100:.1f}%")
print(f"  Total: {t.pnl.sum():,.0f}t (${total:,.0f})")
print(f"  Avg: {t.pnl.mean():.1f}t (${t.pnl.mean()*0.50:.2f})")
print(f"  Avg hold: {t.hold_min.mean():.1f} min")
print(f"  Days: {n_days}")
print(f"  Per day: ${total/n_days:,.0f}")
print()

for d in ['LONG', 'SHORT']:
    sub = t[t.dir == d]
    if len(sub) > 0:
        print(f"  {d}: {len(sub)} trades, WR={(sub.pnl>0).mean()*100:.0f}%, "
              f"avg={sub.pnl.mean():.1f}t, total=${sub.pnl.sum()*0.50:,.0f}")

print()
print("  Daily breakdown:")
daily = t.groupby('date').agg(
    n=('pnl', 'count'),
    pnl=('pnl', 'sum'),
    wr=('pnl', lambda x: (x > 0).mean() * 100)
)
daily['usd'] = daily['pnl'] * 0.50
for date, row in daily.iterrows():
    print(f"    {date}: {row.n:.0f} trades  WR={row.wr:.0f}%  ${row.usd:>+8,.0f}")

print()
print("  COMPARISON:")
print(f"    Blind flip:        $7,967  $73/day")
print(f"    Current system:    $1,844  $56/day")
print(f"    Physics (this):    ${total:,.0f}  ${total/n_days:,.0f}/day")

# Save
output = 'reports/findings/physics_oos_full.txt'
os.makedirs('reports/findings', exist_ok=True)
with open(output, 'w') as f:
    f.write(f"Physics OOS: ${total:,.0f} | {len(t)} trades | {len(wins)/len(t)*100:.0f}% WR | ${total/n_days:,.0f}/day\n")
print(f"\nSaved: {output}")
