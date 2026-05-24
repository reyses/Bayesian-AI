"""Seed Funnel OOS — Match raw IS trend seeds against OOS price action.

Each IS seed defines: a move of X ticks lasting Y bars.
At each OOS bar, track the current move. When it matches an IS seed's
magnitude + duration profile, enter opposite (exhaustion = reversal).

The funnel: as the current move develops, candidate seeds narrow.
At bar 1: many seeds match (move just started, could be anything).
At bar 5: fewer match (move size and duration filtering).
At bar 10: one or two match — that's the signal.

No lookahead. Seeds are IS-derived. Direction from seed profile.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
TICK_VALUE = 0.50

# Load IS trend seeds
print('Loading IS trend seeds...')
with open('DATA/regime_seeds/auto_swing/auto_seeds_all_20260313_200730.json') as f:
    auto = json.load(f)

all_seeds = []
for d in auto['days'].values():
    all_seeds.extend(d['seeds'])

# Build seed profiles: (magnitude_ticks, duration_bars, direction)
seed_profiles = []
for s in all_seeds:
    seed_profiles.append({
        'mag': abs(s['change_ticks']),
        'bars': s['n_bars'],
        'dur_min': s['duration_mins'],
        'direction': s['direction'],
        'mfe': s['mfe_ticks'],
        'mae': s['mae_ticks'],
    })

sp = pd.DataFrame(seed_profiles)
print(f"IS seeds: {len(sp)}")
print(f"  Magnitude: p25={sp.mag.quantile(0.25):.0f}t p50={sp.mag.median():.0f}t p75={sp.mag.quantile(0.75):.0f}t")
print(f"  Duration: p25={sp.bars.quantile(0.25):.0f} p50={sp.bars.median():.0f} p75={sp.bars.quantile(0.75):.0f} bars")

# Bin seeds by magnitude + duration for fast lookup
# Magnitude bins: [0-30, 30-60, 60-100, 100-150, 150-250, 250+]
# Duration bins: [3-5, 5-8, 8-12, 12-20, 20+] bars at 1m
MAG_BINS = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 250), (250, 9999)]
DUR_BINS = [(3, 5), (5, 8), (8, 12), (12, 20), (20, 999)]

seed_lookup = defaultdict(list)
for _, s in sp.iterrows():
    for mi, (mlo, mhi) in enumerate(MAG_BINS):
        if mlo <= s.mag < mhi:
            for di, (dlo, dhi) in enumerate(DUR_BINS):
                if dlo <= s.bars < dhi:
                    seed_lookup[(mi, di)].append(s.to_dict())
                    break
            break

print(f"Seed bins: {len(seed_lookup)} non-empty")
for k, v in sorted(seed_lookup.items()):
    mag_range = MAG_BINS[k[0]]
    dur_range = DUR_BINS[k[1]]
    pct_long = sum(1 for s in v if s['direction'] == 'LONG') / len(v) * 100
    print(f"  mag={mag_range[0]}-{mag_range[1]}t dur={dur_range[0]}-{dur_range[1]}bars: "
          f"{len(v)} seeds ({pct_long:.0f}% LONG)")

# Load OOS 1m
print('\nLoading OOS 1m...')
oos_files = sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))
oos_1m = pd.concat([pd.read_parquet(f) for f in oos_files], ignore_index=True)
closes = oos_1m['close'].values
highs = oos_1m['high'].values
lows = oos_1m['low'].values
timestamps = oos_1m['timestamp'].values
print(f"OOS 1m bars: {len(oos_1m)}")

# Simulate: track current move, match against seed profiles
trades = []
in_trade = False
entry_price = 0.0
trade_dir = ''
entry_bar = 0
entry_ts = 0.0

# Track current move
move_start_price = closes[0]
move_start_bar = 0
move_high = closes[0]
move_low = closes[0]

MIN_MOVE_BARS = 3  # minimum bars before checking seeds

for i in range(1, len(closes)):
    # Update move tracking
    if closes[i] > move_high:
        move_high = closes[i]
    if closes[i] < move_low:
        move_low = closes[i]

    move_bars = i - move_start_bar
    move_up = (move_high - move_start_price) / TICK
    move_down = (move_start_price - move_low) / TICK

    # Determine current move direction and magnitude
    if move_up > move_down:
        current_mag = move_up
        current_dir = 'LONG'  # move was UP
    else:
        current_mag = move_down
        current_dir = 'SHORT'  # move was DOWN

    # Exit check
    if in_trade and i >= entry_bar + trade_hold:
        ep = closes[i]
        if trade_dir == 'LONG':
            pnl = (ep - entry_price) / TICK
        else:
            pnl = (entry_price - ep) / TICK
        hold_min = (timestamps[i] - entry_ts) / 60
        date = datetime.fromtimestamp(entry_ts, tz=timezone.utc).strftime('%Y-%m-%d')
        trades.append({'pnl': pnl, 'dir': trade_dir, 'hold_min': hold_min,
                       'date': date, 'matched_seeds': trade_n_seeds})
        in_trade = False
        # Reset move from here
        move_start_price = closes[i]
        move_start_bar = i
        move_high = closes[i]
        move_low = closes[i]
        continue

    if in_trade:
        continue

    if move_bars < MIN_MOVE_BARS:
        continue

    # Funnel: find matching seed profiles
    # Which magnitude bin?
    mag_bin = None
    for mi, (mlo, mhi) in enumerate(MAG_BINS):
        if mlo <= current_mag < mhi:
            mag_bin = mi
            break
    if mag_bin is None:
        continue

    # Which duration bin?
    dur_bin = None
    for di, (dlo, dhi) in enumerate(DUR_BINS):
        if dlo <= move_bars < dhi:
            dur_bin = di
            break
    if dur_bin is None:
        continue

    # Look up matching seeds
    matching = seed_lookup.get((mag_bin, dur_bin), [])
    if not matching:
        continue

    # How many seeds match? Count direction consensus
    n_long = sum(1 for s in matching if s['direction'] == 'LONG')
    n_short = len(matching) - n_long

    # The seed's direction = what happened DURING the seed
    # Current move is in current_dir
    # If current move matches seed direction, the seed says this move will END
    # So enter OPPOSITE (reversal)
    same_dir = [s for s in matching if s['direction'] == current_dir]
    if len(same_dir) < 3:
        continue  # need at least 3 matching seeds for confidence

    # Consensus: what fraction of matching seeds are same direction?
    consensus = len(same_dir) / len(matching)
    if consensus < 0.4:
        continue  # no clear consensus

    # Entry: opposite of current move (reversal)
    trade_dir = 'SHORT' if current_dir == 'LONG' else 'LONG'

    # Hold: median duration of matching seeds
    median_hold = int(np.median([s['bars'] for s in same_dir]))
    trade_hold = max(3, min(median_hold, 20))

    entry_price = closes[i]
    entry_bar = i
    entry_ts = timestamps[i]
    trade_n_seeds = len(same_dir)
    in_trade = True

    # Reset move
    move_start_price = closes[i]
    move_start_bar = i
    move_high = closes[i]
    move_low = closes[i]

t = pd.DataFrame(trades)
if len(t) == 0:
    print("No trades")
    sys.exit()

wins = t[t.pnl > 0]
total = t.pnl.sum() * TICK_VALUE
n_days = t.date.nunique()

print()
print('=' * 70)
print('SEED FUNNEL OOS — IS seeds matched against OOS price action')
print('Entry: current move matches IS seed profile -> enter reversal')
print('Exit: hold for median matched seed duration')
print('=' * 70)
print(f"  Trades: {len(t):,}")
print(f"  WR: {len(wins)/len(t)*100:.1f}%")
print(f"  Total: {t.pnl.sum():,.0f}t (${total:,.0f})")
print(f"  Avg: {t.pnl.mean():.1f}t (${t.pnl.mean()*TICK_VALUE:.2f})")
print(f"  Avg hold: {t.hold_min.mean():.1f} min")
print(f"  Avg matched seeds: {t.matched_seeds.mean():.0f}")
print(f"  Days: {n_days}")
print(f"  Per day: ${total/n_days:,.0f}")
print()

for d in ['LONG', 'SHORT']:
    sub = t[t.dir == d]
    if len(sub) > 0:
        wr = (sub.pnl > 0).mean() * 100
        print(f"  {d}: {len(sub)} trades, WR={wr:.0f}%, avg={sub.pnl.mean():.1f}t, "
              f"total=${sub.pnl.sum()*TICK_VALUE:,.0f}")

print()
print("  Daily:")
daily = t.groupby('date').agg(n=('pnl', 'count'), pnl=('pnl', 'sum'),
                               wr=('pnl', lambda x: (x > 0).mean() * 100))
daily['usd'] = daily['pnl'] * TICK_VALUE
for date, row in daily.iterrows():
    print(f"    {date}: {row.n:.0f} trades  WR={row.wr:.0f}%  ${row.usd:>+8,.0f}")

print()
print("  COMPARISON:")
print(f"    Blind flip:        $7,967  $73/day")
print(f"    Current system:    $1,844  $56/day")
print(f"    Pure physics 5m:   $-3,420  $-90/day")
print(f"    Seed funnel (this): ${total:,.0f}  ${total/n_days:,.0f}/day")
