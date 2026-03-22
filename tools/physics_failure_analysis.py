"""Analyze why worst days fail vs why best days succeed in pure physics."""
import os, sys, glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine

TICK = 0.25

# IS magnitude distribution
print('Loading IS...')
is_1m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))], ignore_index=True)
is_1m['ts_5m'] = (is_1m['timestamp'] // 300) * 300
is_5m = is_1m.groupby('ts_5m').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'timestamp': 'first'}).reset_index(drop=True)
del is_1m

engine = StatisticalFieldEngine()
raw_is = engine.batch_compute_states(is_5m)
states_is = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_is]

prior_mags = []
prev_fm = 0
ms = is_5m['close'].values[0]
for i in range(1, len(states_is)):
    s = states_is[i]
    if s is None: continue
    fm = s.F_momentum
    if prev_fm != 0 and fm != 0 and np.sign(prev_fm) != np.sign(fm):
        prior_mags.append(abs(is_5m['close'].values[i] - ms) / TICK)
        ms = is_5m['close'].values[i]
    prev_fm = fm
del is_5m, raw_is, states_is

# OOS
print('Loading OOS...')
oos_1m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS_OOS/1m/*.parquet'))], ignore_index=True)
oos_1m['ts_5m'] = (oos_1m['timestamp'] // 300) * 300
oos_5m = oos_1m.groupby('ts_5m').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'timestamp': 'first'}).reset_index(drop=True)
del oos_1m

raw_oos = engine.batch_compute_states(oos_5m)
states = [s['state'] if s and isinstance(s, dict) and 'state' in s else None for s in raw_oos]
closes = oos_5m['close'].values
ts = oos_5m['timestamp'].values

mag_window = deque(prior_mags[-200:], maxlen=200)

# Detect + trade
peaks = []
prev_fm = 0
move_start = closes[0]
for i in range(1, len(states)):
    s = states[i]
    if s is None: continue
    fm = s.F_momentum
    if prev_fm != 0 and fm != 0 and np.sign(prev_fm) != np.sign(fm):
        prior_move = abs(closes[i] - move_start) / TICK
        pctile = sum(1 for m in mag_window if m < prior_move) / max(len(mag_window), 1)
        peaks.append({
            'bar': i, 'ts': ts[i], 'price': closes[i],
            'prior_move': prior_move, 'direction': 'LONG' if prev_fm < 0 else 'SHORT',
            'pctile': pctile, 'is_big': pctile >= 0.50,
            'fm': fm, 'dmi_diff': s.dmi_plus - s.dmi_minus,
            'adx': s.adx_strength, 'z': s.z_score, 'vol': s.volume_delta,
        })
        mag_window.append(prior_move)
        move_start = closes[i]
    prev_fm = fm

pk = pd.DataFrame(peaks)
big = pk[pk.is_big]

trades = []
in_trade = False
entry_price = 0.0
trade_dir = ''
entry_ts_val = 0.0
es = {}

for _, p in big.iterrows():
    if in_trade:
        pnl = ((p.price - entry_price) if trade_dir == 'LONG' else (entry_price - p.price)) / TICK
        date = datetime.fromtimestamp(entry_ts_val, tz=timezone.utc).strftime('%Y-%m-%d')
        trades.append({
            'pnl': pnl, 'dir': trade_dir, 'date': date,
            'hold_min': (p.ts - entry_ts_val) / 60,
            'entry_dmi': es.get('dmi_diff', 0), 'entry_adx': es.get('adx', 0),
            'entry_z': es.get('z', 0), 'entry_vol': es.get('vol', 0),
            'entry_fm': es.get('fm', 0), 'entry_prior_move': es.get('prior_move', 0),
        })
        in_trade = False
    trade_dir = p.direction
    entry_price = p.price
    entry_ts_val = p.ts
    es = p.to_dict()
    in_trade = True

t = pd.DataFrame(trades)

worst_days = ['2026-02-03', '2026-02-16', '2026-02-19', '2026-02-20', '2026-03-17']
best_days = ['2026-02-21', '2026-03-09', '2026-02-05', '2026-02-06', '2026-02-26']

print()
print('=' * 70)
print('WORST vs BEST DAYS')
print('=' * 70)

for label, days in [('WORST', worst_days), ('BEST', best_days)]:
    sub = t[t.date.isin(days)]
    if len(sub) == 0: continue
    wins = sub[sub.pnl > 0]
    longs = sub[sub.dir == 'LONG']
    shorts = sub[sub.dir == 'SHORT']
    print(f"\n=== {label} ({len(sub)} trades) ===")
    print(f"  PnL: ${sub.pnl.sum()*0.50:+,.0f} | WR: {len(wins)/len(sub)*100:.0f}%")
    print(f"  Hold: {sub.hold_min.mean():.0f} min")
    l_wr = (longs.pnl > 0).mean() * 100 if len(longs) > 0 else 0
    s_wr = (shorts.pnl > 0).mean() * 100 if len(shorts) > 0 else 0
    print(f"  LONG:  n={len(longs)} WR={l_wr:.0f}% avg={longs.pnl.mean():.1f}t")
    print(f"  SHORT: n={len(shorts)} WR={s_wr:.0f}% avg={shorts.pnl.mean():.1f}t")
    for f in ['entry_dmi', 'entry_adx', 'entry_z', 'entry_vol', 'entry_fm', 'entry_prior_move']:
        print(f"    |{f}|: {sub[f].abs().mean():.1f}")

print()
print('=== FEATURE SEPARATION ===')
worst = t[t.date.isin(worst_days)]
best = t[t.date.isin(best_days)]
print(f"  {'Feature':<22} {'Worst':>8} {'Best':>8} {'Delta':>8}")
for f in ['entry_dmi', 'entry_adx', 'entry_z', 'entry_vol', 'entry_fm', 'entry_prior_move', 'hold_min']:
    w = worst[f].abs().mean() if 'entry' in f else worst[f].mean()
    b = best[f].abs().mean() if 'entry' in f else best[f].mean()
    print(f"  {f:<22} {w:>8.1f} {b:>8.1f} {b-w:>+8.1f}")

print()
print('=== BIG LOSERS vs BIG WINNERS ===')
losers = t[t.pnl < -50]
winners = t[t.pnl > 50]
print(f"Big losers (<-50t): {len(losers)}")
print(f"Big winners (>+50t): {len(winners)}")
print(f"  {'Feature':<22} {'Loser':>8} {'Winner':>8} {'Delta':>8}")
for f in ['entry_dmi', 'entry_adx', 'entry_z', 'entry_vol', 'entry_fm', 'entry_prior_move', 'hold_min']:
    w = losers[f].abs().mean() if 'entry' in f else losers[f].mean()
    b = winners[f].abs().mean() if 'entry' in f else winners[f].mean()
    print(f"  {f:<22} {w:>8.1f} {b:>8.1f} {b-w:>+8.1f}")
