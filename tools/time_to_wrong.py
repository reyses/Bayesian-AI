"""
Time-to-wrong diagnostic.

For each trade, walk forward 1s bars from entry and record:
  - seconds to first cross of -$1, -$3, -$5, -$10 (adverse)
  - seconds to first cross of +$1, +$3, +$5, +$10 (favorable)

Split by final outcome (winner / loser). Answer:
  "On average, how fast do losers tell us they're losing?"

Usage:
    python tools/time_to_wrong.py
Output:
    research/rm_pivot/findings/YYYY-MM-DD_time_to_wrong.md
"""
import os
import sys
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ATLAS_1S_DIR = 'DATA/ATLAS/1s'
TRADES_PKL = 'training_RM_physics/output/trades/rm_is.pkl'
DPP = 2.0  # $/point

# Thresholds in POINTS (dollars / $2)
ADVERSE_PTS = [-0.5, -1.5, -2.5, -5.0]   # -$1, -$3, -$5, -$10
FAVORABLE_PTS = [0.5, 1.5, 2.5, 5.0]     # +$1, +$3, +$5, +$10


def main():
    with open(TRADES_PKL, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades)} trades')

    # Group trades by day
    by_day = defaultdict(list)
    for t in trades:
        by_day[t['day']].append(t)

    # For each trade, compute time-to-threshold
    rows = []
    for day in tqdm(sorted(by_day.keys()), desc='Days'):
        day_trades = by_day[day]
        path_1s = os.path.join(ATLAS_1S_DIR, f'{day}.parquet')
        if not os.path.exists(path_1s):
            continue
        df = pd.read_parquet(path_1s).sort_values('timestamp').reset_index(drop=True)
        ts = df['timestamp'].values.astype(np.int64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)

        for tr in day_trades:
            entry_ts = int(tr['timestamp'])
            entry_price = float(tr['entry_price'])
            direction = tr['dir']
            final_pnl = float(tr['pnl'])

            # Find first 1s bar at or after entry
            start_idx = int(np.searchsorted(ts, entry_ts, side='left'))
            if start_idx >= len(ts):
                continue

            # We walk forward for min(held_minutes * 60 + 120, end_of_day) seconds
            held_sec = max(60, int(tr['held']) * 60 + 120)
            end_idx = min(start_idx + held_sec, len(ts))

            # Record time to first cross of each adverse / favorable threshold.
            # Use 1s high/low so crosses are caught intra-bar.
            # Adverse for long = (price below entry by X pts) → use 1s low.
            # Favorable for long = (price above entry by X pts) → use 1s high.
            # For short: swap.
            adverse_times = {a: None for a in ADVERSE_PTS}
            favorable_times = {f: None for f in FAVORABLE_PTS}

            for i in range(start_idx, end_idx):
                secs = int(ts[i] - entry_ts)
                if secs < 0:
                    continue
                # Signed PnL of current 1s bar high/low relative to entry
                if direction == 'long':
                    adverse_move = low[i] - entry_price   # negative = adverse
                    favorable_move = high[i] - entry_price  # positive = favorable
                else:
                    adverse_move = entry_price - high[i]
                    favorable_move = entry_price - low[i]
                # Check adverse thresholds (fill in first time reached)
                for a in ADVERSE_PTS:
                    if adverse_times[a] is None and adverse_move <= a:
                        adverse_times[a] = secs
                for f in FAVORABLE_PTS:
                    if favorable_times[f] is None and favorable_move >= f:
                        favorable_times[f] = secs
                # Early exit if all thresholds filled
                if (all(v is not None for v in adverse_times.values()) and
                    all(v is not None for v in favorable_times.values())):
                    break

            row = {'day': day, 'final_pnl': final_pnl, 'dir': direction,
                   'exit_reason': tr.get('exit_reason', '?')}
            for a in ADVERSE_PTS:
                dollars = -int(a * DPP)
                row[f'adverse_{dollars}'] = adverse_times[a]
            for f in FAVORABLE_PTS:
                dollars = int(f * DPP)
                row[f'favorable_{dollars}'] = favorable_times[f]
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print('No data.')
        return

    # Split winners / losers by final PnL sign
    winners = df[df.final_pnl > 0]
    losers = df[df.final_pnl < 0]

    lines = []
    lines.append('# Time-to-wrong diagnostic')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'Trades analyzed: {len(df)} (winners {len(winners)}, losers {len(losers)})')
    lines.append('')
    lines.append('## Seconds to first adverse threshold')
    lines.append('')
    lines.append('(How fast does price cross below entry by N dollars, counting 1s bar lows)')
    lines.append('')
    lines.append('| Threshold | N winners crossed | Win median (s) | Win p25 | Win p75 '
                 '| N losers crossed | Lose median (s) | Lose p25 | Lose p75 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for a in ADVERSE_PTS:
        dollars = -int(a * DPP)
        col = f'adverse_{dollars}'
        w = winners[col].dropna().values
        l = losers[col].dropna().values
        if len(w) == 0 or len(l) == 0:
            continue
        lines.append(
            f'| −${dollars} | {len(w)} | {int(np.median(w))} | {int(np.percentile(w,25))} | {int(np.percentile(w,75))} '
            f'| {len(l)} | {int(np.median(l))} | {int(np.percentile(l,25))} | {int(np.percentile(l,75))} |'
        )
    lines.append('')
    lines.append('## Seconds to first favorable threshold')
    lines.append('')
    lines.append('| Threshold | N winners crossed | Win median (s) | Win p25 | Win p75 '
                 '| N losers crossed | Lose median (s) | Lose p25 | Lose p75 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for f in FAVORABLE_PTS:
        dollars = int(f * DPP)
        col = f'favorable_{dollars}'
        w = winners[col].dropna().values
        l = losers[col].dropna().values
        if len(w) == 0 or len(l) == 0:
            continue
        lines.append(
            f'| +${dollars} | {len(w)} | {int(np.median(w))} | {int(np.percentile(w,25))} | {int(np.percentile(w,75))} '
            f'| {len(l)} | {int(np.median(l))} | {int(np.percentile(l,25))} | {int(np.percentile(l,75))} |'
        )
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append('If losers cross −$X much faster than winners cross +$X, we have an '
                 '**early-detection signal**: "if down $X in Y seconds, flip."')
    lines.append('')

    out_dir = 'research/rm_pivot/findings'
    os.makedirs(out_dir, exist_ok=True)
    date_tag = datetime.now().strftime('%Y-%m-%d')
    out_path = os.path.join(out_dir, f'{date_tag}_time_to_wrong.md')
    with open(out_path, 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(lines))
    print(f'Wrote: {out_path}')
    print()
    print('\n'.join(lines[-30:]))


if __name__ == '__main__':
    main()
