"""Standalone L5Decider zigzag diagnostic.

Replays a single day's 5s bars through the L5Decider's pivot/R-trigger
detector and reports:
  - count of pivots / R-triggers fired
  - per-pivot timing + price + r_price at time of fire
  - comparison against the production pivot dataset (zigzag_pivot_dataset_NT8_OOS_atr4.parquet)

Usage:
    python tools/_diag_l5_zigzag.py
    python tools/_diag_l5_zigzag.py --day 2026_05_11
"""
from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from live.l5_decider import L5Decider, L5Context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default='2026_05_11')
    args = ap.parse_args()

    ctx = L5Context.load()
    dec = L5Decider(ctx)

    # Prime ATR from prior days' 1m history (matches production builder)
    import glob
    all_1m_files = sorted(glob.glob('DATA/ATLAS_NT8/1m/*.parquet'))
    prior_files = [f for f in all_1m_files
                    if os.path.basename(f).replace('.parquet', '') < args.day]
    if prior_files:
        # Load last ~50 days for solid ATR warmup
        recent = prior_files[-50:]
        dfs = [pd.read_parquet(f) for f in recent]
        prior_1m = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
        prior_1m = prior_1m.drop_duplicates(subset='timestamp', keep='last')
        dec.prime_atr_from_history(prior_1m)
        print(f'  ATR primed from {len(prior_1m)} 1m bars '
              f'({len(recent)} prior days), r_price={dec._r_price}')

    bars = pd.read_parquet(f'DATA/ATLAS_NT8/5s/{args.day}.parquet')
    print(f'Replaying {len(bars)} 5s bars for {args.day}...')

    fires = []
    r_prices = []
    for i, row in bars.iterrows():
        ts = float(row['timestamp'])
        price = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        vol = int(row['volume'])

        dec._update_1m_aggregator(ts, price, high, low, vol)
        dec._this_bar_rtrig_dir = None
        dec._this_bar_rtrig_price = None
        dec._update_zigzag(ts, price)

        if dec._r_price is not None and i % 1000 == 0:
            r_prices.append(dec._r_price)

        if dec._this_bar_rtrig_dir is not None:
            fires.append({
                'i': i,
                'ts': ts,
                'rtrig_price': dec._this_bar_rtrig_price,
                'rtrig_dir': dec._this_bar_rtrig_dir,
                'r_price': dec._r_price,
                'min_rev_ticks': dec._min_rev_ticks,
            })

    print(f'\nL5Decider R-trigger fires for {args.day}: {len(fires)}')
    if r_prices:
        print(f'r_price samples: min={min(r_prices):.2f} max={max(r_prices):.2f} '
              f'mean={sum(r_prices)/len(r_prices):.2f}')

    # Compare against production OOS pivots
    truth_path = f'reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet'
    if os.path.exists(truth_path):
        truth = pd.read_parquet(truth_path)
        truth_day = truth[truth['day'] == args.day]
        prod_pivots = truth_day[truth_day['is_pivot'] == 1]
        # Dedup contiguous (within 60s)
        prod_ts = prod_pivots['timestamp'].values.astype(np.int64)
        prod_ts_dedup = []
        for t in sorted(prod_ts):
            if not prod_ts_dedup or t - prod_ts_dedup[-1] > 60:
                prod_ts_dedup.append(t)
        print(f'Production pivot events for {args.day}: {len(prod_ts_dedup)} (dedup, '
              f'raw {len(prod_pivots)})')

        # Side-by-side first 15 fires vs prod
        print('\nL5 fires vs production pivots (first 15 each, ts ordered):')
        print(f'{"L5 fire ts":>12}  {"L5 dir":<6}  {"L5 price":>9}  '
              f'{"L5 r_price":>10}    {"PROD ts":>12}  {"PROD dir":<6}')
        max_show = max(len(fires), len(prod_ts_dedup))
        prod_lookup = list(zip(prod_pivots['timestamp'].values[:max_show],
                                  prod_pivots['pivot_dir'].values[:max_show]))
        for k in range(min(15, max(len(fires), len(prod_ts_dedup)))):
            l5 = fires[k] if k < len(fires) else None
            pr = prod_lookup[k] if k < len(prod_lookup) else None
            l5_str = (f'{int(l5["ts"]):>12}  {l5["rtrig_dir"]:<6}  '
                       f'{l5["rtrig_price"]:>9.2f}  {l5["r_price"]:>10.2f}') \
                       if l5 else f'{"":>12}  {"":<6}  {"":>9}  {"":>10}'
            pr_str = (f'{int(pr[0]):>12}  {pr[1]:<6}') if pr else ''
            print(f'  {l5_str}    {pr_str}')

        # Calibration metrics
        if fires and prod_ts_dedup:
            ratio = len(fires) / len(prod_ts_dedup)
            print(f'\nCAPTURE RATIO: {ratio*100:.0f}% '
                  f'(L5/prod = {len(fires)}/{len(prod_ts_dedup)})')
    else:
        print(f'\n(no production truth file at {truth_path})')


if __name__ == '__main__':
    main()
