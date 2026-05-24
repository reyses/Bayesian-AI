"""
Run regret analysis on max-fill trades — compute oracle-optimal PnL per tier.

For each trade: loads the day's 1m prices, computes SAME/COUNTER PnL curves,
finds the oracle's best action and optimal PnL. Then reports per-tier
oracle stats so we can see the TRUE potential of each tier.

Usage:
    python tools/maxfill_regret.py
    python tools/maxfill_regret.py --tier KILL_SHOT
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_1M = 'DATA/ATLAS/1m'
TRADES_DIR = 'training/output/trades'
TICK = 0.25
TV = 0.50
LOOKAHEAD = 60  # 60 1m bars (1 hour) to find optimal exit


def load_day_prices(day_name):
    """Load 1m close prices for a day."""
    path = os.path.join(ATLAS_1M, f'{day_name}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    return df


def compute_trade_regret(trade, day_df):
    """Compute oracle-optimal PnL for one trade."""
    entry_price = trade['entry_price']
    direction = trade['dir']
    ts = trade['timestamp']

    # Find entry bar in 1m data
    ts_arr = day_df['timestamp'].values
    entry_idx = np.searchsorted(ts_arr, ts, side='right') - 1
    if entry_idx < 0 or entry_idx >= len(ts_arr):
        return None

    closes = day_df['close'].values
    end_idx = min(entry_idx + LOOKAHEAD, len(closes))

    # PnL curves from entry
    same_pnls = []
    counter_pnls = []
    for i in range(entry_idx, end_idx):
        p = closes[i]
        if direction == 'long':
            same = (p - entry_price) / TICK * TV
            counter = (entry_price - p) / TICK * TV
        else:
            same = (entry_price - p) / TICK * TV
            counter = (p - entry_price) / TICK * TV
        same_pnls.append(same)
        counter_pnls.append(counter)

    if not same_pnls:
        return None

    same_pnls = np.array(same_pnls)
    counter_pnls = np.array(counter_pnls)

    same_best = float(same_pnls.max())
    same_best_bar = int(same_pnls.argmax())
    counter_best = float(counter_pnls.max())
    counter_best_bar = int(counter_pnls.argmax())

    if same_best >= counter_best:
        oracle_pnl = same_best
        oracle_bar = same_best_bar
        oracle_action = 'same'
        oracle_dir = direction
    else:
        oracle_pnl = counter_best
        oracle_bar = counter_best_bar
        oracle_action = 'counter'
        oracle_dir = 'short' if direction == 'long' else 'long'

    return {
        'oracle_pnl': oracle_pnl,
        'oracle_bar': oracle_bar,
        'oracle_action': oracle_action,
        'oracle_dir': oracle_dir,
        'same_best': same_best,
        'counter_best': counter_best,
        'actual_pnl': trade['pnl'],
        'regret': oracle_pnl - trade['pnl'],
    }


def run_regret_for_tier(tier_name, trades):
    """Run regret on all trades for one tier."""
    # Group by day for efficient price loading
    by_day = defaultdict(list)
    for t in trades:
        by_day[t['day']].append(t)

    results = []
    for day_name in tqdm(sorted(by_day.keys()), desc=f'  {tier_name}', unit='day', leave=False):
        day_df = load_day_prices(day_name)
        if day_df is None:
            continue

        for t in by_day[day_name]:
            r = compute_trade_regret(t, day_df)
            if r is not None:
                r['tier'] = tier_name
                r['day'] = day_name
                results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=str, nargs='*', default=None)
    args = parser.parse_args()

    tiers = ['cascade', 'kill_shot', 'fade_against', 'fade_calm', 'fade_momentum',
             'freight_train', 'ride_against', 'absorption', 'regime_flip',
             'exhaustion_bar', 'mtf_exhaustion']

    if args.tier:
        tiers = [t.lower() for t in args.tier]

    print(f'MAXFILL REGRET ANALYSIS')
    print(f'  Lookahead: {LOOKAHEAD} bars (1m)')
    print()

    all_results = []

    for tier in tiers:
        path = os.path.join(TRADES_DIR, f'maxfill_{tier}.pkl')
        if not os.path.exists(path):
            continue

        with open(path, 'rb') as f:
            trades = pickle.load(f)

        results = run_regret_for_tier(tier.upper(), trades)
        all_results.extend(results)

    if not all_results:
        print('No results.')
        return

    df = pd.DataFrame(all_results)

    # Per-tier oracle report
    print(f'\n{"="*80}')
    print(f'ORACLE REPORT — what each tier is WORTH with perfect exits')
    print(f'{"="*80}')
    print(f'{"Tier":<20} {"Trades":>7} {"Actual$/tr":>11} {"Oracle$/tr":>11} '
          f'{"Regret$/tr":>11} {"Oracle Total":>13} {"Same%":>6}')
    print(f'{"-"*80}')

    tier_summary = []
    for tier_name, grp in df.groupby('tier'):
        n = len(grp)
        actual_avg = grp['actual_pnl'].mean()
        oracle_avg = grp['oracle_pnl'].mean()
        regret_avg = grp['regret'].mean()
        oracle_total = grp['oracle_pnl'].sum()
        same_pct = (grp['oracle_action'] == 'same').mean() * 100

        tier_summary.append((tier_name, n, actual_avg, oracle_avg, regret_avg, oracle_total, same_pct))
        print(f'{tier_name:<20} {n:>7} {actual_avg:>11.1f} {oracle_avg:>11.1f} '
              f'{regret_avg:>11.1f} {oracle_total:>13,.0f} {same_pct:>5.0f}%')

    total_oracle = df['oracle_pnl'].sum()
    total_actual = df['actual_pnl'].sum()
    print(f'{"-"*80}')
    print(f'{"TOTAL":<20} {len(df):>7} {total_actual/len(df):>11.1f} '
          f'{total_oracle/len(df):>11.1f} {"":>11} {total_oracle:>13,.0f}')
    print(f'\nOracle: ${total_oracle/277:.0f}/day | Actual: ${total_actual/277:.0f}/day | '
          f'Gap: ${(total_oracle-total_actual)/277:.0f}/day')

    # Per-tier: oracle direction breakdown
    print(f'\nORACLE DIRECTION (should we flip?):')
    for tier_name, grp in df.groupby('tier'):
        same = grp[grp['oracle_action'] == 'same']
        counter = grp[grp['oracle_action'] == 'counter']
        print(f'  {tier_name:<20} SAME={len(same):>5} (${same["oracle_pnl"].mean():.0f}/tr)  '
              f'COUNTER={len(counter):>5} (${counter["oracle_pnl"].mean():.0f}/tr)')

    # Save
    out_path = os.path.join(TRADES_DIR, 'maxfill_regret.csv')
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
