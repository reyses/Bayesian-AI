"""
Kill Shot Test — standalone NMP rejection strategy.

Runs ONE strategy on IS/OOS data with strict no-lookahead:
  - Reads 5s features sequentially (bar by bar)
  - Decisions only at 1m boundaries
  - Entry: 5m_wick_ratio > 0.83 AND 15m_wick_ratio > 0.77
           AND |1m_z_se| > 2.0 AND 1m_variance_ratio < 1.0
  - Direction: fade the z (z > 0 → short, z < 0 → long)
  - Exit: 1m_p_at_center > 0.60 (price reached center)
          OR end of day

All decisions use ONLY features available at decision time.
No future information. No regret. No tree. Pure physics.

Usage:
    python tools/killshot_test.py                    # IS only
    python tools/killshot_test.py --target oos       # OOS
    python tools/killshot_test.py --target all       # both
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

FEATURES_DIR = 'DATA/FEATURES_79D_5s_v2'
TICK = 0.25
TV = 0.50

# 79D feature indices
IDX = {name: i for i, name in enumerate(FEATURE_NAMES_79D)}

# Entry conditions (NMP kill shot)
ENTRY_Z_THRESHOLD = 2.0          # |1m_z_se| must exceed this
ENTRY_VR_THRESHOLD = 1.0         # 1m_variance_ratio must be below this
ENTRY_5M_WICK_MIN = 0.83         # 5m rejection candle
ENTRY_15M_WICK_MIN = 0.77        # 15m rejection confirms

# Exit condition
EXIT_P_CENTER_THRESHOLD = 0.60   # price reached near regression center


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Kill shot strategy test')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'])
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def get_day_files(target='is'):
    """Get feature files for target days."""
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    if target == 'is':
        files = [f for f in files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        files = [f for f in files if '2026_' in os.path.basename(f)]
    return files


def run_one_day(feat_file, verbose=False):
    """Run kill shot strategy on one day. Strict sequential, no lookahead.

    Returns list of trade dicts.
    """
    df = pd.read_parquet(feat_file).sort_values('timestamp').reset_index(drop=True)
    feats = df[FEATURE_NAMES_79D].values
    timestamps = df['timestamp'].values
    n = len(df)

    # State
    in_pos = False
    direction = None
    entry_price = 0.0
    entry_bar = 0
    entry_ts = 0.0
    peak_pnl = 0.0
    trades = []

    # We need price — approximate from z_se and band position
    # Actually we don't have raw price in the features. We need it for PnL.
    # Use the approach: track PnL via z_se movement relative to entry z_se.
    # Better: load price from ATLAS
    day_name = os.path.basename(feat_file).replace('.parquet', '')
    price_file = os.path.join('DATA/ATLAS/5s', f'{day_name}.parquet')
    if not os.path.exists(price_file):
        # Try 1m
        price_file = os.path.join('DATA/ATLAS/1m', f'{day_name}.parquet')
    if not os.path.exists(price_file):
        return []

    price_df = pd.read_parquet(price_file).sort_values('timestamp').reset_index(drop=True)
    price_ts = price_df['timestamp'].values
    prices = price_df['close'].values

    def get_price(ts):
        """Get price at timestamp from price data. No lookahead — uses latest available."""
        idx = int(np.searchsorted(price_ts, ts, side='right')) - 1
        if idx < 0:
            return prices[0]
        return prices[min(idx, len(prices) - 1)]

    for bar_idx in range(n):
        ts = timestamps[bar_idx]
        feat = feats[bar_idx]

        # Only make decisions at 1m boundaries
        is_1m = (int(ts) % 60) < 5

        # Read features (available NOW, no lookahead)
        z_1m = feat[IDX['1m_z_se']]
        vr_1m = feat[IDX['1m_variance_ratio']]
        wick_5m = feat[IDX['5m_wick_ratio']]
        wick_15m = feat[IDX['15m_wick_ratio']]
        p_center_1m = feat[IDX['1m_p_at_center']]

        price = get_price(ts)

        if in_pos:
            # Track PnL
            if direction == 'long':
                pnl = (price - entry_price) / TICK * TV
            else:
                pnl = (entry_price - price) / TICK * TV
            peak_pnl = max(peak_pnl, pnl)

            # EXIT CHECK — only at 1m boundaries
            if is_1m:
                exit_reason = None

                # Physics exit: price reached center
                if p_center_1m > EXIT_P_CENTER_THRESHOLD:
                    exit_reason = 'center_reached'

                if exit_reason:
                    time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')
                    trades.append({
                        'day': day_name,
                        'time': time_str,
                        'timestamp': ts,
                        'dir': direction,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'held': bar_idx - entry_bar,
                        'peak': peak_pnl,
                        'exit': exit_reason,
                        # Features at exit (for analysis)
                        'exit_z_1m': z_1m,
                        'exit_p_center': p_center_1m,
                        'exit_wick_5m': wick_5m,
                    })
                    in_pos = False
                    direction = None

        else:
            # ENTRY CHECK — only at 1m boundaries
            if is_1m:
                # All conditions must be true AT THIS BAR (no lookahead)
                if (abs(z_1m) > ENTRY_Z_THRESHOLD and
                    vr_1m < ENTRY_VR_THRESHOLD and
                    wick_5m > ENTRY_5M_WICK_MIN and
                    wick_15m > ENTRY_15M_WICK_MIN):

                    # Direction: fade the z
                    direction = 'short' if z_1m > 0 else 'long'
                    entry_price = price
                    entry_bar = bar_idx
                    entry_ts = ts
                    peak_pnl = 0.0
                    in_pos = True

    # Force close at end of day
    if in_pos:
        price = get_price(timestamps[-1])
        if direction == 'long':
            pnl = (price - entry_price) / TICK * TV
        else:
            pnl = (entry_price - price) / TICK * TV
        time_str = datetime.utcfromtimestamp(timestamps[-1]).strftime('%H:%M')
        trades.append({
            'day': day_name,
            'time': time_str,
            'timestamp': timestamps[-1],
            'dir': direction,
            'entry_price': entry_price,
            'exit_price': price,
            'pnl': pnl,
            'held': n - entry_bar,
            'peak': peak_pnl,
            'exit': 'end_of_day',
            'exit_z_1m': feat[IDX['1m_z_se']],
            'exit_p_center': feat[IDX['1m_p_at_center']],
            'exit_wick_5m': feat[IDX['5m_wick_ratio']],
        })

    return trades


def main():
    args = parse_args()
    files = get_day_files(args.target)

    if not files:
        print(f'No files for target={args.target}')
        return

    print(f'KILL SHOT TEST — NMP Rejection Strategy')
    print(f'  Target: {args.target.upper()} | Days: {len(files)}')
    print(f'  Entry: |z_1m|>{ENTRY_Z_THRESHOLD} + vr<{ENTRY_VR_THRESHOLD} + '
          f'5m_wick>{ENTRY_5M_WICK_MIN} + 15m_wick>{ENTRY_15M_WICK_MIN}')
    print(f'  Exit: 1m_p_center>{EXIT_P_CENTER_THRESHOLD}')
    print()

    all_trades = []
    daily_results = []

    for fpath in tqdm(files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        trades = run_one_day(fpath, verbose=args.verbose)
        all_trades.extend(trades)

        day_pnl = sum(t['pnl'] for t in trades)
        day_n = len(trades)
        day_wr = sum(1 for t in trades if t['pnl'] > 0) / max(day_n, 1) * 100

        daily_results.append({
            'day': day_name, 'trades': day_n, 'pnl': day_pnl, 'wr': day_wr,
        })

        if args.verbose and day_n > 0:
            tqdm.write(f'  {day_name}: {day_n} trades  WR={day_wr:.0f}%  ${day_pnl:.0f}')

    # Summary
    n_days = len(daily_results)
    n_trades = len(all_trades)
    total_pnl = sum(t['pnl'] for t in all_trades)
    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    active_days = sum(1 for d in daily_results if d['trades'] > 0)
    winning_days = sum(1 for d in daily_results if d['pnl'] > 0)

    print(f'\n{"="*60}')
    print(f'KILL SHOT RESULTS — {args.target.upper()}')
    print(f'{"="*60}')
    print(f'  Days: {n_days} (active: {active_days})')
    print(f'  Winning days: {winning_days}/{active_days} ({winning_days/max(active_days,1)*100:.0f}%)')
    print(f'  Trades: {n_trades}')
    print(f'  Trade WR: {wins}/{n_trades} ({wins/max(n_trades,1)*100:.0f}%)')
    print(f'  Total PnL: ${total_pnl:,.0f}')
    print(f'  $/day (active): ${total_pnl/max(active_days,1):.0f}')
    print(f'  $/trade: ${total_pnl/max(n_trades,1):.1f}')

    if all_trades:
        held = [t['held'] for t in all_trades]
        print(f'  Avg held: {np.mean(held):.0f} bars ({np.mean(held)*5/60:.1f} min)')
        print(f'  Avg peak: ${np.mean([t["peak"] for t in all_trades]):.1f}')

        # Exit reason breakdown
        from collections import Counter
        exits = Counter(t['exit'] for t in all_trades)
        print(f'\n  Exit reasons:')
        for reason, count in exits.most_common():
            sub = [t for t in all_trades if t['exit'] == reason]
            sub_wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            sub_pnl = sum(t['pnl'] for t in sub)
            print(f'    {reason:<20} {count:>4} ({sub_wr:.0f}% WR)  ${sub_pnl:,.0f}')

    # Save results
    os.makedirs('nn_v2/output/reports', exist_ok=True)
    report_path = f'nn_v2/output/reports/killshot_{args.target}.csv'
    pd.DataFrame(daily_results).to_csv(report_path, index=False)
    print(f'\nCSV: {report_path}')

    if all_trades:
        trade_path = f'nn_v2/output/reports/killshot_{args.target}_trades.csv'
        pd.DataFrame(all_trades).to_csv(trade_path, index=False)
        print(f'Trades: {trade_path}')


if __name__ == '__main__':
    main()
