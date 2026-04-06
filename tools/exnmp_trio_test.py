"""
ExNMP Trio Test — runs all 3 ExNMP strategies on IS/OOS sequentially.

1. NMP Kill Shot:    wick rejection + p_center exit
2. NMP Overshoot:    wick entry + overshoot exit (opposite extreme / momentum)
3. NMP Cascade:      wick + 1h alignment + p_center exit

Each runs independently. Output: per-strategy results + combined.

Usage:
    python tools/exnmp_trio_test.py                    # IS
    python tools/exnmp_trio_test.py --target oos       # OOS
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.nightmare_killshot import KillShotEngine
from nn_v2.nightmare_wick_overshoot import WickOvershootEngine
from nn_v2.nightmare_cascade import CascadeEngine

FEATURES_DIR = 'DATA/FEATURES_79D_5s_v2'
ATLAS_1M = 'DATA/ATLAS/1m'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='ExNMP trio test')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'])
    return p.parse_args()


def get_day_files(target='is'):
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    if target == 'is':
        files = [f for f in files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        files = [f for f in files if '2026_' in os.path.basename(f)]
    return files


def run_strategy(engine, files, label):
    """Run one strategy on all days. Returns (daily_results, all_trades)."""
    all_results = []
    all_trades = []
    cumul = 0

    for fpath in tqdm(files, desc=label, unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        pf = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(pf):
            pf = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=pf)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        day_pnl = engine.daily_pnl
        day_n = len(engine.trades)
        cumul += day_pnl

        for t in engine.trades:
            t['day'] = day_name
            t['strategy'] = label
        all_trades.extend(engine.get_full_trades())

        all_results.append({
            'day': day_name, 'trades': day_n, 'pnl': day_pnl,
            'wr': sum(1 for t in engine.trades if t['pnl'] > 0) / max(day_n, 1) * 100,
        })

    return all_results, all_trades


def print_summary(label, results, trades):
    """Print strategy summary."""
    n_days = len(results)
    active = sum(1 for r in results if r['trades'] > 0)
    winning = sum(1 for r in results if r['pnl'] > 0)
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)

    print(f'\n  {label}:')
    print(f'    Days: {active} active | Winning: {winning}/{active} ({winning/max(active,1)*100:.0f}%)')
    print(f'    Trades: {total_trades} | WR: {wins}/{total_trades} ({wins/max(total_trades,1)*100:.0f}%)')
    print(f'    Total: ${total_pnl:,.0f} | $/day: ${total_pnl/max(active,1):.0f} | $/trade: ${total_pnl/max(total_trades,1):.1f}')
    print(f'    Trades/day: {total_trades/max(active,1):.1f}')


def main():
    args = parse_args()
    files = get_day_files(args.target)

    if not files:
        print(f'No files for target={args.target}')
        return

    print(f'ExNMP TRIO TEST — {args.target.upper()} ({len(files)} days)')
    print(f'{"="*60}')

    strategies = [
        ('KILL_SHOT', KillShotEngine()),
        ('WICK_OVERSHOOT', WickOvershootEngine()),
        ('CASCADE', CascadeEngine()),
    ]

    all_strategy_trades = {}

    for label, engine in strategies:
        results, trades = run_strategy(engine, files, label)
        all_strategy_trades[label] = (results, trades)
        print_summary(label, results, trades)

    # Combined summary
    print(f'\n{"="*60}')
    print(f'COMPARISON — {args.target.upper()}')
    print(f'{"="*60}')
    print(f'  {"Strategy":<20} {"Trades":>7} {"WR":>6} {"$/trade":>8} {"$/day":>7} {"Total$":>9}')
    print(f'  {"-"*60}')
    for label in ['KILL_SHOT', 'WICK_OVERSHOOT', 'CASCADE']:
        results, trades = all_strategy_trades[label]
        active = sum(1 for r in results if r['trades'] > 0)
        total_pnl = sum(r['pnl'] for r in results)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        n = len(trades)
        wr = wins / max(n, 1) * 100
        avg = total_pnl / max(n, 1)
        per_day = total_pnl / max(active, 1)
        print(f'  {label:<20} {n:>7} {wr:>5.0f}% ${avg:>7.1f} ${per_day:>6.0f} ${total_pnl:>8,.0f}')

    # Save
    os.makedirs('nn_v2/output/reports', exist_ok=True)
    for label in all_strategy_trades:
        results, trades = all_strategy_trades[label]
        csv_path = f'nn_v2/output/reports/{label.lower()}_{args.target}.csv'
        pd.DataFrame(results).to_csv(csv_path, index=False)

        # Save trades for regret analysis
        import pickle
        trade_path = f'nn_v2/output/reports/{label.lower()}_{args.target}_trades.pkl'
        with open(trade_path, 'wb') as f:
            pickle.dump(trades, f)

    print(f'\nResults saved to nn_v2/output/reports/')


if __name__ == '__main__':
    main()
