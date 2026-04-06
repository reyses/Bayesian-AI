"""
Kill Shot Test — runs NMP+wick rejection strategy on IS/OOS.

Uses KillShotEngine (NMP with wick filter + p_at_center exit).
Same state management as NMP — no re-entry into same excursion.
Zero lookahead. Decisions at 1m boundaries only.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.nightmare_killshot import KillShotEngine

FEATURES_DIR = 'DATA/FEATURES_79D_5s_v2'
ATLAS_1M = 'DATA/ATLAS/1m'


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


def main():
    args = parse_args()
    files = get_day_files(args.target)

    if not files:
        print(f'No files for target={args.target}')
        return

    print(f'KILL SHOT TEST — NMP + Wick Rejection + p_center Exit')
    print(f'  Target: {args.target.upper()} | Days: {len(files)}')
    print()

    engine = KillShotEngine()
    all_results = []
    cumul = 0

    for fpath in tqdm(files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)

        for state in ft:
            engine.on_state(state)

        engine.force_close()

        day_pnl = engine.daily_pnl
        day_n = len(engine.trades)
        cumul += day_pnl

        all_results.append({
            'day': day_name,
            'trades': day_n,
            'pnl': day_pnl,
            'wr': sum(1 for t in engine.trades if t['pnl'] > 0) / max(day_n, 1) * 100,
        })

        if args.verbose and day_n > 0:
            tqdm.write(f'  {day_name}: {engine.summary()}  cumul=${cumul:.0f}')

    # Summary
    n_days = len(all_results)
    all_trades = sum(r['trades'] for r in all_results)
    total_pnl = sum(r['pnl'] for r in all_results)
    active_days = sum(1 for r in all_results if r['trades'] > 0)
    winning_days = sum(1 for r in all_results if r['pnl'] > 0)
    trade_wr = sum(1 for r in all_results for _ in range(1) if r['wr'] > 50)  # rough

    print(f'\n{"="*60}')
    print(f'KILL SHOT RESULTS — {args.target.upper()}')
    print(f'{"="*60}')
    print(f'  Days: {n_days} (active: {active_days})')
    print(f'  Winning days: {winning_days}/{active_days} ({winning_days/max(active_days,1)*100:.0f}%)')
    print(f'  Trades: {all_trades}')
    print(f'  Total PnL: ${total_pnl:,.0f}')
    print(f'  $/day (active): ${total_pnl/max(active_days,1):.0f}')
    print(f'  $/trade: ${total_pnl/max(all_trades,1):.1f}')
    print(f'  Trades/day (active): {all_trades/max(active_days,1):.1f}')

    # Save
    os.makedirs('nn_v2/output/reports', exist_ok=True)
    csv_path = f'nn_v2/output/reports/killshot_{args.target}.csv'
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
