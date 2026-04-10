"""
Blended Engine Test — one NMP engine with tiered exits.

Outputs CSV with entry_tier and exit_reason per trade.

Usage:
    python tools/blended_test.py                    # IS
    python tools/blended_test.py --target oos       # OOS
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.nightmare_blended import BlendedEngine

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Blended engine test')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'])
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def get_day_files(target='is'):
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

    print(f'BLENDED ENGINE TEST — {args.target.upper()} ({len(files)} days)')
    print()

    engine = BlendedEngine()
    all_results = []
    all_trades = []
    cumul = 0

    for fpath in tqdm(files, desc='Days', unit='day'):
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
        all_trades.extend(engine.trades)

        all_results.append({
            'day': day_name, 'trades': day_n, 'pnl': day_pnl,
            'wr': sum(1 for t in engine.trades if t['pnl'] > 0) / max(day_n, 1) * 100,
        })

        if args.verbose and day_n > 0:
            tqdm.write(f'  {day_name}: {engine.summary()}  cumul=${cumul:.0f}')

    # Summary
    n_days = len(all_results)
    total_trades = len(all_trades)
    total_pnl = sum(t['pnl'] for t in all_trades)
    active = sum(1 for r in all_results if r['trades'] > 0)
    winning = sum(1 for r in all_results if r['pnl'] > 0)
    wins = sum(1 for t in all_trades if t['pnl'] > 0)

    print(f'\n{"="*60}')
    print(f'BLENDED RESULTS — {args.target.upper()}')
    print(f'{"="*60}')
    print(f'  Days: {n_days} (active: {active})')
    print(f'  Winning days: {winning}/{active} ({winning/max(active,1)*100:.0f}%)')
    print(f'  Trades: {total_trades} | WR: {wins}/{total_trades} ({wins/max(total_trades,1)*100:.0f}%)')
    print(f'  Total PnL: ${total_pnl:,.0f}')
    print(f'  $/day: ${total_pnl/max(active,1):.0f}')
    print(f'  $/trade: ${total_pnl/max(total_trades,1):.1f}')

    # By tier
    print(f'\n  BY ENTRY TIER:')
    print(f'  {"Tier":<15} {"N":>6} {"WR":>6} {"$/trade":>8} {"Total$":>9}')
    print(f'  {"-"*48}')
    for tier in ['CASCADE', 'KILL_SHOT', 'BASE_NMP']:
        sub = [t for t in all_trades if t['entry_tier'] == tier]
        if not sub:
            continue
        w = sum(1 for t in sub if t['pnl'] > 0)
        tot = sum(t['pnl'] for t in sub)
        avg = tot / len(sub)
        wr = w / len(sub) * 100
        print(f'  {tier:<15} {len(sub):>6} {wr:>5.0f}% ${avg:>7.1f} ${tot:>8,.0f}')

    # By exit reason
    print(f'\n  BY EXIT REASON:')
    print(f'  {"Reason":<30} {"N":>6} {"WR":>6} {"$/trade":>8}')
    print(f'  {"-"*53}')
    exits = Counter(t['exit_reason'] for t in all_trades)
    for reason, count in exits.most_common():
        sub = [t for t in all_trades if t['exit_reason'] == reason]
        w = sum(1 for t in sub if t['pnl'] > 0)
        avg = sum(t['pnl'] for t in sub) / len(sub)
        wr = w / len(sub) * 100
        print(f'  {reason:<30} {count:>6} {wr:>5.0f}% ${avg:>7.1f}')

    # Save CSV with tier + exit reason
    os.makedirs('nn_v2/output/reports', exist_ok=True)
    flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
            for t in all_trades]
    csv_path = f'nn_v2/output/reports/blended_{args.target}_trades.csv'
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    print(f'\nTrade CSV: {csv_path}')

    daily_path = f'nn_v2/output/reports/blended_{args.target}_daily.csv'
    pd.DataFrame(all_results).to_csv(daily_path, index=False)
    print(f'Daily CSV: {daily_path}')


if __name__ == '__main__':
    main()
