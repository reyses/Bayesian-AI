"""
Baseline Runner — physics-only forward pass with clean report.

Runs BlendedEngine (no CNN) on IS + OOS + OOS-NT8.
Prints a single pasteable report at the end.

Usage:
    python training/run_baseline.py              # full run
    python training/run_baseline.py --oos-only   # skip IS
    python training/run_baseline.py --atlas DATA/ATLAS_NT8  # NT8 data only
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine
from collections import Counter

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
FEATURES_NT8 = 'DATA/FEATURES_NT8_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
ATLAS_NT8_1M = 'DATA/ATLAS_NT8/1m'


def run_forward(feat_files, price_dir, label=''):
    """Run physics-only forward pass. Returns (results, trades)."""
    engine = BlendedEngine(use_cnn=False)
    all_results = []
    all_trades = []

    for fpath in tqdm(feat_files, desc=f'  {label}', unit='day', leave=False):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(price_dir, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            t['day'] = day_name
        all_trades.extend(engine.get_full_trades())

        all_results.append({
            'day': day_name,
            'trades': len(engine.trades),
            'pnl': engine.daily_pnl,
        })

    return all_results, all_trades


def print_report(datasets, tier_trades=None):
    """Print clean pasteable report."""
    print()
    print('=' * 65)
    print('BASELINE REPORT (physics only, no CNN)')
    print('=' * 65)
    print()
    print(f'{"Dataset":<16} {"$/day":>8} {"Trades":>8} {"WinDays":>10} {"Days":>6}')
    print(f'{"-"*52}')

    for label, results in datasets.items():
        if not results:
            continue
        n = len(results)
        total_pnl = sum(r['pnl'] for r in results)
        per_day = total_pnl / max(n, 1)
        total_trades = sum(r['trades'] for r in results)
        win_days = sum(1 for r in results if r['pnl'] > 0)
        print(f'{label:<16} {per_day:>+8,.0f} {total_trades:>8,} '
              f'{win_days:>4}/{n:<4}  {n:>6}')

    if tier_trades:
        print()
        print(f'{"Tier":<20} {"Trades":>7} {"WR":>5} {"$/tr":>8} {"$/day":>8}')
        print(f'{"-"*52}')

        tiers = Counter(t.get('entry_tier', '?') for t in tier_trades)
        n_days = len(set(t.get('day', '') for t in tier_trades)) or 1

        for tier, count in sorted(tiers.items(), key=lambda x: -sum(
                t['pnl'] for t in tier_trades if t.get('entry_tier') == x[0])):
            sub = [t for t in tier_trades if t.get('entry_tier') == tier]
            wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            avg = np.mean([t['pnl'] for t in sub])
            per_day = sum(t['pnl'] for t in sub) / n_days
            print(f'{str(tier):<20} {count:>7} {wr:>4.0f}% {avg:>8.1f} {per_day:>+8.0f}')

        # Chain lightning stats
        chains = [t for t in tier_trades if any(
            p.get('chain') for p in t.get('path', []) if isinstance(p, dict))]
        if chains:
            print(f'\n  Chained lightning: {len(chains)} trades upgraded mid-position')

    print()
    print('=' * 65)


def main():
    parser = argparse.ArgumentParser(description='Baseline Runner')
    parser.add_argument('--oos-only', action='store_true')
    parser.add_argument('--atlas', type=str, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    datasets = {}

    if args.atlas:
        # Single dataset mode
        atlas_name = os.path.basename(args.atlas.rstrip('/'))
        feat_name = atlas_name.replace('ATLAS', 'FEATURES')
        feat_dir = os.path.join('DATA', f'{feat_name}_5s')
        price_dir = os.path.join(args.atlas, '1m')

        feat_files = sorted(glob.glob(os.path.join(feat_dir, '*.parquet')))
        if not feat_files:
            print(f'No features in {feat_dir}/')
            return

        results, trades = run_forward(feat_files, price_dir, atlas_name)
        datasets[atlas_name] = results
        print_report(datasets, trades)

    else:
        # Full run: IS + OOS + OOS-NT8
        all_trades = []

        if not args.oos_only:
            is_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                              if '2025_' in os.path.basename(f))
            if is_files:
                results, trades = run_forward(is_files, ATLAS_1M, 'IS')
                datasets['IS'] = results
                all_trades.extend(trades)

        oos_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                           if '2026_' in os.path.basename(f))
        if oos_files:
            results, trades = run_forward(oos_files, ATLAS_1M, 'OOS')
            datasets['OOS'] = results

        nt8_files = sorted(glob.glob(os.path.join(FEATURES_NT8, '*.parquet')))
        if nt8_files:
            results, trades = run_forward(nt8_files, ATLAS_NT8_1M, 'OOS-NT8')
            datasets['OOS-NT8'] = results

        print_report(datasets, all_trades if all_trades else None)

    elapsed = time.perf_counter() - t0
    print(f'  Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
