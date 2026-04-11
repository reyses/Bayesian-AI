"""
Isolated pipeline for non-NMP entries.

Runs REGIME_FLIP, EXHAUSTION_BAR, ABSORPTION through their own
CNN training without contaminating the main NMP pipeline.

Usage:
    python training_iso/run_iso.py                    # full pipeline
    python training_iso/run_iso.py --from 3           # from phase 3
    python training_iso/run_iso.py --phase1-only      # just collect trades
"""
import os
import sys
import glob
import time as _time
import subprocess
import pickle
import shutil
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_DIR_SEQ = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'training_iso/output'


def _resolve_days(target, source_dir):
    all_files = sorted(glob.glob(os.path.join(source_dir, '*.parquet')))
    if target == 'is':
        return [f for f in all_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        return [f for f in all_files if '2026_' in os.path.basename(f)]
    return all_files


def _print_summary(results):
    if not results:
        print('No results.')
        return
    n_days = len(results)
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    winning_days = sum(1 for r in results if r['pnl'] > 0)
    print(f'\n{"="*60}')
    print(f'RESULTS: {n_days} days | {total_trades} trades | ${total_pnl:.2f}')
    print(f'  $/day: ${total_pnl / max(n_days, 1):.2f}')
    print(f'  Winning days: {winning_days}/{n_days}')
    print(f'{"="*60}')


def run_iso_forward(target='is'):
    """Run isolated engine on feature files."""
    from training_iso.nightmare_iso import IsoEngine
    from training.sfe_ticker import FeatureTicker
    from tqdm import tqdm
    from collections import Counter

    feat_files = _resolve_days(target, FEATURES_DIR_SEQ)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return [], []

    print(f'ISO FORWARD — {len(feat_files)} day(s)')
    engine = IsoEngine()
    all_results = []
    all_trades = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        engine.trades = []
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            t['day'] = day_name
        all_trades.extend(engine.get_full_trades())

        day_pnl = engine.daily_pnl
        day_trades = len(engine.trades)
        all_results.append({
            'day': day_name,
            'trades': day_trades,
            'pnl': day_pnl,
            'wr': sum(1 for t in engine.trades if t['pnl'] > 0) / max(day_trades, 1) * 100,
        })

    _print_summary(all_results)

    # Tier breakdown
    if all_trades:
        tiers = Counter(t.get('entry_tier', '?') for t in all_trades)
        for tier, count in tiers.most_common():
            sub = [t for t in all_trades if t.get('entry_tier') == tier]
            wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            total = sum(t['pnl'] for t in sub)
            print(f'  {tier}: {count} trades, WR={wr:.0f}%, ${total:,.0f}')

    return all_results, all_trades


def run_regret():
    """Run regret on ISO trades."""
    from training.regret import compute_all_regrets

    trade_path = os.path.join(OUTPUT_DIR, 'trades', 'iso_is.pkl')
    with open(trade_path, 'rb') as f:
        trades = pickle.load(f)
    print(f'Regret on {len(trades)} ISO trades...')

    regret_df = compute_all_regrets(trades)
    actual = regret_df['actual_pnl'].sum()
    optimal = regret_df['best_pnl'].sum()
    print(f'  Actual:  ${actual:,.0f}')
    print(f'  Optimal: ${optimal:,.0f}')
    print(f'  Capture: {actual / max(optimal, 1) * 100:.1f}%')

    # Per-tier
    if 'entry_tier' in regret_df.columns:
        for tier in regret_df['entry_tier'].unique():
            sub = regret_df[regret_df['entry_tier'] == tier]
            print(f'    {tier}: {len(sub)} trades, '
                  f'actual=${sub["actual_pnl"].sum():,.0f}, '
                  f'optimal=${sub["best_pnl"].sum():,.0f}')

    # Best action
    n_counter = regret_df['best_action'].str.contains('counter').sum()
    print(f'  Counter: {n_counter} ({n_counter / len(regret_df) * 100:.0f}%)')

    os.makedirs(os.path.join(OUTPUT_DIR, 'tree'), exist_ok=True)
    regret_df.to_csv(os.path.join(OUTPUT_DIR, 'tree', 'regret_analysis.csv'), index=False)
    # Also save as nmp_is.pkl for CNN flip compatibility
    with open(os.path.join(OUTPUT_DIR, 'trades', 'nmp_is.pkl'), 'wb') as f:
        pickle.dump(trades, f)
    with open(os.path.join(OUTPUT_DIR, 'trades', 'blended_is.pkl'), 'wb') as f:
        pickle.dump(trades, f)


def main():
    args = sys.argv[1:]
    phase1_only = '--phase1-only' in args

    print(f'{"="*60}')
    print(f'ISOLATED PIPELINE (non-NMP entries)')
    print(f'  REGIME_FLIP + EXHAUSTION_BAR + ABSORPTION')
    print(f'  Separate CNN training, no NMP contamination')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}')

    pipeline_start = _time.perf_counter()

    # Phase 1: Collect trades
    print(f'\n{"="*40}')
    print(f'PHASE 1: ISO forward pass (no CNN)')
    print(f'{"="*40}')
    t0 = _time.perf_counter()
    results, trades = run_iso_forward('is')
    if trades:
        os.makedirs(os.path.join(OUTPUT_DIR, 'trades'), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'trades', 'iso_is.pkl'), 'wb') as f:
            pickle.dump(trades, f)
        flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
                for t in trades]
        pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'trades', 'iso_is.csv'), index=False)
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    if phase1_only:
        print('Phase 1 only. Done.')
        return

    # Phase 2: Regret
    print(f'\n{"="*40}')
    print(f'PHASE 2: Regret on ISO trades')
    print(f'{"="*40}')
    t0 = _time.perf_counter()
    run_regret()
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Phase 3: Train CNN flip (uses iso output paths)
    print(f'\n{"="*40}')
    print(f'PHASE 3: Train CNN flip (SAME/COUNTER) on ISO trades')
    print(f'{"="*40}')
    t0 = _time.perf_counter()
    # Set env vars so CNN flip reads from iso paths
    env = os.environ.copy()
    env['CNN_TRADES_PATH'] = os.path.join(OUTPUT_DIR, 'trades', 'blended_is.pkl')
    env['CNN_REGRET_PATH'] = os.path.join(OUTPUT_DIR, 'tree', 'regret_analysis.csv')
    env['CNN_OUTPUT_DIR'] = os.path.join(OUTPUT_DIR, 'tree')
    result = subprocess.run(
        [sys.executable, 'training/cnn_flip.py', '--no-path'],
        timeout=3600, capture_output=False, env=env)
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Phase 4: Report
    print(f'\n{"="*40}')
    print(f'PHASE 4: Summary')
    print(f'{"="*40}')
    elapsed = _time.perf_counter() - pipeline_start
    print(f'ISO PIPELINE COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# Need numpy for flat export
import numpy as np

if __name__ == '__main__':
    main()
