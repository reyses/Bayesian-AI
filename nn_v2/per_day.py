"""
Per-Day Epoch Learning — book gets schooled, day by day.

LEARN phase of the Bayesian pipeline:
  For each IS day sequentially:
    1. AI trades with current book → regret on trades
    2. Bayesian update book (Dirichlet profiles, exit bars, paths)
    3. AI retries same day with updated book
    4. If improved → keep, epoch++. If not → revert, stop.
    5. Spot-check a previous day (non-interference)
    6. Freeze book version → carry to next day

The book is the student. Each day is a lesson.

Usage:
    python nn_v2/per_day.py                    # LEARN phase on IS
    python nn_v2/per_day.py --verbose          # print per-epoch detail
    python nn_v2/per_day.py --max-epochs 3     # limit epochs per day
"""
import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.ai import AIEngine
from nn_v2.regret import compute_regret
from nn_v2.book import (VersionedBook, BOOK_DIR, MAX_EPOCHS_PER_DAY,
                         MIN_PNL_IMPROVEMENT, SPOT_CHECK_DEGRADATION)

FEATURES_DIR = 'DATA/FEATURES_79D_5s_v2'
FEATURES_DIR_BULK = 'DATA/FEATURES_79D_1m'
PRICE_DIR = 'DATA/ATLAS/1m'
TREE_DIR = 'nn_v2/output/tree'
TREE_PATH = os.path.join(TREE_DIR, 'strategy_tree.pkl')


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Per-day epoch learning')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--max-epochs', type=int, default=MAX_EPOCHS_PER_DAY)
    return p.parse_args()


def get_day_files(target='is'):
    """Get feature + price file pairs."""
    import glob
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    source = 'sequential'
    if not feat_files:
        feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR_BULK, '*.parquet')))
        source = 'bulk'

    if target == 'is':
        feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        feat_files = [f for f in feat_files if '2026_' in os.path.basename(f)]

    pairs = []
    for ff in feat_files:
        day_name = os.path.basename(ff).replace('.parquet', '')
        pf = os.path.join(PRICE_DIR, f'{day_name}.parquet')
        pairs.append((ff, pf if os.path.exists(pf) else None, day_name))
    return pairs, source


def run_epoch(ai, feat_file, price_file, book, day_name):
    """Run AI on one day with current book. Returns (trades, pnl, regret_df).

    Injects book into AI gate, runs forward pass, computes regret.
    """
    # Inject current book
    ai.set_book(book.export_for_gate())
    ai.reset()

    ft = FeatureTicker(feat_file, price_file=price_file)
    for state in ft:
        ai.on_state(state)
    ai.force_close()

    trades = ai.get_full_trades()
    day_pnl = ai.daily_pnl

    # Tag trades with day
    for t in trades:
        t['day'] = day_name

    # Compute regret for this day's trades
    regret_rows = []
    if trades and price_file and os.path.exists(price_file):
        price_df = pd.read_parquet(price_file).sort_values('timestamp')
        closes = price_df['close'].values
        timestamps = price_df['timestamp'].values

        for t in trades:
            entry_ts = t.get('timestamp', 0)
            entry_idx = int(np.searchsorted(timestamps, entry_ts, side='left'))
            entry_idx = min(entry_idx, len(closes) - 1)
            r = compute_regret(t, closes, entry_idx)
            r['leaf_id'] = t.get('leaf_id', t.get('branch', -1))
            r['day'] = day_name
            regret_rows.append(r)

    # Build regret DataFrame (without curve arrays)
    if regret_rows:
        flat = [{k: v for k, v in r.items()
                 if k not in ('same_curve', 'counter_curve', 'early_entries')}
                for r in regret_rows]
        regret_df = pd.DataFrame(flat)
    else:
        regret_df = pd.DataFrame()

    return trades, day_pnl, regret_df


def learn_one_day(ai, feat_file, price_file, book, day_name,
                  max_epochs, verbose=False):
    """Multi-epoch learning on a single day.

    Epoch 1: trade → regret → Bayesian update
    Epoch 2+: retry same day → if improved, keep; if not, revert + stop.
    """
    # Snapshot before any updates
    pre_snapshots = book.snapshot_all()

    # Epoch 1: baseline run + update
    trades, base_pnl, regret_df = run_epoch(ai, feat_file, price_file, book, day_name)

    if len(trades) < 2 or regret_df.empty:
        return {
            'day': day_name, 'epochs': 0, 'trades': len(trades),
            'pnl_before': base_pnl, 'pnl_after': base_pnl, 'improved': False,
        }

    # Bayesian update from epoch 1
    traded_leaves = regret_df['leaf_id'].unique() if 'leaf_id' in regret_df.columns else []
    changes = []
    for lid in traded_leaves:
        lid = int(lid)
        leaf_regrets = regret_df[regret_df['leaf_id'] == lid]
        leaf_trades = [t for t in trades if t.get('leaf_id', t.get('branch', -1)) == lid]
        ch = book.bayesian_update(lid, leaf_regrets, leaf_trades)
        if ch.get('changed'):
            changes.append(ch)

    if not changes:
        return {
            'day': day_name, 'epochs': 1, 'trades': len(trades),
            'pnl_before': base_pnl, 'pnl_after': base_pnl, 'improved': False,
        }

    best_pnl = base_pnl
    best_snapshots = book.snapshot_all()
    epochs_run = 1

    if verbose:
        tqdm.write(f'    {day_name} e1: ${base_pnl:.0f} | {len(changes)} leaves updated')

    # Epoch 2+: retry with updated book
    for epoch in range(2, max_epochs + 1):
        _, retry_pnl, retry_regret = run_epoch(ai, feat_file, price_file, book, day_name)
        epochs_run = epoch

        if verbose:
            delta = retry_pnl - best_pnl
            tqdm.write(f'    {day_name} e{epoch}: ${retry_pnl:.0f} ({"+" if delta>=0 else ""}{delta:.0f})')

        if retry_pnl >= best_pnl + MIN_PNL_IMPROVEMENT:
            # Improved — keep, update again, continue
            best_pnl = retry_pnl
            best_snapshots = book.snapshot_all()

            # Another Bayesian update from this epoch's data
            if not retry_regret.empty and 'leaf_id' in retry_regret.columns:
                for lid in retry_regret['leaf_id'].unique():
                    lid = int(lid)
                    leaf_regrets = retry_regret[retry_regret['leaf_id'] == lid]
                    leaf_trades = [t for t in trades if t.get('leaf_id', t.get('branch', -1)) == lid]
                    book.bayesian_update(lid, leaf_regrets, leaf_trades)
        else:
            # No improvement — revert to best, stop
            book.revert_all(best_snapshots)
            break

    return {
        'day': day_name, 'epochs': epochs_run, 'trades': len(trades),
        'pnl_before': base_pnl, 'pnl_after': best_pnl,
        'improved': best_pnl > base_pnl + MIN_PNL_IMPROVEMENT,
    }


def learn_phase(book, verbose=False, max_epochs=MAX_EPOCHS_PER_DAY):
    """Full LEARN phase: iterate IS days, epoch-learn each, freeze versions."""
    file_pairs, source = get_day_files('is')

    if not file_pairs:
        print('No IS feature files found.')
        return book

    print(f'LEARN PHASE — {len(file_pairs)} IS days')
    print(f'  Source: {source} | Max epochs/day: {max_epochs}')
    print(f'  Book: v{book.version} with {len(book.leaves)} leaves')

    ai = AIEngine(TREE_PATH)

    # Track spot-check baselines
    spot_baselines = {}
    results = []

    pbar = tqdm(file_pairs, desc='LEARN', unit='day',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    for i, (feat_file, price_file, day_name) in enumerate(pbar):
        result = learn_one_day(ai, feat_file, price_file, book, day_name,
                               max_epochs, verbose)
        results.append(result)
        spot_baselines[day_name] = result['pnl_after']

        # Spot-check: re-run a random previous day
        if i > 4 and random.random() < 0.2:  # 20% chance, after 5 days
            check_idx = random.randint(0, i - 1)
            check_feat, check_price, check_day = file_pairs[check_idx]
            _, check_pnl, _ = run_epoch(ai, check_feat, check_price, book, check_day)
            original = spot_baselines.get(check_day, 0)
            degradation = check_pnl - original
            if degradation < SPOT_CHECK_DEGRADATION:
                tqdm.write(f'  WARN: {check_day} degraded ${degradation:.0f} '
                           f'(was ${original:.0f}, now ${check_pnl:.0f})')

        # Freeze version
        book.freeze(day_name=day_name)

        # Progress
        improved = sum(1 for r in results if r['improved'])
        total_delta = sum(r['pnl_after'] - r['pnl_before'] for r in results)
        pbar.set_postfix_str(f'v{book.version} | {improved} improved | +${total_delta:.0f}')

    # Summary
    n_improved = sum(1 for r in results if r['improved'])
    total_epochs = sum(r['epochs'] for r in results)
    total_delta = sum(r['pnl_after'] - r['pnl_before'] for r in results)

    print(f'\n{"="*60}')
    print(f'LEARN PHASE COMPLETE')
    print(f'  Days: {len(results)} | Improved: {n_improved}')
    print(f'  Total epochs: {total_epochs} | Avg: {total_epochs/max(len(results),1):.1f}/day')
    print(f'  Total PnL delta: ${total_delta:,.0f}')
    print(f'  Book version: v{book.version}')

    # Save evolution CSV
    os.makedirs(BOOK_DIR, exist_ok=True)
    book.evolution_csv(os.path.join(BOOK_DIR, 'evolution.csv'))

    # Save results
    pd.DataFrame(results).to_csv(os.path.join(BOOK_DIR, 'learn_results.csv'), index=False)

    return book


def main():
    args = parse_args()

    # Load book v0
    book_path = os.path.join(TREE_DIR, 'strategy_book.pkl')
    if not os.path.exists(book_path):
        print(f'No book found at {book_path}. Run book.py first.')
        return

    book = VersionedBook.from_nmp_book(book_path)

    # Freeze v0 as baseline
    book.freeze(day_name='baseline')

    # Run LEARN phase
    book = learn_phase(book, verbose=args.verbose, max_epochs=args.max_epochs)

    # Save final book
    book.save()
    print(f'Final book saved: {BOOK_DIR}/book_latest.pkl')


if __name__ == '__main__':
    main()
