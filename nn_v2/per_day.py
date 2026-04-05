"""
Per-Day Iteration — run every IS day through the AI, build dense brain.

The tree is FROZEN. No retraining. No label rewriting.
The book carries raw strategy + regret profiles per leaf.
The brain accumulates evidence day by day — later days benefit from earlier evidence.

For EACH IS day:
  1. Run AI engine (continuous positioning with chain logic)
  2. Record every trade to brain (chain history, path adherence, entry match)
  3. Brain carries forward to next day
  4. After all days: save brain + report

The brain becomes the selector: given a leaf + its regret profile from the book,
which action actually works based on accumulated evidence?

Usage:
    python nn_v2/per_day.py                    # all IS days
    python nn_v2/per_day.py --target oos       # OOS days (validation only)
    python nn_v2/per_day.py --verbose          # print per-trade detail
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.ai import AIEngine
from nn_v2.memory import BayesianMemory
from core.features_79d import FEATURE_NAMES_79D

FEATURES_DIR = 'DATA/FEATURES_79D_1m_seq'
FEATURES_DIR_BULK = 'DATA/FEATURES_79D_1m'
PRICE_DIR = 'DATA/ATLAS/1m'
TREE_DIR = 'DATA/NMP_TREE'

TREE_PATH = os.path.join(TREE_DIR, 'strategy_tree.pkl')
BOOK_PATH = os.path.join(TREE_DIR, 'strategy_book.pkl')


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Per-day iteration — build dense brain')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'],
                   help='Which days to run (default: is)')
    p.add_argument('--verbose', action='store_true', help='Print per-trade detail')
    return p.parse_args()


def get_day_files(target='is'):
    """Get feature + price file pairs for target days."""
    import glob

    # Prefer sequential (honest), fall back to bulk
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    source = 'sequential'
    if not feat_files:
        feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR_BULK, '*.parquet')))
        source = 'bulk'

    if target == 'is':
        feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        feat_files = [f for f in feat_files if '2026_' in os.path.basename(f)]
    # 'all' keeps everything

    pairs = []
    for ff in feat_files:
        day_name = os.path.basename(ff).replace('.parquet', '')
        pf = os.path.join(PRICE_DIR, f'{day_name}.parquet')
        pairs.append((ff, pf if os.path.exists(pf) else None, day_name))

    return pairs, source


def run_one_day(ai, feat_file, price_file):
    """Run AI engine on one day. Returns trade list and daily PnL."""
    ai.reset()
    ft = FeatureTicker(feat_file, price_file=price_file)

    for state in ft:
        ai.on_state(state)

    ai.force_close()
    return ai.trades, ai.daily_pnl


def compute_path_adherence(trade, book_entry):
    """Compute how well the trade followed the expected path from the book.

    Returns a divergence score: 0 = perfectly followed, higher = more divergent.
    """
    if not book_entry:
        return 0.0

    # Use same_path or counter_path based on tree strategy
    tree_strat = book_entry.get('tree_strategy', '')
    if 'counter' in tree_strat:
        expected = book_entry.get('counter_path', [])
    else:
        expected = book_entry.get('same_path', [])

    if not expected:
        return 0.0

    held = trade.get('held', 0)
    actual_pnl = trade.get('pnl', 0)

    # Compare actual PnL at exit bar to expected PnL at that bar
    if held < len(expected):
        expected_pnl = expected[held]
    elif expected:
        expected_pnl = expected[-1]
    else:
        return 0.0

    # Divergence = actual vs expected (negative = underperforming)
    divergence = actual_pnl - expected_pnl
    return divergence


def main():
    args = parse_args()
    file_pairs, source = get_day_files(args.target)

    if not file_pairs:
        print(f'No feature files found for target={args.target}')
        return

    print(f'PER-DAY BRAIN BUILDER — {args.target.upper()}')
    print(f'  Days: {len(file_pairs)} | Source: {source} | Tree: FROZEN')
    print(f'  Tree: {TREE_PATH}')
    print(f'  Book: {BOOK_PATH}')

    # Load brain (fresh for IS, or load existing for continuation)
    memory = BayesianMemory()

    # Commit tree branches as priors
    with open(TREE_PATH, 'rb') as f:
        tree_data = pickle.load(f)
    memory.commit_branches(tree_data['branches'])

    # Load book for path adherence calculation
    book = {}
    if os.path.exists(BOOK_PATH):
        with open(BOOK_PATH, 'rb') as f:
            book = pickle.load(f)

    # Create AI engine (uses frozen tree + book)
    ai = AIEngine(TREE_PATH, BOOK_PATH)

    all_results = []
    cumul_pnl = 0.0

    pbar = tqdm(file_pairs, desc='Days', unit='day',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    for feat_file, price_file, day_name in pbar:
        trades, day_pnl = run_one_day(ai, feat_file, price_file)

        # Record every trade to brain
        for t in trades:
            leaf_id = t.get('branch', -1)
            book_entry = book.get(leaf_id, {})
            adherence = compute_path_adherence(t, book_entry)

            memory.record_trade(
                leaf_id=leaf_id,
                pnl=t['pnl'],
                peak=t.get('peak', 0),
                held=t.get('held', 0),
                chain_length=t.get('chain_length', 0),
                exit_reason=t.get('exit', ''),
                entry_match=0.0,  # gate provides this at runtime
                path_adherence=adherence,
            )

        cumul_pnl += day_pnl
        n_trades = len(trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / max(n_trades, 1) * 100
        chained = sum(1 for t in trades if t.get('chain_length', 0) > 0)

        all_results.append({
            'day': day_name,
            'trades': n_trades,
            'pnl': day_pnl,
            'wr': wr,
            'cumul': cumul_pnl,
            'chained': chained,
        })

        flag = '<<<' if day_pnl > 50 else '!!!' if day_pnl < -50 else ''
        pbar.set_postfix_str(f'${cumul_pnl:+,.0f}')

        if args.verbose:
            tqdm.write(f'  {day_name}: {n_trades:>3} trades  WR={wr:>4.0f}%  '
                       f'${day_pnl:>8.0f}  cumul=${cumul_pnl:>8.0f}  '
                       f'chain={chained} {flag}')

    # Summary
    n_days = len(all_results)
    total_pnl = sum(r['pnl'] for r in all_results)
    total_trades = sum(r['trades'] for r in all_results)
    winning = sum(1 for r in all_results if r['pnl'] > 0)
    total_chained = sum(r['chained'] for r in all_results)

    print(f'\n{"="*60}')
    print(f'PER-DAY BRAIN BUILDER — {args.target.upper()} COMPLETE')
    print(f'{"="*60}')
    print(f'  Days: {n_days} | Winning: {winning}/{n_days} ({winning/max(n_days,1)*100:.0f}%)')
    print(f'  Trades: {total_trades} | Chained: {total_chained}')
    print(f'  PnL: ${total_pnl:,.0f} | $/day: ${total_pnl/max(n_days,1):.0f}')
    print(f'\n{memory.summary()}')

    # Save brain
    brain_path = os.path.join(TREE_DIR, f'memory_{args.target}.pkl')
    memory.save(brain_path)

    # Save report
    report_path = os.path.join(TREE_DIR, f'perday_{args.target}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Per-Day Brain Builder — {args.target.upper()}\n{"="*60}\n')
        f.write(f'Days: {n_days} | Winning: {winning}/{n_days} ({winning/max(n_days,1)*100:.0f}%)\n')
        f.write(f'Trades: {total_trades} | Chained: {total_chained}\n')
        f.write(f'PnL: ${total_pnl:,.0f} | $/day: ${total_pnl/max(n_days,1):.0f}\n\n')
        f.write(f'Daily breakdown:\n')
        cumul = 0
        for r in all_results:
            cumul += r['pnl']
            flag = '<<<' if r['pnl'] > 50 else '!!!' if r['pnl'] < -50 else ''
            f.write(f'  {r["day"]}  {r["trades"]:>3} trades  {r["wr"]:>4.0f}%  '
                    f'${r["pnl"]:>8.0f}  cumul=${cumul:>8.0f}  chain={r["chained"]} {flag}\n')
        f.write(f'\n{memory.summary()}\n')
    print(f'Report: {report_path}')

    # Save CSV
    csv_path = os.path.join(TREE_DIR, f'perday_{args.target}_daily.csv')
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
