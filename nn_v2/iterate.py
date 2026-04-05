"""
Iteration Loop — runs strategies on IS, finds gaps, refines, repeats.

Loop:
  1. Run book strategies on IS data → daily PnL
  2. Find losing days
  3. For losing days: analyze which conditions lacked profitable strategies
  4. Feed new trade seeds from losing days back through:
     NMP → regret → tree retrain → book rebuild
  5. Repeat until all days positive or max iterations

Usage:
    python nn_v2/iterate.py                    # iterate until done
    python nn_v2/iterate.py --max-iter 10      # max 10 iterations
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.nightmare import NightmareEngine
from nn_v2.gate import Gate
from nn_v2.regret import compute_all_regrets, summarize_regret_by_branch
from nn_v2.memory import BayesianMemory
from core.features_79d import FEATURE_NAMES_79D

FEATURES_DIR = 'DATA/FEATURES_79D_1m'
PRICE_DIR = 'DATA/ATLAS/1m'
TRADE_LOG = 'DATA/NMP_TRADES/nmp_is.pkl'
TREE_DIR = 'DATA/NMP_TREE'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Iterate strategies until all days positive')
    p.add_argument('--max-iter', type=int, default=20, help='Max iterations')
    return p.parse_args()


def get_is_files():
    """Get IS feature + price file pairs."""
    import glob
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]

    pairs = []
    for ff in feat_files:
        day_name = os.path.basename(ff).replace('.parquet', '')
        pf = os.path.join(PRICE_DIR, f'{day_name}.parquet')
        pairs.append((ff, pf if os.path.exists(pf) else None, day_name))
    return pairs


def run_strategies(gate, file_pairs):
    """Run strategies on IS data. Returns daily results + all trades."""
    nmp = NightmareEngine()
    daily_results = []
    all_trades = []

    for feat_file, price_file, day_name in tqdm(file_pairs, desc='  Running', unit='day'):
        nmp.reset()
        ft = FeatureTicker(feat_file, price_file=price_file)

        for state in ft:
            decision = gate.evaluate(state)

            if nmp.in_pos:
                # In trade — let NMP manage exit
                nmp.on_state(state)
            elif decision['allowed']:
                # Tree says trade
                branch = decision['branch']
                # If branch says counter → flip NMP direction
                if branch and 'counter' in branch.get('strategy', ''):
                    # Flip the 79D z_se sign to make NMP enter opposite direction
                    flipped = state.copy()
                    feat = state['features_79d'].copy()
                    # Flip 1m z_se (index 10 = 1m offset + 0)
                    feat[10] = -feat[10]
                    flipped['features_79d'] = feat
                    nmp.on_state(flipped)
                else:
                    nmp.on_state(state)

        nmp.force_close()

        for t in nmp.trades:
            t['day'] = day_name
            # Tag with branch
            entry_feat = np.array(t['entry_79d']).reshape(1, -1)
            entry_feat = np.nan_to_num(entry_feat)
            t['leaf_id'] = int(gate.tree.apply(entry_feat)[0])
        all_trades.extend(nmp.get_full_trades())

        daily_results.append({
            'day': day_name,
            'trades': len(nmp.trades),
            'pnl': nmp.daily_pnl,
        })

    return daily_results, all_trades


def analyze_losing_days(daily_results, all_trades, file_pairs):
    """Find losing days and analyze what went wrong."""
    losing_days = [d for d in daily_results if d['pnl'] < 0]
    winning_days = [d for d in daily_results if d['pnl'] >= 0]

    print(f'\n  Days: {len(winning_days)} winning, {len(losing_days)} losing')
    print(f'  Total PnL: ${sum(d["pnl"] for d in daily_results):,.0f}')

    if not losing_days:
        return None

    # Analyze losing days: what trades happened? What was the 79D?
    losing_day_names = set(d['day'] for d in losing_days)
    losing_trades = [t for t in all_trades if t.get('day') in losing_day_names]

    print(f'\n  Losing days: {len(losing_days)}')
    print(f'  Trades on losing days: {len(losing_trades)}')
    if losing_trades:
        losing_pnl = sum(t['pnl'] for t in losing_trades)
        losing_wr = sum(1 for t in losing_trades if t['pnl'] > 0) / len(losing_trades)
        print(f'  Losing day PnL: ${losing_pnl:,.0f} | WR: {losing_wr:.0%}')

    # Top 5 worst days
    worst = sorted(losing_days, key=lambda d: d['pnl'])[:5]
    print(f'\n  Worst 5 days:')
    for d in worst:
        print(f'    {d["day"]}: {d["trades"]} trades, ${d["pnl"]:.0f}')

    return {
        'losing_days': losing_days,
        'losing_trades': losing_trades,
        'losing_day_names': losing_day_names,
    }


def retrain_from_losses(all_trades, losing_info):
    """Retrain tree with new regret data including losing day analysis."""
    from nn_v2.regret import compute_all_regrets
    from nn_v2.tree import load_data, train_tree, analyze_branches

    print(f'\n  Recomputing regret for all {len(all_trades)} trades...')
    regret_df = compute_all_regrets(all_trades)
    regret_df.to_csv(os.path.join(TREE_DIR, 'regret_analysis.csv'), index=False)

    print(f'  Retraining tree...')
    # Load feature data + new regret labels
    # Save updated trades for tree.py to read
    with open(TRADE_LOG, 'wb') as f:
        pickle.dump(all_trades, f)

    # Run tree training
    from nn_v2.tree import load_data, train_tree, analyze_branches, print_report, STRATEGIES
    df = load_data()

    # Increase depth for more branches to cover gaps
    from sklearn.tree import DecisionTreeClassifier
    tree, X, y, cv_results = train_tree(df, FEATURE_NAMES_79D, max_depth=10, min_leaf=15)

    branches = analyze_branches(tree, X, y, df)

    # Save
    save_path = os.path.join(TREE_DIR, 'strategy_tree.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'branches': branches,
            'feature_names': FEATURE_NAMES_79D,
            'strategies': STRATEGIES,
            'cv_results': cv_results,
        }, f)

    report_path = os.path.join(TREE_DIR, 'strategy_tree_report.txt')
    print_report(tree, branches, cv_results, FEATURE_NAMES_79D, save_path=report_path)

    return tree, branches


def main():
    args = parse_args()
    file_pairs = get_is_files()

    print(f'ITERATION LOOP — target: all {len(file_pairs)} IS days positive')
    print(f'  Max iterations: {args.max_iter}')

    for iteration in range(1, args.max_iter + 1):
        print(f'\n{"="*60}')
        print(f'ITERATION {iteration}')
        print(f'{"="*60}')

        # Load current gate
        tree_path = os.path.join(TREE_DIR, 'strategy_tree.pkl')
        if not os.path.exists(tree_path):
            print(f'  No tree found. Run tree.py first.')
            return

        gate = Gate(tree_path)

        # Run strategies on IS data
        daily_results, all_trades = run_strategies(gate, file_pairs)

        # Analyze
        losing_info = analyze_losing_days(daily_results, all_trades, file_pairs)

        if losing_info is None:
            print(f'\n  ALL DAYS POSITIVE! Stopping at iteration {iteration}.')
            break

        n_losing = len(losing_info['losing_days'])
        n_total = len(daily_results)
        pct_winning = (n_total - n_losing) / n_total * 100

        print(f'\n  Progress: {pct_winning:.0f}% winning days ({n_total - n_losing}/{n_total})')

        # Retrain
        retrain_from_losses(all_trades, losing_info)

        # Rebuild book
        print(f'\n  Rebuilding strategy book...')
        from nn_v2.book import load_all_data, build_book, print_book
        trades, regret_df, tree_data = load_all_data()
        strategies = build_book(trades, regret_df, tree_data)
        book_path = os.path.join(TREE_DIR, 'strategy_book.txt')
        print_book(strategies, save_path=book_path)

    # Final summary
    print(f'\n{"="*60}')
    print(f'FINAL RESULTS after {iteration} iterations')
    print(f'{"="*60}')
    total_pnl = sum(d['pnl'] for d in daily_results)
    winning = sum(1 for d in daily_results if d['pnl'] >= 0)
    print(f'  {winning}/{len(daily_results)} winning days')
    print(f'  Total PnL: ${total_pnl:,.0f}')
    print(f'  $/day: ${total_pnl / len(daily_results):,.0f}')


if __name__ == '__main__':
    main()
