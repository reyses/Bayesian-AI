"""
Trial Run — runs tree branches on IS data, evaluates, iterates.

Loop:
  1. Load tree + memory
  2. Run all IS days: NMP enters, tree classifies, memory records
  3. Evaluate: which branches held? Which failed?
  4. Refine: demote losers, promote winners
  5. Save updated memory
  6. Repeat until stable

Usage:
    python nn_v2/trial.py                      # single iteration
    python nn_v2/trial.py --iterations 5       # run 5 iterations
    python nn_v2/trial.py --target-wr 0.70     # stop when avg WR >= 70%
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_v2.sfe_ticker import FeatureTicker
from nn_v2.nightmare import NightmareEngine
from nn_v2.gate import Gate
from nn_v2.memory import BayesianMemory
from core.features_79d import FEATURE_NAMES_79D

FEATURES_DIR = 'DATA/FEATURES_79D_1m'
PRICE_DIR = 'DATA/ATLAS/1m'
TREE_PATH = 'nn_v2/output/tree/tree.pkl'
MEMORY_PATH = 'nn_v2/output/memory/memory.pkl'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Trial run — iterate tree branches on IS data')
    p.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    p.add_argument('--target-wr', type=float, default=0.80, help='Stop when avg WR >= this')
    p.add_argument('--period', type=str, default='is', choices=['is', 'oos', 'all'],
                   help='Which data period to run on')
    return p.parse_args()


def resolve_files(period: str):
    """Get feature + price file pairs for the period."""
    import glob
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))

    if period == 'is':
        feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    elif period == 'oos':
        feat_files = [f for f in feat_files if '2026_' in os.path.basename(f)]

    pairs = []
    for ff in feat_files:
        day_name = os.path.basename(ff).replace('.parquet', '')
        pf = os.path.join(PRICE_DIR, f'{day_name}.parquet')
        pairs.append((ff, pf if os.path.exists(pf) else None))

    return pairs


def run_iteration(gate: Gate, memory: BayesianMemory, file_pairs: list) -> dict:
    """Run one iteration: NMP + tree gate on all days, record to memory."""
    nmp = NightmareEngine()
    total_allowed = 0
    total_blocked = 0
    total_trades = 0
    all_results = []

    for feat_file, price_file in tqdm(file_pairs, desc='  Days', unit='day'):
        day_name = os.path.basename(feat_file).replace('.parquet', '')
        nmp.reset()

        ft = FeatureTicker(feat_file, price_file=price_file)
        day_allowed = 0
        day_blocked = 0

        for state in ft:
            feat = state['features_79d']

            # Tree classifies this bar
            decision = gate.evaluate(state)
            leaf_id = decision['leaf_id']

            if nmp.in_pos:
                # Already in trade — let NMP manage exit
                nmp.on_state(state)
            elif decision['allowed']:
                # Tree says trade — let NMP enter if conditions met
                nmp.on_state(state)
                day_allowed += 1
            else:
                # Tree says skip — block NMP entry
                day_blocked += 1

        nmp.force_close()

        # Record outcomes to memory
        for t in nmp.trades:
            # Find which leaf this trade's entry 79D maps to
            entry_feat = np.array(t['entry_79d']).reshape(1, -1)
            entry_feat = np.nan_to_num(entry_feat)
            trade_leaf = int(gate.tree.apply(entry_feat)[0])
            memory.record_trade(trade_leaf, t['pnl'], t['peak'], t['held'])

        total_allowed += day_allowed
        total_blocked += day_blocked
        total_trades += len(nmp.trades)
        all_results.append({
            'day': day_name,
            'trades': len(nmp.trades),
            'pnl': nmp.daily_pnl,
        })

    total_pnl = sum(r['pnl'] for r in all_results)
    winning_days = sum(1 for r in all_results if r['pnl'] > 0)
    n_days = len(all_results)

    return {
        'trades': total_trades,
        'pnl': total_pnl,
        'pnl_per_day': total_pnl / max(n_days, 1),
        'winning_days': winning_days,
        'n_days': n_days,
        'allowed': total_allowed,
        'blocked': total_blocked,
        'block_rate': total_blocked / max(total_allowed + total_blocked, 1),
    }


def main():
    args = parse_args()

    print(f'TRIAL RUN — iterate tree branches on {args.period.upper()} data')
    print(f'  Iterations: {args.iterations} | Target WR: {args.target_wr:.0%}')

    # Load tree
    gate = Gate(TREE_PATH)

    # Load or create memory
    if os.path.exists(MEMORY_PATH):
        memory = BayesianMemory.load(MEMORY_PATH)
        print(f'  Loaded existing memory (iteration {memory.iteration})')
    else:
        memory = BayesianMemory()
        # Commit tree branches as priors
        with open(TREE_PATH, 'rb') as f:
            tree_data = pickle.load(f)
        memory.commit_branches(tree_data['branches'])

    file_pairs = resolve_files(args.period)
    print(f'  Data: {len(file_pairs)} days')

    for iteration in range(args.iterations):
        print(f'\n{"="*60}')
        print(f'ITERATION {memory.iteration + 1}')
        print(f'{"="*60}')

        # Run
        result = run_iteration(gate, memory, file_pairs)

        print(f'\n  Trades: {result["trades"]} | PnL: ${result["pnl"]:.0f} | '
              f'$/day: ${result["pnl_per_day"]:.0f}')
        print(f'  Winning days: {result["winning_days"]}/{result["n_days"]}')
        print(f'  Allowed: {result["allowed"]} | Blocked: {result["blocked"]} '
              f'({result["block_rate"]:.0%} block rate)')

        # Evaluate and refine
        eval_result = memory.refine()
        print(f'\n{memory.summary()}')

        # Save
        memory.save(MEMORY_PATH)

        # Check target
        avg_wr = eval_result.get('avg_wr', 0)
        if avg_wr >= args.target_wr:
            print(f'\n  Target WR {args.target_wr:.0%} reached ({avg_wr:.0%}). Stopping.')
            break

    print(f'\nDone. Memory saved to {MEMORY_PATH}')


if __name__ == '__main__':
    main()
