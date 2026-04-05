"""
Per-Day Iteration — iterate EVERY IS day until maximized.

For EACH day (not just losers):
  1. Run all strategies on this day
  2. Compute regret: what's the gap to optimal?
  3. If gap > threshold: fix branches, retrain, re-run this day
  4. Repeat until no improvement or max rounds
  5. Record to brain (evidence per branch per condition)
  6. Move to next day

Every day gets squeezed for maximum PnL. Winners get improved too.
Brain accumulates dense evidence across all days and conditions.

Usage:
    python nn_v2/per_day.py                    # all IS days
    python nn_v2/per_day.py --rounds 5         # max 5 rounds per day
    python nn_v2/per_day.py --min-gap 10       # stop when gap < $10
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
from nn_v2.nightmare import NightmareEngine
from nn_v2.gate import Gate
from nn_v2.memory import BayesianMemory
from nn_v2.regret import compute_regret
from core.features_79d import FEATURE_NAMES_79D

FEATURES_DIR = 'DATA/FEATURES_79D_1m_seq'
FEATURES_DIR_BULK = 'DATA/FEATURES_79D_1m'
PRICE_DIR = 'DATA/ATLAS/1m'
TREE_DIR = 'DATA/NMP_TREE'
TRADE_LOG = 'DATA/NMP_TRADES/nmp_is.pkl'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Per-day iteration — maximize every IS day')
    p.add_argument('--rounds', type=int, default=5, help='Max rounds per day')
    p.add_argument('--min-gap', type=float, default=10.0, help='Stop when regret gap < this')
    return p.parse_args()


def get_is_files():
    import glob
    # Prefer sequential, fall back to bulk
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    if not feat_files:
        feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR_BULK, '*.parquet')))
    feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]

    pairs = []
    for ff in feat_files:
        day_name = os.path.basename(ff).replace('.parquet', '')
        pf = os.path.join(PRICE_DIR, f'{day_name}.parquet')
        pairs.append((ff, pf if os.path.exists(pf) else None, day_name))
    return pairs


def run_day_gated(gate, feat_file, price_file):
    """Run one day with gate. Returns trades + PnL."""
    nmp = NightmareEngine()
    ft = FeatureTicker(feat_file, price_file=price_file)

    for state in ft:
        decision = gate.evaluate(state)
        if nmp.in_pos:
            nmp.on_state(state)
        elif decision['allowed']:
            branch = decision['branch']
            if branch and 'counter' in branch.get('strategy', ''):
                flipped = state.copy()
                feat = state['features_79d'].copy()
                feat[10] = -feat[10]
                flipped['features_79d'] = feat
                nmp.on_state(flipped)
            else:
                nmp.on_state(state)

    nmp.force_close()
    return nmp.trades, nmp.daily_pnl


def compute_day_regret(trades, price_file):
    """Compute regret for one day's trades."""
    if not trades or not price_file or not os.path.exists(price_file):
        return 0.0, []

    price_df = pd.read_parquet(price_file).sort_values('timestamp')
    closes = price_df['close'].values
    timestamps = price_df['timestamp'].values

    total_regret = 0.0
    fixes = []

    # Group by branch
    branch_trades = defaultdict(list)
    for t in trades:
        entry_feat = np.array(t['entry_79d']).reshape(1, -1)
        branch_trades[t.get('leaf_id', -1)].append(t)

    for lid, bt in branch_trades.items():
        branch_pnl = sum(t['pnl'] for t in bt)

        # Regret per trade
        branch_regret = 0.0
        best_actions = defaultdict(float)
        for t in bt:
            entry_ts = t.get('timestamp', 0)
            entry_idx = int(np.searchsorted(timestamps, entry_ts, side='left'))
            entry_idx = min(entry_idx, len(closes) - 1)
            r = compute_regret(t, closes, entry_idx)
            branch_regret += r['regret']

            # Accumulate best action PnL
            best_actions[r['best_action']] += r['best_pnl']

        if best_actions:
            best_fix = max(best_actions, key=best_actions.get)
            fix_pnl = best_actions[best_fix]
        else:
            best_fix = 'same_at_exit'
            fix_pnl = branch_pnl

        total_regret += branch_regret
        if branch_regret > 0:
            fixes.append({
                'leaf_id': lid,
                'n_trades': len(bt),
                'actual_pnl': branch_pnl,
                'best_fix': best_fix,
                'fix_pnl': fix_pnl,
                'regret': branch_regret,
            })

    return total_regret, sorted(fixes, key=lambda f: -f['regret'])


def apply_day_fixes(fixes, day_name, all_trades_df, iteration):
    """Apply fixes for one day by updating tree labels and retraining."""
    from nn_v2.tree import load_data, train_tree, analyze_branches, STRATEGIES

    df = load_data()
    X = np.nan_to_num(df[FEATURE_NAMES_79D].values.astype(np.float32))

    tree_path = os.path.join(TREE_DIR, 'strategy_tree.pkl')
    with open(tree_path, 'rb') as f:
        tree_data = pickle.load(f)
    current_tree = tree_data['tree']
    leaf_ids = current_tree.apply(X)
    df['leaf_id'] = leaf_ids

    overrides = 0
    for fix in fixes[:5]:  # top 5 fixes per day
        mask = (df['day'] == day_name) & (df['leaf_id'] == fix['leaf_id'])
        if mask.sum() > 0 and fix['best_fix'] in STRATEGIES:
            df.loc[mask, 'strategy_idx'] = STRATEGIES.index(fix['best_fix'])
            overrides += mask.sum()

    if overrides == 0:
        return False

    # Retrain
    from sklearn.tree import DecisionTreeClassifier
    y = df['strategy_idx'].values
    tree = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=10, random_state=42 + iteration,
    )
    tree.fit(X, y)
    branches = analyze_branches(tree, X, y, df)

    with open(tree_path, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'branches': branches,
            'feature_names': FEATURE_NAMES_79D,
            'strategies': STRATEGIES,
            'cv_results': [],
            'iteration': iteration,
        }, f)

    return True


def main():
    args = parse_args()
    file_pairs = get_is_files()
    tree_path = os.path.join(TREE_DIR, 'strategy_tree.pkl')

    print(f'PER-DAY ITERATION — maximize every IS day')
    print(f'  Days: {len(file_pairs)} | Rounds/day: {args.rounds} | Min gap: ${args.min_gap}')

    # Load brain for evidence accumulation
    memory = BayesianMemory()
    with open(tree_path, 'rb') as f:
        tree_data = pickle.load(f)
    memory.commit_branches(tree_data['branches'])

    # Load all trades for label overrides
    with open(TRADE_LOG, 'rb') as f:
        all_trades = pickle.load(f)

    total_improved = 0
    total_pnl_before = 0
    total_pnl_after = 0
    global_iteration = 0

    for day_idx, (feat_file, price_file, day_name) in enumerate(file_pairs):
        gate = Gate(tree_path)
        trades, pnl = run_day_gated(gate, feat_file, price_file)

        if len(trades) < 2:
            continue

        total_pnl_before += pnl
        day_pnl = pnl
        best_pnl = pnl

        for round_num in range(args.rounds):
            global_iteration += 1

            # Compute regret
            total_regret, fixes = compute_day_regret(trades, price_file)

            if total_regret < args.min_gap:
                break  # close enough to optimal

            # Apply fixes
            changed = apply_day_fixes(fixes, day_name, None, global_iteration)
            if not changed:
                break

            # Re-run with updated tree
            gate = Gate(tree_path)
            trades, new_pnl = run_day_gated(gate, feat_file, price_file)

            if new_pnl > day_pnl:
                day_pnl = new_pnl
                best_pnl = new_pnl
            else:
                break  # didn't improve, stop

        # Record to brain
        for t in trades:
            entry_feat = np.array(t['entry_79d']).reshape(1, -1)
            lid = int(gate.tree.apply(np.nan_to_num(entry_feat))[0])
            memory.record_trade(lid, t['pnl'], t['peak'], t['held'])

        improved = best_pnl > pnl
        total_pnl_after += best_pnl
        if improved:
            total_improved += 1

        status = '+' if best_pnl > 0 else '-'
        delta = best_pnl - pnl
        tqdm.write(f'  {day_name}: ${pnl:>7.0f} → ${best_pnl:>7.0f} '
                   f'({"+" if delta>=0 else ""}{delta:.0f}) {status}')

    # Summary
    print(f'\n{"="*60}')
    print(f'PER-DAY ITERATION COMPLETE')
    print(f'{"="*60}')
    print(f'  Days processed: {len(file_pairs)}')
    print(f'  Days improved: {total_improved}')
    print(f'  PnL before: ${total_pnl_before:,.0f} (${total_pnl_before/len(file_pairs):.0f}/day)')
    print(f'  PnL after:  ${total_pnl_after:,.0f} (${total_pnl_after/len(file_pairs):.0f}/day)')

    # Save brain
    memory.save(os.path.join(TREE_DIR, 'memory_is_perday.pkl'))

    # Save report
    report_path = os.path.join(TREE_DIR, 'per_day_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Per-Day Iteration Report\n{"="*60}\n')
        f.write(f'PnL before: ${total_pnl_before:,.0f}\n')
        f.write(f'PnL after: ${total_pnl_after:,.0f}\n')
        f.write(f'Days improved: {total_improved}/{len(file_pairs)}\n')
        f.write(f'\n{memory.summary()}\n')
    print(f'Report: {report_path}')
    print(f'Brain: {TREE_DIR}/memory_is_perday.pkl')


if __name__ == '__main__':
    main()
