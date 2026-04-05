"""
Decision Tree V2 — classifies 79D into strategies using regret-derived labels.

Instead of predicting win/loss, predicts the OPTIMAL ACTION per setup:
  - same_extended:    hold same direction longer (peak at ~16 bars)
  - counter_extended: flip direction and hold (peak at ~17 bars)
  - same_early:       same direction but exit fast (1-3 bars)
  - counter_early:    flip and exit fast
  - skip:             no profitable action exists

Each branch = a complete strategy: direction + exit timing.
Trained on regret analysis ground truth from IS NMP trades.

Usage:
    python nn_v2/tree.py                         # train with defaults
    python nn_v2/tree.py --max-depth 8           # deeper tree
    python nn_v2/tree.py --min-leaf 20           # smaller branches
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

TRADE_LOG = 'nn_v2/output/trades/nmp_is.pkl'
CORRECTED_LOG = 'nn_v2/output/trades/corrected_is.pkl'
REGRET_FILE = 'nn_v2/output/tree/regret_analysis.csv'
OUTPUT_DIR = 'nn_v2/output/tree'

# Strategy classes (old: regret action labels)
STRATEGIES = ['same_extended', 'counter_extended', 'same_early', 'counter_early', 'same_at_exit', 'counter_at_exit']
STRAT_TO_IDX = {s: i for i, s in enumerate(STRATEGIES)}

# Corrected trade labels: direction + hold bucket
HOLD_BUCKETS = [(0, 3, 'fast'), (3, 8, 'medium'), (8, 16, 'long'), (16, 999, 'extended')]
CORRECTED_STRATEGIES = []
for d in ['long', 'short']:
    for _, _, label in HOLD_BUCKETS:
        CORRECTED_STRATEGIES.append(f'{d}_{label}')
CORR_STRAT_TO_IDX = {s: i for i, s in enumerate(CORRECTED_STRATEGIES)}


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Strategy tree from regret labels')
    p.add_argument('--max-depth', type=int, default=8, help='Max tree depth')
    p.add_argument('--min-leaf', type=int, default=20, help='Min trades per leaf')
    return p.parse_args()


def load_data():
    """Load 79D features + regret labels."""
    # 79D from trades
    with open(TRADE_LOG, 'rb') as f:
        trades = pickle.load(f)

    # Regret labels
    regret = pd.read_csv(REGRET_FILE)

    rows = []
    for i, t in enumerate(trades):
        entry_79d = np.array(t['entry_79d'])
        r = regret.iloc[i] if i < len(regret) else None

        row = {
            'pnl': t['pnl'],
            'dir': t['dir'],
            'held': t['held'],
            'day': t.get('day', ''),
        }

        # 79D features
        for j, name in enumerate(FEATURE_NAMES_79D):
            row[name] = entry_79d[j] if j < len(entry_79d) else 0.0

        # Regret labels
        if r is not None:
            row['best_action'] = r['best_action']
            row['best_pnl'] = r['best_pnl']
            row['regret'] = r['regret']
            row['same_best_bar'] = r['same_best_bar']
            row['counter_best_bar'] = r['counter_best_bar']
            row['strategy_idx'] = STRAT_TO_IDX.get(r['best_action'], 0)
        else:
            row['best_action'] = 'same_at_exit'
            row['best_pnl'] = t['pnl']
            row['regret'] = 0
            row['same_best_bar'] = t['held']
            row['counter_best_bar'] = 0
            row['strategy_idx'] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def load_corrected_data():
    """Load corrected trades (regret-corrected direction + exit)."""
    with open(CORRECTED_LOG, 'rb') as f:
        trades = pickle.load(f)

    rows = []
    for t in trades:
        entry_79d = np.array(t['entry_79d'])
        direction = t['dir']
        held = t['held']

        # Classify into direction + hold bucket
        bucket_label = 'extended'  # default
        for lo, hi, label in HOLD_BUCKETS:
            if lo <= held < hi:
                bucket_label = label
                break
        strategy = f'{direction}_{bucket_label}'

        row = {
            'pnl': t['pnl'],
            'dir': direction,
            'held': held,
            'day': t.get('day', ''),
            'best_action': t.get('best_action', strategy),
            'best_pnl': t['pnl'],  # corrected PnL IS the best
            'regret': 0.0,          # no regret — this is the corrected trade
            'same_best_bar': held if direction == t.get('original_dir', direction) else 0,
            'counter_best_bar': held if direction != t.get('original_dir', direction) else 0,
            'strategy_idx': CORR_STRAT_TO_IDX.get(strategy, 0),
        }

        # 79D features
        for j, name in enumerate(FEATURE_NAMES_79D):
            row[name] = entry_79d[j] if j < len(entry_79d) else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def train_tree(df, feature_cols, max_depth, min_leaf):
    """Train tree to classify 79D → strategy (from regret labels)."""
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
    y = df['strategy_idx'].values

    # Cross-validate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        tree = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf,
            random_state=42 + fold,
        )
        tree.fit(X[train_idx], y[train_idx])
        val_pred = tree.predict(X[val_idx])
        val_y = y[val_idx]
        acc = (val_pred == val_y).mean()

        # PnL if we followed the tree's recommended strategy
        val_best_pnl = df.iloc[val_idx]['best_pnl'].values
        val_actual_pnl = df.iloc[val_idx]['pnl'].values

        # For correctly classified trades, use best_pnl. For wrong, use actual.
        correct_mask = val_pred == val_y
        sim_pnl = np.where(correct_mask, val_best_pnl, val_actual_pnl)

        cv_results.append({
            'fold': fold, 'accuracy': acc,
            'sim_pnl': sim_pnl.sum(), 'actual_pnl': val_actual_pnl.sum(),
            'optimal_pnl': val_best_pnl.sum(),
        })

    # Train final tree on all data
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42,
    )
    tree.fit(X, y)

    return tree, X, y, cv_results


def analyze_branches(tree, X, y, df, strat_list=None):
    """Analyze each leaf — what strategy does it recommend?"""
    if strat_list is None:
        strat_list = STRATEGIES
    leaves = tree.apply(X)
    unique_leaves = np.unique(leaves)

    branches = []
    for lid in unique_leaves:
        mask = leaves == lid
        leaf_df = df.iloc[np.where(mask)[0]]
        n = len(leaf_df)
        leaf_y = y[mask]

        # Dominant strategy in this leaf
        strategy_counts = pd.Series(leaf_y).value_counts()
        dominant_idx = strategy_counts.index[0]
        dominant_strategy = strat_list[dominant_idx] if dominant_idx < len(strat_list) else f'class_{dominant_idx}'
        dominant_pct = strategy_counts.iloc[0] / n * 100

        # PnL stats
        actual_pnl = leaf_df['pnl'].sum()
        optimal_pnl = leaf_df['best_pnl'].sum()
        avg_regret = leaf_df['regret'].mean()

        # Exit timing
        avg_same_bar = leaf_df['same_best_bar'].mean()
        avg_counter_bar = leaf_df['counter_best_bar'].mean()

        # Direction
        long_pct = (leaf_df['dir'] == 'long').mean() * 100

        branches.append({
            'leaf_id': int(lid),
            'n_trades': n,
            'strategy': dominant_strategy,
            'strategy_pct': dominant_pct,
            'actual_pnl': actual_pnl,
            'optimal_pnl': optimal_pnl,
            'recoverable': optimal_pnl - actual_pnl,
            'avg_regret': avg_regret,
            'exit_bar_same': avg_same_bar,
            'exit_bar_counter': avg_counter_bar,
            'long_pct': long_pct,
            'wr': (leaf_df['pnl'] > 0).mean() * 100,
        })

    return sorted(branches, key=lambda b: -b['optimal_pnl'])


def print_report(tree, branches, cv_results, feature_cols, save_path=None):
    """Print and save report."""
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out(f'\n{"="*70}')
    out(f'STRATEGY TREE (regret-trained)')
    out(f'{"="*70}')
    out(f'  Depth: {tree.get_depth()} | Leaves: {tree.get_n_leaves()}')

    # CV results
    avg_acc = np.mean([r['accuracy'] for r in cv_results])
    avg_sim = np.mean([r['sim_pnl'] for r in cv_results])
    avg_actual = np.mean([r['actual_pnl'] for r in cv_results])
    avg_optimal = np.mean([r['optimal_pnl'] for r in cv_results])
    out(f'  CV Accuracy: {avg_acc:.1%}')
    out(f'  CV PnL (simulated): ${avg_sim:,.0f} (actual: ${avg_actual:,.0f}, optimal: ${avg_optimal:,.0f})')

    # Strategy distribution across branches
    out(f'\n  Strategy Distribution:')
    strat_counts = {}
    for b in branches:
        s = b['strategy']
        if s not in strat_counts:
            strat_counts[s] = {'branches': 0, 'trades': 0, 'optimal': 0}
        strat_counts[s]['branches'] += 1
        strat_counts[s]['trades'] += b['n_trades']
        strat_counts[s]['optimal'] += b['optimal_pnl']
    for s in STRATEGIES:
        if s in strat_counts:
            sc = strat_counts[s]
            out(f'    {s:<22} {sc["branches"]:>3} branches  {sc["trades"]:>5} trades  '
                f'optimal=${sc["optimal"]:>8,.0f}')

    # Feature importance
    importances = tree.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:10]
    out(f'\n  Top Features:')
    for name, imp in top_features:
        if imp > 0.005:
            bar = '#' * int(imp * 50)
            out(f'    {name:<25} {imp:.3f} {bar}')

    # Branch detail
    out(f'\n  Branches (by optimal PnL):')
    out(f'  {"ID":>4} {"N":>5} {"Strategy":<22} {"Pct":>4} {"Actual$":>8} {"Optimal$":>9} {"Recover$":>9} {"ExitBar":>7} {"WR":>5}')
    out(f'  {"-"*80}')
    for b in branches[:25]:
        exit_bar = b['exit_bar_same'] if 'same' in b['strategy'] else b['exit_bar_counter']
        out(f'  {b["leaf_id"]:>4} {b["n_trades"]:>5} {b["strategy"]:<22} '
            f'{b["strategy_pct"]:>3.0f}% ${b["actual_pnl"]:>7.0f} ${b["optimal_pnl"]:>8.0f} '
            f'${b["recoverable"]:>8.0f} {exit_bar:>6.1f} {b["wr"]:>4.0f}%')

    # Tree rules (first 60 lines)
    out(f'\n  Tree Rules:')
    rules = export_text(tree, feature_names=feature_cols, max_depth=10)
    rule_lines = rules.split('\n')
    for rl in rule_lines[:60]:
        out(rl)
    if len(rule_lines) > 60:
        out(f'  ... ({len(rule_lines) - 60} more lines)')

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'\nReport saved: {save_path}')


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prefer corrected trades if available
    use_corrected = os.path.exists(CORRECTED_LOG)
    if use_corrected:
        print(f'Loading CORRECTED trades (regret-corrected direction + exit)...')
        df = load_corrected_data()
        strat_list = CORRECTED_STRATEGIES
        strat_map = CORR_STRAT_TO_IDX
    else:
        print(f'Loading raw trades + regret labels...')
        df = load_data()
        strat_list = STRATEGIES
        strat_map = STRAT_TO_IDX

    print(f'  {len(df)} trades | {len(strat_list)} strategy classes')
    print(f'  Label distribution:')
    for s in strat_list:
        idx = strat_map[s]
        n = (df['strategy_idx'] == idx).sum()
        if n > 0:
            print(f'    {s:<22} {n:>5} ({n/len(df)*100:>4.0f}%)')

    print(f'\nTraining tree (depth={args.max_depth}, min_leaf={args.min_leaf})...')
    tree, X, y, cv_results = train_tree(df, FEATURE_NAMES_79D, args.max_depth, args.min_leaf)

    branches = analyze_branches(tree, X, y, df, strat_list=strat_list)
    report_path = os.path.join(OUTPUT_DIR, 'strategy_tree_report.txt')
    print_report(tree, branches, cv_results, FEATURE_NAMES_79D, save_path=report_path)

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'strategy_tree.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'branches': branches,
            'feature_names': FEATURE_NAMES_79D,
            'strategies': strat_list,
            'cv_results': cv_results,
            'args': vars(args),
            'corrected': use_corrected,
        }, f)
    print(f'Tree saved: {save_path}')

    branches_df = pd.DataFrame(branches)
    branches_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_branches.csv'), index=False)
    print(f'Branches saved: {OUTPUT_DIR}/strategy_branches.csv')


if __name__ == '__main__':
    main()
