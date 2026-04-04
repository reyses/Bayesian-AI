"""
Decision Tree — fractures NMP into transparent strategy branches.

Reads NMP IS trade log. Trains a decision tree on entry 79D features
to predict win/loss. The tree splits are the strategy branches.

Each leaf = a named strategy with known WR, avg PnL, trade count.
Fully transparent — every split is a readable if/then on a 79D feature.

Usage:
    python nn_v2/tree.py                    # train on IS trades
    python nn_v2/tree.py --max-depth 6      # deeper tree (more branches)
    python nn_v2/tree.py --min-leaf 50      # fewer branches, more trades per leaf
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

TRADE_LOG = 'DATA/NMP_TRADES/nmp_is.pkl'
OUTPUT_DIR = 'DATA/NMP_TREE'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Train decision tree on NMP trades')
    p.add_argument('--max-depth', type=int, default=5, help='Max tree depth')
    p.add_argument('--min-leaf', type=int, default=30, help='Min trades per leaf')
    return p.parse_args()


def load_trades():
    """Load IS trade log with full 79D at entry."""
    with open(TRADE_LOG, 'rb') as f:
        trades = pickle.load(f)

    rows = []
    for t in trades:
        entry_79d = np.array(t['entry_79d'])
        row = {
            'pnl': t['pnl'],
            'dir': t['dir'],
            'held': t['held'],
            'peak': t['peak'],
            'exit': t['exit'],
            'win': 1 if t['pnl'] > 0 else 0,
            'day': t.get('day', ''),
        }
        # Add all 79D features
        for i, name in enumerate(FEATURE_NAMES_79D):
            row[name] = entry_79d[i] if i < len(entry_79d) else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def train_tree(df: pd.DataFrame, max_depth: int, min_leaf: int):
    """Train decision tree to predict win/loss from entry 79D."""
    feature_cols = FEATURE_NAMES_79D
    X = df[feature_cols].values
    y = df['win'].values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight='balanced',  # handle imbalanced win/loss
        random_state=42,
    )
    tree.fit(X, y)

    # Cross-validation
    cv_scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')

    return tree, X, y, cv_scores


def analyze_leaves(tree, X, y, df):
    """Analyze each leaf of the tree — these ARE the strategy branches."""
    leaf_ids = tree.apply(X)
    unique_leaves = np.unique(leaf_ids)

    branches = []
    for leaf_id in unique_leaves:
        mask = leaf_ids == leaf_id
        leaf_y = y[mask]
        leaf_df = df.iloc[mask]

        n = len(leaf_y)
        wins = leaf_y.sum()
        wr = wins / n * 100
        avg_pnl = leaf_df['pnl'].mean()
        total_pnl = leaf_df['pnl'].sum()
        avg_held = leaf_df['held'].mean()

        # Direction breakdown
        long_pct = (leaf_df['dir'] == 'long').mean() * 100
        short_pct = (leaf_df['dir'] == 'short').mean() * 100

        # Top 3 features that vary most in this leaf (for naming)
        leaf_features = X[mask]
        feature_means = leaf_features.mean(axis=0)

        branches.append({
            'leaf_id': int(leaf_id),
            'n_trades': n,
            'wins': int(wins),
            'wr': wr,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'avg_held': avg_held,
            'long_pct': long_pct,
            'short_pct': short_pct,
            'prediction': 'WIN' if tree.predict(leaf_features[:1])[0] == 1 else 'LOSS',
        })

    return sorted(branches, key=lambda b: -b['total_pnl'])


def print_report(tree, branches, cv_scores, feature_cols):
    """Print full tree report."""
    print(f'\n{"="*70}')
    print(f'NMP DECISION TREE')
    print(f'{"="*70}')
    print(f'  Depth: {tree.get_depth()} | Leaves: {tree.get_n_leaves()}')
    print(f'  CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})')
    print()

    # Feature importance
    importances = tree.feature_importances_
    top_features = sorted(zip(feature_cols, importances),
                         key=lambda x: -x[1])[:10]
    print(f'  Top 10 Features:')
    for name, imp in top_features:
        bar = '#' * int(imp * 50)
        print(f'    {name:<25} {imp:.3f} {bar}')

    # Branch summary
    print(f'\n  Strategy Branches (leaves):')
    print(f'  {"ID":>4} {"Pred":>4} {"N":>5} {"WR":>6} {"Avg$":>7} {"Total$":>8} {"Hold":>5} {"L%":>4} {"S%":>4}')
    print(f'  {"-"*55}')

    total_profitable = 0
    total_losing = 0
    for b in branches:
        flag = '+' if b['total_pnl'] > 0 else '-'
        if b['total_pnl'] > 0:
            total_profitable += 1
        else:
            total_losing += 1
        print(f'  {b["leaf_id"]:>4} {b["prediction"]:>4} {b["n_trades"]:>5} '
              f'{b["wr"]:>5.0f}% ${b["avg_pnl"]:>6.1f} ${b["total_pnl"]:>7.0f} '
              f'{b["avg_held"]:>5.1f} {b["long_pct"]:>3.0f}% {b["short_pct"]:>3.0f}% {flag}')

    print(f'\n  Profitable branches: {total_profitable} | Losing: {total_losing}')

    # Tree rules (human readable)
    print(f'\n  Tree Rules:')
    rules = export_text(tree, feature_names=feature_cols, max_depth=10)
    # Truncate long trees
    lines = rules.split('\n')
    if len(lines) > 60:
        print('\n'.join(lines[:60]))
        print(f'  ... ({len(lines) - 60} more lines)')
    else:
        print(rules)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Loading IS trades from {TRADE_LOG}...')
    df = load_trades()
    print(f'  {len(df)} trades | WR={df["win"].mean()*100:.0f}%')

    print(f'\nTraining tree (depth={args.max_depth}, min_leaf={args.min_leaf})...')
    tree, X, y, cv_scores = train_tree(df, args.max_depth, args.min_leaf)

    branches = analyze_leaves(tree, X, y, df)
    print_report(tree, branches, cv_scores, FEATURE_NAMES_79D)

    # Save tree + branches
    save_path = os.path.join(OUTPUT_DIR, 'tree.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'branches': branches,
            'feature_names': FEATURE_NAMES_79D,
            'cv_scores': cv_scores,
            'max_depth': args.max_depth,
            'min_leaf': args.min_leaf,
        }, f)
    print(f'\nTree saved: {save_path}')

    # Save branches as CSV for easy viewing
    branches_df = pd.DataFrame(branches)
    branches_csv = os.path.join(OUTPUT_DIR, 'branches.csv')
    branches_df.to_csv(branches_csv, index=False)
    print(f'Branches saved: {branches_csv}')


if __name__ == '__main__':
    main()
