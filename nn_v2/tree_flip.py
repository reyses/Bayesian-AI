"""
Flip Predictor Tree — predicts SAME vs COUNTER from 79D at entry.

Trained on blended NMP trades + regret labels. For each trade, regret
tells us if the NMP direction was correct (same_*) or wrong (counter_*).
The tree learns: given this 79D + tier, will the direction flip?

If the tree can predict flips at 70%+ accuracy, we skip or flip the
counter trades and dramatically improve WR.

Usage:
    python nn_v2/tree_flip.py                    # train + report
    python nn_v2/tree_flip.py --max-depth 10     # deeper tree
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

BLENDED_TRADES = 'nn_v2/output/trades/blended_is.pkl'
REGRET_FILE = 'nn_v2/output/tree/regret_analysis.csv'
OUTPUT_DIR = 'nn_v2/output/tree'

# Tier encoding
TIER_MAP = {'CASCADE': 2, 'KILL_SHOT': 1, 'BASE_NMP': 0}


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Flip predictor tree')
    p.add_argument('--max-depth', type=int, default=8)
    p.add_argument('--min-leaf', type=int, default=20)
    return p.parse_args()


def load_data():
    """Load blended trades + regret labels → build flip dataset."""
    with open(BLENDED_TRADES, 'rb') as f:
        trades = pickle.load(f)

    # Load regret to get best_action per trade
    regret = pd.read_csv(REGRET_FILE)

    print(f'Blended trades: {len(trades)}')
    print(f'Regret records: {len(regret)}')

    rows = []
    for i, t in enumerate(trades):
        entry_79d = np.array(t.get('entry_79d', []))
        if len(entry_79d) != len(FEATURE_NAMES_79D):
            continue

        # Get regret label for this trade
        if i < len(regret):
            best_action = regret.iloc[i]['best_action']
        else:
            # Fallback: use PnL as proxy (won = same, lost = counter)
            best_action = 'same_extended' if t['pnl'] > 0 else 'counter_extended'

        # Binary: same (0) or counter (1)
        is_counter = 1 if 'counter' in best_action else 0

        tier = TIER_MAP.get(t.get('entry_tier', 'BASE_NMP'), 0)

        row = {
            'flip': is_counter,
            'tier': tier,
            'pnl': t['pnl'],
            'dir': t.get('dir', ''),
            'entry_tier': t.get('entry_tier', 'BASE_NMP'),
        }

        # 79D features
        for j, name in enumerate(FEATURE_NAMES_79D):
            row[name] = float(entry_79d[j])

        rows.append(row)

    return pd.DataFrame(rows)


def train_flip_tree(df, feature_cols, max_depth, min_leaf):
    """Train tree to predict same vs counter."""
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
    y = df['flip'].values

    # Cross-validate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        tree = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42 + fold,
        )
        tree.fit(X[train_idx], y[train_idx])
        val_pred = tree.predict(X[val_idx])
        val_y = y[val_idx]
        acc = (val_pred == val_y).mean()

        # PnL impact: if we skip counter predictions, what's the PnL?
        val_pnl = df.iloc[val_idx]['pnl'].values
        # Trades where tree says SAME (pred=0) → keep
        keep_mask = val_pred == 0
        kept_pnl = val_pnl[keep_mask].sum()
        skipped_pnl = val_pnl[~keep_mask].sum()
        kept_n = keep_mask.sum()
        skipped_n = (~keep_mask).sum()

        cv_results.append({
            'fold': fold, 'accuracy': acc,
            'kept_n': kept_n, 'skipped_n': skipped_n,
            'kept_pnl': kept_pnl, 'skipped_pnl': skipped_pnl,
        })

    # Train final tree on all data
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42,
    )
    tree.fit(X, y)

    return tree, X, y, cv_results


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading blended trades + regret labels...')
    df = load_data()
    print(f'Dataset: {len(df)} trades')
    print(f'  SAME (direction correct): {(df["flip"]==0).sum()} ({(df["flip"]==0).mean()*100:.0f}%)')
    print(f'  COUNTER (direction wrong): {(df["flip"]==1).sum()} ({(df["flip"]==1).mean()*100:.0f}%)')

    # Per tier
    print(f'\n  By tier:')
    for tier_name in ['CASCADE', 'KILL_SHOT', 'BASE_NMP']:
        sub = df[df['entry_tier'] == tier_name]
        if len(sub) == 0: continue
        same_pct = (sub['flip'] == 0).mean() * 100
        print(f'    {tier_name:<15} {len(sub):>5} trades  SAME={same_pct:.0f}%  COUNTER={100-same_pct:.0f}%')

    # Features: 79D + tier
    feature_cols = list(FEATURE_NAMES_79D) + ['tier']

    print(f'\nTraining flip predictor (depth={args.max_depth}, min_leaf={args.min_leaf})...')
    tree, X, y, cv_results = train_flip_tree(df, feature_cols, args.max_depth, args.min_leaf)

    # CV results
    avg_acc = np.mean([r['accuracy'] for r in cv_results])
    avg_kept_pnl = np.mean([r['kept_pnl'] for r in cv_results])
    avg_skipped_pnl = np.mean([r['skipped_pnl'] for r in cv_results])
    avg_kept_n = np.mean([r['kept_n'] for r in cv_results])
    avg_skipped_n = np.mean([r['skipped_n'] for r in cv_results])

    print(f'\n{"="*60}')
    print(f'FLIP PREDICTOR RESULTS')
    print(f'{"="*60}')
    print(f'  CV Accuracy: {avg_acc:.1%}')
    print(f'  Tree depth: {tree.get_depth()}, leaves: {tree.get_n_leaves()}')
    print(f'\n  If we SKIP trades predicted as COUNTER:')
    print(f'    Kept: {avg_kept_n:.0f} trades, PnL=${avg_kept_pnl:,.0f}')
    print(f'    Skipped: {avg_skipped_n:.0f} trades, PnL=${avg_skipped_pnl:,.0f}')
    print(f'    Kept $/trade: ${avg_kept_pnl/max(avg_kept_n,1):.1f}')
    print(f'    Skipped $/trade: ${avg_skipped_pnl/max(avg_skipped_n,1):.1f}')

    # If we FLIP trades predicted as counter (instead of skipping)
    print(f'\n  If we FLIP trades predicted as COUNTER:')
    for r in cv_results:
        # Flipping counter predictions inverts their PnL
        pass
    # Approximate: flipped PnL = kept_pnl + abs(skipped_pnl) (rough)
    print(f'    (approximate — flipping inverts the loss trades)')

    # Feature importance
    importances = tree.feature_importances_
    top = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
    print(f'\n  Top features:')
    for name, imp in top:
        if imp > 0.001:
            print(f'    {name:<25} {imp:.3f}')

    # Per-fold detail
    print(f'\n  Per-fold:')
    for r in cv_results:
        kept_avg = r['kept_pnl'] / max(r['kept_n'], 1)
        skip_avg = r['skipped_pnl'] / max(r['skipped_n'], 1)
        print(f'    Fold {r["fold"]}: acc={r["accuracy"]:.1%} | '
              f'kept={r["kept_n"]:.0f} (${kept_avg:.1f}/tr) | '
              f'skipped={r["skipped_n"]:.0f} (${skip_avg:.1f}/tr)')

    # Save tree
    save_path = os.path.join(OUTPUT_DIR, 'flip_tree.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'feature_cols': feature_cols,
            'cv_results': cv_results,
            'accuracy': avg_acc,
            'args': vars(args),
        }, f)
    print(f'\nTree saved: {save_path}')

    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'flip_tree_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Flip Predictor Tree\n{"="*40}\n')
        f.write(f'Accuracy: {avg_acc:.1%}\n')
        f.write(f'Depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}\n\n')
        f.write(f'Top features:\n')
        for name, imp in top:
            if imp > 0.001:
                f.write(f'  {name:<25} {imp:.3f}\n')
        f.write(f'\nTree rules (first 50 lines):\n')
        rules = export_text(tree, feature_names=feature_cols, max_depth=5)
        f.write(rules[:3000])
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()
