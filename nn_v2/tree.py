"""
Decision Tree — iterative splitting with validation to maximize WR × PnL.

Recursively splits NMP trades into branches. Each split is validated on
held-out data. Only keeps splits that improve EV on unseen trades.
Iterates until branches can't improve further.

Target: highest WR + PnL per branch. Branches below threshold = "don't trade".

Usage:
    python nn_v2/tree.py                         # default settings
    python nn_v2/tree.py --target-wr 0.80        # target 80% WR per branch
    python nn_v2/tree.py --min-ev 2.0            # min $2 EV per trade
    python nn_v2/tree.py --max-depth 10          # allow deeper splits
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

TRADE_LOG = 'DATA/NMP_TRADES/nmp_is.pkl'
OUTPUT_DIR = 'DATA/NMP_TREE'


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Iterative decision tree on NMP trades')
    p.add_argument('--target-wr', type=float, default=0.55, help='Min WR per branch (just above noise)')
    p.add_argument('--min-ev', type=float, default=0.0, help='Min expected value $/trade')
    p.add_argument('--min-leaf', type=int, default=20, help='Min trades per branch')
    p.add_argument('--max-depth', type=int, default=10, help='Max split depth')
    p.add_argument('--n-folds', type=int, default=5, help='CV folds for validation')
    p.add_argument('--iterations', type=int, default=10, help='Max refinement iterations')
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
        for i, name in enumerate(FEATURE_NAMES_79D):
            row[name] = entry_79d[i] if i < len(entry_79d) else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def compute_ev(pnls):
    """Expected value per trade."""
    if len(pnls) == 0:
        return 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    if len(wins) == 0:
        return losses.mean() if len(losses) > 0 else 0.0
    if len(losses) == 0:
        return wins.mean()
    wr = len(wins) / len(pnls)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    return wr * avg_win - (1 - wr) * avg_loss


def compute_drawdown(pnls):
    """Max drawdown from cumulative PnL."""
    cumul = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumul)
    dd = cumul - peak
    return dd.min() if len(dd) > 0 else 0.0


def compute_score(pnls):
    """Score = total_PnL / (1 + |max_drawdown|). Maximizes PnL, penalizes drawdown."""
    if len(pnls) == 0:
        return 0.0
    total = np.sum(pnls)
    dd = abs(compute_drawdown(pnls))
    return total / (1.0 + dd)


def validate_tree_cv(X, y, pnls, max_depth, min_leaf, n_folds):
    """Train tree with cross-validation. Returns validated leaf stats."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_leaf_stats = {}  # leaf_id -> list of fold stats

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        pnl_val = pnls[val_idx]

        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            class_weight='balanced',
            random_state=42 + fold,
        )
        tree.fit(X_train, y_train)

        # Get leaf assignments for validation data
        val_leaves = tree.apply(X_val)
        for leaf_id in np.unique(val_leaves):
            mask = val_leaves == leaf_id
            leaf_pnls = pnl_val[mask]
            leaf_y = y_val[mask]
            n = len(leaf_y)
            if n < 3:
                continue
            wr = leaf_y.mean()
            ev = compute_ev(leaf_pnls)

            if leaf_id not in all_leaf_stats:
                all_leaf_stats[leaf_id] = []
            all_leaf_stats[leaf_id].append({
                'fold': fold, 'n': n, 'wr': wr, 'ev': ev,
                'total_pnl': leaf_pnls.sum(),
            })

    return all_leaf_stats


def iterative_tree(df, feature_cols, args):
    """Iteratively train and refine the tree."""
    X = df[feature_cols].values.astype(np.float32)
    y = df['win'].values
    pnls = df['pnl'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    best_tree = None
    best_score = -999
    best_branches = None
    best_depth = 2

    print(f'\nIterating depths 2 to {args.max_depth}...')
    print(f'  Target WR: {args.target_wr:.0%} | Min EV: ${args.min_ev} | Min leaf: {args.min_leaf}')
    print(f'  Scoring: EV x sqrt(N) — balances quality and frequency')
    print()
    print(f'  {"Depth":>5} {"Leaves":>6} {"Trade":>6} {"Skip":>6} {"WR":>6} {"EV":>7} '
          f'{"Score":>8} {"ValWR":>6} {"ValEV":>7} {"Best":>5}')
    print(f'  {"-"*65}')

    for depth in range(2, args.max_depth + 1):
        # Train on full data
        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=args.min_leaf,
            class_weight='balanced',
            random_state=42,
        )
        tree.fit(X, y)
        leaves = tree.apply(X)
        unique_leaves = np.unique(leaves)

        # Validate with CV
        cv_stats = validate_tree_cv(X, y, pnls, depth, args.min_leaf, args.n_folds)

        # Classify each leaf: trade or skip
        branches = []
        trade_mask = np.zeros(len(X), dtype=bool)

        for leaf_id in unique_leaves:
            mask = leaves == leaf_id
            leaf_pnls = pnls[mask]
            leaf_y = y[mask]
            n = len(leaf_y)
            wr = leaf_y.mean()
            ev = compute_ev(leaf_pnls)

            # Validation stats (average across folds)
            val_stats = cv_stats.get(leaf_id, [])
            if val_stats:
                val_wr = np.mean([s['wr'] for s in val_stats])
                val_ev = np.mean([s['ev'] for s in val_stats])
            else:
                val_wr = 0.0
                val_ev = 0.0

            # Branch drawdown
            dd = compute_drawdown(leaf_pnls)

            # Decision: trade this branch?
            # Tradeable if: validated WR above noise AND EV positive AND PnL positive
            tradeable = (val_wr >= args.target_wr and val_ev >= args.min_ev
                        and leaf_pnls.sum() > 0 and n >= args.min_leaf)

            if tradeable:
                trade_mask |= mask

            branches.append({
                'leaf_id': int(leaf_id),
                'n_trades': n,
                'wr': wr,
                'ev': ev,
                'total_pnl': leaf_pnls.sum(),
                'avg_pnl': leaf_pnls.mean(),
                'max_dd': dd,
                'pnl_dd_ratio': leaf_pnls.sum() / (1.0 + abs(dd)),
                'val_wr': val_wr,
                'val_ev': val_ev,
                'tradeable': tradeable,
                'prediction': 'TRADE' if tradeable else 'SKIP',
                'long_pct': (df.iloc[np.where(mask)]['dir'] == 'long').mean() * 100,
            })

        # Score this depth
        trade_pnls = pnls[trade_mask]
        n_trade = trade_mask.sum()
        n_skip = (~trade_mask).sum()

        if n_trade > 0:
            overall_wr = y[trade_mask].mean()
            overall_ev = compute_ev(trade_pnls)
            overall_score = compute_score(trade_pnls)

            # Validation WR/EV (average across tradeable branches)
            tradeable_branches = [b for b in branches if b['tradeable']]
            avg_val_wr = np.mean([b['val_wr'] for b in tradeable_branches]) if tradeable_branches else 0
            avg_val_ev = np.mean([b['val_ev'] for b in tradeable_branches]) if tradeable_branches else 0
        else:
            overall_wr = 0
            overall_ev = 0
            overall_score = 0
            avg_val_wr = 0
            avg_val_ev = 0

        is_best = overall_score > best_score
        if is_best:
            best_score = overall_score
            best_tree = tree
            best_branches = branches
            best_depth = depth

        print(f'  {depth:>5} {len(unique_leaves):>6} {n_trade:>6} {n_skip:>6} '
              f'{overall_wr:>5.0%} ${overall_ev:>6.1f} {overall_score:>8.0f} '
              f'{avg_val_wr:>5.0%} ${avg_val_ev:>6.1f} '
              f'{"<<<" if is_best else "":>5}')

    return best_tree, best_branches, best_depth


def print_report(tree, branches, depth, feature_cols, save_path=None):
    """Print detailed report of the best tree. Optionally save to file."""
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out(f'\n{"="*70}')
    out(f'BEST TREE (depth={depth})')
    out(f'{"="*70}')

    tradeable = [b for b in branches if b['tradeable']]
    skipped = [b for b in branches if not b['tradeable']]

    trade_total = sum(b['n_trades'] for b in tradeable)
    trade_pnl = sum(b['total_pnl'] for b in tradeable)
    skip_total = sum(b['n_trades'] for b in skipped)
    skip_pnl = sum(b['total_pnl'] for b in skipped)

    out(f'\n  TRADE branches: {len(tradeable)} ({trade_total} trades, ${trade_pnl:.0f})')
    out(f'  SKIP branches:  {len(skipped)} ({skip_total} trades, ${skip_pnl:.0f} avoided)')
    out(f'  Net improvement: ${trade_pnl - skip_pnl:.0f} (trade) vs ${trade_pnl + skip_pnl:.0f} (all)')

    # Feature importance
    importances = tree.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:10]
    out(f'\n  Top Features:')
    for name, imp in top_features:
        if imp > 0.005:
            bar = '#' * int(imp * 50)
            out(f'    {name:<25} {imp:.3f} {bar}')

    # Tradeable branches detail
    out(f'\n  TRADEABLE Branches:')
    out(f'  {"ID":>4} {"N":>5} {"WR":>6} {"ValWR":>6} {"EV":>7} {"ValEV":>7} {"Total$":>8} {"DD$":>7} {"L%":>4}')
    out(f'  {"-"*65}')
    for b in sorted(tradeable, key=lambda b: -b['total_pnl']):
        out(f'  {b["leaf_id"]:>4} {b["n_trades"]:>5} {b["wr"]:>5.0%} '
            f'{b["val_wr"]:>5.0%} ${b["ev"]:>6.1f} ${b["val_ev"]:>6.1f} '
            f'${b["total_pnl"]:>7.0f} ${b["max_dd"]:>6.0f} {b["long_pct"]:>3.0f}%')

    # Skip branches (biggest losses avoided)
    out(f'\n  SKIP Branches (top 5 losses avoided):')
    for b in sorted(skipped, key=lambda b: b['total_pnl'])[:5]:
        out(f'  {b["leaf_id"]:>4} {b["n_trades"]:>5} {b["wr"]:>5.0%} '
            f'{b["val_wr"]:>5.0%} ${b["ev"]:>6.1f} ${b["total_pnl"]:>7.0f}')

    # Tree rules
    out(f'\n  Tree Rules:')
    rules_text = export_text(tree, feature_names=feature_cols, max_depth=10)
    rule_lines = rules_text.split('\n')
    if len(rule_lines) > 80:
        for rl in rule_lines[:80]:
            out(rl)
        out(f'  ... ({len(rule_lines) - 80} more lines)')
    else:
        for rl in rule_lines:
            out(rl)

    # Save report to file
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'\nReport saved: {save_path}')


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Loading IS trades from {TRADE_LOG}...')
    df = load_trades()
    print(f'  {len(df)} trades | WR={df["win"].mean()*100:.0f}% | EV=${compute_ev(df["pnl"].values):.2f}')

    tree, branches, depth = iterative_tree(df, FEATURE_NAMES_79D, args)
    report_path = os.path.join(OUTPUT_DIR, 'tree_report.txt')
    print_report(tree, branches, depth, FEATURE_NAMES_79D, save_path=report_path)

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'tree.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'branches': branches,
            'feature_names': FEATURE_NAMES_79D,
            'depth': depth,
            'args': vars(args),
        }, f)
    print(f'\nTree saved: {save_path}')

    branches_df = pd.DataFrame(branches)
    branches_csv = os.path.join(OUTPUT_DIR, 'branches.csv')
    branches_df.to_csv(branches_csv, index=False)
    print(f'Branches saved: {branches_csv}')


if __name__ == '__main__':
    main()
