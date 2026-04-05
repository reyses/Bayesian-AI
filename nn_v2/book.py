"""
Strategy Book — generates the complete playbook from tree + regret data.

Each branch becomes a page in the book:
  SETUP:  what the 79D looks like in the 10 bars before entry
  ENTRY:  the tree split conditions (79D at entry)
  PATH:   expected PnL curve bar by bar (from regret same_curve / counter_curve)
  EXIT:   79D conditions at the optimal exit bar

The book is the FINAL output of the learning pipeline.
The trial run executes FROM the book.

Usage:
    python nn_v2/book.py                    # generate from tree + regret
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

TRADE_LOG = 'DATA/NMP_TRADES/nmp_is.pkl'
REGRET_FILE = 'DATA/NMP_TREE/regret_analysis.csv'
TREE_FILE = 'DATA/NMP_TREE/strategy_tree.pkl'
OUTPUT_DIR = 'DATA/NMP_TREE'


class Strategy:
    """One page in the strategy book."""

    def __init__(self, leaf_id: int):
        self.leaf_id = leaf_id
        self.n_trades = 0
        self.direction = ''          # 'same' or 'counter'
        self.action = ''             # full action name

        # SETUP: mean 79D at entry
        self.entry_79d_mean = np.zeros(len(FEATURE_NAMES_79D))
        self.entry_79d_std = np.zeros(len(FEATURE_NAMES_79D))

        # PATH: expected PnL curve (mean across trades in this branch)
        self.expected_path = []      # list of floats, PnL at each bar
        self.path_std = []           # std at each bar

        # EXIT: optimal bar + 79D at exit
        self.optimal_exit_bar = 0.0
        self.exit_79d_mean = np.zeros(len(FEATURE_NAMES_79D))

        # STATS
        self.actual_pnl = 0.0
        self.optimal_pnl = 0.0
        self.wr = 0.0
        self.avg_regret = 0.0

    def to_dict(self):
        return {
            'leaf_id': self.leaf_id,
            'n_trades': self.n_trades,
            'direction': self.direction,
            'action': self.action,
            'optimal_exit_bar': self.optimal_exit_bar,
            'actual_pnl': self.actual_pnl,
            'optimal_pnl': self.optimal_pnl,
            'wr': self.wr,
            'avg_regret': self.avg_regret,
            'expected_path': self.expected_path,
            'entry_79d_mean': self.entry_79d_mean.tolist(),
            'exit_79d_mean': self.exit_79d_mean.tolist(),
        }


def load_all_data():
    """Load trades (with paths), regret analysis, and tree."""
    with open(TRADE_LOG, 'rb') as f:
        trades = pickle.load(f)

    regret = pd.read_csv(REGRET_FILE)

    with open(TREE_FILE, 'rb') as f:
        tree_data = pickle.load(f)

    tree = tree_data['tree']

    # Classify each trade into its branch
    for i, t in enumerate(trades):
        feat = np.array(t['entry_79d']).reshape(1, -1)
        feat = np.nan_to_num(feat)
        t['leaf_id'] = int(tree.apply(feat)[0])
        t['regret'] = regret.iloc[i].to_dict() if i < len(regret) else {}

    return trades, regret, tree_data


def build_book(trades, regret_df, tree_data):
    """Build the strategy book from tree branches + regret data."""
    strategies = {}

    # Group trades by branch
    by_branch = defaultdict(list)
    for t in trades:
        by_branch[t['leaf_id']].append(t)

    # For each branch, build the strategy page
    for lid, branch_trades in by_branch.items():
        strat = Strategy(lid)
        strat.n_trades = len(branch_trades)

        # Get regret info for these trades
        branch_regrets = regret_df[regret_df['trade_id'].isin(
            [t.get('trade_id', i) for i, t in enumerate(branch_trades)]
        )] if 'trade_id' in regret_df.columns else pd.DataFrame()

        # Use index-based matching if trade_id doesn't work
        branch_indices = [i for i, t in enumerate(trades) if t['leaf_id'] == lid]
        branch_regrets = regret_df.iloc[branch_indices] if branch_indices else pd.DataFrame()

        if len(branch_regrets) == 0:
            continue

        # Dominant action
        action = branch_regrets['best_action'].mode().iloc[0] if len(branch_regrets) > 0 else 'same_extended'
        strat.action = action
        strat.direction = 'counter' if 'counter' in action else 'same'

        # ENTRY 79D: mean of entry features
        entry_feats = np.array([np.array(t['entry_79d']) for t in branch_trades])
        entry_feats = np.nan_to_num(entry_feats)
        strat.entry_79d_mean = entry_feats.mean(axis=0)
        strat.entry_79d_std = entry_feats.std(axis=0)

        # EXPECTED PATH: from trade paths
        # Use same_curve or counter_curve from regret (stored in pkl trades)
        all_paths = []
        for t in branch_trades:
            path = t.get('path', [])
            if path:
                pnls = [p['pnl'] for p in path]
                if strat.direction == 'counter':
                    pnls = [-p for p in pnls]  # flip for counter
                all_paths.append(pnls)

        if all_paths:
            # Pad to same length
            max_len = min(30, max(len(p) for p in all_paths))
            padded = []
            for p in all_paths:
                if len(p) >= max_len:
                    padded.append(p[:max_len])
                else:
                    padded.append(p + [p[-1]] * (max_len - len(p)))
            path_array = np.array(padded)
            strat.expected_path = path_array.mean(axis=0).tolist()
            strat.path_std = path_array.std(axis=0).tolist()

        # OPTIMAL EXIT BAR
        if strat.direction == 'counter':
            strat.optimal_exit_bar = branch_regrets['counter_best_bar'].mean()
        else:
            strat.optimal_exit_bar = branch_regrets['same_best_bar'].mean()

        # EXIT 79D: from trade paths at optimal exit bar
        exit_feats = []
        for t in branch_trades:
            path = t.get('path', [])
            opt_bar = int(strat.optimal_exit_bar)
            if path and opt_bar < len(path) and 'features_79d' in path[min(opt_bar, len(path)-1)]:
                exit_feats.append(np.array(path[min(opt_bar, len(path)-1)]['features_79d']))
        if exit_feats:
            exit_array = np.nan_to_num(np.array(exit_feats))
            strat.exit_79d_mean = exit_array.mean(axis=0)

        # STATS
        strat.actual_pnl = sum(t['pnl'] for t in branch_trades)
        strat.optimal_pnl = branch_regrets['best_pnl'].sum()
        strat.wr = sum(1 for t in branch_trades if t['pnl'] > 0) / max(len(branch_trades), 1)
        strat.avg_regret = branch_regrets['regret'].mean()

        strategies[lid] = strat

    return strategies


def print_book(strategies, save_path=None):
    """Print the strategy book."""
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out(f'\n{"="*70}')
    out(f'STRATEGY BOOK — {len(strategies)} strategies')
    out(f'{"="*70}')

    total_actual = sum(s.actual_pnl for s in strategies.values())
    total_optimal = sum(s.optimal_pnl for s in strategies.values())
    total_trades = sum(s.n_trades for s in strategies.values())

    out(f'  Total trades: {total_trades}')
    out(f'  Actual PnL:  ${total_actual:,.0f}')
    out(f'  Optimal PnL: ${total_optimal:,.0f}')

    # Group by action
    by_action = defaultdict(list)
    for s in strategies.values():
        by_action[s.action].append(s)

    out(f'\n  By strategy type:')
    for action in sorted(by_action.keys()):
        strats = by_action[action]
        n = sum(s.n_trades for s in strats)
        opt = sum(s.optimal_pnl for s in strats)
        out(f'    {action:<22} {len(strats):>3} branches  {n:>5} trades  optimal=${opt:>9,.0f}')

    # Top 20 strategies by optimal PnL
    sorted_strats = sorted(strategies.values(), key=lambda s: -s.optimal_pnl)

    out(f'\n  Top 20 Strategies:')
    out(f'  {"ID":>4} {"N":>5} {"Action":<18} {"Dir":<7} {"ExitBar":>7} {"Actual$":>8} {"Optimal$":>9} {"WR":>5}')
    out(f'  {"-"*70}')

    for s in sorted_strats[:20]:
        out(f'  {s.leaf_id:>4} {s.n_trades:>5} {s.action:<18} {s.direction:<7} '
            f'{s.optimal_exit_bar:>6.1f} ${s.actual_pnl:>7.0f} ${s.optimal_pnl:>8.0f} '
            f'{s.wr:>4.0%}')

    # Detail for top 5
    out(f'\n  {"="*70}')
    out(f'  DETAILED PLAYBOOKS (top 5)')
    out(f'  {"="*70}')

    for s in sorted_strats[:5]:
        out(f'\n  --- Strategy {s.leaf_id} ({s.action}) ---')
        out(f'  Trades: {s.n_trades} | WR: {s.wr:.0%} | Optimal exit: bar {s.optimal_exit_bar:.0f}')
        out(f'  Actual: ${s.actual_pnl:.0f} | Optimal: ${s.optimal_pnl:.0f} | Regret: ${s.avg_regret:.0f}/trade')

        # Entry signature (top 5 non-zero features)
        entry_importance = sorted(
            zip(FEATURE_NAMES_79D, s.entry_79d_mean, s.entry_79d_std),
            key=lambda x: -abs(x[1])
        )
        out(f'  ENTRY signature (top features):')
        for name, mean, std in entry_importance[:7]:
            if abs(mean) > 0.01:
                out(f'    {name:<25} mean={mean:>+8.2f}  std={std:>6.2f}')

        # Expected path
        if s.expected_path:
            out(f'  EXPECTED PATH (PnL per bar):')
            path_str = '  '.join(f'${p:>+6.1f}' for p in s.expected_path[:15])
            out(f'    {path_str}')
            # Find peak
            peak_bar = np.argmax(s.expected_path)
            peak_pnl = s.expected_path[peak_bar]
            out(f'    Peak: ${peak_pnl:.1f} at bar {peak_bar}')

        # Exit signature
        exit_importance = sorted(
            zip(FEATURE_NAMES_79D, s.exit_79d_mean),
            key=lambda x: -abs(x[1])
        )
        out(f'  EXIT signature (top features at optimal exit):')
        for name, mean in exit_importance[:5]:
            if abs(mean) > 0.01:
                out(f'    {name:<25} mean={mean:>+8.2f}')

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'\nBook saved: {save_path}')


def main():
    print('Building Strategy Book...')

    trades, regret_df, tree_data = load_all_data()
    print(f'  {len(trades)} trades, {len(regret_df)} regret records')

    strategies = build_book(trades, regret_df, tree_data)
    print(f'  {len(strategies)} strategies built')

    report_path = os.path.join(OUTPUT_DIR, 'strategy_book.txt')
    print_book(strategies, save_path=report_path)

    # Save book as pkl
    book_path = os.path.join(OUTPUT_DIR, 'strategy_book.pkl')
    book_data = {lid: s.to_dict() for lid, s in strategies.items()}
    with open(book_path, 'wb') as f:
        pickle.dump(book_data, f)
    print(f'Book data saved: {book_path}')


if __name__ == '__main__':
    main()
