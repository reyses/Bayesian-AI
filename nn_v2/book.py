"""
Strategy Book — raw playbook per leaf with full regret profiles.

Each leaf gets a page with:
  RAW STRATEGY:   what the tree classified (no rewriting)
  REGRET PROFILE: distribution of best actions (same_early, counter_extended, etc.)
  APPROACH:       mean 79D path in bars before entry
  ENTRY:          mean 79D at entry
  SAME PATH:      expected PnL curve if following NMP direction
  COUNTER PATH:   expected PnL curve if counter-trading
  EXIT:           79D at optimal exit bar

The book does NOT pick a single answer — it carries the full picture.
The AI + brain decide which action to take based on accumulated evidence.

Usage:
    python nn_v2/book.py                    # generate from tree + regret + trades
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

# All 6 regret actions
REGRET_ACTIONS = [
    'same_early', 'same_at_exit', 'same_extended',
    'counter_early', 'counter_at_exit', 'counter_extended',
]


class Strategy:
    """One page in the strategy book — raw data, no opinions."""

    def __init__(self, leaf_id: int):
        self.leaf_id = leaf_id
        self.n_trades = 0

        # RAW: what the tree says (not overridden)
        self.tree_strategy = ''     # from tree branch data

        # REGRET PROFILE: full distribution of what regret recommends
        # {action: {'count': int, 'pct': float, 'avg_pnl': float, 'total_pnl': float}}
        self.regret_profile = {}

        # APPROACH: 79D signature in bars before entry
        self.approach_79d_mean = None   # shape (N_bars, 79) or None if no data
        self.approach_length = 0        # how many bars of approach we have

        # ENTRY: 79D at entry
        self.entry_79d_mean = np.zeros(len(FEATURE_NAMES_79D))
        self.entry_79d_std = np.zeros(len(FEATURE_NAMES_79D))

        # SAME PATH: PnL curve following NMP direction
        self.same_path = []
        self.same_path_std = []

        # COUNTER PATH: PnL curve counter-trading NMP direction
        self.counter_path = []
        self.counter_path_std = []

        # EXIT: 79D at optimal exit for same and counter
        self.same_exit_bar = 0.0
        self.counter_exit_bar = 0.0
        self.same_exit_79d_mean = np.zeros(len(FEATURE_NAMES_79D))
        self.counter_exit_79d_mean = np.zeros(len(FEATURE_NAMES_79D))

        # STATS (raw, no opinions)
        self.actual_pnl = 0.0
        self.wr = 0.0
        self.avg_regret = 0.0

    def to_dict(self):
        return {
            'leaf_id': self.leaf_id,
            'n_trades': self.n_trades,
            'tree_strategy': self.tree_strategy,
            'regret_profile': self.regret_profile,
            'approach_79d_mean': self.approach_79d_mean.tolist() if self.approach_79d_mean is not None else [],
            'approach_length': self.approach_length,
            'entry_79d_mean': self.entry_79d_mean.tolist(),
            'entry_79d_std': self.entry_79d_std.tolist(),
            'same_path': self.same_path,
            'same_path_std': self.same_path_std,
            'counter_path': self.counter_path,
            'counter_path_std': self.counter_path_std,
            'same_exit_bar': self.same_exit_bar,
            'counter_exit_bar': self.counter_exit_bar,
            'same_exit_79d_mean': self.same_exit_79d_mean.tolist(),
            'counter_exit_79d_mean': self.counter_exit_79d_mean.tolist(),
            'actual_pnl': self.actual_pnl,
            'wr': self.wr,
            'avg_regret': self.avg_regret,
        }


def load_all_data():
    """Load trades (with paths + approach), regret analysis, and tree."""
    with open(TRADE_LOG, 'rb') as f:
        trades = pickle.load(f)

    regret = pd.read_csv(REGRET_FILE)

    with open(TREE_FILE, 'rb') as f:
        tree_data = pickle.load(f)

    tree = tree_data['tree']
    branches = {b['leaf_id']: b for b in tree_data['branches']}

    # Classify each trade into its branch
    for i, t in enumerate(trades):
        feat = np.array(t['entry_79d']).reshape(1, -1)
        feat = np.nan_to_num(feat)
        t['leaf_id'] = int(tree.apply(feat)[0])
        t['regret'] = regret.iloc[i].to_dict() if i < len(regret) else {}

    return trades, regret, tree_data, branches


def _build_path_from_trades(branch_trades, max_bars=30):
    """Build same and counter PnL paths from trade path data."""
    same_paths = []
    counter_paths = []

    for t in branch_trades:
        path = t.get('path', [])
        if not path:
            continue

        pnls = [p['pnl'] for p in path]
        # Counter is the mirror: if trade was long +$10, counter would be -$10
        counter_pnls = [-p for p in pnls]

        same_paths.append(pnls)
        counter_paths.append(counter_pnls)

    def aggregate_paths(paths):
        if not paths:
            return [], []
        max_len = min(max_bars, max(len(p) for p in paths))
        padded = []
        for p in paths:
            if len(p) >= max_len:
                padded.append(p[:max_len])
            else:
                padded.append(p + [p[-1]] * (max_len - len(p)))
        arr = np.array(padded)
        return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

    same_mean, same_std = aggregate_paths(same_paths)
    counter_mean, counter_std = aggregate_paths(counter_paths)
    return same_mean, same_std, counter_mean, counter_std


def _build_approach_signature(branch_trades):
    """Build approach 79D signature from pre-entry buffers."""
    # Collect approach paths that have 79D data
    approach_arrays = []
    for t in branch_trades:
        approach = t.get('approach', [])
        if not approach:
            continue
        # Extract 79D arrays from approach buffer
        approach_79d = [a['features_79d'] for a in approach if 'features_79d' in a]
        if approach_79d:
            approach_arrays.append(np.array(approach_79d))

    if not approach_arrays:
        return None, 0

    # Pad/truncate to same length (use shortest common length)
    lengths = [a.shape[0] for a in approach_arrays]
    common_len = min(lengths)  # use shortest to avoid padding artifacts
    trimmed = [a[-common_len:] for a in approach_arrays]  # take last N bars
    stacked = np.nan_to_num(np.stack(trimmed))
    return stacked.mean(axis=0), common_len


def _build_regret_profile(branch_regrets):
    """Build full regret profile: distribution of best actions with PnL."""
    profile = {}
    n = len(branch_regrets)
    if n == 0:
        return profile

    for action in REGRET_ACTIONS:
        mask = branch_regrets['best_action'] == action
        count = int(mask.sum())
        if count > 0:
            avg_pnl = float(branch_regrets.loc[mask, 'best_pnl'].mean())
            total_pnl = float(branch_regrets.loc[mask, 'best_pnl'].sum())
        else:
            avg_pnl = 0.0
            total_pnl = 0.0
        profile[action] = {
            'count': count,
            'pct': count / n if n > 0 else 0.0,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
        }
    return profile


def build_book(trades, regret_df, tree_data, branches):
    """Build the strategy book — raw data per leaf, no opinions."""
    strategies = {}

    # Group trades by branch
    by_branch = defaultdict(list)
    for t in trades:
        by_branch[t['leaf_id']].append(t)

    for lid, branch_trades in by_branch.items():
        strat = Strategy(lid)
        strat.n_trades = len(branch_trades)

        # RAW tree strategy (what the tree says, never overridden)
        branch_info = branches.get(lid, {})
        strat.tree_strategy = branch_info.get('strategy', 'unknown')

        # Get regret data for this branch
        branch_indices = [i for i, t in enumerate(trades) if t['leaf_id'] == lid]
        branch_regrets = regret_df.iloc[branch_indices] if branch_indices else pd.DataFrame()

        if len(branch_regrets) == 0:
            continue

        # REGRET PROFILE: full distribution
        strat.regret_profile = _build_regret_profile(branch_regrets)

        # APPROACH: 79D path before entry
        approach_mean, approach_len = _build_approach_signature(branch_trades)
        strat.approach_79d_mean = approach_mean
        strat.approach_length = approach_len

        # ENTRY: 79D at entry
        entry_feats = np.nan_to_num(np.array([np.array(t['entry_79d']) for t in branch_trades]))
        strat.entry_79d_mean = entry_feats.mean(axis=0)
        strat.entry_79d_std = entry_feats.std(axis=0)

        # SAME + COUNTER PATHS
        same_path, same_std, counter_path, counter_std = _build_path_from_trades(branch_trades)
        strat.same_path = same_path
        strat.same_path_std = same_std
        strat.counter_path = counter_path
        strat.counter_path_std = counter_std

        # EXIT: optimal bars from regret
        strat.same_exit_bar = float(branch_regrets['same_best_bar'].mean())
        strat.counter_exit_bar = float(branch_regrets['counter_best_bar'].mean())

        # EXIT 79D: from trade paths at optimal exit bar
        for exit_type, exit_bar_attr, exit_79d_attr in [
            ('same', 'same_exit_bar', 'same_exit_79d_mean'),
            ('counter', 'counter_exit_bar', 'counter_exit_79d_mean'),
        ]:
            opt_bar = int(getattr(strat, exit_bar_attr))
            exit_feats = []
            for t in branch_trades:
                path = t.get('path', [])
                if path and opt_bar < len(path) and 'features_79d' in path[min(opt_bar, len(path) - 1)]:
                    exit_feats.append(np.array(path[min(opt_bar, len(path) - 1)]['features_79d']))
            if exit_feats:
                setattr(strat, exit_79d_attr, np.nan_to_num(np.array(exit_feats)).mean(axis=0))

        # STATS
        strat.actual_pnl = sum(t['pnl'] for t in branch_trades)
        strat.wr = sum(1 for t in branch_trades if t['pnl'] > 0) / max(len(branch_trades), 1)
        strat.avg_regret = float(branch_regrets['regret'].mean())

        strategies[lid] = strat

    return strategies


def print_book(strategies, save_path=None):
    """Print the strategy book with regret profiles."""
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out(f'\n{"="*70}')
    out(f'STRATEGY BOOK — {len(strategies)} leaves (raw + regret profiles)')
    out(f'{"="*70}')

    total_actual = sum(s.actual_pnl for s in strategies.values())
    total_trades = sum(s.n_trades for s in strategies.values())

    out(f'  Total trades: {total_trades}')
    out(f'  Actual PnL:  ${total_actual:,.0f}')

    # Regret profile summary across all leaves
    action_totals = defaultdict(lambda: {'count': 0, 'pnl': 0.0})
    for s in strategies.values():
        for action, info in s.regret_profile.items():
            action_totals[action]['count'] += info['count']
            action_totals[action]['pnl'] += info['total_pnl']

    out(f'\n  Regret action distribution (what SHOULD have happened):')
    for action in REGRET_ACTIONS:
        info = action_totals[action]
        pct = info['count'] / max(total_trades, 1) * 100
        out(f'    {action:<22} {info["count"]:>5} ({pct:>4.0f}%)  optimal=${info["pnl"]:>9,.0f}')

    # Top 20 leaves by trade count
    sorted_strats = sorted(strategies.values(), key=lambda s: -s.n_trades)

    out(f'\n  All leaves (by trade count):')
    out(f'  {"ID":>4} {"N":>5} {"TreeStrat":<18} {"WR":>5} {"Actual$":>8} {"Regret":>7} {"TopAction":<18} {"TopPct":>5}')
    out(f'  {"-"*80}')

    for s in sorted_strats:
        # Find the action with highest total PnL in profile
        if s.regret_profile:
            top_action = max(s.regret_profile, key=lambda a: s.regret_profile[a]['total_pnl'])
            top_pct = s.regret_profile[top_action]['pct']
        else:
            top_action = '?'
            top_pct = 0
        out(f'  {s.leaf_id:>4} {s.n_trades:>5} {s.tree_strategy:<18} {s.wr:>4.0%} '
            f'${s.actual_pnl:>7.0f} ${s.avg_regret:>6.0f} {top_action:<18} {top_pct:>4.0%}')

    # Detail for top 5 by trade count
    out(f'\n  {"="*70}')
    out(f'  DETAILED PLAYBOOKS (top 5 by trade count)')
    out(f'  {"="*70}')

    for s in sorted_strats[:5]:
        out(f'\n  --- Leaf {s.leaf_id} (tree: {s.tree_strategy}) ---')
        out(f'  Trades: {s.n_trades} | WR: {s.wr:.0%} | Actual: ${s.actual_pnl:.0f} | Regret: ${s.avg_regret:.0f}/trade')

        # Regret profile
        out(f'  REGRET PROFILE:')
        for action in REGRET_ACTIONS:
            info = s.regret_profile.get(action, {'count': 0, 'pct': 0, 'avg_pnl': 0})
            if info['count'] > 0:
                bar = '#' * int(info['pct'] * 20)
                out(f'    {action:<22} {info["count"]:>4} ({info["pct"]:>4.0%}) '
                    f'avg=${info["avg_pnl"]:>6.0f}  {bar}')

        # Approach signature
        if s.approach_79d_mean is not None and s.approach_length > 0:
            out(f'  APPROACH: {s.approach_length} bars pre-entry captured')

        # Entry signature (top 5 features)
        entry_importance = sorted(
            zip(FEATURE_NAMES_79D, s.entry_79d_mean, s.entry_79d_std),
            key=lambda x: -abs(x[1])
        )
        out(f'  ENTRY signature (top features):')
        for name, mean, std in entry_importance[:7]:
            if abs(mean) > 0.01:
                out(f'    {name:<25} mean={mean:>+8.2f}  std={std:>6.2f}')

        # Same + counter paths
        if s.same_path:
            path_str = '  '.join(f'${p:>+6.1f}' for p in s.same_path[:10])
            out(f'  SAME PATH:    {path_str} ...')
            peak_bar = int(np.argmax(s.same_path))
            out(f'    Peak: ${s.same_path[peak_bar]:.1f} at bar {peak_bar} | Exit bar: {s.same_exit_bar:.0f}')
        if s.counter_path:
            path_str = '  '.join(f'${p:>+6.1f}' for p in s.counter_path[:10])
            out(f'  COUNTER PATH: {path_str} ...')
            peak_bar = int(np.argmax(s.counter_path))
            out(f'    Peak: ${s.counter_path[peak_bar]:.1f} at bar {peak_bar} | Exit bar: {s.counter_exit_bar:.0f}')

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'\nBook saved: {save_path}')


def main():
    print('Building Strategy Book (raw + regret profiles)...')

    trades, regret_df, tree_data, branches = load_all_data()
    print(f'  {len(trades)} trades, {len(regret_df)} regret records')

    strategies = build_book(trades, regret_df, tree_data, branches)
    print(f'  {len(strategies)} leaves with data')

    # Count approach data availability
    with_approach = sum(1 for s in strategies.values() if s.approach_length > 0)
    print(f'  {with_approach} leaves with approach path data')

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
