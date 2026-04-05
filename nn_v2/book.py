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

TRADE_LOG = 'nn_v2/output/trades/nmp_is.pkl'
REGRET_FILE = 'nn_v2/output/tree/regret_analysis.csv'
TREE_FILE = 'nn_v2/output/tree/strategy_tree.pkl'
OUTPUT_DIR = 'nn_v2/output/tree'

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


# ============================================================
# BAYESIAN BOOK — versioned, Dirichlet-updated, human-readable
# ============================================================

PRIOR_WEIGHT = 10.0           # effective NMP sample size for Dirichlet prior
PRIOR_FLOOR = 0.5             # Laplace floor per action (prevents zero prior)
MAX_EPOCHS_PER_DAY = 5        # max retries per day in LEARN phase
MIN_PNL_IMPROVEMENT = 1.0     # $ improvement required to keep epoch result
SPOT_CHECK_DEGRADATION = -5.0 # max $ PnL drop on spot-check before warning
BOOK_DIR = 'nn_v2/output/books'


class BayesianLeaf:
    """One leaf with Dirichlet prior → evidence → posterior."""

    def __init__(self, leaf_id: int, strategy_dict: dict):
        self.leaf_id = leaf_id
        self.tree_strategy = strategy_dict.get('tree_strategy', 'unknown')

        # Dirichlet parameters per action
        self.alpha_prior = {}
        self.alpha_post = {}
        self.evidence_counts = {a: 0 for a in REGRET_ACTIONS}

        # Exit bars (precision-weighted)
        self.prior_same_exit = strategy_dict.get('same_exit_bar', 16.0)
        self.prior_counter_exit = strategy_dict.get('counter_exit_bar', 16.0)
        self.post_same_exit = self.prior_same_exit
        self.post_counter_exit = self.prior_counter_exit

        # Paths (precision-weighted, element-wise)
        self.prior_same_path = list(strategy_dict.get('same_path', []))
        self.prior_counter_path = list(strategy_dict.get('counter_path', []))
        self.post_same_path = list(self.prior_same_path)
        self.post_counter_path = list(self.prior_counter_path)

        # Entry/exit 79D (carried from base, not updated)
        self.entry_79d_mean = strategy_dict.get('entry_79d_mean', [])
        self.entry_79d_std = strategy_dict.get('entry_79d_std', [])

        # Tracking
        self.prior_n = PRIOR_WEIGHT
        self.evidence_n = 0
        self.confidence = 0.0
        self.n_trades = strategy_dict.get('n_trades', 0)
        self.actual_pnl = strategy_dict.get('actual_pnl', 0.0)
        self.wr = strategy_dict.get('wr', 0.0)

        # Initialize Dirichlet prior from NMP regret profile
        self._init_prior(strategy_dict.get('regret_profile', {}))

    def _init_prior(self, regret_profile: dict):
        """Set Dirichlet prior from NMP regret counts."""
        total = sum(info.get('count', 0) for info in regret_profile.values())
        total = max(total, 1)

        for action in REGRET_ACTIONS:
            info = regret_profile.get(action, {})
            count = info.get('count', 0)
            self.alpha_prior[action] = PRIOR_WEIGHT * (count / total) + PRIOR_FLOOR
            self.alpha_post[action] = self.alpha_prior[action]

    def update(self, day_regrets: pd.DataFrame, day_trades: list) -> dict:
        """Bayesian update from one day's evidence. Returns changelog."""
        n_new = len(day_regrets)
        if n_new == 0:
            return {'leaf_id': self.leaf_id, 'n_evidence': 0, 'changed': False}

        prev_profile = self._get_profile()
        prev_same_exit = self.post_same_exit
        prev_counter_exit = self.post_counter_exit

        # 1. Dirichlet update: count best_action distribution
        day_counts = {a: 0 for a in REGRET_ACTIONS}
        for _, row in day_regrets.iterrows():
            action = row.get('best_action', '')
            if action in day_counts:
                day_counts[action] += 1

        for action in REGRET_ACTIONS:
            self.evidence_counts[action] += day_counts[action]
            self.alpha_post[action] = self.alpha_prior[action] + self.evidence_counts[action]

        # 2. Exit bar update (precision-weighted)
        self.evidence_n += n_new
        total_n = self.prior_n + self.evidence_n

        if 'same_best_bar' in day_regrets.columns:
            day_same_exit = float(day_regrets['same_best_bar'].mean())
            self.post_same_exit = (self.prior_n * self.prior_same_exit +
                                   self.evidence_n * day_same_exit) / total_n

        if 'counter_best_bar' in day_regrets.columns:
            day_counter_exit = float(day_regrets['counter_best_bar'].mean())
            self.post_counter_exit = (self.prior_n * self.prior_counter_exit +
                                      self.evidence_n * day_counter_exit) / total_n

        # 3. Path update (precision-weighted, element-wise)
        for t in day_trades:
            path = t.get('path', [])
            if path:
                pnls = [p['pnl'] for p in path]
                counter_pnls = [-p for p in pnls]
                self._blend_path('same', pnls, total_n)
                self._blend_path('counter', counter_pnls, total_n)

        # 4. Confidence
        self.confidence = self.evidence_n / (self.evidence_n + self.prior_n)

        # Build changelog
        new_profile = self._get_profile()
        return {
            'leaf_id': self.leaf_id,
            'n_evidence': n_new,
            'changed': True,
            'confidence': self.confidence,
            'same_exit_delta': self.post_same_exit - prev_same_exit,
            'counter_exit_delta': self.post_counter_exit - prev_counter_exit,
            'profile_deltas': {a: new_profile[a] - prev_profile[a] for a in REGRET_ACTIONS},
        }

    def _blend_path(self, direction: str, new_pnls: list, total_n: float):
        """Blend new path data into posterior path."""
        if direction == 'same':
            prior_path = self.post_same_path
        else:
            prior_path = self.post_counter_path

        if not prior_path:
            if direction == 'same':
                self.post_same_path = new_pnls[:30]
            else:
                self.post_counter_path = new_pnls[:30]
            return

        max_len = min(30, max(len(prior_path), len(new_pnls)))
        result = []
        for i in range(max_len):
            prior_val = prior_path[i] if i < len(prior_path) else prior_path[-1]
            new_val = new_pnls[i] if i < len(new_pnls) else new_pnls[-1]
            blended = (self.prior_n * prior_val + self.evidence_n * new_val) / total_n
            result.append(blended)

        if direction == 'same':
            self.post_same_path = result
        else:
            self.post_counter_path = result

    def _get_profile(self) -> dict:
        """Get current posterior profile as {action: probability}."""
        total = sum(self.alpha_post.values())
        return {a: self.alpha_post[a] / max(total, 1e-10) for a in REGRET_ACTIONS}

    def snapshot(self) -> dict:
        """Deep copy for revert capability."""
        return {
            'alpha_post': dict(self.alpha_post),
            'evidence_counts': dict(self.evidence_counts),
            'evidence_n': self.evidence_n,
            'post_same_exit': self.post_same_exit,
            'post_counter_exit': self.post_counter_exit,
            'post_same_path': list(self.post_same_path),
            'post_counter_path': list(self.post_counter_path),
            'confidence': self.confidence,
        }

    def revert(self, snap: dict):
        """Revert to a previous snapshot."""
        self.alpha_post = dict(snap['alpha_post'])
        self.evidence_counts = dict(snap['evidence_counts'])
        self.evidence_n = snap['evidence_n']
        self.post_same_exit = snap['post_same_exit']
        self.post_counter_exit = snap['post_counter_exit']
        self.post_same_path = list(snap['post_same_path'])
        self.post_counter_path = list(snap['post_counter_path'])
        self.confidence = snap['confidence']

    def to_book_entry(self) -> dict:
        """Export as Gate-compatible dict."""
        profile = self._get_profile()

        # Gate reads 'expected_path' and 'optimal_exit_bar' — pick based on
        # which direction has higher posterior probability
        same_weight = sum(profile.get(a, 0) for a in REGRET_ACTIONS if 'counter' not in a)
        counter_weight = sum(profile.get(a, 0) for a in REGRET_ACTIONS if 'counter' in a)

        if counter_weight > same_weight:
            expected_path = self.post_counter_path
            optimal_exit_bar = self.post_counter_exit
        else:
            expected_path = self.post_same_path
            optimal_exit_bar = self.post_same_exit

        return {
            'leaf_id': self.leaf_id,
            'tree_strategy': self.tree_strategy,
            'n_trades': self.n_trades,
            'regret_profile': {a: {'pct': profile[a]} for a in REGRET_ACTIONS},
            'entry_79d_mean': self.entry_79d_mean,
            'entry_79d_std': self.entry_79d_std,
            'expected_path': expected_path,
            'optimal_exit_bar': optimal_exit_bar,
            'same_path': self.post_same_path,
            'counter_path': self.post_counter_path,
            'same_exit_bar': self.post_same_exit,
            'counter_exit_bar': self.post_counter_exit,
            'confidence': self.confidence,
            'actual_pnl': self.actual_pnl,
            'wr': self.wr,
        }

    def diff(self, prev_snap: dict) -> str:
        """Human-readable diff vs a previous snapshot."""
        profile = self._get_profile()
        prev_total = sum(prev_snap['alpha_post'].values())
        prev_profile = {a: prev_snap['alpha_post'][a] / max(prev_total, 1e-10) for a in REGRET_ACTIONS}

        parts = [f'L{self.leaf_id}:']
        # Exit bar changes
        se_delta = self.post_same_exit - prev_snap['post_same_exit']
        ce_delta = self.post_counter_exit - prev_snap['post_counter_exit']
        if abs(se_delta) > 0.1:
            parts.append(f'same_exit {prev_snap["post_same_exit"]:.1f}→{self.post_same_exit:.1f}')
        if abs(ce_delta) > 0.1:
            parts.append(f'counter_exit {prev_snap["post_counter_exit"]:.1f}→{self.post_counter_exit:.1f}')
        # Profile changes > 2pp
        for a in REGRET_ACTIONS:
            delta = profile[a] - prev_profile[a]
            if abs(delta) > 0.02:
                parts.append(f'{a} {prev_profile[a]:.0%}→{profile[a]:.0%}')
        parts.append(f'conf={self.confidence:.0%}')
        return ' | '.join(parts)


class VersionedBook:
    """Versioned, Bayesian-updating strategy book."""

    def __init__(self):
        self.leaves = {}         # {leaf_id: BayesianLeaf}
        self.version = 0
        self.changelog = []      # list of {version, day, n_changed, changes}
        self._evolution = []     # for CSV export

    @classmethod
    def from_nmp_book(cls, book_pkl_path: str) -> 'VersionedBook':
        """Create v0 from existing NMP book pkl."""
        vb = cls()
        with open(book_pkl_path, 'rb') as f:
            book_data = pickle.load(f)

        for lid, strat_dict in book_data.items():
            vb.leaves[lid] = BayesianLeaf(lid, strat_dict)

        vb.version = 0
        vb._record_evolution('baseline')
        print(f'  VersionedBook v0: {len(vb.leaves)} leaves from {book_pkl_path}')
        return vb

    def bayesian_update(self, leaf_id: int, day_regrets: pd.DataFrame,
                        day_trades: list) -> dict:
        """Update one leaf. Returns changelog dict."""
        if leaf_id not in self.leaves:
            return {'leaf_id': leaf_id, 'changed': False}
        return self.leaves[leaf_id].update(day_regrets, day_trades)

    def snapshot_all(self) -> dict:
        """Snapshot all leaves for revert."""
        return {lid: leaf.snapshot() for lid, leaf in self.leaves.items()}

    def revert_all(self, snapshots: dict):
        """Revert all leaves to snapshots."""
        for lid, snap in snapshots.items():
            if lid in self.leaves:
                self.leaves[lid].revert(snap)

    def freeze(self, day_name: str = ''):
        """Freeze current version, save immutable snapshot, increment."""
        os.makedirs(BOOK_DIR, exist_ok=True)

        # Save pkl
        pkl_path = os.path.join(BOOK_DIR, f'book_v{self.version:03d}.pkl')
        book_data = self.export_for_gate()
        with open(pkl_path, 'wb') as f:
            pickle.dump(book_data, f)

        # Save human-readable
        txt_path = os.path.join(BOOK_DIR, f'book_v{self.version:03d}.txt')
        self.print_version(save_path=txt_path)

        self._record_evolution(day_name)
        self.version += 1

    def export_for_gate(self) -> dict:
        """Export as dict compatible with Gate's book format."""
        return {lid: leaf.to_book_entry() for lid, leaf in self.leaves.items()}

    def print_version(self, save_path: str = None):
        """Human-readable: prior → evidence → posterior per leaf."""
        lines = []
        lines.append(f'BAYESIAN BOOK v{self.version} — {len(self.leaves)} leaves')
        lines.append(f'=' * 60)

        for lid in sorted(self.leaves.keys()):
            leaf = self.leaves[lid]
            profile = leaf._get_profile()
            lines.append(f'\nLeaf {lid} (tree: {leaf.tree_strategy}) — '
                         f'confidence: {leaf.confidence:.0%} | '
                         f'evidence: {leaf.evidence_n:.0f} trades')

            # Prior vs posterior
            lines.append(f'  REGRET PROFILE:')
            for a in REGRET_ACTIONS:
                prior_total = sum(leaf.alpha_prior.values())
                prior_p = leaf.alpha_prior[a] / max(prior_total, 1e-10)
                post_p = profile[a]
                delta = post_p - prior_p
                arrow = '→' if abs(delta) > 0.01 else '='
                bar = '#' * int(post_p * 20)
                lines.append(f'    {a:<22} prior={prior_p:>4.0%} {arrow} '
                             f'post={post_p:>4.0%} ({delta:>+.0%})  {bar}')

            # Exit bars
            lines.append(f'  EXIT BARS: same={leaf.post_same_exit:.1f} '
                         f'(prior={leaf.prior_same_exit:.1f}) | '
                         f'counter={leaf.post_counter_exit:.1f} '
                         f'(prior={leaf.prior_counter_exit:.1f})')

        output = '\n'.join(lines)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(output)
        return output

    def print_diff(self, prev_snapshots: dict, save_path: str = None):
        """Changelog between current state and previous snapshots."""
        lines = [f'DIFF: → v{self.version}', '-' * 40]
        changed = 0
        for lid in sorted(self.leaves.keys()):
            if lid in prev_snapshots:
                diff_str = self.leaves[lid].diff(prev_snapshots[lid])
                if '→' in diff_str:
                    lines.append(f'  {diff_str}')
                    changed += 1
        lines.insert(1, f'Leaves changed: {changed}/{len(self.leaves)}')

        output = '\n'.join(lines)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(output)
        return output

    def _record_evolution(self, day_name: str):
        """Record current state for evolution CSV."""
        for lid, leaf in self.leaves.items():
            profile = leaf._get_profile()
            row = {
                'leaf_id': lid,
                'version': self.version,
                'day': day_name,
                'confidence': leaf.confidence,
                'evidence_n': leaf.evidence_n,
                'same_exit_bar': leaf.post_same_exit,
                'counter_exit_bar': leaf.post_counter_exit,
            }
            for a in REGRET_ACTIONS:
                row[f'{a}_pct'] = profile[a]
            self._evolution.append(row)

    def evolution_csv(self, save_path: str):
        """Save evolution as CSV for EDA."""
        if self._evolution:
            pd.DataFrame(self._evolution).to_csv(save_path, index=False)
            print(f'Evolution CSV: {save_path} ({len(self._evolution)} rows)')

    def save(self, path: str = None):
        """Save current state (for resume)."""
        if path is None:
            path = os.path.join(BOOK_DIR, 'book_latest.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'VersionedBook':
        """Load saved state."""
        with open(path, 'rb') as f:
            return pickle.load(f)


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
