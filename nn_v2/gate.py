"""
Gate — classifies 79D into strategy branches with full calibration.

Uses the decision tree for classification + strategy book for calibration.
Returns: direction, expected path, exit conditions, optimal timing.

The gate provides the FULL playbook for each trade:
  - Which branch (tree classification)
  - Direction: same as NMP or counter
  - Expected path: PnL curve bar by bar
  - Exit bar: when to exit (optimal timing from regret)
  - Exit 79D: what the features should look like at exit
  - Path tolerance: how far actual can deviate before cutting
"""
import os
import sys
import pickle
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D


class Gate:
    """Classifies + calibrates trades from tree + book."""

    def __init__(self, tree_path: str = 'DATA/NMP_TREE/strategy_tree.pkl',
                 book_path: str = 'DATA/NMP_TREE/strategy_book.pkl'):

        with open(tree_path, 'rb') as f:
            data = pickle.load(f)
        self.tree = data['tree']
        self.branches = {b['leaf_id']: b for b in data['branches']}

        # Load book for calibration (optional — falls back to branch stats)
        self.book = {}
        if os.path.exists(book_path):
            with open(book_path, 'rb') as f:
                self.book = pickle.load(f)

        self.tradeable_leaves = set(b['leaf_id'] for b in data['branches'])
        n_calibrated = sum(1 for lid in self.tradeable_leaves if lid in self.book)
        print(f'  Gate: {len(self.tradeable_leaves)} branches, {n_calibrated} calibrated from book')

    def evaluate(self, state: Dict) -> Dict:
        """Evaluate 79D state. Returns full playbook for this bar.

        Returns:
            {
                'allowed': bool,
                'leaf_id': int,
                'branch': dict,
                'strategy': str (same_extended, counter_extended, etc.),
                'direction': str ('same' or 'counter'),
                'expected_path': list of floats (PnL per bar),
                'optimal_exit_bar': float,
                'exit_79d': list of floats (79D at expected exit),
                'entry_79d_match': float (how well current 79D matches branch entry),
                'reason': str,
            }
        """
        feat = state['features_79d']
        feat_2d = np.nan_to_num(feat.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

        leaf_id = int(self.tree.apply(feat_2d)[0])
        branch = self.branches.get(leaf_id, {})
        allowed = leaf_id in self.tradeable_leaves
        strategy = branch.get('strategy', 'same_extended')
        direction = 'counter' if 'counter' in strategy else 'same'

        # Calibration from book
        book_entry = self.book.get(leaf_id, {})
        expected_path = book_entry.get('expected_path', [])
        optimal_exit_bar = book_entry.get('optimal_exit_bar', branch.get('exit_bar_same', 16))
        exit_79d = book_entry.get('exit_79d_mean', [])
        entry_79d_mean = book_entry.get('entry_79d_mean', [])

        # How well does current 79D match this branch's entry signature?
        entry_match = 0.0
        if len(entry_79d_mean) == len(feat):
            entry_mean = np.array(entry_79d_mean)
            diff = np.abs(feat - entry_mean)
            # Normalize by std (avoid div by zero)
            entry_std = np.array(book_entry.get('entry_79d_std',
                                                np.ones(len(feat)))).clip(min=0.01)
            # Mean z-score distance
            z_dist = np.mean(diff / entry_std)
            entry_match = max(0, 1.0 - z_dist / 5.0)  # 1.0 = perfect match, 0 = far

        reason = f'{strategy} branch {leaf_id}'
        if entry_match > 0:
            reason += f' (match={entry_match:.0%})'

        return {
            'allowed': allowed,
            'leaf_id': leaf_id,
            'branch': branch,
            'strategy': strategy,
            'direction': direction,
            'expected_path': expected_path,
            'optimal_exit_bar': optimal_exit_bar,
            'exit_79d': exit_79d,
            'entry_match': entry_match,
            'reason': reason,
        }

    def should_exit(self, state: Dict, bars_held: int, pnl: float,
                    entry_decision: Dict) -> Dict:
        """Check if current bar should exit based on calibration.

        Compares actual trade progress to expected path from book.

        Args:
            state: current 79D
            bars_held: bars since entry
            pnl: current unrealized PnL
            entry_decision: the evaluate() result from entry time

        Returns:
            {'exit': bool, 'reason': str, 'divergence': float}
        """
        path = entry_decision.get('expected_path', [])
        exit_bar = entry_decision.get('optimal_exit_bar', 16)

        # Past optimal exit bar → exit
        if bars_held >= exit_bar:
            return {'exit': True, 'reason': 'optimal_bar_reached', 'divergence': 0}

        # Path divergence check
        if path and bars_held < len(path):
            expected_pnl = path[bars_held]
            divergence = abs(pnl - expected_pnl)

            # If actual PnL is worse than expected by more than 2x the expected magnitude
            expected_magnitude = max(abs(expected_pnl), 1.0)
            if pnl < expected_pnl - 2.0 * expected_magnitude:
                return {
                    'exit': True,
                    'reason': f'path_divergence (actual=${pnl:.0f} vs expected=${expected_pnl:.0f})',
                    'divergence': divergence,
                }

        # Exit 79D match check
        exit_79d = entry_decision.get('exit_79d', [])
        if exit_79d and len(exit_79d) == len(state['features_79d']):
            feat = state['features_79d']
            exit_mean = np.array(exit_79d)
            exit_dist = np.mean(np.abs(feat - exit_mean))
            # If current 79D is close to exit signature → exit
            if exit_dist < 0.5:
                return {
                    'exit': True,
                    'reason': 'exit_79d_match',
                    'divergence': exit_dist,
                }

        return {'exit': False, 'reason': 'hold', 'divergence': 0}
