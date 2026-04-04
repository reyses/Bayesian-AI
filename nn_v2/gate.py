"""
Gate — sits between SFE and NMP. Decides if NMP is allowed to trade.

Receives 79D from SFE. Runs it through the decision tree.
If the tree says TRADE (tradeable branch) → pass to NMP.
If the tree says SKIP → block NMP from entering.

The gate does NOT modify NMP decisions. It only allows or blocks entry.
Once NMP is in a trade, the gate does not interfere with exits.

Usage:
    from nn_v2.gate import Gate
    from nn_v2.nightmare import NightmareEngine

    gate = Gate('DATA/NMP_TREE/tree.pkl')
    nmp = NightmareEngine()

    for state in sfe_ticker:
        decision = gate.evaluate(state)
        if decision['allowed']:
            nmp.on_state(state)
        elif nmp.in_pos:
            # Already in a trade — let NMP manage the exit
            nmp.on_state(state)
"""
import os
import sys
import pickle
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D


class Gate:
    """Decides if NMP is allowed to trade based on decision tree."""

    def __init__(self, tree_path: str = 'DATA/NMP_TREE/tree.pkl'):
        with open(tree_path, 'rb') as f:
            data = pickle.load(f)

        self.tree = data['tree']
        self.branches = {b['leaf_id']: b for b in data['branches']}
        self.feature_names = data['feature_names']

        # Build set of tradeable leaf IDs
        self.tradeable_leaves = set(
            b['leaf_id'] for b in data['branches'] if b['tradeable']
        )

        n_trade = sum(1 for b in data['branches'] if b['tradeable'])
        n_skip = sum(1 for b in data['branches'] if not b['tradeable'])
        print(f'  Gate loaded: {n_trade} TRADE branches, {n_skip} SKIP branches')

    def evaluate(self, state: Dict) -> Dict:
        """Evaluate a 79D state. Returns trade/skip decision with context.

        Args:
            state: dict with 'features_79d' (np.ndarray of 79D)

        Returns:
            {
                'allowed': bool,
                'leaf_id': int,
                'branch': dict (branch stats) or None,
                'reason': str,
            }
        """
        feat = state['features_79d']
        feat_2d = np.nan_to_num(feat.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

        leaf_id = int(self.tree.apply(feat_2d)[0])
        branch = self.branches.get(leaf_id)
        allowed = leaf_id in self.tradeable_leaves

        if allowed:
            reason = f'TRADE branch {leaf_id} (WR={branch["wr"]:.0%}, EV=${branch["ev"]:.1f})'
        else:
            wr = branch['wr'] if branch else 0
            reason = f'SKIP branch {leaf_id} (WR={wr:.0%})'

        return {
            'allowed': allowed,
            'leaf_id': leaf_id,
            'branch': branch,
            'reason': reason,
        }
