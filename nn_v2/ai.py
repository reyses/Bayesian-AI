"""
AI Engine — continuous positioning system using tree strategies.

NOT trade-by-trade. Decides at EVERY bar: should I be LONG, SHORT, or FLAT?
When the signal changes → flip position (2 contracts: close + open).

Reads 79D from SFE (live or test). Consults tree for strategy.
The tree's branch determines: direction + expected path.
If next bar's branch disagrees → reposition.

Usage (test mode):
    from nn_v2.ai import AIEngine
    from nn_v2.sfe_ticker import FeatureTicker

    ai = AIEngine('nn_v2/output/tree/strategy_tree.pkl')
    for state in FeatureTicker(feat_file, price_file):
        action = ai.on_state(state)
        # action = {'position': 'long'/'short'/'flat', 'changed': bool, ...}
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

TICK = 0.25
TV = 0.50

# Minimum confidence from book to trade a leaf (skip low-evidence leaves)
MIN_LEAF_CONFIDENCE = 0.0  # 0 = trade all leaves the tree assigns


class AIEngine:
    """Continuous positioning: LONG / SHORT / FLAT at every bar."""

    def __init__(self, tree_path: str = 'nn_v2/output/tree/strategy_tree.pkl',
                 book_path: str = 'nn_v2/output/tree/strategy_book.pkl'):
        from nn_v2.gate import Gate
        self.gate = Gate(tree_path, book_path)

        # Current position
        self.position = 'flat'  # 'long', 'short', 'flat'
        self.entry_price = 0.0
        self.entry_bar = 0
        self.entry_branch = None
        self.bars_held = 0
        self.peak_pnl = 0.0

        # Chain tracking: branch updates within a trade
        self._entry_decision = None
        self._chain_history = []  # list of {bar, from_branch, to_branch, pnl_at_chain}

        # Full trade recording (same format as NMP for regret compatibility)
        self._entry_79d = None
        self._trade_path = []
        self._approach_buffer = []  # rolling 10-bar buffer when flat

        # Trade log
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0
        self._last_feat = None

    def on_state(self, state: Dict) -> Dict:
        """Process one bar. Returns action taken.

        Chain logic:
          - Same direction signal while in position: STAY + update expected path
          - Different direction signal: EXIT current + ENTER opposite (flip)
          - Calibrated exit from book: exit or flip if new signal ready

        Args:
            state: {'features_79d': np.array(79), 'price': float, 'timestamp': float}

        Returns:
            {
                'position': 'long'/'short'/'flat',
                'changed': bool,
                'action': 'hold'/'enter_long'/'enter_short'/'flip_to_long'/'flip_to_short'/'exit'/'chain_update',
                'branch': dict or None,
                'pnl': current unrealized PnL,
            }
        """
        self._bar_count += 1
        feat = state['features_79d']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price
        self._last_feat = feat

        # Approach buffer: record when flat for pre-entry context
        APPROACH_SIZE = 10
        if self.position == 'flat':
            self._approach_buffer.append({
                'timestamp': ts, 'price': price,
                'features_79d': feat.copy(),
            })
            if len(self._approach_buffer) > APPROACH_SIZE:
                self._approach_buffer = self._approach_buffer[-APPROACH_SIZE:]

        # Gate classifies + calibrates (tree + book)
        decision = self.gate.evaluate(state)
        strategy = decision['strategy']
        leaf_id = decision['leaf_id']
        branch = decision['branch']

        # Determine desired position from tree classification
        desired = self._get_desired_position(decision)

        # Current unrealized PnL
        if self.position == 'long':
            unrealized = (price - self.entry_price) / TICK * TV
        elif self.position == 'short':
            unrealized = (self.entry_price - price) / TICK * TV
        else:
            unrealized = 0.0

        self.peak_pnl = max(self.peak_pnl, unrealized)
        self.bars_held += 1 if self.position != 'flat' else 0

        # Trade path: record 79D at every bar while in position
        if self.position != 'flat':
            self._trade_path.append({
                'bar': self.bars_held, 'timestamp': ts, 'price': price,
                'pnl': unrealized, 'peak_pnl': self.peak_pnl,
                'features_79d': feat.copy(),
            })

        action = 'hold'
        changed = False

        # === WHEN IN POSITION ===
        if self.position != 'flat':

            # 1. Calibrated exit check (per-branch from book)
            if self._entry_decision is not None:
                exit_check = self.gate.should_exit(state, self.bars_held, unrealized,
                                                    self._entry_decision)
                if exit_check['exit']:
                    if desired != 'flat' and desired != self.position:
                        # Exit + flip
                        action = f'flip_to_{desired}'
                        self._exit(price, ts, exit_check['reason'])
                        self._enter(desired, price, ts, leaf_id, branch)
                        self._entry_decision = decision
                    else:
                        action = 'exit'
                        self._exit(price, ts, exit_check['reason'])
                    changed = True

            # 2. Same direction signal: STAY + update path (chain)
            if not changed and desired == self.position:
                prev_leaf = self._entry_decision['leaf_id'] if self._entry_decision else -1
                if leaf_id != prev_leaf:
                    # New branch confirms same direction — chain update
                    self._chain_history.append({
                        'bar': self.bars_held,
                        'from_branch': prev_leaf,
                        'to_branch': leaf_id,
                        'pnl_at_chain': unrealized,
                    })
                    self._entry_decision = decision
                    action = 'chain_update'
                    changed = True

            # 3. Different direction signal: flip
            if not changed and desired != 'flat' and desired != self.position:
                action = f'flip_to_{desired}'
                self._exit(price, ts, 'flip')
                self._enter(desired, price, ts, leaf_id, branch)
                self._entry_decision = decision
                changed = True

            # 4. Signal gone flat
            if not changed and desired == 'flat':
                action = 'exit'
                self._exit(price, ts, 'signal_flat')
                changed = True

        # === WHEN FLAT ===
        elif desired != 'flat':
            action = f'enter_{desired}'
            self._enter(desired, price, ts, leaf_id, branch)
            self._entry_decision = decision
            changed = True

        return {
            'position': self.position,
            'changed': changed,
            'action': action,
            'branch': branch,
            'leaf_id': leaf_id,
            'strategy': strategy,
            'desired': desired,
            'pnl': unrealized,
            'bars_held': self.bars_held,
            'entry_match': decision.get('entry_match', 0),
            'chain_length': len(self._chain_history),
        }

    def _get_desired_position(self, decision: dict) -> str:
        """Determine desired position from tree + book classification.

        The tree already evaluated the full 79D. Its leaf assignment IS the
        entry signal. The strategy label contains the direction.
        No hardcoded z/vr gate — the tree's splits encode those conditions.
        """
        strategy = decision.get('strategy', '')
        allowed = decision.get('allowed', False)

        # Tree says this leaf is not tradeable
        if not allowed:
            return 'flat' if self.position == 'flat' else self.position

        # Skip low-confidence leaves (if book provides confidence)
        confidence = decision.get('branch', {}).get('confidence', 1.0)
        if confidence < MIN_LEAF_CONFIDENCE:
            return 'flat' if self.position == 'flat' else self.position

        # Direction from corrected trade labels (long_fast, short_extended, etc.)
        if strategy.startswith('long'):
            return 'long'
        elif strategy.startswith('short'):
            return 'short'

        # Fallback for unknown strategy labels
        return 'flat' if self.position == 'flat' else self.position

    def _enter(self, direction: str, price: float, ts: float,
               leaf_id: int, branch: dict):
        """Enter a position."""
        self.position = direction
        self.entry_price = price
        self.entry_bar = self._bar_count
        self.entry_branch = branch
        self.bars_held = 0
        self.peak_pnl = 0.0
        self._chain_history = []
        self._entry_79d = self._last_feat.copy() if self._last_feat is not None else None
        self._entry_approach = list(self._approach_buffer)
        self._trade_path = []

    def _exit(self, price: float, ts: float, reason: str):
        """Exit current position, log trade."""
        if self.position == 'flat':
            return

        if self.position == 'long':
            pnl = (price - self.entry_price) / TICK * TV
        else:
            pnl = (self.entry_price - price) / TICK * TV

        self.daily_pnl += pnl
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')
        feat = self._last_feat if self._last_feat is not None else np.zeros(len(FEATURE_NAMES_79D))

        self.trades.append({
            'trade_id': len(self.trades),
            'time': time_str,
            'timestamp': ts,
            'dir': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'held': self.bars_held,
            'peak': self.peak_pnl,
            'exit': reason,
            'branch': self.entry_branch.get('leaf_id', -1) if self.entry_branch else -1,
            'leaf_id': self.entry_branch.get('leaf_id', -1) if self.entry_branch else -1,
            'chain': list(self._chain_history),
            'chain_length': len(self._chain_history),
            # Full data for regret analysis (NMP-compatible)
            'entry_79d': self._entry_79d.tolist() if self._entry_79d is not None else [],
            'exit_79d': feat.tolist(),
            'approach': self._entry_approach if hasattr(self, '_entry_approach') else [],
            'approach_length': len(self._entry_approach) if hasattr(self, '_entry_approach') else 0,
            'path': self._trade_path.copy(),
            'path_length': len(self._trade_path),
        })

        self.position = 'flat'
        self.entry_price = 0.0
        self.bars_held = 0
        self.peak_pnl = 0.0

    def force_close(self):
        """End of day — close any open position."""
        if self.position != 'flat':
            self._exit(self._last_price, 0, 'end_of_day')

    def reset(self):
        """Reset for next day."""
        self.position = 'flat'
        self.entry_price = 0.0
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self.bars_held = 0
        self.peak_pnl = 0.0
        self._entry_decision = None
        self._chain_history = []
        self._entry_79d = None
        self._trade_path = []
        self._approach_buffer = []
        self._entry_approach = []
        self._last_feat = None

    def set_book(self, book_dict: dict):
        """Hot-swap book into gate without reloading tree."""
        self.gate.set_book(book_dict)

    def get_full_trades(self) -> list:
        """Get trades with full 79D data (for regret analysis)."""
        return self.trades

    def summary(self) -> str:
        n = len(self.trades)
        if n == 0:
            return 'AI: 0 trades'
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = sum(t['pnl'] for t in self.trades)
        flips = sum(1 for t in self.trades if t['exit'] == 'flip')
        chained = sum(1 for t in self.trades if t.get('chain_length', 0) > 0)
        return (f'AI: {n} trades | WR={wins/n*100:.0f}% | ${total:.0f} | '
                f'{flips} flips | {chained} chained')
