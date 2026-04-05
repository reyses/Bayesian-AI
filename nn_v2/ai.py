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

    ai = AIEngine('DATA/NMP_TREE/strategy_tree.pkl')
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

# NMP entry threshold (only enter if z_se is extreme enough)
ROCHE = 2.0
VR_ENTRY = 1.0

# 79D offsets
_1M_OFFSET = 10  # 1m is TF index 1, 10 features per TF


class AIEngine:
    """Continuous positioning: LONG / SHORT / FLAT at every bar."""

    def __init__(self, tree_path: str = 'DATA/NMP_TREE/strategy_tree.pkl',
                 book_path: str = 'DATA/NMP_TREE/strategy_book.pkl'):
        from nn_v2.gate import Gate
        self.gate = Gate(tree_path, book_path)

        # Current position
        self.position = 'flat'  # 'long', 'short', 'flat'
        self.entry_price = 0.0
        self.entry_bar = 0
        self.entry_branch = None
        self.bars_held = 0
        self.peak_pnl = 0.0

        # Trade log
        self.trades = []
        self.daily_pnl = 0.0
        self._bar_count = 0
        self._last_price = 0.0

    def on_state(self, state: Dict) -> Dict:
        """Process one bar. Returns action taken.

        Args:
            state: {'features_79d': np.array(79), 'price': float, 'timestamp': float}

        Returns:
            {
                'position': 'long'/'short'/'flat',
                'changed': bool,
                'action': 'hold'/'enter_long'/'enter_short'/'flip_to_long'/'flip_to_short'/'exit',
                'branch': dict or None,
                'pnl': current unrealized PnL,
            }
        """
        self._bar_count += 1
        feat = state['features_79d']
        price = state['price']
        ts = state['timestamp']
        self._last_price = price

        # Read 1m state
        z = feat[_1M_OFFSET + 0]
        vr = feat[_1M_OFFSET + 2]

        # Gate classifies + calibrates
        decision = self.gate.evaluate(state)
        strategy = decision['strategy']
        direction = decision['direction']
        leaf_id = decision['leaf_id']
        branch = decision['branch']

        # Determine desired position
        desired = self._get_desired_position(z, vr, strategy)

        # Current unrealized PnL
        if self.position == 'long':
            unrealized = (price - self.entry_price) / TICK * TV
        elif self.position == 'short':
            unrealized = (self.entry_price - price) / TICK * TV
        else:
            unrealized = 0.0

        self.peak_pnl = max(self.peak_pnl, unrealized)
        self.bars_held += 1 if self.position != 'flat' else 0

        # Decide action
        action = 'hold'
        changed = False

        # === CALIBRATED EXIT CHECK (when in position) ===
        if self.position != 'flat' and self._entry_decision is not None:
            exit_check = self.gate.should_exit(state, self.bars_held, unrealized,
                                                self._entry_decision)
            if exit_check['exit']:
                # Book says exit — check if we should flip or go flat
                if desired != 'flat' and desired != self.position:
                    action = f'flip_to_{desired}'
                    self._exit(price, ts, exit_check['reason'])
                    self._enter(desired, price, ts, leaf_id, branch)
                    self._entry_decision = decision
                else:
                    action = 'exit'
                    self._exit(price, ts, exit_check['reason'])
                changed = True

        # === ENTRY / FLIP (when flat or signal changed) ===
        if not changed:
            if self.position == 'flat' and desired != 'flat':
                action = f'enter_{desired}'
                self._enter(desired, price, ts, leaf_id, branch)
                self._entry_decision = decision
                changed = True

            elif self.position != 'flat' and desired == 'flat':
                action = 'exit'
                self._exit(price, ts, 'signal_flat')
                changed = True

            elif self.position != 'flat' and desired != 'flat' and self.position != desired:
                action = f'flip_to_{desired}'
                self._exit(price, ts, 'flip')
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
        }

    def _get_desired_position(self, z: float, vr: float, strategy: str) -> str:
        """Determine desired position from 79D state + tree strategy."""
        # Must have NMP-level signal to trade
        if abs(z) < ROCHE or vr >= VR_ENTRY:
            # No extreme z → stay flat (or hold current if already in)
            if self.position != 'flat' and self.bars_held < 3:
                return self.position  # don't exit too fast
            return 'flat' if self.position == 'flat' else self.position

        # NMP direction: fade the z
        nmp_dir = 'short' if z > 0 else 'long'

        # Tree strategy modifies direction
        if 'counter' in strategy:
            # Counter: flip NMP direction
            return 'long' if nmp_dir == 'short' else 'short'
        else:
            # Same: follow NMP
            return nmp_dir

    def _enter(self, direction: str, price: float, ts: float,
               leaf_id: int, branch: dict):
        """Enter a position."""
        self.position = direction
        self.entry_price = price
        self.entry_bar = self._bar_count
        self.entry_branch = branch
        self.bars_held = 0
        self.peak_pnl = 0.0

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

        self.trades.append({
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

    def summary(self) -> str:
        n = len(self.trades)
        if n == 0:
            return 'AI: 0 trades'
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = sum(t['pnl'] for t in self.trades)
        flips = sum(1 for t in self.trades if t['exit'] == 'flip')
        return (f'AI: {n} trades | WR={wins/n*100:.0f}% | ${total:.0f} | '
                f'{flips} flips')
