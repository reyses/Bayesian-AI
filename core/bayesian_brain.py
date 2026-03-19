"""
Bayesian-AI - Bayesian Probability Engine
HashMap-based learning system: state_key -> WinRate
"""
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any
from core.market_state import MarketState

@dataclass
class TradeOutcome:
    """Single trade result for learning"""
    state: Union[MarketState, str, int]
    entry_price: float
    exit_price: float
    pnl: float
    result: str  # 'WIN' or 'LOSS'
    timestamp: float
    exit_reason: str  # 'trail_stop', 'structure_break', 'time_exit'
    entry_time: float = 0.0
    exit_time: float = 0.0
    duration: float = 0.0
    direction: str = 'LONG'  # 'LONG' or 'SHORT'
    template_id: Optional[Union[str, int]] = None

class BayesianBrain:
    """
    Probability table that learns from outcomes
    Core logic: probability_table[state_key] = {'wins': X, 'losses': Y}
    """
    def __init__(self):
        self.table: Dict[Any, Dict[str, int]] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0}
        )
        # Direction-aware table: (tid, 'LONG'|'SHORT') -> {wins, losses, total}
        self.dir_table: Dict[tuple, Dict[str, int]] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0}
        )
        self.trade_history = []
        # H0/H1 direction bias: tid -> {long_w, long_l, short_w, short_l}
        self.dir_bias: Dict[Any, Dict[str, int]] = {}
        self._frozen = False  # when True, update() is a no-op

    def freeze(self):
        """Disable learning. Brain state is read-only (for OOS validation)."""
        self._frozen = True

    def unfreeze(self):
        """Re-enable learning."""
        self._frozen = False

    def update(self, outcome: TradeOutcome):
        """
        Bayesian update after trade completion
        Args:
            outcome: TradeOutcome with state vector and result
        """
        if self._frozen:
            return  # frozen brain  -- no learning (OOS validation mode)
        # Prefer template_id if available, otherwise use state as key
        key = outcome.template_id if outcome.template_id is not None else outcome.state
        
        # Update counts
        if outcome.result == 'WIN':
            self.table[key]['wins'] += 1
        else:
            self.table[key]['losses'] += 1
        
        self.table[key]['total'] += 1

        # Direction-aware tracking
        if outcome.direction in ('LONG', 'SHORT'):
            dir_key = (key, outcome.direction)
            if outcome.result == 'WIN':
                self.dir_table[dir_key]['wins'] += 1
            else:
                self.dir_table[dir_key]['losses'] += 1
            self.dir_table[dir_key]['total'] += 1

        # Log trade
        self.trade_history.append(outcome)

    def direction_learn(self, tid, side: str, pnl: float,
                        tick_value: float = 0.50):
        """H0/H1 direction learning from actual outcomes.

        H0 (actual): we went this side  -- record 1 win or 1 loss.
        H1 (counterfactual): opposite side gets the inverse signal,
            but with count=1 (we don't know the actual counterfactual PnL).

        Uses simple counts so one big loss doesn't permanently poison the table.
        PnL tracking is separate (for expected profit computation only).

        Args:
            tid: template ID (PP_ prefix stripped automatically)
            side: 'long' or 'short'
            pnl: realized PnL in dollars
            tick_value: dollar value per tick (MNQ = 0.50)
        Returns:
            dict with current bias for this tid
        """
        if tid is None:
            return None
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid

        if base_tid not in self.dir_bias:
            self.dir_bias[base_tid] = {
                'long_w': 0, 'long_l': 0, 'short_w': 0, 'short_l': 0,
                'long_pnl': 0.0, 'long_n': 0, 'short_pnl': 0.0, 'short_n': 0}

        bias = self.dir_bias[base_tid]
        key = side.lower()
        alt_key = 'short' if key == 'long' else 'long'

        # H0: actual outcome (simple count  -- 1 win or 1 loss)
        if pnl > 0:
            bias[f'{key}_w'] += 1
        else:
            bias[f'{key}_l'] += 1

        # H1: counterfactual (opposite side gets inverse signal, count=1)
        if pnl > 0:
            # We won going this side -> opposite side would have lost
            bias[f'{alt_key}_l'] += 1
        else:
            # We lost going this side -> opposite side would have won
            bias[f'{alt_key}_w'] += 1

        # Dollar PnL tracking (actual only  -- no counterfactual PnL assumed)
        bias[f'{key}_pnl'] = bias.get(f'{key}_pnl', 0.0) + pnl
        bias[f'{key}_n'] = bias.get(f'{key}_n', 0) + 1

        return bias

    def record_hold_bars(self, tid, side: str, hold_bars: int):
        """Track actual hold time per template×direction for time anchor calibration."""
        if tid is None:
            return
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid
        if base_tid not in self.dir_bias:
            return
        bias = self.dir_bias[base_tid]
        key = side.lower()
        bias[f'{key}_hold_sum'] = bias.get(f'{key}_hold_sum', 0) + hold_bars
        bias[f'{key}_hold_n'] = bias.get(f'{key}_hold_n', 0) + 1

    def get_dir_bias(self, tid) -> dict | None:
        """Get direction bias for a template ID."""
        base_tid = tid[3:] if isinstance(tid, str) and tid.startswith('PP_') else tid
        return self.dir_bias.get(base_tid)

    def get_probability(self, state: Any) -> float:
        """
        Get win probability for given state
        Returns:
            float: Win probability (0.0 to 1.0)
        """
        if state not in self.table:
            return 0.09  # Pessimistic prior Beta(1, 10) -> 1/11 ~ 9%
        
        data = self.table[state]
        
        # Bayesian estimate with Pessimistic Prior Beta(1, 10)
        # This assumes most strategies fail, requiring proof to rise above 50%
        wins = data['wins'] + 1
        total = data['total'] + 11  # alpha=1 + beta=10 = 11
        
        return wins / total
    
    def get_confidence(self, state: Any) -> float:
        """
        How confident are we in this probability estimate?
        Based on sample size
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if state not in self.table:
            return 0.0
        
        total = self.table[state]['total']
        
        # Confidence grows with sample size
        # 100 trades = full confidence (100%)
        # 30 trades = 30% confidence (minimum for validation)
        return min(total / 100.0, 1.0)

    def should_fire(self, state, min_prob: float = 0.80, min_conf: float = 0.30) -> bool:
        """Fire trade only if win probability and confidence meet thresholds."""
        return (self.get_probability(state) >= min_prob
                and self.get_confidence(state) >= min_conf)

    def get_dir_probability(self, state, direction: str):
        """Direction-specific win rate. Returns None if < 3 samples."""
        dir_key = (state, direction)
        if dir_key not in self.dir_table:
            return None
        data = self.dir_table[dir_key]
        if data['total'] < 3:
            return None
        return (data['wins'] + 1) / (data['total'] + 11)

    def get_expected_pnl(self, tid, side: str):
        """Average PnL per trade for (template, direction). None if < 3 obs."""
        bias = self.get_dir_bias(tid)
        if bias is None:
            return None
        key = side.lower()
        n = bias.get(f'{key}_n', 0)
        if n < 3:
            return None
        return bias.get(f'{key}_pnl', 0.0) / n

    def get_stats(self, state) -> Dict:
        """Get detailed statistics for a state."""
        if state not in self.table:
            return {'probability': 0.50, 'confidence': 0.0,
                    'wins': 0, 'losses': 0, 'total': 0, 'sample_size': 0}
        data = self.table[state]
        return {
            'probability': self.get_probability(state),
            'confidence': self.get_confidence(state),
            'wins': data['wins'], 'losses': data['losses'],
            'total': data['total'], 'sample_size': data['total'],
        }

    def batch_update(self, outcomes: list):
        """Batch update for bulk outcome recording (Monte Carlo)."""
        for outcome in outcomes:
            self.update(outcome)

    def save(self, filepath: str):
        """Persist probability table to disk"""
        save_data = {
            'table': dict(self.table),  # Convert defaultdict to dict
            'dir_table': dict(self.dir_table),
            'trade_history': self.trade_history,
            'dir_bias': self.dir_bias,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"[BAYESIAN] Saved {len(self.table)} state patterns to {filepath}")
    
    def load(self, filepath: str):
        """Load probability table from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore table
        self.table = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0},
            save_data['table']
        )
        self.trade_history = save_data.get('trade_history', [])
        # Restore direction table (backward-compatible)
        self.dir_table = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0},
            save_data.get('dir_table', {})
        )
        self.dir_bias = save_data.get('dir_bias', {})

        print(f"[BAYESIAN] Loaded {len(self.table)} state patterns from {filepath}")
    
def record_trade(brain: 'BayesianBrain', *, tid, entry_price: float,
                 exit_price: float, pnl: float, side: str,
                 exit_reason: str, timestamp: float,
                 entry_time: float = 0.0, exit_time: float = 0.0,
                 tick_value: float = 0.50,
                 hold_bars: int = 0) -> TradeOutcome:
    """Shared trade recording  -- used by both trainer and live.

    Constructs TradeOutcome, updates brain table + direction learning.
    Returns the outcome for caller-specific bookkeeping.
    """
    outcome = TradeOutcome(
        state=tid if tid is not None else 'UNKNOWN',
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        result='WIN' if pnl > 0 else 'LOSS',
        timestamp=timestamp,
        exit_reason=exit_reason,
        entry_time=entry_time,
        exit_time=exit_time,
        duration=exit_time - entry_time if exit_time and entry_time else 0.0,
        direction='LONG' if side == 'long' else 'SHORT',
        template_id=tid,
    )
    brain.update(outcome)
    brain.direction_learn(tid, side, pnl, tick_value=tick_value)
    if hold_bars > 0:
        brain.record_hold_bars(tid, side, hold_bars)
    return outcome


# Backward compat: old checkpoints may pickle these class names
MarketBayesianBrain = BayesianBrain
QuantumBayesianBrain = BayesianBrain

