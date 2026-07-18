"""
ProjectX v2.0 - Bayesian Probability Engine
HashMap-based learning system: StateVector -> WinRate
"""
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional
from core.state_vector import StateVector

@dataclass
class TradeOutcome:
    """Single trade result for learning"""
    state: StateVector
    entry_price: float
    exit_price: float
    pnl: float
    result: str  # 'WIN' or 'LOSS'
    timestamp: float
    exit_reason: str  # 'trail_stop', 'structure_break', 'time_exit'

class BayesianBrain:
    """
    Probability table that learns from outcomes
    Core logic: probability_table[StateVector] = {'wins': X, 'losses': Y}
    """
    def __init__(self):
        self.table: Dict[StateVector, Dict[str, int]] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0}
        )
        self.trade_history = []
        
    def update(self, outcome: TradeOutcome):
        """
        Bayesian update after trade completion
        Args:
            outcome: TradeOutcome with state vector and result
        """
        state = outcome.state
        
        # Update counts
        if outcome.result == 'WIN':
            self.table[state]['wins'] += 1
        else:
            self.table[state]['losses'] += 1
        
        self.table[state]['total'] += 1
        
        # Log trade
        self.trade_history.append(outcome)
    
    def get_probability(self, state: StateVector) -> float:
        """
        Get win probability for given state
        Returns:
            float: Win probability (0.0 to 1.0)
        """
        if state not in self.table:
            return 0.50  # Neutral prior (no historical data)
        
        data = self.table[state]
        if data['total'] == 0:
            return 0.50
        
        # Bayesian estimate with Laplace smoothing (avoid 0/0)
        wins = data['wins'] + 1
        total = data['total'] + 2
        
        return wins / total
    
    def get_confidence(self, state: StateVector) -> float:
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
        # 30 trades = full confidence (100%)
        # 10 trades = 33% confidence
        # 1 trade = 3% confidence
        return min(total / 30.0, 1.0)
    
    def should_fire(self, state: StateVector, min_prob: float = 0.80, min_conf: float = 0.30) -> bool:
        """
        CORE DECISION FUNCTION
        Fire trade only if:
        1. Win probability > min_prob (default 80%)
        2. Confidence > min_conf (default 30% = ~10 prior trades)
        
        Args:
            state: Current market state
            min_prob: Minimum required win probability
            min_conf: Minimum required confidence
            
        Returns:
            bool: True if should fire, False otherwise
        """
        prob = self.get_probability(state)
        conf = self.get_confidence(state)
        
        return prob >= min_prob and conf >= min_conf
    
    def get_stats(self, state: StateVector) -> Dict:
        """Get detailed statistics for a state"""
        if state not in self.table:
            return {
                'probability': 0.50,
                'confidence': 0.0,
                'wins': 0,
                'losses': 0,
                'total': 0,
                'sample_size': 0
            }
        
        data = self.table[state]
        return {
            'probability': self.get_probability(state),
            'confidence': self.get_confidence(state),
            'wins': data['wins'],
            'losses': data['losses'],
            'total': data['total'],
            'sample_size': data['total']
        }
    
    def get_all_states_above_threshold(self, min_prob: float = 0.80) -> list:
        """
        Find all learned states with win probability above threshold
        Useful for analysis
        """
        results = []
        for state, data in self.table.items():
            prob = self.get_probability(state)
            if prob >= min_prob and data['total'] >= 10:  # Minimum 10 samples
                results.append({
                    'state': state,
                    'probability': prob,
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'total': data['total']
                })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)
    
    def save(self, filepath: str):
        """Persist probability table to disk"""
        save_data = {
            'table': dict(self.table),  # Convert defaultdict to dict
            'trade_history': self.trade_history
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
        
        print(f"[BAYESIAN] Loaded {len(self.table)} state patterns from {filepath}")
    
    def get_summary(self) -> Dict:
        """Overall learning statistics"""
        total_states = len(self.table)
        total_trades = sum(data['total'] for data in self.table.values())
        
        # Count high-probability states (>80%)
        high_prob_states = len([s for s in self.table if self.get_probability(s) >= 0.80 and self.table[s]['total'] >= 10])
        
        return {
            'total_unique_states': total_states,
            'total_trades': total_trades,
            'high_probability_states': high_prob_states,
            'avg_trades_per_state': total_trades / total_states if total_states > 0 else 0
        }
