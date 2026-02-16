"""
Bayesian-AI - Bayesian Probability Engine
HashMap-based learning system: StateVector -> WinRate
"""
import pickle
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any
from core.state_vector import StateVector
from core.three_body_state import ThreeBodyQuantumState

# Statistical validation components (optional imports)
try:
    from training.integrated_statistical_system import (
        BayesianStateValidator,
        MonteCarloRiskAnalyzer
    )
    STATISTICAL_VALIDATION_AVAILABLE = True
except ImportError:
    STATISTICAL_VALIDATION_AVAILABLE = False

@dataclass
class TradeOutcome:
    """Single trade result for learning"""
    state: Union[StateVector, ThreeBodyQuantumState, str, int]
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
    Core logic: probability_table[StateVector] = {'wins': X, 'losses': Y}
    """
    def __init__(self):
        self.table: Dict[Any, Dict[str, int]] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total': 0}
        )
        self.trade_history = []
        
    def update(self, outcome: TradeOutcome):
        """
        Bayesian update after trade completion
        Args:
            outcome: TradeOutcome with state vector and result
        """
        # Prefer template_id if available, otherwise use state as key
        key = outcome.template_id if outcome.template_id is not None else outcome.state
        
        # Update counts
        if outcome.result == 'WIN':
            self.table[key]['wins'] += 1
        else:
            self.table[key]['losses'] += 1
        
        self.table[key]['total'] += 1
        
        # Log trade
        self.trade_history.append(outcome)
    
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
    
    def should_fire(self, state: Any, min_prob: float = 0.80, min_conf: float = 0.30) -> bool:
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

    def should_fire_validated(self, state: Any, use_statistical_validation: bool = True) -> Dict:
        """
        ENHANCED DECISION FUNCTION with Statistical Validation

        Applies multiple layers of validation:
        1. Bayesian validation: P(win_rate > 50%) > 80%
        2. Monte Carlo risk analysis: Expected DD, consecutive losses
        3. Sample size requirements

        Args:
            state: Current market state
            use_statistical_validation: Enable rigorous statistical checks

        Returns:
            dict with decision and validation results
        """
        if state not in self.table:
            return {
                'should_fire': False,
                'reason': 'Unknown state (no history)',
                'validations': None
            }

        record = self.table[state]
        wins = record['wins']
        losses = record['losses']
        total = record['total']

        # Basic threshold check
        prob = self.get_probability(state)
        conf = self.get_confidence(state)

        if not (prob >= 0.80 and conf >= 0.30):
            return {
                'should_fire': False,
                'reason': f'Basic threshold not met: P={prob:.1%}, Conf={conf:.1%}',
                'probability': prob,
                'confidence': conf,
                'validations': None
            }

        # If statistical validation not requested or not available, return basic result
        if not use_statistical_validation or not STATISTICAL_VALIDATION_AVAILABLE:
            return {
                'should_fire': True,
                'reason': f'Basic validation passed: P={prob:.1%}, Conf={conf:.1%}',
                'probability': prob,
                'confidence': conf,
                'validations': None
            }

        # STATISTICAL VALIDATION (requires imports)
        # Phase 1: Bayesian validation
        bayesian_validator = BayesianStateValidator(
            prior_wins=50,
            prior_losses=50,
            min_samples=30,
            confidence_threshold=0.80
        )

        bayesian_result = bayesian_validator.validate_state(wins, losses)

        if not bayesian_result['approved']:
            return {
                'should_fire': False,
                'reason': f"Bayesian validation failed: {bayesian_result['reason']}",
                'probability': prob,
                'confidence': conf,
                'validations': {
                    'bayesian': bayesian_result,
                    'monte_carlo': None
                }
            }

        # Phase 2: Monte Carlo risk analysis (if enough history)
        if total >= 10:
            # Get average win/loss amounts from trade history
            state_trades = [t for t in self.trade_history if t.state == state]

            if state_trades:
                pnls = [t.pnl for t in state_trades]
                avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 200
                avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else -100

                mc_analyzer = MonteCarloRiskAnalyzer(n_simulations=10000)
                mc_results = mc_analyzer.simulate_drawdown(
                    win_rate=bayesian_result['expected_win_rate'],
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    n_trades=100
                )

                risk_validation = mc_analyzer.validate_risk_profile(mc_results)

                if not risk_validation['risk_approved']:
                    return {
                        'should_fire': False,
                        'reason': f"Risk validation failed: {', '.join(risk_validation['concerns'])}",
                        'probability': prob,
                        'confidence': conf,
                        'validations': {
                            'bayesian': bayesian_result,
                            'monte_carlo': risk_validation
                        }
                    }
            else:
                risk_validation = {'risk_approved': True, 'concerns': []}
        else:
            risk_validation = {'risk_approved': True, 'concerns': []}

        # ALL VALIDATIONS PASSED
        return {
            'should_fire': True,
            'reason': f"All validations passed: {bayesian_result['confidence']:.1%} confidence",
            'probability': prob,
            'confidence': conf,
            'validations': {
                'bayesian': bayesian_result,
                'monte_carlo': risk_validation if total >= 10 else None
            },
            'expected_win_rate': bayesian_result['expected_win_rate'],
            'credible_interval': bayesian_result['credible_interval']
        }
    
    def get_stats(self, state: Any) -> Dict:
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

class QuantumBayesianBrain(BayesianBrain):
    """Extends BayesianBrain for ThreeBodyQuantumState"""
    
    def get_quantum_probability(self, state: ThreeBodyQuantumState) -> float:
        """Get learned tunnel probability for quantum state"""
        # Bin continuous values for lookup
        # Note: The ThreeBodyQuantumState.__hash__ already bins values, 
        # so using state as key works.
        
        # Use hashed state for lookup
        return self.get_probability(state)
    
    def should_fire_quantum(
        self, 
        state: ThreeBodyQuantumState, 
        min_prob: float = 0.80,
        min_conf: float = 0.30
    ) -> bool:
        """
        Quantum decision function
        Fire if:
        1. At Roche limit
        2. Wave function collapsed
        3. Learned probability > threshold
        4. Confidence sufficient
        """
        if state.lagrange_zone not in ['L2_ROCHE', 'L3_ROCHE']:
            return False
        
        if not (state.structure_confirmed and state.cascade_detected):
            return False
        
        if state.F_momentum > state.F_reversion * 1.5:
            return False
        
        prob = self.get_quantum_probability(state)
        conf = self.get_confidence(state)
        
        return prob >= min_prob and conf >= min_conf
