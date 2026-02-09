"""
Adaptive Confidence Bootstrap System
Learns optimal thresholds through progressive tightening
Starts at 0% → Converges to 80% over 600 trades
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np
from core.three_body_state import ThreeBodyQuantumState

@dataclass
class ConfidenceEvolution:
    """Tracks learning progression"""
    phase: int
    phase_name: str
    current_prob_threshold: float
    current_conf_threshold: float
    total_trades: int
    total_states_learned: int
    high_confidence_states: int
    elite_states: int
    overall_winrate: float
    recent_winrate: float
    recent_sharpe: float
    trades_until_next_phase: int
    next_phase_criteria: Dict
    probability_field_decay_learned: bool
    decay_states_count: int
    avg_sample_size: float
    state_coverage: float

class AdaptiveConfidenceManager:
    """Manages progressive tightening of trading criteria"""
    
    PHASES = {
        1: {
            'name': 'EXPLORATION',
            'prob_threshold': 0.00,
            'conf_threshold': 0.00,
            'duration_trades': 200,
            'goal': 'Build initial probability map'
        },
        2: {
            'name': 'REFINEMENT',
            'prob_threshold': 0.45,
            'conf_threshold': 0.20,
            'duration_trades': 400,  # Slower transition (was 200)
            'goal': 'Filter obvious losers'
        },
        3: {
            'name': 'OPTIMIZATION',
            'prob_threshold': 0.55,  # FIXED: Was 0.65 (unreachable), now 0.55
            'conf_threshold': 0.30,
            'duration_trades': 400,  # Slower transition (was 200)
            'goal': 'Focus on high-probability setups'
        },
        4: {
            'name': 'MASTERY',
            'prob_threshold': 0.80,
            'conf_threshold': 0.40,
            'duration_trades': float('inf'),
            'goal': 'Exploit proven edge'
        }
    }
    
    def __init__(self, brain):
        self.brain = brain
        self.phase = 1
        self.total_trades = 0
        self.trades_in_phase = 0
        self.decay_observations = []
    
    def should_fire(self, state: ThreeBodyQuantumState) -> dict:
        """Adaptive firing decision based on learning phase"""
        phase_config = self.PHASES[self.phase]
        prob = self.brain.get_probability(state)
        conf = self.brain.get_confidence(state)
        
        # Phase 1: Fire at everything (exploration)
        if self.phase == 1:
            if state.lagrange_zone in ['L2_ROCHE', 'L3_ROCHE']:
                return {
                    'should_fire': True,
                    'reason': 'EXPLORATION: Learning all Roche states',
                    'phase': 1
                }
        
        # Phases 2-4: Use learned probabilities
        meets_threshold = prob >= phase_config['prob_threshold'] and conf >= phase_config['conf_threshold']
        
        return {
            'should_fire': meets_threshold,
            'reason': f"{phase_config['name']}: P={prob:.2%}, Conf={conf:.2%}",
            'current_threshold': phase_config['prob_threshold'],
            'state_probability': prob,
            'state_confidence': conf,
            'phase': self.phase
        }
    
    def record_trade(self, outcome):
        """Record trade and check for phase advancement"""
        self.total_trades += 1
        self.trades_in_phase += 1
        if self._should_advance_phase():
            self._advance_phase()
    
    def _should_advance_phase(self) -> bool:
        """Check if ready for next phase"""
        if self.phase >= 4:
            return False
        
        phase_config = self.PHASES[self.phase]
        
        if self.trades_in_phase < phase_config['duration_trades']:
            return False
        
        # Performance checks
        recent_trades = self.brain.trade_history[-50:]
        recent_wr = sum(1 for t in recent_trades if t.result == 'WIN') / len(recent_trades) if recent_trades else 0
        
        high_conf_count = sum(
            1 for state in self.brain.table 
            if self.brain.get_confidence(state) >= 0.30
        )
        
        recent_pnls = [t.pnl for t in recent_trades]
        recent_sharpe = np.mean(recent_pnls) / (np.std(recent_pnls) + 1e-6) if recent_pnls else 0
        
        return recent_wr > 0.55 and high_conf_count >= 10 and recent_sharpe > 0.5
    
    def _advance_phase(self):
        """Advance to next learning phase"""
        old_phase = self.phase
        self.phase = min(self.phase + 1, 4)
        self.trades_in_phase = 0
        print(f"\n🎯 PHASE ADVANCEMENT: {self.PHASES[old_phase]['name']} → {self.PHASES[self.phase]['name']}")
        print(f"New threshold: {self.PHASES[self.phase]['prob_threshold']:.0%}")
        
    def generate_progress_report(self) -> str:
        """Generate a string report of the current learning status."""
        phase_config = self.PHASES[self.phase]
        recent_trades = self.brain.trade_history[-50:]
        recent_wr = sum(1 for t in recent_trades if t.result == 'WIN') / len(recent_trades) if recent_trades else 0
        
        return f"""
        [ADAPTIVE LEARNING] Phase {self.phase}: {phase_config['name']}
        Trades: {self.total_trades} (In Phase: {self.trades_in_phase})
        Recent WR: {recent_wr:.1%}
        Current Thresholds: P>={phase_config['prob_threshold']:.0%}, Conf>={phase_config['conf_threshold']:.0%}
        """
