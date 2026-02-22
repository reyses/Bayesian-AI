"""
Phase 0: Unconstrained Exploration
No rules. No gates. Pure pattern discovery.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Set, Optional
import random
from core.three_body_state import ThreeBodyQuantumState

logger = logging.getLogger(__name__)

@dataclass
class ExplorationConfig:
    """Configuration for pure exploration mode"""
    max_trades: int = 500              # Explore until 500 trades
    min_unique_states: int = 50        # Or until 50 unique patterns
    fire_probability: float = 1.0      # 100% = fire every opportunity
    randomize_entries: bool = False    # If True, randomly skip some trades
    ignore_all_gates: bool = True      # Fire regardless of L8/L9
    allow_chaos_zone: bool = True      # Trade even at L1 (neutral zone)
    learn_from_failures: bool = True   # Record losses too

class UnconstrainedExplorer:
    """
    Exploration engine with ZERO constraints
    Discovers patterns by experiencing everything
    """

    def __init__(self, config: Optional[ExplorationConfig] = None):
        self.config = config or ExplorationConfig()
        self.trades_executed = 0
        self.unique_states_seen: Set[int] = set()

    def should_fire(self, state: ThreeBodyQuantumState) -> dict:
        """
        PHASE 0 LOGIC: Fire on (almost) everything

        NO CHECKS FOR:
        - Lagrange zone (trade anywhere)
        - Measurements (L8/L9 don't matter)
        - Momentum override (ignore F_momentum)
        - Probability (no table yet)
        - Confidence (building from zero)

        ONLY CHECK:
        - Trade count limit (stop at max_trades)
        - Optional: Random sampling (if fire_probability < 1.0)
        """

        # Stop if hit trade limit
        if self.trades_executed >= self.config.max_trades:
            return {
                'should_fire': False,
                'reason': f'Exploration complete ({self.trades_executed} trades)',
                'phase': 0
            }

        # Optional: Random sampling (for partial exploration)
        if self.config.fire_probability < 1.0:
            if random.random() > self.config.fire_probability:
                return {
                    'should_fire': False,
                    'reason': f'Random skip (sampling {self.config.fire_probability:.0%})',
                    'phase': 0
                }

        # FIRE DECISION (permissive)
        if self.config.allow_chaos_zone:
            # Trade EVERYWHERE (even L1 neutral zone)
            fire = True
            reason = f"UNCONSTRAINED: Trading all zones (z={state.z_score:.2f})"
        else:
            # Only trade near extremes (still very permissive)
            fire = abs(state.z_score) >= 1.0  # Any displacement from center
            reason = f"EXPLORATION: z={state.z_score:.2f} (threshold: 1.0σ)"

        # Track unique states
        self.unique_states_seen.add(hash(state))

        return {
            'should_fire': fire,
            'reason': reason,
            'phase': 0,
            'exploration_progress': {
                'trades': self.trades_executed,
                'max_trades': self.config.max_trades,
                'unique_states': len(self.unique_states_seen),
                'target_states': self.config.min_unique_states
            }
        }

    def record_trade(self, outcome):
        """Track exploration progress"""
        self.trades_executed += 1

        # Progress report every 50 trades
        if self.trades_executed % 50 == 0:
            logger.info(f"EXPLORATION PROGRESS: {self.trades_executed}/{self.config.max_trades} trades")
            logger.info(f"Unique states discovered: {len(self.unique_states_seen)}")
            logger.info(f"Last outcome: {outcome.result} | P&L: ${outcome.pnl:.2f}")

    def is_complete(self) -> bool:
        """Check if exploration phase is done"""
        trades_complete = self.trades_executed >= self.config.max_trades
        states_complete = len(self.unique_states_seen) >= self.config.min_unique_states
        return trades_complete or states_complete

    def get_completion_report(self) -> str:
        """Summary of exploration phase"""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║              EXPLORATION PHASE COMPLETE                          ║
╚══════════════════════════════════════════════════════════════════╝

📊 DISCOVERY METRICS:
   Trades Executed:        {self.trades_executed}
   Unique States Found:    {len(self.unique_states_seen)}

🎯 PHASE 0 COMPLETE - Ready for Phase 1 (Adaptive Learning)

Next Step: Initialize AdaptiveConfidenceManager with learned states
{'='*68}
"""