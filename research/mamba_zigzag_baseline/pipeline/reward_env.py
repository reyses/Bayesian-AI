import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RewardConfig:
    # Swappable Knobs (Beta Status)
    density: str = "terminal"  # "terminal" or "dense"
    architecture: str = "multi_head"  # "multi_head" (3 heads) or "single_head"
    weight_selectivity: float = 1.0
    weight_direction: float = 1.0
    weight_exit: float = 2.0  # Asymmetric edge is here
    entropy_decay: float = 0.99
    regret_source: str = "oracle"  # "cubic" or "oracle"
    
    # Locked Invariants (Configured here for visibility, but theoretically locked)
    vol_normalization_window: int = 120  # Bars for ATR/StdDev

class BetaRewardPolicy:
    """
    Beta scaffold for the Direction-Exhaustion Reward & Exit Policy.
    Provisional until Stage 1 (Inflection Ablation) yields a MAKE verdict.
    """
    def __init__(self, config: RewardConfig):
        self.config = config
        
    def _compute_selectivity_reward(self, 
                                    took_trade: bool, 
                                    setup_quality_mfe: float, 
                                    setup_quality_mae: float, 
                                    volatility: float) -> float:
        """
        Component 1: Selectivity (Trade / No-Trade)
        Did you take the high-quality move and skip the wiggle?
        - Regret lives ONLY on entry.
        """
        # Objective quality gate: MFE/MAE (vol-normalized)
        norm_mfe = setup_quality_mfe / (volatility + 1e-8)
        norm_mae = setup_quality_mae / (volatility + 1e-8)
        
        # High quality = high MFE, low MAE
        is_high_quality = (norm_mfe > 2.0) and (norm_mae < 1.0)
        
        if took_trade:
            # Reward taking a good setup, mildly penalize taking a wiggle
            return 1.0 if is_high_quality else -0.2
        else:
            # Regret only fires if you missed a causally-readable, high-quality setup
            # Zero regret for sitting out a wiggle
            return -1.0 if is_high_quality else 0.1

    def _compute_direction_reward(self, 
                                  took_trade: bool, 
                                  predicted_dir: int, 
                                  actual_dir: int) -> float:
        """
        Component 2: Direction
        Of entries, were you on the correct side?
        """
        if not took_trade:
            return 0.0
            
        return 1.0 if (predicted_dir == actual_dir) else -1.0
        
    def _compute_exit_reward(self, 
                             took_trade: bool, 
                             is_right_direction: bool, 
                             holding_time: int, 
                             capture_rate: float,
                             accumulated_mae: float) -> float:
        """
        Component 3: Exit/Capture (The Edge)
        Asymmetric PnL: cut fast on wrong, ride to inflection on right.
        - Path independent (no memory of deficit).
        - Oracle-anchored capture.
        """
        if not took_trade:
            return 0.0
            
        if not is_right_direction:
            # WRONG trade: reward fast cutting.
            # Speed/adversity tension: cut-reward decays with holding time and MAE.
            speed_penalty = (holding_time / 10.0) 
            mae_penalty = accumulated_mae
            cut_score = 1.0 - (speed_penalty + mae_penalty)
            # Ensure it is a small positive if cut fast, but goes negative if bagged
            return max(cut_score * 0.5, -2.0)
        else:
            # RIGHT trade: reward capture rate vs the oracle half-cycle.
            # 100% capture = 1.0. 
            return capture_rate

    def compute_reward(self, state: Dict[str, Any], action: Dict[str, Any], hindsight_oracle: Dict[str, Any]) -> Dict[str, float]:
        """
        The leak wall: state (observation) is causal, hindsight_oracle is used ONLY here in the reward.
        Returns independent additive components (scorecard).
        """
        took_trade = action.get('take_trade', False)
        predicted_dir = action.get('direction', 0)
        holding_time = action.get('holding_time', 0)
        
        volatility = hindsight_oracle.get('volatility', 1.0)
        
        # 1. Selectivity
        sel_reward = self._compute_selectivity_reward(
            took_trade,
            hindsight_oracle.get('mfe', 0.0),
            hindsight_oracle.get('mae', 0.0),
            volatility
        )
        
        # 2. Direction
        actual_dir = hindsight_oracle.get('direction', 0)
        dir_reward = self._compute_direction_reward(took_trade, predicted_dir, actual_dir)
        
        # 3. Exit
        is_right = (predicted_dir == actual_dir) and (actual_dir != 0)
        exit_reward = self._compute_exit_reward(
            took_trade,
            is_right,
            holding_time,
            hindsight_oracle.get('capture_rate', 0.0),
            hindsight_oracle.get('accumulated_mae', 0.0) / (volatility + 1e-8)
        )
        
        # Additive components - never multiply the scorecard into a funnel!
        total_reward = (
            self.config.weight_selectivity * sel_reward +
            self.config.weight_direction * dir_reward +
            self.config.weight_exit * exit_reward
        )
        
        return {
            'selectivity': sel_reward,
            'direction': dir_reward,
            'exit': exit_reward,
            'total': total_reward
        }
