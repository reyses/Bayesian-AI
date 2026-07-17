import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RewardConfig:
    # V2 Config weights
    w_c: float = 1.00     # Capture
    w_x: float = 0.35     # Cut bonus
    w_r: float = 0.75     # Regret (raised 0.25->0.75, anti-freeze: missing a
                          #   knowable high-P fire must STING, not whisper)
    w_s: float = 0.30     # Selectivity credit (anti-freeze: taking a knowable,
                          #   aligned setup pays immediately, independent of exit)
    w_d: float = 0.20     # Direction
    w_w: float = 0.15     # Wiggle penalty
    w_aux: float = 0.20   # Aux hazard loss (used in training loop)
    w_cost: float = 1.00  # Cost weight (real)
    
    # Thresholds and parameters
    theta_rem: float = 1.5      # Denominator floor for remaining extent (vol-normalized)
    theta_Q_mfe: float = 2.0    # Quality gate MFE
    theta_Q_mae: float = 1.0    # Quality gate MAE limit
    theta_c: float = 0.5        # Classifier confidence threshold for regret
    tau: float = 5.0            # Time decay constant for fast cut (bars)
    cost_ticks: float = 3.0     # Spread + commission + slippage in ticks
    
    vol_normalization_window: int = 120  # Bars for ATR/StdDev

class BetaRewardPolicy:
    """
    Direction-Exhaustion Reward & Exit Policy (v2)
    Implements path-independence, leak wall, additive components, and causal observation boundaries.
    """
    def __init__(self, config: RewardConfig):
        self.config = config
        self._regret_capped_swings = set() # Track swung IDs to cap regret once per swing

    def compute_reward(self, 
                       state: Dict[str, Any], 
                       action_type: str, 
                       hindsight_oracle: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates the V2 scorecard.
        action_type: 'FLAT_STEP', 'ENTRY', 'EXIT', 'IN_POSITION_STEP'
        """
        volatility = hindsight_oracle.get('volatility', 1.0)
        sigma_ticks = hindsight_oracle.get('sigma_ticks', volatility)
        
        scorecard = {
            'cost': 0.0,
            'capture': 0.0,
            'direction': 0.0,
            'cut_bonus': 0.0,
            'wiggle': 0.0,
            'regret': 0.0,
            'selectivity': 0.0,
            'total': 0.0
        }

        # Handle flat actions (Regret and Wiggle)
        if action_type == 'FLAT_STEP':
            c_t = hindsight_oracle.get('c_t', 0.0)
            is_label_covered = hindsight_oracle.get('is_label_covered', True)

            # Regret, windowed (P2 fix)
            # Credited only to the flat-action bars where c_t >= theta_c during the readable entry window.
            # Capped once per swing.
            if is_label_covered and c_t >= self.config.theta_c:
                swing_id = hindsight_oracle.get('swing_id', None)
                qualifying = hindsight_oracle.get('is_qualifying', False)
                if qualifying and swing_id and swing_id not in self._regret_capped_swings:
                    scorecard['regret'] = -self.config.w_r * c_t
                    self._regret_capped_swings.add(swing_id)

            scorecard['total'] = scorecard['regret']
            return scorecard

        # Handle Trade Terminal Actions (Entry -> Exit)
        if action_type == 'EXIT':
            is_label_covered = hindsight_oracle.get('is_label_covered', True)
            
            # Cost
            if sigma_ticks > 0:
                scorecard['cost'] = -self.config.w_cost * (self.config.cost_ticks / sigma_ticks)
                
            # Entry metrics
            predicted_dir = hindsight_oracle.get('predicted_dir', 0)
            actual_dir = hindsight_oracle.get('actual_dir', 0)
            is_right = (predicted_dir == actual_dir) and (actual_dir != 0)
            
            # Direction
            if actual_dir != 0:
                scorecard['direction'] = self.config.w_d if is_right else -self.config.w_d
                
            # Wiggle penalty (taken trade that doesn't qualify)
            qualifying = hindsight_oracle.get('is_qualifying', False)
            if not qualifying and is_label_covered:
                scorecard['wiggle'] = -self.config.w_w
                
            # Capture and Cut Bonus
            if is_right:
                # Right trade: Calculate capture rate
                captured = hindsight_oracle.get('captured_vol_norm', 0.0)
                remaining_extent = hindsight_oracle.get('remaining_extent_vol_norm', 0.0)
                
                # Denominator floor (θ_rem)
                denom = max(remaining_extent, self.config.theta_rem)
                capture_rate = captured / denom
                
                # If entry was so late that remaining_extent < θ_rem, it scores zero capture.
                if remaining_extent < self.config.theta_rem:
                    scorecard['capture'] = 0.0
                else:
                    scorecard['capture'] = self.config.w_c * capture_rate
            else:
                # Wrong trade: Cut bonus
                t_hold = hindsight_oracle.get('t_hold', 0.0)
                mae_norm = hindsight_oracle.get('mae_vol_norm', 0.0)
                
                cut_score = self.config.w_x * math.exp(-t_hold / self.config.tau) * math.exp(-mae_norm)
                scorecard['cut_bonus'] = cut_score

            scorecard['total'] = sum(scorecard.values())
            return scorecard

        # Handle Entry Action (Selectivity credit)
        if action_type == 'ENTRY':
            # Anti-freeze: taking a KNOWABLE, ALIGNED setup pays immediately and
            # additively -- independent of the later exit outcome. A direction
            # mismatch or an absent live signal scores 0 (NO penalty; the wiggle
            # term already charges junk entries at exit). The gate is the REAL
            # calibrated c_t from phit_feed, so this only rewards entries that
            # coincide with a decile-9/0 fire the agent could actually read.
            c_t = hindsight_oracle.get('c_t', 0.0)
            signal_dir = hindsight_oracle.get('signal_dir', 0)
            entry_dir = hindsight_oracle.get('predicted_dir', 0)
            if c_t >= self.config.theta_c and signal_dir != 0 and entry_dir == signal_dir:
                scorecard['selectivity'] = self.config.w_s * c_t
            scorecard['total'] = sum(scorecard.values())
            return scorecard

        return scorecard

def run_synthetic_tests():
    config = RewardConfig()
    policy = BetaRewardPolicy(config)
    
    print("Running Synthetic Tests for BetaRewardPolicy (V2 Patches)...")
    
    # Test 1: Fast cut on wrong trade nets positive (~+0.15)
    hindsight_1 = {
        'sigma_ticks': 10.0,
        'predicted_dir': 1,
        'actual_dir': -1,
        't_hold': 0.0, # instant cut
        'mae_vol_norm': 0.0,
        'is_label_covered': True,
        'is_qualifying': True
    }
    score_1 = policy.compute_reward({}, 'EXIT', hindsight_1)
    net_cut_dir = score_1['cut_bonus'] + score_1['direction']
    assert abs(net_cut_dir - 0.15) < 1e-5, f"Test 1 Failed: {net_cut_dir}"
    print(f"[PASS] (1) fast-cut-on-wrong nets positive: {net_cut_dir:.2f}")
    
    # Test 2: Entry at 90% of swing (below theta_rem) scores zero capture
    hindsight_2 = {
        'sigma_ticks': 10.0,
        'predicted_dir': 1,
        'actual_dir': 1,
        'is_label_covered': True,
        'is_qualifying': True,
        'captured_vol_norm': 0.5,
        'remaining_extent_vol_norm': 1.0 # Less than theta_rem (1.5)
    }
    score_2 = policy.compute_reward({}, 'EXIT', hindsight_2)
    assert score_2['capture'] == 0.0, "Test 2 Failed"
    print(f"[PASS] (2) late entry below theta_rem -> zero capture: {score_2['capture']:.2f}")
    
    # Test 3: Entry in label-gap region -> no wiggle penalty
    hindsight_3 = {
        'sigma_ticks': 10.0,
        'predicted_dir': 1,
        'actual_dir': 1,
        'is_label_covered': False, # GAP
        'is_qualifying': False # Wiggle
    }
    score_3 = policy.compute_reward({}, 'EXIT', hindsight_3)
    assert score_3['wiggle'] == 0.0, "Test 3 Failed"
    print(f"[PASS] (3) label-gap entry -> no wiggle penalty: {score_3['wiggle']:.2f}")
    
    # Test 4: Capture is net of cost -> tiny swing nets negative.
    hindsight_4 = {
        'sigma_ticks': 10.0,
        'predicted_dir': 1,
        'actual_dir': 1,
        'is_label_covered': True,
        'is_qualifying': True,
        'captured_vol_norm': 0.05, # Tiny capture
        'remaining_extent_vol_norm': 2.0
    }
    score_4 = policy.compute_reward({}, 'EXIT', hindsight_4)
    assert score_4['total'] < 0.0, f"Test 4 Failed"
    print(f"[PASS] (4) tiny swing nets negative after cost: {score_4['total']:.3f} (Cost={score_4['cost']}, Cap={score_4['capture']})")
    
    # Test 5: Two missed swings -> regret twice, capped, only on c_t>=theta_c
    policy2 = BetaRewardPolicy(config)
    hindsight_5a = {'c_t': 0.8, 'is_label_covered': True, 'swing_id': 's1', 'is_qualifying': True}
    hindsight_5b = {'c_t': 0.9, 'is_label_covered': True, 'swing_id': 's1', 'is_qualifying': True}
    hindsight_5c = {'c_t': 0.6, 'is_label_covered': True, 'swing_id': 's2', 'is_qualifying': True}
    hindsight_5d = {'c_t': 0.2, 'is_label_covered': True, 'swing_id': 's3', 'is_qualifying': True}
    
    r_a = policy2.compute_reward({}, 'FLAT_STEP', hindsight_5a)['regret']
    r_b = policy2.compute_reward({}, 'FLAT_STEP', hindsight_5b)['regret']
    r_c = policy2.compute_reward({}, 'FLAT_STEP', hindsight_5c)['regret']
    r_d = policy2.compute_reward({}, 'FLAT_STEP', hindsight_5d)['regret']

    # NOTE: magnitude tracks config.w_r (raised 0.25->0.75); the SEMANTICS of the
    # test are unchanged (capped once/swing, credited only on c_t>=theta_c).
    assert abs(r_a - (-config.w_r * 0.8)) < 1e-9
    assert r_b == 0.0
    assert abs(r_c - (-config.w_r * 0.6)) < 1e-9
    assert r_d == 0.0
    print(f"[PASS] (5) regret 2x capped, c_t-window-credited only: Capped: {r_b==0.0}, Windowed: {r_d==0.0}")

    # Test 6: REAL gate -- regret fires only when c_t >= theta_c (0.5), now that
    # c_t is the live calibrated probability (no longer mocked to 1.0).
    policy6 = BetaRewardPolicy(config)
    r_hi = policy6.compute_reward({}, 'FLAT_STEP',
        {'c_t': 0.51, 'is_label_covered': True, 'swing_id': 'g_hi', 'is_qualifying': True})['regret']
    r_lo = policy6.compute_reward({}, 'FLAT_STEP',
        {'c_t': 0.49, 'is_label_covered': True, 'swing_id': 'g_lo', 'is_qualifying': True})['regret']
    assert abs(r_hi - (-config.w_r * 0.51)) < 1e-9, f"Test 6 hi Failed: {r_hi}"
    assert r_lo == 0.0, f"Test 6 lo Failed: {r_lo}"
    print(f"[PASS] (6) regret real-gated at theta_c: c_t=.51->{r_hi:.3f}, c_t=.49->{r_lo:.3f}")

    # Test 7: Selectivity credit on a KNOWABLE, ALIGNED entry (additive, +w_s*c_t).
    s_aligned = policy.compute_reward({}, 'ENTRY',
        {'c_t': 0.8, 'signal_dir': 1, 'predicted_dir': 1})
    assert abs(s_aligned['selectivity'] - config.w_s * 0.8) < 1e-9, f"Test 7 Failed: {s_aligned}"
    assert abs(s_aligned['total'] - config.w_s * 0.8) < 1e-9, f"Test 7 total Failed: {s_aligned}"
    print(f"[PASS] (7) selectivity credit on aligned entry: {s_aligned['selectivity']:.3f}")

    # Test 8: NO selectivity (and NO penalty) on misaligned / absent / sub-gate signal.
    s_mis = policy.compute_reward({}, 'ENTRY',
        {'c_t': 0.8, 'signal_dir': 1, 'predicted_dir': -1})   # direction mismatch
    s_nosig = policy.compute_reward({}, 'ENTRY',
        {'c_t': 0.0, 'signal_dir': 0, 'predicted_dir': 1})    # no live signal
    s_weak = policy.compute_reward({}, 'ENTRY',
        {'c_t': 0.4, 'signal_dir': 1, 'predicted_dir': 1})    # aligned but below theta_c
    assert s_mis['selectivity'] == 0.0 and s_mis['total'] == 0.0, f"Test 8 mismatch Failed: {s_mis}"
    assert s_nosig['selectivity'] == 0.0, f"Test 8 nosig Failed: {s_nosig}"
    assert s_weak['selectivity'] == 0.0, f"Test 8 weak Failed: {s_weak}"
    print(f"[PASS] (8) no selectivity on misaligned/absent/weak signal (no penalty)")

    print("All 8 Synthetic Tests Passed explicitly (5 original + 3 anti-freeze)!")

if __name__ == '__main__':
    run_synthetic_tests()
