"""
Context Detector - 200-Parameter Context-Aware Activation System
Activates only relevant 10-40 parameters based on market context

Philosophy:
- NOT optimizing all 200 parameters simultaneously (overfitting)
- Activates subsets based on what matters NOW
- Each context has validated parameter ranges
"""
from typing import Dict, List, Any
from dataclasses import dataclass
from core.state_vector import StateVector
from core.three_body_state import ThreeBodyQuantumState
import pandas as pd
import numpy as np


@dataclass
class MarketContext:
    """Represents an active market context"""
    name: str
    active: bool
    strength: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    reason: str


class ContextDetector:
    """
    Detects active market contexts and returns relevant parameters

    10 Context Types:
    1. CORE (always active) - 10 params
    2. KILL_ZONE (at support/resistance) - 18 params
    3. PATTERN_SETUP (L7 active) - 15 params
    4. CONFIRMATION (L8 = True) - 12 params
    5. VELOCITY_SPIKE (L9 = True) - 10 params
    6. VOLATILITY_DIFFERENTIAL (3σ→2σ detected) - 25 params
    7. FRACTAL_RESONANCE (3-body alignment) - 40 params
    8. TRANSITION (state changing) - 15 params
    9. SESSION_SPECIFIC (time-based) - 20 params
    10. MICROSTRUCTURE (order flow) - 35 params

    Total: 200 parameters, but only 10-40 active at once
    """

    def __init__(self):
        # Context history for transition detection
        self.context_history = []
        self.state_history = []

        # Parameter libraries per context
        self.context_params = self._initialize_parameter_libraries()

    def _initialize_parameter_libraries(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all 200 parameters organized by context"""
        return {
            'CORE': {
                'stop_loss_ticks': 15,
                'take_profit_ticks': 40,
                'min_samples_required': 30,
                'confidence_threshold': 0.80,
                'max_hold_seconds': 600,
                'trail_activation_profit': 50,
                'trail_distance_tight': 10,
                'trail_distance_wide': 30,
                'max_consecutive_losses': 5,
                'min_sharpe_ratio': 0.5
            },
            'KILL_ZONE': {
                'require_at_killzone': True,
                'killzone_tolerance_ticks': 5,
                'require_rejection_wick': True,
                'min_rejection_wick_ticks': 5,
                'wick_to_body_ratio': 2.0,
                'wick_cluster_count': 3,
                'multiple_touches_required': 2,
                'zone_strength_multiplier': 1.5,
                'absorption_bars': 3,
                'trade_after_absorption': True,
                'roche_limit_approach_ticks': 10,
                'structure_stress_threshold': 0.8,
                'breakdown_probability_threshold': 0.70,
                'breakdown_velocity_required': 10,
                'commitment_threshold_ticks': 12,
                'commitment_volume_multiple': 2.5,
                'false_breakout_tolerance_ticks': 5,
                'identify_vacuum_zones': True
            },
            'PATTERN_SETUP': {
                'trade_flag': True,
                'trade_wedge': True,
                'trade_compression': True,
                'trade_breakdown': False,
                'pattern_confirmation_bars': 5,
                'pattern_min_range_ticks': 10,
                'pattern_min_bars': 3,
                'require_engulfing': False,
                'pinbar_body_max_pct': 0.3,
                'pinbar_wick_min_pct': 0.6,
                'avoid_doji_entry': True,
                'doji_body_max_ticks': 3,
                'engulfing_lookback': 2,
                'min_engulfing_size_ticks': 10,
                'compression_range_threshold': 0.7
            },
            'CONFIRMATION': {
                'volume_spike_threshold': 2.0,
                'min_volume_ratio': 1.0,
                'volume_confirmation_bars': 2,
                'require_increasing_volume': True,
                'volume_decline_exit': True,
                'volume_lookback_bars': 20,
                'aggressive_buyer_pct': 0.7,
                'bid_ask_imbalance_threshold': 0.7,
                'large_print_size_contracts': 25,
                'high_volume_rejection': True,
                'rejection_volume_min': 1.5,
                'exit_on_volume_drop': True
            },
            'VELOCITY_SPIKE': {
                'cascade_min_points': 10,
                'cascade_time_window': 0.5,
                'min_entry_velocity': 3,
                'max_entry_velocity': 30,
                'deceleration_exit': True,
                'tick_imbalance_threshold': 0.7,
                'min_tick_flow': 20,
                'tick_reversal_exit': True,
                'require_acceleration': False,
                'min_acceleration': 0
            },
            'VOLATILITY_DIFFERENTIAL': {
                'layer_high_volatility_sigma': 3.0,
                'layer_low_volatility_sigma': 2.0,
                'min_sigma_differential': 1.0,
                'max_sigma_differential': 2.5,
                'volatility_lookback_bars': 20,
                'monitor_layer_pairs': [['L7','L8'], ['L6','L7'], ['L5','L6']],
                'high_sigma_layer_must_be': 'either',
                'compression_must_be_recent': True,
                'compression_age_max_bars': 5,
                'predict_breakout_direction': True,
                'use_energy_flow_vector': True,
                'error_band_method': 'bollinger',
                'error_band_period': 20,
                'error_band_multiplier': 2.0,
                'require_band_touch': True,
                'band_touch_layer': 'low_sigma',
                'band_rejection_required': False,
                'squeeze_threshold_pct': 0.60,
                'squeeze_duration_min_bars': 5,
                'calculate_vol_gradient': True,
                'gradient_threshold': 0.3,
                'vol_correlation_threshold': -0.3,
                'correlation_lookback_bars': 20,
                'energy_accumulation_min_bars': 10,
                'energy_release_velocity_threshold': 15
            },
            'FRACTAL_RESONANCE': {
                # Lagrange point detection (12)
                'require_triad1_stable': True,
                'require_triad2_stable': True,
                'require_triad3_stable': True,
                'min_stable_triads': 2,
                'lagrange_point_tolerance': 0.1,
                'lagrange_duration_min_bars': 5,
                'transitional_state_action': 'skip',
                'macro_lagrange_points': 5,
                'meso_lagrange_points': 3,
                'micro_lagrange_points': 4,
                'unstable_config_penalty': 0.7,
                'stable_config_bonus': 1.5,

                # Resonance scoring (10)
                'resonance_score_method': 'multiplicative',
                'min_resonance_score': 8.0,
                'perfect_alignment_required': False,
                'triad_alignment_weights': {'macro_meso': 1.5, 'meso_micro': 2.0},
                'partial_alignment_acceptable': True,
                'conflicting_triads_action': 'skip',
                'resonance_decay_rate': 0.95,
                'layer_harmonic_weights': {'L5_L6': 1.5, 'L6_L7': 1.5, 'L7_L8': 2.0, 'L8_L9': 1.5},
                'cascade_trigger_layers': 4,
                'perfect_alignment_multiplier': 3.0,

                # Energy cascade detection (10)
                'require_energy_cascade': False,
                'cascade_direction': 'downward',
                'cascade_source_triad': 'macro',
                'high_energy_threshold_sigma': 3.0,
                'low_energy_threshold_sigma': 2.0,
                'cascade_time_window_bars': 10,
                'energy_transfer_efficiency': 0.7,
                'cascade_completion_required': False,
                'partial_cascade_acceptable': True,
                'reverse_cascade_warning': True,

                # Chaos avoidance (8)
                'detect_chaos': True,
                'max_chaos_score': 4,
                'chaos_state_changes_threshold': 5,
                'chaos_lookback_bars': 10,
                'exit_on_chaos_detected': False,
                'avoid_trading_during_chaos': True,
                'chaos_cooldown_bars': 10,
                'chaos_recovery_confirmation': 'three_bars'
            },
            'TRANSITION': {
                'transition_detection': True,
                'transition_speed_threshold': 2,
                'trade_during_transition': 'reduce',
                'hysteresis_factor': 0.3,
                'min_bars_in_state': 5,
                'state_flip_confirmation': 2,
                'detect_whipsaw_transitions': True,
                'whipsaw_lookback_bars': 10,
                'detect_regime_shift': True,
                'regime_shift_volatility_threshold': 3.0,
                'pause_trading_during_regime_shift': True,
                'regime_shift_cooldown_bars': 50,
                'state_change_energy_threshold': 1.0,
                'calculate_activation_energy': True,
                'valid_transition_sequences': 'predefined'
            },
            'SESSION_SPECIFIC': {
                'opening_range_minutes': 15,
                'opening_range_breakout_trade': True,
                'lunch_hour_avoid': True,
                'time_of_day_volatility_curve': 'hourly_profile',
                'high_volatility_hours': ['09:30-11:00', '14:00-16:00'],
                'reduce_size_low_vol_hours': True,
                'avoid_first_15min': False,
                'session_start': '09:30',
                'session_end': '15:30',
                'min_hold_seconds': 60,
                'max_hold_seconds': 900,
                'monday_bias': 'neutral',
                'friday_closeout': False,
                'opex_week_behavior': 'cautious',
                'avoid_fomc_minutes': 60,
                'news_event_cooldown': 30,
                'pre_market_reference': True,
                'overnight_gap_trade': False,
                'vix_spike_pause': True,
                'correlation_breakdown_threshold': 0.5
            },
            'MICROSTRUCTURE': {
                # Order flow imbalance (5)
                'bid_ask_imbalance_threshold': 0.7,
                'aggressive_vs_passive_ratio': 0.7,
                'order_flow_lookback_seconds': 10,
                'imbalance_reversal_exit': True,
                'flow_momentum_threshold': 0.6,

                # Large player detection (6)
                'institutional_size_threshold_contracts': 50,
                'iceberg_order_detection': True,
                'sweep_detection': True,
                'track_dark_pool_prints': False,
                'smart_money_confirmation': True,
                'institutional_follow_threshold': 100,

                # Liquidity analysis (6)
                'min_bid_ask_liquidity_contracts': 100,
                'spread_width_max_ticks': 2,
                'avoid_thin_liquidity': True,
                'liquidity_imbalance_threshold': 0.7,
                'depth_of_book_levels': 5,
                'order_book_pressure_score': 0.6,

                # Tape reading (8)
                'print_velocity': 50,
                'size_per_print': 10,
                'time_and_sales_lookback': 30,
                'uptick_downtick_ratio': 0.7,
                'block_trade_threshold': 100,
                'repeated_size_detection': True,
                'spoofing_detection': True,
                'layering_detection': False,

                # Intermarket (5)
                'correlation_with_futures': True,
                'spy_nq_correlation_threshold': 0.8,
                'vix_inverse_correlation': -0.7,
                'require_es_confirmation': False,
                'bond_yield_divergence_threshold': 0.2,

                # Higher-order stats (5)
                'return_skewness_threshold': 0.5,
                'trade_with_skew': True,
                'kurtosis_threshold': 5.0,
                'avoid_high_kurtosis': True,
                'hurst_exponent_threshold': 0.55
            }
        }

    def detect(self, state: Any, market_data: Dict, time_of_day: str = 'open') -> List[MarketContext]:
        """
        Detect active market contexts

        Args:
            state: StateVector or ThreeBodyQuantumState
            market_data: Dictionary with market data (df_ticks, df_bars, etc.)
            time_of_day: Session time ('open', 'mid', 'close')

        Returns:
            List of active MarketContext objects
        """
        contexts = []

        # Context 1: CORE (always active)
        contexts.append(MarketContext(
            name='CORE',
            active=True,
            strength=1.0,
            parameters=self.context_params['CORE'],
            reason='Always active'
        ))

        # Detect L4 zone for KILL_ZONE context
        if hasattr(state, 'L4_zone'):
            if state.L4_zone == 'at_killzone':
                contexts.append(MarketContext(
                    name='KILL_ZONE',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['KILL_ZONE'],
                    reason='L4 at killzone'
                ))
        elif hasattr(state, 'lagrange_zone'):
            if state.lagrange_zone in ['L2_ROCHE', 'L3_ROCHE']:
                contexts.append(MarketContext(
                    name='KILL_ZONE',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['KILL_ZONE'],
                    reason='At Roche limit'
                ))

        # Context 3: PATTERN_SETUP (L7 active)
        if hasattr(state, 'L7_pattern'):
            if state.L7_pattern != 'none':
                contexts.append(MarketContext(
                    name='PATTERN_SETUP',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['PATTERN_SETUP'],
                    reason=f'L7 pattern: {state.L7_pattern}'
                ))

        # Context 4: CONFIRMATION (L8 = True)
        if hasattr(state, 'L8_confirm'):
            if state.L8_confirm:
                contexts.append(MarketContext(
                    name='CONFIRMATION',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['CONFIRMATION'],
                    reason='L8 confirmation active'
                ))
        elif hasattr(state, 'structure_confirmed'):
            if state.structure_confirmed:
                contexts.append(MarketContext(
                    name='CONFIRMATION',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['CONFIRMATION'],
                    reason='Structure confirmed'
                ))

        # Context 5: VELOCITY_SPIKE (L9 = True)
        if hasattr(state, 'L9_cascade'):
            if state.L9_cascade:
                contexts.append(MarketContext(
                    name='VELOCITY_SPIKE',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['VELOCITY_SPIKE'],
                    reason='L9 cascade detected'
                ))
        elif hasattr(state, 'cascade_detected'):
            if state.cascade_detected:
                contexts.append(MarketContext(
                    name='VELOCITY_SPIKE',
                    active=True,
                    strength=1.0,
                    parameters=self.context_params['VELOCITY_SPIKE'],
                    reason='Cascade detected'
                ))

        # Context 6: VOLATILITY_DIFFERENTIAL (detect 3σ→2σ pattern)
        vol_diff_detected = self._detect_volatility_differential(market_data)
        if vol_diff_detected:
            contexts.append(MarketContext(
                name='VOLATILITY_DIFFERENTIAL',
                active=True,
                strength=vol_diff_detected['strength'],
                parameters=self.context_params['VOLATILITY_DIFFERENTIAL'],
                reason=vol_diff_detected['reason']
            ))

        # Context 7: FRACTAL_RESONANCE (3-body alignment)
        resonance_score = self._calculate_resonance_score(state)
        if resonance_score >= 8.0:
            contexts.append(MarketContext(
                name='FRACTAL_RESONANCE',
                active=True,
                strength=min(resonance_score / 10.0, 1.0),
                parameters=self.context_params['FRACTAL_RESONANCE'],
                reason=f'Resonance score: {resonance_score:.1f}'
            ))

        # Context 8: TRANSITION (state changing)
        if self._is_transitioning(state):
            contexts.append(MarketContext(
                name='TRANSITION',
                active=True,
                strength=0.5,
                parameters=self.context_params['TRANSITION'],
                reason='State transition detected'
            ))

        # Context 9: SESSION_SPECIFIC (always check time)
        contexts.append(MarketContext(
            name='SESSION_SPECIFIC',
            active=True,
            strength=1.0,
            parameters=self.context_params['SESSION_SPECIFIC'],
            reason=f'Session time: {time_of_day}'
        ))

        # Store state for transition detection
        self.state_history.append(state)
        if len(self.state_history) > 50:
            self.state_history.pop(0)

        return contexts

    def get_active_parameters(self, contexts: List[MarketContext]) -> Dict[str, Any]:
        """
        Merge all active context parameters

        Returns combined parameter dictionary with all active params
        """
        merged_params = {}

        for context in contexts:
            if context.active:
                merged_params.update(context.parameters)

        return merged_params

    def _detect_volatility_differential(self, market_data: Dict) -> Dict:
        """Detect 3σ→2σ volatility cascade pattern"""
        # Simplified detection logic
        # In production, would calculate σ levels per layer

        if 'ticks' not in market_data or market_data['ticks'] is None:
            return None

        # Placeholder: Would calculate rolling volatility
        # and detect high-to-low sigma differential
        return None

    def _calculate_resonance_score(self, state: Any) -> float:
        """Calculate 3-body resonance alignment score (0-10)"""
        score = 0.0

        # Check if StateVector type (9-layer system)
        if hasattr(state, 'L1_bias'):
            # Triad 1 (Macro): L1, L2, L3
            if state.L1_bias == 'bull' and state.L2_regime == 'trending' and state.L3_swing == 'higher_highs':
                score += 3.0
            elif state.L1_bias == 'bear' and state.L2_regime == 'trending' and state.L3_swing == 'lower_lows':
                score += 3.0

            # Triad 2 (Meso): L4, L5, L6
            if hasattr(state, 'L5_trend') and hasattr(state, 'L6_structure'):
                if state.L5_trend == 'up' and state.L6_structure == 'bullish':
                    score += 3.0
                elif state.L5_trend == 'down' and state.L6_structure == 'bearish':
                    score += 3.0

            # Triad 3 (Micro): L7, L8, L9
            if hasattr(state, 'L7_pattern') and hasattr(state, 'L8_confirm') and hasattr(state, 'L9_cascade'):
                if state.L7_pattern != 'none' and state.L8_confirm and state.L9_cascade:
                    score += 4.0

        # Check if ThreeBodyQuantumState
        elif hasattr(state, 'lagrange_zone'):
            # At Lagrange point = stable configuration
            if state.lagrange_zone in ['L1_MACRO', 'L2_MESO', 'L3_MICRO']:
                score += 5.0

            # Structure confirmed + cascade = resonance
            if hasattr(state, 'structure_confirmed') and hasattr(state, 'cascade_detected'):
                if state.structure_confirmed and state.cascade_detected:
                    score += 5.0

        return score

    def _is_transitioning(self, state: Any) -> bool:
        """Detect if system is in transition between states"""
        if len(self.state_history) < 5:
            return False

        # Check if recent states are different (unstable)
        recent_states = self.state_history[-5:]
        unique_states = len(set(hash(s) for s in recent_states))

        # If more than 3 different states in last 5 bars = transitioning
        return unique_states >= 3

    def get_context_summary(self, contexts: List[MarketContext]) -> str:
        """Generate human-readable context summary"""
        active_contexts = [c.name for c in contexts if c.active]
        total_params = len(self.get_active_parameters(contexts))

        summary = f"Active Contexts ({len(active_contexts)}):\n"
        for context in contexts:
            if context.active:
                summary += f"  • {context.name} (strength: {context.strength:.1f}) - {context.reason}\n"

        summary += f"\nTotal Active Parameters: {total_params}/200"

        return summary


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("CONTEXT DETECTOR - DEMO")
    print("="*80)

    detector = ContextDetector()

    # Create test state
    from core.state_vector import StateVector

    test_state = StateVector(
        L1_bias='bull',
        L2_regime='trending',
        L3_swing='higher_highs',
        L4_zone='at_killzone',
        L5_trend='up',
        L6_structure='bullish',
        L7_pattern='flag',
        L8_confirm=True,
        L9_cascade=True,
        timestamp=0.0,
        price=21500.0
    )

    # Detect contexts
    contexts = detector.detect(test_state, {}, 'open')

    # Print summary
    print(detector.get_context_summary(contexts))

    # Print active parameters
    active_params = detector.get_active_parameters(contexts)
    print(f"\nSample Active Parameters:")
    for key, value in list(active_params.items())[:10]:
        print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("✅ CONTEXT DETECTOR READY")
    print("="*80)
