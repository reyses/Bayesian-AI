"""
Comprehensive test suite for all 5 layers
"""
import pytest
import numpy as np
import pandas as pd
from dataclasses import replace
from core.three_body_state import ThreeBodyQuantumState
from core.quantum_field_engine import QuantumFieldEngine
from core.fractal_three_body import FractalMarketState, FractalTradingLogic, FractalThreeBodyLayer
from core.resonance_cascade import ResonanceCascadeDetector
from core.adaptive_confidence import AdaptiveConfidenceManager
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome

def test_quantum_state_creation():
    """Test basic state creation"""
    state = ThreeBodyQuantumState.null_state()
    assert state.lagrange_zone == 'L1_STABLE'
    assert hash(state) is not None

def test_roche_limit_detection():
    """Test system detects Roche limit correctly"""
    # Create data with price at +2σ
    dates = pd.date_range('2025-01-01', periods=100, freq='15min')
    base_price = 21500
    
    df_macro = pd.DataFrame({
        'close': [base_price] * 80 + [base_price + 100] * 20,
        'high': [base_price + 10] * 80 + [base_price + 110] * 20,
        'low': [base_price - 10] * 80 + [base_price + 90] * 20,
        'open': [base_price] * 100,
        'volume': [2000] * 100
    }, index=dates)
    
    df_micro = pd.DataFrame({
        'close': [base_price + 100] * 100,
        'high': [base_price + 105] * 100,
        'low': [base_price + 95] * 100,
        'open': [base_price + 100] * 100,
        'volume': [200] * 100
    }, index=pd.date_range('2025-01-01', periods=100, freq='15s'))
    
    engine = QuantumFieldEngine()
    state = engine.calculate_three_body_state(
        df_macro, df_micro, base_price + 100, 2000, 0.0
    )
    
    assert state.lagrange_zone in ['L2_ROCHE', 'L3_ROCHE']

def test_fractal_alignment():
    """Test fractal multi-scale detection"""
    layers = []
    for i in range(9):
        layers.append(FractalThreeBodyLayer(
            timeframe='15m', parent_timeframe='1h', child_timeframe='1m',
            center_mass=100, upper_singularity=102, lower_singularity=98,
            local_position=102.5, local_z_score=2.5, local_velocity=0.1,
            F_reversion_local=1.0, F_momentum_local=0.1,
            lagrange_zone_local='L2_ROCHE',
            wave_function_local=1+0j, tunnel_prob_local=0.85,
            phase=0.1  # Aligned phases
        ))
        
    decision = FractalTradingLogic.check_fractal_alignment(layers)
    assert decision['confidence_level'] == 'EXTREME'
    assert decision['roche_alignment_count'] == 9

def test_resonance_detection():
    """Test harmonic alignment detector"""
    detector = ResonanceCascadeDetector()
    state = ThreeBodyQuantumState.null_state()
    
    # Aligned phases (all 0)
    deviations = {f'L{i}': 1.0 for i in range(1, 10)}
    velocities = {f'L{i}': 0.1 for i in range(1, 10)} # small positive velocity, phase ~ 0
    
    res_state = detector.detect_resonance(
        state, deviations, velocities,
        {'current_volume': 1000, 'avg_volume': 1000},
        10000, []
    )
    
    assert res_state.phase_coherence > 0.9 # Should be very high
    assert res_state.resonance_type in ['FULL', 'CRITICAL']

def test_adaptive_confidence_progression():
    """Test phase advancement logic"""
    brain = QuantumBayesianBrain()
    mgr = AdaptiveConfidenceManager(brain)
    
    assert mgr.phase == 1
    assert mgr.PHASES[1]['prob_threshold'] == 0.0
    
    # Create 10 different states with high confidence and good win rate
    base_state = ThreeBodyQuantumState.null_state()
    
    for k in range(15): # 15 states
        # Create unique state by modifying z_score which affects hash
        s = replace(base_state, z_score=float(k))
        
        # Add 30 trades (full confidence) with all WINs (100% winrate)
        for _ in range(30):
            outcome = TradeOutcome(
                 state=s, entry_price=100, exit_price=110, pnl=10,
                 result='WIN', timestamp=0, exit_reason='test'
            )
            brain.update(outcome)
            mgr.record_trade(outcome) # Call record_trade to advance
            
    # Now check if we advanced
    # We have 15 * 30 = 450 trades
    # Phase 1 needs 200 trades
    
    assert mgr.phase >= 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
