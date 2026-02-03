"""
Bayesian-AI - Math Topic Test
Tests mathematical core functions.
"""
import pytest
import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.bayesian_brain import BayesianBrain, TradeOutcome
from core.state_vector import StateVector

def test_laplace_smoothing():
    """
    Verify Laplace Smoothing: (7 wins, 3 losses) must return exactly 0.666666...
    """
    brain = BayesianBrain()
    state = StateVector.null_state()

    # 7 wins
    for _ in range(7):
        brain.update(TradeOutcome(
            state=state, entry_price=0, exit_price=0, pnl=0, result='WIN',
            timestamp=0, exit_reason='test'
        ))

    # 3 losses
    for _ in range(3):
        brain.update(TradeOutcome(
            state=state, entry_price=0, exit_price=0, pnl=0, result='LOSS',
            timestamp=0, exit_reason='test'
        ))

    prob = brain.get_probability(state)

    # (7 + 1) / (10 + 2) = 8 / 12 = 2/3
    expected = 2.0 / 3.0

    assert prob == pytest.approx(expected, abs=1e-9)

def test_synthetic_baseline_10w_2l():
    """
    Verify 10W-2L synthetic baseline (P=78.57%)
    (10 wins, 2 losses) -> (10+1)/(12+2) = 11/14 = 0.785714...
    """
    brain = BayesianBrain()
    state = StateVector.null_state()

    for _ in range(10):
        brain.update(TradeOutcome(
            state=state, entry_price=0, exit_price=0, pnl=0, result='WIN',
            timestamp=0, exit_reason='test'
        ))

    for _ in range(2):
        brain.update(TradeOutcome(
            state=state, entry_price=0, exit_price=0, pnl=0, result='LOSS',
            timestamp=0, exit_reason='test'
        ))

    prob = brain.get_probability(state)
    expected = 11.0 / 14.0
    # 78.57% ~ 0.7857
    assert prob == pytest.approx(expected, abs=1e-4)

def test_hash_determinism():
    """
    Verify Hash Determinism: Identical StateVector inputs must yield identical hashes.
    """
    sv1 = StateVector(
        L1_bias='bull', L2_regime='trending', L3_swing='higher_highs', L4_zone='mid_range',
        L5_trend='up', L6_structure='bullish', L7_pattern='flag', L8_confirm=True, L9_cascade=False,
        timestamp=100.0, price=5000.0
    )

    sv2 = StateVector(
        L1_bias='bull', L2_regime='trending', L3_swing='higher_highs', L4_zone='mid_range',
        L5_trend='up', L6_structure='bullish', L7_pattern='flag', L8_confirm=True, L9_cascade=False,
        timestamp=200.0, price=6000.0 # Metadata differs
    )

    assert sv1 == sv2
    assert hash(sv1) == hash(sv2)

    # Different state should have different hash (high probability)
    sv3 = StateVector(
        L1_bias='bear', L2_regime='trending', L3_swing='higher_highs', L4_zone='mid_range',
        L5_trend='up', L6_structure='bullish', L7_pattern='flag', L8_confirm=True, L9_cascade=False
    )
    assert sv1 != sv3
    assert hash(sv1) != hash(sv3)

def test_9_layer_consistency():
    """
    Verify Static (L1-L4) vs. Fluid (L5-L9) state transitions.
    This test verifies that StateVector correctly holds the 9 layers.
    """
    sv = StateVector(
        L1_bias='bull', L2_regime='trending', L3_swing='higher_highs', L4_zone='mid_range',
        L5_trend='up', L6_structure='bullish', L7_pattern='flag', L8_confirm=True, L9_cascade=False
    )

    # Check Static Layers
    assert sv.L1_bias == 'bull'
    assert sv.L2_regime == 'trending'
    assert sv.L3_swing == 'higher_highs'
    assert sv.L4_zone == 'mid_range'

    # Check Fluid Layers
    assert sv.L5_trend == 'up'
    assert sv.L6_structure == 'bullish'
    assert sv.L7_pattern == 'flag'
    assert sv.L8_confirm is True
    assert sv.L9_cascade is False

if __name__ == "__main__":
    pytest.main([__file__])
