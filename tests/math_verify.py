import pytest
from core.bayesian_brain import BayesianBrain, TradeOutcome
from core.state_vector import StateVector

def test_laplace_smoothing():
    brain = BayesianBrain()
    # Case: 7 wins, 3 losses -> (7+1)/(10+2) = 0.666...

    # Create a dummy state for testing
    state = StateVector.null_state()

    # 7 wins
    for _ in range(7):
        # Adapting the prompt's brain.update("state_test", 1) to the actual API
        outcome = TradeOutcome(
            state=state,
            entry_price=100.0,
            exit_price=110.0,
            pnl=10.0,
            result='WIN',
            timestamp=1234567890,
            exit_reason='test'
        )
        brain.update(outcome)

    # 3 losses
    for _ in range(3):
        # Adapting the prompt's brain.update("state_test", 0) to the actual API
        outcome = TradeOutcome(
            state=state,
            entry_price=100.0,
            exit_price=90.0,
            pnl=-10.0,
            result='LOSS',
            timestamp=1234567890,
            exit_reason='test'
        )
        brain.update(outcome)

    assert abs(brain.get_probability(state) - (8/12)) < 1e-6

def test_hash_consistency():
    # Adapting the prompt's StateVector usage to match the actual class definition
    # s1 = StateVector(l1_bias=1, l9_velocity=10.5)

    s1 = StateVector(
        L1_bias='bull',
        L2_regime='trending',
        L3_swing='higher_highs',
        L4_zone='at_support',
        L5_trend='up',
        L6_structure='bullish',
        L7_pattern='flag',
        L8_confirm=True,
        L9_cascade=True
    )

    s2 = StateVector(
        L1_bias='bull',
        L2_regime='trending',
        L3_swing='higher_highs',
        L4_zone='at_support',
        L5_trend='up',
        L6_structure='bullish',
        L7_pattern='flag',
        L8_confirm=True,
        L9_cascade=True
    )

    assert hash(s1) == hash(s2)
