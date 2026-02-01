import pytest
from core.bayesian_brain import BayesianBrain, TradeOutcome
from core.state_vector import StateVector

def test_laplace_smoothing():
    brain = BayesianBrain()
    # Case: 7 wins, 3 losses -> (7+1)/(10+2) = 0.666...

    # Create a dummy state for testing
    state = StateVector.null_state()

    def add_trade(is_win):
        outcome = TradeOutcome(
            state=state,
            entry_price=100.0,
            exit_price=110.0 if is_win else 90.0,
            pnl=10.0 if is_win else -10.0,
            result='WIN' if is_win else 'LOSS',
            timestamp=1234567890.0,
            exit_reason='test'
        )
        brain.update(outcome)

    # 7 wins
    for _ in range(7):
        add_trade(True)

    # 3 losses
    for _ in range(3):
        add_trade(False)

    assert abs(brain.get_probability(state) - (8/12)) < 1e-6

def test_hash_consistency():
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
