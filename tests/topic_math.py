import pytest
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock imports if core modules are missing or need mocking,
# but assuming they exist based on manifest
from core.bayesian_brain import BayesianBrain, TradeOutcome
from core.state_vector import StateVector

def test_synthetic_baseline():
    """
    Verify that switching to LEARNING mode correctly identifies the 10W-2L synthetic baseline ($P=78.57\%$).
    """
    brain = BayesianBrain()
    state = StateVector.null_state() # Use null state or any fixed state

    # 10 Wins
    for _ in range(10):
        outcome = TradeOutcome(
            state=state,
            entry_price=100.0,
            exit_price=110.0,
            pnl=10.0,
            result='WIN',
            timestamp=time.time(),
            exit_reason='test'
        )
        brain.update(outcome)

    # 2 Losses
    for _ in range(2):
        outcome = TradeOutcome(
            state=state,
            entry_price=100.0,
            exit_price=90.0,
            pnl=-10.0,
            result='LOSS',
            timestamp=time.time(),
            exit_reason='test'
        )
        brain.update(outcome)

    prob = brain.get_probability(state)

    # Expected: (10 + 1) / (12 + 2) = 11 / 14 = 0.785714...
    expected = 11/14

    # Assert approx equal
    assert abs(prob - expected) < 1e-4, f"Expected {expected:.4f}, got {prob:.4f}"
    # Verify strict string formatting matches prompt expectation roughly
    assert f"{prob*100:.2f}%" == "78.57%"
