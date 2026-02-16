"""
Bayesian AI v2.0 - Quantum Integration Test
Validates ThreeBodyQuantumState + QuantumBayesianBrain + QuantumFieldEngine work together
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.three_body_state import ThreeBodyQuantumState
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.quantum_field_engine import QuantumFieldEngine
from core.multi_timeframe_context import TimeframeContext
from config.symbols import MNQ
from tests.utils import load_test_data

def test_quantum_state():
    """Test ThreeBodyQuantumState hashing and equality"""
    print("\n=== TEST 1: ThreeBodyQuantumState ===")

    # Create two identical states (conceptually)
    state1 = ThreeBodyQuantumState.null_state()
    # Mocking fields for testing hash
    # Note: dataclasses are immutable (frozen=True), so we must create new instances
    from dataclasses import replace
    state1 = replace(state1,
        z_score=2.5,
        momentum_strength=0.8,
        lagrange_zone='L2_ROCHE',
        structure_confirmed=True,
        cascade_detected=True,
        trend_direction_15m='UP',
        tunnel_probability=0.95  # High enough to trigger trade
    )

    state2 = replace(state1,
        particle_position=99999.0, # Differs but shouldn't affect hash
        timestamp=123456.0         # Differs but shouldn't affect hash
    )

    # Should be equal (timestamp/price not part of hash)
    assert state1 == state2, "States should be equal despite different metadata"
    assert hash(state1) == hash(state2), "Hashes should match"

    # Test inequality
    state3 = replace(state1, z_score=-2.5, lagrange_zone='L3_ROCHE')
    assert state1 != state3, "States with different z_score/zone should not be equal"

    print("[OK] ThreeBodyQuantumState hashing works correctly")
    print(f"  Hash: {hash(state1)}")

    # Test trade directive
    directive = state1.get_trade_directive()
    print(f"  Trade Directive (L2_ROCHE): {directive['action']}")
    if directive['action'] == 'WAIT':
        print(f"    Reason: {directive.get('reason')}")

def test_quantum_brain():
    """Test QuantumBayesianBrain probability learning"""
    print("\n=== TEST 2: QuantumBayesianBrain ===")

    brain = QuantumBayesianBrain()

    # Create test state
    state = ThreeBodyQuantumState.null_state()
    from dataclasses import replace
    state = replace(state,
        z_score=3.0,
        lagrange_zone='L2_ROCHE',
        structure_confirmed=True,
        cascade_detected=True
    )

    # Initial probability (no data)
    initial_prob = brain.get_quantum_probability(state)
    expected_prior = 0.09
    assert abs(initial_prob - expected_prior) < 1e-9, f"Initial probability should be {expected_prior}, got {initial_prob}"
    print(f"[OK] Initial probability: {initial_prob:.2%}")

    # Simulate 10 wins, 2 losses
    for i in range(10):
        outcome = TradeOutcome(
            state=state, entry_price=21500, exit_price=21520,
            pnl=40, result='WIN', timestamp=float(i), exit_reason='trail_stop'
        )
        brain.update(outcome)

    for i in range(2):
        outcome = TradeOutcome(
            state=state, entry_price=21500, exit_price=21490,
            pnl=-20, result='LOSS', timestamp=float(i), exit_reason='structure_break'
        )
        brain.update(outcome)

    # Updated probability
    learned_prob = brain.get_quantum_probability(state)
    confidence = brain.get_confidence(state)

    print(f"[OK] After 12 trades (10W-2L):")
    print(f"  Probability: {learned_prob:.2%}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Should fire (80% threshold): {brain.should_fire_quantum(state)}")

    # Test statistics
    stats = brain.get_stats(state)
    print(f"[OK] Stats: {stats}")

def test_quantum_field_engine():
    """Test QuantumFieldEngine computation"""
    print("\n=== TEST 3: QuantumFieldEngine ===")

    # Load test data
    data = load_test_data()
    print(f"Loaded {len(data)} rows of data.")

    engine = QuantumFieldEngine(regression_period=21)

    # Use batch_compute_states as it's the primary method used in Orchestrator
    # We need a small subset to test
    subset = data.head(100).copy()

    # Ensure columns exist
    if 'price' not in subset.columns and 'close' in subset.columns:
        subset['price'] = subset['close']
    if 'volume' not in subset.columns:
        subset['volume'] = 0.0

    print("  Running batch_compute_states...")

    # Check environment variable for deterministic testing
    use_cuda_env = os.environ.get("TEST_USE_CUDA", "auto").lower()
    if use_cuda_env == "true":
        use_cuda = True
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("TEST_USE_CUDA is 'true' but CUDA is not available.")
    elif use_cuda_env == "false":
        use_cuda = False
    else:  # 'auto'
        use_cuda = torch.cuda.is_available()

    print(f"  Using CUDA: {use_cuda}")
    results = engine.batch_compute_states(subset, use_cuda=use_cuda)

    print(f"  Computed {len(results)} states.")

    if len(results) > 0:
        first_res = results[0]
        state = first_res['state']
        print(f"[OK] Sample State (Bar {first_res['bar_idx']}):")
        print(f"  Zone: {state.lagrange_zone}")
        print(f"  Z-Score: {state.z_score:.2f}")
        print(f"  Tunnel Prob: {state.tunnel_probability:.2%}")

        # Validation checks
        assert not np.isnan(state.z_score), "Z-Score should not be NaN"
        assert not np.isnan(state.tunnel_probability), "Tunnel Prob should not be NaN"
    else:
        print("  WARNING: No states computed (maybe too few data points?)")

def test_full_quantum_integration():
    """Test full integration: QuantumFieldEngine → ThreeBodyQuantumState → QuantumBayesianBrain"""
    print("\n=== TEST 4: Full Quantum Integration ===")

    # Setup
    data = load_test_data()
    # Use a chunk where price moves significantly to trigger Roche zones
    # We'll just use the tail of the data
    subset = data.tail(200).copy()

    if 'price' not in subset.columns and 'close' in subset.columns:
        subset['price'] = subset['close']

    engine = QuantumFieldEngine(regression_period=21)
    brain = QuantumBayesianBrain()

    print("  Computing states...")

    # Check environment variable for deterministic testing
    use_cuda_env = os.environ.get("TEST_USE_CUDA", "auto").lower()
    if use_cuda_env == "true":
        use_cuda = True
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("TEST_USE_CUDA is 'true' but CUDA is not available.")
    elif use_cuda_env == "false":
        use_cuda = False
    else:  # 'auto'
        use_cuda = torch.cuda.is_available()

    print(f"  Using CUDA: {use_cuda}")
    results = engine.batch_compute_states(subset, use_cuda=use_cuda)

    trades_simulated = 0

    for res in results:
        state = res['state']
        price = res['price']

        # Basic directive check
        directive = state.get_trade_directive()

        if directive['action'] in ['BUY', 'SELL']:
            # Simulate a trade outcome
            entry_price = price
            # Random win/loss for learning test
            pnl = 20.0 if np.random.random() > 0.4 else -20.0

            outcome = TradeOutcome(
                state=state,
                entry_price=entry_price,
                exit_price=entry_price + (pnl/2.0), # Approximate
                pnl=pnl,
                result='WIN' if pnl > 0 else 'LOSS',
                timestamp=state.timestamp,
                exit_reason='test',
                direction=directive['action']
            )

            brain.update(outcome)
            trades_simulated += 1

    print(f"  Simulated {trades_simulated} trades based on directives.")

    # Summary
    summary = brain.get_summary()
    print(f"\n[OK] Learning Summary:")
    print(f"  Unique states learned: {summary['total_unique_states']}")
    print(f"  Total trades: {summary['total_trades']}")

def test_rolling_cascade_and_context():
    """Test Rolling Window Cascade logic and Multi-Timeframe Context injection"""
    print("\n=== TEST 5: Rolling Cascade & Context Injection ===")

    engine = QuantumFieldEngine(regression_period=21)

    # Create synthetic data with a "Cascade" (large range > 10 in 5 bars)
    # Bars 0-20: flat
    # Bar 21: High=21000, Low=21000
    # Bar 22: High=21000, Low=21000
    # Bar 23: High=21000, Low=21000
    # Bar 24: High=21015, Low=21000 (Range 15 > 10 threshold)

    timestamps = pd.date_range('2025-01-01', periods=30, freq='1s')
    prices = [21000.0] * 30
    highs = [21000.0] * 30
    lows = [21000.0] * 30

    # Induce cascade in last 5 bars
    highs[-1] = 21015.0 # Last bar spikes up

    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'price': prices,
        'high': highs,
        'low': lows,
        'volume': [100.0] * 30,
        'open': prices
    }).set_index('timestamp')

    # Context
    ctx = {
        'daily': TimeframeContext(trend='BULL', volatility='HIGH'),
        'h4': TimeframeContext(trend='UP', session='US')
    }

    # Call calculate_three_body_state
    # Pass last 21 bars as macro, last 21 as micro (contains the cascade at the end)
    df_window = df.iloc[-21:]

    state = engine.calculate_three_body_state(
        df_macro=df_window,
        df_micro=df_window,
        current_price=21000.0,
        current_volume=100.0,
        tick_velocity=0.0,
        context=ctx
    )

    # Verify Cascade Detected
    # Range is 21015 - 21000 = 15 > 10
    print(f"  Cascade Detected: {state.cascade_detected}")
    assert state.cascade_detected == True, "Should detect rolling cascade (range=15)"

    # Verify Context Injection
    print(f"  Daily Trend: {state.daily_trend}")
    print(f"  H4 Session: {state.session}")

    assert state.daily_trend == 'BULL', "Daily trend should be BULL"
    assert state.daily_volatility == 'HIGH', "Daily vol should be HIGH"
    assert state.session == 'US', "Session should be US"

    print("[OK] Rolling Cascade and Context Injection verified")

if __name__ == "__main__":
    print("Bayesian AI v2.0 - Quantum Integration Validation")
    print("=" * 60)

    test_quantum_state()
    test_quantum_brain()
    test_quantum_field_engine()
    test_full_quantum_integration()
    test_rolling_cascade_and_context()

    print("\n" + "=" * 60)
    print("[OK] ALL QUANTUM INTEGRATION TESTS PASSED")
