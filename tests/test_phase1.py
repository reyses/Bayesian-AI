"""
ProjectX v2.0 - Phase 1 Integration Test
Validates StateVector + BayesianBrain + LayerEngine work together
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_vector import StateVector
from core.bayesian_brain import BayesianBrain, TradeOutcome
from core.layer_engine import LayerEngine
from config.symbols import MNQ, calculate_pnl

def generate_test_data():
    """Generate minimal synthetic data for testing"""
    # 90 days of daily OHLC data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    daily_data = pd.DataFrame({
        'open': np.random.uniform(21400, 21600, 90),
        'high': np.random.uniform(21500, 21700, 90),
        'low': np.random.uniform(21300, 21500, 90),
        'close': np.random.uniform(21400, 21600, 90),
        'volume': np.random.randint(10000, 50000, 90)
    }, index=dates)
    
    # Add upward trend to last 30 days
    daily_data.loc[dates[-30]:, 'close'] += np.linspace(0, 200, 30)
    
    return daily_data

def test_state_vector():
    """Test StateVector hashing and equality"""
    print("\n=== TEST 1: StateVector ===")
    
    state1 = StateVector(
        L1_bias='bull',
        L2_regime='trending',
        L3_swing='higher_highs',
        L4_zone='at_killzone',
        L5_trend='up',
        L6_structure='bullish',
        L7_pattern='flag',
        L8_confirm=True,
        L9_cascade=True,
        timestamp=123456.0,
        price=21550.0
    )
    
    state2 = StateVector(
        L1_bias='bull',
        L2_regime='trending',
        L3_swing='higher_highs',
        L4_zone='at_killzone',
        L5_trend='up',
        L6_structure='bullish',
        L7_pattern='flag',
        L8_confirm=True,
        L9_cascade=True,
        timestamp=999999.0,  # Different timestamp
        price=99999.0        # Different price
    )
    
    # Should be equal (timestamp/price not part of hash)
    assert state1 == state2, "States should be equal despite different metadata"
    assert hash(state1) == hash(state2), "Hashes should match"
    
    print("✓ StateVector hashing works correctly")
    print(f"  Hash: {hash(state1)}")
    print(f"  State dict: {state1.to_dict()}")

def test_bayesian_brain():
    """Test Bayesian probability learning"""
    print("\n=== TEST 2: BayesianBrain ===")
    
    brain = BayesianBrain()
    
    # Create test state
    state = StateVector(
        L1_bias='bull', L2_regime='trending', L3_swing='higher_highs',
        L4_zone='at_killzone', L5_trend='up', L6_structure='bullish',
        L7_pattern='flag', L8_confirm=True, L9_cascade=True
    )
    
    # Initial probability (no data)
    initial_prob = brain.get_probability(state)
    assert initial_prob == 0.50, "Initial probability should be 50% (neutral prior)"
    print(f"✓ Initial probability: {initial_prob:.2%}")
    
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
    learned_prob = brain.get_probability(state)
    confidence = brain.get_confidence(state)
    
    print(f"✓ After 12 trades (10W-2L):")
    print(f"  Probability: {learned_prob:.2%}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Should fire (80% threshold): {brain.should_fire(state)}")
    
    # Test statistics
    stats = brain.get_stats(state)
    print(f"✓ Stats: {stats}")

def test_layer_engine():
    """Test LayerEngine computation"""
    print("\n=== TEST 3: LayerEngine ===")
    
    # Generate test data
    daily_data = generate_test_data()
    
    # Initialize engine
    engine = LayerEngine()
    engine.initialize_static_context(daily_data, kill_zones=[21500, 21600, 21700])
    
    # Create current snapshot
    current_data = {
        'price': 21550.0,
        'timestamp': datetime.now().timestamp(),
        'bars_4hr': daily_data.tail(6),
        'bars_1hr': daily_data.tail(10),
        'bars_15m': daily_data.tail(20),
        'bars_5m': daily_data.tail(20),
        'ticks': np.array([21540, 21545, 21550, 21555, 21560] * 10)  # 50 ticks
    }
    
    # Compute state
    state = engine.compute_current_state(current_data)
    
    print(f"✓ Computed state:")
    for key, value in state.to_dict().items():
        if key not in ['timestamp', 'price']:
            print(f"  {key}: {value}")

def test_integration():
    """Test full integration: LayerEngine → StateVector → BayesianBrain"""
    print("\n=== TEST 4: Full Integration ===")
    
    # Setup
    daily_data = generate_test_data()
    engine = LayerEngine()
    engine.initialize_static_context(daily_data, kill_zones=[21500, 21600])
    brain = BayesianBrain()
    
    # Simulate 5 trades
    for i in range(5):
        current_data = {
            'price': 21500 + i * 10,
            'timestamp': datetime.now().timestamp(),
            'bars_4hr': daily_data.tail(6),
            'bars_1hr': daily_data.tail(10),
            'bars_15m': daily_data.tail(20),
            'bars_5m': daily_data.tail(20),
            'ticks': np.random.uniform(21490, 21510, 50)
        }
        
        # Compute state
        state = engine.compute_current_state(current_data)
        
        # Simulate trade outcome
        entry_price = current_data['price']
        exit_price = entry_price + np.random.choice([20, -10])  # Random win/loss
        pnl = calculate_pnl(MNQ, entry_price, exit_price, 'short')
        
        outcome = TradeOutcome(
            state=state,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            result='WIN' if pnl > 0 else 'LOSS',
            timestamp=float(i),
            exit_reason='test'
        )
        
        brain.update(outcome)
        
        print(f"Trade {i+1}: {outcome.result} | PnL: ${pnl:.2f}")
    
    # Summary
    summary = brain.get_summary()
    print(f"\n✓ Learning Summary:")
    print(f"  Unique states learned: {summary['total_unique_states']}")
    print(f"  Total trades: {summary['total_trades']}")

if __name__ == "__main__":
    print("ProjectX v2.0 - Phase 1 Validation")
    print("=" * 50)
    
    test_state_vector()
    test_bayesian_brain()
    test_layer_engine()
    test_integration()
    
    print("\n" + "=" * 50)
    print("✓ ALL PHASE 1 TESTS PASSED")
    print("Ready for Phase 2: CUDA optimization")
