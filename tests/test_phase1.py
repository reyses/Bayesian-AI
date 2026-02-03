"""
Bayesian AI v2.0 - Phase 1 Integration Test
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
from tests.utils import load_test_data

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
    
    # Load test data
    data = load_test_data()
    print(f"Loaded {len(data)} rows of data.")

    # Resample for different timeframes
    bars_4hr = data.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars_1hr = data.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars_15m = data.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars_5m = data.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    
    # Initialize engine
    try:
        from numba import cuda
        use_gpu = cuda.is_available()
    except:
        use_gpu = False

    engine = LayerEngine(use_gpu=use_gpu)
    engine.initialize_static_context(data, kill_zones=[21500, 21600, 21700])
    
    last_price = data.iloc[-1]['close']
    last_timestamp = data.index[-1].timestamp()

    # Create current snapshot
    current_data = {
        'price': last_price,
        'timestamp': last_timestamp,
        'bars_4hr': bars_4hr.tail(6),
        'bars_1hr': bars_1hr.tail(10),
        'bars_15m': bars_15m.tail(20),
        'bars_5m': bars_5m.tail(20),
        'ticks': data['close'].values[-50:]  # Use last 50 closes as ticks
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
    data = load_test_data()
    bars_4hr = data.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars_1hr = data.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars_15m = data.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    bars_5m = data.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

    try:
        from numba import cuda
        use_gpu = cuda.is_available()
    except:
        use_gpu = False

    engine = LayerEngine(use_gpu=use_gpu)
    engine.initialize_static_context(data, kill_zones=[21500, 21600])
    brain = BayesianBrain()
    
    # Simulate 5 trades
    # We will just iterate over the last 5 data points
    subset = data.tail(5)

    for i in range(5):
        row = subset.iloc[i]
        price = row['close']
        timestamp = subset.index[i].timestamp()

        current_data = {
            'price': price,
            'timestamp': timestamp,
            'bars_4hr': bars_4hr,
            'bars_1hr': bars_1hr,
            'bars_15m': bars_15m,
            'bars_5m': bars_5m,
            'ticks': data['close'].values[-(50+i):-i] if i > 0 else data['close'].values[-50:]
        }
        
        # Compute state
        state = engine.compute_current_state(current_data)
        
        # Simulate trade outcome
        entry_price = price
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
    print("Bayesian AI v2.0 - Phase 1 Validation")
    print("=" * 50)
    
    test_state_vector()
    test_bayesian_brain()
    test_layer_engine()
    test_integration()
    
    print("\n" + "=" * 50)
    print("✓ ALL PHASE 1 TESTS PASSED")
    print("Ready for Phase 2: CUDA optimization")
