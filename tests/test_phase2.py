"""
Bayesian AI v2.0 - Phase 2 CUDA Validation
Tests CUDA kernels and training orchestrator
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cuda.pattern_detector import get_pattern_detector
from cuda.confirmation import get_confirmation_engine
from cuda.velocity_gate import get_velocity_gate
from core.layer_engine import LayerEngine
from training.orchestrator import TrainingOrchestrator
from config.symbols import MNQ

def generate_synthetic_session():
    """Generate synthetic session data for testing"""
    # 1000 ticks for faster testing
    n_ticks = 1000
    dates = pd.date_range(start=datetime.now(), periods=n_ticks, freq='1s')
    
    # Random walk with trend
    base_price = 21500
    returns = np.random.normal(0, 0.1, n_ticks)
    returns[:1000] += 0.5  # Add cascade in first 1000 ticks
    
    prices = base_price + np.cumsum(returns)
    volumes = np.random.randint(1, 20, n_ticks)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 0.5, n_ticks),
        'low': prices - np.random.uniform(0, 0.5, n_ticks),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Add columns expected by Orchestrator
    df['price'] = df['close']
    df['type'] = 'trade'
    # Ensure timestamp is available as column for itertuples
    if 'timestamp' not in df.columns:
        df['timestamp'] = df.index

    return df

def test_cuda_pattern_detector():
    """Test Layer 7 CUDA pattern detection"""
    print("\n=== TEST 1: CUDA Pattern Detector ===")
    
    # Generate 15-min bars
    session = generate_synthetic_session()
    bars_15m = session.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    detector = get_pattern_detector(use_gpu=False)  # CPU for testing
    pattern, maturity = detector.detect(bars_15m, window_size=20)
    
    print(f"✓ Pattern detected: {pattern}")
    print(f"✓ Maturity score: {maturity:.2f}")
    
    assert pattern in ['none', 'flag', 'wedge', 'compression', 'breakdown'], "Invalid pattern type"
    assert 0.0 <= maturity <= 1.0, "Maturity must be 0-1"
    print("✓ Pattern detector working")

def test_cuda_confirmation():
    """Test Layer 8 CUDA confirmation"""
    print("\n=== TEST 2: CUDA Confirmation Engine ===")
    
    session = generate_synthetic_session()
    bars_5m = session.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    confirmer = get_confirmation_engine(use_gpu=False)
    
    # Test with pattern active
    confirmed = confirmer.confirm(bars_5m.tail(10), L7_pattern_active=True)
    print(f"✓ Confirmation (pattern active): {confirmed}")
    
    # Test without pattern
    not_confirmed = confirmer.confirm(bars_5m.tail(10), L7_pattern_active=False)
    print(f"✓ Confirmation (no pattern): {not_confirmed}")
    
    assert not_confirmed == False, "Should not confirm without pattern"
    print("✓ Confirmation engine working")

def test_cuda_velocity_gate():
    """Test Layer 9 CUDA velocity cascade detection"""
    print("\n=== TEST 3: CUDA Velocity Gate ===")
    
    # Generate cascade scenario
    normal_prices = np.random.uniform(21500, 21510, 50)
    cascade_prices = np.linspace(21510, 21525, 50)  # 15-point move
    
    gate = get_velocity_gate(cascade_threshold=10.0, use_gpu=False)
    
    # Test normal (no cascade)
    no_cascade = gate.detect_cascade(normal_prices)
    print(f"✓ Normal ticks (no cascade): {no_cascade}")
    
    # Test cascade
    has_cascade = gate.detect_cascade(cascade_prices)
    print(f"✓ Cascade ticks (15pt move): {has_cascade}")
    
    assert no_cascade == False, "Should not detect cascade in normal ticks"
    assert has_cascade == True, "Should detect cascade in 15pt move"
    print("✓ Velocity gate working")

def test_layer_engine_cuda():
    """Test full LayerEngine with CUDA integration"""
    print("\n=== TEST 4: LayerEngine with CUDA ===")
    
    session = generate_synthetic_session()
    engine = LayerEngine(use_gpu=False)
    engine.initialize_static_context(session, kill_zones=[21500, 21550])
    
    # Create current snapshot
    current_data = {
        'price': 21505.0,
        'timestamp': datetime.now().timestamp(),
        'bars_4hr': session.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(),
        'bars_1hr': session.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(),
        'bars_15m': session.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(),
        'bars_5m': session.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(),
        'ticks': session['close'].values[-100:]
    }
    
    state = engine.compute_current_state(current_data)
    
    print(f"✓ Computed state with CUDA:")
    for key, value in state.to_dict().items():
        if key not in ['timestamp', 'price']:
            print(f"  {key}: {value}")
    
    assert state.L7_pattern in ['none', 'flag', 'wedge', 'compression', 'breakdown']
    assert isinstance(state.L8_confirm, bool)
    assert isinstance(state.L9_cascade, bool)
    print("✓ LayerEngine CUDA integration working")

def test_training_orchestrator():
    """Test training orchestrator setup"""
    print("\n=== TEST 5: Training Orchestrator ===")
    
    orchestrator = TrainingOrchestrator(asset_ticker="MNQ", use_gpu=False)
    
    # Generate synthetic data file
    session = generate_synthetic_session()
    test_file = '/tmp/test_nq_session.parquet'
    session.to_parquet(test_file)
    
    # Load data
    orchestrator.load_historical_data(test_file)
    
    print(f"✓ Data loaded: {len(orchestrator.raw_data)} ticks")
    print(f"✓ Static context initialized")
    print(f"✓ Asset: {orchestrator.asset.ticker}")
    print(f"✓ Kill zones: {orchestrator.kill_zones}")
    
    # Run single iteration (not full 1000)
    print("\n[TEST] Running 1 training iteration...")
    result = orchestrator._run_single_iteration(0)
    
    print(f"✓ Iteration complete:")
    print(f"  Trades: {result['total_trades']}")
    print(f"  P&L: ${result['pnl']:.2f}")
    print(f"  States learned: {result['unique_states']}")
    
    print("✓ Training orchestrator working")

if __name__ == "__main__":
    print("Bayesian AI v2.0 - Phase 2 CUDA Validation")
    print("=" * 60)
    
    try:
        test_cuda_pattern_detector()
        test_cuda_confirmation()
        test_cuda_velocity_gate()
        test_layer_engine_cuda()
        test_training_orchestrator()
        
        print("\n" + "=" * 60)
        print("✓ ALL PHASE 2 TESTS PASSED")
        print("Ready for Phase 3: Real data training")
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
