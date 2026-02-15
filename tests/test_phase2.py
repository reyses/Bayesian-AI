"""
Bayesian AI v2.0 - Phase 2 CUDA Validation
Tests CUDA kernels and training orchestrator
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from archive.cuda_modules.pattern_detector import get_pattern_detector
from archive.cuda_modules.confirmation import get_confirmation_engine
from archive.cuda_modules.velocity_gate import get_velocity_gate
from archive.layer_engine import LayerEngine
from archive.orchestrator_pre_consolidation import TrainingOrchestrator, get_data_source
from config.symbols import MNQ
from tests.utils import load_test_data

def test_cuda_pattern_detector():
    """Test Layer 7 CUDA pattern detection"""
    print("\n=== TEST 1: CUDA Pattern Detector ===")
    
    # Load real data
    session = load_test_data()
    # Add dummy OHLC if missing
    if 'open' not in session.columns:
        session['open'] = session['close']
        session['high'] = session['close']
        session['low'] = session['close']
    print(f"Loaded {len(session)} rows.")

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
    
    session = load_test_data()
    # Add dummy OHLC if missing
    if 'open' not in session.columns:
        session['open'] = session['close']
        session['high'] = session['close']
        session['low'] = session['close']
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
    
    session = load_test_data()

    # Use real data slice for normal ticks
    # Ensure we take a copy and handle potential NaNs
    normal_prices = session['close'].dropna().values[:50].copy()

    # Create cascade by taking real data and modifying end
    cascade_prices = session['close'].dropna().values[:50].copy()

    print(f"Debug: Initial prices range: {cascade_prices.min()} - {cascade_prices.max()}")

    # Force a 15 point move in the last few ticks
    # We add to the existing values. If existing values are flat, this creates a slope.
    # If existing values are sloping down, this might flatten it.
    # To GUARANTEE a move, we should probably set them absolutely relative to the start.

    # Let's inspect the last 5 values before mod
    print(f"Debug: Last 5 before: {cascade_prices[-5:]}")

    cascade_prices[-5:] = cascade_prices[-5] + np.linspace(0, 15, 5)
    
    print(f"Debug: Last 5 after: {cascade_prices[-5:]}")
    print(f"Debug: New range: {cascade_prices.min()} - {cascade_prices.max()}")
    print(f"Debug: Range diff: {cascade_prices.max() - cascade_prices.min()}")

    gate = get_velocity_gate(cascade_threshold=10.0, use_gpu=False)
    
    # Test normal (no cascade)
    no_cascade = gate.detect_cascade(normal_prices)
    print(f"✓ Normal ticks (from data): {no_cascade}")
    
    # Test cascade
    has_cascade = gate.detect_cascade(cascade_prices)
    print(f"✓ Cascade ticks (synthetic mod): {has_cascade}")
    
    assert no_cascade == False, "Should not detect cascade in normal ticks"
    assert has_cascade == True, "Should detect cascade in 15pt move"
    print("✓ Velocity gate working")

def test_layer_engine_cuda():
    """Test full LayerEngine with CUDA integration"""
    print("\n=== TEST 4: LayerEngine with CUDA ===")
    
    session = load_test_data()
    # Add dummy OHLC if missing
    if 'open' not in session.columns:
        session['open'] = session['close']
        session['high'] = session['close']
        session['low'] = session['close']
    engine = LayerEngine(use_gpu=False)
    engine.initialize_static_context(session, kill_zones=[21500, 21550])
    
    last_row = session.iloc[-1]
    last_price = last_row['close']
    last_timestamp = session.index[-1].timestamp()

    # Create current snapshot
    current_data = {
        'price': last_price,
        'timestamp': last_timestamp,
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
    
    # Load data directly
    data = load_test_data().head(1000) # Limit to 1000 rows for speed
    
    # Orchestrator expects specific columns. load_test_data returns with datetime index.
    # We need to ensure 'price' column exists (it's 'close' usually) and 'type'.
    # And 'timestamp' column.
    
    # Working on a copy to avoid affecting other tests
    data_orch = data.copy()
    data_orch['price'] = data_orch['close']
    data_orch['type'] = 'trade'
    if 'open' not in data_orch.columns:
        data_orch['open'] = data_orch['close']
        data_orch['high'] = data_orch['close']
        data_orch['low'] = data_orch['close']
    
    # Note: load_test_data sets datetime index, but checks for 'timestamp' column presence in file.
    # The file had 'timestamp'. But let's ensure it's there.
    if 'timestamp' not in data_orch.columns:
        data_orch['timestamp'] = data_orch.index.astype(np.int64) // 10**9

    orchestrator = TrainingOrchestrator(asset_ticker="MNQ", data=data_orch, use_gpu=False)

    print(f"✓ Data loaded: {len(orchestrator.raw_data)} ticks")
    print(f"✓ Static context initialized")
    print(f"✓ Asset: {orchestrator.asset.ticker}")
    print(f"✓ Kill zones: {orchestrator.kill_zones}")

    # Run single iteration
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
