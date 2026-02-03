"""
Bayesian AI v2.0 - Phase 4 Validation Script
File: bayesian_ai/test_full_system.py
"""
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_core import BayesianEngine
from config.symbols import MNQ
from core.state_vector import StateVector
from tests.utils import load_test_data

def run_validation():
    print("=== BAYESIAN AI V2.0 VALIDATION ===")
    engine = BayesianEngine(MNQ)

    # Initialize session with real data
    print("Loading test data...")
    hist_data = load_test_data()
    engine.initialize_session(hist_data, [21500, 21600])

    # Get baseline from end of history
    last_row = hist_data.iloc[-1]
    base_price = last_row['close']
    base_time = hist_data.index[-1].timestamp()

    print(f"Baseline Price: {base_price}, Time: {base_time}")

    # 1. LOAD: Create a synthetic cascade event starting from baseline
    # Inject 50 flat ticks to satisfy VelocityGate requirement (min 50 ticks)
    ticks = []

    # We advance time slightly from history end
    start_time = base_time + 1.0

    for i in range(50):
        ticks.append({
            'timestamp': start_time + i*0.001,  # High density: 1ms spacing
            'price': base_price,
            'volume': 10,
            'type': 'trade'
        })

    # Add Cascade
    last_tick_time = ticks[-1]['timestamp']
    # Cascade: 15pt drop.
    # Tick 1: start
    ticks.append({'timestamp': last_tick_time + 0.01, 'price': base_price, 'volume': 10, 'type': 'trade'})
    # Tick 2: -5
    ticks.append({'timestamp': last_tick_time + 0.05, 'price': base_price - 5.0, 'volume': 15, 'type': 'trade'})
    # Tick 3: -15 (Total 15pt drop from base in < 0.1s)
    ticks.append({'timestamp': last_tick_time + 0.1, 'price': base_price - 15.0, 'volume': 50, 'type': 'trade'})

    # Mock Probability Engine to ensure fire
    engine.prob_table.get_probability = MagicMock(return_value=0.95)
    engine.prob_table.get_confidence = MagicMock(return_value=1.0)

    # 2. TRANSFORM & ANALYZE: Inject ticks
    print(f"Injecting {len(ticks)} ticks...")

    for i, t in enumerate(ticks):
        # Manually override engine state for validation focus
        engine.on_tick(t)
        if i == len(ticks) - 1:
            print("Last tick processed.")

    # 3. VISUALIZE: Check if WaveRider opened position
    if engine.wave_rider.position:
        pos = engine.wave_rider.position
        print(f"✓ Position Opened: {pos.side} @ {pos.entry_price}")
        print(f"✓ Initial Stop: {pos.stop_loss}")

        # Simulate profit move to trigger adaptive trail
        print("Simulating profit move...")
        # Move price down (assuming short on drop)
        target_price = base_price - 200.0
        exit_tick = {'timestamp': last_tick_time + 10.0, 'price': target_price, 'volume': 10}
        engine.on_tick(exit_tick)

        if engine.wave_rider.position is None:
            print(f"✓ Position Closed. Daily PnL: ${engine.daily_pnl:.2f}")
        else:
            print(f"Position still open at {target_price}.")
            current_pnl = engine.wave_rider.calculate_unrealized_pnl(target_price)
            print(f"Unrealized PnL: ${current_pnl:.2f}")

    else:
        print("✗ Position failed to fire. Check L9/Probability thresholds.")
        # Debug
        current_data = engine.aggregator.get_current_data()
        print(f"Ticks in buffer: {len(current_data['ticks'])}")
        # Check logic manually
        from cuda_modules.velocity_gate import get_velocity_gate
        vg = get_velocity_gate(use_gpu=False)
        is_cascade = vg.detect_cascade(current_data['ticks'])
        print(f"Manual Cascade Check: {is_cascade}")

if __name__ == "__main__":
    run_validation()
