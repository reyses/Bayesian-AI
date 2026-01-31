"""
ProjectX v2.0 - Phase 4 Validation Script
File: projectx/test_full_system.py
"""
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine_core import ProjectXEngine
from config.symbols import MNQ
from core.state_vector import StateVector

def run_validation():
    print("=== PROJECTX V2.0 VALIDATION ===")
    engine = ProjectXEngine(MNQ)

    # Initialize session with dummy data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    dummy_hist = pd.DataFrame({
        'open': 21500.0, 'high': 21600.0, 'low': 21400.0, 'close': 21500.0, 'volume': 1000
    }, index=dates)
    engine.initialize_session(dummy_hist, [21500, 21600])

    # 1. LOAD: Create a synthetic cascade event
    # Inject 50 flat ticks to satisfy VelocityGate requirement (min 50 ticks)
    ticks = []
    base_time = 100.0
    for i in range(50):
        ticks.append({
            'timestamp': base_time + i*0.001,  # High density: 1ms spacing
            'price': 21500.0,
            'volume': 10,
            'type': 'trade'
        })

    # Add Cascade
    last_time = ticks[-1]['timestamp']
    ticks.append({'timestamp': last_time + 0.01, 'price': 21500.0, 'volume': 10, 'type': 'trade'})
    ticks.append({'timestamp': last_time + 0.05, 'price': 21495.0, 'volume': 15, 'type': 'trade'})
    ticks.append({'timestamp': last_time + 0.1, 'price': 21485.0, 'volume': 50, 'type': 'trade'}) # Cascade: 15pt drop in 0.1s

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
        print("Simulating $200 profit move...")
        exit_tick = {'timestamp': last_time + 10.0, 'price': 21400.0, 'volume': 10}
        engine.on_tick(exit_tick)

        if engine.wave_rider.position is None:
            print(f"✓ Position Closed. Daily PnL: ${engine.daily_pnl:.2f}")
        else:
            print("Position still open.")
    else:
        print("✗ Position failed to fire. Check L9/Probability thresholds.")
        # Debug
        current_data = engine.aggregator.get_current_data()
        print(f"Ticks in buffer: {len(current_data['ticks'])}")
        # Check logic manually
        from cuda.velocity_gate import get_velocity_gate
        vg = get_velocity_gate(use_gpu=False)
        is_cascade = vg.detect_cascade(current_data['ticks'])
        print(f"Manual Cascade Check: {is_cascade}")

if __name__ == "__main__":
    run_validation()
