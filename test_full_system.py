"""
ProjectX v2.0 - Phase 4 Validation Script
File: projectx/test_full_system.py
"""
import numpy as np
import pandas as pd
from engine_core import ProjectXEngine
from config.symbols import MNQ
from core.state_vector import StateVector

def run_validation():
    print("=== PROJECTX V2.0 VALIDATION ===")
    engine = ProjectXEngine(MNQ)
    
    # 1. LOAD: Create a synthetic cascade event
    # [Price, Vol, Velocity, Density]
    ticks = [
        {'timestamp': 100.0, 'price': 21500.0, 'volume': 10, 'type': 'trade'},
        {'timestamp': 100.1, 'price': 21495.0, 'volume': 15, 'type': 'trade'},
        {'timestamp': 100.2, 'price': 21485.0, 'volume': 50, 'type': 'trade'} # Cascade
    ]
    
    # Force a high-probability state for testing
    test_state = StateVector.null_state()
    # Mock some historical wins for this state [cite: 13, 14, 15]
    for _ in range(20):
        from core.bayesian_brain import TradeOutcome
        engine.prob_table.update(TradeOutcome(
            state=test_state, entry_price=0, exit_price=0, 
            pnl=50.0, result='WIN', timestamp=0, exit_reason='test'
        ))

    # 2. TRANSFORM & ANALYZE: Inject ticks
    print(f"Initial Table Size: {len(engine.prob_table.table)} states [cite: 17]")
    
    for t in ticks:
        # Manually override engine state for validation focus
        engine.on_tick(t)
        
    # 3. VISUALIZE: Check if WaveRider opened position [cite: 27, 28]
    if engine.wave_rider.position:
        pos = engine.wave_rider.position
        print(f"✓ Position Opened: {pos.side} @ {pos.entry_price}")
        print(f"✓ Initial Stop: {pos.stop_loss}")
        
        # Simulate profit move to trigger adaptive trail 
        print("Simulating $200 profit move...")
        exit_tick = {'timestamp': 105.0, 'price': 21400.0, 'volume': 10}
        engine.on_tick(exit_tick)
        
        if engine.wave_rider.position is None:
            print(f"✓ Position Closed. Daily PnL: ${engine.daily_pnl:.2f} [cite: 49]")
    else:
        print("✗ Position failed to fire. Check L9/Probability thresholds.")

if __name__ == "__main__":
    run_validation()