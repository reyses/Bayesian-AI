import sys
import os
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.append(os.getcwd())

from training.fractal_discovery_agent import FractalDiscoveryAgent, PatternEvent
from core.three_body_state import ThreeBodyQuantumState

def verify_agent():
    print("Verifying FractalDiscoveryAgent...")
    try:
        agent = FractalDiscoveryAgent()

        # Create a dummy state
        state = ThreeBodyQuantumState.null_state()

        # Create a PatternEvent
        p = PatternEvent(
            pattern_type='ROCHE_SNAP',
            timestamp=1234567890.0,
            price=100.0,
            z_score=2.5,
            velocity=0.1,
            momentum=5.0,
            coherence=0.8,
            file_source='test.parquet',
            idx=0,
            state=state,
            timeframe='15s'
        )

        chain = agent._build_parent_chain(p)
        entry = chain[0]

        expected_keys = ['rel_volume', 'adx', 'hurst', 'dmi_plus', 'dmi_minus', 'pid', 'osc_coh']
        for key in expected_keys:
            if key not in entry:
                print(f"FAIL: Key '{key}' missing in parent chain entry")
                return False

        print("PASS: Parent chain has all required keys")
        return True

    except Exception as e:
        print(f"FAIL: Error verifying agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_agent():
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
        sys.exit(1)
