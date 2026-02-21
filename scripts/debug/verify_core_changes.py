import sys
import os
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.append(os.getcwd())

from core.three_body_state import ThreeBodyQuantumState
from core.quantum_field_engine import QuantumFieldEngine

def verify_three_body_state():
    print("Verifying ThreeBodyQuantumState...")
    try:
        # Check if rel_volume exists and has default 1.0
        state = ThreeBodyQuantumState.null_state()
        if not hasattr(state, 'rel_volume'):
            print("FAIL: rel_volume attribute missing in ThreeBodyQuantumState")
            return False
        if state.rel_volume != 1.0:
            print(f"FAIL: rel_volume default is {state.rel_volume}, expected 1.0")
            return False
        print("PASS: ThreeBodyQuantumState has rel_volume")
        return True
    except Exception as e:
        print(f"FAIL: Error verifying ThreeBodyQuantumState: {e}")
        return False

def verify_quantum_field_engine():
    print("Verifying QuantumFieldEngine...")
    try:
        engine = QuantumFieldEngine()

        # Create dummy data
        n = 100
        df = pd.DataFrame({
            'price': np.random.rand(n) + 100,
            'open': np.random.rand(n) + 100,
            'high': np.random.rand(n) + 101,
            'low': np.random.rand(n) + 99,
            'close': np.random.rand(n) + 100,
            'volume': np.random.rand(n) * 1000 + 100, # Ensure non-zero volume
            'timestamp': np.arange(n)
        })

        # Force CPU execution for consistent verification
        engine.use_gpu = False

        results = engine.batch_compute_states(df, use_cuda=False)

        if not results:
            print("FAIL: No results from batch_compute_states")
            return False

        last_result = results[-1]
        state = last_result['state']

        if not hasattr(state, 'rel_volume'):
             print("FAIL: Computed state missing rel_volume")
             return False

        print(f"PASS: Computed state has rel_volume={state.rel_volume}")
        return True

    except Exception as e:
        print(f"FAIL: Error verifying QuantumFieldEngine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_three_body_state() and verify_quantum_field_engine():
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
        sys.exit(1)
