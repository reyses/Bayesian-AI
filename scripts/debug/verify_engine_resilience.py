
import pandas as pd
import numpy as np
from core.quantum_field_engine import QuantumFieldEngine

def test_key_error():
    engine = QuantumFieldEngine()
    # Create dataframe with 'high' but missing 'open'
    df = pd.DataFrame({
        'price': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100)
    })
    # 'open' is missing

    try:
        engine.batch_compute_states(df, use_cuda=False)
        print("Success!")
    except KeyError as e:
        print(f"Caught expected KeyError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

if __name__ == "__main__":
    test_key_error()
