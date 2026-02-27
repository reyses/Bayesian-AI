
import logging
import argparse
import pandas as pd
import numpy as np
from core.quantum_field_engine import QuantumFieldEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info("Success! Engine handled missing columns gracefully.")
    except KeyError as e:
        logger.info(f"Caught expected KeyError: {e}")
    except Exception as e:
        logger.error(f"Caught unexpected exception: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify QuantumFieldEngine resilience to missing columns.")
    args = parser.parse_args()

    test_key_error()
