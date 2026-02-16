"""
CUDA AI Implementation Readiness Verification Script
Checks PyTorch CUDA availability and validates QuantumFieldEngine on GPU.
Logs output to debug_outputs/cuda_readiness.log
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
import logging
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_field_engine import QuantumFieldEngine

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_outputs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'cuda_readiness.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def verify_cuda_readiness():
    logger.info("="*60)
    logger.info("CUDA AI READINESS VERIFICATION")
    logger.info("="*60)

    # 1. Check PyTorch CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: YES")
        device_count = torch.cuda.device_count()
        logger.info(f"Device Count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {props.name}")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            logger.info(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    else:
        logger.info("CUDA Available: NO (Running on CPU)")
        logger.info("  NOTE: This script is running in an environment without a GPU.")
        logger.info("  Real training will require a CUDA-capable GPU for acceleration.")

    # 2. Check QuantumFieldEngine
    logger.info("-" * 60)
    logger.info("Initializing QuantumFieldEngine...")
    try:
        engine = QuantumFieldEngine()
        logger.info(f"Engine Device: {engine.device}")
        logger.info(f"Engine Use GPU: {engine.use_gpu}")

        if torch.cuda.is_available() and not engine.use_gpu:
            logger.info("WARNING: CUDA is available but Engine is using CPU!")
        elif not torch.cuda.is_available() and engine.use_gpu:
            logger.info("ERROR: Engine claims to use GPU but CUDA is not available!")
        else:
            logger.info("Engine configuration matches environment.")

        # 3. Run Dummy Batch
        logger.info("-" * 60)
        logger.info("Running Dummy Batch Computation...")

        # Create dummy data (100 bars)
        df = pd.DataFrame({
            'price': np.random.randn(100) + 100,
            'volume': np.random.rand(100) * 1000,
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='15s')
        })
        # Ensure close exists
        df['close'] = df['price']

        try:
            # Force use_cuda=True if available, else False
            use_cuda = torch.cuda.is_available()
            results = engine.batch_compute_states(df, use_cuda=use_cuda)
            logger.info(f"Successfully computed {len(results)} states.")
            if len(results) > 0:
                logger.info(f"Sample State Z-Score: {results[0]['state'].z_score}")
                logger.info(f"Sample Tunnel Prob: {results[0]['state'].tunnel_probability}")
            logger.info("Batch computation verified.")
        except Exception as e:
            logger.info(f"ERROR during batch computation: {e}")
            logger.info(traceback.format_exc())

    except RuntimeError as e:
        logger.info(f"CRITICAL: QuantumFieldEngine Initialization Failed: {e}")
        logger.info("This is expected if no CUDA device is available and the engine strictly requires one.")
    except Exception as e:
        logger.info(f"UNEXPECTED ERROR: {e}")
        logger.info(traceback.format_exc())

    logger.info("="*60)
    logger.info("VERIFICATION COMPLETE")

if __name__ == "__main__":
    verify_cuda_readiness()
