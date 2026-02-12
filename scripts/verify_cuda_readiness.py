"""
CUDA AI Implementation Readiness Verification Script
Checks PyTorch CUDA availability and validates QuantumFieldEngine on GPU.
"""
import sys
import os
import torch
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_field_engine import QuantumFieldEngine

def verify_cuda_readiness():
    print("="*60)
    print("CUDA AI READINESS VERIFICATION")
    print("="*60)

    # 1. Check PyTorch CUDA
    if torch.cuda.is_available():
        print(f"CUDA Available: YES")
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA Available: NO (Running on CPU)")
        print("  NOTE: This script is running in an environment without a GPU.")
        print("  Real training will require a CUDA-capable GPU for acceleration.")

    # 2. Check QuantumFieldEngine
    print("-" * 60)
    print("Initializing QuantumFieldEngine...")
    engine = QuantumFieldEngine()
    print(f"Engine Device: {engine.device}")
    print(f"Engine Use GPU: {engine.use_gpu}")

    if torch.cuda.is_available() and not engine.use_gpu:
        print("WARNING: CUDA is available but Engine is using CPU!")
    elif not torch.cuda.is_available() and engine.use_gpu:
        print("ERROR: Engine claims to use GPU but CUDA is not available!")
    else:
        print("Engine configuration matches environment.")

    # 3. Run Dummy Batch
    print("-" * 60)
    print("Running Dummy Batch Computation...")

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
        print(f"Successfully computed {len(results)} states.")
        if len(results) > 0:
            print("Sample State Z-Score:", results[0]['state'].z_score)
            print("Sample Tunnel Prob:", results[0]['state'].tunnel_probability)
        print("Batch computation verified.")
    except Exception as e:
        print(f"ERROR during batch computation: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)
    print("VERIFICATION COMPLETE")

if __name__ == "__main__":
    verify_cuda_readiness()
