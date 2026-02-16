
import time
import numpy as np
import sys
import os

# Ensure we can import core modules
sys.path.append(os.getcwd())

from core.dynamic_binner import DynamicBinner

def benchmark_transform():
    # Setup data
    np.random.seed(42)
    n_samples = 10000
    z_data = np.random.normal(0, 1, n_samples)
    mom_data = np.random.uniform(-1, 1, n_samples)

    data = {'z_score': z_data, 'momentum': mom_data}

    # Fit binner
    binner = DynamicBinner()
    binner.fit(data)

    print("Binner fitted.")
    print(binner.summary())

    # Benchmark transform
    iterations = 100000
    start_time = time.time()

    # Simulate the access pattern in ThreeBodyQuantumState (scalar calls)
    for i in range(iterations):
        # Generate random values to transform
        z_val = np.random.normal(0, 1)
        mom_val = np.random.uniform(-1, 1)

        binner.transform('z_score', z_val)
        binner.transform('momentum', mom_val)

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nTransform benchmark:")
    print(f"Iterations: {iterations}")
    print(f"Total time: {duration:.4f} seconds")
    print(f"Time per iteration (2 transforms): {duration/iterations*1e6:.2f} microseconds")
    print(f"Transforms per second: {iterations*2/duration:.2f}")

if __name__ == "__main__":
    benchmark_transform()
