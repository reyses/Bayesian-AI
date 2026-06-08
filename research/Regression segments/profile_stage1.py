import os
import sys
import cProfile
import pstats
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from stage1_speed_pass import screen_pipeline_cpu

def profile_cpu_solvers():
    # Generate synthetic data similar to 120 bars of 100 features
    N = 120
    p = 100
    X_raw = np.random.randn(N, p)
    Y = np.random.randn(N)
    
    # Create fake groups (size 3 each)
    groups = np.repeat(np.arange(p // 3 + 1), 3)[:p]
    
    print("Starting cProfile on screen_pipeline_cpu...")
    start = time.time()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    active_idx, best_reg, best_alpha = screen_pipeline_cpu(X_raw, Y, groups)
    
    profiler.disable()
    
    print(f"Total time for 1 block: {time.time() - start:.3f} seconds")
    
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

if __name__ == "__main__":
    profile_cpu_solvers()
