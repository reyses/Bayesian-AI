import numpy as np
import time
from training.orchestrator_worker import _extract_arrays_from_df
import pandas as pd

n = 1000
df = pd.DataFrame({
    'price': np.random.randn(n),
    'timestamp': np.arange(n) * 1e9, # ns
    'z_score': np.random.randn(n),
    'velocity': np.random.randn(n)
})

t0 = time.perf_counter()
for i in range(100):
    prices, timestamps, periods, dampings = _extract_arrays_from_df(df)
t1 = time.perf_counter()

print(f"Original _extract_arrays_from_df time: {t1-t0:.4f}")
