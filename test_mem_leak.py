import os
import psutil
import gc
from core_v2.features import load_features

days = ['2025_01_02', '2025_01_03', '2025_01_06', '2025_01_07', '2025_01_08']
process = psutil.Process(os.getpid())

print("Initial RAM:", process.memory_info().rss / 1024 / 1024, "MB")

for i in range(20):
    for day in days:
        df = load_features([day], root='DATA/ATLAS/FEATURES_5s_v2')
    gc.collect()
    print(f"Iter {i} RAM:", process.memory_info().rss / 1024 / 1024, "MB")
