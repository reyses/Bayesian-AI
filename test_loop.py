import sys, os
sys.path.append(r'c:\Users\reyse\OneDrive\Desktop\Bayesian-AI')
import importlib.util

spec = importlib.util.spec_from_file_location('m', r'c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\ATR-09_Statistical_Fade\ag_deepdive_09_atr.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

all_days = sorted([f.replace('.parquet', '') for f in os.listdir(r'c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS\5s') if f.endswith('.parquet')])
day_idx = all_days.index('2025_12_01')
yest_profile = None
l0_dir_local = r'c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS\5s'

for offset in range(1, min(6, day_idx + 1)):
    candidate_yest_day = all_days[day_idx - offset]
    candidate_path = os.path.join(l0_dir_local, f"{candidate_yest_day}.parquet")
    temp_prof = m.compute_daily_profile(candidate_path)
    print(f'offset={offset}, day={candidate_yest_day}, temp_prof is None: {temp_prof is None}')
    if temp_prof is not None:
        yest_profile = temp_prof
        break

print("Final yest_profile is None:", yest_profile is None)
