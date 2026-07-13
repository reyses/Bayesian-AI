import torch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'research/mamba_zigzag_baseline/pipeline')
from mamba_env import MambaRLTradingEnv
env = MambaRLTradingEnv(atlas_root='DATA/ATLAS', features_root='DATA/ATLAS/FEATURES_5s_v2', labels_csv='DATA/ATLAS/regime_labels_2d.csv', days=['2024_03_04'], seq_len=30)
tss = []
for bar in iter(env.fps):
    if bar.v2_vector is not None:
        tss.append(bar.timestamp)
        if len(tss) == 35:
            break

print("Resetting env...")
state = env.reset()
print(f'ts_day[29] = {tss[29]}, env.current_bar = {env.current_bar.timestamp}')
for i in range(35):
    print(f"{i}: tss={tss[i]}")
