import sys
sys.path.append('.')
from core_v2.FPS.forward_pass_system import ForwardPassSystem

day = '2026_03_20'
atlas_root = 'DATA/ATLAS'
features_root = 'DATA/ATLAS/FEATURES_5s_v2'
labels_csv = 'DATA/ATLAS/regime_labels_2d.csv'

fps = ForwardPassSystem(day=day, atlas_root=atlas_root, features_root=features_root, labels_csv=labels_csv)

print('Iterating through states of FPS where 1m data is available:')
count = 0
for state in fps:
    if state.ohlcv_1m is not None:
        print(f'ts={state.timestamp} 5s_close={state.ohlcv_5s["close"]} 1m_close={state.ohlcv_1m["close"]}')
        count += 1
        if count >= 30:
            break
