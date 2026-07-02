import json

with open('DATA/cusp_picks/picks_2024-01-02_multi.json', 'r') as f:
    data = json.load(f)

for i, p in enumerate(data['picks']):
    if p.get('end_pnl_ticks', 0) < 0:
        print(f"Pick {i}: snap={p['snap']}, dir={p['direction']}, MFE={p['mfe_ticks']}, MAE={p['mae_ticks']}, End={p.get('end_pnl_ticks')}")
