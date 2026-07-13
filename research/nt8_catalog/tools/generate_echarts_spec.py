import os
import glob
import json
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORTS = os.path.join(ROOT, "research", "nt8_catalog", "reports")
HORIZONS = os.path.join(REPORTS, "fps_horizons.parquet")
LABELS_DIR = os.path.join(ROOT, "DATA", "ai_cusp_picks")

def main():
    # 1. Load Horizons
    df = pd.read_parquet(HORIZONS)
    orb_mask = df['doss'] == 'ORB-02'
    df.loc[orb_mask, 'entry_ts'] += 1800
    df = df[~df['doss'].isin(['SEASON-12', 'RENKO-24'])]
    
    # 2. Load Labels
    files = glob.glob(os.path.join(LABELS_DIR, "ai_picks_*_multi.json"))
    trades = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if 'trades' in data:
                trades.extend(data['trades'])
    df_labels = pd.DataFrame(trades)
    df_labels['doss'] = 'GOLDEN_LABEL'
    
    # 3. Filter to a portion (e.g., 2 weeks in May 2024)
    start_ts = pd.Timestamp('2024-05-01', tz='UTC').timestamp()
    end_ts = pd.Timestamp('2024-05-14', tz='UTC').timestamp()
    
    df_sub = df[(df['entry_ts'] >= start_ts) & (df['entry_ts'] <= end_ts)].copy()
    labels_sub = df_labels[(df_labels['entry_ts'] >= start_ts) & (df_labels['entry_ts'] <= end_ts)].copy()
    
    # Map dossiers to Y-axis indexes
    dossiers = sorted(df_sub['doss'].unique())
    dossiers.insert(0, 'GOLDEN_LABEL') # Labels at the top/bottom
    
    y_map = {d: i for i, d in enumerate(dossiers)}
    
    series = []
    
    # Add series for each dossier
    for d in dossiers:
        if d == 'GOLDEN_LABEL':
            ts_vals = labels_sub['entry_ts'].values
            symbol_size = 12
            color = 'gold'
        else:
            ts_vals = df_sub[df_sub['doss'] == d]['entry_ts'].values
            symbol_size = 6
            color = '#5470c6'
            
        data_pts = [[float(ts * 1000), y_map[d]] for ts in ts_vals]
        
        series.append({
            "name": d,
            "type": "scatter",
            "symbolSize": symbol_size,
            "itemStyle": {"color": color} if d == 'GOLDEN_LABEL' else {},
            "data": data_pts
        })
        
    spec = {
        "title": {"text": "Catalog Events vs Golden Labels (Portion: May 1-14 2024)"},
        "tooltip": {
            "trigger": "item",
            "formatter": "{a}"
        },
        "grid": {"left": 200, "right": 20, "bottom": 30},
        "xAxis": {
            "type": "time",
            "splitLine": {"show": True}
        },
        "yAxis": {
            "type": "category",
            "data": dossiers,
            "splitLine": {"show": True}
        },
        "series": series
    }
    
    with open(os.path.join(ROOT, 'artifacts', 'echarts_spec.json'), 'w') as f:
        json.dump(spec, f)
        
    print(f"Spec generated at {os.path.join(ROOT, 'artifacts', 'echarts_spec.json')}")

if __name__ == "__main__":
    main()
