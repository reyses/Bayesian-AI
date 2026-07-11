import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def compute_features(df):
    s = df['close']
    net_change = s.diff(60).abs()
    sum_change = s.diff().abs().rolling(60).sum()
    er = net_change / sum_change
    
    vol = s.diff().abs().rolling(60).mean()
    
    return er, vol

def run_conditioning():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    l0_dir = os.path.join(base_dir, '..', '..', 'DATA', 'ATLAS', '5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    day_features = {}
    print(f"Precomputing features for {len(all_files)} days...")
    for f in all_files:
        day = os.path.basename(f).replace('.parquet', '')
        df = pd.read_parquet(f, columns=['close', 'timestamp'])
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        df_rth = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
        
        # 60m is 720 5s bars
        if len(df_rth) < 720:
            continue
            
        s = df_rth['close']
        net_change = s.diff(720).abs()
        sum_change = s.diff().abs().rolling(720).sum()
        er = net_change / sum_change
        vol = s.diff().abs().rolling(720).mean()
        
        day_features[day] = {
            'er': er.values,
            'vol': vol.values,
        }
    all_er = []
    all_vol = []
    for d in day_features.values():
        all_er.extend(d['er'][~np.isnan(d['er'])])
        all_vol.extend(d['vol'][~np.isnan(d['vol'])])
    
    er_bins = np.nanquantile(all_er, [0.33, 0.66])
    vol_bins = np.nanquantile(all_vol, [0.33, 0.66])
    
    def get_tercile(val, bins):
        if np.isnan(val): return 'mid'
        if val <= bins[0]: return 'low'
        if val > bins[1]: return 'high'
        return 'mid'
        
    events_files = glob.glob(os.path.join(base_dir, 'tests', '**', 'events.parquet'), recursive=True)
    events_files = [f for f in events_files if 'archive' not in f.lower()]
    
    report_lines = []
    report_lines.append("# Document ID: AG-CAT-01-CONDITIONING-SWEEP")
    report_lines.append("**Title:** Phase 4: Multi-Dimensional Conditioning Sweep")
    report_lines.append("**Status:** Audit Completed")
    report_lines.append("")
    report_lines.append("## Objective")
    report_lines.append("Evaluate whether Hour-of-Day, Regime (Efficiency Ratio), Volatility, or Depth conditionally rescues any setups.")
    report_lines.append("")
    
    for ef in events_files:
        dossier = os.path.basename(os.path.dirname(ef))
        try:
            ev_df = pd.read_parquet(ef)
        except:
            continue
            
        if len(ev_df) == 0: continue
        
        er_list = []
        vol_list = []
        hour_list = []
        
        for _, row in ev_df.iterrows():
            day = row['day']
            idx = int(row['event_idx'])
            
            h = 8 + (30 + idx) // 60
            hour_list.append(h)
            
            idx_5s = idx * 12
            if day in day_features and idx_5s < len(day_features[day]['er']):
                e = day_features[day]['er'][idx_5s]
                v = day_features[day]['vol'][idx_5s]
            else:
                e = np.nan
                v = np.nan
            er_list.append(e)
            vol_list.append(v)
            
        ev_df['hour'] = hour_list
        ev_df['er'] = er_list
        ev_df['vol'] = vol_list
        
        ev_df['er_tercile'] = ev_df['er'].apply(lambda x: get_tercile(x, er_bins))
        ev_df['vol_tercile'] = ev_df['vol'].apply(lambda x: get_tercile(x, vol_bins))
        
        has_depth = 'depth' in ev_df.columns
        if has_depth:
            depth_vals = ev_df['depth'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(depth_vals) > 10:
                depth_bins = np.nanquantile(depth_vals, [0.33, 0.66])
                ev_df['depth_tercile'] = ev_df['depth'].apply(lambda x: get_tercile(x, depth_bins))
            else:
                has_depth = False
                
        report_lines.append(f"### Dossier: {dossier}")
        report_lines.append(f"Total Base Events: {len(ev_df)}")
        report_lines.append("")
        
        def render_agg(groupby_col):
            agg = ev_df.groupby(groupby_col).agg(
                N=('hit', 'count'),
                WR=('hit', lambda x: np.mean(x)*100),
                EV=('magnitude', 'mean')
            ).reset_index()
            
            # Find robust edge
            robust = agg[(agg['N'] >= 30) & (agg['EV'] > 0.05) & (agg['WR'] > 55.0)]
            
            lines = [f"**Condition: {groupby_col}**"]
            lines.append(f"| {groupby_col} | N | WR% | EV (σ) |")
            lines.append("|---|---|---|---|")
            for _, r in agg.iterrows():
                lines.append(f"| {r[groupby_col]} | {r['N']} | {r['WR']:.1f}% | {r['EV']:.3f} |")
            lines.append("")
            
            if not robust.empty:
                lines.append("> **POTENTIAL CONDITIONAL EDGE IDENTIFIED:**")
                for _, r in robust.iterrows():
                    lines.append(f"> Subset {r[groupby_col]}: N={r['N']}, EV={r['EV']:.3f}")
                lines.append("")
                
            return lines
            
        report_lines.extend(render_agg('hour'))
        report_lines.extend(render_agg('er_tercile'))
        report_lines.extend(render_agg('vol_tercile'))
        
        if has_depth:
            report_lines.extend(render_agg('depth_tercile'))
            
        report_lines.append("---")
        report_lines.append("")

    out_path = os.path.join(base_dir, 'reports', 'AG_cat_01_CONDITIONING_SWEEP.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"Conditioning sweep written to {out_path}")

if __name__ == '__main__':
    run_conditioning()
