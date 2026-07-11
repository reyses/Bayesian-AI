import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def calc_wr(mags_array):
    wins = (mags_array > 0).sum()
    total = len(mags_array)
    if total == 0: return 0.0
    return wins / total

def bootstrap_ev(df_group, n_iter=4000):
    if len(df_group) == 0: return 0, 0, 0, 0, 0, False
    
    day_mags = df_group.groupby('day')['magnitude'].apply(list).values
    n_days = len(day_mags)
    evs = []
    
    for _ in range(n_iter):
        sampled_idx = np.random.choice(n_days, n_days, replace=True)
        sampled_mags = [m for idx in sampled_idx for m in day_mags[idx]]
        if len(sampled_mags) > 0:
            evs.append(np.mean(sampled_mags))
        else:
            evs.append(0.0)
            
    mags = df_group['magnitude'].dropna().values
    if len(mags) == 0: return 0, 0, 0, 0, 0, False
    
    real_wr = df_group['hit'].mean()
    
    counts, bin_edges = np.histogram(mags, bins=50)
    mode_idx = np.argmax(counts)
    real_mag_mode = (bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2.0
    
    ev_mean = np.mean(mags)
    ev_lb = np.percentile(evs, 2.5)
    ev_ub = np.percentile(evs, 97.5)
    is_significant = (ev_lb > 0) or (ev_ub < 0)
    return real_wr, real_mag_mode, ev_mean, ev_lb, ev_ub, is_significant

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
        try:
            df = pd.read_parquet(f, columns=['close', 'timestamp'])
        except Exception as e:
            continue
        df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        df_rth = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
        
        # Resample to 1min so that event_idx aligns with day_features indices
        df_1m = df_rth.resample('1min', on='dt').agg({'close': 'last'}).reset_index()
        
        if len(df_1m) < 60: # 60 minutes
            continue
            
        s = df_1m['close']
        # Calculate ER over 60 minutes
        net_change = s.diff(60).abs()
        sum_change = s.diff().abs().rolling(60).sum()
        er = net_change / sum_change
        vol = s.diff().abs().rolling(60).mean()
        
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
    
    master_lines = []
    master_lines.append("# Document ID: AG-CAT-00-CONDITIONING-REVISED")
    master_lines.append("**Title:** [REVISED] Phase 4: Master Multi-Dimensional Conditioning Sweep")
    master_lines.append("**Status:** Audit Completed (Restandardized to P0)")
    master_lines.append("")
    master_lines.append("## Objective")
    master_lines.append("Evaluate whether Hour-of-Day, Regime (Efficiency Ratio), Volatility, or Depth conditionally rescues any setups, evaluated against Raw Points and Day-Block Bootstrapped 95% CI.")
    master_lines.append("")
    
    for ef in events_files:
        dossier_dir = os.path.dirname(ef)
        dossier = os.path.basename(dossier_dir)
        if dossier == 'tests': continue
        dossier_id = dossier.split('_')[0] if '_' in dossier else dossier
        
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
            
            if day in day_features and idx < len(day_features[day]['er']):
                e = day_features[day]['er'][idx]
                v = day_features[day]['vol'][idx]
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
        
        dossier_lines = []
        dossier_lines.append(f"# Document ID: COND_{dossier_id}-REVISED")
        dossier_lines.append(f"**Dossier:** {dossier}")
        dossier_lines.append(f"**Total Base Events:** {len(ev_df)}")
        dossier_lines.append("")
        
        master_lines.append(f"### Dossier: {dossier}")
        master_lines.append(f"Total Base Events: {len(ev_df)}")
        master_lines.append("")
        
        def render_agg(groupby_col, df):
            lines = []
            for year in sorted(df['year'].unique()):
                df_year = df[df['year'] == year]
                if len(df_year) == 0: continue
                lines.append(f"#### Year: {year} | Condition: {groupby_col}")
                lines.append(f"| {groupby_col} | N | Resp Freq (%) | Mag (Mode) | EV (Raw Pts) | EV 95% CI (Day-Block) | Sig? |")
                lines.append("|---|---|---|---|---|---|---|")
                
                groups = df_year.groupby(groupby_col)
                for name, group in groups:
                    if len(group) < 30:
                        lines.append(f"| *{name}* | *{len(group)}* | *-* | *-* | *-* | *-* | *Insufficient N* |")
                        continue
                    wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(group)
                    sig_str = "Yes" if is_sig else "No"
                    lines.append(f"| {name} | {len(group)} | {wr*100:.1f}% | {mag_mode:.2f} | **{ev_mean:.3f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
                lines.append("")
            return lines
            
        res_hour = render_agg('hour', ev_df)
        res_er = render_agg('er_tercile', ev_df)
        res_vol = render_agg('vol_tercile', ev_df)
        
        dossier_lines.extend(res_hour)
        dossier_lines.extend(res_er)
        dossier_lines.extend(res_vol)
        
        master_lines.extend(res_hour)
        master_lines.extend(res_er)
        master_lines.extend(res_vol)
        master_lines.append("---")
        master_lines.append("")
        
        dossier_out_path = os.path.join(dossier_dir, f'COND_{dossier_id}.md')
        with open(dossier_out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(dossier_lines))
            
    master_out_path = os.path.join(base_dir, 'reports', 'AG_cat_00_CONDITIONING.md')
    with open(master_out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(master_lines))
    print(f"Master Conditioning sweep written to {master_out_path}")

if __name__ == '__main__':
    run_conditioning()
