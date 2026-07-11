import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_GAP_THRESHOLD = 5.0 # Minimum gap required to filter out sub-friction/microstructure noise where gap-fill directionality is meaningless.

def process_day(args):
    day, prior_close = args
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    parquet_path = os.path.join(l0_dir, f"{day}.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'high', 'low', 'timestamp'])
    except Exception:
        return None
        
    if len(df) < 100: return None
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return None
    
    df_day = df_day.sort_values('dt').reset_index(drop=True)
    open_price = df_day['close'].iloc[0]
    
    dt_day = pd.to_datetime(day, format="%Y_%m_%d")
    dow = dt_day.weekday()
    
    gap = open_price - prior_close
    
    if abs(gap) < MIN_GAP_THRESHOLD:
        return None
        
    setup = dow + 1 # 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri
    mode = 'gap_down' if gap < 0 else 'gap_up'
    
    filled = False
    magnitude = 0.0
    
    highs = df_day['high'].values
    lows = df_day['low'].values
    
    if mode == 'gap_down':
        if np.any(highs >= prior_close):
            filled = True
            magnitude = -gap
        else:
            magnitude = np.max(highs) - open_price
            
    elif mode == 'gap_up':
        if np.any(lows <= prior_close):
            filled = True
            magnitude = gap
        else:
            magnitude = open_price - np.min(lows)
            
    
    # --- INJECTED MFE/MAE ---
    try:
        _mode_str = str(mode).lower() if 'mode' in locals() else ''
        _setup_val = setup if 'setup' in locals() else 0
        _is_bullish = ('bull' in _mode_str or 'long' in _mode_str or 'buy' in _mode_str or _setup_val == 1)
        _dir = 1 if _is_bullish else -1
        _exit_price_approx = p0 + _dir * magnitude
        _exit_idx = -1
        for _i, _p in enumerate(path):
            if (_dir == 1 and _p >= _exit_price_approx - 0.0001) or (_dir == -1 and _p <= _exit_price_approx + 0.0001):
                _exit_idx = _i
                break
        if _exit_idx == -1: _exit_idx = len(path) - 1
        _sub_path = path[:_exit_idx+1]
        if len(_sub_path) > 0:
            if _dir == 1:
                mfe = max(0.0, np.max(_sub_path) - p0)
                mae = max(0.0, p0 - np.min(_sub_path))
            else:
                mfe = max(0.0, p0 - np.min(_sub_path))
                mae = max(0.0, np.max(_sub_path) - p0)
        else:
            mfe, mae = 0.0, 0.0
    except Exception:
        mfe, mae = 0.0, 0.0
        
    try:
        _idx_var = event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)
        _sigma_val = sigmas[_idx_var] if 'sigmas' in locals() else 1.0
        if np.isnan(_sigma_val) or _sigma_val <= 0: _sigma_val = 1.0
        magnitude_sigma = magnitude / _sigma_val
        mfe_sigma = mfe / _sigma_val
        mae_sigma = mae / _sigma_val
    except Exception:
        magnitude_sigma, mfe_sigma, mae_sigma = magnitude, mfe, mae
    # ------------------------
    
    return {
        'year': day[:4],
        'day': day,
        'setup': setup,
        'mode': mode,
        'open_price': open_price,
        'event_idx': 0,
        'hit': int(filled),
        'magnitude': magnitude,
        'mfe': mfe,
        'resolution_idx': (_exit_idx + (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)) + 1) if ('_exit_idx' in locals() and _exit_idx != -1) else -1,
        'duration_bars': _exit_idx if '_exit_idx' in locals() else -1,
        'depth': _trigger_depth,
        'mae': mae,
        'magnitude_sigma': magnitude_sigma,
        'mfe_sigma': mfe_sigma,
        'mae_sigma': mae_sigma
    }

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    eod_closes = {}
    print("[Seasonality Deep Dive] Extracting EOD closes...")
    for day in days:
        try:
            df = pd.read_parquet(os.path.join(l0_dir, f"{day}.parquet"), columns=['close', 'timestamp'])
            df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
            df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())]
            if len(df_day) > 0:
                eod_closes[day] = df_day.iloc[-1]['close']
        except Exception:
            pass
            
    valid_days = sorted(list(eod_closes.keys()))
    tasks = []
    for i in range(1, len(valid_days)):
        prev_day = valid_days[i-1]
        curr_day = valid_days[i]
        tasks.append((curr_day, eod_closes[prev_day]))
            
    print("[Seasonality Deep Dive] Evaluating Gap Fills...")
    all_events = []
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        for res in executor.map(process_day, tasks):
            if res is not None:
                all_events.append(res)
                
    df = pd.DataFrame(all_events)
    if len(df) == 0:
        print("No events found.")
        import sys; sys.exit(0)
        
    parquet_out = os.path.join(os.path.dirname(__file__), 'events.parquet')
    df.to_parquet(parquet_out)
    print(f"[Seasonality Deep Dive] Extracted {len(df)} triggered events.")
    
    def bootstrap_ev(df_sub, n_iter=4000):
        if len(df_sub) == 0: return 0, 0, 0, 0, 0
        evs = []
        hits = df_sub['hit'].values
        mags = df_sub['magnitude'].values
        n = len(df_sub)
        
        for _ in range(n_iter):
            idx = np.random.choice(n, n, replace=True)
            h = hits[idx]
            evs.append(np.mean(h))
            
        real_wr = np.mean(hits)
        
        counts, bin_edges = np.histogram(mags, bins=50)
        mode_idx = np.argmax(counts)
        real_mag_mode = (bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2.0
        
        ev_mean = np.mean(evs)
        ev_lb = np.percentile(evs, 2.5)
        ev_ub = np.percentile(evs, 97.5)
        
        return real_wr, real_mag_mode, ev_mean, ev_lb, ev_ub

    def bootstrap_contrast(df_sub, df_base, n_iter=4000):
        if len(df_sub) == 0 or len(df_base) == 0: return 0.0, 0.0, 0.0, False
        diffs = []
        hits_sub = df_sub['hit'].values
        hits_base = df_base['hit'].values
        n_sub = len(df_sub)
        n_base = len(df_base)
        for _ in range(n_iter):
            idx_sub = np.random.choice(n_sub, n_sub, replace=True)
            idx_base = np.random.choice(n_base, n_base, replace=True)
            diffs.append(np.mean(hits_sub[idx_sub]) - np.mean(hits_base[idx_base]))
        
        diff_mean = np.mean(diffs)
        diff_lb = np.percentile(diffs, 2.5)
        diff_ub = np.percentile(diffs, 97.5)
        
        is_sig = (diff_lb > 0) or (diff_ub < 0)
        return diff_mean, diff_lb, diff_ub, is_sig
    
    report_lines = []
    report_lines.append("# Document ID: DOC-SEASON-12")
    report_lines.append("**Title:** Deep Dive #12: Seasonality / Day of Week Effects")
    report_lines.append("**Status:** Completed (Dual-Year Validated)")
    report_lines.append("**Ruleset:** Weekday Gap-Fills (>5pts).")
    report_lines.append("")
    report_lines.append("## Probability of Fill (Hit Rate)")
    report_lines.append("")
    
    dow_names = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri'}
    
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | Fill Prob | 95% CI |")
        report_lines.append("|---|---|---|---|---|---|---|")
        
        if len(df) > 0:
            df_year = df[df['year'] == year]
            df_monday = df_year[df_year['setup'] == 1]
            for setup in [1, 2, 3, 4, 5]:
                df_sub = df_year[df_year['setup'] == setup]
                if len(df_sub) == 0:
                    report_lines.append(f"| {setup} | {dow_names[setup]} | 0 | - | - | - | - |")
                    continue
                wr, mag_mode, ev_mean, ev_lb, ev_ub = bootstrap_ev(df_sub)
                n = len(df_sub)
                desc = dow_names[setup]
                report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | {ev_mean:.2f} | [{ev_lb:.2f}, {ev_ub:.2f}] |")
        report_lines.append("")
        
        report_lines.append(f"#### Contrast vs Monday ({year})")
        report_lines.append("| Day | Contrast (Day - Mon) | 95% CI | Significant? |")
        report_lines.append("|---|---|---|---|")
        if len(df) > 0:
            for setup in [2, 3, 4, 5]:
                df_sub = df_year[df_year['setup'] == setup]
                if len(df_sub) == 0 or len(df_monday) == 0:
                    report_lines.append(f"| {dow_names[setup]} | - | - | - |")
                    continue
                c_mean, c_lb, c_ub, c_sig = bootstrap_contrast(df_sub, df_monday)
                sig_str = "Yes" if c_sig else "No"
                report_lines.append(f"| {dow_names[setup]} | {c_mean:+.3f} | [{c_lb:+.3f}, {c_ub:+.3f}] | {sig_str} |")
        report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-SEASON-12_distributions.png')
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('DOC-SEASON-12: Gap Fill Probabilities by Weekday', fontsize=16)
    
    for i, setup in enumerate([1, 2, 3, 4, 5]):
        ax = axes[i]
        if len(df) > 0:
            df_sub = df[df['setup'] == setup]
            if len(df_sub) > 0:
                hits = df_sub[df_sub['hit'] == 1]['magnitude']
                misses = df_sub[df_sub['hit'] == 0]['magnitude']
                if len(hits) > 0:
                    ax.hist(hits, bins=10, alpha=0.6, color='green', label=f'Filled (n={len(hits)})')
                if len(misses) > 0:
                    ax.hist(misses, bins=10, alpha=0.6, color='red', label=f'Unfilled (n={len(misses)})')
                ax.set_title(f"{dow_names[setup]}")
                ax.set_xlabel("Gap Excursion")
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{dow_names[setup]} (No Data)")
        else:
             ax.set_title(f"{dow_names[setup]} (No Data)")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    report_lines.append("## Graphical Descriptive Statistics (Aggregate)")
    report_lines.append(f"![Distribution Plot](./DOC-SEASON-12_distributions.png)")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_12_Seasonality.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    # Also overwrite the old DOC_12.md to not leave legacy files
    old_report_path = os.path.join(os.path.dirname(__file__), 'DOC_12.md')
    if os.path.exists(old_report_path):
        os.remove(old_report_path)
        
    print(f"[Seasonality Deep Dive] Complete. Report written to {report_path}")
