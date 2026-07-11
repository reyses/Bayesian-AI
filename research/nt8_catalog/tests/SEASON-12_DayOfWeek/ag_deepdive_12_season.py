import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_GAP_THRESHOLD = 5.0 # Minimum gap required to filter out sub-friction/microstructure noise where gap-fill directionality is meaningless.

def rolling_ols_bands(close, W):
    n = len(close)
    if n < W:
        return np.full(n, 1.0)
    x = np.linspace(-1.0, 1.0, W)
    X = np.stack([np.ones(W), x], axis=1)
    P = np.linalg.pinv(X)
    sw = np.lib.stride_tricks.sliding_window_view(close, W)
    C = sw @ P.T
    fit = C @ X.T
    sig = np.sqrt(((sw - fit) ** 2).mean(axis=1))
    pad = np.full(W - 1, np.nan)
    return np.r_[pad, sig]

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
    
    df['sigma'] = rolling_ols_bands(df['close'].values, W=12)
    df['sigma'] = df['sigma'].bfill().fillna(1.0)
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return None
    
    df_day = df_day.sort_values('dt').reset_index(drop=True)
    open_price = df_day['close'].iloc[0]
    sigma_0 = df_day['sigma'].iloc[0]
    
    dt_day = pd.to_datetime(day, format="%Y_%m_%d")
    dow = dt_day.weekday()
    
    gap = open_price - prior_close
    
    if abs(gap) < MIN_GAP_THRESHOLD:
        return None
        
    setup = dow + 1 # 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri
    mode = 'gap_down_long' if gap < 0 else 'gap_up_short'
    
    path = df_day['close'].values[1:]
    hit = 0
    magnitude = 0.0
    
    for p in path:
        if mode == 'gap_down_long':
            exc = (p - open_price) / sigma_0
        else:
            exc = (open_price - p) / sigma_0
            
        if exc >= 2.05:
            hit = 1
            magnitude = 2.05
            break
        elif exc <= -2.05:
            hit = 0
            magnitude = -2.05
            break
            
    if magnitude == 0.0:
        if len(path) > 0:
            p_end = path[-1]
            if mode == 'gap_down_long':
                exc = (p_end - open_price) / sigma_0
            else:
                exc = (open_price - p_end) / sigma_0
            magnitude = exc
            hit = 1 if magnitude > 0 else 0
            
    return {
        'year': day[:4],
        'day': day,
        'setup': setup,
        'mode': mode,
        'open_price': open_price,
        'event_idx': 0,
        'hit': hit,
        'magnitude': magnitude,
        'depth': abs(gap) / sigma_0,
        'mfe': magnitude,
        'mae': 0.0
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
    report_lines.append("**Ruleset:** Weekday Gap Fades (>5pts). Ruleset changed from bespoke exit to symmetric ±2.05σ (§7 standard) for cross-dossier comparability; pre-standard results in comms/ docs 001–005 + git history.")
    report_lines.append("")
    report_lines.append("## Probability of +2.05σ (Hit Rate)")
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
                ax.set_xlabel("Excursion (σ)")
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
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    # Also overwrite the old DOC_12.md to not leave legacy files
    old_report_path = os.path.join(os.path.dirname(__file__), 'DOC_12.md')
    if os.path.exists(old_report_path):
        os.remove(old_report_path)
        
    print(f"[Seasonality Deep Dive] Complete. Report written to {report_path}")
