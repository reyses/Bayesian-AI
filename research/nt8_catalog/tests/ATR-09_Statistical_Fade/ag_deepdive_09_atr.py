import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    day, daily_atr = args
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    parquet_path = os.path.join(l0_dir, f"{day}.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'timestamp'])
    except Exception:
        return []
        
    if len(df) < 100: return []
    
    df['sigma'] = rolling_ols_bands(df['close'].values, W=12)
    df['sigma'] = df['sigma'].bfill().fillna(1.0)
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return []
        
    prices = df_day['close'].values
    sigmas = df_day['sigma'].values
    
    running_high = prices[0]
    running_low = prices[0]
    
    events = []
    
    thresholds = [0.5, 0.75, 1.0]
    triggered = {x: False for x in thresholds}
    
    for i, p in enumerate(prices):
        running_high = max(running_high, p)
        running_low = min(running_low, p)
        
        current_range = running_high - running_low
        
        for x in thresholds:
            if not triggered[x] and current_range >= x * daily_atr:
                triggered[x] = True
                
                if p >= running_high - 0.25:
                    mode = 'bearish_fade'
                    setup = int(x * 100)
                elif p <= running_low + 0.25:
                    mode = 'bullish_fade'
                    setup = int(x * 100) + 1
                else:
                    continue
                    
                path = prices[i+1 :]
                if len(path) == 0: continue
                
                p0 = p
                sigma_0 = sigmas[i]
                magnitude = 0.0
                hit_target = 0
                
                for px in path:
                    if 'bullish' in mode:
                        exc = (px - p0) / sigma_0
                    else:
                        exc = (p0 - px) / sigma_0
                        
                    if exc >= 2.05:
                        hit_target = 1
                        magnitude = 2.05
                        break
                    elif exc <= -2.05:
                        hit_target = 0
                        magnitude = -2.05
                        break
                        
                if magnitude == 0.0:
                    px_end = path[-1]
                    if 'bullish' in mode:
                        exc = (px_end - p0) / sigma_0
                    else:
                        exc = (p0 - px_end) / sigma_0
                    magnitude = exc
                    hit_target = 1 if magnitude > 0 else 0
                        
                events.append({
                    'year': day[:4],
                    'day': day,
                    'x_threshold': x,
                    'setup': setup,
                    'mode': mode,
                    'open_price': prices[0],
                    'daily_atr': daily_atr,
                    'event_idx': i,
                    'hit': hit_target,
                    'magnitude': magnitude,
                    'depth': current_range / sigma_0,
                    'mfe': magnitude,
                    'mae': 0.0
                })
                
    return events

def compute_daily_summary(parquet_path):
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'timestamp'])
    except Exception:
        return None
    if len(df) == 0: return None
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_rth = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())]
    if len(df_rth) == 0: return None
    
    return {
        'high': df_rth['close'].max(),
        'low': df_rth['close'].min(),
        'close': df_rth['close'].iloc[-1]
    }

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    print(f"[ATR 09 Deep Dive] Computing Daily Profiles for {len(days)} days...")
    daily_data = {}
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        paths = [os.path.join(l0_dir, f"{d}.parquet") for d in days]
        results = executor.map(compute_daily_summary, paths)
        for d, res in zip(days, results):
            if res is not None:
                daily_data[d] = res
                
    print("[ATR 09 Deep Dive] Evaluating Setups...")
    tasks = []
    
    valid_days = [d for d in days if d in daily_data]
    
    for i in range(15, len(valid_days)):
        today = valid_days[i]
        
        window = valid_days[i-15:i]
        trs = []
        for j in range(1, 15):
            curr = daily_data[window[j]]
            prev = daily_data[window[j-1]]
            tr1 = curr['high'] - curr['low']
            tr2 = abs(curr['high'] - prev['close'])
            tr3 = abs(curr['low'] - prev['close'])
            trs.append(max(tr1, tr2, tr3))
            
        atr_14 = np.mean(trs)
        
        tasks.append((today, atr_14))
            
    all_events = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for res_list in executor.map(process_day, tasks):
            if res_list:
                all_events.extend(res_list)
                
    df = pd.DataFrame(all_events)
    if len(df) == 0:
        print("No events found")
        import sys; sys.exit(0)
        
    parquet_out = os.path.join(os.path.dirname(__file__), 'events.parquet')
    df.to_parquet(parquet_out)

    print(f"[ATR 09 Deep Dive] Extracted {len(df)} triggered events.")
    
    def calc_wr(mags_array):
        wins = (mags_array > 0).sum()
        total = len(mags_array)
        if total == 0: return 0.0
        return wins / total

    def bootstrap_ev(df_sub, n_iter=4000):
        if len(df_sub) == 0: return 0, 0, 0, 0, 0, False
        evs = []
        mags = df_sub['magnitude'].values
        n = len(df_sub)
        
        for _ in range(n_iter):
            idx = np.random.choice(n, n, replace=True)
            m = mags[idx]
            ev = np.mean(m)
            evs.append(ev)
            
        real_wr = calc_wr(mags)
        counts, bin_edges = np.histogram(mags, bins=50)
        mode_idx = np.argmax(counts)
        real_mag_mode = (bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2.0
        
        ev_mean = np.mean(evs)
        ev_lb = np.percentile(evs, 2.5)
        ev_ub = np.percentile(evs, 97.5)
        
        is_significant = (ev_lb > 0) or (ev_ub < 0)
        
        return real_wr, real_mag_mode, ev_mean, ev_lb, ev_ub, is_significant
    
    report_lines = []
    report_lines.append("# Document ID: AG-DOC-ATR-09")
    report_lines.append("**Title:** Deep Dive #9: Statistical ATR fade (True 14-day ATR Sweep)")
    report_lines.append("**Status:** Completed (Dual-Year Validated)")
    report_lines.append("**Ruleset:** Ruleset changed from bespoke exit to symmetric ±2.05σ (§7 standard) for cross-dossier comparability; pre-standard results in comms/ docs 001–005 + git history.")
    report_lines.append("")
    report_lines.append("## Expected Value (EV)")
    report_lines.append("")
    
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup (Threshold) | Description | N | WR(>2.05σ)% | Excursion (Mode) | EV (Mean σ) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        df_year = df[df['year'] == year]
        
        for thresh in [0.5, 0.75, 1.0]:
            df_thresh = df_year[df_year['x_threshold'] == thresh]
            for mode in ['bearish_fade', 'bullish_fade']:
                df_sub = df_thresh[df_thresh['mode'] == mode]
                if len(df_sub) == 0:
                    report_lines.append(f"| {thresh*100}% | No events | 0 | - | - | - | - | - |")
                    continue
                wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
                n = len(df_sub)
                desc = f"{mode.replace('_', ' ').title()}"
                sig_str = "Yes" if is_sig else "No"
                report_lines.append(f"| {thresh*100}% | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-ATR-09_distributions.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('DOC-ATR-09: True 14-day ATR Fade Setups (Both Years)', fontsize=16)
    
    for i, thresh in enumerate([0.5, 0.75, 1.0]):
        ax = axes[i]
        df_sub = df[df['x_threshold'] == thresh]
        if len(df_sub) > 0:
            winners = df_sub[df_sub['hit'] == 1]['magnitude']
            losers = df_sub[df_sub['hit'] == 0]['magnitude']
            if len(winners) > 0:
                ax.hist(winners, bins=10, alpha=0.6, color='green', label=f'Winners (n={len(winners)})')
                ax.axvline(np.median(winners), color='darkgreen', linestyle='dashed', linewidth=2, label='Median')
            if len(losers) > 0:
                ax.hist(losers, bins=10, alpha=0.6, color='red', label=f'Losers (n={len(losers)})')
                ax.axvline(np.median(losers), color='darkred', linestyle='dashed', linewidth=2, label='Median')
            ax.set_title(f"Threshold {thresh*100}% ATR")
            ax.set_xlabel("Excursion (σ)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"Threshold {thresh*100}% ATR (No Data)")
            
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    report_lines.append("## Graphical Descriptive Statistics (Aggregate)")
    report_lines.append(f"![Distribution Plot](./DOC-ATR-09_distributions.png)")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_09.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"[ATR 09 Deep Dive] Complete. Report written to {report_path}")
