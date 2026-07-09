import os
import glob
import sys
import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

def compute_daily_profile(parquet_path):
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'volume'])
    except Exception:
        return None
        
    if len(df) == 0:
        return None
        
    prices = df['close'].values
    volumes = df['volume'].values
    
    high = np.max(prices)
    low = np.min(prices)
    total_vol = np.sum(volumes)
    
    if total_vol == 0:
        return None
        
    # Bin prices by tick size (0.25 for ES)
    tick_size = 0.25
    bins = np.arange(low, high + tick_size, tick_size)
    if len(bins) < 2:
        return {'high': high, 'low': low, 'poc': low, 'vah': high, 'val': low, 'total_vol': total_vol}
        
    digitized = np.digitize(prices, bins)
    
    vol_by_bin = np.zeros(len(bins))
    for i in range(len(prices)):
        idx = digitized[i] - 1
        if idx >= len(vol_by_bin):
            idx = len(vol_by_bin) - 1
        vol_by_bin[idx] += volumes[i]
        
    poc_idx = np.argmax(vol_by_bin)
    poc = bins[poc_idx]
    
    target_vol = 0.7 * total_vol
    va_vol = vol_by_bin[poc_idx]
    
    up = poc_idx + 1
    down = poc_idx - 1
    
    while va_vol < target_vol:
        vol_up = vol_by_bin[up] if up < len(bins) else -1
        vol_down = vol_by_bin[down] if down >= 0 else -1
        
        if vol_up == -1 and vol_down == -1:
            break
            
        if vol_up > vol_down:
            va_vol += vol_up
            up += 1
        else:
            va_vol += vol_down
            down -= 1
            
    vah = bins[min(up, len(bins)-1)]
    val = bins[max(down, 0)]
    
    return {
        'high': high,
        'low': low,
        'poc': poc,
        'vah': vah,
        'val': val,
        'total_vol': total_vol
    }

def process_day(args):
    day, yest_profile = args
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    parquet_path = os.path.join(l0_dir, f"{day}.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, columns=['close'])
    except Exception:
        return None
        
    prices = df['close'].values
    if len(prices) < 100:
        return None
        
    open_price = prices[0]
    
    # Check Setups
    setup = 0
    mode = 'none'
    event_idx = -1
    
    vh = yest_profile['vah']
    vl = yest_profile['val']
    ph = yest_profile['high']
    pl = yest_profile['low']
    poc = yest_profile['poc']
    
    # Bullish Signal
    if vh < open_price < ph:
        setup = 1
        # Trigger: retraces back to POC
        for i, p in enumerate(prices):
            if p <= poc:
                event_idx = i
                mode = 'bullish_bounce'
                break
                
    # Bearish Signal
    elif pl < open_price < vl:
        setup = 2
        # Trigger: retraces up to POC
        for i, p in enumerate(prices):
            if p >= poc:
                event_idx = i
                mode = 'bearish_bounce'
                break
                
    # Runner (Bullish or Bearish gap)
    elif open_price > ph:
        setup = 3
        event_idx = 0
        mode = 'bullish_runner'
    elif open_price < pl:
        setup = 3
        event_idx = 0
        mode = 'bearish_runner'
        
    if event_idx == -1:
        return None
        
    # Evaluate Response
    horizon = 12 * 60 # 60 minutes in 5s bars
    if event_idx + 10 >= len(prices):
        return None
        
    p0 = prices[event_idx]
    
    # Local sigma over previous 30 mins (360 bars)
    lookback = max(0, event_idx - 360)
    if event_idx > lookback:
        sigma = np.std(np.diff(prices[lookback:event_idx+1]))
    else:
        sigma = 1.0
        
    if sigma == 0 or np.isnan(sigma): sigma = 1.0
    
    path = prices[event_idx+1 : event_idx+1+horizon]
    if len(path) == 0: return None
    
    k = 2.0
    hit_target = False
    magnitude = 0.0
    
    if mode in ['bullish_bounce', 'bullish_runner']:
        tp = p0 + (k * sigma)
        sl = p0 - (k * sigma)
        for p in path:
            if p >= tp:
                hit_target = True
                magnitude = (np.max(path) - p0) / sigma
                break
            elif p <= sl:
                hit_target = False
                magnitude = (np.min(path) - p0) / sigma
                break
        if magnitude == 0.0:
            magnitude = (path[-1] - p0) / sigma
            hit_target = magnitude > 0
            
    elif mode in ['bearish_bounce', 'bearish_runner']:
        tp = p0 - (k * sigma)
        sl = p0 + (k * sigma)
        for p in path:
            if p <= tp:
                hit_target = True
                magnitude = (p0 - np.min(path)) / sigma
                break
            elif p >= sl:
                hit_target = False
                magnitude = (p0 - np.max(path)) / sigma
                break
        if magnitude == 0.0:
            magnitude = (p0 - path[-1]) / sigma
            hit_target = magnitude > 0
            
    return {
        'day': day,
        'setup': setup,
        'mode': mode,
        'open_price': open_price,
        'yest_poc': poc,
        'event_idx': event_idx,
        'hit': int(hit_target),
        'magnitude': magnitude
    }

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if '2024' in f]
    days = sorted(days)
    
    print(f"[VP Deep Dive] Computing Daily Profiles for {len(days)} days...")
    daily_profiles = {}
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        paths = [os.path.join(l0_dir, f"{d}.parquet") for d in days]
        results = executor.map(compute_daily_profile, paths)
        for d, res in zip(days, results):
            if res is not None:
                daily_profiles[d] = res
                
    print("[VP Deep Dive] Evaluating Setups over days...")
    tasks = []
    for i in range(1, len(days)):
        yest = days[i-1]
        today = days[i]
        if yest in daily_profiles:
            tasks.append((today, daily_profiles[yest]))
            
    all_events = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for res in executor.map(process_day, tasks):
            if res is not None:
                all_events.append(res)
                
    df = pd.DataFrame(all_events)
    print(f"[VP Deep Dive] Extracted {len(df)} triggered events.")
    
    def bootstrap_ev(df_sub, n_iter=1000):
        if len(df_sub) == 0: return 0, 0, 0, 0, 0, 0
        evs = []
        hits = df_sub['hit'].values
        mags = df_sub['magnitude'].values
        n = len(df_sub)
        
        for _ in range(n_iter):
            idx = np.random.choice(n, n, replace=True)
            h = hits[idx]
            m = mags[idx]
            
            p_w = h.mean()
            p_l = 1 - p_w
            # Use MEDIAN (bulk) instead of MEAN to ignore massive outliers
            mag_w = np.median(m[h == 1]) if p_w > 0 else 0
            mag_l = np.median(m[h == 0]) if p_l > 0 else 0
            
            ev = (p_w * mag_w) - (p_l * abs(mag_l))
            evs.append(ev)
            
        real_p_w = hits.mean()
        real_mag_w = np.median(mags[hits == 1]) if real_p_w > 0 else 0
        real_mag_l = np.median(mags[hits == 0]) if real_p_w < 1 else 0
            
        return real_p_w, real_mag_w, real_mag_l, np.mean(evs), np.percentile(evs, 2.5), np.percentile(evs, 97.5)
    
    report_lines = []
    report_lines.append("# Deep Dive #1: Volume Profile Trading Strategies")
    report_lines.append("**Source:** `3-volume-profile-trading-strategies.md`")
    report_lines.append("")
    report_lines.append("## PQ: Empirical Expectation (EV)")
    report_lines.append("| Setup | Description | N | Win Rate | Mag(W) | Mag(L) | EV ($\sigma$) | EV 95% CI |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    
    for setup in [1, 2, 3]:
        df_sub = df[df['setup'] == setup]
        if len(df_sub) == 0:
            report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
            continue
            
        wr, mag_w, mag_l, ev_mean, ev_lb, ev_ub = bootstrap_ev(df_sub)
        
        n = len(df_sub)
        desc = "Bullish Bounce" if setup == 1 else "Bearish Bounce" if setup == 2 else "Runner"
        report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2%} | {mag_w:.2f}$\sigma$ | {mag_l:.2f}$\sigma$ | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] |")
        
    # --- Generate Graphical Descriptive Statistics ---
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-VP-01_distributions.png')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('DOC-VP-01: Volume Profile Setup Distributions (Magnitude in $\sigma$)', fontsize=16)
    
    for i, setup in enumerate([1, 2, 3]):
        ax = axes[i]
        df_sub = df[df['setup'] == setup]
        if len(df_sub) > 0:
            winners = df_sub[df_sub['hit'] == 1]['magnitude']
            losers = df_sub[df_sub['hit'] == 0]['magnitude']
            
            if len(winners) > 0:
                ax.hist(winners, bins=10, alpha=0.6, color='green', label=f'Winners (n={len(winners)})')
                ax.axvline(np.median(winners), color='darkgreen', linestyle='dashed', linewidth=2, label='Median')
            if len(losers) > 0:
                ax.hist(losers, bins=10, alpha=0.6, color='red', label=f'Losers (n={len(losers)})')
                ax.axvline(np.median(losers), color='darkred', linestyle='dashed', linewidth=2, label='Median')
                
            ax.set_title(f"Setup {setup}")
            ax.set_xlabel("Magnitude ($\sigma$)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"Setup {setup} (No Data)")
            
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    report_lines.append("")
    report_lines.append("## Graphical Descriptive Statistics")
    report_lines.append(f"![Distribution Plot](file:///{plot_path.replace(os.sep, '/')})")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_VP_01.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[VP Deep Dive] Complete. Report written to {report_path}")
    print(f"[VP Deep Dive] Plot written to {plot_path}")
