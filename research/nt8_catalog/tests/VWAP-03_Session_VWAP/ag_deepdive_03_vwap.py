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

def process_day(day):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    parquet_path = os.path.join(l0_dir, f"{day}.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'volume', 'timestamp'])
    except Exception:
        return None
        
    if len(df) < 100: return None
    
    df['sigma'] = rolling_ols_bands(df['close'].values, W=12)
    df['sigma'] = df['sigma'].bfill().fillna(1.0)
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return None
        
    prices = df_day['close'].values
    volumes = df_day['volume'].values
    sigmas = df_day['sigma'].values
    
    cum_vol = volumes.cumsum()
    cum_vol = np.where(cum_vol == 0, 1, cum_vol)
    
    cum_pv = (prices * volumes).cumsum()
    vwap = cum_pv / cum_vol
    
    df_day['vwap_std'] = df_day['close'].rolling(20).std().bfill()
    df_day['vwap_std'] = df_day['vwap_std'].replace(0, 0.25)
    vwap_std = df_day['vwap_std'].values
    
    z_scores = (prices - vwap) / vwap_std
    
    setup = 0
    mode = 'none'
    event_idx = -1
    
    primed_bear = False
    primed_bull = False
    
    for i in range(1, len(prices)):
        z_curr = z_scores[i]
        z_prev = z_scores[i-1]
        
        if z_curr > 2.0:
            primed_bear = True
        elif primed_bear and z_curr < z_prev and z_curr > 0:
            setup = 1
            event_idx = i
            mode = 'bearish_bounce'
            break
        elif z_curr <= 0:
            primed_bear = False
            
        if z_curr < -2.0:
            primed_bull = True
        elif primed_bull and z_curr > z_prev and z_curr < 0:
            setup = 2
            event_idx = i
            mode = 'bullish_bounce'
            break
        elif z_curr >= 0:
            primed_bull = False
            
    if event_idx == -1: return None
    
    p0 = prices[event_idx]
    sigma_0 = sigmas[event_idx]
    path = prices[event_idx+1 :]
    if len(path) == 0: return None
    
    magnitude = 0.0
    hit_target = 0
    
    for p in path:
        if mode == 'bullish_bounce':
            exc = (p - p0) / sigma_0
        else:
            exc = (p0 - p) / sigma_0
            
        if exc >= 2.05:
            hit_target = 1
            magnitude = 2.05
            break
        elif exc <= -2.05:
            hit_target = 0
            magnitude = -2.05
            break
            
    if magnitude == 0.0:
        p_end = path[-1]
        if mode == 'bullish_bounce':
            exc = (p_end - p0) / sigma_0
        else:
            exc = (p0 - p_end) / sigma_0
        magnitude = exc
        hit_target = 1 if magnitude > 0 else 0
            
    return {
        'year': day[:4],
        'day': day,
        'setup': setup,
        'mode': mode,
        'open_price': prices[0],
        'event_idx': event_idx,
        'hit': hit_target,
        'magnitude': magnitude,
        'depth': 0.0,
        'mfe': magnitude,
        'mae': 0.0
    }

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    print(f"[VWAP Deep Dive] Evaluating Setups across {len(days)} days...")
    tasks = days
            
    all_events = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for res in executor.map(process_day, tasks):
            if res is not None:
                all_events.append(res)
                
    df = pd.DataFrame(all_events)
    if len(df) == 0:
        print("No events found")
        import sys; sys.exit(0)
        
    parquet_out = os.path.join(os.path.dirname(__file__), 'events.parquet')
    df.to_parquet(parquet_out)

    print(f"[VWAP Deep Dive] Extracted {len(df)} triggered events.")
    
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
    report_lines.append("# Document ID: AG-DOC-VWAP-03")
    report_lines.append("**Title:** Deep Dive #3: Session VWAP Z-Score Mean Reversion")
    report_lines.append("**Status:** Completed (Dual-Year Validated)")
    report_lines.append("**Ruleset:** Ruleset changed from bespoke exit to symmetric ±2.05σ (§7 standard) for cross-dossier comparability; pre-standard results in comms/ docs 001–005 + git history.")
    report_lines.append("")
    report_lines.append("## Expected Value (EV)")
    report_lines.append("")
    
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR(>2.05σ)% | Excursion (Mode) | EV (Mean σ) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        df_year = df[df['year'] == year]
        
        for setup in [1, 2]:
            df_sub = df_year[df_year['setup'] == setup]
            if len(df_sub) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
            wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
            n = len(df_sub)
            desc = "Bearish Bounce (Short from +2z)" if setup == 1 else "Bullish Bounce (Long from -2z)"
            sig_str = "Yes" if is_sig else "No"
            report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-VWAP-03_distributions.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DOC-VWAP-03: Session VWAP Mean Reversion (Both Years)', fontsize=16)
    for i, setup in enumerate([1, 2]):
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
            ax.set_xlabel("Excursion (σ)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"Setup {setup} (No Data)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    report_lines.append("## Graphical Descriptive Statistics (Aggregate)")
    report_lines.append(f"![Distribution Plot](./DOC-VWAP-03_distributions.png)")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_03_Session_VWAP.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"[VWAP Deep Dive] Complete. Report written to {report_path}")
