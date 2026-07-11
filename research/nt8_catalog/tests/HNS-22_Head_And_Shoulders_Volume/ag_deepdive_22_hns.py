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
        df = pd.read_parquet(parquet_path, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    except Exception:
        return None
        
    if len(df) < 100: return None
    
    df['sigma'] = rolling_ols_bands(df['close'].values, W=12)
    df['sigma'] = df['sigma'].bfill().fillna(1.0) 
    
    # Filter for RTH
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return None
    
    df_day = df_day.reset_index(drop=True)
    
    # Find Local Peaks (Highs) and Troughs (Lows) with a 21-bar window (10 bars each side)
    df_day['is_peak'] = (df_day['high'] == df_day['high'].rolling(21, center=True).max()) & (df_day['high'] > df_day['high'].shift(1))
    df_day['is_trough'] = (df_day['low'] == df_day['low'].rolling(21, center=True).min()) & (df_day['low'] < df_day['low'].shift(1))
    
    op = df_day['open'].values
    hi = df_day['high'].values
    lo = df_day['low'].values
    cl = df_day['close'].values
    vo = df_day['volume'].values
    sigmas = df_day['sigma'].values
    is_peak = df_day['is_peak'].values
    is_trough = df_day['is_trough'].values
    
    events_found = []
    
    # Store valid peaks and troughs
    peaks = []
    troughs = []
    
    cooldown = 0
    
    # We can only process up to len - 60 (for the outcome window) 
    # AND we must respect the 10-bar lookahead for the peak detection by lagging our signal check.
    # Actually, if we just scan from left to right, we can't 'trade' a peak until 10 bars AFTER it occurs.
    # But a neckline break will usually happen > 10 bars after the right shoulder anyway.
    
    for i in range(10, len(cl) - 60):
        if cooldown > 0:
            cooldown -= 1
            
        # Register peaks/troughs 10 bars ago to avoid lookahead bias
        check_idx = i - 10
        if is_peak[check_idx]:
            peaks.append(check_idx)
        if is_trough[check_idx]:
            troughs.append(check_idx)
            
        if cooldown > 0: continue
            
        if len(peaks) >= 3 and len(troughs) >= 2:
            p3, p2, p1 = peaks[-3], peaks[-2], peaks[-1]
            t2, t1 = troughs[-2], troughs[-1]
            
            # Sequence: p3 (LS) -> t2 -> p2 (Head) -> t1 -> p1 (RS)
            if p3 < t2 < p2 < t1 < p1:
                # Top HNS Geometry
                if hi[p2] > hi[p3] and hi[p2] > hi[p1]: # Head is highest
                    if abs(hi[p3] - hi[p1]) < (hi[p2] - hi[p1]) * 0.5: # Shoulders roughly equal
                        
                        # Volume Divergence Check
                        v_ls = vo[p3-2:p3+3].mean()
                        v_h = vo[p2-2:p2+3].mean()
                        v_rs = vo[p1-2:p1+3].mean()
                        
                        if v_ls > v_h > v_rs:
                            # Neckline is formed by t2 and t1
                            # Slope of neckline
                            dx = t1 - t2
                            dy = lo[t1] - lo[t2]
                            slope = dy / dx if dx > 0 else 0
                            
                            neckline_price = lo[t1] + slope * (i - t1)
                            
                            # Breakout trigger
                            if cl[i-1] >= neckline_price and cl[i] < neckline_price:
                                setup = 1
                                mode = 'hns_breakdown'
                                event_idx = i
                                
                                p0 = cl[event_idx]
                                path = cl[event_idx+1 : event_idx+61]
                                std_path = sigmas[event_idx+1 : event_idx+61]
                                
                                magnitude = 0.0
                                hit_target = False
                                
                                for p, std in zip(path, std_path):
                                    if p <= p0 - 3.0 * std:
                                        magnitude = p0 - p
                                        hit_target = True
                                        break
                                    elif p >= p0 + 3.0 * std:
                                        magnitude = p0 - p
                                        hit_target = False
                                        break
                                if magnitude == 0.0:
                                    magnitude = p0 - path[-1]
                                    hit_target = magnitude > 0
                                
                                mfe = 0.0
                                mae = 0.0
                                try:
                                    exit_price_approx = p0 - magnitude
                                    if hit_target:
                                        idx_candidates = np.where(path <= exit_price_approx + 0.0001)[0]
                                    else:
                                        idx_candidates = np.where(path >= exit_price_approx - 0.0001)[0]
                                        
                                    exit_idx = idx_candidates[0] if len(idx_candidates) > 0 else len(path) - 1
                                    sub_path = path[:exit_idx+1]
                                    
                                    mfe = p0 - np.min(sub_path)
                                    mae = p0 - np.max(sub_path)
                                except Exception:
                                    pass
                                    
                                
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
                                
                                events_found.append({
                                    'year': day[:4],
                                    'day': day,
                                    'setup': setup,
                                    'mode': mode,
                                    'open_price': p0,
                                    'event_idx': event_idx,
                                    'hit': int(hit_target),
                                    'magnitude': magnitude,
                                    'mfe': mfe,
        'resolution_idx': (_exit_idx + (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)) + 1) if ('_exit_idx' in locals() and _exit_idx != -1) else -1,
        'duration_bars': _exit_idx if '_exit_idx' in locals() else -1,
                        'depth': (lambda l: next((abs(float(l[k])) for k in ['magnitude', 'div', 'adx_val', 'z', 'z_val', 'z_score', 'distance', 'gap'] if k in l and l[k] is not None), abs(l.get('p0',0) - l.get('open_price',0)) if 'p0' in l and 'open_price' in l else 0.0))(locals()),
                                    'mae': mae,
                                    'magnitude_sigma': magnitude_sigma,
                                    'mfe_sigma': mfe_sigma,
                                    'mae_sigma': mae_sigma
                                })
                                
                                cooldown = 60
                                peaks.clear() # Reset structure

    return events_found

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    print(f"[HNS-22 Deep Dive] Evaluating Setups across {len(days)} days...")
    tasks = days
            
    all_events = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for res_list in executor.map(process_day, tasks):
            if res_list is not None:
                all_events.extend(res_list)
                
    if len(all_events) == 0:
        print("[HNS-22 Deep Dive] No events found.")
        import sys; sys.exit(0)
        
    df = pd.DataFrame(all_events)
    parquet_out = os.path.join(os.path.dirname(__file__), 'events.parquet')
    df.to_parquet(parquet_out)

    print(f"[HNS-22 Deep Dive] Extracted {len(df)} triggered events.")
    
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
    report_lines.append("# Document ID: AG-DOC-HNS-22")
    report_lines.append("**Title:** Deep Dive #22: Head and Shoulders Volume Divergence")
    report_lines.append("**Status:** Completed (Dual-Year Validated)")
    report_lines.append("**Ruleset:** Neckline Breakdown 3.0$\\sigma$ Target / 3.0$\\sigma$ Stop.")
    report_lines.append("")
    report_lines.append("## LR: Unnormalized Expected Value (EV)")
    report_lines.append("> *Note: Magnitudes are in raw points. Win Rate is binary (%).*")
    report_lines.append("")
    
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        df_year = df[df['year'] == year]
        
        for setup in [1]:
            df_sub = df_year[df_year['setup'] == setup]
            if len(df_sub) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
            wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
            n = len(df_sub)
            desc = "HNS Breakdown"
            sig_str = "Yes" if is_sig else "No"
            report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-HNS-22_distributions.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DOC-HNS-22: HNS Volume Divergence (Both Years)', fontsize=16)
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
            ax.set_xlabel("Magnitude (Raw Points)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"Setup {setup} (No Data)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    report_lines.append("## Graphical Descriptive Statistics (Aggregate)")
    report_lines.append(f"![Distribution Plot](./DOC-HNS-22_distributions.png)")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_22_Head_And_Shoulders_Volume.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[HNS-22 Deep Dive] Complete. Report written to {report_path}")
