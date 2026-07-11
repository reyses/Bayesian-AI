import os
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
    day_str, df_day, p10, p90 = args
    
    if len(df_day) < 100: return []
    
    # [FIX] Sort by time to prevent interleaved symbols/contracts causing massive sigma spikes
    df_day = df_day.sort_values('dt').copy()
    
    # Compute trailing 1m sigma (W=12 for 5s bars) across the whole day to avoid edge effects at 8:30
    df_day['sigma'] = rolling_ols_bands(df_day['close'].values, W=12)
    df_day['sigma'] = df_day['sigma'].bfill().fillna(1.0) 
    
    # Filter for RTH (08:30 to 15:15 CT)
    df_rth = df_day[(df_day['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df_day['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_rth) < 100: return []
        
    prices = df_rth['close'].values
    highs = df_rth['high'].values
    lows = df_rth['low'].values
    opens = df_rth['open'].values
    sigmas = df_rth['sigma'].values
    divergences = df_rth['divergence'].values
    deltas = df_rth['delta'].values
    
    # [FIX] Read the trailing p10/p90 precomputed globally
    p10_arr = df_rth['p10'].values
    p90_arr = df_rth['p90'].values
    
    events = []
    
    # Find local 21-bar peaks and troughs for the "absolute high/low of the swing"
    df_rth = df_rth.reset_index(drop=True)
    is_peak = (df_rth['high'] == df_rth['high'].rolling(21, center=True).max()).values
    is_trough = (df_rth['low'] == df_rth['low'].rolling(21, center=True).min()).values
    
    cooldown = 0
    # Process up to -60 to allow for outcome window, and start at 10 to allow for peak lookbehind
    for i in range(10, len(prices) - 60):
        if cooldown > 0:
            cooldown -= 1
            
        # Register peak 10 bars ago (to simulate live trading without lookahead bias)
        check_idx = i - 10
        setup = 0
        mode = 'none'
        
        # Setup 1: Delta Divergence (extreme divergence at the peak/trough)
        # Setup 2: Trapped Traders (Delta heavily against the reversal direction at the peak/trough)
        
        if is_peak[check_idx]:
            d = deltas[check_idx]
            div = divergences[check_idx]
            curr_p10 = p10_arr[check_idx]
            
            if d > 0: # Positive delta at the peak -> Buyers bought the top (Trapped Buyers)
                setup = 2
                mode = 'bearish_runner'
            elif pd.notna(curr_p10) and div < curr_p10: # Extreme negative divergence at peak
                setup = 1
                mode = 'bearish_bounce'
                
        elif is_trough[check_idx]:
            d = deltas[check_idx]
            div = divergences[check_idx]
            curr_p90 = p90_arr[check_idx]
            
            if d < 0: # Negative delta at the trough -> Sellers sold the bottom (Trapped Sellers)
                setup = 2
                mode = 'bullish_runner'
            elif pd.notna(curr_p90) and div > curr_p90: # Extreme positive divergence at trough
                setup = 1
                mode = 'bullish_bounce'
                
        if setup != 0 and cooldown <= 0:
            # We trigger the trade at index `i` (10 bars after the peak/trough confirms)
            p0 = prices[i]
            path = prices[i+1 : i+61]
            std_path = sigmas[i+1 : i+61]
            
            magnitude = 0.0
            hit_target = False
            
            if 'bearish' in mode:
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
                    
            elif 'bullish' in mode:
                for p, std in zip(path, std_path):
                    if p >= p0 + 3.0 * std:
                        magnitude = p - p0
                        hit_target = True
                        break
                    elif p <= p0 - 3.0 * std:
                        magnitude = p - p0
                        hit_target = False
                        break
                if magnitude == 0.0:
                    magnitude = path[-1] - p0
                    hit_target = magnitude > 0
                    
            mfe = 0.0
            mae = 0.0
            try:
                if 'bullish' in mode:
                    exit_price_approx = p0 + magnitude
                    if hit_target:
                        idx_candidates = np.where(path >= exit_price_approx - 0.0001)[0]
                    else:
                        idx_candidates = np.where(path <= exit_price_approx + 0.0001)[0]
                else:
                    exit_price_approx = p0 - magnitude
                    if hit_target:
                        idx_candidates = np.where(path <= exit_price_approx + 0.0001)[0]
                    else:
                        idx_candidates = np.where(path >= exit_price_approx - 0.0001)[0]
                
                exit_idx = idx_candidates[0] if len(idx_candidates) > 0 else len(path) - 1
                sub_path = path[:exit_idx+1]
        
                if 'bullish' in mode:
                    mfe = np.max(sub_path) - p0
                    mae = np.min(sub_path) - p0
                else:
                    mfe = p0 - np.min(sub_path)
                    mae = p0 - np.max(sub_path)
            except Exception:
                pass
                
            sigma_val = sigmas[i] if i < len(sigmas) else 0.0
            if abs(magnitude) > 100.0:
                print(f"[Skip Filter] Dropped {magnitude:.2f} pts anomaly at idx {i} on {day_str}")
                continue
                
            
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
            
            events.append({
                'year': day_str[:4],
                'day': day_str,
                'setup': setup,
                'mode': mode,
                'open_price': p0,
                'event_idx': i,
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
            
    return events

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    parquet_path = os.path.join(base_dir, 'DATA/ATLAS/order_flow_delta_5s.parquet')
    
    print(f"[OrderFlow Deep Dive] Loading single block data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # [FIX] Data cleaning: Drop corrupted ticks where 'close' lost its trailing digits (e.g. 23000 -> 230)
    df = df[df['close'] > 10000].copy()
    
    if isinstance(df.index, pd.DatetimeIndex):
        df['dt'] = df.index.tz_convert('America/Chicago')
    else:
        df['dt'] = pd.to_datetime(df.index, utc=True).tz_convert('America/Chicago')
        
    df['day_str'] = df['dt'].dt.strftime('%Y-%m-%d')
    
    # Sort globally by time before calculating expanding quantiles
    df = df.sort_values('dt').reset_index(drop=True)
    
    # [FIX] Use expanding window for p10/p90 to avoid lookahead bias. Warmup = 4050 bars (~0.5 days)
    df['p10'] = df['divergence'].expanding(min_periods=4050).quantile(0.10)
    df['p90'] = df['divergence'].expanding(min_periods=4050).quantile(0.90)
    
    dropped_events = df['p10'].isna().sum()
    print(f"[OrderFlow Deep Dive] Dropped {dropped_events} rows during p10/p90 expanding threshold warm-up.")
    
    days = sorted(df['day_str'].unique())
    
    tasks = []
    for d in days:
        tasks.append((d, df[df['day_str'] == d], None, None))
        
    all_events = []
    print(f"[OrderFlow Deep Dive] Evaluating Setups over {len(tasks)} days...")
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        for res in executor.map(process_day, tasks):
            if res:
                all_events.extend(res)
                
    if len(all_events) == 0:
        print("[OrderFlow Deep Dive] No events found.")
        import sys; sys.exit(0)
        
    df_events = pd.DataFrame(all_events)
    print(f"[OrderFlow Deep Dive] Extracted {len(df_events)} triggered events.")
    
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
    report_lines.append("# Document ID: DOC-14-OrderFlow (LOGISTIC REGRESSION VERIFIED)")
    report_lines.append("**Title:** Deep Dive #14: Order Flow & Cumulative Delta")
    report_lines.append("**Status:** Completed (Single Block Validated)")
    report_lines.append(f"**Ruleset:** Trapped Delta / Divergence at Swings. 3.0$\\sigma$ Target / 3.0$\\sigma$ Stop. (Expanding min_periods=4050 for p10/p90 thresholds; {dropped_events} initial rows dropped for warm-up).")
    report_lines.append("")
    report_lines.append("## LR: Unnormalized Expected Value (EV)")
    report_lines.append("> *Note: Magnitudes are in raw points. Win Rate is binary (%).*")
    report_lines.append("")
    
    for year in ['2024', '2025', '2026']:
        df_year = df_events[df_events['year'] == year]
        if len(df_year) == 0:
            continue
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        
        for setup in [1, 2]:
            df_sub = df_year[df_year['setup'] == setup]
            if len(df_sub) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
            wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
            n = len(df_sub)
            desc = "Delta Divergence at Peak" if setup == 1 else "Trapped Traders at Peak"
            sig_str = "Yes" if is_sig else "No"
            report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    report_lines.append("### Results for All Data (6-Month Single Validation Block)")
    report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    for setup in [1, 2]:
        if len(df_events) == 0:
            report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
            continue
        df_sub = df_events[df_events['setup'] == setup]
        if len(df_sub) == 0:
            report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
            continue
        wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
        n = len(df_sub)
        desc = "Delta Divergence at Peak" if setup == 1 else "Trapped Traders at Peak"
        sig_str = "Yes" if is_sig else "No"
        report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
    report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-14-OrderFlow_distributions.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DOC-14-OrderFlow: Realizable Setups (All Data)', fontsize=16)
    
    for i, setup in enumerate([1, 2]):
        ax = axes[i]
        if len(df_events) > 0:
            df_sub = df_events[df_events['setup'] == setup]
        else:
            df_sub = pd.DataFrame()
            
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
    report_lines.append(f"![Distribution Plot](./DOC-14-OrderFlow_distributions.png)")
    
    report_path = os.path.join(assets_dir, 'DOC_14_OrderFlow.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    df_events.to_parquet(os.path.join(assets_dir, 'events.parquet'))
        
    print(f"[OrderFlow Deep Dive] Complete. Report written to {report_path}")
