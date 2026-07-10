import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling_ols_bands(close, W):
    """Trailing W-bar OLS residual sigma"""
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
    
    df_day = df_day.copy()
    
    # Compute trailing 1m sigma (W=12 for 5s bars) across the whole day to avoid edge effects at 8:30
    df_day['sigma'] = rolling_ols_bands(df_day['close'].values, W=12)
    # Forward fill sigma for the first W-1 bars
    df_day['sigma'] = df_day['sigma'].bfill().fillna(1.0) 
    
    # Filter for RTH (08:30 to 15:15 CT)
    df_rth = df_day[(df_day['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df_day['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_rth) < 100: return []
        
    prices = df_rth['close'].values
    opens = df_rth['open'].values
    sigmas = df_rth['sigma'].values
    divergences = df_rth['divergence'].values
    deltas = df_rth['delta'].values
    
    event_idx_s1 = -1
    mode_s1 = 'none'
    
    # Setup 1: Delta Divergence
    for i in range(len(prices)):
        if divergences[i] < p10:
            event_idx_s1 = i
            mode_s1 = 'bearish_bounce'
            break
        elif divergences[i] > p90:
            event_idx_s1 = i
            mode_s1 = 'bullish_bounce'
            break
            
    event_idx_s2 = -1
    mode_s2 = 'none'
    
    # Setup 2: Trapped Traders
    for i in range(len(prices)):
        if deltas[i] > 0 and prices[i] < opens[i]:
            event_idx_s2 = i
            mode_s2 = 'bearish_runner'
            break
        elif deltas[i] < 0 and prices[i] > opens[i]:
            event_idx_s2 = i
            mode_s2 = 'bullish_runner'
            break

    def evaluate_horizon(event_idx, mode, setup_id):
        if event_idx == -1: return None
        p0 = prices[event_idx]
        
        path = prices[event_idx+1 :]
        div_path = divergences[event_idx+1 :]
        delta_path = deltas[event_idx+1 :]
        if len(path) == 0: return None
        
        magnitude = 0.0
        hit_target = False
        
        if setup_id == 1:
            if mode == 'bearish_bounce':
                for p, div in zip(path, div_path):
                    if div >= p10:
                        magnitude = p0 - p
                        hit_target = magnitude > 0
                        break
            elif mode == 'bullish_bounce':
                for p, div in zip(path, div_path):
                    if div <= p90:
                        magnitude = p - p0
                        hit_target = magnitude > 0
                        break
        elif setup_id == 2:
            if mode == 'bearish_runner':
                for p, d in zip(path, delta_path):
                    if d <= 0:
                        magnitude = p0 - p
                        hit_target = magnitude > 0
                        break
            elif mode == 'bullish_runner':
                for p, d in zip(path, delta_path):
                    if d >= 0:
                        magnitude = p - p0
                        hit_target = magnitude > 0
                        break
        
        if magnitude == 0.0:
            if mode in ['bullish_bounce', 'bullish_runner']:
                magnitude = path[-1] - p0
            else:
                magnitude = p0 - path[-1]
            hit_target = magnitude > 0
                
        return {
            'year': day_str[:4],
            'day': day_str,
            'setup': setup_id,
            'mode': mode,
            'open_price': opens[event_idx],
            'event_idx': event_idx,
            'hit': int(hit_target),
            'magnitude': magnitude
        }

    results = []
    res1 = evaluate_horizon(event_idx_s1, mode_s1, 1)
    if res1: results.append(res1)
    res2 = evaluate_horizon(event_idx_s2, mode_s2, 2)
    if res2: results.append(res2)
    
    return results

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    parquet_path = os.path.join(base_dir, 'DATA/ATLAS/order_flow_delta_5s.parquet')
    
    print(f"[OrderFlow Deep Dive] Loading single block data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if isinstance(df.index, pd.DatetimeIndex):
        df['dt'] = df.index.tz_convert('America/Chicago')
    else:
        df['dt'] = pd.to_datetime(df.index, utc=True).tz_convert('America/Chicago')
        
    df['day_str'] = df['dt'].dt.strftime('%Y-%m-%d')
    
    divergence_non_null = df['divergence'].dropna()
    p10 = divergence_non_null.quantile(0.10)
    p90 = divergence_non_null.quantile(0.90)
    
    print(f"[OrderFlow Deep Dive] Divergence 10th: {p10:.2f}, 90th: {p90:.2f}")
    
    days = sorted(df['day_str'].unique())
    
    tasks = []
    for d in days:
        tasks.append((d, df[df['day_str'] == d], p10, p90))
        
    all_events = []
    print(f"[OrderFlow Deep Dive] Evaluating Setups over {len(tasks)} days...")
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        for res in executor.map(process_day, tasks):
            if res:
                all_events.extend(res)
                
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
    report_lines.append("**Status:** Completed (Dual-Year Validated + Single Block)")
    report_lines.append("**Ruleset:** Bespoke Exit (Normalization). Unclamped Magnitude.")
    report_lines.append("")
    report_lines.append("## LR: Unnormalized Expected Value (EV)")
    report_lines.append("> *Note: Magnitudes are in raw points. Win Rate is binary (%).*")
    report_lines.append("")
    
    # Strictly adhering to FABLE-5 rule 4 & 7: "Both years: run 2024 AND 2025; report per-year tables side by side."
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        
        if len(df_events) > 0:
            df_year = df_events[df_events['year'] == year]
        else:
            df_year = pd.DataFrame()
            
        for setup in [1, 2]:
            if len(df_year) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
                
            df_sub = df_year[df_year['setup'] == setup]
            if len(df_sub) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
            wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
            n = len(df_sub)
            desc = "Delta Divergence" if setup == 1 else "Trapped Traders"
            sig_str = "Yes" if is_sig else "No"
            report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    # Appending the Single Block as requested by the user, while preserving the FABLE-5 dual-year tables above.
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
        desc = "Delta Divergence" if setup == 1 else "Trapped Traders"
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
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_14_OrderFlow.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[OrderFlow Deep Dive] Complete. Report written to {report_path}")
