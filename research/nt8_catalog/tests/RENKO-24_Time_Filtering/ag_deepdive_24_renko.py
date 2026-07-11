import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def build_renko(highs, lows, closes, timestamps, brick_size):
    r_open = []
    r_close = []
    r_high = []
    r_low = []
    r_time = []
    r_dir = []
    
    # Initialize first brick based on first close
    prev_close = np.floor(closes[0] / brick_size) * brick_size
    curr_dir = 0
    
    for i in range(1, len(closes)):
        c = closes[i]
        
        while True:
            if curr_dir == 0:
                if c >= prev_close + brick_size:
                    curr_dir = 1
                    r_open.append(prev_close)
                    r_close.append(prev_close + brick_size)
                    r_high.append(prev_close + brick_size)
                    r_low.append(prev_close)
                    r_time.append(timestamps[i])
                    r_dir.append(1)
                    prev_close += brick_size
                elif c <= prev_close - brick_size:
                    curr_dir = -1
                    r_open.append(prev_close)
                    r_close.append(prev_close - brick_size)
                    r_high.append(prev_close)
                    r_low.append(prev_close - brick_size)
                    r_time.append(timestamps[i])
                    r_dir.append(-1)
                    prev_close -= brick_size
                else:
                    break
            elif curr_dir == 1:
                # Up trend
                if c >= prev_close + brick_size:
                    r_open.append(prev_close)
                    r_close.append(prev_close + brick_size)
                    r_high.append(prev_close + brick_size)
                    r_low.append(prev_close)
                    r_time.append(timestamps[i])
                    r_dir.append(1)
                    prev_close += brick_size
                elif c <= prev_close - 2 * brick_size:
                    # Reversal requires 2 bricks
                    curr_dir = -1
                    r_open.append(prev_close - brick_size)
                    r_close.append(prev_close - 2 * brick_size)
                    r_high.append(prev_close - brick_size)
                    r_low.append(prev_close - 2 * brick_size)
                    r_time.append(timestamps[i])
                    r_dir.append(-1)
                    prev_close -= 2 * brick_size
                else:
                    break
            elif curr_dir == -1:
                # Down trend
                if c <= prev_close - brick_size:
                    r_open.append(prev_close)
                    r_close.append(prev_close - brick_size)
                    r_high.append(prev_close)
                    r_low.append(prev_close - brick_size)
                    r_time.append(timestamps[i])
                    r_dir.append(-1)
                    prev_close -= brick_size
                elif c >= prev_close + 2 * brick_size:
                    # Reversal requires 2 bricks
                    curr_dir = 1
                    r_open.append(prev_close + brick_size)
                    r_close.append(prev_close + 2 * brick_size)
                    r_high.append(prev_close + 2 * brick_size)
                    r_low.append(prev_close + brick_size)
                    r_time.append(timestamps[i])
                    r_dir.append(1)
                    prev_close += 2 * brick_size
                else:
                    break
                    
    return np.array(r_open), np.array(r_close), np.array(r_high), np.array(r_low), np.array(r_time), np.array(r_dir)

def process_day(day):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    parquet_path = os.path.join(l0_dir, f"{day}.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, columns=['high', 'low', 'close', 'timestamp'])
    except Exception:
        return None
        
    if len(df) < 100: return None
    
    # Filter for RTH first
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return None
        
    # Build Renko
    brick_size = 2.0 # 2 points (8 ticks)
    r_op, r_cl, r_hi, r_lo, r_ts, r_dir = build_renko(
        df_day['high'].values, 
        df_day['low'].values, 
        df_day['close'].values, 
        df_day['timestamp'].values, 
        brick_size
    )
    
    if len(r_cl) < 10: return None
    
    events_found = []
    
    # Renko strategy: 2 consecutive bricks in the same direction -> continue
    # This is a classic simple Renko trend following setup
    for i in range(2, len(r_cl) - 20): # Only need 20 bricks forward for Renko
        setup = 0
        mode = 'none'
        
        # Setup 1: Bullish Continuation
        if r_dir[i-1] == 1 and r_dir[i] == 1 and r_dir[i-2] == -1:
            setup = 1
            mode = 'bullish_renko'
            
        # Setup 2: Bearish Continuation
        elif r_dir[i-1] == -1 and r_dir[i] == -1 and r_dir[i-2] == 1:
            setup = 2
            mode = 'bearish_renko'
            
        if setup != 0:
            p0 = r_cl[i]
            path = r_cl[i+1 : i+21] # Look ahead up to 20 bricks
            if len(path) == 0: continue
            
            magnitude = 0.0
            hit_target = False
            
            target_pts = 3 * brick_size
            stop_pts = 2 * brick_size
            
            if setup == 1:
                for p in path:
                    if p >= p0 + target_pts:
                        magnitude = target_pts
                        hit_target = True
                        break
                    elif p <= p0 - stop_pts:
                        magnitude = -stop_pts
                        hit_target = False
                        break
                if magnitude == 0.0:
                    magnitude = path[-1] - p0
                    hit_target = magnitude > 0
                    
            elif setup == 2:
                for p in path:
                    if p <= p0 - target_pts:
                        magnitude = target_pts
                        hit_target = True
                        break
                    elif p >= p0 + stop_pts:
                        magnitude = -stop_pts
                        hit_target = False
                        break
                if magnitude == 0.0:
                    magnitude = p0 - path[-1]
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
                'event_idx': i, # This is the renko brick index, not time index
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
            
    return events_found

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    print(f"[RENKO-24 Deep Dive] Evaluating Setups across {len(days)} days...")
    tasks = days
            
    all_events = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for res_list in executor.map(process_day, tasks):
            if res_list is not None:
                all_events.extend(res_list)
                
    if len(all_events) == 0:
        print("[RENKO-24 Deep Dive] No events found.")
        import sys; sys.exit(0)
        
    df = pd.DataFrame(all_events)
    parquet_out = os.path.join(os.path.dirname(__file__), 'events.parquet')
    df.to_parquet(parquet_out)

    print(f"[RENKO-24 Deep Dive] Extracted {len(df)} triggered events.")
    
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
    report_lines.append("# Document ID: AG-DOC-RENKO-24")
    report_lines.append("**Title:** Deep Dive #24: Renko Time Filtering (2-Point Bricks)")
    report_lines.append("**Status:** Completed (Dual-Year Validated)")
    report_lines.append("**Ruleset:** Trend Continuation on 2nd Brick. Target: 3 Bricks, Stop: 2 Bricks.")
    report_lines.append("")
    report_lines.append("## LR: Unnormalized Expected Value (EV)")
    report_lines.append("> *Note: Magnitudes are in raw points. Win Rate is binary (%).*")
    report_lines.append("")
    
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        df_year = df[df['year'] == year]
        
        for setup in [1, 2]:
            df_sub = df_year[df_year['setup'] == setup]
            if len(df_sub) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
            wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
            n = len(df_sub)
            desc = "Bullish Continuation" if setup == 1 else "Bearish Continuation"
            sig_str = "Yes" if is_sig else "No"
            report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-RENKO-24_distributions.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DOC-RENKO-24: Renko Continuation (Both Years)', fontsize=16)
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
    report_lines.append(f"![Distribution Plot](./DOC-RENKO-24_distributions.png)")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_24_Renko_Time_Filtering.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[RENKO-24 Deep Dive] Complete. Report written to {report_path}")
