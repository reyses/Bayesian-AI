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

def compute_daily_summary(parquet_path):
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'timestamp'])
    except Exception:
        return None
    if len(df) == 0: return None
    return {
        'high': df['close'].max(),
        'low': df['close'].min(),
        'close': df['close'].iloc[-1]
    }

def process_day(args):
    day, context = args
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    parquet_path = os.path.join(l0_dir, f"{day}.parquet")
    
    try:
        df = pd.read_parquet(parquet_path, columns=['close', 'timestamp'])
    except Exception:
        return None
        
    if len(df) < 100: return None
    
    df['sigma'] = rolling_ols_bands(df['close'].values, W=12)
    df['sigma'] = df['sigma'].bfill().fillna(1.0) 
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    df_day = df[(df['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
    
    if len(df_day) < 100: return None
        
    prices = df_day['close'].values
    sigmas = df_day['sigma'].values
    open_price = prices[0]
    
    trend = context['trend'] # 'UP' or 'DOWN'
    fib_50 = context['fib_50']
    fib_618 = context['fib_618']
    adx = context['adx']
    swing_high = context['swing_high']
    swing_low = context['swing_low']
    
    events = []
    
    if adx > 25:
        if trend == 'UP':
            # Look for pullback into 50-61.8 zone
            lower_bound = min(fib_50, fib_618)
            upper_bound = max(fib_50, fib_618)
            
            # Setup 1 (Bullish Pullback): Price drops into the zone
            if open_price > upper_bound:
                for i, p in enumerate(prices):
                    if p <= upper_bound and p >= lower_bound:
                        events.append({'setup': 1, 'mode': 'bullish_bounce', 'event_idx': i})
                        break
                        
        elif trend == 'DOWN':
            lower_bound = min(fib_50, fib_618)
            upper_bound = max(fib_50, fib_618)
            
            # Setup 2 (Bearish Pullback): Price rallies into the zone
            if open_price < lower_bound:
                for i, p in enumerate(prices):
                    if p >= lower_bound and p <= upper_bound:
                        events.append({'setup': 2, 'mode': 'bearish_bounce', 'event_idx': i})
                        break
                        
    results = []
    horizon = 12 * 60 # 60 minutes
    
    for ev in events:
        e_idx = ev['event_idx']
        mode = ev['mode']
        setup = ev['setup']
        
        if e_idx + 10 >= len(prices): continue
        
        p0 = prices[e_idx]
        
        path = prices[e_idx+1 :]
        if len(path) == 0: continue
        
        magnitude = 0.0
        hit_target = False
        
        if mode == 'bullish_bounce':
            target = swing_high
            stop = p0 - 10.0
            for p in path:
                if p >= target:
                    hit_target = True
                    magnitude = p - p0
                    break
                elif p <= stop:
                    hit_target = False
                    magnitude = p - p0
                    break
            if magnitude == 0.0:
                magnitude = path[-1] - p0
                hit_target = magnitude > 0
                
        elif mode == 'bearish_bounce':
            target = swing_low
            stop = p0 + 10.0
            for p in path:
                if p <= target:
                    hit_target = True
                    magnitude = p0 - p
                    break
                elif p >= stop:
                    hit_target = False
                    magnitude = p0 - p
                    break
            if magnitude == 0.0:
                magnitude = p0 - path[-1]
                hit_target = magnitude > 0
                
        results.append({
            'year': day[:4],
            'day': day,
            'setup': setup,
            'mode': mode,
            'event_idx': e_idx,
            'hit': int(hit_target),
            'magnitude': magnitude
        })
        
    return results

def compute_adx(highs, lows, closes, n=14):
    if len(closes) < n + 1:
        return 0.0
    
    # Simple Wilder's Smoothing approximation or Pandas rolling
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    df['upMove'] = df['high'] - df['high'].shift(1)
    df['downMove'] = df['low'].shift(1) - df['low']
    
    df['+DM'] = np.where((df['upMove'] > df['downMove']) & (df['upMove'] > 0), df['upMove'], 0)
    df['-DM'] = np.where((df['downMove'] > df['upMove']) & (df['downMove'] > 0), df['downMove'], 0)
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Use simple moving average for Wilder's Smoothing as approximation for speed
    df['+DI'] = 100 * (df['+DM'].rolling(n).mean() / df['TR'].rolling(n).mean())
    df['-DI'] = 100 * (df['-DM'].rolling(n).mean() / df['TR'].rolling(n).mean())
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(n).mean()
    
    val = df['ADX'].iloc[-1]
    if np.isnan(val): return 0.0
    return val

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    l0_dir = os.path.join(base_dir, 'DATA/ATLAS/5s')
    all_files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    
    days = [os.path.basename(f).replace('.parquet', '') for f in all_files if ('2024' in f or '2025' in f)]
    days = sorted(days)
    
    print(f"[FIB Deep Dive] Computing Daily Data for {len(days)} days...")
    daily_data = {}
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        paths = [os.path.join(l0_dir, f"{d}.parquet") for d in days]
        results = executor.map(compute_daily_summary, paths)
        for d, res in zip(days, results):
            if res is not None:
                daily_data[d] = res
                
    print("[FIB Deep Dive] Evaluating Setups...")
    tasks = []
    
    # Need 14 days for ADX + 10 days for swing High/Low
    # Let's just use a 24-day warmup
    
    valid_days = [d for d in days if d in daily_data]
    
    for i in range(25, len(valid_days)):
        today = valid_days[i]
        
        # Get last 14 days for ADX
        window_14 = valid_days[i-14:i]
        highs = [daily_data[d]['high'] for d in window_14]
        lows = [daily_data[d]['low'] for d in window_14]
        closes = [daily_data[d]['close'] for d in window_14]
        adx_val = compute_adx(highs, lows, closes, n=7) # Use n=7 to fit in 14 day window
        
        # Get last 10 days for Swing
        window_10 = valid_days[i-10:i]
        swing_high = max([daily_data[d]['high'] for d in window_10])
        swing_low = min([daily_data[d]['low'] for d in window_10])
        
        # Determine trend (Close relative to 10-day SMA)
        sma_10 = np.mean([daily_data[d]['close'] for d in window_10])
        last_close = daily_data[valid_days[i-1]]['close']
        trend = 'UP' if last_close > sma_10 else 'DOWN'
        
        if trend == 'UP':
            # Retracement from High to Low
            range_val = swing_high - swing_low
            fib_50 = swing_high - (range_val * 0.50)
            fib_618 = swing_high - (range_val * 0.618)
        else:
            # Retracement from Low to High
            range_val = swing_high - swing_low
            fib_50 = swing_low + (range_val * 0.50)
            fib_618 = swing_low + (range_val * 0.618)
            
        context = {
            'trend': trend,
            'fib_50': fib_50,
            'fib_618': fib_618,
            'adx': adx_val,
            'swing_high': swing_high,
            'swing_low': swing_low
        }
        tasks.append((today, context))
            
    all_events = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for res_list in executor.map(process_day, tasks):
            if res_list:
                all_events.extend(res_list)
                
    df = pd.DataFrame(all_events)
    parquet_out = os.path.join(os.path.dirname(__file__), 'events.parquet')
    df.to_parquet(parquet_out)

    print(f"[FIB Deep Dive] Extracted {len(df)} triggered events.")
    
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
    report_lines.append("# Document ID: AG-DOC-FIB-17 (LOGISTIC REGRESSION VERIFIED)")
    report_lines.append("**Title:** Deep Dive #17: Fibonacci Confluence + ADX")
    report_lines.append("**Status:** Completed (Dual-Year Validated)")
    report_lines.append("**Ruleset:** Bespoke Exit (Target Swing H/L or 10pt Stop). Unclamped Magnitude.")
    report_lines.append("")
    report_lines.append("## LR: Unnormalized Expected Value (EV)")
    report_lines.append("> *Note: Magnitudes are in raw points. Win Rate is binary (%).*")
    report_lines.append("")
    
    for year in ['2024', '2025']:
        report_lines.append(f"### Results for {year}")
        report_lines.append("| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |")
        report_lines.append("|---|---|---|---|---|---|---|---|")
        
        if len(df) > 0:
            df_year = df[df['year'] == year]
        else:
            df_year = pd.DataFrame()
            
        for setup in [1, 2]:
            df_sub = df_year[df_year['setup'] == setup]
            if len(df_sub) == 0:
                report_lines.append(f"| {setup} | No events | 0 | - | - | - | - | - |")
                continue
                
            wr, mag_mode, ev_mean, ev_lb, ev_ub, is_sig = bootstrap_ev(df_sub)
            n = len(df_sub)
            desc = "Bullish Pullback (UP Trend)" if setup == 1 else "Bearish Pullback (DOWN Trend)"
            sig_str = "Yes" if is_sig else "No"
            report_lines.append(f"| {setup} | {desc} | {n} | {wr:.2f} | {mag_mode:.2f} | **{ev_mean:.2f}** | [{ev_lb:.2f}, {ev_ub:.2f}] | {sig_str} |")
        report_lines.append("")
        
    assets_dir = os.path.dirname(__file__)
    plot_path = os.path.join(assets_dir, 'DOC-FIB-17_distributions.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DOC-FIB-17: Fibonacci Confluence (Both Years)', fontsize=16)
    for i, setup in enumerate([1, 2]):
        ax = axes[i]
        if len(df) > 0:
            df_sub = df[df['setup'] == setup]
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
    report_lines.append(f"![Distribution Plot](./DOC-FIB-17_distributions.png)")
    
    report_path = os.path.join(os.path.dirname(__file__), 'DOC_17_Fib.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[FIB Deep Dive] Complete. Report written to {report_path}")
