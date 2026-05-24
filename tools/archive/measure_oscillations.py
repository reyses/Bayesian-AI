"""
Measure natural oscillation at every timeframe.

For each TF, identifies:
  - Oscillation period (bars between consecutive troughs)
  - Oscillation amplitude (ticks from trough to peak)
  - Persistence (how many cycles before the pattern breaks)
  - How oscillations at each TF nest inside higher TFs

Output: reports/findings/oscillation_measurements.txt + PNG chart

Usage:
  python -m tools.measure_oscillations
"""
import gc
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TICK = 0.25
ATLAS_ROOT = 'DATA/ATLAS'


def detrend(prices, window=30):
    """Remove trend via rolling mean, return oscillation component."""
    trend = pd.Series(prices).rolling(window, min_periods=1, center=True).mean().values
    return prices - trend, trend


def measure_oscillation(prices, timestamps, tf_name, order=8):
    """Measure oscillation characteristics from price data.

    Returns dict with period, amplitude, persistence stats.
    """
    n = len(prices)
    if n < 10:
        return None

    # Detrend: remove rolling mean to isolate oscillation
    # Window scales with TF (larger TF = larger detrend window)
    detrend_windows = {'1s': 60, '5s': 40, '15s': 30, '1m': 30, '5m': 20, '15m': 20, '1h': 10}
    window = detrend_windows.get(tf_name, 30)
    osc, trend = detrend(prices, window=window)

    # Find peaks and troughs in the detrended oscillation
    # Order = minimum bars between peaks (scales with TF)
    peak_orders = {'1s': 5, '5s': 4, '15s': 4, '1m': 4, '5m': 3, '15m': 3, '1h': 2}
    peak_order = peak_orders.get(tf_name, 4)

    peaks = argrelextrema(osc, np.greater, order=peak_order)[0]
    troughs = argrelextrema(osc, np.less, order=peak_order)[0]

    if len(peaks) < 3 or len(troughs) < 3:
        return None

    # Period: bars between consecutive peaks
    peak_periods = np.diff(peaks)
    trough_periods = np.diff(troughs)
    all_periods = np.concatenate([peak_periods, trough_periods])

    # Amplitude: distance from trough to next peak (in ticks)
    amplitudes = []
    for i in range(len(troughs) - 1):
        t_idx = troughs[i]
        # Find next peak after this trough
        next_peaks = peaks[peaks > t_idx]
        if len(next_peaks) > 0:
            p_idx = next_peaks[0]
            amp = abs(prices[p_idx] - prices[t_idx]) / TICK
            amplitudes.append(amp)
    amplitudes = np.array(amplitudes)

    # Half-cycle: bars from trough to next peak
    half_cycles = []
    for i in range(len(troughs)):
        t_idx = troughs[i]
        next_peaks = peaks[peaks > t_idx]
        if len(next_peaks) > 0:
            half_cycles.append(next_peaks[0] - t_idx)
    half_cycles = np.array(half_cycles)

    # Time units
    tf_seconds = {'1s': 1, '5s': 5, '15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    bar_sec = tf_seconds.get(tf_name, 60)

    result = {
        'tf': tf_name,
        'n_bars': n,
        'n_peaks': len(peaks),
        'n_troughs': len(troughs),
        'period_bars_mean': all_periods.mean(),
        'period_bars_median': np.median(all_periods),
        'period_bars_std': all_periods.std(),
        'period_bars_p25': np.percentile(all_periods, 25),
        'period_bars_p75': np.percentile(all_periods, 75),
        'period_seconds': all_periods.mean() * bar_sec,
        'period_minutes': all_periods.mean() * bar_sec / 60,
        'half_cycle_bars_mean': half_cycles.mean() if len(half_cycles) > 0 else 0,
        'half_cycle_bars_median': np.median(half_cycles) if len(half_cycles) > 0 else 0,
        'half_cycle_seconds': half_cycles.mean() * bar_sec if len(half_cycles) > 0 else 0,
        'amplitude_ticks_mean': amplitudes.mean() if len(amplitudes) > 0 else 0,
        'amplitude_ticks_median': np.median(amplitudes) if len(amplitudes) > 0 else 0,
        'amplitude_ticks_std': amplitudes.std() if len(amplitudes) > 0 else 0,
        'amplitude_dollars': amplitudes.mean() * 0.5 if len(amplitudes) > 0 else 0,
        'peaks': peaks,
        'troughs': troughs,
        'osc': osc,
        'trend': trend,
        'periods': all_periods,
        'amplitudes': amplitudes,
        'half_cycles': half_cycles,
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2026-03-25', help='Single day to analyze (YYYY-MM-DD)')
    parser.add_argument('--mode', default='1d', choices=['1d', '1w', '1m', '1y'],
                        help='1d=single day, 1w=weekly rolling, 1m=monthly rolling, 1y=yearly rolling')
    args = parser.parse_args()

    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    if args.mode == '1d':
        run_single_day(args.date, log)
    elif args.mode == '1w':
        run_rolling(args.date, window_days=5, label='1W', log_fn=log)
    elif args.mode == '1m':
        run_rolling(args.date, window_days=20, label='1M', log_fn=log)
    elif args.mode == '1y':
        run_rolling(args.date, window_days=250, label='1Y', log_fn=log)

    # Save report
    report_path = f'reports/findings/oscillation_{args.mode}_{args.date}.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        for line in report_lines:
            f.write(line + '\n')
    log(f"\n  Report saved: {report_path}")


def run_single_day(date_str, log):
    """Measure oscillation for a single day across all TFs."""
    log("=" * 70)
    log(f"OSCILLATION MEASUREMENT: {date_str} (single day)")
    log("=" * 70)

    tfs_to_measure = ['1h', '15m', '5m', '1m', '15s', '5s', '1s']
    target_date = pd.Timestamp(date_str).date()
    all_results = {}

    for tf in tfs_to_measure:
        # Find the right monthly parquet
        month_str = date_str[:7].replace('-', '_')
        path = os.path.join(ATLAS_ROOT, tf, f'{month_str}.parquet')
        if not os.path.exists(path):
            log(f"\n{tf}: no data for {month_str}")
            continue

        df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        day_df = df[df['date'] == target_date].reset_index(drop=True)

        min_bars = 10 if tf == '1h' else 30
        if len(day_df) < min_bars:
            log(f"\n{tf}: only {len(day_df)} bars on {date_str}, skipping")
            del df; gc.collect()
            continue

        prices = day_df['close'].values
        timestamps = day_df['timestamp'].values
        r = measure_oscillation(prices, timestamps, tf)
        if r is None:
            log(f"\n{tf}: couldn't measure oscillation on {date_str}")
            del df; gc.collect()
            continue

        all_results[tf] = r
        all_results[tf]['prices'] = prices
        all_results[tf]['timestamps'] = timestamps

        log(f"\n{'='*50}")
        log(f"{tf.upper()} — {date_str}")
        log(f"{'='*50}")
        log(f"  Bars: {len(day_df):,}")
        log(f"  Peaks: {r['n_peaks']} | Troughs: {r['n_troughs']}")
        log(f"")
        log(f"  PERIOD (full cycle):")
        log(f"    Mean:   {r['period_bars_mean']:.1f} bars ({r['period_minutes']:.1f} min)")
        log(f"    Median: {r['period_bars_median']:.0f} bars")
        log(f"    IQR:    [{r['period_bars_p25']:.0f}, {r['period_bars_p75']:.0f}]")
        log(f"")
        log(f"  HALF-CYCLE (trough to peak):")
        log(f"    Mean:   {r['half_cycle_bars_mean']:.1f} bars ({r['half_cycle_seconds']:.0f}s)")
        log(f"    Median: {r['half_cycle_bars_median']:.0f} bars")
        log(f"")
        log(f"  AMPLITUDE:")
        log(f"    Mean:   {r['amplitude_ticks_mean']:.1f} ticks (${r['amplitude_dollars']:.2f})")
        log(f"    Median: {r['amplitude_ticks_median']:.1f} ticks")

        del df; gc.collect()

    # Cross-TF summary
    log(f"\n\n{'='*70}")
    log(f"CROSS-TF NESTING — {date_str}")
    log(f"{'='*70}")
    log(f"")
    log(f"  {'TF':>4} {'Bars':>6} {'Period':>8} {'Min':>8} {'HalfCyc':>8} {'Amp(t)':>8} {'Cycles':>7}")
    log(f"  {'-'*55}")
    for tf in tfs_to_measure:
        if tf in all_results:
            r = all_results[tf]
            log(f"  {tf:>4} {r['n_bars']:>6} {r['period_bars_mean']:>7.1f} "
                f"{r['period_minutes']:>7.1f} {r['half_cycle_bars_mean']:>7.1f} "
                f"{r['amplitude_ticks_mean']:>7.1f} {r['n_peaks']:>7}")

    # Nesting ratios
    log(f"\n  NESTING RATIOS:")
    for higher, lower in [('1h', '15m'), ('15m', '1m'), ('1m', '15s'), ('15s', '1s')]:
        if higher in all_results and lower in all_results:
            ratio = all_results[higher]['period_minutes'] / max(0.01, all_results[lower]['period_minutes'])
            log(f"    {higher}/{lower}: {ratio:.1f}x")

    # Chart: all TFs stacked — candlesticks, bar-index x-axis, timestamp labels
    active_tfs = [tf for tf in tfs_to_measure if tf in all_results]
    n_tfs = len(active_tfs)
    matplotlib.rcParams['savefig.directory'] = os.path.abspath('examples')

    if n_tfs >= 2:
        fig, axes = plt.subplots(n_tfs, 1, figsize=(18, 3.5 * n_tfs), sharex=False)
        fig.suptitle(f'Oscillation at Every Timeframe — {date_str}', fontsize=14, fontweight='bold')

        for idx, tf in enumerate(active_tfs):
            r = all_results[tf]
            ax = axes[idx] if n_tfs > 1 else axes
            prices = r['prices']
            timestamps_tf = r['timestamps']
            n = len(prices)
            x = np.arange(n)

            # Try to load OHLC for candlesticks
            month_str = date_str[:7].replace('-', '_')
            ohlc_path = os.path.join(ATLAS_ROOT, tf, f'{month_str}.parquet')
            has_candles = False
            if os.path.exists(ohlc_path):
                df_ohlc = pd.read_parquet(ohlc_path).sort_values('timestamp')
                target = pd.Timestamp(date_str).date()
                df_ohlc['date'] = pd.to_datetime(df_ohlc['timestamp'], unit='s').dt.date
                df_ohlc = df_ohlc[df_ohlc['date'] == target].reset_index(drop=True)
                if len(df_ohlc) == n:
                    has_candles = True
                    cw = 0.6
                    for i in range(n):
                        o, c = df_ohlc.iloc[i]['open'], df_ohlc.iloc[i]['close']
                        h, l = df_ohlc.iloc[i]['high'], df_ohlc.iloc[i]['low']
                        color = '#26A69A' if c >= o else '#EF5350'
                        ax.plot([x[i], x[i]], [l, h], color='#555', linewidth=0.4)
                        ax.bar(x[i], max(abs(c - o), TICK), bottom=min(o, c),
                               width=cw, color=color, edgecolor='#555', linewidth=0.2)

            if not has_candles:
                ax.plot(x, prices, 'k-', linewidth=0.8)

            # Trend overlay
            ax.plot(x, r['trend'], 'r-', linewidth=2, alpha=0.7, label='Trend')

            # Timestamp labels
            n_ticks = min(15, n)
            tick_step = max(1, n // n_ticks)
            tick_pos = list(range(0, n, tick_step))
            tf_fmt = '%m/%d %H:%M' if tf in ('4h', '1h') else '%H:%M'
            tick_labels = [pd.to_datetime(timestamps_tf[i], unit='s').strftime(tf_fmt) for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)

            ax.set_title(f"{tf.upper()}: period={r['period_bars_mean']:.0f} bars "
                         f"({r['period_minutes']:.1f}min) | "
                         f"half={r['half_cycle_bars_mean']:.0f} | "
                         f"amp={r['amplitude_ticks_mean']:.0f}t | "
                         f"cycles={r['n_peaks']}", fontsize=10)
            ax.set_ylabel('Price')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        chart_path = f'examples/oscillation_1d_{date_str}.png'
        os.makedirs('examples', exist_ok=True)
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        log(f"\n  Chart saved: {chart_path}")

    # Overlay chart: all TF trends on one plot (the kids chart with real data)
    if n_tfs >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(18, 8))
        fig.suptitle(f'Nested Oscillation — All Timeframes on One Chart — {date_str}',
                     fontsize=14, fontweight='bold')

        tf_colors = {
            '1h': '#CC0000',   # red — structural
            '15m': '#FF8C00',  # orange — session
            '5m': '#9370DB',   # purple — swing
            '1m': '#0066CC',   # blue — oscillation
            '15s': '#2ECC71',  # green — fast
            '5s': '#99CC00',   # lime — micro
            '1s': '#DAA520',   # gold — tick
        }
        tf_widths = {'1h': 4.5, '15m': 3, '5m': 2.5, '1m': 1.5, '15s': 1, '5s': 0.7, '1s': 0.5}
        tf_alpha = {'1h': 1.0, '15m': 1.0, '5m': 0.9, '1m': 0.9, '15s': 0.7, '5s': 0.5, '1s': 0.4}
        # Skip sub-15s on overlay — at day scale they're noise
        overlay_tfs = active_tfs

        # Normalize all to 1m bar index (x-axis = minutes of the day)
        # Common time base
        t_min = min(r['timestamps'][0] for r in all_results.values())

        if '1m' in all_results:
            r_1m = all_results['1m']
            x_1m = (r_1m['timestamps'] - t_min) / 60.0
            p_1m_bg = r_1m['prices'].copy().astype(float)
            diffs_1m = np.diff(r_1m['timestamps'])
            for gi in range(len(diffs_1m)):
                if diffs_1m[gi] > 180:  # 3 min gap = maintenance
                    p_1m_bg[gi] = np.nan
            ax.plot(x_1m, p_1m_bg, color='gray', linewidth=0.3,
                    alpha=0.3, label='1m price')

        _tf_secs = {'1s': 1, '5s': 5, '15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600}

        for tf in overlay_tfs:
            r = all_results[tf]
            x = (r['timestamps'] - t_min) / 60.0
            prices = r['prices']
            tf_sec = _tf_secs.get(tf, 60)

            # Break line at gaps (maintenance windows, weekends)
            # Insert NaN where gap > 3x the TF bar duration
            gap_threshold = tf_sec * 3
            diffs = np.diff(r['timestamps'])
            x_plot = x.copy().astype(float)
            p_plot = prices.copy().astype(float)
            for gi in range(len(diffs)):
                if diffs[gi] > gap_threshold:
                    p_plot[gi] = np.nan  # breaks the line

            ax.plot(x_plot, p_plot, color=tf_colors.get(tf, 'black'),
                    linewidth=tf_widths.get(tf, 1),
                    alpha=tf_alpha.get(tf, 0.7),
                    label=f'{tf} price (period={r["period_bars_mean"]:.0f} bars, '
                          f'{r["period_minutes"]:.0f}min)')

        ax.set_xlabel('Minutes from session start')
        ax.set_ylabel('Price')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        overlay_path = f'examples/oscillation_overlay_{date_str}.png'
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        log(f"  Overlay chart saved: {overlay_path}")


def run_rolling(end_date_str, window_days, label, log_fn):
    """Rolling window analysis: measure oscillation stability over time."""
    log_fn(f"{'='*70}")
    log_fn(f"ROLLING {label} OSCILLATION — ending {end_date_str} (window={window_days} days)")
    log_fn(f"{'='*70}")

    tfs_to_measure = ['1h', '15m', '1m']  # skip sub-minute for rolling (speed)
    end_date = pd.Timestamp(end_date_str).date()

    for tf in tfs_to_measure:
        files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

        dates = sorted(df['date'].unique())
        # Find the window
        end_idx = None
        for i, d in enumerate(dates):
            if d >= end_date:
                end_idx = i
                break
        if end_idx is None:
            end_idx = len(dates) - 1
        start_idx = max(0, end_idx - window_days + 1)
        window_dates = dates[start_idx:end_idx + 1]

        daily_results = []
        for date in window_dates:
            day_df = df[df['date'] == date]
            if len(day_df) < 30:
                continue
            r = measure_oscillation(day_df['close'].values, day_df['timestamp'].values, tf)
            if r is None:
                continue
            daily_results.append({
                'date': str(date),
                'period_bars': r['period_bars_mean'],
                'period_minutes': r['period_minutes'],
                'half_cycle_bars': r['half_cycle_bars_mean'],
                'amplitude_ticks': r['amplitude_ticks_mean'],
            })

        if not daily_results:
            continue

        dr = pd.DataFrame(daily_results)
        log_fn(f"\n{tf.upper()} ({len(dr)} days in window):")
        log_fn(f"  Period: mean={dr['period_bars'].mean():.1f} std={dr['period_bars'].std():.1f} "
               f"CV={dr['period_bars'].std()/dr['period_bars'].mean():.2f}")
        log_fn(f"  Half-cycle: mean={dr['half_cycle_bars'].mean():.1f} std={dr['half_cycle_bars'].std():.1f}")
        log_fn(f"  Amplitude: mean={dr['amplitude_ticks'].mean():.1f} std={dr['amplitude_ticks'].std():.1f}")
        log_fn(f"  Stability (CV<0.5 = stable): {('STABLE' if dr['period_bars'].std()/dr['period_bars'].mean() < 0.5 else 'UNSTABLE')}")

        csv_path = f'reports/findings/oscillation_{tf}_{label}_{end_date_str}.csv'
        dr.to_csv(csv_path, index=False)
        log_fn(f"  Saved: {csv_path}")

        del df; gc.collect()

    # Save report
    report_path = 'reports/findings/oscillation_measurements.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        for line in report_lines:
            f.write(line + '\n')
    log(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    main()
