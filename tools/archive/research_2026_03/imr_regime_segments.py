#!/usr/bin/env python
"""
I-MR Regime Segment Extractor — identify tradeable regimes from price structure.

Uses SPC I-MR control charts on 1m bars to segment price into natural regimes
(MR UCL breaks = regime boundaries). Then measures each regime with 1s resolution.

Start manual: pick 1 day or 1 week, visually verify, then scale to full year.

Usage:
    python tools/imr_regime_segments.py --seed 42              # random week
    python tools/imr_regime_segments.py --date 2025-07-14      # specific day
    python tools/imr_regime_segments.py --week 2025-07-14      # week starting July 14
    python tools/imr_regime_segments.py --all-months            # full year scan
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.imr import compute_price_imr, detect_regimes
from tools.golden_path import load_1s_index, load_1s_window

TICK_SIZE = 0.25
TICK_VALUE = 0.50


def extract_regime_segments(df_1m, index_1s, context_days=21, analysis_days=7,
                            min_regime_bars=8, min_profit_ticks=0):
    """Run I-MR on 1m data and extract all regime segments with 1s metrics.

    Returns list of dicts, one per regime segment.
    """
    # Compute I-MR
    price_imr = compute_price_imr(df_1m, context_days=context_days,
                                  analysis_days=analysis_days)
    regime_ids, regime_meta = detect_regimes(price_imr, min_regime_bars=min_regime_bars)

    if not regime_meta:
        print("  No regimes detected.")
        return [], price_imr, regime_ids

    close = price_imr['close']
    timestamps = price_imr['timestamps']
    highs = df_1m['high'].values.astype(float)
    lows = df_1m['low'].values.astype(float)
    cache = {}

    segments = []
    for rm in tqdm(regime_meta, desc='Measuring regimes', unit='regime',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'):
        si = rm['start_idx']
        ei = rm['end_idx']

        entry_price = close[si]
        exit_price = close[ei]
        ts_start = float(timestamps[si])
        ts_end = float(timestamps[ei])

        # Regime direction and magnitude from 1m bars
        price_change = exit_price - entry_price
        price_change_ticks = price_change / TICK_SIZE
        duration_bars = rm['n_bars']
        duration_secs = ts_end - ts_start
        duration_mins = duration_secs / 60.0

        # Direction from price change (not just close-to-close)
        regime_high = float(highs[si:ei + 1].max())
        regime_low = float(lows[si:ei + 1].min())
        max_up_ticks = (regime_high - entry_price) / TICK_SIZE
        max_down_ticks = (entry_price - regime_low) / TICK_SIZE

        if abs(price_change_ticks) < 1:
            direction = 'FLAT'
        elif price_change_ticks > 0:
            direction = 'LONG'
        else:
            direction = 'SHORT'

        # 1s resolution metrics (if 1s data available)
        mfe_ticks_1s = 0.0
        mae_ticks_1s = 0.0
        time_to_mfe_secs = 0.0

        if index_1s:
            try:
                df_1s = load_1s_window(index_1s, ts_start, ts_end, cache)
                if len(df_1s) >= 5:
                    p1s = df_1s['close'].values.astype(float)
                    ts1s = df_1s['timestamp'].values.astype(float)

                    if direction == 'LONG':
                        fav = (p1s - entry_price) / TICK_SIZE
                        adv = (entry_price - p1s) / TICK_SIZE
                    elif direction == 'SHORT':
                        fav = (entry_price - p1s) / TICK_SIZE
                        adv = (p1s - entry_price) / TICK_SIZE
                    else:
                        fav = np.abs(p1s - entry_price) / TICK_SIZE
                        adv = np.zeros(len(p1s))

                    mfe_idx = int(np.argmax(fav))
                    mfe_ticks_1s = float(fav[mfe_idx])
                    mae_ticks_1s = float(np.max(adv[:mfe_idx + 1])) if mfe_idx > 0 else 0.0
                    time_to_mfe_secs = float(ts1s[mfe_idx] - ts1s[0])

                    # ── Noise bands: rolling std dev of 1s returns ──
                    returns_1s = np.diff(p1s) / TICK_SIZE  # bar-to-bar change in ticks
                    if len(returns_1s) >= 30:
                        # Rolling 30s std dev (noise level)
                        window = 30
                        rolling_std = np.array([
                            np.std(returns_1s[max(0, j - window):j + 1])
                            for j in range(len(returns_1s))
                        ])
                        noise_std_mean = float(np.mean(rolling_std))
                        noise_std_p75 = float(np.percentile(rolling_std, 75))
                        noise_std_p25 = float(np.percentile(rolling_std, 25))
                        # Suggested SL: 2x avg noise std (2-sigma)
                        noise_sl_ticks = round(2 * noise_std_mean, 1)
                        noise_sl_dollars = round(noise_sl_ticks * TICK_VALUE, 2)
                        # Noise ratio: how much of MAE is just noise?
                        noise_ratio = (2 * noise_std_mean) / mae_ticks_1s if mae_ticks_1s > 0 else float('inf')
                    else:
                        noise_std_mean = noise_std_p75 = noise_std_p25 = 0.0
                        noise_sl_ticks = noise_sl_dollars = 0.0
                        noise_ratio = 0.0
            except Exception:
                noise_std_mean = noise_std_p75 = noise_std_p25 = 0.0
                noise_sl_ticks = noise_sl_dollars = 0.0
                noise_ratio = 0.0

        seg = {
            'regime_id': rm['regime_id'],
            'ts_start': ts_start,
            'ts_end': ts_end,
            'start_time': datetime.fromtimestamp(ts_start, tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
            'end_time': datetime.fromtimestamp(ts_end, tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'price_change_ticks': round(price_change_ticks, 1),
            'price_change_dollars': round(price_change_ticks * TICK_VALUE, 2),
            'regime_high': round(regime_high, 2),
            'regime_low': round(regime_low, 2),
            'max_up_ticks': round(max_up_ticks, 1),
            'max_down_ticks': round(max_down_ticks, 1),
            'duration_bars': duration_bars,
            'duration_mins': round(duration_mins, 1),
            'mfe_ticks_1s': round(mfe_ticks_1s, 1),
            'mae_ticks_1s': round(mae_ticks_1s, 1),
            'mfe_dollars_1s': round(mfe_ticks_1s * TICK_VALUE, 2),
            'mae_dollars_1s': round(mae_ticks_1s * TICK_VALUE, 2),
            'time_to_mfe_mins': round(time_to_mfe_secs / 60, 1),
            'volatility': round(rm['volatility'], 4),
            # Noise bands
            'noise_std_mean': round(noise_std_mean, 2),
            'noise_std_p25': round(noise_std_p25, 2),
            'noise_std_p75': round(noise_std_p75, 2),
            'noise_sl_ticks': noise_sl_ticks,
            'noise_sl_dollars': noise_sl_dollars,
            'noise_ratio': round(noise_ratio, 2),
        }
        segments.append(seg)

    return segments, price_imr, regime_ids


def print_report(segments, label=""):
    """Print regime segment report."""
    if not segments:
        print("No segments to report.")
        return

    df = pd.DataFrame(segments)
    n = len(df)

    print(f"\n{'='*80}")
    print(f"  I-MR REGIME SEGMENTS{f' -- {label}' if label else ''}")
    print(f"{'='*80}")
    print(f"  Total regimes: {n}")
    print(f"  LONG: {len(df[df['direction']=='LONG'])}, "
          f"SHORT: {len(df[df['direction']=='SHORT'])}, "
          f"FLAT: {len(df[df['direction']=='FLAT'])}")

    # Per-regime table
    print(f"\n  {'ID':>3} {'Start':>16} {'End':>16} {'Dir':>5} "
          f"{'Change':>8} {'Dollars':>8} {'Bars':>5} {'Mins':>6} "
          f"{'MFE$':>7} {'MAE$':>7} {'R:R':>5} {'Noise':>6} {'NoiseSL':>8}")
    print(f"  {'-'*3} {'-'*16} {'-'*16} {'-'*5} "
          f"{'-'*8} {'-'*8} {'-'*5} {'-'*6} "
          f"{'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*8}")

    for _, s in df.iterrows():
        rr = (s['mfe_dollars_1s'] / s['mae_dollars_1s']
              if s['mae_dollars_1s'] > 0 else float('inf'))
        rr_str = f"{rr:.1f}" if rr < 100 else "inf"
        noise_str = f"{s.get('noise_std_mean', 0):.1f}t"
        nsl_str = f"${s.get('noise_sl_dollars', 0):.0f}"
        print(f"  {s['regime_id']:>3} {s['start_time']:>16} {s['end_time']:>16} "
              f"{s['direction']:>5} "
              f"{s['price_change_ticks']:>+7.0f}t "
              f"${s['price_change_dollars']:>+6.0f} "
              f"{s['duration_bars']:>5} {s['duration_mins']:>5.0f}m "
              f"${s['mfe_dollars_1s']:>6.0f} ${s['mae_dollars_1s']:>6.0f} "
              f"{rr_str:>5} {noise_str:>6} {nsl_str:>8}")

    # Summary stats
    profitable = df[df['mfe_dollars_1s'] >= 30]
    print(f"\n  -- SUMMARY --")
    print(f"  All regimes:     {n}")
    print(f"  Avg MFE:         ${df['mfe_dollars_1s'].mean():.2f}")
    print(f"  Avg MAE:         ${df['mae_dollars_1s'].mean():.2f}")
    print(f"  Avg duration:    {df['duration_mins'].mean():.1f} min")

    print(f"\n  Profitable ($30+ MFE): {len(profitable)}")
    if len(profitable) > 0:
        print(f"    Avg MFE:       ${profitable['mfe_dollars_1s'].mean():.2f}")
        print(f"    Avg MAE:       ${profitable['mae_dollars_1s'].mean():.2f}")
        print(f"    Avg duration:  {profitable['duration_mins'].mean():.1f} min")
        print(f"    Avg R:R:       1:{(profitable['mfe_dollars_1s'].mean() / max(profitable['mae_dollars_1s'].mean(), 0.01)):.1f}")

    # SL survival with $10 risk
    if len(profitable) > 0:
        sl_ticks = 20
        survivors = profitable[profitable['mae_ticks_1s'] <= sl_ticks]
        print(f"\n  With $10 SL (20 ticks):")
        print(f"    Survivors:     {len(survivors)}/{len(profitable)} "
              f"({len(survivors)/len(profitable)*100:.0f}%)")
        if len(survivors) > 0:
            print(f"    Avg reward:    ${survivors['mfe_dollars_1s'].mean():.2f}")
            print(f"    Total reward:  ${survivors['mfe_dollars_1s'].sum():.2f}")

    # Noise band analysis
    if 'noise_std_mean' in df.columns:
        noise_data = df[df['noise_std_mean'] > 0]
        if len(noise_data) > 0:
            print(f"\n  -- NOISE BAND ANALYSIS --")
            print(f"    Avg noise (1s std):  {noise_data['noise_std_mean'].mean():.2f} ticks")
            print(f"    p25 noise:           {noise_data['noise_std_p25'].mean():.2f} ticks")
            print(f"    p75 noise:           {noise_data['noise_std_p75'].mean():.2f} ticks")
            print(f"    Suggested SL (2x):   {noise_data['noise_sl_ticks'].mean():.1f} ticks "
                  f"(${noise_data['noise_sl_dollars'].mean():.2f})")

            # Noise-adaptive SL survival
            noise_sl_survivors = df[df['mae_ticks_1s'] <= df['noise_sl_ticks'] * 2]
            print(f"\n    Noise-adaptive SL (2x noise):")
            print(f"      Survivors:  {len(noise_sl_survivors)}/{n} "
                  f"({len(noise_sl_survivors)/n*100:.0f}%)")

            # Quiet vs noisy regimes
            median_noise = noise_data['noise_std_mean'].median()
            quiet = df[df['noise_std_mean'] <= median_noise]
            noisy = df[df['noise_std_mean'] > median_noise]
            if len(quiet) > 0 and len(noisy) > 0:
                q_surv = quiet[quiet['mae_ticks_1s'] <= 20]
                n_surv = noisy[noisy['mae_ticks_1s'] <= 20]
                print(f"\n    Quiet regimes (noise <= {median_noise:.1f}t): "
                      f"{len(quiet)}, $10 SL survives {len(q_surv)/len(quiet)*100:.0f}%, "
                      f"avg MFE ${quiet['mfe_dollars_1s'].mean():.0f}")
                print(f"    Noisy regimes (noise > {median_noise:.1f}t):  "
                      f"{len(noisy)}, $10 SL survives {len(n_surv)/len(noisy)*100:.0f}%, "
                      f"avg MFE ${noisy['mfe_dollars_1s'].mean():.0f}")


def plot_regime_chart(segments, price_imr, regime_ids, output_path, label=""):
    """Create annotated regime chart for manual review."""
    close = price_imr['close']
    timestamps = price_imr['timestamps']
    mr_abs = price_imr['mr_abs']
    ucl_mr = price_imr['ucl_mr']
    center = price_imr['center']
    ucl_i = price_imr['ucl_i']
    lcl_i = price_imr['lcl_i']
    warmup_end = price_imr['warmup_end_idx']
    analysis_mask = price_imr['analysis_mask']

    # Convert timestamps to datetime for x-axis
    dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]

    # Analysis window only
    a_idx = np.where(analysis_mask)[0]
    if len(a_idx) == 0:
        print("  No analysis bars to plot.")
        return

    a_start, a_end = a_idx[0], a_idx[-1] + 1

    fig, axes = plt.subplots(3, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.set_facecolor('white')

    # Regime colors
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
              '#00BCD4', '#E91E63', '#8BC34A', '#FF5722', '#3F51B5']

    # Panel 1: Price with regime coloring
    ax = axes[0]
    ax.set_facecolor('#FAFAFA')

    # Plot each regime as a colored segment
    for seg in segments:
        si = seg.get('_start_idx', None)
        ei = seg.get('_end_idx', None)
        # Find indices from timestamps
        if si is None:
            mask = (timestamps >= seg['ts_start']) & (timestamps <= seg['ts_end'])
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            si, ei = idx[0], idx[-1]

        rid = seg['regime_id']
        c = colors[rid % len(colors)]
        ax.plot(dt_stamps[si:ei + 1], close[si:ei + 1], color=c, linewidth=1.5, alpha=0.8)

        # Annotate profitable regimes
        if seg['mfe_dollars_1s'] >= 30:
            mid = (si + ei) // 2
            ax.annotate(
                f"${seg['mfe_dollars_1s']:.0f}\n{seg['direction']}",
                xy=(dt_stamps[mid], close[mid]),
                fontsize=8, fontweight='bold',
                ha='center', va='bottom' if seg['direction'] == 'LONG' else 'top',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8),
            )

    # Regime boundaries (vertical lines)
    for seg in segments:
        mask = timestamps >= seg['ts_start']
        idx = np.where(mask)[0]
        if len(idx) > 0:
            ax.axvline(x=dt_stamps[idx[0]], color='gray', linestyle=':', alpha=0.3)

    ax.set_ylabel('Price', fontsize=11)
    ax.set_title(f'I-MR REGIME SEGMENTS{f" -- {label}" if label else ""}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.15)

    # Panel 2: MR chart with UCL
    ax = axes[1]
    ax.set_facecolor('#FAFAFA')
    ax.bar(dt_stamps[a_start:a_end], mr_abs[a_start:a_end],
           width=timedelta(seconds=50), color='steelblue', alpha=0.6)
    ax.axhline(y=ucl_mr, color='red', linestyle='--', linewidth=1.5,
               label=f'UCL_MR = {ucl_mr:.2f}')
    ax.set_ylabel('|MR|', fontsize=11)
    ax.set_title('Moving Range (regime breaks = bars above red line)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Panel 3: Regime map (colored bars)
    ax = axes[2]
    ax.set_facecolor('#FAFAFA')
    for seg in segments:
        mask = (timestamps >= seg['ts_start']) & (timestamps <= seg['ts_end'])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        rid = seg['regime_id']
        c = colors[rid % len(colors)]
        ax.axvspan(dt_stamps[idx[0]], dt_stamps[idx[-1]], alpha=0.4, color=c)
        mid = idx[len(idx) // 2]
        ax.text(dt_stamps[mid], 0.5, f"R{rid}\n{seg['direction']}\n{seg['duration_mins']:.0f}m",
                ha='center', va='center', fontsize=7, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Regime Map', fontsize=11)
    ax.set_yticks([])
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nChart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='I-MR Regime Segment Extractor')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random week from IS data')
    parser.add_argument('--date', default=None,
                        help='Specific date (YYYY-MM-DD) for single day')
    parser.add_argument('--week', default=None,
                        help='Week starting at date (YYYY-MM-DD)')
    parser.add_argument('--all-months', action='store_true',
                        help='Full year scan (all months)')
    parser.add_argument('--min-regime-bars', type=int, default=8,
                        help='Minimum bars per regime (default 8)')
    parser.add_argument('--output', default=None,
                        help='Output directory for charts and CSV')
    args = parser.parse_args()

    out_dir = args.output or 'tools/plots/standalone/imr_regimes'

    print(f"I-MR Regime Segment Extractor")
    print(f"  Data: {args.data_dir}")

    # Load 1m data
    print(f"\nLoading 1m data...")
    df_1m = load_atlas_tf(args.data_dir, '1m')
    if df_1m.empty:
        print("ERROR: No 1m data found")
        sys.exit(1)
    print(f"  Loaded {len(df_1m)} 1m bars")

    # Load 1s index
    print(f"\nLoading 1s data index...")
    index_1s = load_1s_index(args.data_dir)

    # Determine time window
    _ctx_days = None  # computed by --date path; None = default 21
    if args.date:
        # Single day: all prior data as context, target day as analysis
        dt = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        t_day_start = dt.timestamp()
        t_day_end = t_day_start + 86400
        # Use all data up to end of target day (prior bars = context)
        mask = df_1m['timestamp'] <= t_day_end
        df_window = df_1m[mask].reset_index(drop=True)
        # Count how many bars are ON the target day → that's the analysis window
        _on_day = (df_window['timestamp'] >= t_day_start).sum()
        _total = len(df_window)
        _ctx_bars = _total - _on_day
        # Convert context bars to approximate days for compute_price_imr
        _ctx_days = max(0, int(_ctx_bars / 1380))  # ~1380 1m bars/day
        print(f"  Window: {_total} bars, Day: {args.date}")
        print(f"  Context: {_ctx_bars} bars (~{_ctx_days}d), Analysis: {_on_day} bars")
        label = f"Day: {args.date}"
        analysis_days = 1
        file_tag = args.date.replace('-', '')

    elif args.week:
        # Specific week: 21 days context + 7 days analysis
        dt = datetime.strptime(args.week, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        t_start = dt.timestamp() - 21 * 86400
        t_end = dt.timestamp() + 7 * 86400
        mask = (df_1m['timestamp'] >= t_start) & (df_1m['timestamp'] <= t_end)
        df_window = df_1m[mask].reset_index(drop=True)
        label = f"Week: {args.week}"
        analysis_days = 7
        file_tag = f"week_{args.week.replace('-', '')}"

    elif args.seed is not None:
        # Random week
        from tools.imr_golden_path import pick_random_week
        ctx_start, week_start, week_end = pick_random_week(df_1m, seed=args.seed)
        mask = (df_1m['timestamp'] >= ctx_start) & (df_1m['timestamp'] <= week_end)
        df_window = df_1m[mask].reset_index(drop=True)
        ws = datetime.fromtimestamp(week_start, tz=timezone.utc)
        label = f"Random week (seed={args.seed}): {ws:%Y-%m-%d}"
        analysis_days = 7
        file_tag = f"seed{args.seed}"

    elif args.all_months:
        # Full year: 21 day context, rest is analysis
        df_window = df_1m.copy()
        label = "Full IS Dataset"
        analysis_days = 0  # use all remaining
        file_tag = "full_year"
    else:
        # Default: random week with seed 42
        from tools.imr_golden_path import pick_random_week
        ctx_start, week_start, week_end = pick_random_week(df_1m, seed=42)
        mask = (df_1m['timestamp'] >= ctx_start) & (df_1m['timestamp'] <= week_end)
        df_window = df_1m[mask].reset_index(drop=True)
        ws = datetime.fromtimestamp(week_start, tz=timezone.utc)
        label = f"Random week (seed=42): {ws:%Y-%m-%d}"
        analysis_days = 7
        file_tag = "seed42"

    # Use computed context days for --date mode, default 21 otherwise
    _context_days = _ctx_days if args.date else 21
    print(f"\n  Window: {len(df_window)} bars, {label}")

    # Extract segments
    segments, price_imr, regime_ids = extract_regime_segments(
        df_window, index_1s,
        context_days=_context_days, analysis_days=analysis_days,
        min_regime_bars=args.min_regime_bars
    )

    if not segments:
        print("No segments extracted.")
        return

    # Report
    print_report(segments, label)

    # Save CSV
    csv_path = os.path.join(out_dir, f'regime_segments_{file_tag}.csv')
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(segments).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # Chart
    chart_path = os.path.join(out_dir, f'regime_chart_{file_tag}.png')
    plot_regime_chart(segments, price_imr, regime_ids, chart_path, label)


if __name__ == '__main__':
    main()
