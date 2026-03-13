#!/usr/bin/env python
"""
L2 Risk Budget Analysis — How much do you need to risk to capture $30+ trades?

For each golden path L2 segment ($30+), computes:
  - MFE: max favorable excursion (the profit)
  - MAE: max adverse excursion BEFORE MFE (the cost of waiting)
  - Time to MFE: how long until peak profit
  - Risk/Reward: MAE / MFE ratio

This answers: "What stop loss do I need to survive long enough to capture L2 moves?"

Usage:
    python tools/l2_risk_budget.py                  # full IS dataset
    python tools/l2_risk_budget.py --seed 17        # specific random week
    python tools/l2_risk_budget.py --all-months     # scan all 12 months
"""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.golden_path import load_1s_index, load_1s_window


def analyze_l2_segments(df_1s, tick_size=0.25, l2_min_ticks=60):
    """Find all L2 segments and compute MAE/MFE/time metrics.

    Scans 1s data in 5-minute windows, finds $30+ moves,
    then measures the drawdown path to get there.

    Returns list of dicts with full risk metrics per segment.
    """
    prices = df_1s['close'].values.astype(float)
    timestamps = df_1s['timestamp'].values.astype(float)
    n = len(prices)

    window_bars = 300  # 5 min at 1s
    segments = []
    i = 0

    while i < n - 10:
        end = min(i + window_bars, n)
        window_prices = prices[i:end]
        window_ts = timestamps[i:end]

        if len(window_prices) < 10:
            i = end
            continue

        entry = window_prices[0]
        max_up = float(window_prices.max() - entry)
        max_down = float(entry - window_prices.min())

        if max_up >= max_down and max_up >= tick_size * 2:
            direction = 'LONG'
            peak_idx = int(np.argmax(window_prices))
            captured_ticks = max_up / tick_size
        elif max_down >= tick_size * 2:
            direction = 'SHORT'
            peak_idx = int(np.argmin(window_prices))
            captured_ticks = max_down / tick_size
        else:
            i += window_bars // 2
            continue

        # Only analyze L2 segments ($30+)
        if captured_ticks >= l2_min_ticks:
            # Compute MAE: worst drawdown BEFORE reaching MFE
            path_to_peak = window_prices[:peak_idx + 1]

            if direction == 'LONG':
                # For LONG: MAE = how far price drops below entry before peak
                running_min = np.minimum.accumulate(path_to_peak)
                mae_ticks = float((entry - running_min.min()) / tick_size)
                # Also track running PnL path
                pnl_path = (path_to_peak - entry) / tick_size
            else:
                # For SHORT: MAE = how far price rises above entry before peak
                running_max = np.maximum.accumulate(path_to_peak)
                mae_ticks = float((running_max.max() - entry) / tick_size)
                pnl_path = (entry - path_to_peak) / tick_size

            time_to_mfe_seconds = float(window_ts[peak_idx] - window_ts[0])
            time_to_mfe_minutes = time_to_mfe_seconds / 60.0

            segments.append({
                'direction': direction,
                'entry_price': float(entry),
                'mfe_ticks': round(captured_ticks, 2),
                'mfe_dollars': round(captured_ticks * 0.50, 2),
                'mae_ticks': round(mae_ticks, 2),
                'mae_dollars': round(mae_ticks * 0.50, 2),
                'time_to_mfe_sec': round(time_to_mfe_seconds),
                'time_to_mfe_min': round(time_to_mfe_minutes, 1),
                'risk_reward': round(mae_ticks / captured_ticks, 4) if captured_ticks > 0 else 999,
                'timestamp': float(window_ts[0]),
                'pnl_path_min': round(float(pnl_path.min()) * 0.50, 2),
                'pnl_path_max': round(float(pnl_path.max()) * 0.50, 2),
            })

        seg_end = max(peak_idx + 1, 2)
        i = max(i + seg_end, i + 10)

    return segments


def print_report(segments, label=""):
    """Print risk budget report."""
    if not segments:
        print("  No L2 segments found.")
        return

    n = len(segments)
    maes = np.array([s['mae_ticks'] for s in segments])
    mfes = np.array([s['mfe_ticks'] for s in segments])
    mae_dollars = maes * 0.50
    mfe_dollars = mfes * 0.50
    rr = np.array([s['risk_reward'] for s in segments])
    times = np.array([s['time_to_mfe_min'] for s in segments])

    print(f"\n{'='*70}")
    print(f"L2 RISK BUDGET ANALYSIS{f' — {label}' if label else ''}")
    print(f"{'='*70}")
    print(f"  L2 segments ($30+): {n}")
    print(f"  Direction split: {sum(1 for s in segments if s['direction']=='LONG')} LONG, "
          f"{sum(1 for s in segments if s['direction']=='SHORT')} SHORT")

    print(f"\n  -- MFE (What You Get) --")
    print(f"    Mean:   ${mfe_dollars.mean():.2f}  ({mfes.mean():.0f} ticks)")
    print(f"    Median: ${np.median(mfe_dollars):.2f}  ({np.median(mfes):.0f} ticks)")
    print(f"    p25:    ${np.percentile(mfe_dollars, 25):.2f}")
    print(f"    p75:    ${np.percentile(mfe_dollars, 75):.2f}")
    print(f"    Max:    ${mfe_dollars.max():.2f}")

    print(f"\n  -- MAE Before MFE (What You Risk) --")
    print(f"    Mean:   ${mae_dollars.mean():.2f}  ({maes.mean():.0f} ticks)")
    print(f"    Median: ${np.median(mae_dollars):.2f}  ({np.median(maes):.0f} ticks)")
    print(f"    p75:    ${np.percentile(mae_dollars, 75):.2f}  <- SL must survive this")
    print(f"    p90:    ${np.percentile(mae_dollars, 90):.2f}  <- conservative SL")
    print(f"    p95:    ${np.percentile(mae_dollars, 95):.2f}")
    print(f"    Max:    ${mae_dollars.max():.2f}")

    print(f"\n  -- Risk/Reward (MAE/MFE) --")
    print(f"    Mean:   {rr.mean():.3f}  (risk {rr.mean()*100:.1f}% of reward)")
    print(f"    Median: {np.median(rr):.3f}")
    print(f"    p75:    {np.percentile(rr, 75):.3f}")
    print(f"    p90:    {np.percentile(rr, 90):.3f}")

    print(f"\n  -- Time to MFE --")
    print(f"    Mean:   {times.mean():.1f} min")
    print(f"    Median: {np.median(times):.1f} min")
    print(f"    p75:    {np.percentile(times, 75):.1f} min")
    print(f"    p90:    {np.percentile(times, 90):.1f} min")

    # Zero-MAE segments (no drawdown at all — price moves immediately in our favor)
    zero_mae = (maes == 0).sum()
    print(f"\n  -- Zero-MAE trades (immediate profit) --")
    print(f"    {zero_mae} of {n} ({zero_mae/n*100:.1f}%)")

    # MAE buckets
    print(f"\n  -- MAE Distribution (what SL do you need?) --")
    buckets = [0, 2, 5, 10, 15, 20, 30, 50, 100, 999]
    for i_b in range(len(buckets) - 1):
        lo, hi = buckets[i_b], buckets[i_b + 1]
        count = ((mae_dollars >= lo) & (mae_dollars < hi)).sum()
        pct = count / n * 100
        if count > 0:
            avg_mfe = mfe_dollars[(mae_dollars >= lo) & (mae_dollars < hi)].mean()
            print(f"    MAE ${lo:>3}-${hi:>3}: {count:>4} trades ({pct:5.1f}%) "
                  f"-> avg MFE ${avg_mfe:.2f}")

    # CRITICAL ANSWER: What stop loss captures 80% of L2 trades?
    print(f"\n  -- STOP LOSS SIZING (to capture X% of L2 trades) --")
    for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
        sl = np.percentile(mae_dollars, pct)
        surviving = (mae_dollars <= sl).sum()
        avg_profit = mfe_dollars[mae_dollars <= sl].mean()
        total_profit = mfe_dollars[mae_dollars <= sl].sum()
        print(f"    SL=${sl:>6.2f} -> captures {surviving:>4}/{n} trades ({pct}%) "
              f"| avg profit ${avg_profit:.2f} | total ${total_profit:,.0f}")

    return {
        'n': n,
        'mae_mean': float(mae_dollars.mean()),
        'mae_median': float(np.median(mae_dollars)),
        'mae_p75': float(np.percentile(mae_dollars, 75)),
        'mae_p90': float(np.percentile(mae_dollars, 90)),
        'mfe_mean': float(mfe_dollars.mean()),
        'rr_mean': float(rr.mean()),
        'time_mean': float(times.mean()),
    }


def plot_risk_budget(segments, output_path):
    """Create risk budget visualization."""
    if not segments:
        return

    maes = np.array([s['mae_dollars'] for s in segments])
    mfes = np.array([s['mfe_dollars'] for s in segments])
    times = np.array([s['time_to_mfe_min'] for s in segments])
    dirs = np.array([s['direction'] for s in segments])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.set_facecolor('white')
    for row in axes:
        for ax in row:
            ax.set_facecolor('white')

    # -- Panel 1: MAE vs MFE scatter --
    ax = axes[0, 0]
    long_mask = dirs == 'LONG'
    ax.scatter(maes[long_mask], mfes[long_mask], c='#00C853', alpha=0.5,
               s=20, label=f'LONG ({long_mask.sum()})', edgecolors='none')
    ax.scatter(maes[~long_mask], mfes[~long_mask], c='#FF1744', alpha=0.5,
               s=20, label=f'SHORT ({(~long_mask).sum()})', edgecolors='none')
    # 1:1 line (breakeven)
    max_val = max(maes.max(), mfes.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='1:1 (breakeven)')
    ax.set_xlabel('MAE (risk) $', fontsize=10)
    ax.set_ylabel('MFE (reward) $', fontsize=10)
    ax.set_title('RISK vs REWARD\n(every dot = one $30+ trade opportunity)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # -- Panel 2: MAE histogram --
    ax = axes[0, 1]
    ax.hist(maes, bins=50, color='#FF6F00', alpha=0.7, edgecolor='white')
    for pct, color, ls in [(75, '#AA0000', '--'), (90, '#AA0000', ':')]:
        val = np.percentile(maes, pct)
        ax.axvline(x=val, color=color, linestyle=ls, linewidth=1.5,
                   label=f'p{pct}=${val:.2f}')
    ax.set_xlabel('MAE (drawdown before profit) $', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('MAE DISTRIBUTION\n(how much pain before the gain?)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # -- Panel 3: Time to MFE histogram --
    ax = axes[1, 0]
    ax.hist(times, bins=50, color='#1565C0', alpha=0.7, edgecolor='white')
    ax.axvline(x=np.median(times), color='#AA0000', linestyle='--', linewidth=1.5,
               label=f'Median={np.median(times):.1f} min')
    ax.set_xlabel('Time to MFE (minutes)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('TIME TO PROFIT\n(how long do you hold?)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # -- Panel 4: SL capture curve --
    ax = axes[1, 1]
    sl_range = np.linspace(0, np.percentile(maes, 99), 100)
    capture_pct = [100 * (maes <= sl).sum() / len(maes) for sl in sl_range]
    avg_profit = [float(mfes[maes <= sl].mean()) if (maes <= sl).any() else 0
                  for sl in sl_range]

    ax.plot(sl_range, capture_pct, color='#1565C0', linewidth=2, label='% trades captured')
    ax.set_xlabel('Stop Loss Size $', fontsize=10)
    ax.set_ylabel('% L2 Trades Captured', fontsize=10, color='#1565C0')

    ax2 = ax.twinx()
    ax2.plot(sl_range, avg_profit, color='#FF6F00', linewidth=2, linestyle='--',
             label='Avg profit per trade $')
    ax2.set_ylabel('Avg Profit per Trade $', fontsize=10, color='#FF6F00')

    # Mark key SL levels
    for target_pct in [75, 90]:
        sl_val = np.percentile(maes, target_pct)
        ax.axvline(x=sl_val, color='#888888', linestyle=':', alpha=0.5)
        ax.annotate(f'SL=${sl_val:.0f}\n({target_pct}%)',
                    xy=(sl_val, target_pct), fontsize=8, ha='left',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    ax.set_title('STOP LOSS SIZING CURVE\n(bigger SL = more trades captured)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'L2 RISK BUDGET — {len(segments)} trades ($30+)\n'
                 f'MAE median=${np.median(maes):.2f} | '
                 f'MFE median=${np.median(mfes):.2f} | '
                 f'R:R={np.median(maes)/np.median(mfes):.2f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='L2 Risk Budget Analysis')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (picks one week). Omit for full scan.')
    parser.add_argument('--all-months', action='store_true',
                        help='Scan all 12 months of IS data')
    parser.add_argument('--month', default=None,
                        help='Specific month (e.g., 2025_07)')
    parser.add_argument('--l2-min', type=float, default=30.0,
                        help='L2 minimum profit threshold in dollars (default: $30)')
    args = parser.parse_args()

    l2_min_ticks = args.l2_min / 0.50  # convert $ to ticks

    print("=" * 70)
    print(f"L2 RISK BUDGET ANALYSIS (trades >= ${args.l2_min:.0f})")
    print("=" * 70)

    index_1s = load_1s_index(args.data_dir)

    if args.seed is not None:
        # Single random week
        df_1m = load_atlas_tf(args.data_dir, '1m')
        from tools.imr_golden_path import pick_random_week
        ctx_start, week_start, week_end = pick_random_week(df_1m, seed=args.seed)
        ws_dt = datetime.fromtimestamp(week_start, tz=timezone.utc)
        we_dt = datetime.fromtimestamp(week_end, tz=timezone.utc)
        print(f"\n  Week: {ws_dt:%Y-%m-%d} to {we_dt:%Y-%m-%d}")

        cache = {}
        df_1s = load_1s_window(index_1s, week_start, week_end, cache)
        segments = analyze_l2_segments(df_1s, l2_min_ticks=l2_min_ticks)
        stats = print_report(segments, f"{ws_dt:%Y-%m-%d} to {we_dt:%Y-%m-%d}")

    else:
        # Full scan: all available 1s data (or specific month)
        months = [args.month] if args.month else sorted(index_1s.keys())
        all_segments = []

        for month_key in months:
            print(f"\n  Processing {month_key}...")
            cache = {}
            df_1s = pd.read_parquet(index_1s[month_key])
            if 'timestamp' in df_1s.columns:
                if pd.api.types.is_datetime64_any_dtype(df_1s['timestamp']):
                    df_1s['timestamp'] = df_1s['timestamp'].astype('int64') // 10**9
                df_1s = df_1s.sort_values('timestamp').reset_index(drop=True)

            segs = analyze_l2_segments(df_1s, l2_min_ticks=l2_min_ticks)
            print(f"    {month_key}: {len(segs)} L2 segments")
            all_segments.extend(segs)

        stats = print_report(all_segments, "Full IS Dataset (12 months)")

        # Plot
        os.makedirs('tools/plots/standalone/risk_budget', exist_ok=True)
        plot_path = 'tools/plots/standalone/risk_budget/l2_risk_budget.png'
        plot_risk_budget(all_segments, plot_path)


if __name__ == '__main__':
    main()
