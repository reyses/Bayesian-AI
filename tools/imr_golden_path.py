#!/usr/bin/env python
"""
I-MR Chart with Golden Path Overlay — 1-minute resolution.

Picks a random week from IS data (DATA/ATLAS), computes price I-MR control
charts on 1m bars, detects regimes, then overlays the golden path (oracle
optimal segments from 1s data) on the price panel.

Usage:
    python tools/imr_golden_path.py                  # random week
    python tools/imr_golden_path.py --month 2025_07  # random week from July
    python tools/imr_golden_path.py --seed 42        # reproducible pick
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
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.imr import compute_price_imr, detect_regimes, D4, E2
from tools.golden_path import load_1s_index, load_1s_window, oracle_segments

# Regime colors
_REGIME_COLORS = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#E91E63', '#8BC34A', '#FF5722', '#3F51B5',
]


def pick_random_week(df_1m, seed=None):
    """Pick a random 5-trading-day (7-calendar-day) window from 1m data.

    Returns (week_start_ts, week_end_ts, context_start_ts).
    Context = 21 calendar days before week start for warmup.
    """
    rng = random.Random(seed)
    timestamps = df_1m['timestamp'].values
    t_min, t_max = float(timestamps[0]), float(timestamps[-1])

    # Need at least 28 days of data (21 warmup + 7 analysis)
    total_days = (t_max - t_min) / 86400
    if total_days < 30:
        print(f"ERROR: Need 30+ days of data, have {total_days:.0f}")
        sys.exit(1)

    # Pick a random start for the analysis week
    # Must have 21 days of warmup before and 7 days of analysis
    earliest_week_start = t_min + 21 * 86400
    latest_week_start = t_max - 7 * 86400

    week_start = rng.uniform(earliest_week_start, latest_week_start)
    week_end = week_start + 7 * 86400
    context_start = week_start - 21 * 86400

    return context_start, week_start, week_end


def compute_golden_path_segments(index_1s, week_start, week_end, tick_size=0.25):
    """Compute oracle optimal segments from 1s data for the analysis week.

    Returns list of dicts with: ts_start, ts_end, direction, captured_ticks,
    prices (1s close array for the segment).
    """
    print("  Loading 1s data for golden path...")
    cache = {}
    df_1s = load_1s_window(index_1s, week_start, week_end, cache)

    if df_1s.empty or len(df_1s) < 100:
        print(f"  WARNING: Only {len(df_1s)} 1s bars in window")
        return [], np.array([]), np.array([])

    print(f"  Loaded {len(df_1s):,} 1s bars for golden path")

    prices_1s = df_1s['close'].values.astype(float)
    timestamps_1s = df_1s['timestamp'].values.astype(float)

    # Segment into ~5-min windows and compute oracle segments for each
    # This gives us directional segments to overlay on price
    window_bars = 300  # 5 minutes at 1s
    segments = []
    i = 0
    n = len(prices_1s)

    while i < n - 10:
        end = min(i + window_bars, n)
        window_prices = prices_1s[i:end]
        window_ts = timestamps_1s[i:end]

        if len(window_prices) < 10:
            i = end
            continue

        entry = window_prices[0]

        # Try both directions, pick the one with more favorable movement
        max_up = float(window_prices.max() - entry)
        max_down = float(entry - window_prices.min())

        if max_up >= max_down and max_up >= tick_size * 2:
            direction = 'LONG'
            # Find the peak
            peak_idx = int(np.argmax(window_prices))
            captured = max_up / tick_size
        elif max_down >= tick_size * 2:
            direction = 'SHORT'
            peak_idx = int(np.argmin(window_prices))
            captured = max_down / tick_size
        else:
            # No meaningful move, skip
            i += window_bars // 2
            continue

        seg_end = max(peak_idx + 1, 2)
        segments.append({
            'ts_start': float(window_ts[0]),
            'ts_end': float(window_ts[min(seg_end, len(window_ts) - 1)]),
            'direction': direction,
            'captured_ticks': round(captured, 2),
            'entry_price': float(entry),
            'exit_price': float(window_prices[min(peak_idx, len(window_prices) - 1)]),
        })

        i = max(i + seg_end, i + 10)

    # Also compute full-window oracle segments for summary stats
    seg_result = oracle_segments(prices_1s, tick_size=tick_size)

    print(f"  Golden path: {len(segments)} directional segments, "
          f"oracle captured {seg_result['captured_ticks']:.0f} ticks "
          f"(efficiency={seg_result['efficiency']:.1f})")

    return segments, prices_1s, timestamps_1s


def plot_imr_golden_path(df_context, price_imr, regime_ids, regime_meta,
                         golden_segments, week_start, week_end, output_path):
    """Create the 4-panel I-MR chart with golden path overlay on price."""

    close = price_imr['close']
    mr = price_imr['mr']
    mr_abs = price_imr['mr_abs']
    timestamps = price_imr['timestamps']
    center = price_imr['center']
    ucl_i = price_imr['ucl_i']
    lcl_i = price_imr['lcl_i']
    mr_bar = price_imr['mr_bar']
    ucl_mr = price_imr['ucl_mr']
    warmup_end = price_imr['warmup_end_idx']

    n = len(close)
    x = np.arange(n)

    # Analysis window mask (bars in the target week)
    analysis_mask = (timestamps >= week_start) & (timestamps < week_end)
    analysis_start_idx = int(np.argmax(analysis_mask)) if analysis_mask.any() else warmup_end
    analysis_end_idx = n - 1 - int(np.argmax(analysis_mask[::-1])) if analysis_mask.any() else n - 1

    # Date labels for x-axis
    date_labels = []
    date_positions = []
    prev_day = None
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day = dt.strftime('%m/%d')
        if day != prev_day:
            date_labels.append(day)
            date_positions.append(i)
            prev_day = day

    # ── Figure: 4 panels ──
    fig, axes = plt.subplots(4, 1, figsize=(24, 20), sharex=True,
                              gridspec_kw={'height_ratios': [4, 2, 2, 1]})
    fig.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    n_regimes = len(regime_meta)

    # ═══ Panel 1: Price with golden path overlay ═══
    ax = axes[0]

    # Draw warmup (context) in gray
    if warmup_end > 1:
        ax.plot(x[:warmup_end], close[:warmup_end], color='#CCCCCC',
                linewidth=0.6, alpha=0.5, label='Context (warmup)')

    # Draw analysis week price colored by regime
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.plot(x[s:e+1], close[s:e+1], color=color, linewidth=1.0, alpha=0.6,
                label=f"R{rm['regime_id']} {rm['direction']} ({rm['n_bars']}b)")
        # Regime mean reference
        ax.hlines(y=rm['mean_price'], xmin=s, xmax=e, color=color,
                  linestyle=':', linewidth=0.6, alpha=0.4)

    # ── Golden path overlay (Level 1 + Level 2) ──
    # Map golden path segments (by timestamp) onto the bar index
    ts_to_idx = {}
    for i_bar, ts in enumerate(timestamps):
        ts_to_idx[int(ts)] = i_bar

    # Pre-sort bar timestamps for faster lookup
    sorted_bar_ts = sorted(ts_to_idx.keys())

    # $30 MNQ = 60 ticks (tick_size=0.25, tick_value=$0.50)
    L2_MIN_TICKS = 60  # $30 threshold
    L2_DOLLAR = L2_MIN_TICKS * 0.50

    n_l1_long = 0
    n_l1_short = 0
    n_l2_long = 0
    n_l2_short = 0
    total_l1_captured = 0.0
    total_l2_captured = 0.0

    def _find_bar_idx(seg_ts):
        """Find the 1m bar index containing this 1s timestamp."""
        for bar_ts in sorted_bar_ts:
            if bar_ts <= seg_ts <= bar_ts + 60:
                return ts_to_idx[bar_ts]
        return None

    # Two passes: Level 1 first (thin, background), Level 2 on top (thick, prominent)
    for level in (1, 2):
        for seg in golden_segments:
            is_l2 = seg['captured_ticks'] >= L2_MIN_TICKS

            # Level 1 pass: draw all segments thin
            # Level 2 pass: only draw $30+ segments thick
            if level == 1 and is_l2:
                continue  # will draw in Level 2 pass
            if level == 2 and not is_l2:
                continue

            start_idx = _find_bar_idx(seg['ts_start'])
            end_idx = _find_bar_idx(seg['ts_end'])

            if start_idx is None or end_idx is None:
                continue
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            if end_idx >= n:
                continue

            seg_x = x[start_idx:end_idx+1]
            seg_y = close[start_idx:end_idx+1]

            if len(seg_x) < 2:
                continue

            is_long = seg['direction'] == 'LONG'
            pnl_dollars = seg['captured_ticks'] * 0.50

            if level == 1:
                # Level 1: thin, semi-transparent
                color = '#81C784' if is_long else '#E57373'  # muted green/red
                ax.plot(seg_x, seg_y, color=color, linewidth=1.5, alpha=0.4, zorder=8)
                if is_long:
                    n_l1_long += 1
                else:
                    n_l1_short += 1
                total_l1_captured += seg['captured_ticks']
            else:
                # Level 2: thick, bold, with $ label
                color = '#00C853' if is_long else '#FF1744'  # bright green/red
                ax.plot(seg_x, seg_y, color=color, linewidth=4.5, alpha=0.85, zorder=12)

                # Arrow showing direction
                ax.annotate('',
                            xy=(seg_x[min(2, len(seg_x)-1)], seg_y[min(2, len(seg_y)-1)]),
                            xytext=(seg_x[0], seg_y[0]),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2.5),
                            zorder=13)

                # Dollar label at midpoint
                mid = len(seg_x) // 2
                y_offset = 3 if is_long else -3
                ax.annotate(f'${pnl_dollars:.0f}',
                            xy=(seg_x[mid], seg_y[mid]),
                            xytext=(0, y_offset),
                            textcoords='offset points',
                            fontsize=7, fontweight='bold', color=color,
                            ha='center', va='bottom' if is_long else 'top',
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec=color, alpha=0.85, linewidth=0.5),
                            zorder=14)

                if is_long:
                    n_l2_long += 1
                else:
                    n_l2_short += 1
                total_l2_captured += seg['captured_ticks']

    # Golden path legend entries
    ax.plot([], [], color='#81C784', linewidth=1.5, alpha=0.5,
            label=f'L1 All ({n_l1_long + n_l1_short} segs, '
                  f'${total_l1_captured * 0.50:,.0f})')
    ax.plot([], [], color='#00C853', linewidth=4.5,
            label=f'L2 LONG ${L2_DOLLAR:.0f}+ ({n_l2_long} segs, '
                  f'${total_l2_captured * 0.50:,.0f})')
    ax.plot([], [], color='#FF1744', linewidth=4.5,
            label=f'L2 SHORT ${L2_DOLLAR:.0f}+ ({n_l2_short} segs, '
                  f'${total_l2_captured * 0.50:,.0f})')

    # Regime boundaries
    for rm in regime_meta[1:]:
        for a in axes:
            a.axvline(x=rm['start_idx'], color='#888888', linestyle=':',
                      linewidth=0.5, alpha=0.4)

    # Warmup boundary
    for a in axes:
        a.axvline(x=warmup_end, color='#333333', linestyle='--',
                  linewidth=1, alpha=0.5)

    # I-chart control limits on price
    ax.axhline(y=ucl_i, color='#AA0000', linestyle='--', linewidth=0.8,
               alpha=0.4, label=f'UCL={ucl_i:.1f}')
    ax.axhline(y=lcl_i, color='#AA0000', linestyle='--', linewidth=0.8,
               alpha=0.4, label=f'LCL={lcl_i:.1f}')
    ax.axhline(y=center, color='#666666', linestyle='-', linewidth=0.8,
               alpha=0.3, label=f'Center={center:.1f}')

    week_start_dt = datetime.fromtimestamp(week_start, tz=timezone.utc)
    week_end_dt = datetime.fromtimestamp(week_end, tz=timezone.utc)

    total_all = n_l1_long + n_l1_short + n_l2_long + n_l2_short
    total_l2 = n_l2_long + n_l2_short
    ax.set_title(f'PRICE (1m Close) + GOLDEN PATH — '
                 f'{week_start_dt:%Y-%m-%d} to {week_end_dt:%Y-%m-%d}\n'
                 f'L1: {total_all} segments | '
                 f'L2 (${L2_DOLLAR:.0f}+): {total_l2} trades, '
                 f'${total_l2_captured * 0.50:,.0f} captured | '
                 f'{n_regimes} regimes',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(fontsize=7, loc='upper left', ncol=min(n_regimes + 4, 8))
    ax.grid(True, alpha=0.12)

    # ═══ Panel 2: I chart ═══
    ax = axes[1]
    # Gray for warmup, colored for analysis
    ax.plot(x[:warmup_end], close[:warmup_end], color='#CCCCCC',
            linewidth=0.4, alpha=0.4)

    inside = (close <= ucl_i) & (close >= lcl_i)
    outside = ~inside

    # Analysis bars as dots colored by regime
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        mask = np.zeros(n, dtype=bool)
        mask[s:e+1] = True
        in_mask = mask & inside
        out_mask = mask & outside
        if in_mask.any():
            ax.scatter(x[in_mask], close[in_mask], color=color, s=4, zorder=4, alpha=0.6)
        if out_mask.any():
            ax.scatter(x[out_mask], close[out_mask], color='black', s=8, zorder=5,
                       marker='x', linewidths=0.8)

    ax.axhline(y=center, color='#888888', linestyle='-', linewidth=1.2, alpha=0.5,
               label=f'Center={center:.1f}')
    ax.axhline(y=ucl_i, color='#AA0000', linestyle='--', linewidth=0.8, alpha=0.5,
               label=f'UCL={ucl_i:.1f}')
    ax.axhline(y=lcl_i, color='#AA0000', linestyle='--', linewidth=0.8, alpha=0.5,
               label=f'LCL={lcl_i:.1f}')

    ax.set_title('I CHART — Individual 1m Close (dots colored by regime, x = outside limits)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Close', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.12)

    # ═══ Panel 3: MR chart ═══
    ax = axes[2]

    # Color bars by sign
    colors_mr = np.where(mr >= 0, '#4CAF50', '#F44336')
    ax.bar(x, mr, color=colors_mr, width=1.0, alpha=0.6, edgecolor='none')

    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.6)
    ax.axhline(y=ucl_mr, color='#AA0000', linestyle='--', linewidth=0.8, alpha=0.5,
               label=f'+UCL={ucl_mr:.2f}')
    ax.axhline(y=-ucl_mr, color='#AA0000', linestyle='--', linewidth=0.8, alpha=0.5,
               label=f'-UCL={-ucl_mr:.2f}')
    ax.axhline(y=mr_bar, color='#888888', linestyle=':', linewidth=0.8, alpha=0.4,
               label=f'MR_bar={mr_bar:.2f}')
    ax.axhline(y=-mr_bar, color='#888888', linestyle=':', linewidth=0.8, alpha=0.4)

    # Mark UCL breaks
    mr_break = mr_abs > ucl_mr
    if mr_break.any():
        ax.scatter(x[mr_break], mr[mr_break], color='black', s=10, zorder=5,
                   marker='x', linewidths=0.8, label='UCL break')

    ax.set_title('MR CHART — Signed Moving Range (green=up, red=down)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Price Change', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.12)

    # ═══ Panel 4: Regime map ═══
    ax = axes[3]
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.barh(0, e - s + 1, left=s, height=0.8, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        mid = (s + e) / 2
        label_text = f"R{rm['regime_id']}\n{rm['direction']}\n{rm['n_bars']}b"
        if rm['n_bars'] > 20:
            ax.text(mid, 0, label_text, ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

    ax.set_yticks([])
    ax.set_title('REGIME MAP — Natural price segments from MR UCL breaks',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Bar index (1m)', fontsize=10)

    # Date tick labels
    if date_positions:
        step = max(1, len(date_positions) // 20)
        ax.set_xticks([date_positions[i] for i in range(0, len(date_positions), step)])
        ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), step)],
                           rotation=45, ha='right', fontsize=8)

    fig.suptitle(f'1-MINUTE I-MR CHART + GOLDEN PATH\n'
                 f'{n} bars | Warmup: {warmup_end} bars | '
                 f'UCL_MR={ucl_mr:.2f} | {n_regimes} regimes',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='I-MR + Golden Path (1m)')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--month', default=None,
                        help='Restrict random pick to this month (e.g., 2025_07)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--date', default=None,
                        help='Target date (YYYY-MM-DD) — centers analysis around this day')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup context days before analysis week')
    parser.add_argument('--analysis-days', type=int, default=7,
                        help='Analysis window in calendar days')
    args = parser.parse_args()

    print("=" * 70)
    print("I-MR CHART + GOLDEN PATH OVERLAY (1-minute)")
    print("=" * 70)

    # Load 1m data
    print("\n[1] Loading 1m ATLAS data...")
    months = [args.month] if args.month else None
    df_1m = load_atlas_tf(args.data_dir, '1m', months=months)
    if df_1m.empty:
        print("ERROR: No 1m data found")
        sys.exit(1)
    print(f"  Loaded {len(df_1m):,} 1m bars")

    # Pick analysis window
    if args.date:
        print(f"\n[2] Targeting date: {args.date}")
        _target_dt = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        week_start = _target_dt.timestamp()
        week_end = week_start + args.analysis_days * 86400
        context_start = week_start - args.context_days * 86400
    else:
        print("\n[2] Picking random week...")
        context_start, week_start, week_end = pick_random_week(
            df_1m, seed=args.seed)

    ctx_dt = datetime.fromtimestamp(context_start, tz=timezone.utc)
    ws_dt = datetime.fromtimestamp(week_start, tz=timezone.utc)
    we_dt = datetime.fromtimestamp(week_end, tz=timezone.utc)
    print(f"  Context starts: {ctx_dt:%Y-%m-%d %H:%M}")
    print(f"  Analysis week:  {ws_dt:%Y-%m-%d %H:%M} to {we_dt:%Y-%m-%d %H:%M}")

    # Filter 1m data to context + analysis window
    mask = (df_1m['timestamp'] >= context_start) & (df_1m['timestamp'] < week_end)
    df_context = df_1m[mask].reset_index(drop=True)
    print(f"  Context + analysis: {len(df_context):,} 1m bars")

    # Compute I-MR
    print("\n[3] Computing price I-MR chart...")
    price_imr = compute_price_imr(
        df_context,
        context_days=args.context_days,
        analysis_days=args.analysis_days
    )

    # Detect regimes
    print("\n[4] Detecting regimes...")
    regime_ids, regime_meta = detect_regimes(price_imr, min_regime_bars=8)

    # Compute golden path from 1s data
    print("\n[5] Computing golden path from 1s data...")
    index_1s = load_1s_index(args.data_dir)
    golden_segments, prices_1s, ts_1s = compute_golden_path_segments(
        index_1s, week_start, week_end)

    # Plot
    print("\n[6] Generating plot...")
    os.makedirs('tools/plots/standalone/imr_golden', exist_ok=True)
    seed_label = args.seed if args.seed else 'random'
    output_path = (f'tools/plots/standalone/imr_golden/'
                   f'imr_golden_{ws_dt:%Y%m%d}_{seed_label}.png')
    plot_imr_golden_path(df_context, price_imr, regime_ids, regime_meta,
                         golden_segments, week_start, week_end, output_path)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Week:     {ws_dt:%Y-%m-%d} to {we_dt:%Y-%m-%d}")
    print(f"  1m bars:  {len(df_context):,} (context + analysis)")
    print(f"  Regimes:  {len(regime_meta)}")

    analysis_mask = price_imr['analysis_mask']
    analysis_close = price_imr['close'][analysis_mask]
    if len(analysis_close) > 1:
        week_return = analysis_close[-1] - analysis_close[0]
        week_range = analysis_close.max() - analysis_close.min()
        print(f"  Week return:  {week_return:+.2f} pts")
        print(f"  Week range:   {week_range:.2f} pts")

    if golden_segments:
        total_cap = sum(s['captured_ticks'] for s in golden_segments)
        n_long = sum(1 for s in golden_segments if s['direction'] == 'LONG')
        n_short = sum(1 for s in golden_segments if s['direction'] == 'SHORT')

        # Level 2 ($30+) = 60 ticks
        l2_segs = [s for s in golden_segments if s['captured_ticks'] >= 60]
        l2_cap = sum(s['captured_ticks'] for s in l2_segs)
        l2_long = sum(1 for s in l2_segs if s['direction'] == 'LONG')
        l2_short = sum(1 for s in l2_segs if s['direction'] == 'SHORT')

        print(f"  L1 All:       {len(golden_segments)} segments "
              f"({n_long} LONG, {n_short} SHORT)")
        print(f"  L1 captured:  {total_cap:.0f} ticks "
              f"(${total_cap * 0.50:,.2f})")
        print(f"  L2 ($30+):    {len(l2_segs)} segments "
              f"({l2_long} LONG, {l2_short} SHORT)")
        print(f"  L2 captured:  {l2_cap:.0f} ticks "
              f"(${l2_cap * 0.50:,.2f})")
        if l2_segs:
            avg_l2 = l2_cap / len(l2_segs) * 0.50
            print(f"  L2 avg/trade: ${avg_l2:,.2f}")

    for rm in regime_meta:
        print(f"    R{rm['regime_id']}: {rm['n_bars']:>4} bars, "
              f"vol={rm['volatility']:.2f}, dir={rm['direction']}, "
              f"chg={rm['price_change']:+.1f}")


if __name__ == '__main__':
    main()
