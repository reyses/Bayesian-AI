"""
Multi-TF zigzag pivot overlay for ONE DAY.
==========================================

Eyeball test for the resonance cascade hypothesis: do zigzag pivots from
different TFs cluster around the same times, or are they scattered?

Plots:
  - Black line: 1m close (price backbone for the day)
  - Markers per TF: zigzag pivots placed at (extreme_time, extreme_price)
    * triangle UP   = low pivot  (= leg flips from down to up)
    * triangle DOWN = high pivot (= leg flips from up to down)
  - Color/size by TF (small/light = short TF, big/dark = long TF)

When a vertical "stack" of markers from multiple TFs aligns at roughly the
same time AND same direction, that's a candidate cascade event.

Usage:
    python tools/zigzag_multitf_overlay.py --date 2026-02-09
    python tools/zigzag_multitf_overlay.py --date 2026-04-24 --tfs 1m 5m 15m 1h
    python tools/zigzag_multitf_overlay.py --date 2026-02-09 --r 30 --out custom.png

Default TFs: 1m, 5m, 15m, 1h. Default R = 30 points (MNQ standard).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.zigzag_backtest import zigzag_pivots_with_confirmation


# Map TF label -> ATLAS subdir + period in seconds
TF_DIRS = {
    '15s': ('15s', 15),
    '30s': ('30s', 30),
    '1m':  ('1m', 60),
    '5m':  ('5m', 300),
    '15m': ('15m', 900),
    '30m': ('30m', 1800),
    '1h':  ('1h', 3600),
    '4h':  ('4h', 14400),
}

# (color, marker_size, line_width, alpha) — short TFs are thin/light, long TFs thick/bold
# so the visual "stack" reads bottom-up: macro structure on top, micro under.
TF_STYLE = {
    '15s': ('#cccccc', 30,  0.5, 0.45),
    '30s': ('#999999', 50,  0.7, 0.50),
    '1m':  ('#4daf4a', 60,  0.9, 0.55),   # green   — micro
    '5m':  ('#377eb8', 110, 1.4, 0.70),   # blue
    '15m': ('#e41a1c', 170, 2.0, 0.80),   # red
    '30m': ('#984ea3', 230, 2.4, 0.85),   # purple
    '1h':  ('#ff7f00', 320, 2.8, 0.90),   # orange  — macro
    '4h':  ('#000000', 450, 3.4, 0.95),   # black
}


def load_day_bars(atlas_root: str, day_label: str, tf: str) -> pd.DataFrame | None:
    if tf not in TF_DIRS:
        return None
    subdir, _ = TF_DIRS[tf]
    p = os.path.join(atlas_root, subdir, f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description='Multi-TF zigzag overlay for one day')
    ap.add_argument('--atlas', default='DATA/ATLAS')
    ap.add_argument('--date', required=True, help='Date as YYYY-MM-DD')
    ap.add_argument('--tfs', nargs='+', default=['1m', '5m', '15m', '1h'],
                    help='Timeframes to overlay (default: 1m 5m 15m 1h)')
    ap.add_argument('--r', type=float, default=30.0, help='Zigzag R in points (MNQ)')
    ap.add_argument('--out', default=None, help='Output PNG path')
    args = ap.parse_args()

    day_label = args.date.replace('-', '_')
    out_path = args.out or f'reports/findings/multitf_overlay_{day_label}.png'

    # Backbone: 1m close as a continuous price line
    df_1m = load_day_bars(args.atlas, day_label, '1m')
    if df_1m is None:
        print(f'[ERROR] No 1m parquet for {day_label}')
        sys.exit(1)

    times_1m = pd.to_datetime(df_1m['timestamp'].values, unit='s', utc=True)
    closes_1m = df_1m['close'].values.astype(np.float64)

    fig, (ax, ax_lanes) = plt.subplots(
        2, 1, figsize=(18, 10),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    ax.plot(times_1m, closes_1m, color='black', linewidth=0.5, alpha=0.6, label='1m close')

    # Per-TF pivot overlay on price chart + lanes panel
    summary_rows = []
    lane_y = {tf: i for i, tf in enumerate(args.tfs)}

    # Sort TFs short->long so longer-TF lines render ON TOP (zorder grows w/ index).
    tfs_sorted = sorted(args.tfs, key=lambda t: TF_DIRS.get(t, ('', 0))[1])

    for z_order, tf in enumerate(tfs_sorted, start=2):
        df_tf = load_day_bars(args.atlas, day_label, tf)
        if df_tf is None or len(df_tf) < 2:
            print(f'  [{tf}] no parquet -- skipped')
            continue
        closes = df_tf['close'].values.astype(np.float64)
        ts_tf = df_tf['timestamp'].values.astype(np.int64)
        period_s = TF_DIRS[tf][1]

        pivots = zigzag_pivots_with_confirmation(closes, args.r)
        if not pivots:
            print(f'  [{tf}] 0 pivots at R={args.r}')
            continue

        color, base_size, lw, alpha = TF_STYLE.get(tf, ('gray', 80, 1.0, 0.7))

        # Place each marker at (extreme bar's CLOSE time, extreme price).
        # ts_tf[i] is bar START; close is at start + period_s.
        ext_times = pd.to_datetime(
            [int(ts_tf[ext_idx]) + period_s for ext_idx, _, _, _ in pivots],
            unit='s', utc=True,
        )
        ext_prices = [ext_price for _, ext_price, _, _ in pivots]
        ext_kinds = [kind for _, _, kind, _ in pivots]

        # ── ZIGZAG POLYLINE ────────────────────────────────────────────────
        # Connect pivot extremes in order. This is the actual zigzag at this TF.
        # Higher TFs draw thicker/darker so the macro skeleton rises to the top of
        # the visual stack and the micro structure fills in beneath it.
        ax.plot(ext_times, ext_prices, color=color, linewidth=lw, alpha=alpha,
                zorder=z_order, solid_joinstyle='round', solid_capstyle='round')

        # Markers at each pivot — UP triangle for lows, DOWN triangle for highs.
        for t, p, k in zip(ext_times, ext_prices, ext_kinds):
            mk = '^' if k == 'low' else 'v'
            ax.scatter(t, p, s=base_size, marker=mk, c=color, alpha=min(1.0, alpha + 0.1),
                       edgecolors='black', linewidths=0.6, zorder=z_order + 10)

        # Lanes panel: timeline of pivots per TF.
        for t, k in zip(ext_times, ext_kinds):
            ax_lanes.scatter(t, lane_y[tf], s=base_size * 0.8,
                             marker=('^' if k == 'low' else 'v'),
                             c=color, alpha=0.9,
                             edgecolors='black', linewidths=0.5)

        n_low = sum(1 for _, _, k, _ in pivots if k == 'low')
        n_high = sum(1 for _, _, k, _ in pivots if k == 'high')
        summary_rows.append((tf, len(pivots), n_low, n_high))
        # Legend handle that shows BOTH the line AND a marker.
        ax.plot([], [], color=color, linewidth=lw, marker='s', markersize=8,
                markeredgecolor='black', markeredgewidth=0.6,
                label=f'{tf}: {len(pivots)} pivots ({n_low}L/{n_high}H)')

    # Format price chart
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=None))
    ax.set_title(f'Multi-TF zigzag pivots overlay — {args.date}  (R={args.r:g} pts, MNQ)',
                 fontsize=13, weight='bold')
    ax.set_ylabel('Price')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Format lanes panel
    ax_lanes.set_yticks(list(lane_y.values()))
    ax_lanes.set_yticklabels(list(lane_y.keys()))
    ax_lanes.set_ylabel('TF')
    ax_lanes.set_xlabel('Time (UTC)')
    ax_lanes.set_ylim(-0.5, len(args.tfs) - 0.5)
    ax_lanes.invert_yaxis()
    ax_lanes.grid(alpha=0.3)
    ax_lanes.set_title('Pivot lane chart — vertical alignment = simultaneous pivots across TFs', fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote: {out_path}')

    # Console summary
    print(f'\n{args.date} — pivot counts at R={args.r:g}:')
    for tf, total, n_low, n_high in summary_rows:
        print(f'  {tf:>4}: {total:>3} pivots  ({n_low} lows / {n_high} highs)')


if __name__ == '__main__':
    main()
