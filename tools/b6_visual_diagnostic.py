"""B6 visual diagnostic — chart per-day directional pivot predictions
in time-segmented panels so we can verify the model's calls visually.

Renders for each test day:
  - Day chopped into N segments (typically 60-min each)
  - One PNG per segment showing:
    * 1m candles
    * Zigzag swings (truth)
    * B6 P(PIVOT_TO_LONG) and P(PIVOT_TO_SHORT) ribbons above chart
    * Markers where B6 predicts high-confidence directional pivot

Tests several days stratified by regime (when regime label available).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless — saves PNGs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')
REGIME_CSV = Path('DATA/ATLAS/regime_labels_2d.csv')

SEGMENT_MIN = 90    # minutes per chart segment


def load_regime_label(day_str: str):
    """Lookup regime for day. day_str is YYYY_MM_DD format from NT8."""
    if not REGIME_CSV.exists():
        return None
    df = pd.read_csv(REGIME_CSV, usecols=['date', 'regime_2d'])
    iso_day = day_str.replace('_', '-')
    row = df[df['date'] == iso_day]
    if len(row) == 0:
        return None
    return str(row['regime_2d'].iloc[0])


def draw_candles(ax, bars: pd.DataFrame, width_seconds=50):
    if len(bars) == 0:
        return
    import matplotlib.dates as mdates
    w_days = width_seconds / 86400.0
    half_w = w_days / 2
    ts_num = mdates.date2num(bars['ts_dt'].values)
    o = bars['open'].values; c = bars['close'].values
    h = bars['high'].values; l = bars['low'].values
    up = c >= o
    ax.vlines(bars['ts_dt'].values, l, h, color='black',
              linewidth=0.5, alpha=0.7)
    for i in range(len(bars)):
        x = ts_num[i] - half_w
        bottom = min(o[i], c[i])
        height = max(abs(c[i] - o[i]), 0.01)
        color = '#d0d0d0' if up[i] else '#606060'
        rect = Rectangle((x, bottom), w_days, height,
                          facecolor=color, edgecolor='black',
                          linewidth=0.3, alpha=0.7, zorder=2)
        ax.add_patch(rect)


def render_day_segments(day: str, bars1m: pd.DataFrame,
                          truth_df: pd.DataFrame, b6_df: pd.DataFrame,
                          out_dir: Path, regime: str,
                          K_min: int = 10,
                          thr_high: float = 0.50,
                          thr_med: float = 0.40):
    """Render N segment PNGs for one day."""
    bars1m['ts_dt'] = pd.to_datetime(bars1m['timestamp'], unit='s')
    truth_df = truth_df.copy()
    truth_df['ts_dt'] = pd.to_datetime(truth_df['timestamp'], unit='s')
    b6_df = b6_df.copy()
    b6_df['ts_dt'] = pd.to_datetime(b6_df['timestamp'], unit='s')

    # Day pivots (truth swings)
    piv = truth_df[truth_df['is_pivot'] == 1].sort_values('timestamp')

    # Segment the day into SEGMENT_MIN windows
    ts_min = bars1m['timestamp'].min()
    ts_max = bars1m['timestamp'].max()
    seg_starts = list(range(ts_min, ts_max, SEGMENT_MIN * 60))

    n_segs = len(seg_starts)
    for si, seg_lo in enumerate(seg_starts):
        seg_hi = seg_lo + SEGMENT_MIN * 60
        seg_bars = bars1m[(bars1m['timestamp'] >= seg_lo) &
                           (bars1m['timestamp'] < seg_hi)]
        if len(seg_bars) < 5:
            continue
        seg_piv = piv[(piv['timestamp'] >= seg_lo) & (piv['timestamp'] < seg_hi)]
        seg_b6  = b6_df[(b6_df['timestamp'] >= seg_lo) &
                         (b6_df['timestamp'] < seg_hi)]

        fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(12, 6),
                                          gridspec_kw={'height_ratios': [1, 4]},
                                          sharex=True)
        plt.subplots_adjust(hspace=0.04, top=0.92)

        # === Top panel: B6 probability ribbons ===
        p_long_col  = f'p_PIVOT_TO_LONG_{K_min}m'
        p_short_col = f'p_PIVOT_TO_SHORT_{K_min}m'
        if p_long_col in seg_b6.columns and len(seg_b6) > 0:
            ax_top.fill_between(seg_b6['ts_dt'], 0, seg_b6[p_long_col],
                                 color='lime', alpha=0.5, label='P(LONG pivot)')
            ax_top.fill_between(seg_b6['ts_dt'], 0, -seg_b6[p_short_col],
                                 color='orangered', alpha=0.5,
                                 label='P(SHORT pivot)')
            ax_top.axhline(0, color='black', linewidth=0.5)
            ax_top.axhline( thr_high, color='lime',      linewidth=0.6, ls='--', alpha=0.6)
            ax_top.axhline(-thr_high, color='orangered', linewidth=0.6, ls='--', alpha=0.6)
        ax_top.set_ylim(-1.0, 1.0)
        ax_top.set_ylabel(f'B6 P (K={K_min}m)', fontsize=8)
        ax_top.legend(loc='upper left', fontsize=6)
        ax_top.grid(True, alpha=0.3)

        # === Bottom panel: candles + truth pivots ===
        draw_candles(ax, seg_bars)
        ylo = seg_bars['low'].min(); yhi = seg_bars['high'].max()
        ypad = (yhi - ylo) * 0.04
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.set_xlim(seg_bars['ts_dt'].iloc[0], seg_bars['ts_dt'].iloc[-1])

        # Mark truth pivots
        for _, p in seg_piv.iterrows():
            color = 'green' if p['pivot_dir'] == 'LONG' else 'red'
            marker = '^' if p['pivot_dir'] == 'LONG' else 'v'
            ax.scatter([p['ts_dt']], [p['pivot_price']], marker=marker,
                        s=80, color=color, edgecolor='white', linewidths=1.0,
                        zorder=10)

        # Mark B6 high-confidence directional fires
        if p_long_col in seg_b6.columns:
            high_long  = seg_b6[seg_b6[p_long_col]  >= thr_high]
            high_short = seg_b6[seg_b6[p_short_col] >= thr_high]
            # Place markers at price-low (for long-pivot warning) and price-high
            # (for short-pivot warning) so they don't visually clash with the
            # truth markers placed at pivot price.
            for _, r in high_long.iterrows():
                pos_bar = seg_bars[seg_bars['timestamp'] <= r['timestamp']].tail(1)
                if len(pos_bar) == 0:
                    continue
                y = pos_bar['low'].iloc[0] - 0.3 * ypad
                ax.scatter([r['ts_dt']], [y], marker='^', s=30,
                            color='lime', alpha=0.7, zorder=6)
            for _, r in high_short.iterrows():
                pos_bar = seg_bars[seg_bars['timestamp'] <= r['timestamp']].tail(1)
                if len(pos_bar) == 0:
                    continue
                y = pos_bar['high'].iloc[0] + 0.3 * ypad
                ax.scatter([r['ts_dt']], [y], marker='v', s=30,
                            color='orangered', alpha=0.7, zorder=6)

        ax.set_xlabel('time')
        ax.set_ylabel('price')
        ax.grid(True, alpha=0.3)
        title = f'{day}  seg {si+1}/{n_segs}  ({SEGMENT_MIN}min)'
        if regime:
            title += f'  regime={regime}'
        title += f'  thr_high={thr_high} K={K_min}m'
        fig.suptitle(title, fontsize=10)

        # Save
        seg_path = out_dir / f'{day}_seg{si+1:02d}.png'
        fig.savefig(seg_path, dpi=110, bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--b6-cache',
                    default='reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')
    ap.add_argument('--days', nargs='*', default=None,
                    help='Specific days to render (else picks several diverse)')
    ap.add_argument('--n-days', type=int, default=6,
                    help='Number of test days to render if --days not given')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--thr-high', type=float, default=0.50)
    ap.add_argument('--out-dir',
                    default='reports/findings/regret_oracle/b6_visual_diagnostic/')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading truth + B6...')
    truth = pd.read_parquet(args.truth)
    if not Path(args.b6_cache).exists():
        print(f'B6 cache not found: {args.b6_cache}')
        print('Run train_b6_directional_pivot.py first.')
        return
    b6 = pd.read_parquet(args.b6_cache)
    print(f'  truth: {len(truth):,}   B6: {len(b6):,}')

    # Pick days
    all_days = sorted(truth['day'].unique())
    if args.days:
        days = args.days
    else:
        # Try to pick a diverse set: stratify by regime when available
        regimes = {d: load_regime_label(d) for d in all_days}
        # Group days by regime
        groups = {}
        for d, r in regimes.items():
            groups.setdefault(r or 'UNKNOWN', []).append(d)
        # Pick one or two from each regime group
        days = []
        for r, ds in groups.items():
            days.extend(ds[:max(1, args.n_days // max(len(groups), 1))])
        days = days[:args.n_days]
    print(f'Rendering {len(days)} days × ~{1440//SEGMENT_MIN} segments each...')

    for day in days:
        bars1m_path = NT8_1M_DIR / f'{day}.parquet'
        if not bars1m_path.exists():
            print(f'  skip {day}: no 1m bars')
            continue
        bars1m = pd.read_parquet(bars1m_path)
        truth_day = truth[truth['day'] == day]
        b6_day    = b6[b6['day'] == day]
        regime = load_regime_label(day)
        regime_str = regime or 'unknown'
        day_out = out_dir / day
        day_out.mkdir(parents=True, exist_ok=True)
        print(f'  {day} [{regime_str}] ...')
        render_day_segments(day, bars1m, truth_day, b6_day, day_out,
                              regime_str, K_min=args.K,
                              thr_high=args.thr_high)

    print(f'\nWrote charts to: {out_dir}')


if __name__ == '__main__':
    main()
