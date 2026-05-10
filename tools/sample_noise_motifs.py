"""Sample NOISE motifs (shape_class='NOISE') and render examples in a grid.

Reads `reports/findings/segments/all_motifs_labeled.csv`, filters to NOISE
motifs, picks examples spanning different (length, peak_z, ride_pnl, sigma_rank)
combinations to illustrate what 40% of motifs actually look like.

Usage:
    python tools/sample_noise_motifs.py
    python tools/sample_noise_motifs.py --n 9
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s, TF_WINDOW, PERIOD_S
)


def sample_noise_examples(noise_df: pd.DataFrame, n: int) -> list[pd.Series]:
    """Pick N motifs spanning diverse (length, peak_z, ride_pnl) combos."""
    if len(noise_df) <= n:
        return [r for _, r in noise_df.iterrows()]

    # Six descriptive picks, then fill with random
    picks = []
    used = set()

    def _add(row):
        idx = row.name
        if idx not in used:
            picks.append(row)
            used.add(idx)

    # 1. Shortest length
    _add(noise_df.sort_values('length_min').iloc[0])
    # 2. Longest length
    _add(noise_df.sort_values('length_min', ascending=False).iloc[0])
    # 3. Highest peak |z|
    _add(noise_df.sort_values('peak_abs_z', ascending=False).iloc[0])
    # 4. Lowest peak |z|
    _add(noise_df.sort_values('peak_abs_z').iloc[0])
    # 5. Highest ride_pnl_pts
    _add(noise_df.sort_values('ride_pnl_pts', ascending=False).iloc[0])
    # 6. Most-negative ride_pnl_pts
    _add(noise_df.sort_values('ride_pnl_pts').iloc[0])
    # 7. Median length
    med_len = noise_df['length_min'].median()
    _add(noise_df.iloc[(noise_df['length_min'] - med_len).abs().argsort().values[0]])

    # Fill rest with random uniform across length quintiles
    remaining = n - len(picks)
    if remaining > 0:
        unused_df = noise_df.loc[~noise_df.index.isin(used)]
        if len(unused_df) > 0:
            sample = unused_df.sample(min(remaining, len(unused_df)), random_state=42)
            for _, r in sample.iterrows():
                _add(r)

    return picks[:n]


def render_panel(ax, motif: pd.Series):
    day = motif['day']
    df_5s = _load_5s(day)
    if df_5s.empty:
        ax.text(0.5, 0.5, f'no data: {day}', transform=ax.transAxes,
                ha='center', va='center')
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    dt_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s]
    close_5s = df_5s['close'].values

    ax.plot(dt_5s, close_5s, color='black', lw=0.4, alpha=0.85, label='5s close')

    # Overlay 15m and 5m means
    for tf, color, label in [('15m', '#1E88E5', '15m'), ('5m', '#FB8C00', '5m')]:
        oh = _load_tf_ohlcv(tf, day)
        if not oh.empty:
            N = TF_WINDOW[tf]
            M = oh['close'].rolling(N, min_periods=2).mean().values
            tf_ts = oh['timestamp'].values.astype(np.int64)
            M5s = _ffill_to_5s(M, tf_ts, ts_5s, PERIOD_S[tf])
            ax.plot(dt_5s, M5s, color=color, lw=1.0, alpha=0.85, label=f'{label} M_close')

    # Highlight the noise motif
    s_dt = datetime.fromtimestamp(int(motif['start_ts']), tz=timezone.utc)
    e_dt = datetime.fromtimestamp(int(motif['end_ts']), tz=timezone.utc)
    ax.axvspan(s_dt, e_dt, color='#9C27B0', alpha=0.25, label='NOISE motif')
    ax.axvline(s_dt, color='#9C27B0', lw=1.4, alpha=0.85)
    ax.axvline(e_dt, color='#9C27B0', lw=1.4, alpha=0.55, linestyle=':')

    title = (
        f'{day}  M{int(motif["seg_idx"])}  '
        f'{s_dt.strftime("%H:%M")}-{e_dt.strftime("%H:%M")}  '
        f'{motif["length_min"]:.0f}min\n'
        f'slope={motif["slope_pts_per_min"]:+.3f}/min  '
        f'r2={motif["r2adj"]:.2f}  pk_z={motif["peak_abs_z"]:.1f}  '
        f'sigma_rank={motif["sigma_rank_mid"]:.2f}\n'
        f'shape_r={motif["shape_pearson_r"]:+.2f} (NOISE = below 0.75)  '
        f'ride={motif["ride_pnl_pts"]:+.1f}pts  '
        f'mfe={motif["max_mfe_ride_pts"]:+.1f}/mae={motif["max_mae_ride_pts"]:+.1f}\n'
        f'cascade={motif["resolved_as_cascade"]}  '
        f'next_cont={motif["next_seg_continued"]}'
    )
    ax.set_title(title, fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv',
                    default='reports/findings/segments/all_motifs_labeled.csv')
    ap.add_argument('--out',
                    default='chart/segments/noise_motif_examples.png')
    ap.add_argument('--n', type=int, default=9)
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.split != 'BOTH':
        df = df[df['split'] == args.split]
    noise = df[df['shape_class'] == 'NOISE'].copy()
    if noise.empty:
        print('No NOISE motifs found')
        sys.exit(1)
    print(f'NOISE motif population: {len(noise)} ({args.split})')
    print(f'  length_min:    median={noise["length_min"].median():.0f}  '
          f'min={noise["length_min"].min():.0f}  max={noise["length_min"].max():.0f}')
    print(f'  peak_abs_z:    median={noise["peak_abs_z"].median():.2f}  '
          f'min={noise["peak_abs_z"].min():.2f}  max={noise["peak_abs_z"].max():.2f}')
    print(f'  ride_pnl_pts:  median={noise["ride_pnl_pts"].median():+.1f}  '
          f'min={noise["ride_pnl_pts"].min():+.1f}  max={noise["ride_pnl_pts"].max():+.1f}')
    print(f'  sigma_rank:    median={noise["sigma_rank_mid"].median():.2f}')
    print(f'  shape_pearson: median={noise["shape_pearson_r"].median():+.2f}  '
          f'(threshold for non-NOISE: r >= 0.75)')

    examples = sample_noise_examples(noise, args.n)
    print(f'\nRendering {len(examples)} examples...')
    for i, ex in enumerate(examples):
        print(f'  {i+1}. {ex["day"]}  M{int(ex["seg_idx"])}  '
              f'{ex["length_min"]:.0f}min  '
              f'pk_z={ex["peak_abs_z"]:.1f}  '
              f'ride={ex["ride_pnl_pts"]:+.1f}  '
              f'r={ex["shape_pearson_r"]:+.2f}')

    n = len(examples)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.5, rows * 4.0),
                              squeeze=False)
    for i, ex in enumerate(examples):
        render_panel(axes[i // cols][i % cols], ex)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f'NOISE motif examples (n={len(noise)} {args.split} total).  '
        f'These are 15m-CRM-anchored macro segments where shape_class did NOT '
        f'match any of 20 templates above r=0.75.\n'
        f'Purple span = the NOISE motif within its day context.',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {args.out}')


if __name__ == '__main__':
    main()
