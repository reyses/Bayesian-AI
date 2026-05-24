"""Look at what's INSIDE the NOISE buckets — they're not idiosyncratic, they're
the population HDBSCAN couldn't carve into density clusters with the chosen
parameters. Most of the trading-relevant structure lives there.

For a given primitive shape (e.g. LINEAR_DOWN), the NOISE bucket contains
phrases that didn't fit any HDBSCAN cluster. We visualize a sample to see
whether NOISE is genuinely heterogeneous or whether we missed a sub-pattern.

Pure 2D shape — no chord features. NO oracle PnL reporting at this stage.

USAGE
    python tools/inspect_noise_bucket.py --shape LINEAR_DOWN
    python tools/inspect_noise_bucket.py --shape LINEAR_DOWN --bucket-level 5m  # motif NOISE
    python tools/inspect_noise_bucket.py --parent-15m-cluster FLATLINE_C3
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


SEG_FEATURES = [
    'slope_pts_per_min', 'mean_sigma', 'sigma_rank_mid', 'r2adj',
    'length_min', 'peak_abs_z', 'net_move_pts', 'tod_start_hour_utc',
]


def render_segment_panel(ax, seg: pd.Series, level: str = 'phrase',
                          window_pad_min: float = 5.0):
    day = seg['day']
    df_5s = _load_5s(day)
    if df_5s.empty:
        ax.text(0.5, 0.5, f'no data: {day}', transform=ax.transAxes,
                ha='center', va='center')
        return
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    close_5s = df_5s['close'].values
    pad = int(window_pad_min * 60)
    t_start = int(seg['start_ts']) - pad
    t_end   = int(seg['end_ts']) + pad
    m = (ts_5s >= t_start) & (ts_5s <= t_end)
    if not m.any():
        ax.text(0.5, 0.5, 'window empty', transform=ax.transAxes,
                ha='center', va='center')
        return
    dt_w = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_5s[m]]
    cl_w = close_5s[m]
    ax.plot(dt_w, cl_w, color='black', lw=0.5, alpha=0.85, label='5s close')

    # Anchor TF mean: 15m for phrases, 5m for motifs
    anchor_tf = '15m' if level == 'phrase' else '5m'
    secondary_tf = '5m' if level == 'phrase' else '1m'
    for tf, color, lw in [(anchor_tf, '#1E88E5', 1.6), (secondary_tf, '#FB8C00', 1.0)]:
        oh = _load_tf_ohlcv(tf, day)
        if oh.empty:
            continue
        N = TF_WINDOW[tf]
        M = oh['close'].rolling(N, min_periods=2).mean().values
        tf_ts = oh['timestamp'].values.astype(np.int64)
        M5s = _ffill_to_5s(M, tf_ts, ts_5s, PERIOD_S[tf])[m]
        ax.plot(dt_w, M5s, color=color, lw=lw, alpha=0.85,
                label=f'{tf} M_close')

    # Highlight segment span
    s_dt = datetime.fromtimestamp(int(seg['start_ts']), tz=timezone.utc)
    e_dt = datetime.fromtimestamp(int(seg['end_ts']), tz=timezone.utc)
    ax.axvspan(s_dt, e_dt, color='#9C27B0', alpha=0.18, label='NOISE segment')
    ax.axvline(s_dt, color='#9C27B0', lw=1.2, alpha=0.85)
    ax.axvline(e_dt, color='#9C27B0', lw=1.0, alpha=0.55, linestyle=':')

    title = (f'{day}  seg{int(seg["seg_idx"])}  '
              f'{s_dt.strftime("%H:%M")}-{e_dt.strftime("%H:%M")}  '
              f'{seg["length_min"]:.0f}min\n'
              f'shape={seg["shape_class"]}  '
              f'slope={seg["slope_pts_per_min"]:+.3f}/min  '
              f'sigma={seg["mean_sigma"]:.2f}  '
              f'r2={seg["r2adj"]:.2f}\n'
              f'peak_z={seg["peak_abs_z"]:.1f}  '
              f'net_move={seg["net_move_pts"]:+.1f}  '
              f'tod={int(seg["tod_start_hour_utc"]):02d}h')
    ax.set_title(title, fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


def feature_distribution(df: pd.DataFrame, features: list) -> str:
    """Per-feature stats for a NOISE bucket."""
    lines = ['feature                  q10        q50        q90        std']
    lines.append('-' * 75)
    for f in features:
        if f not in df.columns:
            continue
        s = df[f].dropna()
        if len(s) == 0:
            continue
        lines.append(f'{f:<22s} {s.quantile(0.10):>+10.4f} '
                     f'{s.quantile(0.50):>+10.4f} {s.quantile(0.90):>+10.4f} '
                     f'{s.std():>+10.4f}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/layer1b/phrases_with_15m_clusters.csv')
    ap.add_argument('--motif-csv',
        default='reports/findings/segments/layer1b/motifs_with_5m_subclusters.csv')
    ap.add_argument('--shape', default='LINEAR_DOWN',
                    help='Phrase shape to inspect NOISE for')
    ap.add_argument('--bucket-level', default='15m', choices=['15m', '5m'],
                    help='15m = phrase-level NOISE; 5m = motif-level NOISE within parent')
    ap.add_argument('--parent-15m-cluster', default=None,
                    help='If bucket-level=5m, restrict to motifs whose parent_15m_cluster matches')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--split', default='IS')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.bucket_level == '15m':
        df = pd.read_csv(args.phrase_csv)
        full = pd.read_csv('reports/findings/segments/all_motifs_labeled.csv')
        new_cols = [c for c in ['mean_sigma', 'sigma_rank_mid', 'r2adj',
                                  'slope_pts_per_min', 'net_move_pts',
                                  'tod_start_hour_utc', 'start_ts', 'end_ts']
                     if c not in df.columns]
        df = df.merge(full[['day', 'seg_idx'] + new_cols],
                       on=['day', 'seg_idx'], how='left')
        # NOTE: the layer1b phrase CSV stores 'cluster_15m' as an integer -1 for NOISE
        noise = df[(df['shape_class'] == args.shape) & (df['cluster_15m'] == -1)]
        if 'split' in df.columns and args.split:
            noise = noise[noise['split'] == args.split] if 'split' in noise.columns else noise
        suptitle = f'15m PHRASE NOISE for shape={args.shape}'
        level = 'phrase'
    else:
        df = pd.read_csv(args.motif_csv)
        full = pd.read_csv('reports/findings/segments/all_melodies_labeled.csv')
        new_cols = [c for c in ['mean_sigma', 'sigma_rank_mid', 'r2adj',
                                  'slope_pts_per_min', 'net_move_pts',
                                  'tod_start_hour_utc', 'start_ts', 'end_ts']
                     if c not in df.columns]
        df = df.merge(full[['day', 'seg_idx', 'parent_motif_idx'] + new_cols],
                       on=['day', 'seg_idx', 'parent_motif_idx'], how='left')
        noise = df[df['cluster_5m'] == -1]
        if args.parent_15m_cluster:
            noise = noise[noise['parent_15m_cluster'] == args.parent_15m_cluster]
        if args.split and 'split' in noise.columns:
            noise = noise[noise['split'] == args.split]
        suptitle = (f'5m MOTIF NOISE'
                    + (f' within parent {args.parent_15m_cluster}'
                       if args.parent_15m_cluster else '')
                    + f' (shape={args.shape})')
        level = 'motif'

    print(f'NOISE bucket: {len(noise)} segments at {args.bucket_level} level')
    if noise.empty:
        print('No NOISE segments found for these filters.')
        sys.exit(1)

    # Feature distribution
    print('\nFeature distribution within NOISE:')
    print(feature_distribution(noise, SEG_FEATURES))

    # Sample N examples spread across length quintiles to ensure variety
    sorted_n = noise.sort_values('length_min').reset_index(drop=True)
    if len(sorted_n) <= args.n:
        picks = sorted_n
    else:
        idx = (np.linspace(0.05, 0.95, args.n) * (len(sorted_n) - 1)).round().astype(int)
        picks = sorted_n.iloc[idx].drop_duplicates(subset=['day', 'seg_idx']).head(args.n)
    print(f'\nRendering {len(picks)} samples spanning length q5 to q95...')
    for _, p in picks.iterrows():
        print(f'  {p["day"]}  seg{int(p["seg_idx"])}  '
              f'{p["length_min"]:.0f}min  '
              f'slope={p["slope_pts_per_min"]:+.3f}/min  '
              f'r2={p["r2adj"]:.2f}  pkz={p["peak_abs_z"]:.1f}')

    # Render
    n = len(picks)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 11, rows * 4.5),
                              squeeze=False)
    for i, (_, seg) in enumerate(picks.iterrows()):
        render_segment_panel(axes[i // cols][i % cols], seg, level=level)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(suptitle + f'  (n={len(noise)})\n'
                 'Pure 2D shape inspection. No chord/feature analysis.',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or f'chart/segments/noise_inspect_{args.bucket_level}_{args.shape}.png'
    if args.parent_15m_cluster:
        out = out.replace('.png', f'_parent_{args.parent_15m_cluster}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out}')


if __name__ == '__main__':
    main()
