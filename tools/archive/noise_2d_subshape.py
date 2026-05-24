"""Discover finer 2D sub-shapes within a NOISE bucket — pure shape clustering.

The 20-shape primitive library (LINEAR/EXPONENTIAL/ROUNDED_U/etc.) is too
coarse. NOISE = phrases that don't match any of those 20 templates above
r=0.75. But many NOISE phrases ARE shape-coherent — they just don't fit
the predefined library. This tool discovers EMPIRICAL sub-shapes by
clustering the trajectories themselves.

Algorithm (pure 2D, no features):
    1. For each NOISE phrase, extract M_5s line over the segment span
    2. Resample to fixed length (16 points) and normalize 0-1
    3. Pairwise Pearson correlation between resampled trajectories
    4. Hierarchical agglomerative clustering on (1 - r) distance
    5. Cut at r >= cluster_threshold (default 0.85)
    6. Each resulting cluster = an empirical sub-shape

Output: cluster assignments per phrase + visualization of cluster centroids
+ sample trajectories per cluster overlaid.

Stays purely in 2D price-time space. No feature aggregation, no chord
fingerprint, no UMAP/HDBSCAN.

USAGE
    python tools/noise_2d_subshape.py --shape LINEAR_DOWN
    python tools/noise_2d_subshape.py --shape FLATLINE --threshold 0.80
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s, TF_WINDOW, PERIOD_S
)


def _extract_trajectory(day: str, start_ts: int, end_ts: int,
                         tf: str = '15m', n_points: int = 16) -> np.ndarray:
    """Extract M_5s line over [start_ts, end_ts] resampled to n_points,
    normalized 0-1. Returns NaN array if missing."""
    df_5s = _load_5s(day)
    if df_5s.empty:
        return np.full(n_points, np.nan)
    ts_5s = df_5s['timestamp'].values.astype(np.int64)
    oh = _load_tf_ohlcv(tf, day)
    if oh.empty:
        return np.full(n_points, np.nan)
    N = TF_WINDOW[tf]
    M = oh['close'].rolling(N, min_periods=2).mean().values
    tf_ts = oh['timestamp'].values.astype(np.int64)
    M5s = _ffill_to_5s(M, tf_ts, ts_5s, PERIOD_S[tf])
    mask = (ts_5s >= start_ts) & (ts_5s <= end_ts)
    seg = M5s[mask]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 4:
        return np.full(n_points, np.nan)
    src_x = np.linspace(0, 1, len(seg))
    tgt_x = np.linspace(0, 1, n_points)
    resampled = np.interp(tgt_x, src_x, seg)
    mn, mx = resampled.min(), resampled.max()
    if mx - mn < 1e-9:
        return np.zeros(n_points)
    return (resampled - mn) / (mx - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/layer1b/phrases_with_15m_clusters.csv')
    ap.add_argument('--full-phrase-csv',
        default='reports/findings/segments/all_motifs_labeled.csv',
        help='Provides start_ts/end_ts columns missing from layer1b CSV')
    ap.add_argument('--shape', default='LINEAR_DOWN',
                    help='Inspect NOISE bucket of this shape')
    ap.add_argument('--threshold', type=float, default=0.85,
                    help='Pearson r threshold for clustering (cut hierarchical at 1-r)')
    ap.add_argument('--n-points', type=int, default=16,
                    help='Resample length for trajectories')
    ap.add_argument('--min-cluster-size', type=int, default=8,
                    help='Min phrases per cluster to keep')
    ap.add_argument('--split', default='IS')
    ap.add_argument('--out-dir', default='chart/segments/noise_2d_subshape')
    args = ap.parse_args()

    df = pd.read_csv(args.phrase_csv)
    full = pd.read_csv(args.full_phrase_csv)
    df = df.merge(full[['day', 'seg_idx', 'start_ts', 'end_ts',
                         'slope_pts_per_min', 'mean_sigma', 'r2adj',
                         'shape_pearson_r']],
                   on=['day', 'seg_idx'], how='left',
                   suffixes=('', '_full'))
    if args.split and 'split' in df.columns:
        df = df[df['split'] == args.split]
    noise = df[(df['shape_class'] == args.shape) & (df['cluster_15m'] == -1)].reset_index(drop=True)
    print(f'NOISE bucket for {args.shape}: {len(noise)} phrases')
    if len(noise) < args.min_cluster_size * 2:
        print('Not enough phrases for clustering.'); sys.exit(1)

    # Extract trajectories
    print(f'Extracting trajectories ({args.n_points} points each)...')
    trajectories = []
    valid_idx = []
    for i, r in noise.iterrows():
        traj = _extract_trajectory(r['day'], int(r['start_ts']), int(r['end_ts']),
                                    n_points=args.n_points)
        if np.all(np.isfinite(traj)) and traj.std() > 1e-9:
            trajectories.append(traj)
            valid_idx.append(i)
    trajectories = np.array(trajectories)
    print(f'Valid trajectories: {len(trajectories)} of {len(noise)}')
    if len(trajectories) < args.min_cluster_size * 2:
        print('Too few valid trajectories.'); sys.exit(1)

    # Pearson correlation distance matrix
    print('Computing pairwise correlations...')
    corr = np.corrcoef(trajectories)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # ensure symmetry
    dist = np.clip(dist, 0, 2)
    condensed = squareform(dist, checks=False)

    # Hierarchical agglomerative clustering
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=(1.0 - args.threshold), criterion='distance')
    print(f'\nHierarchical clustering at r >= {args.threshold} (i.e. distance <= {1-args.threshold})')

    # Filter clusters by min size
    from collections import Counter
    label_counts = Counter(cluster_labels)
    keep = {l for l, n in label_counts.items() if n >= args.min_cluster_size}
    final_labels = np.array([l if l in keep else 0 for l in cluster_labels])
    n_clusters = len(keep)
    n_noise = (final_labels == 0).sum()
    print(f'Clusters with n>={args.min_cluster_size}: {n_clusters}')
    print(f'Re-NOISE (clusters too small to keep): {n_noise}')

    # Cluster summary
    print('\nPer-cluster summary:')
    for cid in [0] + sorted(keep):
        mask = final_labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        cid_str = 'NOISE' if cid == 0 else f'subC{cid}'
        # mean trajectory shape signature
        avg_traj = trajectories[mask].mean(axis=0)
        # describe shape: monotonic? V-shape? etc.
        diffs = np.diff(avg_traj)
        n_sign_changes = int(((diffs[1:] * diffs[:-1]) < 0).sum())
        net = float(avg_traj[-1] - avg_traj[0])
        peak_idx = int(np.argmax(np.abs(avg_traj - 0.5)))
        print(f'  {cid_str:>8s}  n={n:>4}  '
              f'net_norm={net:+.2f}  n_sign_flips={n_sign_changes}  '
              f'peak_idx={peak_idx}/{args.n_points-1}')

    # Render: for each cluster, overlay sample trajectories + centroid
    cols = 3
    rows = max(1, int(np.ceil((n_clusters + 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
                              squeeze=False)
    axes_flat = axes.ravel()
    panel = 0

    cmap = plt.get_cmap('tab10')
    cluster_ids_sorted = sorted(keep) + [0]  # last = re-NOISE
    for ci, cid in enumerate(cluster_ids_sorted):
        mask = final_labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        ax = axes_flat[panel]; panel += 1
        # Show all trajectories in cluster (light) + centroid (bold)
        traj_in_cluster = trajectories[mask]
        for t in traj_in_cluster[:50]:  # cap render to 50 traces
            ax.plot(t, color=cmap(ci % 10) if cid != 0 else '#999', lw=0.5, alpha=0.35)
        centroid = traj_in_cluster.mean(axis=0)
        ax.plot(centroid, color=cmap(ci % 10) if cid != 0 else '#444',
                lw=2.5, alpha=1.0, label='centroid')
        cid_str = 'NOISE_residual' if cid == 0 else f'subC{cid}'
        ax.set_title(f'{args.shape} {cid_str}\nn={n} (showing up to 50 traces)',
                      fontsize=10)
        ax.set_xlabel('resampled time index'); ax.set_ylabel('normalized price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    for j in range(panel, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'2D sub-shape clustering of NOISE bucket: {args.shape}\n'
                 f'Pearson r >= {args.threshold} hierarchical clustering on '
                 f'normalized M_5s trajectories ({args.n_points} points). Pure 2D.',
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'subshape_{args.shape}_r{int(args.threshold*100)}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out}')

    # Also save per-phrase mapping
    noise_out = noise.iloc[valid_idx].copy()
    noise_out['subshape_id'] = final_labels
    map_csv = os.path.join(args.out_dir,
                           f'subshape_map_{args.shape}_r{int(args.threshold*100)}.csv')
    noise_out[['day', 'seg_idx', 'shape_class', 'subshape_id',
                'length_min', 'shape_pearson_r']].to_csv(map_csv, index=False)
    print(f'Mapping -> {map_csv}')


if __name__ == '__main__':
    main()
