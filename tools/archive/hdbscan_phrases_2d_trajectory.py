"""HDBSCAN on raw 2D trajectories (no feature aggregates).

For each phrase, take its M_close line over [start_ts, end_ts], resample to
N normalized points, and cluster the resulting trajectory vectors with
HDBSCAN. This is HDBSCAN STRAIGHT UP AT 2D SPACE — the input is the shape
of the line itself, not scalar summaries (slope, sigma, peak_z).

Per primitive shape (LINEAR_DOWN, FLATLINE, etc.) we run HDBSCAN within
the shape's phrases. The 2D trajectory IS the input.

Pipeline:
    1. For each phrase, extract resampled normalized M_close trajectory
    2. Per shape, take the matrix of trajectories (N_phrases x N_points)
    3. UMAP to 2D for density structure (still operating on trajectory rows)
    4. HDBSCAN on UMAP embedding
    5. Output cluster assignments + centroid trajectories

USAGE
    python tools/hdbscan_phrases_2d_trajectory.py --shape LINEAR_DOWN
    python tools/hdbscan_phrases_2d_trajectory.py  # all shapes
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import (
    _load_5s, _load_tf_ohlcv, _ffill_to_5s, TF_WINDOW, PERIOD_S
)


def _extract_normalized_trajectory(day: str, start_ts: int, end_ts: int,
                                    tf: str = '15m', n_points: int = 16) -> np.ndarray:
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
        default='reports/findings/segments/all_motifs_labeled.csv')
    ap.add_argument('--shape', default=None,
                    help='Limit to one shape; default: all shapes >= min_n')
    ap.add_argument('--min-shape-n', type=int, default=80)
    ap.add_argument('--n-points', type=int, default=16)
    ap.add_argument('--min-cluster-size', type=int, default=30)
    ap.add_argument('--min-samples', type=int, default=10)
    ap.add_argument('--umap-n-neighbors', type=int, default=15)
    ap.add_argument('--split', default='IS')
    ap.add_argument('--out-dir',
        default='reports/findings/segments/hdbscan_2d_trajectory')
    args = ap.parse_args()

    import hdbscan
    import umap.umap_ as umap

    df = pd.read_csv(args.phrase_csv)
    if args.split:
        df = df[df['split'] == args.split]
    print(f'Loaded {len(df)} phrases ({args.split})')

    if args.shape:
        shapes_to_run = [args.shape]
    else:
        shape_counts = df['shape_class'].value_counts()
        shapes_to_run = [s for s, n in shape_counts.items() if n >= args.min_shape_n]
        print(f'Shapes with >= {args.min_shape_n}: {shapes_to_run}')

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    for shape in shapes_to_run:
        sub = df[df['shape_class'] == shape].reset_index(drop=True)
        print(f'\n=== {shape}: extracting {len(sub)} 2D trajectories ===')
        trajectories = []
        keep_idx = []
        for i, r in sub.iterrows():
            traj = _extract_normalized_trajectory(
                r['day'], int(r['start_ts']), int(r['end_ts']),
                n_points=args.n_points)
            if np.all(np.isfinite(traj)):
                trajectories.append(traj)
                keep_idx.append(i)
        traj_mat = np.array(trajectories)
        sub_clean = sub.iloc[keep_idx].reset_index(drop=True)
        print(f'  {len(traj_mat)} valid trajectories')
        if len(traj_mat) < args.min_cluster_size * 2:
            print(f'  too few; skipping')
            continue

        n_neighbors = min(args.umap_n_neighbors, max(5, len(traj_mat) - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.05,
                             metric='euclidean', random_state=42, n_components=2)
        emb = reducer.fit_transform(traj_mat)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                      min_samples=args.min_samples,
                                      cluster_selection_method='eom')
        labels = clusterer.fit_predict(emb)
        n_clusters = len([c for c in set(labels) if c >= 0])
        n_noise = int((labels == -1).sum())
        print(f'  HDBSCAN: {n_clusters} clusters + {n_noise} NOISE ({100*n_noise/len(traj_mat):.1f}%)')

        # Per cluster: centroid trajectory + count
        for cid in sorted(set(labels)):
            mask = labels == cid
            n = int(mask.sum())
            cid_str = 'NOISE' if cid == -1 else f'C{cid}'
            print(f'    {cid_str:<6s} n={n}')
            summary_rows.append({
                'shape': shape, 'cluster_id': cid_str, 'n': n,
                'pct': round(100 * n / len(traj_mat), 1),
            })

        # Render: UMAP scatter + per-cluster centroid trajectories
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        cmap = plt.get_cmap('tab10')
        for cid in sorted(set(labels)):
            mask = labels == cid
            color = '#999' if cid == -1 else cmap(cid % 10)
            label_str = f'NOISE (n={mask.sum()})' if cid == -1 else f'C{cid} (n={mask.sum()})'
            axes[0].scatter(emb[mask, 0], emb[mask, 1], c=[color], s=14,
                            alpha=0.65, label=label_str, edgecolor='none')
        axes[0].set_title(f'{shape}: HDBSCAN at 2D-trajectory space\n'
                           f'(UMAP of {args.n_points}-point normalized M_close lines)\n'
                           f'n_total={len(traj_mat)}  clusters={n_clusters}  noise={n_noise}',
                           fontsize=11)
        axes[0].legend(loc='best', fontsize=9, markerscale=1.5)
        axes[0].set_xlabel('UMAP dim 1'); axes[0].set_ylabel('UMAP dim 2')
        axes[0].grid(True, alpha=0.25)

        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask = labels == cid
            color = cmap(cid % 10)
            traj_in = traj_mat[mask]
            for t in traj_in[:50]:
                axes[1].plot(t, color=color, lw=0.4, alpha=0.25)
            axes[1].plot(traj_in.mean(axis=0), color=color, lw=2.5,
                          label=f'C{cid} centroid (n={mask.sum()})')
        # NOISE centroid
        mask = labels == -1
        if mask.sum() > 0:
            traj_in = traj_mat[mask]
            for t in traj_in[:30]:
                axes[1].plot(t, color='#999', lw=0.4, alpha=0.20)
            axes[1].plot(traj_in.mean(axis=0), color='#444', lw=2.0,
                          linestyle='--',
                          label=f'NOISE centroid (n={mask.sum()})')
        axes[1].set_title(f'{shape}: per-cluster trajectory centroids\n'
                           f'(thin = sample traces, thick = centroid)',
                           fontsize=11)
        axes[1].set_xlabel('resampled time index')
        axes[1].set_ylabel('normalized price')
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.05, 1.05)

        plt.tight_layout()
        out = os.path.join(args.out_dir, f'2d_traj_hdbscan_{shape}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Chart -> {out}')

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
        print(f'\nSummary CSV -> {os.path.join(args.out_dir, "summary.csv")}')


if __name__ == '__main__':
    main()
