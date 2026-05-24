"""Single-step HDBSCAN at the 15m phrase level — global, not per-shape.

Runs UMAP+HDBSCAN once on the ENTIRE IS phrase population (no shape
stratification). The clusters that emerge are the natural 15m-level
structure of the data. Then we cross-tabulate against shape_class to see
whether the natural clusters align with the shape templates or reveal a
different organization.

This is the focused 15m-only analysis. Per-shape HDBSCAN, multi-level
recursion, and motif-level drill-down all come AFTER we understand what's
happening at 15m.

USAGE
    python tools/hdbscan_phrases_global.py
    python tools/hdbscan_phrases_global.py --min-cluster-size 50
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


DEFAULT_FEATURES = [
    'slope_15m__mean', 'slope_15m__std',
    'z_close_15m__mean', 'z_close_15m__std',
    'sigma_rank_15m__mean', 'sigma_rank_15m__std',
    'slope_5m__mean', 'slope_5m__std',
    'z_close_5m__mean', 'z_close_5m__std',
    'sigma_rank_5m__mean', 'sigma_rank_5m__std',
    'r2adj_5m__mean', 'r2adj_5m__std',
    'length_min', 'peak_abs_z',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/all_motifs_labeled_with_chord.csv')
    ap.add_argument('--min-cluster-size', type=int, default=50)
    ap.add_argument('--min-samples', type=int, default=15)
    ap.add_argument('--umap-n-neighbors', type=int, default=30)
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--out-dir',
        default='reports/findings/segments/hdbscan_global')
    args = ap.parse_args()

    import hdbscan
    import umap.umap_ as umap

    df = pd.read_csv(args.phrase_csv)
    if args.split != 'BOTH':
        df = df[df['split'] == args.split]
    print(f'Loaded {len(df)} phrases ({args.split})')

    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    print(f'Features ({len(feature_cols)}): {feature_cols}')

    X = df[feature_cols].values.astype(np.float64)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    sub = df[finite].reset_index(drop=True)
    print(f'Finite rows: {len(sub)}')

    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-9] = 1.0
    X_std = (X - mu) / sd

    print(f'Running UMAP (n_neighbors={args.umap_n_neighbors})...')
    reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors, min_dist=0.05,
                        metric='euclidean', random_state=42, n_components=2)
    emb = reducer.fit_transform(X_std)

    print(f'Running HDBSCAN (min_cluster_size={args.min_cluster_size})...')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                 min_samples=args.min_samples,
                                 cluster_selection_method='eom')
    labels = clusterer.fit_predict(emb)
    sub['cluster_id'] = labels

    cluster_ids = sorted(set(labels))
    n_clusters = len([c for c in cluster_ids if c >= 0])
    n_noise = int((labels == -1).sum())
    print(f'\nResult: {n_clusters} clusters + {n_noise} NOISE ({100*n_noise/len(sub):.1f}%)')

    # Per-cluster summary: size, ride PnL distribution, shape composition
    print('\nPer-cluster summary:')
    print('=' * 110)
    print(f'{"cid":>4s} {"n":>6s} {"%":>5s} '
          f'{"ride_mean":>10s} {"ride_med":>9s} {"ride_q10":>9s} {"ride_q90":>9s} '
          f'{"%cascade":>9s} {"top_shape":>20s} (% of cluster)')
    print('-' * 110)
    for cid in cluster_ids:
        c = sub[sub['cluster_id'] == cid]
        ride = c['ride_pnl_pts']
        shape_top = c['shape_class'].value_counts().head(2)
        shape_str = ', '.join(f'{s}({100*n/len(c):.0f}%)' for s, n in shape_top.items())
        cid_str = 'NOISE' if cid == -1 else f'C{cid}'
        print(f'{cid_str:>4s} {len(c):>6d} {100*len(c)/len(sub):>4.1f}% '
              f'{ride.mean():>+9.1f} {ride.median():>+8.1f} '
              f'{ride.quantile(0.10):>+8.1f} {ride.quantile(0.90):>+8.1f} '
              f'{100*c["resolved_as_cascade"].mean():>8.1f}% {shape_str}')

    # Shape composition matrix (rows = cluster, cols = shape)
    print('\nShape composition per cluster (rows=cluster, cols=shape, n_phrases):')
    crosstab = pd.crosstab(sub['cluster_id'], sub['shape_class'])
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    print(crosstab.to_string())

    # Per-shape: which clusters absorb each shape?
    print('\nPer-shape distribution across clusters (rows=shape, cols=cluster, % of shape):')
    crosstab_t = pd.crosstab(sub['shape_class'], sub['cluster_id'])
    crosstab_t_pct = crosstab_t.div(crosstab_t.sum(axis=1), axis=0) * 100
    print(crosstab_t_pct.round(1).to_string())

    # Render charts
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. UMAP scatter colored by cluster
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    cmap = plt.get_cmap('tab10')
    for cid in cluster_ids:
        mask = labels == cid
        color = '#999999' if cid == -1 else cmap(cid % 10)
        label = f'NOISE (n={mask.sum()})' if cid == -1 else f'C{cid} (n={mask.sum()})'
        axes[0].scatter(emb[mask, 0], emb[mask, 1], c=[color], s=10,
                       alpha=0.65, label=label, edgecolor='none')
    axes[0].set_title(f'15m phrase population: UMAP + HDBSCAN (global, all shapes)\n'
                       f'n_total={len(sub)}  n_clusters={n_clusters}  '
                       f'noise={n_noise} ({100*n_noise/len(sub):.1f}%)',
                       fontsize=12)
    axes[0].legend(loc='best', fontsize=9, markerscale=1.7)
    axes[0].set_xlabel('UMAP dim 1'); axes[0].set_ylabel('UMAP dim 2')
    axes[0].grid(True, alpha=0.25)

    # 2. UMAP scatter colored by shape_class
    shapes_sorted = sub['shape_class'].value_counts().index.tolist()
    shape_colors = plt.get_cmap('tab20')
    for i, shape in enumerate(shapes_sorted):
        mask = sub['shape_class'].values == shape
        axes[1].scatter(emb[mask, 0], emb[mask, 1],
                       c=[shape_colors(i % 20)], s=10, alpha=0.65,
                       label=f'{shape} (n={mask.sum()})', edgecolor='none')
    axes[1].set_title(f'Same UMAP, colored by SHAPE_CLASS\n'
                       f'do natural HDBSCAN clusters align with shapes?',
                       fontsize=12)
    axes[1].legend(loc='best', fontsize=8, markerscale=1.5)
    axes[1].set_xlabel('UMAP dim 1'); axes[1].set_ylabel('UMAP dim 2')
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    chart_path = os.path.join(args.out_dir, 'umap_hdbscan_global.png')
    plt.savefig(chart_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {chart_path}')

    # Save crosstab + cluster assignments
    crosstab.to_csv(os.path.join(args.out_dir, 'crosstab_cluster_x_shape.csv'))
    sub[['day', 'seg_idx', 'shape_class', 'cluster_id', 'ride_pnl_pts',
         'resolved_as_cascade', 'length_min', 'peak_abs_z']].to_csv(
        os.path.join(args.out_dir, 'phrases_with_cluster.csv'), index=False)
    print(f'Crosstab CSV -> {os.path.join(args.out_dir, "crosstab_cluster_x_shape.csv")}')


if __name__ == '__main__':
    main()
