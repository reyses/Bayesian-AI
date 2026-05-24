"""Stepped HDBSCAN — find NATURAL variations per phrase shape via density clustering.

Replaces the discrete-axis step walker with data-driven clustering:
    1. Per phrase shape (LINEAR_DOWN, EXPONENTIAL_UP, FLATLINE, ...):
       a. Extract chord fingerprint features (continuous)
       b. Standardize (z-score)
       c. UMAP to 2D for density structure
       d. HDBSCAN to find natural density clusters
    2. Each cluster = a NATURAL variation (data-defined, not quartile-binned)
    3. HDBSCAN's NOISE label = truly idiosyncratic phrases (no cluster fits)
    4. Per-cluster diagnostics: cluster size, oracle outcomes, chord centroid

The NUMBER of natural clusters per shape tells us the conditioning depth
the data supports for that shape. NOISE fraction tells us how much of the
data is genuinely idiosyncratic and must rely on shape-level prior alone.

Stepped: after the first pass per shape, optionally drill within each
cluster (next-level HDBSCAN on motif-level chord fingerprints inside the
phrases of that cluster). For V0 we report the first-level structure.

USAGE
    python tools/stepped_hdbscan_phrases.py
    python tools/stepped_hdbscan_phrases.py --min-cluster-size 30
    python tools/stepped_hdbscan_phrases.py --shape LINEAR_DOWN  # one shape only
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


# Default chord fingerprint feature set
DEFAULT_FEATURES = [
    # 15m anchor
    'slope_15m__mean', 'slope_15m__std',
    'z_close_15m__mean', 'z_close_15m__std',
    'sigma_rank_15m__mean', 'sigma_rank_15m__std',
    # 5m anchor
    'slope_5m__mean', 'slope_5m__std',
    'z_close_5m__mean', 'z_close_5m__std',
    'sigma_rank_5m__mean', 'sigma_rank_5m__std',
    # variation
    'r2adj_5m__mean', 'r2adj_5m__std',
    # segment-level summary
    'length_min', 'peak_abs_z',
]

# Outcome columns to summarize per cluster
OUTCOME_COLS = ['ride_pnl_pts', 'fade_pnl_pts',
                'max_mfe_ride_pts', 'max_mae_ride_pts',
                'resolved_as_cascade', 'extended_60m']


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def _cluster_one_shape(sub_df: pd.DataFrame, feature_cols: list[str],
                       min_cluster_size: int, min_samples: int,
                       umap_n_neighbors: int) -> dict:
    """Run UMAP+HDBSCAN on one shape's chord-fingerprint feature matrix."""
    import hdbscan
    import umap.umap_ as umap

    X = sub_df[feature_cols].values.astype(np.float64)
    # Drop rows with any NaN
    finite_mask = np.all(np.isfinite(X), axis=1)
    if finite_mask.sum() < min_cluster_size * 2:
        return {'n_finite': int(finite_mask.sum()),
                'too_few': True,
                'labels': None, 'embedding': None,
                'sub_df': sub_df.iloc[0:0]}

    X_clean = X[finite_mask]
    sub_clean = sub_df[finite_mask].copy().reset_index(drop=True)
    X_std = _standardize(X_clean)

    # UMAP to 2D for density structure
    n_neighbors = min(umap_n_neighbors, max(5, len(X_std) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.05,
                         metric='euclidean', random_state=42, n_components=2)
    emb = reducer.fit_transform(X_std)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=min_samples,
                                  cluster_selection_method='eom')
    labels = clusterer.fit_predict(emb)

    return {
        'n_finite': int(finite_mask.sum()),
        'too_few': False,
        'labels': labels,
        'embedding': emb,
        'sub_df': sub_clean,
    }


def _cluster_summary(sub: pd.DataFrame, labels: np.ndarray,
                     feature_cols: list[str]) -> pd.DataFrame:
    """Per-cluster summary (size, NOISE counted as label=-1)."""
    rows = []
    for cid in sorted(set(labels)):
        mask = labels == cid
        cluster_sub = sub[mask]
        row = {
            'cluster_id': cid,
            'n':          int(mask.sum()),
            'pct':        round(100 * float(mask.mean()), 1),
        }
        for col in OUTCOME_COLS:
            if col in cluster_sub.columns:
                if cluster_sub[col].dtype == bool:
                    row[f'{col}_pct'] = round(100 * float(cluster_sub[col].mean()), 1)
                else:
                    row[f'{col}_mean'] = round(float(cluster_sub[col].mean()), 1)
                    row[f'{col}_med']  = round(float(cluster_sub[col].median()), 1)
        # Chord centroid (mean of features in the cluster)
        for fc in feature_cols[:6]:  # first 6 for compactness
            if fc in cluster_sub.columns:
                row[f'cent_{fc}'] = round(float(cluster_sub[fc].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _render_shape_chart(shape: str, result: dict, out_path: str):
    if result['too_few'] or result['embedding'] is None:
        return
    emb = result['embedding']
    labels = result['labels']
    sub = result['sub_df']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: UMAP embedding colored by cluster
    cmap = plt.get_cmap('tab10')
    unique = sorted(set(labels))
    for cid in unique:
        mask = labels == cid
        if cid == -1:
            color = '#999999'
            label = f'NOISE ({mask.sum()})'
        else:
            color = cmap(cid % 10)
            label = f'C{cid} ({mask.sum()})'
        axes[0].scatter(emb[mask, 0], emb[mask, 1], c=[color], s=14,
                        alpha=0.7, label=label, edgecolor='none')
    axes[0].set_title(f'{shape}: UMAP embedding + HDBSCAN clusters\n'
                       f'n_total={len(sub)}  n_clusters={len([c for c in unique if c >= 0])}  '
                       f'noise={sum(1 for l in labels if l == -1)}',
                       fontsize=11)
    axes[0].legend(loc='best', fontsize=8, markerscale=1.5)
    axes[0].set_xlabel('UMAP dim 1')
    axes[0].set_ylabel('UMAP dim 2')
    axes[0].grid(True, alpha=0.25)

    # Right: per-cluster ride_pnl distribution
    pnl_data = []
    pnl_labels = []
    for cid in unique:
        mask = labels == cid
        if mask.sum() == 0:
            continue
        pnl_data.append(sub.loc[mask, 'ride_pnl_pts'].values)
        pnl_labels.append(f'C{cid}\nn={mask.sum()}' if cid >= 0
                          else f'NOISE\nn={mask.sum()}')
    if pnl_data:
        bp = axes[1].boxplot(pnl_data, labels=pnl_labels, showfliers=False,
                              patch_artist=True)
        for patch, cid in zip(bp['boxes'], unique):
            color = cmap(cid % 10) if cid >= 0 else '#999999'
            patch.set_facecolor(color); patch.set_alpha(0.55)
    axes[1].axhline(0, color='black', lw=0.7, alpha=0.6)
    axes[1].set_title(f'{shape}: ride_pnl_pts per cluster (boxplot, no fliers)',
                       fontsize=11)
    axes[1].set_ylabel('ride_pnl_pts')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/all_motifs_labeled_with_chord.csv')
    ap.add_argument('--shape', default=None,
                    help='Limit to one shape; default: all shapes with >= min_n')
    ap.add_argument('--min-shape-n', type=int, default=80,
                    help='Skip shapes with fewer than this many phrases')
    ap.add_argument('--min-cluster-size', type=int, default=30)
    ap.add_argument('--min-samples', type=int, default=10)
    ap.add_argument('--umap-n-neighbors', type=int, default=15)
    ap.add_argument('--features', default=None,
                    help='comma-sep feature columns; default = DEFAULT_FEATURES')
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--out-dir',
        default='reports/findings/segments/stepped_hdbscan')
    args = ap.parse_args()

    df = pd.read_csv(args.phrase_csv)
    if args.split != 'BOTH':
        df = df[df['split'] == args.split]
    print(f'Phrases loaded: {len(df)} ({args.split})')

    feature_cols = (DEFAULT_FEATURES if args.features is None
                    else [c.strip() for c in args.features.split(',')])
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f'Features ({len(feature_cols)}): {feature_cols}')

    if args.shape:
        shapes_to_run = [args.shape]
    else:
        shape_counts = df['shape_class'].value_counts()
        shapes_to_run = [s for s, n in shape_counts.items()
                          if n >= args.min_shape_n]
        print(f'Shapes with >= {args.min_shape_n} phrases: {shapes_to_run}')

    os.makedirs(args.out_dir, exist_ok=True)
    md = ['# Stepped HDBSCAN per phrase shape',
          '',
          f'_Generated {datetime.now().isoformat()}_',
          '',
          f'Split: {args.split}    min_cluster_size: {args.min_cluster_size}    min_samples: {args.min_samples}',
          f'Features ({len(feature_cols)}): `{feature_cols}`',
          '',
          '## How to read this',
          '',
          '- Per shape, UMAP reduces chord-fingerprint features to 2D, then HDBSCAN',
          '  finds natural density clusters.',
          '- `cluster_id = -1` is HDBSCAN NOISE (idiosyncratic phrases that do not',
          '  fit any cluster).',
          '- The NUMBER of clusters per shape tells us the conditioning depth the',
          '  data supports for that shape.',
          '- High noise % means most phrases of that shape are idiosyncratic and',
          '  the Bayesian table must rely on shape-level prior alone.',
          '']

    overall_summary = []

    for shape in shapes_to_run:
        sub_df = df[df['shape_class'] == shape].reset_index(drop=True)
        print(f'\n--- {shape} (n={len(sub_df)}) ---')
        result = _cluster_one_shape(sub_df, feature_cols,
                                     args.min_cluster_size,
                                     args.min_samples,
                                     args.umap_n_neighbors)
        if result['too_few']:
            print(f'  too few finite rows ({result["n_finite"]}); skipped')
            md.append(f'## {shape}: too few finite rows ({result["n_finite"]}); skipped')
            md.append('')
            continue

        labels = result['labels']
        sub = result['sub_df']
        n_clusters = len([c for c in set(labels) if c >= 0])
        n_noise = int((labels == -1).sum())
        pct_noise = round(100 * n_noise / len(labels), 1)

        summary = _cluster_summary(sub, labels, feature_cols)
        overall_summary.append({
            'shape': shape, 'n_total': len(sub),
            'n_clusters': n_clusters, 'n_noise': n_noise,
            'pct_noise': pct_noise,
        })
        print(f'  n_clusters={n_clusters}  n_noise={n_noise} ({pct_noise}%)')

        # Per-cluster ride_pnl_pts means
        for _, r in summary.iterrows():
            cid = r['cluster_id']
            cid_str = f'C{int(cid)}' if cid >= 0 else 'NOISE'
            ride = r.get('ride_pnl_pts_mean', float('nan'))
            casc = r.get('resolved_as_cascade_pct', float('nan'))
            print(f'    {cid_str:<7s} n={int(r["n"]):>4d}  ride_mean={ride:+8.1f}  cascade_pct={casc:>5.1f}%')

        md.append(f'## {shape}  (n={len(sub)}, {n_clusters} clusters, '
                  f'{n_noise} NOISE = {pct_noise}%)')
        md.append('')
        md.append('```')
        cols_show = ['cluster_id', 'n', 'pct']
        for c in OUTCOME_COLS:
            if f'{c}_mean' in summary.columns: cols_show.append(f'{c}_mean')
            if f'{c}_med' in summary.columns: cols_show.append(f'{c}_med')
            if f'{c}_pct' in summary.columns: cols_show.append(f'{c}_pct')
        md.append(summary[cols_show].to_string(index=False))
        md.append('```')
        md.append('')

        # Render chart
        chart_path = os.path.join(args.out_dir, f'hdbscan_{shape}.png')
        _render_shape_chart(shape, result, chart_path)
        md.append(f'Chart: `{chart_path}`')
        md.append('')

    # Overall summary
    print('\n' + '=' * 70)
    print('OVERALL: clusters per shape')
    print('=' * 70)
    overall_df = pd.DataFrame(overall_summary).sort_values('n_total', ascending=False)
    print(overall_df.to_string(index=False))

    md.insert(7, '## Overall: clusters per shape')
    md.insert(8, '')
    md.insert(9, '```')
    md.insert(10, overall_df.to_string(index=False))
    md.insert(11, '```')
    md.insert(12, '')

    out_md = os.path.join(args.out_dir, 'summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'\nReport -> {out_md}')


if __name__ == '__main__':
    main()
