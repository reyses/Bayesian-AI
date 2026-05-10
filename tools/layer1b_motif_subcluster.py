"""Layer 1b: drill within each 15m phrase-cluster by clustering motif 2D shapes.

For each 15m phrase cluster from Layer 1a:
    1. Grab all motifs (5m segments) that belong to phrases in this cluster
    2. Run UMAP+HDBSCAN on motif-level segment features (slope, sigma,
       length, peak_z, r2adj, tod) — same 2D-shape features used at phrase
       level, no chord/feature signatures
    3. Save each motif's sub-cluster assignment

Pure Layer 1: only 2D-shape features (per memory rule). Layer 2 (chord)
remains for later.

Pipeline pre-req: phrase-level cluster assignments must exist. We re-run
per-shape HDBSCAN at phrase level here to generate them, then drill into
each cluster's motif population.

USAGE
    python tools/layer1b_motif_subcluster.py
    python tools/layer1b_motif_subcluster.py --shape LINEAR_DOWN  # one shape only
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


# Segment-level features for HDBSCAN. Identical at phrase and motif levels —
# both are 2D-shape-only. NO chord fingerprint stats (those are Layer 2).
SEGMENT_FEATURES = [
    'slope_pts_per_min',
    'mean_sigma',
    'sigma_rank_mid',
    'r2adj',
    'length_min',
    'peak_abs_z',
    'net_move_pts',
    'tod_start_hour_utc',
]

OUTCOME_COLS = ['ride_pnl_pts', 'fade_pnl_pts',
                'max_mfe_ride_pts', 'max_mae_ride_pts',
                'resolved_as_cascade', 'extended_60m']


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def _cluster(X: np.ndarray, min_cluster_size: int, min_samples: int,
              umap_n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardize -> UMAP -> HDBSCAN. Returns (labels, embedding)."""
    import hdbscan
    import umap.umap_ as umap

    finite = np.all(np.isfinite(X), axis=1)
    if finite.sum() < min_cluster_size * 2:
        return None, None
    X_std = _standardize(X[finite])
    n_neighbors = min(umap_n_neighbors, max(5, len(X_std) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.05,
                         metric='euclidean', random_state=42, n_components=2)
    emb = reducer.fit_transform(X_std)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=min_samples,
                                  cluster_selection_method='eom')
    labels = clusterer.fit_predict(emb)
    # Map back to original index space
    full_labels = np.full(len(X), -1, dtype=int)
    full_emb = np.full((len(X), 2), np.nan)
    full_labels[finite] = labels
    full_emb[finite] = emb
    return full_labels, full_emb


def _summarize_cluster_outcomes(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    rows = []
    for cid, sub in df.groupby(cluster_col):
        row = {'cluster_id': cid, 'n': len(sub)}
        for col in OUTCOME_COLS:
            if col in sub.columns:
                if sub[col].dtype == bool:
                    row[f'{col}_pct'] = round(100 * float(sub[col].mean()), 1)
                else:
                    row[f'{col}_mean'] = round(float(sub[col].mean()), 1)
                    row[f'{col}_med']  = round(float(sub[col].median()), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('cluster_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phrase-csv',
        default='reports/findings/segments/all_motifs_labeled.csv')
    ap.add_argument('--motif-csv',
        default='reports/findings/segments/all_melodies_labeled.csv')
    ap.add_argument('--shape', default=None,
                    help='Limit to one phrase shape; default: all shapes >= min_shape_n')
    ap.add_argument('--min-shape-n', type=int, default=80)
    ap.add_argument('--phrase-mcs', type=int, default=30,
                    help='min_cluster_size at phrase (15m) level')
    ap.add_argument('--motif-mcs', type=int, default=30,
                    help='min_cluster_size at motif (5m) level')
    ap.add_argument('--phrase-msamples', type=int, default=10)
    ap.add_argument('--motif-msamples', type=int, default=10)
    ap.add_argument('--split', default='IS', choices=['IS', 'OOS', 'BOTH'])
    ap.add_argument('--out-dir', default='reports/findings/segments/layer1b')
    args = ap.parse_args()

    phrases = pd.read_csv(args.phrase_csv)
    motifs = pd.read_csv(args.motif_csv)
    if args.split != 'BOTH':
        phrases = phrases[phrases['split'] == args.split].reset_index(drop=True)
        motifs = motifs[motifs['split'] == args.split].reset_index(drop=True)
    print(f'Phrases: {len(phrases)}    Motifs: {len(motifs)}    Split: {args.split}')
    print(f'Features ({len(SEGMENT_FEATURES)}): {SEGMENT_FEATURES}')

    feat = [c for c in SEGMENT_FEATURES if c in phrases.columns]

    if args.shape:
        shapes_to_run = [args.shape]
    else:
        shape_counts = phrases['shape_class'].value_counts()
        shapes_to_run = [s for s, n in shape_counts.items() if n >= args.min_shape_n]
        print(f'Shapes with >= {args.min_shape_n} phrases: {shapes_to_run}')

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: phrase-level cluster per shape ─────────────────────
    phrases['cluster_15m'] = -99   # placeholder
    summary_rows = []

    for shape in shapes_to_run:
        sub_idx = phrases.index[phrases['shape_class'] == shape].tolist()
        sub_X = phrases.loc[sub_idx, feat].values.astype(np.float64)
        print(f'\n=== {shape}: phrase-level HDBSCAN (n={len(sub_idx)}) ===')
        labels_15m, _ = _cluster(sub_X, args.phrase_mcs, args.phrase_msamples,
                                  umap_n_neighbors=15)
        if labels_15m is None:
            print(f'  too few finite rows; assigning all to cluster -1')
            phrases.loc[sub_idx, 'cluster_15m'] = -1
            continue
        # Re-key the cluster labels with the shape prefix to keep them globally unique
        # e.g. LINEAR_DOWN cluster 0 becomes 'LINEAR_DOWN_C0'
        for k, j in enumerate(sub_idx):
            phrases.at[j, 'cluster_15m'] = int(labels_15m[k])
        n_clusters = len([c for c in set(labels_15m) if c >= 0])
        n_noise = int((labels_15m == -1).sum())
        print(f'  {n_clusters} clusters + {n_noise} NOISE ({100*n_noise/len(sub_idx):.1f}%)')

        for cid in sorted(set(labels_15m)):
            mask = labels_15m == cid
            n = int(mask.sum())
            cid_str = 'NOISE' if cid == -1 else f'C{cid}'
            sub_phrases = phrases.loc[np.array(sub_idx)[mask]]
            ride_mean = sub_phrases['ride_pnl_pts'].mean()
            casc_pct = 100 * sub_phrases['resolved_as_cascade'].mean()
            print(f'    {shape}_{cid_str}  n={n:>4}  ride_mean={ride_mean:>+8.1f}  cascade={casc_pct:>5.1f}%')
            summary_rows.append({
                'shape': shape, 'cluster_15m': cid_str, 'level': 'phrase_15m',
                'n': n, 'ride_mean': ride_mean, 'cascade_pct': casc_pct,
            })

    # Save phrase cluster assignments
    phrases_out = phrases[['day', 'seg_idx', 'shape_class', 'cluster_15m',
                           'ride_pnl_pts', 'resolved_as_cascade',
                           'length_min', 'peak_abs_z']].copy()
    phrases_out.to_csv(os.path.join(args.out_dir, 'phrases_with_15m_clusters.csv'),
                        index=False)
    print(f'\nWrote phrases_with_15m_clusters.csv ({len(phrases_out)} rows)')

    # ── Step 2: motif sub-cluster within each 15m cluster ────────────
    motifs['parent_15m_cluster'] = 'NONE'
    motifs['cluster_5m'] = -99

    # Build phrase->cluster lookup
    phrase_lookup = phrases.set_index(['day', 'seg_idx'])['cluster_15m'].to_dict()
    shape_lookup = phrases.set_index(['day', 'seg_idx'])['shape_class'].to_dict()

    # Tag each motif with its parent phrase's cluster (using parent_motif_idx
    # in the motifs CSV, which maps to the phrase's seg_idx)
    for i, mot in motifs.iterrows():
        key = (mot['day'], mot['parent_motif_idx'])
        cid = phrase_lookup.get(key)
        shape = shape_lookup.get(key, 'UNKNOWN')
        if cid is not None:
            cid_str = 'NOISE' if cid == -1 else f'C{cid}'
            motifs.at[i, 'parent_15m_cluster'] = f'{shape}_{cid_str}'

    n_tagged = (motifs['parent_15m_cluster'] != 'NONE').sum()
    print(f'Tagged {n_tagged} of {len(motifs)} motifs with parent 15m cluster')

    # Within each 15m cluster, run motif-level HDBSCAN
    motif_feat = [c for c in SEGMENT_FEATURES if c in motifs.columns]
    print(f'\n=== Step 2: motif-level HDBSCAN within each 15m cluster ===')

    cluster_hierarchy = []
    for parent_cluster, sub_motifs in motifs.groupby('parent_15m_cluster'):
        if parent_cluster == 'NONE' or len(sub_motifs) < args.motif_mcs * 2:
            continue
        idx = sub_motifs.index.tolist()
        X = motifs.loc[idx, motif_feat].values.astype(np.float64)
        labels_5m, _ = _cluster(X, args.motif_mcs, args.motif_msamples,
                                 umap_n_neighbors=15)
        if labels_5m is None:
            print(f'  {parent_cluster}: too few finite motifs; skipping')
            continue
        for k, j in enumerate(idx):
            motifs.at[j, 'cluster_5m'] = int(labels_5m[k])
        n_clusters = len([c for c in set(labels_5m) if c >= 0])
        n_noise = int((labels_5m == -1).sum())
        print(f'  {parent_cluster:<25s}  n={len(idx):>5}  -> {n_clusters} sub-clusters + {n_noise} NOISE ({100*n_noise/len(idx):.0f}%)')

        cluster_hierarchy.append({
            'parent_15m_cluster': parent_cluster,
            'n_motifs': len(idx),
            'n_5m_subclusters': n_clusters,
            'n_noise': n_noise,
        })

        # Per-sub-cluster outcomes
        for cid in sorted(set(labels_5m)):
            mask = labels_5m == cid
            n = int(mask.sum())
            cid_str = 'NOISE' if cid == -1 else f'C{cid}'
            sub = motifs.loc[np.array(idx)[mask]]
            ride_mean = sub['ride_pnl_pts'].mean() if 'ride_pnl_pts' in sub.columns else float('nan')
            casc_pct = 100 * sub['resolved_as_cascade'].mean() if 'resolved_as_cascade' in sub.columns else float('nan')
            print(f'      {parent_cluster}/{cid_str:>5s}  n={n:>4}  ride={ride_mean:>+8.1f}  cascade={casc_pct:>5.1f}%')
            summary_rows.append({
                'shape':       parent_cluster.split('_')[0] + '_' + parent_cluster.split('_')[1] if '_' in parent_cluster else parent_cluster,
                'cluster_15m': parent_cluster,
                'cluster_5m':  cid_str,
                'level':       'motif_5m',
                'n':           n,
                'ride_mean':   ride_mean,
                'cascade_pct': casc_pct,
            })

    # Save motif cluster assignments
    motifs_out = motifs[['day', 'seg_idx', 'parent_motif_idx', 'shape_class',
                         'parent_15m_cluster', 'cluster_5m',
                         'ride_pnl_pts', 'resolved_as_cascade',
                         'length_min', 'peak_abs_z']].copy()
    motifs_out.to_csv(os.path.join(args.out_dir, 'motifs_with_5m_subclusters.csv'),
                       index=False)
    print(f'\nWrote motifs_with_5m_subclusters.csv ({len(motifs_out)} rows)')

    # Hierarchy summary
    hierarchy_df = pd.DataFrame(cluster_hierarchy).sort_values(
        'n_motifs', ascending=False)
    print('\n=== Layer 1 hierarchy summary (15m -> 5m) ===')
    print(hierarchy_df.to_string(index=False))
    hierarchy_df.to_csv(os.path.join(args.out_dir, 'cluster_hierarchy.csv'),
                        index=False)

    # Markdown report
    md = ['# Layer 1b: 5m motif sub-clustering within 15m phrase clusters',
          '',
          f'_Generated {datetime.now().isoformat()}_',
          '',
          f'Split: {args.split}    phrase_mcs: {args.phrase_mcs}    motif_mcs: {args.motif_mcs}',
          f'Pure Layer 1 (2D shape only). Features: `{SEGMENT_FEATURES}`',
          '',
          '## Hierarchy summary (motifs grouped by parent 15m cluster)',
          '',
          '```',
          hierarchy_df.to_string(index=False),
          '```',
          '',
          '## Per-cluster outcomes (phrase + motif levels)',
          '',
          '```',
          pd.DataFrame(summary_rows).to_string(index=False),
          '```',
          '',
          '## Notes',
          '',
          '- Each row in `phrases_with_15m_clusters.csv` is one phrase with its',
          '  15m cluster id (per shape).',
          '- Each row in `motifs_with_5m_subclusters.csv` is one motif tagged',
          '  with its parent 15m cluster AND its own 5m sub-cluster id.',
          '- Bayesian-table cell key is now (shape, cluster_15m, cluster_5m).',
          '']
    with open(os.path.join(args.out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'\nReport -> {os.path.join(args.out_dir, "summary.md")}')


if __name__ == '__main__':
    main()
