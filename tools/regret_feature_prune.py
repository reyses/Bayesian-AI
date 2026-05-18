"""Prune redundant V2 features via correlation clusters + 1D R² ranking.

Per user 2026-05-16 + the regret-research methodology memory: exhaustive
pair/triplet on 184 raw V2 features is computationally heavy AND statistically
suspect (the features are highly correlated — same concept at multiple TFs).

This tool:
  1. Computes 1D regression R² (signed_mfe target) per feature for ranking.
  2. Computes the feature-feature correlation matrix.
  3. Hierarchical-clusters features by correlation distance (1 − |corr|).
  4. Picks ONE representative per cluster (highest 1D R² in the cluster).
  5. Writes the pruned feature list AND the cluster membership.

Default correlation threshold: 0.85 (features with |corr| ≥ 0.85 are
considered redundant and grouped into a cluster). Configurable.

Outputs:
  feature_prune_<name>.csv             one row per feature with cluster_id,
                                        is_representative flag, 1D R², ρ
  feature_prune_representatives_<name>.txt  the pruned feature list, one per line
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='daisy-chain + V2 features parquet')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='IS_full_daisy_v2')
    ap.add_argument('--corr-threshold', type=float, default=0.85,
                    help='Features with |corr| >= this are grouped (default 0.85)')
    ap.add_argument('--target', default='signed_mfe',
                    help='Target column for 1D R² ranking (default signed_mfe)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)
    print(f'Loaded {len(df)} rows × {len(df.columns)} cols from {args.input}')

    # Add signed_mfe if missing
    if 'signed_mfe' not in df.columns and 'mfe_dollars' in df.columns and 'direction' in df.columns:
        df['signed_mfe'] = df['mfe_dollars'] * np.where(df['direction'] == 'LONG', 1, -1)

    if args.target not in df.columns:
        raise SystemExit(f'Target {args.target} not in columns')

    # Identify V2 feature columns: prefix L1_/L2_/L3_
    feat_cols = [c for c in df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 feature columns: {len(feat_cols)}')

    # ── 1. 1D R² + Spearman per feature ──
    print('\n--- 1D regression per feature ---')
    rows = []
    y = df[args.target].astype(float).values
    y_var = float(np.var(y))
    if y_var == 0:
        raise SystemExit('Target has zero variance')

    for f in feat_cols:
        x = df[f].astype(float).values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_, y_ = x[mask], y[mask]
        if len(x_) < 100 or np.std(x_) == 0:
            rows.append({'feature': f, 'n': len(x_), 'lin_r2': float('nan'),
                         'spearman_rho': float('nan'), 'abs_rho': float('nan')})
            continue
        lr = stats.linregress(x_, y_)
        try:
            rho, _ = stats.spearmanr(x_, y_)
        except Exception:
            rho = float('nan')
        rows.append({
            'feature': f, 'n': len(x_),
            'lin_r2': round(float(lr.rvalue ** 2), 5),
            'spearman_rho': round(float(rho), 5) if not np.isnan(rho) else float('nan'),
            'abs_rho': round(abs(float(rho)), 5) if not np.isnan(rho) else float('nan'),
        })
    feat_stats = pd.DataFrame(rows).set_index('feature')

    # ── 2. Correlation matrix ──
    print('--- Correlation matrix ---')
    X = df[feat_cols].astype(float)
    # Fill NaN with column mean for correlation computation
    X = X.fillna(X.mean())
    # Drop zero-variance features
    nz = X.std(axis=0) > 1e-9
    feat_cols_nz = [c for c, k in zip(feat_cols, nz) if k]
    dropped = [c for c, k in zip(feat_cols, nz) if not k]
    if dropped:
        print(f'  Dropped {len(dropped)} zero-variance features')
    Xn = X[feat_cols_nz]
    corr = Xn.corr(method='pearson').values
    print(f'  Correlation matrix shape: {corr.shape}')

    # ── 3. Hierarchical clustering by correlation distance ──
    print('--- Hierarchical clustering on correlation distance ---')
    # Distance = 1 - |corr|; values in [0,1] (0=identical, 1=uncorrelated)
    dist = 1.0 - np.abs(corr)
    # Force diagonal to zero (numerical guard) and ensure symmetric
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2.0
    # Condensed form for scipy.linkage
    from scipy.spatial.distance import squareform
    condensed = squareform(dist, checks=False)
    # Average linkage (less sensitive to chaining than single/complete)
    Z = linkage(condensed, method='average')
    # Cut at threshold = 1 - corr_threshold
    cluster_labels = fcluster(Z, t=(1.0 - args.corr_threshold), criterion='distance')
    n_clusters = int(cluster_labels.max())
    print(f'  {n_clusters} clusters at |corr| >= {args.corr_threshold}')

    cluster_df = pd.DataFrame({
        'feature': feat_cols_nz,
        'cluster_id': cluster_labels,
    }).set_index('feature')
    cluster_df = cluster_df.join(feat_stats[['n', 'lin_r2', 'spearman_rho', 'abs_rho']])

    # ── 4. Pick representative per cluster (highest abs_rho or lin_r2) ──
    print('--- Selecting representatives ---')
    # Rank within each cluster by combined score: lin_r2 + abs_rho (handles both
    # linear and monotone non-linear)
    cluster_df['combined_score'] = (cluster_df['lin_r2'].fillna(0) +
                                     cluster_df['abs_rho'].fillna(0))
    cluster_df['is_representative'] = False
    for cid, sub in cluster_df.groupby('cluster_id'):
        best = sub['combined_score'].idxmax()
        cluster_df.loc[best, 'is_representative'] = True

    reps = cluster_df[cluster_df['is_representative']].sort_values(
        'combined_score', ascending=False)
    print(f'  {len(reps)} representative features selected')

    # ── Outputs ──
    cluster_df = cluster_df.reset_index().sort_values(['cluster_id', 'combined_score'],
                                                       ascending=[True, False])
    out_csv = out_dir / f'feature_prune_{args.name}.csv'
    cluster_df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')

    # Pruned feature list (representatives only, sorted by score)
    reps_list = list(reps.sort_values('combined_score', ascending=False).index)
    txt_path = out_dir / f'feature_prune_representatives_{args.name}.txt'
    with open(txt_path, 'w') as f:
        for r in reps_list:
            f.write(r + '\n')
    print(f'Wrote: {txt_path} ({len(reps_list)} features)')

    # ── Stdout summary ──
    print(f'\n=== Pruned feature list (top 30 by combined score) ===')
    show_cols = ['feature', 'cluster_id', 'lin_r2', 'spearman_rho', 'combined_score']
    print(reps.reset_index()[show_cols].head(30).to_string(index=False))

    print(f'\n=== Largest clusters (most-redundant feature groups, top 10) ===')
    cluster_sizes = cluster_df.groupby('cluster_id').size().sort_values(ascending=False)
    for cid, sz in cluster_sizes.head(10).items():
        rep_name = cluster_df[(cluster_df['cluster_id'] == cid) &
                               (cluster_df['is_representative'])]['feature'].iloc[0]
        print(f'  cluster {cid}: {sz} features, rep = {rep_name}')


if __name__ == '__main__':
    main()
