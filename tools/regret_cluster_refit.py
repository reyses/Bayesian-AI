"""Refit each cluster's PCA line using ALL member trajectories (not just seed).

Per user 2026-05-16: after peel + assignment at a chosen r, the cluster
representative line is the SEED's PCA line — which uses only the seed
trade's trajectory. The cluster contains many trades close to that line;
their joint trajectory should give a more representative archetype line.

This is one EM-style refinement iteration:
    M-step: pool all member bars → PCA → refined (centroid, direction).
    (We don't re-do the E-step / re-assignment here — keep membership
     fixed from the peel for stability.)

For each cluster:
    Pool all bars from all member trades → matrix X (Σ T_bars × 184)
    The features are already globally z-scored (from signatures step)
    Center X (subtract mean), SVD → refined centroid + first PC direction
    Save refined line + per-cluster stats (variance along direction, mean
        member distance from refined line, etc.)

Outputs:
    refined_bayesian_<name>.csv          — human-readable cluster table
    refined_bayesian_<name>.npz          — high-dim arrays (centroids, dirs)

The refined line is what a live selector should match against, and what
the decay metric d(t) should be measured from.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla


def perpendicular_distance(points, c, d):
    delta = points - c[np.newaxis, :]
    along = delta @ d
    proj = np.outer(along, d)
    perp = delta - proj
    return np.linalg.norm(perp, axis=1)


def peel_at_r(centroids, directions, pca_unstable, oracle_idx, meta,
              dir_pool, dir_name, r, min_pool, max_iter, min_cluster_size):
    """Single-level peel at radius r. Returns list of (seed_idx, member_sig_indices)."""
    pool = dir_pool.copy()
    pool = pool[~pca_unstable[pool]]
    seed_feature_vals = meta['mfe_velocity'].values
    clusters = []   # each = dict with seed info + member array

    while len(pool) >= min_pool and len(clusters) < max_iter:
        seed_pos = int(np.argmax(seed_feature_vals[pool]))
        seed_sig_idx = int(pool[seed_pos])
        c = centroids[seed_sig_idx]
        d = directions[seed_sig_idx]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-6:
            pool = np.delete(pool, seed_pos)
            continue
        d = d / d_norm

        dists = perpendicular_distance(centroids[pool], c, d)
        member_pos = np.where(dists <= r)[0]
        member_sig_indices = pool[member_pos]

        if len(member_sig_indices) >= min_cluster_size:
            clusters.append({
                'direction': dir_name,
                'r': float(r),
                'seed_id': len(clusters),
                'seed_sig_idx': seed_sig_idx,
                'seed_oracle_idx': int(oracle_idx[seed_sig_idx]),
                'seed_mfe_velocity': float(seed_feature_vals[seed_sig_idx]),
                'member_sig_indices': member_sig_indices.astype(np.int64),
                'member_oracle_ids': oracle_idx[member_sig_indices].astype(np.int64),
                'n_members': int(len(member_sig_indices)),
            })

        # Always remove members from pool regardless of cluster size
        pool = np.delete(pool, member_pos)

    return clusters


def refit_cluster_line(member_oracle_ids: np.ndarray, X: np.ndarray,
                       oracle_idx_arr: np.ndarray) -> dict | None:
    """Pool all bars from members, run PCA, return refined (centroid, direction, sv1)."""
    # Find all rows in X belonging to any of these member oracle_ids
    mask = np.isin(oracle_idx_arr, member_oracle_ids)
    X_members = X[mask]
    if len(X_members) < 5:
        return None
    centroid = X_members.mean(axis=0)
    X_c = X_members - centroid[np.newaxis, :]
    # Drop near-constant columns to avoid SVD ill-conditioning
    col_std = X_c.std(axis=0)
    active = col_std > 1e-6
    if active.sum() < 2:
        return None
    X_active = X_c[:, active]
    try:
        _, S, Vh = np.linalg.svd(X_active, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            _, S, Vh = sla.svd(X_active, full_matrices=False, lapack_driver='gesvd')
        except (sla.LinAlgError, np.linalg.LinAlgError):
            return None
    n_features = X.shape[1]
    direction = np.zeros(n_features, dtype=np.float32)
    direction[active] = Vh[0, :].astype(np.float32)
    # Per-cluster spread: mean perpendicular distance of members to refined line
    perp = perpendicular_distance(X_members, centroid, direction)
    return {
        'centroid': centroid.astype(np.float32),
        'direction': direction,
        'sv1': float(S[0]),
        'n_bars': int(len(X_members)),
        'mean_perp_distance': float(perp.mean()),
        'median_perp_distance': float(np.median(perp)),
        'max_perp_distance': float(perp.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signatures', required=True)
    ap.add_argument('--trajectories', required=True)
    ap.add_argument('--trades-csv', required=True)
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='IS_full_r7_refit')
    ap.add_argument('--r', type=float, default=7.0)
    ap.add_argument('--min-pool', type=int, default=20)
    ap.add_argument('--min-cluster-size', type=int, default=5)
    ap.add_argument('--max-iterations', type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading signatures: {args.signatures}')
    sig = np.load(args.signatures, allow_pickle=True)
    oracle_idx = sig['oracle_idx']
    centroids_seed = sig['centroids']
    directions_seed = sig['directions']
    pca_unstable = sig['pca_unstable']
    feature_means = sig['feature_means']
    feature_stds = sig['feature_stds']
    feature_names = sig['feature_names']
    print(f'  {len(oracle_idx)} signatures, {pca_unstable.sum()} PCA-unstable')

    print(f'Loading trades: {args.trades_csv}')
    trades = pd.read_csv(args.trades_csv)
    if 'signed_mfe' not in trades.columns:
        trades['signed_mfe'] = trades['mfe_dollars'] * np.where(trades['direction'] == 'LONG', 1, -1)
    meta = trades.set_index('oracle_idx').loc[oracle_idx][
        ['direction', 'mfe_velocity', 'signed_mfe', 'mfe_dollars', 'time_to_mfe_min']
    ].reset_index()

    print(f'Loading trajectories: {args.trajectories}')
    traj = np.load(args.trajectories, allow_pickle=True)
    X = traj['X'].astype(np.float32)
    traj_oracle_idx = traj['oracle_idx']
    print(f'  {X.shape[0]:,} bars x {X.shape[1]} features')

    # Apply the same global z-score the signatures used
    print('Applying global z-score (from signatures)...')
    # First scrub NaN to be safe
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feature_stds_safe = np.where(feature_stds > 1e-8, feature_stds, 1.0).astype(np.float32)
    X = (X - feature_means[np.newaxis, :]) / feature_stds_safe[np.newaxis, :]

    long_pool = np.where(meta['direction'].values == 'LONG')[0]
    short_pool = np.where(meta['direction'].values == 'SHORT')[0]

    # Peel + refit per direction
    print(f'\n--- Peeling at r={args.r} + refitting per cluster ---')
    all_clusters = []
    t0 = time.time()
    for dir_name, dir_pool in [('LONG', long_pool), ('SHORT', short_pool)]:
        print(f'\n=== {dir_name} (pool {len(dir_pool)}) ===')
        peeled = peel_at_r(
            centroids_seed, directions_seed, pca_unstable, oracle_idx, meta,
            dir_pool, dir_name, args.r, args.min_pool, args.max_iterations,
            args.min_cluster_size)
        print(f'  peeled {len(peeled)} clusters (size >= {args.min_cluster_size})')

        # Refit each cluster
        for ci, cl in enumerate(peeled):
            refit = refit_cluster_line(cl['member_oracle_ids'], X, traj_oracle_idx)
            if refit is None:
                continue
            mem_meta = meta.iloc[cl['member_sig_indices']]
            mfes = mem_meta['signed_mfe'].values
            abs_mfes = mem_meta['mfe_dollars'].values
            durs = mem_meta['time_to_mfe_min'].values
            n_long = int((mem_meta['direction'].values == 'LONG').sum())
            all_clusters.append({
                'cluster_id': len(all_clusters),
                'direction': dir_name,
                'seed_id': cl['seed_id'],
                'r': args.r,
                'seed_oracle_idx': cl['seed_oracle_idx'],
                'seed_mfe_velocity': cl['seed_mfe_velocity'],
                'n_members': cl['n_members'],
                'n_bars_pooled': refit['n_bars'],
                'mean_signed_mfe': float(mfes.mean()),
                'std_signed_mfe': float(mfes.std()),
                'mean_$_magnitude': float(abs_mfes.mean()),
                'mean_duration_min': float(durs.mean()),
                'pct_long': round(100 * n_long / len(mem_meta), 1),
                'refined_sv1': refit['sv1'],
                'mean_perp_distance': refit['mean_perp_distance'],
                'median_perp_distance': refit['median_perp_distance'],
                'max_perp_distance': refit['max_perp_distance'],
                '_centroid': refit['centroid'],
                '_direction': refit['direction'],
                '_member_oracle_ids': cl['member_oracle_ids'],
            })
        print(f'  refit done in {time.time()-t0:.1f}s', flush=True)

    n_clusters = len(all_clusters)
    print(f'\nTotal refit clusters: {n_clusters}')

    # CSV (without high-dim arrays)
    csv_rows = []
    for c in all_clusters:
        csv_rows.append({k: v for k, v in c.items() if not k.startswith('_')})
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / f'refined_bayesian_{args.name}.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f'Wrote: {csv_path}')

    # NPZ (high-dim arrays)
    n_features = centroids_seed.shape[1]
    refined_centroids = np.stack([c['_centroid'] for c in all_clusters]) if n_clusters else np.zeros((0, n_features))
    refined_directions = np.stack([c['_direction'] for c in all_clusters]) if n_clusters else np.zeros((0, n_features))
    # Member oracle_ids: variable length, use object array
    member_arrs = np.array([c['_member_oracle_ids'] for c in all_clusters], dtype=object)
    npz_path = out_dir / f'refined_bayesian_{args.name}.npz'
    np.savez_compressed(
        npz_path,
        cluster_id=np.array([c['cluster_id'] for c in all_clusters]),
        direction=np.array([c['direction'] for c in all_clusters]),
        seed_oracle_idx=np.array([c['seed_oracle_idx'] for c in all_clusters]),
        n_members=np.array([c['n_members'] for c in all_clusters]),
        n_bars_pooled=np.array([c['n_bars_pooled'] for c in all_clusters]),
        refined_centroids=refined_centroids,
        refined_directions=refined_directions,
        member_oracle_ids=member_arrs,
        mean_signed_mfe=np.array([c['mean_signed_mfe'] for c in all_clusters]),
        mean_dollars=np.array([c['mean_$_magnitude'] for c in all_clusters]),
        pct_long=np.array([c['pct_long'] for c in all_clusters]),
        feature_names=feature_names,
    )
    print(f'Wrote: {npz_path}')

    # Stats
    if csv_df.empty:
        return
    print('\n--- Per-cluster geometry stats ---')
    print(f'  Mean perp distance from members to refined line (median across clusters): '
          f'{csv_df.mean_perp_distance.median():.3f}σ')
    print(f'  Max perp distance from members (median): {csv_df.median_perp_distance.median():.3f}σ')
    print()
    print('--- Top 15 clusters by |mean_signed| (n >= 5) ---')
    big = csv_df[csv_df.n_members >= 5].copy()
    big['abs_signed'] = big.mean_signed_mfe.abs()
    cols = ['cluster_id', 'direction', 'n_members', 'mean_signed_mfe',
            'mean_$_magnitude', 'mean_perp_distance', 'seed_mfe_velocity']
    print(big.sort_values('abs_signed', ascending=False).head(15)[cols].to_string(index=False))


if __name__ == '__main__':
    main()
