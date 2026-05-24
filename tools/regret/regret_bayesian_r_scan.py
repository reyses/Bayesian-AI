"""Scan a range of r values and report cluster-size distributions.

Per user 2026-05-16: rather than commit to a single r or a hierarchical
ladder, scan single-level clustering at r = 1, 2, 3, ... and find the
sweet spot between samples-per-cluster and distance-from-archetype.

For each r value:
    Run peel-by-velocity SINGLE-LEVEL clustering (each seed gets one cluster
    of all trades within r of its PCA line)
    Report: n_clusters, n_singletons (<min_size), trades_assigned,
            size_distribution (median, max), top archetype magnitudes

Output: summary CSV with one row per (r, direction).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def perpendicular_distance(points, c_seed, d_seed):
    delta = points - c_seed[np.newaxis, :]
    along = delta @ d_seed
    proj = np.outer(along, d_seed)
    perp = delta - proj
    return np.linalg.norm(perp, axis=1)


def scan_one_r(centroids, directions, magnitudes, pca_unstable,
               oracle_idx, meta, r, min_pool, max_iter, direction_pool, dir_name):
    """Peel at a single r value within a direction pool. Returns cluster records."""
    pool = direction_pool.copy()
    pool = pool[~pca_unstable[pool]]
    clusters = []
    seeds_peeled = 0
    seed_feature_vals = meta['mfe_velocity'].values

    while len(pool) >= min_pool and seeds_peeled < max_iter:
        # Pick seed
        seed_pos = int(np.argmax(seed_feature_vals[pool]))
        seed_idx = int(pool[seed_pos])
        c = centroids[seed_idx]
        d = directions[seed_idx]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-6:
            pool = np.delete(pool, seed_pos)
            continue
        d = d / d_norm

        # Distance from each pool centroid to seed's line
        distances = perpendicular_distance(centroids[pool], c, d)
        member_pos = np.where(distances <= r)[0]
        mem_sig_indices = pool[member_pos]
        mfes = meta['signed_mfe'].values[mem_sig_indices]
        abs_mfes = meta['mfe_dollars'].values[mem_sig_indices]
        clusters.append({
            'direction': dir_name,
            'r': float(r),
            'seed_id': seeds_peeled,
            'seed_oracle_idx': int(oracle_idx[seed_idx]),
            'seed_mfe_velocity': float(seed_feature_vals[seed_idx]),
            'n_members': int(len(member_pos)),
            'mean_signed_mfe': float(mfes.mean()) if len(mfes) > 0 else 0.0,
            'mean_$_magnitude': float(abs_mfes.mean()) if len(abs_mfes) > 0 else 0.0,
        })

        # Remove cluster members from pool
        pool = np.delete(pool, member_pos)
        seeds_peeled += 1

    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signatures', required=True)
    ap.add_argument('--trades-csv', required=True)
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='IS_full_r_scan')
    ap.add_argument('--r-values', type=float, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
                    help='r values to scan')
    ap.add_argument('--min-pool', type=int, default=20)
    ap.add_argument('--max-iterations', type=int, default=2000)
    ap.add_argument('--min-cluster-size-for-archetype', type=int, default=5,
                    help='Cluster sizes >= this count as real archetypes')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading signatures: {args.signatures}')
    sig = np.load(args.signatures, allow_pickle=True)
    oracle_idx = sig['oracle_idx']
    centroids = sig['centroids']
    directions = sig['directions']
    magnitudes = sig['magnitudes']
    pca_unstable = sig['pca_unstable']
    n_sig = len(oracle_idx)
    print(f'  {n_sig} signatures, {pca_unstable.sum()} PCA-unstable')

    print(f'Loading trades: {args.trades_csv}')
    trades = pd.read_csv(args.trades_csv)
    if 'signed_mfe' not in trades.columns:
        trades['signed_mfe'] = trades['mfe_dollars'] * np.where(trades['direction'] == 'LONG', 1, -1)
    meta = trades.set_index('oracle_idx').loc[oracle_idx][
        ['direction', 'mfe_velocity', 'signed_mfe', 'mfe_dollars']
    ].reset_index()

    long_pool = np.where(meta['direction'].values == 'LONG')[0]
    short_pool = np.where(meta['direction'].values == 'SHORT')[0]
    print(f'  Pool sizes: LONG={len(long_pool)}, SHORT={len(short_pool)}')

    # Scan
    print(f'\nScanning r = {args.r_values}')
    all_clusters = []
    summary_rows = []
    t0 = time.time()
    for r in args.r_values:
        for dir_name, dir_pool in [('LONG', long_pool), ('SHORT', short_pool)]:
            t_r = time.time()
            clusters = scan_one_r(
                centroids, directions, magnitudes, pca_unstable, oracle_idx,
                meta, r, args.min_pool, args.max_iterations, dir_pool, dir_name)
            elapsed = time.time() - t_r

            # Aggregate stats
            ns = np.array([c['n_members'] for c in clusters])
            mean_signeds = np.array([c['mean_signed_mfe'] for c in clusters])
            archetype_mask = ns >= args.min_cluster_size_for_archetype
            assigned_total = int(ns.sum())
            n_archetypes = int(archetype_mask.sum())
            n_singletons_etc = int((~archetype_mask).sum())
            top_signed = (mean_signeds[archetype_mask].min() if dir_name == 'SHORT'
                          else mean_signeds[archetype_mask].max()) if n_archetypes > 0 else 0.0
            summary_rows.append({
                'r': float(r),
                'direction': dir_name,
                'n_clusters_total': len(clusters),
                'n_archetypes_n_ge_5': n_archetypes,
                'n_micro_n_lt_5': n_singletons_etc,
                'trades_assigned': assigned_total,
                'median_n': float(np.median(ns)) if len(ns) > 0 else 0.0,
                'max_n': int(ns.max()) if len(ns) > 0 else 0,
                'top_archetype_mean_signed': float(top_signed),
                'elapsed_sec': round(elapsed, 1),
            })
            all_clusters.extend(clusters)
            print(f'  r={r:5.2f}  {dir_name}: {len(clusters):4d} clusters, '
                  f'{n_archetypes:3d} archetypes(n>=5), max_n={int(ns.max() if len(ns) else 0):4d}, '
                  f'top_signed={top_signed:+8.1f}, {elapsed:.1f}s', flush=True)

    print(f'\nTotal scan time: {time.time()-t0:.1f}s')

    # Save summary + per-cluster table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f'{args.name}_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\nWrote summary: {summary_path}')

    clusters_df = pd.DataFrame(all_clusters)
    clusters_path = out_dir / f'{args.name}_clusters.csv'
    clusters_df.to_csv(clusters_path, index=False)
    print(f'Wrote clusters: {clusters_path}')

    # Pretty print summary
    print(f'\n=== Scan summary ===')
    print(f'(archetype = cluster with n >= {args.min_cluster_size_for_archetype})\n')
    cols = ['r', 'direction', 'n_clusters_total', 'n_archetypes_n_ge_5',
            'n_micro_n_lt_5', 'trades_assigned', 'median_n', 'max_n',
            'top_archetype_mean_signed']
    print(summary_df[cols].to_string(index=False))


if __name__ == '__main__':
    main()
