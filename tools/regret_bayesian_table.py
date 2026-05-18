"""L3 Phase 3 — hierarchical Bayesian table builder (PCA-line peel).

Per locked protocol (research/bayesian_archetypes/project.md):
    1. Stratify by direction (LONG / SHORT) — Level 1
    2. Within each direction, iterative peel:
         seed = trade with max mfe_velocity in remaining pool
         For each candidate: distance to seed's PCA line (centroid-mode here;
         per-bar trajectory-mode is a TODO)
         Hierarchical r-ladder: gather at multiple radii (coarse → fine)
         Record cluster TREE per seed
         Remove coarse-level members from pool
    3. Output: hierarchical Bayesian table (one row per cluster node)

Match mode:
    --match centroid     (this build): use each candidate trade's CENTROID
        as the test point. Fast. Loses trajectory shape info; a trade with
        a path that drifts far from the seed's line but has a centroid on
        the line will MATCH.
    --match trajectory   (TODO): require all (or quantile of) candidate's
        trajectory points within r of seed's line. True protocol.

Inputs:
    --signatures      output of regret_trade_signatures.py
    --trades-csv      daisy-chain CSV (for mfe_velocity, signed_mfe, direction)

Outputs:
    bayesian_table_<name>.csv      one row per cluster node (hierarchical
                                    cluster_id path)
    cluster_assignments_<name>.csv  trade_idx → leaf cluster path
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def perpendicular_distance(points: np.ndarray, c_seed: np.ndarray, d_seed: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each point (N, F) to the line through
    c_seed with direction d_seed (F,). d_seed must be unit-length."""
    delta = points - c_seed[np.newaxis, :]   # (N, F)
    along = delta @ d_seed                    # (N,) — projection onto line
    proj = np.outer(along, d_seed)            # (N, F)
    perp = delta - proj                        # (N, F)
    return np.linalg.norm(perp, axis=1)       # (N,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signatures', required=True)
    ap.add_argument('--trades-csv', required=True,
                    help='daisy-chain CSV (oracle_idx, mfe_velocity, signed_mfe, direction)')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='IS_full_daisy_bayes')
    ap.add_argument('--r-ladder', type=float, nargs='+', default=[2.0, 1.0, 0.5, 0.25],
                    help='r values for hierarchical levels, coarse to fine (in z-score units)')
    ap.add_argument('--direction-stratify', type=lambda s: s.lower() != 'false',
                    default=True)
    ap.add_argument('--seed-feature', default='mfe_velocity')
    ap.add_argument('--match', choices=['centroid', 'trajectory'], default='centroid')
    ap.add_argument('--min-cluster-size', type=int, default=2,
                    help='Drop sub-clusters smaller than this')
    ap.add_argument('--min-pool', type=int, default=20,
                    help='Stop peeling when pool drops below this')
    ap.add_argument('--max-iterations', type=int, default=500)
    ap.add_argument('--remove-on-peel', choices=['coarse', 'fine'], default='coarse',
                    help='Which r-level membership removes a trade from the pool')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading signatures: {args.signatures}')
    sig = np.load(args.signatures, allow_pickle=True)
    oracle_idx = sig['oracle_idx']
    centroids = sig['centroids']
    directions = sig['directions']
    magnitudes = sig['magnitudes']
    pca_unstable = sig['pca_unstable']
    print(f'  {len(oracle_idx)} signatures loaded ({pca_unstable.sum()} PCA-unstable)')

    print(f'Loading trade data: {args.trades_csv}')
    trades = pd.read_csv(args.trades_csv)
    if 'signed_mfe' not in trades.columns and 'mfe_dollars' in trades.columns:
        trades['signed_mfe'] = trades['mfe_dollars'] * np.where(trades['direction'] == 'LONG', 1, -1)
    trades_indexed = trades.set_index('oracle_idx')
    print(f'  {len(trades)} trades; signed_mfe range '
          f'[{trades.signed_mfe.min():.0f}, {trades.signed_mfe.max():.0f}]')

    # Join trade metadata onto signature order
    # (signatures are sorted by oracle_idx by construction in P2)
    meta = trades_indexed.loc[oracle_idx][[
        'direction', args.seed_feature, 'signed_mfe', 'mfe_dollars',
        'time_to_mfe_min',
    ]].reset_index()
    n_sig = len(oracle_idx)
    if len(meta) != n_sig:
        print(f'  WARN: {len(meta)} matched of {n_sig}; some oracle_idx not in trades CSV')

    # Build pool indices per direction
    if args.direction_stratify:
        long_pool = np.where(meta['direction'].values == 'LONG')[0]
        short_pool = np.where(meta['direction'].values == 'SHORT')[0]
        # Also restrict to PCA-stable signatures
        long_pool = long_pool[~pca_unstable[long_pool]]
        short_pool = short_pool[~pca_unstable[short_pool]]
        pool_groups = [('LONG', long_pool), ('SHORT', short_pool)]
    else:
        all_pool = np.arange(n_sig)
        all_pool = all_pool[~pca_unstable[all_pool]]
        pool_groups = [('ALL', all_pool)]

    n_levels = len(args.r_ladder)
    print(f'\n--- Peeling per direction ---')
    print(f'r-ladder (coarse->fine): {args.r_ladder}    levels: {n_levels}')

    all_rows = []
    assignments = []   # (trade_idx, cluster_path)
    seed_feature_vals = meta[args.seed_feature].values

    for dir_name, pool_init in pool_groups:
        print(f'\n=== Direction: {dir_name}  initial pool size: {len(pool_init)} ===')
        pool = pool_init.copy()
        # For each trade, remember if it's been assigned (so we don't put it
        # into multiple top-level clusters)
        assigned_mask = np.zeros(n_sig, dtype=bool)
        seed_id = 0

        for it in range(args.max_iterations):
            if len(pool) < args.min_pool:
                print(f'  STOP: pool size {len(pool)} < min_pool {args.min_pool}  '
                      f'(after {it} peels)')
                break
            # Pick seed from pool: max seed_feature
            pool_seed_vals = seed_feature_vals[pool]
            seed_pos_in_pool = int(np.argmax(pool_seed_vals))
            seed_idx = int(pool[seed_pos_in_pool])
            c_seed = centroids[seed_idx]
            d_seed = directions[seed_idx]
            # Defensive: skip if direction is zero (PCA-unstable that slipped through)
            d_norm = np.linalg.norm(d_seed)
            if d_norm < 1e-6:
                pool = np.delete(pool, seed_pos_in_pool)
                continue
            d_seed = d_seed / d_norm

            # Distance from each pool candidate's centroid to seed's PCA line
            if args.match == 'centroid':
                pool_centroids = centroids[pool]
                d = perpendicular_distance(pool_centroids, c_seed, d_seed)
            else:
                # TODO: trajectory mode
                raise NotImplementedError('--match trajectory not yet built')

            # Hierarchical r-ladder: gather at each radius, finer is subset of coarser
            members_at_level = []   # list of arrays of pool indices (positions in `pool`)
            for r in args.r_ladder:
                members_pos = np.where(d <= r)[0]
                members_at_level.append(members_pos)

            # Record cluster nodes (one per level)
            seed_oracle_idx = int(oracle_idx[seed_idx])
            for li, (r, mem_pos) in enumerate(zip(args.r_ladder, members_at_level)):
                if len(mem_pos) < args.min_cluster_size:
                    continue
                mem_pool_indices = pool[mem_pos]            # positions in signatures
                mem_oracle_ids = oracle_idx[mem_pool_indices]
                # Per-cluster stats
                mfes = meta['signed_mfe'].values[mem_pool_indices]
                abs_mfes = meta['mfe_dollars'].values[mem_pool_indices]
                durs = meta['time_to_mfe_min'].values[mem_pool_indices]
                n_long = int(np.sum(meta['direction'].values[mem_pool_indices] == 'LONG'))
                cluster_path = f'{dir_name}.S{seed_id:04d}.L{li}'
                all_rows.append({
                    'cluster_path': cluster_path,
                    'direction': dir_name,
                    'seed_id': seed_id,
                    'level': li,
                    'r': float(r),
                    'seed_oracle_idx': seed_oracle_idx,
                    'seed_mfe_velocity': float(seed_feature_vals[seed_idx]),
                    'n_members': len(mem_pos),
                    'mean_signed_mfe': float(mfes.mean()),
                    'std_signed_mfe': float(mfes.std()),
                    'mean_$_magnitude': float(abs_mfes.mean()),
                    'mean_duration_min': float(durs.mean()),
                    'pct_long': round(100 * n_long / len(mem_pos), 1),
                    'seed_magnitude': float(magnitudes[seed_idx]),
                })

            # Remove members at the chosen "remove-on-peel" level from the pool
            level_for_removal = 0 if args.remove_on_peel == 'coarse' else (n_levels - 1)
            removal_pos = members_at_level[level_for_removal]
            removed_pool_indices = pool[removal_pos]
            assigned_mask[removed_pool_indices] = True
            # Assignment record: each removed trade → finest level it qualified for
            for sig_idx in removed_pool_indices:
                # Find the finest level this trade qualified for
                # (i.e., the largest level index where the trade was in members)
                finest_level = -1
                for li, mem_pos in enumerate(members_at_level):
                    if (pool[mem_pos] == sig_idx).any():
                        finest_level = li
                if finest_level >= 0:
                    assignments.append({
                        'oracle_idx': int(oracle_idx[sig_idx]),
                        'cluster_path': f'{dir_name}.S{seed_id:04d}.L{finest_level}',
                        'direction': dir_name,
                        'seed_id': seed_id,
                        'finest_level': finest_level,
                    })

            pool = np.delete(pool, removal_pos)

            seed_id += 1
            if (seed_id) % 10 == 0:
                print(f'  ...{seed_id} seeds peeled  pool size: {len(pool)}', flush=True)
        else:
            print(f'  STOP: max iterations {args.max_iterations} hit')

        print(f'  {dir_name}: {seed_id} seeds, {assigned_mask.sum()} trades assigned, '
              f'{len(pool)} unassigned')

    # Save tables
    out_tbl = pd.DataFrame(all_rows)
    out_tbl_path = out_dir / f'bayesian_table_{args.name}.csv'
    out_tbl.to_csv(out_tbl_path, index=False)

    out_assign = pd.DataFrame(assignments)
    out_assign_path = out_dir / f'cluster_assignments_{args.name}.csv'
    out_assign.to_csv(out_assign_path, index=False)

    print(f'\nWrote: {out_tbl_path} ({len(out_tbl)} cluster nodes)')
    print(f'Wrote: {out_assign_path} ({len(out_assign)} trade assignments)')

    if len(out_tbl) > 0:
        print('\n--- Summary by direction × level ---')
        summary = out_tbl.groupby(['direction', 'level']).agg(
            n_clusters=('cluster_path', 'count'),
            mean_n_members=('n_members', 'mean'),
            mean_mean_signed=('mean_signed_mfe', 'mean'),
        ).round(2)
        print(summary.to_string())

        # Top clusters by mean_signed magnitude (after filtering for n >= 5)
        big = out_tbl[out_tbl['n_members'] >= 5].copy()
        if len(big) > 0:
            print(f'\n--- Top 10 LONG clusters by mean_signed_mfe (n >= 5) ---')
            top_l = big[big['direction'] == 'LONG'].sort_values('mean_signed_mfe', ascending=False).head(10)
            print(top_l[['cluster_path', 'level', 'r', 'n_members',
                          'mean_signed_mfe', 'mean_$_magnitude', 'pct_long']].to_string(index=False))
            print(f'\n--- Top 10 SHORT clusters by |mean_signed_mfe| (n >= 5) ---')
            top_s = big[big['direction'] == 'SHORT'].sort_values('mean_signed_mfe').head(10)
            print(top_s[['cluster_path', 'level', 'r', 'n_members',
                          'mean_signed_mfe', 'mean_$_magnitude', 'pct_long']].to_string(index=False))


if __name__ == '__main__':
    main()
