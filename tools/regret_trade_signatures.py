"""L3 Phase 2 — per-trade PCA signatures from N-D trajectories.

Per locked protocol (research/bayesian_archetypes/project.md): each trade
gets a signature consisting of:
    - centroid (N-D vector): mean of trajectory points
    - direction (N-D unit vector): first principal axis from SVD
    - magnitude (scalar): top singular value (relative trajectory length)

Implementation note: I initially tried batched torch.linalg.svd on CUDA,
but it failed for ~all trades due to ill-conditioning (V2 features are
highly correlated even after corr-cluster pruning, and the NaN-fills
create duplicate rows). Switched to numpy.linalg.svd per-trade — LAPACK's
gesdd is robust to ill-conditioning. Speed is ~50ms/trade on CPU which
is acceptable (full 7,925 trades ≈ 7 min). If speed becomes the bottleneck,
torch.svd_lowrank with q=1 is the GPU fallback.

Features are z-scored globally (using the same μ, σ across all bars)
before SVD so distance is in std units later.

Input: trajectories_*.npz (output of regret_join_v2_trajectories.py)
Output: signatures_*.npz with
    oracle_idx: int array (one per trade)
    centroids: float matrix (n_trades × n_features)
    directions: float matrix (n_trades × n_features) — unit vectors
    magnitudes: float array (n_trades,) — top singular value
    n_bars: int array (n_trades,) — number of bars used per trade
    feature_means: float array (n_features,) — global μ used for z-scoring
    feature_stds: float array (n_features,) — global σ used for z-scoring
    feature_names: array of strings

Trades with T_bars < MIN_BARS_FOR_PCA (default 5) are flagged but still
processed; their PCA is unstable but the centroid is still meaningful.

Edge handling for NaN: replace with column mean before z-scoring.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla


MIN_BARS_FOR_PCA = 5
MIN_COL_STD_PER_TRADE = 1e-6   # drop columns that are essentially constant during a trade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='trajectories npz')
    ap.add_argument('--out', required=True, help='output signatures npz')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.input}')
    data = np.load(args.input, allow_pickle=True)
    oracle_idx = data['oracle_idx']
    X_np = data['X']
    feature_names = data['feature_names']
    n_total_bars, n_features = X_np.shape
    print(f'  total bars: {n_total_bars:,}  features: {n_features}')

    # Global z-score normalization
    # Step 1: replace NaN with column mean; columns that are entirely NaN
    # get filled with 0 (no info to recover).
    col_means = np.nanmean(X_np, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_mask = np.isnan(X_np)
    X_np = np.where(nan_mask, col_means[np.newaxis, :], X_np)
    # Belt-and-suspenders: scrub any remaining NaN/inf
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: z-score
    feature_means = X_np.mean(axis=0).astype(np.float32)
    feature_stds = X_np.std(axis=0).astype(np.float32)
    # Avoid div by zero
    feature_stds_safe = np.where(feature_stds > 1e-8, feature_stds, 1.0).astype(np.float32)
    X_z = ((X_np - feature_means[np.newaxis, :]) / feature_stds_safe[np.newaxis, :]).astype(np.float32)
    print(f'  z-scored. NaN-filled: {nan_mask.sum():,} ({100*nan_mask.sum()/X_np.size:.2f}%)')

    # Group bars by oracle_idx (already sorted within each trade since they came in order)
    unique_oidx, trade_starts, trade_counts = np.unique(
        oracle_idx, return_index=True, return_counts=True
    )
    n_trades = len(unique_oidx)
    print(f'  n_trades: {n_trades:,}')

    # Allocate outputs
    centroids = np.zeros((n_trades, n_features), dtype=np.float32)
    directions = np.zeros((n_trades, n_features), dtype=np.float32)
    magnitudes = np.zeros(n_trades, dtype=np.float32)
    n_bars_per_trade = trade_counts.astype(np.int32)
    pca_unstable = np.zeros(n_trades, dtype=bool)

    # Per-trade PCA via numpy.linalg.svd (LAPACK gesdd — robust to ill-conditioning)
    print('\n--- Computing per-trade PCA (numpy SVD, per trade) ---')
    t0 = time.time()
    n_fail = 0

    for ti, (start, count) in enumerate(zip(trade_starts, trade_counts)):
        cell = X_z[start:start + int(count), :]   # (T × F)
        # Always compute centroid
        centroids[ti, :] = cell.mean(axis=0)
        if count < MIN_BARS_FOR_PCA:
            pca_unstable[ti] = True
            continue
        # Center for SVD
        cell_c = cell - centroids[ti, :][np.newaxis, :]
        # Drop columns that are essentially constant within THIS trade —
        # these are slow-TF features (L_1h/4h/1D) whose update period
        # exceeds the trade duration. After centering they're zero columns,
        # which causes LAPACK gesdd to fail to converge in combination with
        # other near-collinear columns.
        col_std = cell_c.std(axis=0)
        active_mask = col_std > MIN_COL_STD_PER_TRADE
        if active_mask.sum() < 2:
            pca_unstable[ti] = True
            continue
        cell_active = cell_c[:, active_mask]
        # Try numpy gesdd first, then scipy gesvd (more robust)
        try:
            _, S, Vh = np.linalg.svd(cell_active, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                _, S, Vh = sla.svd(cell_active, full_matrices=False,
                                    lapack_driver='gesvd')
            except (sla.LinAlgError, np.linalg.LinAlgError):
                pca_unstable[ti] = True
                n_fail += 1
                continue
        # Map direction back to full feature space (inactive cols = 0)
        direction_full = np.zeros(n_features, dtype=np.float32)
        direction_full[active_mask] = Vh[0, :].astype(np.float32)
        directions[ti, :] = direction_full
        magnitudes[ti] = float(S[0])

        if (ti + 1) % max(1, n_trades // 10) == 0:
            print(f'  ...{ti+1}/{n_trades}  '
                  f'({time.time()-t0:.1f}s, '
                  f'{(time.time()-t0)/(ti+1)*1000:.1f}ms/trade)', flush=True)

    total_processed = n_trades - n_fail
    if n_fail:
        print(f'  SVD failed (numpy LinAlgError) on {n_fail} trades')

    print(f'  Done. Total processed: {total_processed}/{n_trades}  '
          f'({time.time()-t0:.1f}s, '
          f'{(time.time()-t0)/max(1,n_trades)*1000:.1f}ms/trade)')
    n_unstable = int(pca_unstable.sum())
    print(f'  PCA-unstable (T < {MIN_BARS_FOR_PCA}): {n_unstable}')

    # Save
    print(f'\nSaving to {out_path}')
    np.savez_compressed(
        out_path,
        oracle_idx=unique_oidx.astype(np.int64),
        centroids=centroids,
        directions=directions,
        magnitudes=magnitudes,
        n_bars=n_bars_per_trade,
        pca_unstable=pca_unstable,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_names=feature_names,
    )
    print('Saved.')

    # Sanity stats
    print(f'\n--- Sanity stats ---')
    print(f'  centroid norms: median {np.linalg.norm(centroids, axis=1).mean():.2f}, '
          f'max {np.linalg.norm(centroids, axis=1).max():.2f}')
    print(f'  direction norms (should be ~1.0 after SVD): '
          f'median {np.linalg.norm(directions, axis=1).mean():.4f}')
    print(f'  magnitudes: median {np.median(magnitudes):.2f}  '
          f'p25 {np.percentile(magnitudes, 25):.2f}  '
          f'p75 {np.percentile(magnitudes, 75):.2f}  '
          f'max {magnitudes.max():.2f}')


if __name__ == '__main__':
    main()
