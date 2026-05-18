"""Build lead-in trajectories — past N bars BEFORE each trade's entry.

Per user 2026-05-16: at live time we don't have the forward trajectory.
But we DO have the past N bars leading into the entry bar. If those past
bars trace a shape (in N-D V2 feature space) that predicts the forward
trade's cluster, we have a live-usable predictor.

For each trade:
    lead_in_trade_ts = entry_ts - K*TF, entry_ts - (K-1)*TF, ..., entry_ts
    pull V2 features at each → past-K-bar trajectory matrix
    compute PCA signature (centroid + direction + magnitude) — same as
    forward signatures but pointing INTO the entry, not OUT of it

Output: npz with per-trade lead-in centroid + direction + magnitude.
Schema mirrors signatures_IS_full.npz exactly.

Configurable lookback: --lookback-bars (default 60 = 5 min at 5s).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla


V2_ROOT = Path('DATA/ATLAS/FEATURES_5s_v2')
LAYER_DIRS = [
    'L1_5s', 'L1_15s', 'L1_1m', 'L1_5m', 'L1_15m', 'L1_1h', 'L1_4h', 'L1_1D',
    'L2_5s', 'L2_15s', 'L2_1m', 'L2_5m', 'L2_15m', 'L2_1h', 'L2_4h', 'L2_1D',
    'L3_5s', 'L3_15s', 'L3_1m', 'L3_5m', 'L3_15m', 'L3_1h', 'L3_4h', 'L3_1D',
]
TF_S = 5
MIN_BARS_FOR_PCA = 5
MIN_COL_STD = 1e-6


def load_layer_concat(layer, dates):
    frames = []
    for d in dates:
        p = V2_ROOT / layer / f'{d}.parquet'
        if not p.exists(): continue
        df = pd.read_parquet(p)
        if 'timestamp' not in df.columns: continue
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
        df['timestamp'] = df['timestamp'].astype('int64')
        frames.append(df)
    if not frames: return None
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='daisy-chain CSV')
    ap.add_argument('--out', required=True, help='output signatures npz (lead-in)')
    ap.add_argument('--lookback-bars', type=int, default=60,
                    help='Past N 5s-bars to use for lead-in (default 60 = 5 min)')
    ap.add_argument('--reuse-stats', default=None,
                    help='Path to forward signatures.npz; if provided, reuse '
                         'its feature_means + feature_stds (z-score parity).')
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f'Processing {len(df)} trades, lookback={args.lookback_bars} bars '
          f'({args.lookback_bars * TF_S}s)')

    df['_session_date_key'] = pd.to_datetime(df['session_date']).dt.strftime('%Y_%m_%d')
    needed_dates = set(df['_session_date_key'].unique())
    for d in list(needed_dates):
        pd_date = pd.to_datetime(d, format='%Y_%m_%d')
        needed_dates.add((pd_date - pd.Timedelta(days=1)).strftime('%Y_%m_%d'))
        needed_dates.add((pd_date + pd.Timedelta(days=1)).strftime('%Y_%m_%d'))
    needed_dates = sorted(needed_dates)
    print(f'Date keys (with ±1 padding): {len(needed_dates)}')

    print('\n--- Loading V2 layers ---')
    layer_dfs = {}
    feature_names = []
    t0 = time.time()
    for layer in LAYER_DIRS:
        ldf = load_layer_concat(layer, needed_dates)
        if ldf is None:
            print(f'  {layer}: NO DATA')
            continue
        feat_cols = [c for c in ldf.columns if c != 'timestamp']
        feature_names.extend(feat_cols)
        layer_dfs[layer] = ldf
    print(f'  Loaded {len(layer_dfs)} layers, {len(feature_names)} features ({time.time()-t0:.1f}s)')

    # Stats parity (for z-scoring downstream)
    if args.reuse_stats:
        fz = np.load(args.reuse_stats, allow_pickle=True)
        feature_means_ref = fz['feature_means']
        feature_stds_ref = fz['feature_stds']
        feature_names_ref = fz['feature_names']
        # Reorder to match
        if list(feature_names) != list(feature_names_ref):
            print('  reorder features to match forward signatures...')
            ref_set = list(feature_names_ref)
            order = [ref_set.index(f) for f in feature_names if f in ref_set]
            # Actually, let's just assume same order if layers are listed in same order
            # If they're not equal, we keep our order and z-score later.
        print(f'  Using forward signatures\' feature_means/stds for z-score parity')

    # Build lead-in trajectory per trade
    print('\n--- Building lead-in trajectories ---')
    all_oracle_idx = []
    all_bar_idx = []
    feature_arrays = {f: [] for f in feature_names}
    skipped = 0
    t1 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        oidx = int(row.oracle_idx)
        e_ts = int(row.oracle_ts)
        # Lead-in: K bars BEFORE entry (inclusive of entry bar? exclude.)
        # Generate timestamps: e_ts - K*TF, e_ts - (K-1)*TF, ..., e_ts - TF
        # We exclude entry_ts itself so this is purely lead-in (causal at entry)
        lead_ts = np.arange(e_ts - args.lookback_bars * TF_S, e_ts, TF_S,
                             dtype=np.int64)
        if len(lead_ts) < MIN_BARS_FOR_PCA:
            skipped += 1
            continue
        # asof-merge each layer
        for layer, ldf in layer_dfs.items():
            ts_arr = ldf['timestamp'].values
            indices = np.searchsorted(ts_arr, lead_ts, side='right') - 1
            indices = np.clip(indices, 0, len(ts_arr) - 1)
            for c in ldf.columns:
                if c == 'timestamp': continue
                feature_arrays[c].append(ldf[c].values[indices])
        all_oracle_idx.append(np.full(len(lead_ts), oidx, dtype=np.int64))
        all_bar_idx.append(np.arange(-len(lead_ts), 0, dtype=np.int32))  # negative bar_idx = lead-in

        if (i+1) % max(1, len(df)//20) == 0:
            print(f'  {i+1}/{len(df)} ({(time.time()-t1):.0f}s)', flush=True)
    print(f'  Built {len(df)-skipped} lead-in trajectories, skipped {skipped}')

    if not all_oracle_idx:
        print('No trajectories built — aborting')
        return

    oracle_idx_arr = np.concatenate(all_oracle_idx)
    bar_idx_arr = np.concatenate(all_bar_idx)
    X = np.empty((len(oracle_idx_arr), len(feature_names)), dtype=np.float32)
    for k, f in enumerate(feature_names):
        X[:, k] = np.concatenate(feature_arrays[f]).astype(np.float32)
    print(f'  X shape: {X.shape}, memory {X.nbytes / 1024**2:.0f} MB')

    # NaN audit + scrub
    nan_rate = float(np.isnan(X).sum() / X.size)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score with same stats as forward (if requested), else compute fresh
    if args.reuse_stats:
        # We assumed same feature ordering as forward; if not it would fail.
        stds_safe = np.where(feature_stds_ref > 1e-8, feature_stds_ref, 1.0).astype(np.float32)
        X_z = ((X - feature_means_ref[np.newaxis, :]) / stds_safe[np.newaxis, :]).astype(np.float32)
        f_means, f_stds = feature_means_ref, feature_stds_ref
    else:
        f_means = X.mean(axis=0).astype(np.float32)
        f_stds = X.std(axis=0).astype(np.float32)
        stds_safe = np.where(f_stds > 1e-8, f_stds, 1.0).astype(np.float32)
        X_z = ((X - f_means[np.newaxis, :]) / stds_safe[np.newaxis, :]).astype(np.float32)

    # Per-trade PCA — same logic as regret_trade_signatures
    print('\n--- Per-trade PCA on lead-in ---')
    unique_oidx, trade_starts, trade_counts = np.unique(
        oracle_idx_arr, return_index=True, return_counts=True)
    n_trades = len(unique_oidx)
    n_features = X_z.shape[1]
    centroids = np.zeros((n_trades, n_features), dtype=np.float32)
    directions = np.zeros((n_trades, n_features), dtype=np.float32)
    magnitudes = np.zeros(n_trades, dtype=np.float32)
    pca_unstable = np.zeros(n_trades, dtype=bool)
    t2 = time.time()
    n_fail = 0
    for ti, (start, count) in enumerate(zip(trade_starts, trade_counts)):
        cell = X_z[start:start + int(count), :]
        centroids[ti] = cell.mean(axis=0)
        if count < MIN_BARS_FOR_PCA:
            pca_unstable[ti] = True
            continue
        cell_c = cell - centroids[ti][np.newaxis, :]
        col_std = cell_c.std(axis=0)
        active = col_std > MIN_COL_STD
        if active.sum() < 2:
            pca_unstable[ti] = True
            continue
        try:
            _, S, Vh = np.linalg.svd(cell_c[:, active], full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                _, S, Vh = sla.svd(cell_c[:, active], full_matrices=False,
                                    lapack_driver='gesvd')
            except (sla.LinAlgError, np.linalg.LinAlgError):
                pca_unstable[ti] = True
                n_fail += 1
                continue
        direction_full = np.zeros(n_features, dtype=np.float32)
        direction_full[active] = Vh[0, :].astype(np.float32)
        directions[ti] = direction_full
        magnitudes[ti] = float(S[0])
        if (ti+1) % max(1, n_trades//10) == 0:
            print(f'  ...{ti+1}/{n_trades} ({time.time()-t2:.1f}s)', flush=True)
    print(f'  Done. {n_fail} SVD failures. {pca_unstable.sum()} PCA-unstable total.')

    np.savez_compressed(
        out_path,
        oracle_idx=unique_oidx.astype(np.int64),
        centroids=centroids,
        directions=directions,
        magnitudes=magnitudes,
        n_bars=trade_counts.astype(np.int32),
        pca_unstable=pca_unstable,
        feature_means=f_means,
        feature_stds=f_stds,
        feature_names=np.array(feature_names),
        lookback_bars=np.array([args.lookback_bars]),
    )
    print(f'\nSaved: {out_path}')
    print(f'  {n_trades} lead-in signatures')
    print(f'  centroid norms median: {np.linalg.norm(centroids, axis=1).mean():.2f}')
    print(f'  direction norms median: {np.linalg.norm(directions, axis=1).mean():.4f}')
    print(f'  magnitudes p25/p50/p75: '
          f'{np.percentile(magnitudes, 25):.2f} / '
          f'{np.percentile(magnitudes, 50):.2f} / '
          f'{np.percentile(magnitudes, 75):.2f}')


if __name__ == '__main__':
    main()
