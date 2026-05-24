"""Build lead-in sequence dataset for LSTM scenario classifier.

For each trade: pull V2 features at past K 5s bars (lead-in), select a
subset of N features, z-score using IS stats. Save as compact .npz with
sequence tensor + bucket labels.

Inputs:
  --input            daisy_with_v2_features parquet (IS or OOS)
  --buckets          daisy_<X>_buckets.csv (must have bucket_* cols)
  --features-file    feature_prune_representatives_<...>.txt (top-N by score)
  --top-n            number of features to take (default 30)
  --lookback-bars    K (default 60 = 5min @ 5s)
  --stats-in         z-score stats from IS (use for OOS); if None, compute fresh

Output:
  <out>.npz with:
    X:        (N_trades, K, N_features) float32 — z-scored
    y_dir:    (N_trades,) int8           — 1=LONG, 0=SHORT
    y_dur:    (N_trades,) int8           — 0..3 duration bucket
    y_spd:    (N_trades,) int8           — 0..3 speed bucket
    y_traj:   (N_trades,) int8           — 0..3 trajectory bucket
    oracle_idx: (N_trades,) int64
    feature_names: array of str
    feature_means: array of float32
    feature_stds:  array of float32
    lookback_bars: K
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


V2_ROOT = Path('DATA/ATLAS/FEATURES_5s_v2')
TF_S = 5


def date_keys_for(ts_unix: int) -> list:
    d = pd.to_datetime(ts_unix, unit='s')
    return [(d + pd.Timedelta(days=off)).strftime('%Y_%m_%d') for off in (-1, 0, 1)]


def find_feature_layer(feature_name: str) -> str:
    """Map feature name like 'L2_15s_price_velocity_12' to layer dir 'L2_15s'."""
    parts = feature_name.split('_')
    return f'{parts[0]}_{parts[1]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='daisy_with_v2_features parquet')
    ap.add_argument('--buckets', required=True,
                    help='daisy_X_buckets.csv with bucket_* columns')
    ap.add_argument('--features-file', required=True,
                    help='Text file with one feature name per line')
    ap.add_argument('--top-n', type=int, default=30)
    ap.add_argument('--lookback-bars', type=int, default=60)
    ap.add_argument('--stats-in', default=None,
                    help='npz with feature_means + feature_stds for z-score')
    ap.add_argument('--out', required=True, help='output .npz path')
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load features list (top-N by combined score)
    with open(args.features_file) as f:
        all_feats = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    feats = all_feats[:args.top_n]
    print(f'Using top-{len(feats)} features:')
    for f in feats[:10]:
        print(f'  {f}')
    if len(feats) > 10:
        print(f'  ...and {len(feats)-10} more')

    # Map feature -> layer
    feat_by_layer = {}
    for f in feats:
        layer = find_feature_layer(f)
        feat_by_layer.setdefault(layer, []).append(f)
    print(f'\nLayers needed: {list(feat_by_layer.keys())}')

    # Load buckets
    buckets = pd.read_csv(args.buckets)
    print(f'\nLoaded {len(buckets)} bucket labels')

    # Load V2 features for date range
    df_v2 = pd.read_parquet(args.input)
    print(f'Loaded {len(df_v2)} trades with V2 features')

    # Date range -> determine which date files to load (per layer)
    df_v2['_date'] = pd.to_datetime(df_v2['session_date']).dt.strftime('%Y_%m_%d')
    needed_dates = set(df_v2['_date'].unique())
    # Add ±1 padding for lead-in trajectories crossing midnight
    for d in list(needed_dates):
        pd_d = pd.to_datetime(d, format='%Y_%m_%d')
        needed_dates.add((pd_d - pd.Timedelta(days=1)).strftime('%Y_%m_%d'))
        needed_dates.add((pd_d + pd.Timedelta(days=1)).strftime('%Y_%m_%d'))
    needed_dates = sorted(needed_dates)
    print(f'Date keys (with ±1 pad): {len(needed_dates)}')

    # Load each layer's concatenated DF (timestamp + needed feature cols)
    print('\n--- Loading V2 layer parquets ---')
    layer_data = {}
    t0 = time.time()
    for layer, fcols in feat_by_layer.items():
        frames = []
        for d in needed_dates:
            p = V2_ROOT / layer / f'{d}.parquet'
            if not p.exists(): continue
            df_l = pd.read_parquet(p)
            # Filter columns
            keep_cols = ['timestamp'] + [c for c in fcols if c in df_l.columns]
            df_l = df_l[keep_cols]
            if 'timestamp' in df_l.columns and pd.api.types.is_datetime64_any_dtype(df_l['timestamp']):
                df_l['timestamp'] = df_l['timestamp'].astype('int64') // 10**9
            df_l['timestamp'] = df_l['timestamp'].astype('int64')
            frames.append(df_l)
        merged = pd.concat(frames, ignore_index=True).drop_duplicates('timestamp')
        merged = merged.sort_values('timestamp').reset_index(drop=True)
        layer_data[layer] = merged
        print(f'  {layer}: {len(merged)} bars, cols {list(merged.columns)}')
    print(f'  loaded in {time.time()-t0:.1f}s')

    # Build sequence tensor
    print('\n--- Building lead-in sequences ---')
    K = args.lookback_bars
    N = len(feats)
    n_trades = len(df_v2)
    X = np.zeros((n_trades, K, N), dtype=np.float32)
    pca_unstable = np.zeros(n_trades, dtype=bool)
    skipped = 0
    t1 = time.time()
    for ti, row in enumerate(df_v2.itertuples(index=False)):
        e_ts = int(row.oracle_ts)
        # Sequence: e_ts - K*TF .. e_ts - TF  (forward pass, excludes entry bar)
        target_ts = np.arange(e_ts - K*TF_S, e_ts, TF_S, dtype=np.int64)
        # asof-merge per layer
        for layer, fcols in feat_by_layer.items():
            ldf = layer_data[layer]
            ts_arr = ldf['timestamp'].values
            idx_in = np.searchsorted(ts_arr, target_ts, side='right') - 1
            idx_in = np.clip(idx_in, 0, len(ts_arr) - 1)
            for fc in fcols:
                feat_pos = feats.index(fc)
                X[ti, :, feat_pos] = ldf[fc].values[idx_in].astype(np.float32)
        if (ti+1) % max(1, n_trades//20) == 0:
            print(f'  {ti+1}/{n_trades} ({time.time()-t1:.0f}s)', flush=True)
    print(f'  built X shape {X.shape}, memory {X.nbytes / 1024**2:.0f} MB')

    # Handle NaN
    nan_rate = float(np.isnan(X).sum() / X.size)
    if nan_rate > 0:
        print(f'  NaN rate: {nan_rate*100:.2f}%; replacing with 0')
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score
    if args.stats_in:
        print(f'\nLoading z-score stats from {args.stats_in}')
        sz = np.load(args.stats_in)
        means = sz['feature_means']
        stds = sz['feature_stds']
        ref_feats = list(sz['feature_names'])
        # Verify feature ordering
        if ref_feats != feats:
            print('WARNING: feature ordering differs from stats source')
        print(f'  Using IS stats for z-score parity')
    else:
        # Compute fresh stats per feature across all (trade, bar) pairs
        flat = X.reshape(-1, N)
        means = flat.mean(axis=0).astype(np.float32)
        stds = flat.std(axis=0).astype(np.float32)
    stds_safe = np.where(stds > 1e-8, stds, 1.0).astype(np.float32)
    X = ((X - means[None, None, :]) / stds_safe[None, None, :]).astype(np.float32)
    print(f'  z-scored; |X| p99: {np.percentile(np.abs(X), 99):.2f}')

    # Align labels (buckets are indexed by oracle_idx)
    print('\n--- Aligning bucket labels ---')
    bk_by_oid = buckets.set_index('oracle_idx')
    y_dir = np.zeros(n_trades, dtype=np.int8)
    y_dur = np.zeros(n_trades, dtype=np.int8)
    y_spd = np.zeros(n_trades, dtype=np.int8)
    y_traj = np.zeros(n_trades, dtype=np.int8)
    for ti, oid in enumerate(df_v2['oracle_idx'].values):
        if int(oid) in bk_by_oid.index:
            r = bk_by_oid.loc[int(oid)]
            y_dir[ti] = int(r['bucket_direction'])
            y_dur[ti] = int(r['bucket_duration'])
            y_spd[ti] = int(r['bucket_speed'])
            y_traj[ti] = int(r['bucket_trajectory'])

    np.savez_compressed(
        out_path,
        X=X,
        y_dir=y_dir,
        y_dur=y_dur,
        y_spd=y_spd,
        y_traj=y_traj,
        oracle_idx=df_v2['oracle_idx'].values.astype(np.int64),
        feature_names=np.array(feats),
        feature_means=means,
        feature_stds=stds,
        lookback_bars=np.array([K]),
    )
    print(f'\nWrote: {out_path}')
    print(f'  X shape:   {X.shape}')
    print(f'  y_dir distribution:  {dict(zip(*np.unique(y_dir, return_counts=True)))}')
    print(f'  y_dur distribution:  {dict(zip(*np.unique(y_dur, return_counts=True)))}')
    print(f'  y_spd distribution:  {dict(zip(*np.unique(y_spd, return_counts=True)))}')
    print(f'  y_traj distribution: {dict(zip(*np.unique(y_traj, return_counts=True)))}')


if __name__ == '__main__':
    main()
