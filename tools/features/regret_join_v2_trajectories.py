"""Build per-trade N-D feature trajectories from daisy-chain trades × V2 features.

Per user 2026-05-16: this is L3 Phase 1. The single-bar V2 join already
exists (`regret_join_v2_features.py`). This tool extends it: for each
trade, pull the V2 features at EVERY 5s bar from entry_ts to exit_ts.

Output: a numpy npz with:
    oracle_idx: int array (len = total bars across all trades)
    bar_idx:    int array (which bar of the trade — 0 to T_bars-1)
    feature_columns: list of feature names
    X: float matrix (total_bars × n_features) — the trajectory data

This format keeps it long-format (one row per trade-bar) so it's easy to
group by oracle_idx for per-trade PCA in Phase 2. Total size estimate:
7,925 trades × ~444 median bars × 184 features × 4 bytes ≈ 2.6 GB.
For smoke test on 50 trades, ~22k bars × 184 features × 4B ≈ 16 MB.

Usage:
    # smoke (50 trades):
    python tools/regret_join_v2_trajectories.py \\
        --input reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv \\
        --n-trades 50 \\
        --out reports/findings/regret_oracle/trajectories_smoke.npz

    # full:
    python tools/regret_join_v2_trajectories.py \\
        --input reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv \\
        --out reports/findings/regret_oracle/trajectories_IS_full.npz
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


V2_ROOT = Path('DATA/ATLAS/FEATURES_5s_v2')
LAYER_DIRS = [
    'L1_5s', 'L1_15s', 'L1_1m', 'L1_5m', 'L1_15m', 'L1_1h', 'L1_4h', 'L1_1D',
    'L2_5s', 'L2_15s', 'L2_1m', 'L2_5m', 'L2_15m', 'L2_1h', 'L2_4h', 'L2_1D',
    'L3_5s', 'L3_15s', 'L3_1m', 'L3_5m', 'L3_15m', 'L3_1h', 'L3_4h', 'L3_1D',
]
TF_S = 5  # base bar TF for trajectory sampling


def load_layer_concat(layer: str, dates: list[str]) -> pd.DataFrame | None:
    """Load and concat a layer's parquets for a set of YYYY_MM_DD date strings."""
    frames = []
    for d in dates:
        p = V2_ROOT / layer / f'{d}.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if 'timestamp' not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
        df['timestamp'] = df['timestamp'].astype('int64')
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='daisy-chain CSV')
    ap.add_argument('--out', required=True, help='output npz path')
    ap.add_argument('--n-trades', type=int, default=None,
                    help='SMOKE: process only first N trades (default all)')
    ap.add_argument('--layers', nargs='*', default=LAYER_DIRS)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.n_trades:
        df = df.head(args.n_trades).reset_index(drop=True)
    print(f'Processing {len(df)} trades')

    # Identify the dates we need (entry's session_date AND prior date for safety)
    df['_session_date_key'] = (
        pd.to_datetime(df['session_date']).dt.strftime('%Y_%m_%d')
    )
    needed_dates = set(df['_session_date_key'].unique())
    for d in list(needed_dates):
        try:
            pd_date = pd.to_datetime(d, format='%Y_%m_%d')
            needed_dates.add((pd_date - pd.Timedelta(days=1)).strftime('%Y_%m_%d'))
            needed_dates.add((pd_date + pd.Timedelta(days=1)).strftime('%Y_%m_%d'))
        except Exception:
            pass
    needed_dates = sorted(needed_dates)
    print(f'Unique date keys (with ±1 padding): {len(needed_dates)}')

    # Load each layer's data once (concatenated across dates)
    print('\n--- Loading V2 layers ---')
    layer_dfs = {}
    feature_names = []
    t0 = time.time()
    for layer in args.layers:
        ldf = load_layer_concat(layer, needed_dates)
        if ldf is None:
            print(f'  {layer}: NO DATA')
            continue
        # Identify feature columns
        feat_cols = [c for c in ldf.columns if c != 'timestamp']
        feature_names.extend(feat_cols)
        layer_dfs[layer] = ldf
        print(f'  {layer}: {len(ldf):,} rows, {len(feat_cols)} features')
    print(f'  Total features across layers: {len(feature_names)}')
    print(f'  Layer load time: {time.time()-t0:.1f}s')

    # For each trade, generate timestamps from entry_ts to exit_ts (every TF_S sec)
    # then asof-merge each timestamp to each layer's data → stack columns
    print('\n--- Building trajectories ---')
    all_oracle_idx = []
    all_bar_idx = []
    feature_arrays = {f: [] for f in feature_names}  # accumulators
    skipped = 0
    t1 = time.time()

    for i, row in enumerate(df.itertuples(index=False)):
        oidx = int(row.oracle_idx)
        e_ts = int(row.oracle_ts)
        # Use time_to_mfe_min × 60 as the algorithmic trade duration —
        # NOT exit_ts − entry_ts. 17.5% of trades have exit_ts skewed by
        # data gaps the daisy-chain session detector missed (gaps < 30min).
        # time_to_mfe_min is the algorithm's bar-count truth (correct).
        ttm_min = float(row.time_to_mfe_min)
        if ttm_min <= 0:
            skipped += 1
            continue
        n_bars = int(round(ttm_min * 60.0 / TF_S)) + 1   # +1 to include entry bar
        if n_bars < 2:
            skipped += 1
            continue
        trade_ts = e_ts + np.arange(n_bars, dtype=np.int64) * TF_S
        # For each layer, asof-merge to find feature values at each trade_ts
        trade_features_per_layer = {}
        for layer, ldf in layer_dfs.items():
            # asof-backward search
            ts_arr = ldf['timestamp'].values
            # np.searchsorted with 'right' then -1 gives backward asof
            indices = np.searchsorted(ts_arr, trade_ts, side='right') - 1
            indices = np.clip(indices, 0, len(ts_arr) - 1)
            for c in ldf.columns:
                if c == 'timestamp':
                    continue
                trade_features_per_layer[c] = ldf[c].values[indices]
        # Append to accumulators
        all_oracle_idx.append(np.full(n_bars, oidx, dtype=np.int64))
        all_bar_idx.append(np.arange(n_bars, dtype=np.int32))
        for f in feature_names:
            if f in trade_features_per_layer:
                feature_arrays[f].append(trade_features_per_layer[f])
            else:
                feature_arrays[f].append(np.full(n_bars, np.nan, dtype=np.float32))

        if (i + 1) % max(1, len(df) // 20) == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            eta = (len(df) - (i + 1)) / rate
            print(f'  {i+1}/{len(df)}  ({rate:.1f} trades/sec, ETA {eta:.0f}s)', flush=True)

    print(f'  Built trajectories for {len(df) - skipped} trades  ({skipped} skipped)')
    print(f'  Build time: {time.time()-t1:.1f}s')

    if not all_oracle_idx:
        print('NO TRAJECTORIES BUILT — aborting')
        return

    # Concatenate
    print('\n--- Concatenating ---')
    oracle_idx_arr = np.concatenate(all_oracle_idx)
    bar_idx_arr = np.concatenate(all_bar_idx)
    X = np.empty((len(oracle_idx_arr), len(feature_names)), dtype=np.float32)
    for k, f in enumerate(feature_names):
        X[:, k] = np.concatenate(feature_arrays[f]).astype(np.float32)
    print(f'  Final shape: X = {X.shape}    oracle_idx = {oracle_idx_arr.shape}')
    print(f'  Memory: {X.nbytes / 1024**2:.1f} MB for X')

    # Save
    print(f'\nSaving to {out_path}...')
    np.savez_compressed(
        out_path,
        oracle_idx=oracle_idx_arr,
        bar_idx=bar_idx_arr,
        X=X,
        feature_names=np.array(feature_names),
    )
    print(f'Saved.')

    # NaN audit
    nan_rate = np.isnan(X).sum() / X.size
    print(f'NaN rate: {nan_rate*100:.3f}%')

    # Bar-count distribution per trade
    unique_oidx, counts = np.unique(oracle_idx_arr, return_counts=True)
    print(f'\nBars per trade: n_trades={len(unique_oidx)}  '
          f'min={counts.min()}  p25={int(np.percentile(counts, 25))}  '
          f'median={int(np.median(counts))}  p75={int(np.percentile(counts, 75))}  '
          f'max={counts.max()}')


if __name__ == '__main__':
    main()
