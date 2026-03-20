"""Pre-compute TBN states for live startup.

Loads ATLAS + ATLAS_OOS parquets for all TFs, computes MarketState
via StatisticalFieldEngine, and saves as pickle. The live launcher
loads these instead of recomputing on startup.

Usage: python tools/precompute_live_states.py
  Run after training or whenever ATLAS data changes.
  Output: checkpoints/live/precomputed_states.pkl

The live launcher checks if this file exists and loads it.
If missing, falls back to computing from raw parquets (slow).
"""
import os
import sys
import pickle
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine


def load_tf_data(tf_label: str) -> pd.DataFrame:
    """Load all parquets for a TF from ATLAS + ATLAS_OOS."""
    chunks = []
    for root in ['DATA/ATLAS', 'DATA/ATLAS_OOS']:
        tf_dir = os.path.join(root, tf_label)
        if not os.path.isdir(tf_dir):
            continue
        for fn in sorted(os.listdir(tf_dir)):
            if not fn.endswith('.parquet'):
                continue
            df = pd.read_parquet(os.path.join(tf_dir, fn))
            if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
                df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
            chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset='timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)


def main():
    t0 = time.time()
    engine = StatisticalFieldEngine()
    result = {}

    # Only pre-compute TFs needed for live parity.
    # 1m: critical (F_momentum for sensor gate, volume_delta)
    # 4h, 1h, 30m, 15m, 5m, 3m: TBN workers (moderate size)
    # 15s, 5s, 1s: too large to pickle, computed from NT8 history instead
    tfs = ['1m', '4h', '1h', '30m', '15m', '5m', '3m']

    for tf in tfs:
        print(f'Loading {tf}...', end=' ', flush=True)
        df = load_tf_data(tf)
        if len(df) < 25:
            print(f'skipped ({len(df)} bars)')
            continue
        print(f'{len(df):,} bars -> ', end='', flush=True)
        states = engine.batch_compute_states(df, use_cuda=True)
        result[tf] = {
            'df': df,
            'states': states,
            'n_bars': len(df),
            'ts_min': float(df['timestamp'].min()),
            'ts_max': float(df['timestamp'].max()),
        }
        print(f'{len(states):,} states')

    out_path = 'checkpoints/live/precomputed_states.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    elapsed = time.time() - t0
    print(f'\nSaved: {out_path} ({size_mb:.1f} MB)')
    print(f'TFs: {list(result.keys())}')
    print(f'Time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
