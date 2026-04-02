"""
Build Feature Atlas — pre-compute 13D features for ALL timeframes.

Saves to DATA/ATLAS_FEATURES/{tf}/YYYY_MM.parquet
Each parquet has: timestamp + 13D features

These are OBSERVER features — pre-computed, looked up by timestamp.
The active TF (1m) is computed live in the ticker.

Usage:
  python tools/build_feature_atlas.py           # all TFs
  python tools/build_feature_atlas.py 1h,1D     # specific TFs
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')

import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from training.train_trade_cnn import extract_features_13d, FEATURE_NAMES_13D

ATLAS_ROOT = 'DATA/ATLAS'
OUT_ROOT = 'DATA/ATLAS_FEATURES'

TFS = sys.argv[1].split(',') if len(sys.argv) > 1 else [
    '1s', '5s', '15s', '30s', '1m', '3m', '5m', '15m', '30m', '1h', '4h', '1D'
]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f'Building Feature Atlas')
    print(f'  TFs: {TFS}')
    print(f'  Output: {OUT_ROOT}/{{tf}}/YYYY_MM.parquet')
    print()

    for tf in TFS:
        tf_dir = os.path.join(ATLAS_ROOT, tf)
        out_dir = os.path.join(OUT_ROOT, tf)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
        if not files:
            print(f'  {tf}: no data, skipping')
            continue

        print(f'  {tf}: {len(files)} files')

        for fpath in tqdm(files, desc=f'    {tf}'):
            fname = os.path.basename(fpath)
            out_path = os.path.join(out_dir, fname)

            # Skip if already computed
            if os.path.exists(out_path):
                continue

            df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
            if len(df) < 30:
                continue

            sfe = StatisticalFieldEngine()
            states = sfe.batch_compute_states(df)
            feats = extract_features_13d(states, df)
            del states, sfe; gc.collect()

            # Build output: timestamp + 13D features
            out_df = pd.DataFrame({
                'timestamp': df['timestamp'].values,
            })
            for fi, fname_f in enumerate(FEATURE_NAMES_13D):
                out_df[fname_f] = feats[:, fi]

            out_df.to_parquet(out_path, index=False)
            del df, feats, out_df; gc.collect()

        print(f'    Done: {out_dir}/')

    print(f'\nFeature Atlas complete: {OUT_ROOT}/')


if __name__ == '__main__':
    main()
