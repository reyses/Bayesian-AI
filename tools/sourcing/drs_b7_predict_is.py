"""Apply trained B7 leg-amplitude regressor to IS legs.

The deliverable's `b7_leg_sizer_IS.parquet` is the IS TRAINING dataset (features
+ labels). It does NOT carry `pred_amp_R` because predictions weren't cached at
training time. This script:

  1. loads the trained B7 model
  2. predicts `pred_amp_R` on the IS legs (in-sample predictions — note this is
     legitimately train-set inference, but B7's monotonic ranking is well-
     calibrated even in-sample)
  3. writes an enriched parquet for downstream peeky-PnL aggregation

Output: DATA/CROSS_DAY/predictions_IS/b7_leg_sizer_IS_with_preds.parquet

Read-only: does NOT modify the deliverable's pkl or caches.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DELIVER = Path('deliverables/composite_zigzag_pipeline')
B7_PKL  = DELIVER / 'models' / 'b7_leg_sizer.pkl'
IS_LEGS = DELIVER / 'caches' / 'b7_leg_sizer_IS.parquet'

OUT_DIR = Path('DATA/CROSS_DAY/predictions_IS')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'b7_leg_sizer_IS_with_preds.parquet'


def main():
    print(f'Loading B7 model:  {B7_PKL}')
    with open(B7_PKL, 'rb') as f:
        b7 = pickle.load(f)
    model = b7['model']
    v2_cols = b7['v2_cols']
    print(f'  v2_cols: {len(v2_cols)}   IS MAE: {b7.get("is_mae","?"):.3f}   OOS MAE: {b7.get("oos_mae","?"):.3f}')

    print(f'Loading IS legs:   {IS_LEGS}')
    legs = pd.read_parquet(IS_LEGS)
    print(f'  rows: {len(legs):,}  days: {legs["day"].nunique()}')

    X = legs[v2_cols].fillna(0.0).values.astype(np.float32)
    print('Predicting pred_amp_R...')
    legs['pred_amp_R'] = model.predict(X)

    print(f'\npred_amp_R stats: median={legs["pred_amp_R"].median():.3f}  '
          f'mean={legs["pred_amp_R"].mean():.3f}  '
          f'p25={legs["pred_amp_R"].quantile(0.25):.3f}  '
          f'p75={legs["pred_amp_R"].quantile(0.75):.3f}')
    # Sanity: should track actual leg_amp_R monotonically
    rho = float(np.corrcoef(legs['pred_amp_R'], legs['leg_amp_R'])[0, 1])
    print(f'Pearson(pred, actual) on IS: {rho:.4f}  (should be ~0.5+ since model was trained on this)')

    # Drop V2 cols to keep output small — only need the leg-level summary
    keep = ['day', 'entry_ts', 'leg_dir', 'entry_price',
            'leg_amp_pts', 'leg_amp_usd', 'leg_amp_R',
            'pnl_at_R_usd', 'r_price', 'atr_pts', 'leg_duration_s',
            'pred_amp_R']
    legs[keep].to_parquet(OUT_PATH, index=False)
    print(f'\nWrote: {OUT_PATH}  ({len(legs):,} legs)')


if __name__ == '__main__':
    main()
