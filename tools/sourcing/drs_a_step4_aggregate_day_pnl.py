"""DRS Path A - Step 4: aggregate per-leg PnL to per-day, write final target.

Combines IS hardened forward pass output (from step 3) with the existing
deliverable's OOS hardened forward pass output. Aggregates per-leg PnL
under the gbm_ev sizing scheme to per-day totals. Writes a final dataset
that joins cross-day features with hardened day-PnL targets.

Inputs:
  DATA/CROSS_DAY/predictions_IS/composite_forward_pass_hardened_IS.csv
  deliverables/composite_zigzag_pipeline/caches/composite_forward_pass_hardened.csv  (OOS)
  DATA/CROSS_DAY/cross_day_features.parquet

Output:
  DATA/CROSS_DAY/cross_day_features_with_target.parquet
  Final dataset ready for DRS training (replaces Phase 1B's peeky proxy).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

IS_CSV   = Path('DATA/CROSS_DAY/predictions_IS/composite_forward_pass_hardened_IS.csv')
OOS_CSV  = Path('deliverables/composite_zigzag_pipeline/caches/composite_forward_pass_hardened.csv')
FEATS    = Path('DATA/CROSS_DAY/cross_day_features.parquet')

OUT_PATH = Path('DATA/CROSS_DAY/cross_day_features_with_target.parquet')


def gbm_ev(pred_R):
    return float(np.clip(max(pred_R - 1.0, 0.0), 0.0, 3.0))


def aggregate(legs: pd.DataFrame, source: str) -> pd.DataFrame:
    legs = legs.copy()
    legs['size_gbm_ev'] = legs['pred_amp_R_hardened'].apply(gbm_ev)
    legs['pnl_weighted'] = legs['pnl_usd'] * legs['size_gbm_ev']
    out = legs.groupby('day').agg(
        n_legs=('day', 'size'),
        day_pnl_flat_hardened=('pnl_usd', 'sum'),
        day_pnl_gbmev_hardened=('pnl_weighted', 'sum'),
        median_pred_amp_R=('pred_amp_R_hardened', 'median'),
    ).reset_index().rename(columns={'day': 'date_label'})
    out['source'] = source
    return out


def main():
    print('Loading per-leg forward pass outputs...')
    is_legs  = pd.read_csv(IS_CSV)
    oos_legs = pd.read_csv(OOS_CSV)
    print(f'  IS  legs: {len(is_legs):,} / {is_legs["day"].nunique()} days')
    print(f'  OOS legs: {len(oos_legs):,} / {oos_legs["day"].nunique()} days')

    is_day  = aggregate(is_legs,  'ATLAS')
    oos_day = aggregate(oos_legs, 'NT8')
    all_day = pd.concat([is_day, oos_day], ignore_index=True).sort_values('date_label')

    print(f'\nDay-level target rows: {len(all_day)}')

    print(f'Loading cross-day features: {FEATS}')
    feats = pd.read_parquet(FEATS)
    print(f'  {len(feats)} feature rows')

    # Drop the stale NaN target column if present, then join
    feats = feats.drop(columns=['target_day_pnl'], errors='ignore')
    merged = feats.merge(
        all_day[['date_label', 'day_pnl_gbmev_hardened', 'day_pnl_flat_hardened',
                 'n_legs', 'median_pred_amp_R']],
        on='date_label', how='left',
    )
    # Use gbm_ev as primary target
    merged['target_day_pnl'] = merged['day_pnl_gbmev_hardened']

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f'\nWrote: {OUT_PATH}  ({len(merged)} rows)')

    has_target = merged['target_day_pnl'].notna().sum()
    print(f'  rows with target_day_pnl populated: {has_target}/{len(merged)}')
    print()
    print('--- Summary by source ---')
    for src in ('ATLAS', 'NT8'):
        sub = merged[merged['source'] == src].dropna(subset=['target_day_pnl'])
        if len(sub) == 0: continue
        pnl = sub['target_day_pnl'].values
        print(f'  {src:<6}  n={len(sub):3d}  mean ${pnl.mean():+.0f}  '
              f'median ${np.median(pnl):+.0f}  '
              f'min ${pnl.min():+.0f}  max ${pnl.max():+.0f}  '
              f'pos {int((pnl > 0).sum())}/{len(sub)}')


if __name__ == '__main__':
    main()
