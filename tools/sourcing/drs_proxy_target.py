"""Compute peeky day-PnL proxy for the DRS feasibility check.

This is the FAST proxy for `target_day_pnl`. Uses the existing per-leg
amplitude (lookahead-OK at this layer) + B7's predicted-amp_R (for sizing
decisions) to compute the peeky version of:

  - flat sizing day pnl
  - gbm_ev sizing day pnl  (this is what the DRS will be sized on top of)

Peeky formula per leg (matches `composite_forward_pass.py` not `_hardened.py`):
  size_flat   = 1.0
  size_gbm_ev = clip(max(pred_amp_R - 1, 0), 0, 3)
  pnl_per_leg_unit = (leg_amp_pts - 2 * r_price) * $2/pt
  pnl_per_leg_sized = pnl_per_leg_unit * size - friction * size

The "- 2*r_price" subtracts R-trigger entry + R-trigger exit slippage from
the absolute leg amplitude (i.e., what the strategy can capture). Friction
$6/leg per the deliverable's parameters.

Inputs:
  IS:  DATA/CROSS_DAY/predictions_IS/b7_leg_sizer_IS_with_preds.parquet
  OOS: deliverables/composite_zigzag_pipeline/caches/b7_leg_sizer_OOS.parquet
       (already carries pred_amp_R)

Output: DATA/CROSS_DAY/day_pnl_proxy.parquet
        columns: date_label, source, n_legs, day_pnl_flat_peeky, day_pnl_gbmev_peeky
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

IS_LEGS  = Path('DATA/CROSS_DAY/predictions_IS/b7_leg_sizer_IS_with_preds.parquet')
OOS_LEGS = Path('deliverables/composite_zigzag_pipeline/caches/b7_leg_sizer_OOS.parquet')

OUT_PATH = Path('DATA/CROSS_DAY/day_pnl_proxy.parquet')

DOLLAR_PER_POINT   = 2.0
FRICTION_PER_LEG   = 6.0   # commission $4 + slippage $2 per round-trip


def gbm_ev_size(pred_R: pd.Series) -> pd.Series:
    """size = max(pred_R - 1, 0) clipped to [0, 3]."""
    return np.clip(np.maximum(pred_R - 1.0, 0.0), 0.0, 3.0)


def per_leg_pnl(legs: pd.DataFrame, scheme: str) -> pd.Series:
    """Peeky per-leg PnL under given sizing scheme."""
    raw_per_unit = (legs['leg_amp_pts'] - 2.0 * legs['r_price']) * DOLLAR_PER_POINT
    if scheme == 'flat':
        size = pd.Series(1.0, index=legs.index)
    elif scheme == 'gbm_ev':
        size = gbm_ev_size(legs['pred_amp_R'])
    else:
        raise ValueError(scheme)
    return raw_per_unit * size - FRICTION_PER_LEG * size


def aggregate_by_day(legs: pd.DataFrame, source: str) -> pd.DataFrame:
    legs = legs.copy()
    legs['pnl_flat']   = per_leg_pnl(legs, 'flat')
    legs['pnl_gbm_ev'] = per_leg_pnl(legs, 'gbm_ev')
    grp = legs.groupby('day').agg(
        n_legs=('day', 'size'),
        day_pnl_flat_peeky=('pnl_flat', 'sum'),
        day_pnl_gbmev_peeky=('pnl_gbm_ev', 'sum'),
        median_leg_amp_R=('leg_amp_R', 'median'),
        median_pred_amp_R=('pred_amp_R', 'median'),
    ).reset_index().rename(columns={'day': 'date_label'})
    grp['source'] = source
    return grp


def main():
    print(f'Loading IS legs:  {IS_LEGS}')
    is_legs  = pd.read_parquet(IS_LEGS)
    print(f'  {len(is_legs):,} legs / {is_legs["day"].nunique()} days')

    print(f'Loading OOS legs: {OOS_LEGS}')
    oos_legs = pd.read_parquet(OOS_LEGS)
    print(f'  {len(oos_legs):,} legs / {oos_legs["day"].nunique()} days')

    is_day  = aggregate_by_day(is_legs,  'ATLAS')
    oos_day = aggregate_by_day(oos_legs, 'NT8')

    out = pd.concat([is_day, oos_day], ignore_index=True)
    out = out.sort_values('date_label').reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(f'\nWrote: {OUT_PATH}  ({len(out)} days)')
    print()
    print('--- Day-PnL summary (peeky, gbm_ev) ---')
    for src in ('ATLAS', 'NT8'):
        sub = out[out['source'] == src]
        pnl = sub['day_pnl_gbmev_peeky'].values
        print(f'  {src:<6}  n={len(sub):3d}  mean=${pnl.mean():+.0f}/day  '
              f'median=${np.median(pnl):+.0f}/day  '
              f'min=${pnl.min():+.0f}  max=${pnl.max():+.0f}  '
              f'pos {int((pnl > 0).sum())}/{len(sub)}')

    print()
    print('--- Day-PnL summary (peeky, flat) ---')
    for src in ('ATLAS', 'NT8'):
        sub = out[out['source'] == src]
        pnl = sub['day_pnl_flat_peeky'].values
        print(f'  {src:<6}  n={len(sub):3d}  mean=${pnl.mean():+.0f}/day  '
              f'median=${np.median(pnl):+.0f}/day  '
              f'min=${pnl.min():+.0f}  max=${pnl.max():+.0f}  '
              f'pos {int((pnl > 0).sum())}/{len(sub)}')

    print()
    print('--- First/last 5 rows ---')
    print(out.head().to_string())
    print('...')
    print(out.tail().to_string())


if __name__ == '__main__':
    main()
