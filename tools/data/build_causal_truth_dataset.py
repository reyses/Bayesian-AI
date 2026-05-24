"""Retag the IS truth dataset using CAUSAL pivots (not offline).

The existing `zigzag_pivot_dataset_IS_atr4.parquet` carries V2 features per
1m bar plus offline-zigzag `is_pivot` tags. The downstream trainers
(`tools/train_b{1..8}.py`) consume the pivot tags as their event source and
derive per-pivot targets (leg amplitude, direction, etc.) ON THE FLY from the
5s bars between consecutive `is_pivot==1` rows. So to retrain on HONEST
(causal) data, we only need to swap out the pivot tags — the V2 features and
all downstream target derivation stay correct.

This tool zeroes the offline tags and re-tags `is_pivot=1` at the 1m bar
at-or-just-before each causal leg's `entry_ts`.

Default I/O:
  base-truth   reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet
  causal-legs  reports/findings/trade_outcome_table/causal_flat_zigzag_legs_IS.csv
  out          reports/findings/regret_oracle/zigzag_pivot_dataset_CAUSAL_IS_atr4.parquet

Caveat: base-truth's day span (2025-only by default) caps the retag. Legs
on days outside the base-truth span (e.g. Q1 2026 Databento) are dropped
with a count printed; extend the base-truth first if you want them included.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet',
                    help='source V2-per-1m-bar parquet whose pivot tags will be replaced')
    ap.add_argument('--causal-legs',
                    default='reports/findings/trade_outcome_table/causal_flat_zigzag_legs_IS.csv',
                    help='causal leg list whose entry_ts marks the new is_pivot=1 bars')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_CAUSAL_IS_atr4.parquet',
                    help='output parquet')
    args = ap.parse_args()

    base = (REPO / args.base_truth) if not Path(args.base_truth).is_absolute() else Path(args.base_truth)
    legs_path = (REPO / args.causal_legs) if not Path(args.causal_legs).is_absolute() else Path(args.causal_legs)
    out_path = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    print(f'Loading base truth: {base}')
    truth = pd.read_parquet(base)
    n_offline_piv = int((truth['is_pivot'] == 1).sum())
    print(f'  rows={len(truth):,}  days={truth.day.nunique()}  '
          f'orig is_pivot=1: {n_offline_piv:,}')

    # Zero out the offline pivot tags. Downstream V2 columns are untouched.
    truth['is_pivot'] = 0
    truth['pivot_dir'] = ''
    truth['pivot_price'] = np.nan

    print(f'Loading causal legs: {legs_path}')
    legs = pd.read_csv(legs_path)
    legs['day'] = legs['day'].astype(str)
    print(f'  legs={len(legs):,}  days={legs.day.nunique()}  '
          f'span={legs.day.min()} -> {legs.day.max()}')

    truth_days = set(truth.day.astype(str).unique())
    in_span = legs[legs.day.isin(truth_days)]
    dropped = len(legs) - len(in_span)
    print(f'  legs in truth span: {len(in_span):,}  (dropped {dropped:,} '
          'outside truth — extend base-truth to include them if needed)')

    # Tag is_pivot at the 1m bar at-or-just-before each leg entry_ts.
    tagged = 0
    truth_by_day = {d: g for d, g in truth.groupby(truth.day.astype(str))}
    for day, day_legs in tqdm(in_span.groupby('day'),
                              total=in_span.day.nunique(), desc='retag'):
        td = truth_by_day.get(day)
        if td is None or len(td) == 0:
            continue
        ts = td.timestamp.values.astype(np.int64)
        idx = td.index.values
        for _, leg in day_legs.iterrows():
            ent = int(leg['entry_ts'])
            i = int(np.searchsorted(ts, ent, side='right') - 1)
            if i < 0:
                continue
            row_idx = idx[i]
            truth.at[row_idx, 'is_pivot'] = 1
            truth.at[row_idx, 'pivot_dir'] = str(leg['leg_dir'])
            truth.at[row_idx, 'pivot_price'] = float(leg['entry_price'])
            tagged += 1

    print(f'\nTagged {tagged:,} causal pivot bars  '
          f'(causal legs in span: {len(in_span):,})')
    print(f'Final causal is_pivot=1 rows: {int((truth.is_pivot==1).sum()):,} '
          f'(was offline {n_offline_piv:,})')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(out_path, index=False)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
