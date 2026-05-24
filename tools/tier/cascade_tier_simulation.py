"""
cascade_tier_simulation.py -- Apply cascade-decile tier sizing to the trade ledger.
====================================================================================

Loads the per-pivot CSV from `cascade_pivot_quality.py` and simulates several
sizing schemes:
  A)  Baseline             — all trades 1x (= pure zigzag at the chosen R).
  B)  Overweight top       — top decile (9) gets 2x, all else 1x.
  C)  Filter bottom        — deciles 0-1 SKIP, all else 1x.
  D)  Filter + overweight  — deciles 0-1 SKIP, decile 9 = 2x, rest 1x.
  E)  Custom ladder        — full tier ladder (configurable in code).

Computes per-day P&L with and without each scheme, IS vs OOS, accounting for
commission (commission scales with contract count).

Usage:
    python tools/cascade_tier_simulation.py --csv reports/findings/cascade_pivot_quality_R30_z1.5.csv
    python tools/cascade_tier_simulation.py --csv reports/findings/cascade_pivot_quality_R10_z1.5.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DOLLAR_PER_POINT = 2.0     # MNQ
COMMISSION_PER_SIDE = 1.0  # $1/side per contract


def is_2025(d): return d.startswith('2025_')
def is_2026(d): return d.startswith('2026_')


def daily_summary(daily_pnl: pd.Series, label: str) -> dict:
    if len(daily_pnl) == 0:
        return dict(label=label, n_days=0, total=0, per_day=0, day_wr=0,
                    best=0, worst=0, sharpe=0)
    arr = daily_pnl.values
    return dict(
        label=label,
        n_days=len(arr),
        total=float(arr.sum()),
        per_day=float(arr.mean()),
        day_wr=float(100.0 * (arr > 0).mean()),
        best=float(arr.max()),
        worst=float(arr.min()),
        sharpe=float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252)),
    )


def simulate_scheme(df: pd.DataFrame, sizing_fn, label: str) -> dict:
    """Apply size_mult per pivot, compute daily P&L (incl. commission), summarize."""
    sizes = df['decile'].apply(sizing_fn).values  # contracts per trade

    # Per-trade P&L = leg_pts * $/pt * size - commission * 2 sides * size
    leg_pnl = df['leg_pts'].values * DOLLAR_PER_POINT * sizes
    commission = 2.0 * COMMISSION_PER_SIDE * sizes
    pnl_usd = leg_pnl - commission
    pnl_usd[sizes == 0] = 0.0   # skipped trades contribute zero

    df_local = df.copy()
    df_local['size'] = sizes
    df_local['pnl_usd'] = pnl_usd

    n_skipped = int((sizes == 0).sum())
    n_traded = int((sizes > 0).sum())

    daily_total = df_local.groupby('day')['pnl_usd'].sum()

    is_summary = daily_summary(daily_total[daily_total.index.map(is_2025)], 'IS')
    oos_summary = daily_summary(daily_total[daily_total.index.map(is_2026)], 'OOS')

    return {
        'label': label,
        'n_traded': n_traded,
        'n_skipped': n_skipped,
        'IS': is_summary,
        'OOS': oos_summary,
    }


# Sizing schemes — return contracts per trade given the trade's decile.

def scheme_A_baseline(d):     return 1
def scheme_B_overweight(d):   return 2 if d == 9 else 1
def scheme_C_filter(d):       return 0 if d in (0, 1) else 1
def scheme_D_combo(d):        return 0 if d in (0, 1) else (2 if d == 9 else 1)
def scheme_E_ladder(d):
    # Aggressive ladder: skip worst, double best, partial sizes in middle.
    if d in (0, 1): return 0
    if d in (2, 3): return 1
    if d in (4, 5, 6): return 1
    if d in (7, 8): return 1
    if d == 9:      return 2
    return 1


SCHEMES = [
    ('A) Baseline (all 1x)',                scheme_A_baseline),
    ('B) Top-decile 2x',                    scheme_B_overweight),
    ('C) Skip bottom-2 deciles',            scheme_C_filter),
    ('D) Skip bottom-2 + Top-decile 2x',    scheme_D_combo),
    ('E) Custom ladder',                    scheme_E_ladder),
]


def print_row(label, traded, skipped, summ_is, summ_oos):
    print(f'{label:<38} {traded:>6,} {skipped:>6,} | '
          f'${summ_is["per_day"]:>+7.2f} {summ_is["day_wr"]:>5.1f}%  '
          f'${summ_is["worst"]:>+7.0f} | '
          f'${summ_oos["per_day"]:>+7.2f} {summ_oos["day_wr"]:>5.1f}%  '
          f'${summ_oos["worst"]:>+7.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True,
                    help='Per-pivot CSV from cascade_pivot_quality.py')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        print('Empty input.')
        return

    # Quantile-bucket dir_energy into deciles. duplicates='drop' tolerates ties.
    df['decile'] = pd.qcut(df['dir_energy'], 10, labels=False, duplicates='drop')
    df = df.dropna(subset=['decile'])
    df['decile'] = df['decile'].astype(int)

    n_total = len(df)
    n_is = int(df['day'].apply(is_2025).sum())
    n_oos = int(df['day'].apply(is_2026).sum())

    print('=' * 110)
    print(f'CASCADE TIER SIZING SIMULATION — {os.path.basename(args.csv)}')
    print(f'Pivots: {n_total:,}  (IS={n_is:,}, OOS={n_oos:,})')
    print(f'MNQ: $/pt={DOLLAR_PER_POINT}, commission ${COMMISSION_PER_SIDE}/side per contract')
    print('=' * 110)

    print(f'{"Scheme":<38} {"Trade":>6} {"Skip":>6} | '
          f'{"IS $/day":>10} {"IS dWR":>6}  {"IS worst":>8} | '
          f'{"OOS $/day":>10} {"OOS dWR":>6}  {"OOS worst":>9}')
    print('-' * 110)

    results = []
    for label, fn in SCHEMES:
        r = simulate_scheme(df, fn, label)
        results.append(r)
        print_row(label, r['n_traded'], r['n_skipped'], r['IS'], r['OOS'])

    # Show absolute deltas vs baseline for clarity
    print('-' * 110)
    base_is = results[0]['IS']['per_day']
    base_oos = results[0]['OOS']['per_day']
    print(f'\nDelta vs Baseline ($/day):')
    for r in results[1:]:
        delta_is = r['IS']['per_day'] - base_is
        delta_oos = r['OOS']['per_day'] - base_oos
        print(f'  {r["label"]:<38}  IS d/day {delta_is:>+7.2f}  |  OOS d/day {delta_oos:>+7.2f}')


if __name__ == '__main__':
    main()
