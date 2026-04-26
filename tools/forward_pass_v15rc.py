"""
forward_pass_v15rc.py -- Forward pass for v1.5-RC vs v1.3-RC vs v1.0 baseline.
==============================================================================

Computes daily P&L on the same 1m zigzag trade list under four scenarios:

  A) BASELINE          = v1.0 / v1.1-RC.  No trail, no filter. Each trade exits
                         on the opposite pivot confirmation.
  B) TRAIL only        = v1.3-RC.  Trail stop max(TrailDist, TrailPct * HWM).
                         If HWM_pts >= TrailActivate, trail armed; trail exits if
                         leg retraces from HWM by more than eff_dist.
  C) FILTER only       = cascade-context filter without trail.
                         SKIP rules:
                           * regime AGAINST + envelope extreme (|1h_z|>2)
                           * regime WITH    + envelope inside  (|1h_z|<0.5)
  D) TRAIL + FILTER    = v1.5-RC.  Both layers.

Approximation note: trail exit P&L is computed analytically from per-leg MFE,
not bar-by-bar. NT8 actually runs trail on each 1m close. With Calculate.OnBarClose,
the analytic and bar-by-bar values differ at most by one 1m bar's worth of
movement (typically <$5/trade). Good enough to size the policy decision.

Inputs:
  --csv  = output of regime_envelope_quality.py
            (regime_envelope_R1m30_R1h{N}.csv -- has leg_pts/mfe_pts/regime/band).

MNQ: $2/pt, $1/side commission ($2 round-trip per contract).

Usage:
    python tools/forward_pass_v15rc.py --csv reports/findings/regime_envelope_R1m30_R1h75.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DOLLAR_PER_POINT = 2.0
COMMISSION_PER_SIDE = 1.0   # $1 entry + $1 exit = $2 round-trip per contract

# v1.3-RC trail defaults (match NT8 strategy file)
DEFAULT_TRAIL_ACTIVATE_PTS = 10.0
DEFAULT_TRAIL_DIST_PTS     = 5.0
DEFAULT_TRAIL_PCT          = 0.10


def is_2025(d): return d.startswith('2025_')
def is_2026(d): return d.startswith('2026_')


# ── Trail simulation ──────────────────────────────────────────────────────

def trail_pnl(mfe_pts: float, leg_pts: float,
              activate: float, dist: float, pct: float) -> tuple[float, str]:
    """Return (pnl_pts, exit_reason). pnl_pts is BEFORE commission.

    Logic mirrors NT8 v1.3-RC:
      - HWM profit per bar = max profit reached during the leg = mfe_pts.
      - Trail armed if mfe_pts >= activate.
      - eff_dist = max(dist, pct * mfe_pts).
      - Exit if leg retraced from MFE by more than eff_dist.
        retracement_from_MFE = mfe_pts - leg_pts.
      - If trail fires, exit P&L = mfe_pts - eff_dist.
      - Otherwise, exit at opposite pivot (= leg_pts).
    """
    if mfe_pts < activate or mfe_pts <= 0:
        return float(leg_pts), 'natural'

    eff_dist = max(dist, pct * mfe_pts)
    retracement = mfe_pts - leg_pts

    if retracement > eff_dist:
        # Trail fired before opposite pivot. Exit at MFE - eff_dist.
        return float(mfe_pts - eff_dist), 'trail'

    # Trail held; natural opposite-pivot exit.
    return float(leg_pts), 'natural'


# ── Filter ────────────────────────────────────────────────────────────────

def filter_keep(row) -> bool:
    """Skip the 2 worst buckets identified in 2026-04-25 cascade analysis."""
    r = row['regime']
    if r == 0:
        return True   # no macro regime -> trade as-is
    with_reg = (r == row['direction'])
    if (not with_reg) and row['band'] == 'extreme':
        return False  # AGAINST + extreme  (the fakeout-at-exhaustion trap)
    if with_reg and row['band'] == 'inside':
        return False  # WITH + inside     (noise-fakeout)
    return True


# ── Scenario simulators ───────────────────────────────────────────────────

def simulate(df: pd.DataFrame, scenario: str,
             activate=DEFAULT_TRAIL_ACTIVATE_PTS,
             dist=DEFAULT_TRAIL_DIST_PTS,
             pct=DEFAULT_TRAIL_PCT) -> pd.DataFrame:
    """Compute pnl_usd per row under the given scenario."""
    df = df.copy()

    # Filter (drop excluded rows by setting size=0)
    if scenario in ('FILTER', 'TRAIL+FILTER'):
        df['kept'] = df.apply(filter_keep, axis=1)
    else:
        df['kept'] = True

    # Trail vs natural per leg
    if scenario in ('TRAIL', 'TRAIL+FILTER'):
        out = df.apply(lambda row: trail_pnl(row['mfe_pts'], row['leg_pts'],
                                              activate, dist, pct), axis=1)
        df['exit_pts'] = [t[0] for t in out]
        df['exit_reason'] = [t[1] for t in out]
    else:
        df['exit_pts'] = df['leg_pts']
        df['exit_reason'] = 'natural'

    # P&L per trade (USD); commission only when traded.
    gross = df['exit_pts'] * DOLLAR_PER_POINT
    comm  = 2.0 * COMMISSION_PER_SIDE   # round-trip per contract
    df['pnl_usd'] = np.where(df['kept'], gross - comm, 0.0)
    return df


def daily_summary(df: pd.DataFrame, label: str) -> dict:
    daily = df.groupby('day')['pnl_usd'].sum()
    is_d  = daily[daily.index.map(is_2025)]
    oos_d = daily[daily.index.map(is_2026)]
    n_traded = int((df['kept'] & (df['exit_reason'] != '')).sum())
    n_skipped = int((~df['kept']).sum())
    n_trail = int(((df['exit_reason'] == 'trail') & df['kept']).sum())
    n_natural = int(((df['exit_reason'] == 'natural') & df['kept']).sum())

    def stats(s):
        if len(s) == 0:
            return dict(per_day=0, day_wr=0, n=0, worst=0, best=0, std=0, total=0)
        arr = s.values
        return dict(
            per_day=float(arr.mean()),
            day_wr=float(100.0 * (arr > 0).mean()),
            n=len(arr),
            worst=float(arr.min()),
            best=float(arr.max()),
            std=float(arr.std()),
            total=float(arr.sum()),
        )

    return dict(
        label=label, n_traded=n_traded, n_skipped=n_skipped,
        n_trail=n_trail, n_natural=n_natural,
        IS=stats(is_d), OOS=stats(oos_d),
    )


def print_summary(rows: list[dict]):
    print(f'\n{"Scenario":<22} {"Traded":>7} {"Skip":>5} {"Trail":>6} | '
          f'{"IS $/day":>10} {"IS dWR":>7} {"IS worst":>9} {"IS std":>8} | '
          f'{"OOS $/day":>10} {"OOS dWR":>7} {"OOS worst":>10} {"OOS std":>8}')
    print('-' * 142)
    for r in rows:
        print(f'{r["label"]:<22} {r["n_traded"]:>7,} {r["n_skipped"]:>5,} {r["n_trail"]:>6,} | '
              f'${r["IS"]["per_day"]:>+8.2f} {r["IS"]["day_wr"]:>6.1f}% '
              f'${r["IS"]["worst"]:>+7.0f} ${r["IS"]["std"]:>+6.0f} | '
              f'${r["OOS"]["per_day"]:>+8.2f} {r["OOS"]["day_wr"]:>6.1f}% '
              f'${r["OOS"]["worst"]:>+8.0f} ${r["OOS"]["std"]:>+6.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True,
                    help='Per-pivot CSV from regime_envelope_quality.py')
    ap.add_argument('--trail-activate', type=float, default=DEFAULT_TRAIL_ACTIVATE_PTS)
    ap.add_argument('--trail-dist', type=float, default=DEFAULT_TRAIL_DIST_PTS)
    ap.add_argument('--trail-pct', type=float, default=DEFAULT_TRAIL_PCT)
    args = ap.parse_args()

    df_in = pd.read_csv(args.csv)
    if len(df_in) == 0:
        print('Empty input.')
        return

    print('=' * 142)
    print(f'V1.5-RC FORWARD PASS  --  source: {os.path.basename(args.csv)}')
    print(f'Pivots: {len(df_in):,}  '
          f'(IS {int(df_in["day"].apply(is_2025).sum()):,}, '
          f'OOS {int(df_in["day"].apply(is_2026).sum()):,})')
    print(f'Trail: activate={args.trail_activate}pt, dist={args.trail_dist}pt, '
          f'pct={args.trail_pct}.   MNQ $/pt={DOLLAR_PER_POINT}, comm ${COMMISSION_PER_SIDE}/side')
    print('=' * 142)

    summaries = []
    for scenario in ('BASELINE', 'TRAIL', 'FILTER', 'TRAIL+FILTER'):
        sim = simulate(df_in, scenario,
                       args.trail_activate, args.trail_dist, args.trail_pct)
        label = {
            'BASELINE':     'A) Baseline (v1.0)',
            'TRAIL':        'B) Trail only (v1.3-RC)',
            'FILTER':       'C) Filter only',
            'TRAIL+FILTER': 'D) Trail+Filter (v1.5-RC)',
        }[scenario]
        summaries.append(daily_summary(sim, label))

    print_summary(summaries)

    # Deltas vs baseline
    print('\nDeltas vs Baseline:')
    base = summaries[0]
    for r in summaries[1:]:
        d_is = r['IS']['per_day'] - base['IS']['per_day']
        d_oos = r['OOS']['per_day'] - base['OOS']['per_day']
        print(f'  {r["label"]:<22}  IS  d={d_is:>+7.2f}/day  |  OOS d={d_oos:>+7.2f}/day')

    # Key v1.5-RC vs v1.3-RC delta (= incremental cascade benefit ON TOP of trail)
    print('\nKey delta — v1.5-RC vs v1.3-RC (= cascade benefit on top of trail):')
    v13 = summaries[1]
    v15 = summaries[3]
    print(f'  IS:  v1.3-RC ${v13["IS"]["per_day"]:+7.2f}/day  ->  '
          f'v1.5-RC ${v15["IS"]["per_day"]:+7.2f}/day  (d {v15["IS"]["per_day"]-v13["IS"]["per_day"]:+6.2f})')
    print(f'  OOS: v1.3-RC ${v13["OOS"]["per_day"]:+7.2f}/day  ->  '
          f'v1.5-RC ${v15["OOS"]["per_day"]:+7.2f}/day  (d {v15["OOS"]["per_day"]-v13["OOS"]["per_day"]:+6.2f})')

    # Save
    out_csv = 'reports/findings/forward_pass_v15rc.csv'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for r in summaries:
        rows.append({
            'scenario': r['label'],
            'n_traded': r['n_traded'], 'n_skipped': r['n_skipped'],
            'n_trail_exits': r['n_trail'], 'n_natural_exits': r['n_natural'],
            'IS_per_day':  r['IS']['per_day'],  'IS_total':  r['IS']['total'],
            'IS_day_wr':   r['IS']['day_wr'],   'IS_n_days': r['IS']['n'],
            'IS_worst':    r['IS']['worst'],    'IS_best':   r['IS']['best'],
            'OOS_per_day': r['OOS']['per_day'], 'OOS_total': r['OOS']['total'],
            'OOS_day_wr':  r['OOS']['day_wr'],  'OOS_n_days':r['OOS']['n'],
            'OOS_worst':   r['OOS']['worst'],   'OOS_best':  r['OOS']['best'],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')


if __name__ == '__main__':
    main()
