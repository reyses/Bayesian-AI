"""Composite Entry Filter Sweep — find the optimal skip threshold.

Aggressive sizing scheme skipped 16% of legs at CLEAR + b6_match<0.50,
delivering +15.6% per-leg edge. Question: is this the right cutoff, or
can a different filter yield better selectivity?

Sweep filters across:
  - skip_zones      : sets of entry_zones to skip outright
  - b6_match_thr    : minimum B6 directional confidence to take entry
  - p_early_min     : minimum B5 P(EARLY) to take entry (we want fresh legs)

For each filter, report:
  - n_taken / n_skipped
  - mean P&L on kept legs
  - total P&L (kept × size = 1)
  - per-day P&L on kept legs
  - improvement vs baseline (no filter)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


def evaluate_filter(df, skip_zones, b6_thr, p_early_min):
    """Apply filter to legs. Returns dict of metrics."""
    skip_mask = (
        df['entry_zone'].isin(skip_zones) |
        (df['entry_p_b6_match'] < b6_thr) |
        (df['entry_p_early'] < p_early_min)
    )
    keep = df[~skip_mask]
    return {
        'skip_zones': '|'.join(sorted(skip_zones)) if skip_zones else 'none',
        'b6_thr': b6_thr,
        'p_early_min': p_early_min,
        'n_total': len(df),
        'n_kept': len(keep),
        'n_skipped': int(skip_mask.sum()),
        'keep_rate': len(keep) / max(len(df), 1),
        'mean_pnl_kept': float(keep['pnl_at_R_usd'].mean()) if len(keep) > 0 else float('nan'),
        'mean_pnl_skipped': float(df[skip_mask]['pnl_at_R_usd'].mean()) if skip_mask.any() else float('nan'),
        'total_pnl_kept': float(keep['pnl_at_R_usd'].sum()),
        'total_pnl_skipped': float(df[skip_mask]['pnl_at_R_usd'].sum()),
        'per_day_kept': float(keep['pnl_at_R_usd'].sum() / df['day'].nunique()),
        'pos_pct_kept': float((keep['pnl_at_R_usd'] > 0).mean()) if len(keep) > 0 else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.csv')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_entry_filter_sweep.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_entry_filter_sweep.txt')
    args = ap.parse_args()

    df = pd.read_csv(args.legs)
    print(f'Loaded {len(df):,} legs')
    baseline = df['pnl_at_R_usd'].mean()
    baseline_total = df['pnl_at_R_usd'].sum()
    baseline_per_day = baseline_total / df['day'].nunique()
    print(f'Baseline: ${baseline:+.2f}/leg, ${baseline_per_day:+.0f}/day total')

    # Filter combinations
    zone_skip_options = [
        set(),
        {'CLEAR'},
        {'CLEAR', 'WATCH'},
    ]
    b6_thresholds = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70]
    p_early_thresholds = [0.0, 0.20, 0.30, 0.40, 0.50]

    results = []
    for skip_zones, b6_thr, p_early in product(zone_skip_options, b6_thresholds,
                                                  p_early_thresholds):
        results.append(evaluate_filter(df, skip_zones, b6_thr, p_early))

    rdf = pd.DataFrame(results)
    # Compute "improvement over baseline" — per-leg and per-day
    rdf['delta_per_leg'] = rdf['mean_pnl_kept'] - baseline
    rdf['delta_per_day'] = rdf['per_day_kept'] - baseline_per_day
    rdf['pct_capture'] = rdf['total_pnl_kept'] / baseline_total
    # Sort by per-leg improvement
    rdf = rdf.sort_values('delta_per_leg', ascending=False)
    rdf.to_csv(args.out, index=False)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('COMPOSITE ENTRY FILTER SWEEP')
    out('=' * 78)
    out(f'Baseline (all legs taken): ${baseline:+.2f}/leg, ${baseline_per_day:+.0f}/day')
    out('')
    out('Top 20 filters by per-leg P&L improvement:')
    out(f'{"skip_zones":<16}  {"b6_thr":>7}  {"p_early":>7}  '
        f'{"n_kept":>7}  {"keep%":>6}  {"$/leg":>9}  '
        f'{"d/leg":>7}  {"$/day":>8}  {"d/day":>8}  {"%cap":>6}')
    for _, r in rdf.head(20).iterrows():
        out(f'{r["skip_zones"][:15]:<16}  {r["b6_thr"]:>7.2f}  {r["p_early_min"]:>7.2f}  '
            f'{int(r["n_kept"]):>7}  {r["keep_rate"]*100:>5.1f}%  '
            f'${r["mean_pnl_kept"]:>+7.2f}  '
            f'${r["delta_per_leg"]:>+5.2f}  '
            f'${r["per_day_kept"]:>+6.0f}  ${r["delta_per_day"]:>+6.0f}  '
            f'{r["pct_capture"]*100:>5.1f}%')

    out('')
    out('Top 5 by per-DAY improvement (preserves coverage):')
    by_day = rdf.sort_values('per_day_kept', ascending=False)
    out(f'{"skip_zones":<16}  {"b6_thr":>7}  {"p_early":>7}  '
        f'{"n_kept":>7}  {"keep%":>6}  {"$/leg":>9}  '
        f'{"$/day":>8}  {"d/day":>8}')
    for _, r in by_day.head(5).iterrows():
        out(f'{r["skip_zones"][:15]:<16}  {r["b6_thr"]:>7.2f}  {r["p_early_min"]:>7.2f}  '
            f'{int(r["n_kept"]):>7}  {r["keep_rate"]*100:>5.1f}%  '
            f'${r["mean_pnl_kept"]:>+7.2f}  '
            f'${r["per_day_kept"]:>+6.0f}  ${r["delta_per_day"]:>+6.0f}')

    # Sweet-spot: high per-leg + reasonable coverage
    out('')
    out('Sweet-spot filter recommendations:')
    out('  (high per-leg improvement WITHOUT cutting most of the day):')
    sweet = rdf[(rdf['keep_rate'] >= 0.50) & (rdf['delta_per_leg'] > 5)]
    sweet = sweet.sort_values('delta_per_leg', ascending=False).head(5)
    if len(sweet) == 0:
        out('  none found at keep_rate >= 0.50 and delta_per_leg > $5')
    else:
        for _, r in sweet.iterrows():
            out(f'  zones={r["skip_zones"]:<15}  b6>={r["b6_thr"]:.2f}  '
                f'early>={r["p_early_min"]:.2f}  '
                f'keep {int(r["n_kept"])}/{int(r["n_total"])} ({r["keep_rate"]*100:.0f}%)  '
                f'${r["mean_pnl_kept"]:+.2f}/leg  (+${r["delta_per_leg"]:.2f} vs base)  '
                f'${r["per_day_kept"]:+.0f}/day')

    out('')
    out('Note: these P&L numbers assume oracle-optimal entry timing (entering')
    out('at the pivot). Realistic deployment subtracts ~$32/leg for R-trigger')
    out('confirmation lag. The RELATIVE improvements should hold under that')
    out('adjustment.')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
