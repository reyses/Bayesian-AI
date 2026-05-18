"""Composite Sizing Simulator — apply data-driven position size to each leg
based on entry-time composite signal.

Per composite_entry_analyzer.py finding (NT8 OOS):
  Entry at AT_PIVOT zones -> $+121/leg, 2.2x CLEAR entries
  Entry with B6 P(match) >= 0.70 -> $+85/leg, 1.6x baseline
  Entry at CLEAR with weak B6 -> $54/leg (lowest)

Strategy: scale position size by entry-signal strength. Higher confidence
in the directional setup -> larger size. This amplifies the high-edge
entries and de-emphasizes the weak ones, without changing exit logic.

Sizing schemes tested:
  - 'flat': all legs 1.0x (baseline — what we've been computing)
  - 'zone': AT_PIVOT/IMMINENT 1.5x, NEAR_* 1.2x, WATCH 1.0x, CLEAR 0.8x
  - 'b6':   B6 match >=0.70 -> 1.5x; 0.50-0.70 -> 1.0x; <0.50 -> 0.7x
  - 'combo': zone multiplier × B6 multiplier
  - 'aggressive': larger multipliers, harder cutoff for weak entries

For each scheme, compute total P&L = sum(leg_pnl * size_multiplier),
and capital-efficiency = total_P&L / sum(size_multipliers).

The right metric for COMPARING is mean per-leg PnL (capital-weighted),
since real trading has finite capital. We want pnl_per_unit_capital
to RISE under the variable-sizing schemes.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def get_size_multiplier(zone, b6_match, scheme):
    """Return position size multiplier given entry signal + scheme."""
    if scheme == 'flat':
        return 1.0

    if scheme == 'zone':
        if zone in ('AT_PIVOT', 'IMMINENT'):
            return 1.5
        elif zone in ('NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m', 'WIDE_ZONE'):
            return 1.2
        elif zone == 'WATCH':
            return 1.0
        else:  # CLEAR
            return 0.8

    if scheme == 'b6':
        if b6_match >= 0.70:
            return 1.5
        elif b6_match >= 0.50:
            return 1.0
        else:
            return 0.7

    if scheme == 'combo':
        # zone × b6 multiplier (geometric mean approximated by product/2 + 0.5)
        z = get_size_multiplier(zone, b6_match, 'zone')
        b = get_size_multiplier(zone, b6_match, 'b6')
        return (z + b) / 2

    if scheme == 'aggressive':
        # Strong size on AT_PIVOT or B6 >= 0.70; skip CLEAR + B6 < 0.50
        if zone == 'AT_PIVOT' or b6_match >= 0.70:
            return 2.0
        elif zone in ('IMMINENT', 'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m'):
            return 1.2
        elif zone == 'CLEAR' and b6_match < 0.50:
            return 0.0   # SKIP
        else:
            return 0.8

    raise ValueError(f'Unknown scheme: {scheme}')


def evaluate_scheme(df, scheme):
    """Apply size multipliers, compute aggregate metrics."""
    sizes = df.apply(lambda r: get_size_multiplier(r['entry_zone'],
                                                    r['entry_p_b6_match'],
                                                    scheme),
                      axis=1).values
    pnl = df['pnl_at_R_usd'].values
    weighted_pnl = pnl * sizes
    total_pnl = weighted_pnl.sum()
    total_size = sizes.sum()
    n_taken = int((sizes > 0).sum())
    n_skipped = int((sizes == 0).sum())
    # Mean per-leg actual = total pnl / n_legs (regardless of size)
    # Mean per-unit-capital = total pnl / total size
    return {
        'scheme': scheme,
        'n_legs_total': len(df),
        'n_taken': n_taken,
        'n_skipped': n_skipped,
        'total_pnl_usd': float(total_pnl),
        'total_capital_units': float(total_size),
        'pnl_per_leg': float(total_pnl / max(len(df), 1)),
        'pnl_per_unit_capital': float(total_pnl / max(total_size, 1e-9)),
        'mean_size': float(sizes.mean()),
        'per_day_total': float(total_pnl / df['day'].nunique()),
    }


def per_day_stats(df, scheme):
    sizes = df.apply(lambda r: get_size_multiplier(r['entry_zone'],
                                                    r['entry_p_b6_match'],
                                                    scheme),
                      axis=1).values
    pnl = df['pnl_at_R_usd'].values
    weighted = pnl * sizes
    df = df.copy()
    df['weighted_pnl'] = weighted
    df['size'] = sizes
    per_day = df.groupby('day').agg(
        sum_pnl=('weighted_pnl', 'sum'),
        sum_size=('size', 'sum'),
        n_legs=('weighted_pnl', 'count'),
    )
    per_day['pnl_per_unit'] = per_day['sum_pnl'] / per_day['sum_size'].clip(lower=1e-9)
    return per_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.csv')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_sizing_sim.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_sizing_sim.txt')
    args = ap.parse_args()

    df = pd.read_csv(args.legs)
    print(f'Loaded {len(df):,} legs from {args.legs}')

    schemes = ['flat', 'zone', 'b6', 'combo', 'aggressive']
    results = [evaluate_scheme(df, s) for s in schemes]
    rdf = pd.DataFrame(results)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('COMPOSITE SIZING SIMULATOR — variable position size by entry signal')
    out('=' * 78)
    out(f'Total legs: {len(df):,}   Days: {df["day"].nunique()}')
    out(f'Baseline (flat 1.0x) per-leg P&L: ${df["pnl_at_R_usd"].mean():+.2f}')
    out('')
    out(f'{"scheme":<12}  {"n_taken":>7}  {"n_skipped":>9}  '
        f'{"total_pnl":>12}  {"per_leg":>10}  {"per_unit":>10}  '
        f'{"mean_size":>9}  {"per_day":>9}')
    for _, r in rdf.iterrows():
        out(f'{r["scheme"]:<12}  {int(r["n_taken"]):>7}  {int(r["n_skipped"]):>9}  '
            f'${r["total_pnl_usd"]:>10,.0f}  ${r["pnl_per_leg"]:>+8.2f}  '
            f'${r["pnl_per_unit_capital"]:>+8.2f}  '
            f'{r["mean_size"]:>9.2f}  ${r["per_day_total"]:>+7.0f}')

    out('')
    out('Interpretation:')
    out('  per_leg     = total $ / total legs   (raw effect, easy to interpret)')
    out('  per_unit    = total $ / total capital units used (capital efficiency)')
    out('  per_day     = total $ / number of days (daily rate)')
    out('')
    out('The right metric depends on the constraint:')
    out('  - If capital is unconstrained:  maximize per_day (total $)')
    out('  - If capital is constrained:    maximize per_unit_capital')
    out('  - If selectivity matters:       balance per_leg with n_skipped')
    out('')

    # Per-day comparison
    out('--- PER-DAY P&L by scheme (32 days, bootstrap CI on mean) ---')
    for s in schemes:
        pd_per_day = per_day_stats(df, s)
        per_day_pnl = pd_per_day['sum_pnl'].values
        rng = np.random.default_rng(42)
        boots = np.array([per_day_pnl[rng.integers(0, len(per_day_pnl), len(per_day_pnl))].mean()
                           for _ in range(4000)])
        out(f'  {s:<12}  mean ${per_day_pnl.mean():+.2f}/day  '
            f'95% CI [${np.percentile(boots, 2.5):+.2f}, ${np.percentile(boots, 97.5):+.2f}]  '
            f'positive days {(per_day_pnl > 0).sum()}/{len(per_day_pnl)}  '
            f'median ${np.median(per_day_pnl):+.2f}')

    rdf.to_csv(args.out, index=False)
    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
