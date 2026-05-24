"""Composite OOS Forward Pass — realistic R-trigger entries + B7 sizing.

All previous sims assumed ORACLE entry timing: we enter exactly at the
pivot (knowing it's a pivot). In live trading we don't — the indicator
only CONFIRMS a pivot after price reverses R from the extreme. So:

  Entry price  = pivot_price + R    (for LONG leg, pivot is a LOW)
                or pivot_price - R   (for SHORT leg, pivot is a HIGH)
  Exit price   = next_pivot_price - R   (for LONG, exit on next R-trigger)
                or next_pivot_price + R  (for SHORT)
  Per-leg P&L  = leg_amplitude - 2*R - friction

Plus B7 sizing on top. Compare flat vs hand_aggressive vs gbm_ev.

Friction assumptions (project standard from CLAUDE.md context):
  - Round-trip commission: $4/contract
  - Slippage: $2/contract round-trip (1 tick each side)
  - Total per-leg friction: $6/contract

This is the HONEST per-leg cost. NOTE: B7 was trained on features at
PIVOT bar time; here it's being applied to legs with R-trigger entries.
The sizing decision still uses pivot-time features (a small "peek")
but the P&L is realistic. For a fully-honest version, B7 would need
retraining on R-trigger-time features — flagged as caveat.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DOLLAR_PER_POINT = 2.0
COMMISSION_PER_LEG = 4.00    # $4 round-trip
SLIPPAGE_PER_LEG   = 2.00    # 1 tick each side = $1.00 per side × 2
FRICTION_PER_LEG   = COMMISSION_PER_LEG + SLIPPAGE_PER_LEG   # $6


def gbm_ev(pred_R):
    return float(np.clip(max(pred_R - 1.0, 0.0), 0.0, 3.0))


def gbm_quantile(pred_R, percentile_rank):
    if percentile_rank >= 0.80: return 2.0
    if percentile_rank >= 0.50: return 1.5
    if percentile_rank >= 0.20: return 0.8
    return 0.5


def hand_aggressive(zone, b6_match):
    if zone == 'AT_PIVOT' or b6_match >= 0.70: return 2.0
    elif zone in ('IMMINENT', 'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m'): return 1.2
    elif zone == 'CLEAR' and b6_match < 0.50: return 0.0
    else: return 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b7-cache',
                    default='reports/findings/regret_oracle/b7_leg_sizer_OOS.parquet')
    ap.add_argument('--entries',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.csv')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_forward_pass.txt')
    ap.add_argument('--commission', type=float, default=COMMISSION_PER_LEG)
    ap.add_argument('--slippage', type=float, default=SLIPPAGE_PER_LEG)
    args = ap.parse_args()

    print('Loading inputs...')
    b7 = pd.read_parquet(args.b7_cache)
    entries = pd.read_csv(args.entries)
    merged = b7.merge(
        entries[['day', 'entry_ts', 'entry_zone', 'entry_p_b6_match']],
        on=['day', 'entry_ts'], how='inner',
    )
    print(f'  {len(merged):,} legs across {merged["day"].nunique()} days')

    friction = args.commission + args.slippage

    # Realistic per-leg P&L: leg_amplitude - 2*R - friction
    # (give back R at entry from R-trigger lag, R at exit from R-trigger lag)
    realistic_pnl = (merged['leg_amp_usd'].values
                       - 2 * merged['r_price'].values * DOLLAR_PER_POINT
                       - friction)
    merged['realistic_pnl_usd'] = realistic_pnl

    # Skip legs where R-trigger CAN'T complete (i.e., leg_amplitude < 2R + friction)
    # These would be losing trades under R-trigger even at oracle exit
    # No actual skip — these are legitimate losses

    n_legs = len(merged)
    pos_legs = int((realistic_pnl > 0).sum())
    print(f'  Realistic per-leg P&L: mean ${realistic_pnl.mean():+.2f}   '
          f'median ${np.median(realistic_pnl):+.2f}   pos {pos_legs}/{n_legs} ({pos_legs/n_legs*100:.1f}%)')

    # Sizing schemes
    pred = merged['pred_amp_R'].values
    pred_rank = pd.Series(pred).rank(pct=True).values

    schemes = {
        'flat':            np.ones(n_legs),
        'gbm_ev':          np.array([gbm_ev(p) for p in pred]),
        'gbm_quantile':    np.array([gbm_quantile(p, r) for p, r in zip(pred, pred_rank)]),
        'hand_aggressive': np.array([hand_aggressive(z, b)
                                       for z, b in zip(merged['entry_zone'].values,
                                                        merged['entry_p_b6_match'].values)]),
    }

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('OOS FORWARD PASS — realistic R-trigger entries + B7 sizing')
    out('=' * 78)
    out(f'Days: {merged["day"].nunique()}   Legs: {n_legs:,}')
    out(f'Friction per leg: ${friction:.2f}  (commission ${args.commission}, '
        f'slippage ${args.slippage})')
    out(f'Realistic P&L = leg_amplitude - 2*R - friction')
    out('')
    out(f'  Mean leg amp:       ${merged["leg_amp_usd"].mean():.2f}')
    out(f'  Mean 2*R cost:      ${(2 * merged["r_price"].mean() * DOLLAR_PER_POINT):.2f}')
    out(f'  Mean per-leg P&L:   ${realistic_pnl.mean():+.2f}')
    out(f'  Median per-leg P&L: ${np.median(realistic_pnl):+.2f}')
    out(f'  Positive legs:      {pos_legs}/{n_legs} ({pos_legs/n_legs*100:.1f}%)')
    out('')

    out(f'{"scheme":<18}  {"n_taken":>7}  {"n_skip":>7}  '
        f'{"total_$":>10}  {"per_leg":>9}  {"per_unit":>9}  '
        f'{"mean_size":>9}  {"$/day":>9}')

    rng = np.random.default_rng(42)
    per_day_results = {}
    for name, sizes in schemes.items():
        weighted = realistic_pnl * sizes
        total_pnl = float(weighted.sum())
        total_size = float(sizes.sum())
        n_taken = int((sizes > 0).sum())
        n_skipped = int((sizes == 0).sum())
        per_day = total_pnl / merged['day'].nunique()
        per_leg = total_pnl / max(n_legs, 1)
        per_unit = total_pnl / max(total_size, 1e-9)
        out(f'{name:<18}  {n_taken:>7}  {n_skipped:>7}  '
            f'${total_pnl:>9,.0f}  ${per_leg:>+7.2f}  '
            f'${per_unit:>+7.2f}  {sizes.mean():>9.3f}  '
            f'${per_day:>+7.0f}')

        # Per-day series for CI calc
        merged_copy = merged.copy()
        merged_copy['wpnl'] = weighted
        per_day_series = merged_copy.groupby('day')['wpnl'].sum().values
        per_day_results[name] = per_day_series

    out('')
    out('--- Per-day P&L (32 days, bootstrap CI 4000 resamples) ---')
    for name, per_day in per_day_results.items():
        boots = np.array([per_day[rng.integers(0, len(per_day), len(per_day))].mean()
                           for _ in range(4000)])
        out(f'  {name:<18}  mean ${per_day.mean():+.2f}/day  '
            f'95% CI [${np.percentile(boots, 2.5):+.2f}, ${np.percentile(boots, 97.5):+.2f}]  '
            f'pos days {(per_day > 0).sum()}/{len(per_day)}  '
            f'median ${np.median(per_day):+.2f}')

    # Paired-delta vs flat
    out('')
    out('--- Paired delta vs flat (realistic R-trigger baseline) ---')
    base = per_day_results['flat']
    for name, per_day in per_day_results.items():
        if name == 'flat':
            continue
        delta = per_day - base
        boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                           for _ in range(4000)])
        out(f'  {name:<18}  delta ${delta.mean():+.2f}/day  '
            f'CI [${np.percentile(boots, 2.5):+.2f}, ${np.percentile(boots, 97.5):+.2f}]  '
            f'wins {(per_day > base).sum()}/{len(per_day)}')

    out('')
    out('--- IMPORTANT CAVEATS ---')
    out('  1. B7 sizing model was TRAINED on features at the actual pivot bar.')
    out('     In this forward pass it is applied to legs with R-trigger entries.')
    out('     The size DECISION uses pivot-time features (peek). For a fully')
    out('     honest live test, retrain B7 on R-trigger-time features.')
    out('  2. R-trigger detection is itself imperfect: in live trading, the')
    out('     pivot price + R threshold may be slipped, and the R-trigger')
    out('     bar arrives 5-30s after the actual pivot. We use 5s closes here.')
    out('  3. Friction assumed $6/leg (commission $4 + slippage $2). Real')
    out('     deployment may have higher slippage on stop-out entries.')
    out('  4. No partial fills, no overnight risk, no position limits modeled.')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'realistic_pnl_usd': realistic_pnl, **{f'size_{k}': v for k, v in schemes.items()}}).to_csv(args.out, index=False)
    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
