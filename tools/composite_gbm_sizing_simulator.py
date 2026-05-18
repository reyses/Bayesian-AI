"""GBM-driven Sizing Simulator.

Uses B7's per-leg predicted amplitude (in R units) to determine position
size. Compares to flat 1.0× baseline AND the hand-coded "aggressive"
scheme from composite_sizing_simulator.py.

Sizing schemes tested:
  - flat            : 1.0× always (baseline)
  - gbm_linear      : size = pred_amp_R / median_pred_amp_R   (linear scale)
  - gbm_clipped     : size = clip(pred / median, 0.5, 2.0)    (bounded)
  - gbm_quantile    : bucket-based on prediction quantile rank
  - gbm_ev          : size proportional to predicted P&L: (pred_amp_R - 1) clipped to [0, 3]
  - hand_aggressive : the hand-coded scheme from earlier (composite + B6 entry signals)

Metrics:
  - total $ across all OOS legs
  - mean per-leg P&L (raw)
  - mean per-unit-capital P&L (capital efficiency)
  - per-day total + bootstrap CI
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gbm_linear(pred_R, median):
    return pred_R / max(median, 1e-9)


def gbm_clipped(pred_R, median, lo=0.5, hi=2.0):
    return float(np.clip(pred_R / max(median, 1e-9), lo, hi))


def gbm_quantile(pred_R, percentile_rank):
    """percentile_rank in [0, 1]. Top 20% -> 2.0x; top 50% -> 1.5x;
    bottom 50% -> 0.8x; bottom 20% -> 0.5x."""
    if percentile_rank >= 0.80:
        return 2.0
    if percentile_rank >= 0.50:
        return 1.5
    if percentile_rank >= 0.20:
        return 0.8
    return 0.5


def gbm_ev(pred_R):
    """Size = max(pred_amp_R - 1, 0) clipped to [0, 3]."""
    ev = max(pred_R - 1.0, 0.0)
    return float(np.clip(ev, 0.0, 3.0))


def hand_aggressive(zone, b6_match):
    if zone == 'AT_PIVOT' or b6_match >= 0.70:
        return 2.0
    elif zone in ('IMMINENT', 'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m'):
        return 1.2
    elif zone == 'CLEAR' and b6_match < 0.50:
        return 0.0
    else:
        return 0.8


def evaluate(df, sizes, scheme):
    pnl = df['pnl_at_R_usd'].values
    weighted = pnl * sizes
    total_pnl = float(weighted.sum())
    total_size = float(sizes.sum())
    n_taken = int((sizes > 0).sum())
    n_skipped = int((sizes == 0).sum())
    per_day = float(total_pnl / df['day'].nunique())
    per_leg_raw = float(total_pnl / max(len(df), 1))
    per_unit = float(total_pnl / max(total_size, 1e-9))
    return {
        'scheme': scheme,
        'n_taken': n_taken,
        'n_skipped': n_skipped,
        'total_pnl_usd': total_pnl,
        'total_capital_units': total_size,
        'pnl_per_leg': per_leg_raw,
        'pnl_per_unit_capital': per_unit,
        'mean_size': float(sizes.mean()),
        'per_day_total': per_day,
    }


def per_day_pnl(df, sizes):
    df = df.copy()
    df['wpnl'] = df['pnl_at_R_usd'].values * sizes
    return df.groupby('day')['wpnl'].sum().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b7-cache',
                    default='reports/findings/regret_oracle/b7_leg_sizer_OOS.parquet')
    ap.add_argument('--entry-features',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.csv')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_gbm_sizing_sim.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_gbm_sizing_sim.txt')
    args = ap.parse_args()

    print('Loading inputs...')
    b7 = pd.read_parquet(args.b7_cache)
    entries = pd.read_csv(args.entry_features)
    print(f'  b7 cache:  {len(b7):,} legs')
    print(f'  entries:   {len(entries):,} legs')

    # Merge b7 predictions with entry composite signals
    merged = b7.merge(
        entries[['day', 'entry_ts', 'entry_zone', 'entry_p_b6_match']],
        on=['day', 'entry_ts'], how='inner',
    )
    print(f'  merged:    {len(merged):,} legs')

    pred = merged['pred_amp_R'].values
    pred_median = float(np.median(pred))
    pred_rank = pd.Series(pred).rank(pct=True).values

    schemes = {}
    schemes['flat'] = np.ones(len(merged))
    schemes['gbm_linear'] = np.array([gbm_linear(p, pred_median) for p in pred])
    schemes['gbm_clipped'] = np.array([gbm_clipped(p, pred_median) for p in pred])
    schemes['gbm_quantile'] = np.array([gbm_quantile(p, r) for p, r in zip(pred, pred_rank)])
    schemes['gbm_ev'] = np.array([gbm_ev(p) for p in pred])
    schemes['hand_aggressive'] = np.array([
        hand_aggressive(z, b) for z, b in zip(merged['entry_zone'].values,
                                                merged['entry_p_b6_match'].values)
    ])

    results = [evaluate(merged, sizes, s) for s, sizes in schemes.items()]
    rdf = pd.DataFrame(results)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('GBM SIZING SIMULATOR — B7 predicted amp_R drives size')
    out('=' * 78)
    out(f'Total legs: {len(merged):,}   Days: {merged["day"].nunique()}')
    out(f'Predicted amp_R: median {pred_median:.2f}   '
        f'mean {pred.mean():.2f}   min {pred.min():.2f}   max {pred.max():.2f}')
    out(f'Actual amp_R:    median {merged["leg_amp_R"].median():.2f}   '
        f'mean {merged["leg_amp_R"].mean():.2f}')
    out('')
    out(f'{"scheme":<18}  {"n_taken":>7}  {"n_skip":>7}  '
        f'{"total_$":>10}  {"per_leg":>9}  {"per_unit":>9}  '
        f'{"mean_size":>9}  {"$/day":>8}')
    for _, r in rdf.iterrows():
        out(f'{r["scheme"]:<18}  {int(r["n_taken"]):>7}  {int(r["n_skipped"]):>7}  '
            f'${r["total_pnl_usd"]:>9,.0f}  ${r["pnl_per_leg"]:>+7.2f}  '
            f'${r["pnl_per_unit_capital"]:>+7.2f}  '
            f'{r["mean_size"]:>9.3f}  ${r["per_day_total"]:>+7.0f}')

    out('')
    out('--- PER-DAY P&L with bootstrap CI (4000 resamples) ---')
    rng = np.random.default_rng(42)
    for s, sizes in schemes.items():
        per_day = per_day_pnl(merged, sizes)
        boots = np.array([per_day[rng.integers(0, len(per_day), len(per_day))].mean()
                           for _ in range(4000)])
        out(f'  {s:<18}  mean ${per_day.mean():+.2f}/day  '
            f'95% CI [${np.percentile(boots, 2.5):+.2f}, '
            f'${np.percentile(boots, 97.5):+.2f}]  '
            f'pos days {(per_day > 0).sum()}/{len(per_day)}  '
            f'median ${np.median(per_day):+.2f}')

    out('')
    out('--- VS hand_aggressive head-to-head (paired delta) ---')
    base_pd = per_day_pnl(merged, schemes['hand_aggressive'])
    for s, sizes in schemes.items():
        if s == 'hand_aggressive':
            continue
        s_pd = per_day_pnl(merged, sizes)
        delta = s_pd - base_pd
        boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                           for _ in range(4000)])
        out(f'  {s:<18}  vs hand_aggressive: mean delta ${delta.mean():+.2f}/day  '
            f'CI [${np.percentile(boots, 2.5):+.2f}, '
            f'${np.percentile(boots, 97.5):+.2f}]   '
            f'wins {(s_pd > base_pd).sum()}/{len(s_pd)} days')

    out('')
    out('Interpretation:')
    out('  - flat = baseline (no sizing)')
    out('  - gbm_linear/clipped/quantile/ev = data-driven sizing from B7 model')
    out('  - hand_aggressive = previous best hand-coded scheme (+15.6% vs flat)')
    out('')
    out('  If gbm beats hand_aggressive at p<0.05, ML adds genuine sizing edge.')
    out('  If similar, the hand-coded zones+B6 capture most of the available signal.')

    rdf.to_csv(args.out, index=False)
    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
