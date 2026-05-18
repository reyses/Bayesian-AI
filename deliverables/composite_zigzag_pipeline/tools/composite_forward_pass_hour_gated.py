"""Hour-gated forward pass — apply B8 hour-risk prediction as sizing modifier.

Combined sizing: total_size = b7_ev_size * hour_mult

Where hour_mult is derived from B8's predicted forward 60-min P&L:
  pred_hour_pnl < THR_LOW   -> hour_mult = LOW_MULT   (e.g., 0.3 — bad window)
  pred_hour_pnl in mid       -> hour_mult = 1.0       (normal)
  pred_hour_pnl > THR_HIGH  -> hour_mult = HIGH_MULT  (e.g., 1.5 — good window)

Or a simpler binary gate: skip / size_max if pred_hour_pnl < SKIP_THR.

Run sweep to find the best (THR_LOW, LOW_MULT) combination per the user's
target (32/32 days > $200).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm


def gbm_ev(p):
    return float(np.clip(max(p - 1.0, 0.0), 0.0, 3.0))


def hour_mult_linear(pred_hour_pnl, low_thr, high_thr, low_mult, high_mult):
    """Piecewise-linear: pred < low_thr -> low_mult, pred > high_thr -> high_mult,
    interpolate in between."""
    if pred_hour_pnl <= low_thr:
        return low_mult
    if pred_hour_pnl >= high_thr:
        return high_mult
    # linear interp
    frac = (pred_hour_pnl - low_thr) / max(high_thr - low_thr, 1e-9)
    return low_mult + frac * (high_mult - low_mult)


def hour_mult_binary(pred_hour_pnl, skip_thr, default_mult=1.0):
    """Binary: skip if predicted is below threshold."""
    return 0.0 if pred_hour_pnl < skip_thr else default_mult


def run_config(legs, b8_lookup, config):
    """Apply sizing config across legs, return per-day P&L."""
    rows = []
    for day, day_legs in legs.groupby('day'):
        day_legs = day_legs.sort_values('entry_ts').reset_index(drop=True)
        day_pnl = 0.0
        for _, leg in day_legs.iterrows():
            # Find B8 prediction at or before entry_ts
            b8_ts, b8_pred = b8_lookup.get(day, (None, None))
            if b8_ts is None or b8_pred is None:
                pred_hr = 0.0
            else:
                idx = int(np.searchsorted(b8_ts, leg['entry_ts'], side='right') - 1)
                idx = max(0, min(idx, len(b8_pred) - 1))
                pred_hr = float(b8_pred[idx])

            # Compute hour multiplier
            if config['mode'] == 'linear':
                hour_mult = hour_mult_linear(
                    pred_hr,
                    config['low_thr'], config['high_thr'],
                    config['low_mult'], config['high_mult'])
            elif config['mode'] == 'binary':
                hour_mult = hour_mult_binary(
                    pred_hr, config['skip_thr'],
                    default_mult=config.get('default_mult', 1.0))
            elif config['mode'] == 'baseline':
                hour_mult = 1.0
            else:
                raise ValueError(config['mode'])

            # B7 leg sizing
            leg_size = gbm_ev(leg['pred_amp_R_hardened'])
            total_size = leg_size * hour_mult
            wpnl = leg['pnl_usd'] * total_size
            day_pnl += wpnl

        rows.append({'day': day, 'pnl': day_pnl})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--b8',
                    default='reports/findings/regret_oracle/b8_hour_risk_OOS.parquet')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass_hour_gated.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_forward_pass_hour_gated.txt')
    args = ap.parse_args()

    print('Loading inputs...')
    legs = pd.read_csv(args.legs)
    b8 = pd.read_parquet(args.b8)
    print(f'  legs: {len(legs):,}   b8: {len(b8):,}')

    # Build per-day B8 lookup
    b8_lookup = {}
    for day, sub in b8.groupby('day'):
        sub = sub.sort_values('timestamp').reset_index(drop=True)
        b8_lookup[day] = (sub['timestamp'].values.astype(np.int64),
                            sub['pred_hour_pnl_usd'].values)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    # Baseline
    print('Baseline (no hour gate)...')
    base_r = run_config(legs, b8_lookup, {'mode': 'baseline'})
    base = base_r['pnl'].values

    out('=' * 78)
    out('HOUR-GATED FORWARD PASS — B7 sizing × B8 hour multiplier')
    out('=' * 78)
    out(f'Legs: {len(legs):,}   Days: {len(base_r)}')
    out('')
    out(f'BASELINE (B7 sizing only, no hour gate):')
    out(f'  mean ${base.mean():+.0f}/day   median ${np.median(base):+.0f}   '
        f'min ${base.min():+.0f}   max ${base.max():+.0f}')
    out(f'  pos days {(base > 0).sum()}/{len(base)}   days >$200 {(base > 200).sum()}/{len(base)}')
    out('')

    # Sweep binary gates
    out('--- BINARY HOUR GATE (skip if predicted hour P&L < skip_thr) ---')
    out(f'  {"skip_thr":<10}  {"mean":>8}  {"median":>8}  {"min":>8}  '
        f'{"max":>8}  {"pos":>4}  {"#>200":>5}  {"total":>9}')
    for thr in [-200, -100, -50, 0, 50, 100, 200]:
        r = run_config(legs, b8_lookup, {'mode': 'binary', 'skip_thr': thr})
        p = r['pnl'].values
        out(f'  ${thr:>+6.0f}    ${p.mean():>+6.0f}  ${np.median(p):>+6.0f}  '
            f'${p.min():>+6.0f}  ${p.max():>+6.0f}  '
            f'{(p > 0).sum():>4}  {(p > 200).sum():>5}  ${p.sum():>+7.0f}')

    out('')
    out('--- LINEAR HOUR GATE (interp between low_mult and high_mult) ---')
    out(f'  {"low_thr":<8}  {"high_thr":<8}  {"low_mult":<8}  {"high_mult":<8}  '
        f'{"mean":>8}  {"min":>8}  {"pos":>4}  {"#>200":>5}  {"total":>9}')
    sweep = []
    for low_thr in [-200, -100, 0]:
        for high_thr in [100, 200, 300]:
            for low_mult in [0.0, 0.3, 0.5, 0.7]:
                for high_mult in [1.0, 1.5, 2.0]:
                    cfg = dict(mode='linear', low_thr=low_thr, high_thr=high_thr,
                                low_mult=low_mult, high_mult=high_mult)
                    r = run_config(legs, b8_lookup, cfg)
                    p = r['pnl'].values
                    sweep.append({
                        **cfg,
                        'mean': float(p.mean()),
                        'median': float(np.median(p)),
                        'min': float(p.min()),
                        'pos_days': int((p > 0).sum()),
                        'days_over_200': int((p > 200).sum()),
                        'total': float(p.sum()),
                    })
    sdf = pd.DataFrame(sweep)
    # Sort by days_over_200 then mean
    sdf = sdf.sort_values(['days_over_200', 'mean'], ascending=[False, False])
    sdf.to_csv(args.out, index=False)

    out('Top 10 linear configs (sorted by days_over_200 desc, then mean):')
    for _, r in sdf.head(10).iterrows():
        out(f'  ${r["low_thr"]:>+5.0f}    ${r["high_thr"]:>+5.0f}    '
            f'{r["low_mult"]:<8.2f}  {r["high_mult"]:<8.2f}  '
            f'${r["mean"]:>+6.0f}  ${r["min"]:>+6.0f}  '
            f'{int(r["pos_days"]):>4}  {int(r["days_over_200"]):>5}  ${r["total"]:>+7.0f}')

    # Highlight any 100% > $200 config
    out('')
    target = sdf[sdf['days_over_200'] == len(base)]
    if len(target) > 0:
        out(f'*** {len(target)} configs achieve ALL DAYS > $200 ***')
        best = target.sort_values('mean', ascending=False).iloc[0]
        out(f'   Best: low_thr=${best["low_thr"]:.0f} high_thr=${best["high_thr"]:.0f} '
            f'low_mult={best["low_mult"]:.2f} high_mult={best["high_mult"]:.2f}   '
            f'mean ${best["mean"]:+.0f}/day  total ${best["total"]:+.0f}')
    else:
        most = sdf.iloc[0]
        out(f'No config achieves 100% > $200.')
        out(f'   Best: {int(most["days_over_200"])}/{len(base)} days > $200   '
            f'(low_thr=${most["low_thr"]:.0f} high_thr=${most["high_thr"]:.0f} '
            f'low_mult={most["low_mult"]:.2f} high_mult={most["high_mult"]:.2f})')
        out(f'   min day ${most["min"]:+.0f}  mean ${most["mean"]:+.0f}/day')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')
    print(f'Wrote: {args.report}')


if __name__ == '__main__':
    main()
