"""
Cord length from the REGRESSION-LINE perspective.

Companion to cord_length_1m.py (which measures price cord).

The regression line is smooth — its cord length represents the underlying
TREND signal amplitude, absent intra-bar noise. Comparing to price cord
tells us the noise-to-signal ratio:
    efficiency = regression_cord / price_cord
    - high (~1): price and regression move together, clean trends
    - low (<0.3): price oscillates heavily around slow-drifting mean

Regression cord is the theoretical ceiling for a strategy that filters
intra-bar noise (i.e., trades only when the regression line meaningfully
moves). It's a tighter upper bound than price cord.

Usage:
    python tools/cord_length_regression.py
    python tools/cord_length_regression.py --window 60

Output: reports/findings/cord_length_regression.md
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.cord_length_1m import zigzag_pivots, cord_stats


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/cord_length_regression.md'
DOLLAR_PER_POINT = 2.0
THRESHOLDS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]


def rolling_fit(closes, window):
    n = len(closes)
    fitted = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        x = np.arange(window, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            continue
        slope = (dx * (y - ym)).sum() / denom
        intercept = ym - slope * xm
        fitted[i] = intercept + slope * (window - 1)
    return fitted


def analyze_day(df, thresholds, window):
    closes = df['close'].values.astype(np.float64)
    fitted = rolling_fit(closes, window)
    # Drop leading NaN for zigzag
    valid = ~np.isnan(fitted)
    fit_clean = fitted[valid]

    n_bars = len(closes)
    active_hours = n_bars / 60.0
    out = {'_n_bars': n_bars, '_active_hours': active_hours}
    for R in thresholds:
        # Price cord
        p_pivots = zigzag_pivots(closes, R)
        p_stats = cord_stats(closes, p_pivots)
        # Regression cord (only over valid-fitted segment)
        r_pivots = zigzag_pivots(fit_clean, R) if len(fit_clean) > 1 else [0]
        r_stats = cord_stats(fit_clean, r_pivots)
        out[R] = {
            'price_legs': p_stats['n_legs'],
            'price_cord_dollars': p_stats['cord_dollars'],
            'reg_legs': r_stats['n_legs'],
            'reg_cord_dollars': r_stats['cord_dollars'],
            'efficiency': (r_stats['cord_dollars'] / p_stats['cord_dollars']
                           if p_stats['cord_dollars'] > 0 else 0),
        }
    return out


def aggregate(paths, thresholds, label, window):
    per_day = []
    for p in tqdm(paths, desc=label, unit='day'):
        df = pd.read_parquet(p)
        stats = analyze_day(df, thresholds, window)
        stats['day'] = os.path.basename(p).replace('.parquet', '')
        per_day.append(stats)
    return per_day


def summary(per_day, thresholds):
    n_days = len(per_day)
    out = {}
    for R in thresholds:
        p_sum = sum(d[R]['price_cord_dollars'] for d in per_day)
        r_sum = sum(d[R]['reg_cord_dollars'] for d in per_day)
        p_legs = sum(d[R]['price_legs'] for d in per_day)
        r_legs = sum(d[R]['reg_legs'] for d in per_day)
        p_per_day = np.array([d[R]['price_cord_dollars'] for d in per_day])
        r_per_day = np.array([d[R]['reg_cord_dollars'] for d in per_day])
        out[R] = {
            'price_mean': p_sum / n_days,
            'price_median': float(np.median(p_per_day)),
            'reg_mean': r_sum / n_days,
            'reg_median': float(np.median(r_per_day)),
            'price_legs_per_day': p_legs / n_days,
            'reg_legs_per_day': r_legs / n_days,
            'efficiency': r_sum / p_sum if p_sum > 0 else 0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=60,
                    help='Regression window in 1m bars (default 60 = 1h)')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS {len(is_paths)} | OOS {len(oos_paths)} days  window={args.window}')

    is_per_day = aggregate(is_paths, THRESHOLDS, 'IS', args.window)
    oos_per_day = aggregate(oos_paths, THRESHOLDS, 'OOS', args.window)

    is_agg = summary(is_per_day, THRESHOLDS)
    oos_agg = summary(oos_per_day, THRESHOLDS)

    print('\n=== IS aggregate ===')
    print(f'{"R":>6} {"Price $/day":>11} {"Price legs":>11} '
          f'{"Reg $/day":>10} {"Reg legs":>9} {"Efficiency":>10}')
    for R in THRESHOLDS:
        a = is_agg[R]
        label = 'all' if R == 0 else f'${R:.0f}'
        print(f'{label:>6} ${a["price_mean"]:>9,.0f} {a["price_legs_per_day"]:>10.0f} '
              f'${a["reg_mean"]:>8,.0f} {a["reg_legs_per_day"]:>8.0f} '
              f'{a["efficiency"]*100:>9.1f}%')

    print('\n=== OOS aggregate ===')
    print(f'{"R":>6} {"Price $/day":>11} {"Price legs":>11} '
          f'{"Reg $/day":>10} {"Reg legs":>9} {"Efficiency":>10}')
    for R in THRESHOLDS:
        a = oos_agg[R]
        label = 'all' if R == 0 else f'${R:.0f}'
        print(f'{label:>6} ${a["price_mean"]:>9,.0f} {a["price_legs_per_day"]:>10.0f} '
              f'${a["reg_mean"]:>8,.0f} {a["reg_legs_per_day"]:>8.0f} '
              f'{a["efficiency"]*100:>9.1f}%')

    # MD
    out = [f'# Cord length — regression-line perspective', '']
    out.append(f'Regression window: {args.window} 1m bars (= {args.window} min).')
    out.append('')
    out.append('**Price cord** = zigzag path on closes (noise-inclusive).')
    out.append('**Regression cord** = zigzag path on the smoothed regression-line '
               'fitted values.')
    out.append('**Efficiency** = regression_cord / price_cord. '
               'Higher = cleaner trend; lower = more noise.')
    out.append('')
    out.append('## IS aggregate')
    out.append('')
    out.append('| R | Price $/day | Price legs/day | Reg $/day | Reg legs/day | Efficiency |')
    out.append('|---:|---:|---:|---:|---:|---:|')
    for R in THRESHOLDS:
        a = is_agg[R]
        label = 'all bars' if R == 0 else f'${R:.0f}'
        out.append(f'| {label} | ${a["price_mean"]:,.0f} | '
                   f'{a["price_legs_per_day"]:.0f} | '
                   f'**${a["reg_mean"]:,.0f}** | '
                   f'{a["reg_legs_per_day"]:.0f} | '
                   f'{a["efficiency"]*100:.1f}% |')
    out.append('')
    out.append('## OOS aggregate')
    out.append('')
    out.append('| R | Price $/day | Price legs/day | Reg $/day | Reg legs/day | Efficiency |')
    out.append('|---:|---:|---:|---:|---:|---:|')
    for R in THRESHOLDS:
        a = oos_agg[R]
        label = 'all bars' if R == 0 else f'${R:.0f}'
        out.append(f'| {label} | ${a["price_mean"]:,.0f} | '
                   f'{a["price_legs_per_day"]:.0f} | '
                   f'**${a["reg_mean"]:,.0f}** | '
                   f'{a["reg_legs_per_day"]:.0f} | '
                   f'{a["efficiency"]*100:.1f}% |')
    out.append('')

    # Interpretation
    out.append('## Interpretation')
    out.append('')
    out.append('Regression cord = upper bound for a strategy that ONLY captures '
               'smooth trend moves (no intra-bar noise). Price cord − regression '
               'cord = the "unextractable" noise component.')
    out.append('')
    r15_is = is_agg[15.0]['reg_mean']
    r15_oos = oos_agg[15.0]['reg_mean']
    p15_is = is_agg[15.0]['price_mean']
    p15_oos = oos_agg[15.0]['price_mean']
    out.append(f'At R=$15: IS regression cord = ${r15_is:,.0f}/day '
               f'(vs price cord ${p15_is:,.0f}/day, '
               f'efficiency {r15_is/p15_is*100:.1f}%).')
    out.append(f'           OOS regression cord = ${r15_oos:,.0f}/day '
               f'(vs price cord ${p15_oos:,.0f}/day, '
               f'efficiency {r15_oos/p15_oos*100:.1f}%).')
    out.append('')
    out.append('Current NMP engine: +$311/day IS / +$67/day OOS.')
    if r15_is > 0:
        out.append(f'NMP captures {311/r15_is*100:.1f}% of IS regression cord.')
    if r15_oos > 0:
        out.append(f'NMP captures {67/r15_oos*100:.1f}% of OOS regression cord.')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
