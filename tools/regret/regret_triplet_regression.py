"""3-feature regression — every continuous triplet (A, B, C).

Per user 2026-05-16: after the joint triplet clustering, fit four regression
models and report the R² gain at each step:

    additive    : y = β₀ + β_A·A + β_B·B + β_C·C
    + 2-way     : adds β_AB·AB + β_AC·AC + β_BC·BC
    + 3-way     : adds β_ABC·ABC
    quadratic   : (skipped at this level — too many params for sample size)

The marginal R² gains tell us how much non-linearity each level adds.

Output:
    triplet_regression_<name>.csv   one row per triplet with all R²s + coefs
"""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR_DEFAULT = Path('reports/findings/regret_oracle')

CONTINUOUS_FEATS = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'bar_range', 'volume',
    'tod_minutes',
]


def fit_r2(X, y, ss_tot):
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coefs
    ss_res = float(np.sum((y - y_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coefs, r2


def regress_triplet(df, fa, fb, fc):
    sub = df[[fa, fb, fc, 'mfe_dollars']].dropna()
    if len(sub) < 200:
        return None
    a = sub[fa].astype(float).values
    b = sub[fb].astype(float).values
    c = sub[fc].astype(float).values
    y = sub['mfe_dollars'].astype(float).values
    a_std = float(np.std(a)); b_std = float(np.std(b)); c_std = float(np.std(c))
    if min(a_std, b_std, c_std) == 0:
        return None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    one = np.ones(len(y))

    # Additive
    X_add = np.column_stack([one, a, b, c])
    co_add, r2_add = fit_r2(X_add, y, ss_tot)

    # + 2-way interactions
    X_2 = np.column_stack([one, a, b, c, a*b, a*c, b*c])
    co_2, r2_2 = fit_r2(X_2, y, ss_tot)

    # + 3-way interaction
    X_3 = np.column_stack([one, a, b, c, a*b, a*c, b*c, a*b*c])
    co_3, r2_3 = fit_r2(X_3, y, ss_tot)

    return {
        'feat_a': fa, 'feat_b': fb, 'feat_c': fc,
        'n':       int(len(sub)),
        'beta_a_per_1sigma_$':  round(float(co_add[1] * a_std), 2),
        'beta_b_per_1sigma_$':  round(float(co_add[2] * b_std), 2),
        'beta_c_per_1sigma_$':  round(float(co_add[3] * c_std), 2),
        'r2_additive':            round(float(r2_add), 4),
        'r2_with_2way':           round(float(r2_2),   4),
        'r2_with_3way':           round(float(r2_3),   4),
        'gain_from_2way':         round(float(r2_2 - r2_add), 4),
        'gain_from_3way':         round(float(r2_3 - r2_2),   4),
        # 3-way interaction coefficient — sign tells direction-amplification
        'beta_ABC':               round(float(co_3[7]), 6),
        'beta_ABC_per_sigma3_$':  round(float(co_3[7] * a_std * b_std * c_std), 3),
        'sign_ABC':               '+' if co_3[7] > 0 else '-',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--name', default='IS_full_daisy')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} trades')

    feats = [c for c in CONTINUOUS_FEATS if c in df.columns]
    triplets = list(combinations(feats, 3))
    print(f'Features: {len(feats)}    Triplets: {len(triplets)}')

    rows = []
    for i, (fa, fb, fc) in enumerate(triplets):
        r = regress_triplet(df, fa, fb, fc)
        if r:
            rows.append(r)
        if (i + 1) % 200 == 0:
            print(f'  ...processed {i+1}/{len(triplets)} triplets', flush=True)

    table = pd.DataFrame(rows)
    out_path = out_dir / f'triplet_regression_{args.name}.csv'
    table.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}    ({len(table)} triplets regressed)')

    if table.empty:
        return

    # Headline ranking: highest 3-way R²
    print('\n=== Top 20 triplets by FULL R² (additive + 2-way + 3-way) ===')
    tot_sort = table.sort_values('r2_with_3way', ascending=False)
    cols_full = ['feat_a', 'feat_b', 'feat_c',
                 'r2_additive', 'r2_with_2way', 'r2_with_3way',
                 'gain_from_2way', 'gain_from_3way']
    print(tot_sort[cols_full].head(20).to_string(index=False))

    # Which triplets gain the MOST from the 3-way interaction term?
    print('\n=== Top 20 triplets by 3-way interaction gain ===')
    three_sort = table.sort_values('gain_from_3way', ascending=False)
    cols_3 = ['feat_a', 'feat_b', 'feat_c', 'r2_with_2way', 'r2_with_3way',
              'gain_from_3way', 'beta_ABC_per_sigma3_$', 'sign_ABC']
    print(three_sort[cols_3].head(20).to_string(index=False))

    # Which triplets gain the most from adding 2-way interactions (beyond additive)?
    print('\n=== Top 20 triplets by 2-way interaction gain ===')
    two_sort = table.sort_values('gain_from_2way', ascending=False)
    cols_2 = ['feat_a', 'feat_b', 'feat_c', 'r2_additive', 'r2_with_2way',
              'gain_from_2way']
    print(two_sort[cols_2].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
