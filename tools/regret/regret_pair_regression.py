"""Paired 1D regression — every continuous × continuous pair.

Per user 2026-05-16: after the joint quantile clustering, run the matching
pair regressions. For each unordered pair (A, B):

    Additive    : y = β₀ + β_A·A + β_B·B
    Interaction : y = β₀ + β_A·A + β_B·B + β_AB·(A·B)

We report R² for each and the gain from adding the interaction term. A
non-trivial r2_gain means the two features INTERACT (their joint effect
is not just the sum of their individual effects). The sign of β_AB tells
the direction:
    β_AB > 0 = features amplify each other (high-A AND high-B → bigger trade)
    β_AB < 0 = features cancel (high in one, low in the other → bigger trade)

Output:
    pair_regression_<name>.csv   one row per pair
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


def fit_r2(X: np.ndarray, y: np.ndarray, ss_tot: float):
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coefs
    ss_res = float(np.sum((y - y_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coefs, r2


def regress_pair(df: pd.DataFrame, fa: str, fb: str) -> dict | None:
    sub = df[[fa, fb, 'mfe_dollars']].dropna()
    if len(sub) < 200:
        return None
    a = sub[fa].astype(float).values
    b = sub[fb].astype(float).values
    y = sub['mfe_dollars'].astype(float).values
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    one = np.ones(len(y))

    # Additive
    X_add = np.column_stack([one, a, b])
    coef_add, r2_add = fit_r2(X_add, y, ss_tot)

    # With interaction
    X_int = np.column_stack([one, a, b, a * b])
    coef_int, r2_int = fit_r2(X_int, y, ss_tot)

    return {
        'feat_a':                fa,
        'feat_b':                fb,
        'n':                     int(len(sub)),
        'beta_a_per_1sigma_$':   round(float(coef_add[1] * a_std), 2),
        'beta_b_per_1sigma_$':   round(float(coef_add[2] * b_std), 2),
        'r2_additive':           round(float(r2_add), 4),
        'r2_interaction':        round(float(r2_int), 4),
        'r2_gain':               round(float(r2_int - r2_add), 4),
        'beta_AB':               round(float(coef_int[3]), 5),
        'beta_AB_per_sigma2_$':  round(float(coef_int[3] * a_std * b_std), 2),
        'sign_AB':               '+' if coef_int[3] > 0 else '-',
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
    pairs = list(combinations(feats, 2))
    print(f'Features: {len(feats)}    Pairs: {len(pairs)}')

    rows = []
    for fa, fb in pairs:
        r = regress_pair(df, fa, fb)
        if r:
            rows.append(r)

    table = pd.DataFrame(rows)
    out_path = out_dir / f'pair_regression_{args.name}.csv'
    table.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}    ({len(table)} pairs regressed)')

    if table.empty:
        return

    # ── Ranking 1: best additive R² ──
    print('\n=== Top 20 pairs by ADDITIVE R² (each feature contributes independently) ===')
    add_sort = table.sort_values('r2_additive', ascending=False)
    cols = ['feat_a', 'feat_b', 'n', 'beta_a_per_1sigma_$',
            'beta_b_per_1sigma_$', 'r2_additive', 'r2_interaction', 'r2_gain']
    print(add_sort[cols].head(20).to_string(index=False))

    # ── Ranking 2: highest interaction R² gain ──
    print('\n=== Top 20 pairs by INTERACTION R² gain (features interact non-additively) ===')
    int_sort = table.sort_values('r2_gain', ascending=False)
    cols_i = ['feat_a', 'feat_b', 'n', 'r2_additive', 'r2_interaction',
              'r2_gain', 'beta_AB_per_sigma2_$', 'sign_AB']
    print(int_sort[cols_i].head(20).to_string(index=False))

    # ── Ranking 3: total R² (additive + interaction combined) ──
    print('\n=== Top 20 pairs by TOTAL R² (additive + interaction model) ===')
    tot_sort = table.sort_values('r2_interaction', ascending=False)
    print(tot_sort[cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
