"""Simple 1D regression on each entry-time feature against mfe_dollars.

Per user 2026-05-16: linear regression per feature is the headline.

For each continuous feature we fit three things — because the quantile table
already showed several features are U-shaped (slopes especially):

    Linear     y = β₀ + β₁ x           → slope, slope_per_1σ ($), R², p
    Quadratic  y = β₀ + β₁ x + β₂ x²   → R² gain over linear, sign of β₂
                                          (β₂ > 0 = U-shape valley;
                                           β₂ < 0 = inverted-U mountain)
    Spearman   rank correlation ρ      → catches monotone non-linear

Output: one row per feature with all three regressions, plus a stdout
summary ranked by quadratic R² (the best single fit-quality number).

Caveat: with n ~ 7,925, every p-value is going to be tiny. P-values are
not the right measure here — use effect-size ($-per-σ) and R² instead.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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


def regress_feature(df: pd.DataFrame, feat: str) -> dict | None:
    """Run linear + quadratic + Spearman on feat → mfe_dollars."""
    if feat not in df.columns:
        return None
    sub = df[[feat, 'mfe_dollars']].dropna()
    if len(sub) < 100:
        return None
    x = sub[feat].astype(float).values
    y = sub['mfe_dollars'].astype(float).values
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0:
        return None

    # Linear
    lr = stats.linregress(x, y)
    lin_r2 = float(lr.rvalue ** 2)
    lin_slope_per_sigma = float(lr.slope * x_std)

    # Quadratic
    coefs = np.polyfit(x, y, 2)            # [β₂, β₁, β₀]
    y_pred_q = np.polyval(coefs, x)
    ss_res = float(np.sum((y - y_pred_q) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    quad_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    quad_b2 = float(coefs[0])

    # Spearman rank correlation
    sp_rho, sp_p = stats.spearmanr(x, y)

    return {
        'feature':                 feat,
        'n':                       int(len(sub)),
        'x_std':                   round(x_std, 4),
        'y_std':                   round(y_std, 4),
        # Linear
        'lin_slope':               round(float(lr.slope), 5),
        'lin_slope_per_1sigma_$':  round(lin_slope_per_sigma, 2),
        'lin_intercept':           round(float(lr.intercept), 2),
        'lin_r2':                  round(lin_r2, 4),
        'lin_p':                   float(lr.pvalue),
        # Quadratic
        'quad_b2':                 round(quad_b2, 5),
        'quad_shape':              ('U' if quad_b2 > 0 else 'inv-U'),
        'quad_r2':                 round(quad_r2, 4),
        'quad_r2_gain_over_lin':   round(quad_r2 - lin_r2, 4),
        # Spearman
        'spearman_rho':            round(float(sp_rho), 4),
        'spearman_p':              float(sp_p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--name', default='IS_full_daisy')
    ap.add_argument('--exit', action='store_true',
                    help='Run on exit_* features instead of entry')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} trades from {args.input}')

    feats = ([f'exit_{c}' for c in CONTINUOUS_FEATS if f'exit_{c}' in df.columns]
             if args.exit else
             [c for c in CONTINUOUS_FEATS if c in df.columns])
    prefix = 'exit_' if args.exit else ''
    print(f'  features: {len(feats)}')

    rows = []
    for feat in feats:
        r = regress_feature(df, feat)
        if r:
            rows.append(r)

    if not rows:
        print('No features regressed.')
        return

    table = pd.DataFrame(rows)
    out_path = out_dir / f'feature_regression_{prefix}{args.name}.csv'
    table.to_csv(out_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'  total features regressed: {len(table)}')

    # ── Headline ranking: linear R² (the "simple 1D regression" the user asked for) ──
    print('\n=== Linear regression (the headline "simple 1D" — sorted by linear R²) ===')
    lin_sort = table.sort_values('lin_r2', ascending=False)
    cols_lin = ['feature', 'n', 'lin_slope', 'lin_slope_per_1sigma_$',
                'lin_intercept', 'lin_r2']
    print(lin_sort[cols_lin].to_string(index=False))

    # ── Quadratic ranking — catches U-shapes ──
    print('\n=== Adding quadratic term (sorted by quad R² — best fit) ===')
    quad_sort = table.sort_values('quad_r2', ascending=False)
    cols_q = ['feature', 'lin_r2', 'quad_r2', 'quad_r2_gain_over_lin',
              'quad_shape', 'quad_b2']
    print(quad_sort[cols_q].to_string(index=False))

    # ── Spearman ranking — monotone non-linear ──
    print('\n=== Spearman rank correlation (sorted by |rho| — monotone strength) ===')
    table['abs_rho'] = table['spearman_rho'].abs()
    sp_sort = table.sort_values('abs_rho', ascending=False).drop('abs_rho', axis=1)
    cols_sp = ['feature', 'spearman_rho', 'lin_r2', 'quad_r2']
    print(sp_sort[cols_sp].to_string(index=False))


if __name__ == '__main__':
    main()
