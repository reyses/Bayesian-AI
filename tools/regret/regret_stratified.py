"""Stratified pair analysis — subset first, THEN cluster + regress within subset.

Per user 2026-05-16 fallback directive: "smaller like-to-like samples should
help separate the shaft from the seeds."

Approach: pre-subset the dataset by a single stratifier feature (e.g.,
bar_range quartile or tod_minutes bucket), then run the full pair clustering
+ regression analysis WITHIN each subset. This addresses two problems:

  1. Heterogeneous-data noise: global aggregation washes out subgroup-specific
     signals. By restricting to one stratum, we see cleaner relationships.
  2. Direction-aware: pair clustering on signed_mfe still applies — we now
     get per-stratum direction-callable cells.

Output:
    stratified_pair_clusters_<stratifier>_<name>.csv
    stratified_pair_regression_<stratifier>_<name>.csv

Each row has an extra `stratum` column identifying the subset.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR_DEFAULT = Path('reports/findings/regret_oracle')
MIN_N_PER_CELL  = 20
MIN_N_PER_STRATUM = 500   # don't bother stratifying into tiny groups
N_BOOT = 2000
BIN_W_SIGNED = 10.0

CONTINUOUS_FEATS = [
    'z_15s', 'z_1m', 'z_15m', 'z_1h_high', 'z_1h_low',
    'dist_15m_to_Mh', 'dist_15m_to_Ml',
    'dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m', 'fan_width',
    'slope_15s_3m', 'slope_15s_10m', 'slope_1m_10m',
    'slope_15m_5m', 'slope_15m_15m',
    'bar_range', 'volume',
    'tod_minutes',
]


def bootstrap_mean_ci(vals, n_boot=N_BOOT, rng=None):
    if len(vals) < 2:
        m = float(vals[0]) if len(vals) else float('nan')
        return m, m, m
    if rng is None: rng = np.random.default_rng(42)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def wilson_ci(p_hat, n, z=1.96):
    if n < 1: return float('nan'), float('nan')
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n))/denom
    spread = z * np.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n))/denom
    return max(0.0, center - spread), min(1.0, center + spread)


def cluster_pair_in_stratum(df, fa, fb, stratum_label, n_bins, rng):
    sub = df[[fa, fb, 'signed_mfe', 'mfe_dollars', 'direction']].dropna()
    if len(sub) < 200:
        return []
    sub = sub.copy()
    try:
        sub['_ba'] = pd.qcut(sub[fa].astype(float), n_bins, duplicates='drop')
        sub['_bb'] = pd.qcut(sub[fb].astype(float), n_bins, duplicates='drop')
    except Exception:
        return []
    idx_a = {iv: i+1 for i, iv in enumerate(sub['_ba'].cat.categories)}
    idx_b = {iv: i+1 for i, iv in enumerate(sub['_bb'].cat.categories)}
    rows = []
    for (ba, bb), cell in sub.groupby(['_ba', '_bb'], observed=True):
        if len(cell) < MIN_N_PER_CELL: continue
        signed = cell['signed_mfe'].astype(float).values
        pct_long = float((cell['direction']=='LONG').mean())
        pl_lo, pl_hi = wilson_ci(pct_long, len(cell))
        sm_mean, sm_lo, sm_hi = bootstrap_mean_ci(signed, rng=rng)
        rows.append({
            'stratum': stratum_label,
            'feat_a': fa, 'feat_b': fb,
            'qa': idx_a[ba], 'qb': idx_b[bb],
            'n': len(cell),
            'pct_long': round(100*pct_long, 1),
            'pct_long_ci_lo': round(100*pl_lo, 1),
            'pct_long_ci_hi': round(100*pl_hi, 1),
            'mean_signed': round(sm_mean, 2),
            'mean_signed_ci_lo': round(sm_lo, 2),
            'mean_signed_ci_hi': round(sm_hi, 2),
            'mean_$_magnitude': round(float(cell['mfe_dollars'].mean()), 2),
            'long_callable': bool(pl_lo > 0.70),
            'short_callable': bool(pl_hi < 0.30),
        })
    return rows


def regress_pair_in_stratum(df, fa, fb, stratum_label):
    sub = df[[fa, fb, 'signed_mfe']].dropna()
    if len(sub) < 200: return None
    a = sub[fa].astype(float).values
    b = sub[fb].astype(float).values
    y = sub['signed_mfe'].astype(float).values
    a_std = float(np.std(a)); b_std = float(np.std(b))
    if a_std == 0 or b_std == 0: return None
    ss_tot = float(np.sum((y - y.mean())**2))
    one = np.ones(len(y))
    X_add = np.column_stack([one, a, b])
    co_a, *_ = np.linalg.lstsq(X_add, y, rcond=None)
    y_pa = X_add @ co_a
    r2_add = 1 - float(np.sum((y - y_pa)**2))/ss_tot if ss_tot > 0 else 0
    X_int = np.column_stack([one, a, b, a*b])
    co_i, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pi = X_int @ co_i
    r2_int = 1 - float(np.sum((y - y_pi)**2))/ss_tot if ss_tot > 0 else 0
    return {
        'stratum': stratum_label,
        'feat_a': fa, 'feat_b': fb, 'n': len(sub),
        'beta_a_per_sigma': round(float(co_a[1] * a_std), 2),
        'beta_b_per_sigma': round(float(co_a[2] * b_std), 2),
        'r2_additive': round(r2_add, 4),
        'r2_interaction': round(r2_int, 4),
        'r2_gain': round(r2_int - r2_add, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--stratifier', required=True,
                    help='Feature to stratify on (e.g. bar_range, tod_minutes, '
                         'slope_15s_3m)')
    ap.add_argument('--n-strata', type=int, default=4,
                    help='Number of quantile strata (default 4)')
    ap.add_argument('--n-bins', type=int, default=5,
                    help='Pair cell bin count (default 5)')
    ap.add_argument('--name', default='IS_full_daisy')
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if 'signed_mfe' not in df.columns:
        df['signed_mfe'] = df['mfe_dollars'] * np.where(df['direction']=='LONG', 1, -1)
    if args.stratifier not in df.columns:
        raise SystemExit(f'Stratifier {args.stratifier} not in CSV')
    print(f'Loaded {len(df)} trades. Stratifying by {args.stratifier} '
          f'({args.n_strata} quantiles).')

    # Build strata
    df['_stratum'] = pd.qcut(df[args.stratifier].astype(float),
                              args.n_strata, duplicates='drop')
    strata = list(df['_stratum'].cat.categories)
    print(f'Strata: {len(strata)}')
    for i, s in enumerate(strata):
        sub = df[df['_stratum'] == s]
        print(f'  S{i+1} [{s.left:.3g}, {s.right:.3g}]: n={len(sub)}')

    feats = [c for c in CONTINUOUS_FEATS if c in df.columns and c != args.stratifier]
    pairs = list(itertools.combinations(feats, 2))
    print(f'Features (excluding stratifier): {len(feats)}    Pairs per stratum: {len(pairs)}')

    rng = np.random.default_rng(42)
    all_cluster_rows = []
    all_reg_rows = []
    for s_idx, s in enumerate(strata):
        sub = df[df['_stratum'] == s].reset_index(drop=True)
        if len(sub) < MIN_N_PER_STRATUM:
            print(f'  S{s_idx+1}: too few rows ({len(sub)} < {MIN_N_PER_STRATUM}); skipping')
            continue
        stratum_label = f'S{s_idx+1}_{args.stratifier}_[{s.left:.3g},{s.right:.3g}]'
        print(f'\n=== {stratum_label}  (n={len(sub)}) ===')
        for fa, fb in pairs:
            all_cluster_rows.extend(
                cluster_pair_in_stratum(sub, fa, fb, stratum_label, args.n_bins, rng))
            r = regress_pair_in_stratum(sub, fa, fb, stratum_label)
            if r:
                all_reg_rows.append(r)

    cl = pd.DataFrame(all_cluster_rows)
    rg = pd.DataFrame(all_reg_rows)
    cl_path = out_dir / f'stratified_pair_clusters_{args.stratifier}_{args.name}.csv'
    rg_path = out_dir / f'stratified_pair_regression_{args.stratifier}_{args.name}.csv'
    cl.to_csv(cl_path, index=False)
    rg.to_csv(rg_path, index=False)
    print(f'\nWrote: {cl_path}  ({len(cl)} cells)')
    print(f'Wrote: {rg_path}  ({len(rg)} regressions)')

    if not rg.empty:
        print(f'\n=== Top 15 regressions across all strata (by r2_interaction) ===')
        top = rg.sort_values('r2_interaction', ascending=False).head(15)
        print(top.to_string(index=False))

    if not cl.empty:
        long_call = cl[cl['long_callable']]
        short_call = cl[cl['short_callable']]
        print(f'\nLONG-callable cells across strata: {len(long_call)} of {len(cl)}')
        print(f'SHORT-callable cells across strata: {len(short_call)} of {len(cl)}')
        if len(long_call) > 0:
            print(f'\nTop 10 LONG-callable by mean_signed:')
            top_l = long_call.sort_values('mean_signed', ascending=False).head(10)
            cols = ['stratum', 'feat_a', 'qa', 'feat_b', 'qb', 'n',
                    'pct_long', 'mean_signed', 'mean_$_magnitude']
            print(top_l[cols].to_string(index=False))
        if len(short_call) > 0:
            print(f'\nTop 10 SHORT-callable by |mean_signed|:')
            top_s = short_call.sort_values('mean_signed', ascending=True).head(10)
            cols = ['stratum', 'feat_a', 'qa', 'feat_b', 'qb', 'n',
                    'pct_long', 'mean_signed', 'mean_$_magnitude']
            print(top_s[cols].to_string(index=False))


if __name__ == '__main__':
    main()
