"""
v2_features_within_tf_optimized_mirror.py — Optimization pass: re-run
the within-TF feature×feature mirror on PRUNED 8-family-representative
set instead of full 23 features.

Per D1 redundancy families finding (mutual r > 0.5 across TFs), the
23 features collapse to ~7 independent families. Picking one
representative per family gives an 8-feature shortlist:
  - bar_range          (dispersion family)
  - body               (bar direction family)
  - vwap_w             (mean family — vwap_w == price_mean_w r=1.000)
  - vol_velocity_w     (volume kinetic family)
  - vol_mean_w         (volume state family)
  - z_se_w             (z-score family — confirmed universal modifier)
  - hurst_w            (independent — weak modifier but distinct)
  - reversion_prob_w   (independent — distinct from hurst_w in v2)

Compares OPTIMIZED outputs to UNOPTIMIZED:
  - D2 sign-flip rate
  - D8 modifier influence (does z_se_w still dominate? what's the gap?)
  - D9 OOS survival rate (does pruning improve survival?)

Outputs:
  reports/findings/v2_features_within_tf_optimized/
    d2_optimized_summary.csv     pair x regime corr (28 pairs * 8 TFs * 6 regimes)
    d8_optimized_summary.csv     contextualizer triplets (8*7*6 = 336 triplets)
    oos_survival_compare.csv     full vs optimized side-by-side
    summary.md
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_tf_sweep_eda import feature_column_for


PRUNED_CONCEPTS = [
    'bar_range',         # dispersion rep
    'body',              # bar direction rep
    'vwap_w',            # mean rep (drop price_mean_w == vwap_w)
    'vol_velocity_w',    # volume kinetic rep
    'vol_mean_w',        # volume state rep
    'z_se_w',            # z-score rep (universal modifier)
    'hurst_w',           # independent
    'reversion_prob_w',  # independent
]

DEFAULT_TFS = ['5s', '1m', '5m', '15m', '1h']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_optimized')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  OPTIMIZED mirror — pruned to {len(PRUNED_CONCEPTS)} family reps")
    print(f"  Pruned concepts: {PRUNED_CONCEPTS}")
    print(f"{'='*70}")

    # Load IS + OOS
    print(f"\n--- Loading data ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int
    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base_df['date'] = dt_la.dt.date.astype(str)

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')

    is_merged = merged[merged['split'] == 'IS'].reset_index(drop=True)
    oos_merged = merged[merged['split'] == 'OOS'].reset_index(drop=True)
    print(f"  IS: {len(is_merged):,} bars; OOS: {len(oos_merged):,} bars")

    def load_features(merged_df):
        ts = merged_df['ts_int'].values.astype(np.int64)
        feats = load_v2_features(
            v2_dir=args.cache, atlas_root=args.data, day_strs=None,
            ts_range=(int(ts.min()), int(ts.max())), verbose=False,
        )
        aligned = align_v2_to_base_tf(feats, ts)
        return pd.concat([merged_df.reset_index(drop=True),
                            aligned.reset_index(drop=True)], axis=1)

    full_is = load_features(is_merged)
    full_oos = load_features(oos_merged)

    is_regimes = full_is['regime_2d'].values.astype(str)
    oos_regimes = full_oos['regime_2d'].values.astype(str)

    # ---- D2 OPTIMIZED: pair × regime correlations (IS) ----
    print(f"\n--- D2 OPTIMIZED: regime-stratified within-TF correlations ---")
    d2_rows = []
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in PRUNED_CONCEPTS]
        present = [(c, col) for c, col in zip(PRUNED_CONCEPTS, cols)
                       if col in full_is.columns]
        if len(present) < 5:
            continue
        for c1, c2 in itertools.combinations([c for c, _ in present], 2):
            x = full_is[feature_column_for(c1, tf)].values.astype(np.float64)
            y = full_is[feature_column_for(c2, tf)].values.astype(np.float64)
            for regime in REGIME_2D_ORDER:
                m = (is_regimes == regime) & ~np.isnan(x) & ~np.isnan(y)
                if m.sum() < args.min_cell_n:
                    continue
                xv, yv = x[m], y[m]
                if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                    continue
                r = float(np.corrcoef(xv, yv)[0, 1])
                d2_rows.append({
                    'tf': tf, 'c1': c1, 'c2': c2,
                    'regime_2d': regime, 'n': int(m.sum()), 'pearson': r,
                })

    d2 = pd.DataFrame(d2_rows)
    d2.to_csv(os.path.join(args.output_dir, 'd2_optimized_summary.csv'),
               index=False)
    print(f"  [saved] d2_optimized_summary.csv ({len(d2)} rows)")

    # Sign-flip analysis: per (pair, tf), does corr flip sign across regimes?
    d2['pair'] = d2['c1'] + '__' + d2['c2']
    flip_rows = []
    for (tf, pair), g in d2.groupby(['tf', 'pair']):
        rs = g['pearson'].values
        pos = rs[rs > 0.05]
        neg = rs[rs < -0.05]
        if len(pos) > 0 and len(neg) > 0:
            flip_rows.append({
                'tf': tf, 'pair': pair,
                'r_min': float(rs.min()), 'r_max': float(rs.max()),
                'spread': float(rs.max() - rs.min()),
            })
    d2_flips = pd.DataFrame(flip_rows).sort_values('spread', ascending=False)

    n_pairs_total = len(d2.groupby(['tf', 'pair']))
    n_flip = len(d2_flips)
    print(f"  Pruned D2: {n_flip} sign-flip pair-TFs of {n_pairs_total}  "
          f"({100*n_flip/max(n_pairs_total,1):.1f}%)")
    print(f"  Compare unoptimized: 577 / 1,265 ({100*577/1265:.1f}%)")

    # ---- D8 OPTIMIZED: contextualizer triplets ----
    print(f"\n--- D8 OPTIMIZED: contextualizer triplets ---")
    d8_rows = []
    Q = args.quantiles
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in PRUNED_CONCEPTS]
        present_concepts = [c for c, col in zip(PRUNED_CONCEPTS, cols)
                                 if col in full_is.columns]
        tf_arrs = {c: full_is[feature_column_for(c, tf)].values.astype(np.float64)
                       for c in present_concepts}

        z_bins = {}
        for z in present_concepts:
            v = tf_arrs[z]
            valid = ~np.isnan(v)
            if valid.sum() < Q * 5:
                continue
            qs = np.quantile(v[valid], np.linspace(0, 1, Q + 1))
            qs[0] -= 1e-9
            qs[-1] += 1e-9
            bin_idx = np.full(len(v), -1, dtype=np.int32)
            bin_idx[valid] = np.digitize(v[valid], qs[1:-1])
            z_bins[z] = bin_idx

        for c1, c2 in itertools.combinations(present_concepts, 2):
            x = tf_arrs[c1]
            y = tf_arrs[c2]
            valid_xy = ~np.isnan(x) & ~np.isnan(y)
            for z in present_concepts:
                if z == c1 or z == c2:
                    continue
                z_b = z_bins.get(z)
                if z_b is None:
                    continue
                bin_corrs = []
                for q in range(Q):
                    m = valid_xy & (z_b == q)
                    if m.sum() < args.min_cell_n:
                        continue
                    xv, yv = x[m], y[m]
                    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                        continue
                    r = float(np.corrcoef(xv, yv)[0, 1])
                    bin_corrs.append((q, r, int(m.sum())))
                if len(bin_corrs) < 2:
                    continue
                rs = np.array([r for _, r, _ in bin_corrs])
                r_min, r_max = float(rs.min()), float(rs.max())
                lift = r_max - r_min
                pos = rs[rs > 0.05]
                neg = rs[rs < -0.05]
                sign_flip = len(pos) > 0 and len(neg) > 0
                d8_rows.append({
                    'tf': tf, 'X': c1, 'Y': c2, 'Z': z,
                    'r_min': r_min, 'r_max': r_max, 'lift': lift,
                    'sign_flip': sign_flip,
                })

    d8 = pd.DataFrame(d8_rows).sort_values('lift', ascending=False)
    d8.to_csv(os.path.join(args.output_dir, 'd8_optimized_summary.csv'),
               index=False)
    print(f"  [saved] d8_optimized_summary.csv ({len(d8)} triplets)")

    n_d8 = len(d8)
    n_d8_flip = int(d8['sign_flip'].sum())
    print(f"  Pruned D8: {n_d8_flip} sign-flips of {n_d8}  "
          f"({100*n_d8_flip/max(n_d8,1):.1f}%)")
    print(f"  Compare unoptimized: 4960 / 26565 (18.7%)")

    print(f"\n  Top 15 contextualizer triplets in pruned set:")
    print(f"    {'tf':>4}  {'X':>22}  {'Y':>22}  {'Z':>22}  "
          f"{'r_min':>7}  {'r_max':>7}  {'lift':>5}  flip")
    for _, r in d8.head(15).iterrows():
        print(f"    {r['tf']:>4}  {r['X']:>22}  {r['Y']:>22}  {r['Z']:>22}  "
              f"{r['r_min']:>+7.3f}  {r['r_max']:>+7.3f}  {r['lift']:>5.3f}  "
              f"{'YES' if r['sign_flip'] else 'no'}")

    # Modifier influence ranking
    z_inf = (d8.groupby('Z')['lift']
                .agg(['mean', 'max', 'count'])
                .reset_index()
                .sort_values('mean', ascending=False))
    z_inf.columns = ['Z', 'mean_lift', 'max_lift', 'n_triplets']
    print(f"\n  Modifier influence ranking (pruned):")
    for _, r in z_inf.iterrows():
        print(f"    {r['Z']:>22}  mean={r['mean_lift']:>5.3f}  "
              f"max={r['max_lift']:>5.3f}  n={int(r['n_triplets'])}")

    # ---- D9 OPTIMIZED OOS validation ----
    print(f"\n--- D9 OPTIMIZED: OOS validation of pruned D2/D8 findings ---")

    # D2 OOS: for each sign-flip pair, recompute corr in regime_min and regime_max on OOS
    d9_d2_rows = []
    for _, fr in d2_flips.iterrows():
        tf, pair = fr['tf'], fr['pair']
        c1, c2 = pair.split('__')
        col1 = feature_column_for(c1, tf)
        col2 = feature_column_for(c2, tf)
        if col1 not in full_oos.columns or col2 not in full_oos.columns:
            continue
        x_oos = full_oos[col1].values.astype(np.float64)
        y_oos = full_oos[col2].values.astype(np.float64)
        # find which regimes had + and - signs in IS
        is_g = d2[(d2['tf'] == tf) & (d2['pair'] == pair)]
        for _, ir in is_g.iterrows():
            regime = ir['regime_2d']
            is_r = ir['pearson']
            m = (oos_regimes == regime) & ~np.isnan(x_oos) & ~np.isnan(y_oos)
            if m.sum() < 50:
                continue
            xv, yv = x_oos[m], y_oos[m]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                continue
            oos_r = float(np.corrcoef(xv, yv)[0, 1])
            d9_d2_rows.append({
                'tf': tf, 'pair': pair, 'regime_2d': regime,
                'is_r': is_r, 'oos_r': oos_r,
                'sign_match': np.sign(is_r) == np.sign(oos_r),
            })
    d9_d2 = pd.DataFrame(d9_d2_rows)
    if len(d9_d2) > 0:
        # Per pair-tf, did BOTH regimes survive sign-match?
        survivors = []
        for (tf, pair), g in d9_d2.groupby(['tf', 'pair']):
            all_match = bool(g['sign_match'].all())
            survivors.append({
                'tf': tf, 'pair': pair, 'n_regimes': len(g),
                'all_signs_match_oos': all_match,
            })
        surv_df = pd.DataFrame(survivors)
        n_pairs = len(surv_df)
        n_surv = int(surv_df['all_signs_match_oos'].sum())
        print(f"  Pruned D2 sign-flip pairs OOS: {n_surv}/{n_pairs} "
              f"({100*n_surv/max(n_pairs,1):.1f}%) survive")

    # D8 OOS: for top-K pruned triplets, recompute lift on OOS
    d9_d8_rows = []
    top_d8 = d8.head(50)
    for _, r in top_d8.iterrows():
        tf, X, Y, Z = r['tf'], r['X'], r['Y'], r['Z']
        cX = feature_column_for(X, tf)
        cY = feature_column_for(Y, tf)
        cZ = feature_column_for(Z, tf)
        if any(c not in full_oos.columns for c in [cX, cY, cZ]):
            continue
        if cZ not in full_is.columns:
            continue
        x_oos = full_oos[cX].values.astype(np.float64)
        y_oos = full_oos[cY].values.astype(np.float64)
        z_oos = full_oos[cZ].values.astype(np.float64)
        z_is = full_is[cZ].values.astype(np.float64)
        valid_z_is = ~np.isnan(z_is)
        if valid_z_is.sum() < Q * 5:
            continue
        qs = np.quantile(z_is[valid_z_is], np.linspace(0, 1, Q + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        bins = np.digitize(z_oos, qs[1:-1])
        valid_xy = ~np.isnan(x_oos) & ~np.isnan(y_oos) & ~np.isnan(z_oos)
        oos_corrs = []
        for q in range(Q):
            m = valid_xy & (bins == q)
            if m.sum() < 50:
                continue
            xv, yv = x_oos[m], y_oos[m]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                continue
            oos_corrs.append(float(np.corrcoef(xv, yv)[0, 1]))
        if len(oos_corrs) < 2:
            continue
        oos_lift = max(oos_corrs) - min(oos_corrs)
        oos_r_min = min(oos_corrs)
        oos_r_max = max(oos_corrs)
        sign_match = (np.sign(oos_r_min) == np.sign(r['r_min'])
                            and np.sign(oos_r_max) == np.sign(r['r_max']))
        survives = sign_match and oos_lift >= 0.50
        d9_d8_rows.append({
            'tf': tf, 'X': X, 'Y': Y, 'Z': Z,
            'is_lift': r['lift'], 'oos_lift': oos_lift,
            'oos_r_min': oos_r_min, 'oos_r_max': oos_r_max,
            'survives': survives,
        })
    d9_d8 = pd.DataFrame(d9_d8_rows)
    if len(d9_d8) > 0:
        n_d8_oos = len(d9_d8)
        n_surv_d8 = int(d9_d8['survives'].sum())
        print(f"  Pruned D8 top-50 triplets OOS: {n_surv_d8}/{n_d8_oos} "
              f"({100*n_surv_d8/max(n_d8_oos,1):.1f}%) survive")
        print(f"  Compare unoptimized D9: 100/100 (100%)")

    # ---- Side-by-side comparison ----
    cmp_rows = [
        {'metric': 'D2 sign-flip rate',
         'unoptimized': '577 / 1,265 = 45.6%',
         'optimized': f'{n_flip} / {n_pairs_total} = {100*n_flip/max(n_pairs_total,1):.1f}%'},
        {'metric': 'D8 sign-flip rate',
         'unoptimized': '4960 / 26,565 = 18.7%',
         'optimized': f'{n_d8_flip} / {n_d8} = {100*n_d8_flip/max(n_d8,1):.1f}%'},
        {'metric': 'D9 D2 OOS survival',
         'unoptimized': '79 / 83 = 95.2%',
         'optimized': (f'{n_surv} / {n_pairs} = {100*n_surv/max(n_pairs,1):.1f}%'
                          if 'n_surv' in locals() else 'n/a')},
        {'metric': 'D9 D8 OOS survival',
         'unoptimized': '100 / 100 = 100%',
         'optimized': (f'{n_surv_d8} / {n_d8_oos} = {100*n_surv_d8/max(n_d8_oos,1):.1f}%'
                          if 'n_surv_d8' in locals() else 'n/a')},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(os.path.join(args.output_dir, 'oos_survival_compare.csv'),
                    index=False)
    print(f"\n  ===  OPTIMIZED vs UNOPTIMIZED ===")
    for _, r in cmp_df.iterrows():
        print(f"    {r['metric']:>20}: unoptimized={r['unoptimized']:>20}  "
              f"optimized={r['optimized']}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Optimized within-TF mirror — pruned to "
                f"{len(PRUNED_CONCEPTS)} family reps\n\n")
        f.write(f"Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Pruned concepts**: {PRUNED_CONCEPTS}\n\n")
        f.write(f"## Optimized vs unoptimized comparison\n\n")
        f.write(cmp_df.to_string(index=False))
        f.write(f"\n\n## D2 sign-flip pairs (pruned)\n\n")
        f.write(d2_flips.head(40).to_string(index=False))
        f.write(f"\n\n## D8 top contextualizer triplets (pruned)\n\n")
        f.write(d8.head(30).to_string(index=False))
        f.write(f"\n\n## Modifier influence ranking (pruned)\n\n")
        f.write(z_inf.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
