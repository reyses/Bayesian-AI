"""
v2_features_within_tf_oos_validate.py — Layer D9: OOS validation of
feature x feature within-TF findings.

Re-measures the headline findings from D2, D7, D8 on the held-out
OOS 71 days:

(A) D7 CONFIRMED regime-flip pairs (83 pairs that passed both regime-
    stratification and IS-half stability): does the regime difference
    still hold on OOS?

(B) D8 TOP contextualizer triplets (top 100 by lift, restricted to
    sign-flippers): does the contextualization still flip corr(X,Y)
    by Z-bin on OOS?

Survival rule for D7 pairs: in regime_min, OOS corr same sign as IS;
                              in regime_max, OOS corr same sign as IS.

Survival rule for D8 triplets: OOS lift >= 0.50 in same direction as IS.

Outputs:
  reports/findings/v2_features_within_tf_oos/
    d7_pair_oos.csv         per pair: IS sign-flip + OOS sign-flip + survives
    d8_triplet_oos.csv      per triplet: IS lift + OOS lift + survives
    summary.md
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--d7-summary',
                        default='reports/findings/v2_features_within_tf_drift/'
                                  'd2_flip_confirmation.csv')
    parser.add_argument('--d8-summary',
                        default='reports/findings/v2_features_within_tf_contextualizer/'
                                  'top_contextualizers.csv')
    parser.add_argument('--top-d8', type=int, default=100,
                        help='Top K D8 contextualizer triplets to validate')
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--min-cell-n', type=int, default=80)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_oos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  Within-TF feature×feature OOS validation (Layer D9)")
    print(f"{'='*70}")

    # Load D7 confirmed pairs
    print(f"\n--- Loading IS findings ---")
    d7 = pd.read_csv(args.d7_summary)
    d7_confirmed = d7[d7['confirmed']].copy()
    print(f"  D7 confirmed pairs: {len(d7_confirmed)}")

    d8 = pd.read_csv(args.d8_summary)
    d8_top = d8.sort_values('lift', ascending=False).head(args.top_d8).copy()
    print(f"  D8 top-{args.top_d8} contextualizers: {len(d8_top)}")

    # Load OOS data
    print(f"\n--- Loading data ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    base_df['date'] = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert(
        'America/Los_Angeles').dt.date.astype(str)
    merged_oos = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')
    merged_oos = merged_oos[merged_oos['split'] == 'OOS'].reset_index(drop=True)
    print(f"  OOS bars: {len(merged_oos):,}")

    ts_int_oos = merged_oos['ts_int'].values.astype(np.int64)
    feats_oos = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int_oos.min()), int(ts_int_oos.max())), verbose=False,
    )
    aligned_oos = align_v2_to_base_tf(feats_oos, ts_int_oos)
    full_oos = pd.concat([merged_oos.reset_index(drop=True),
                            aligned_oos.reset_index(drop=True)], axis=1)
    regimes_oos = full_oos['regime_2d'].values.astype(str)

    # Also load IS for IS quantile boundary derivation (D8 needs them)
    merged_is = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')
    merged_is = merged_is[merged_is['split'] == 'IS'].reset_index(drop=True)
    ts_int_is = merged_is['ts_int'].values.astype(np.int64)
    feats_is = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int_is.min()), int(ts_int_is.max())), verbose=False,
    )
    aligned_is = align_v2_to_base_tf(feats_is, ts_int_is)
    full_is = pd.concat([merged_is.reset_index(drop=True),
                           aligned_is.reset_index(drop=True)], axis=1)

    # ---- (A) D7 PAIR OOS validation ----
    print(f"\n--- (A) Validating D7 confirmed regime-flip pairs on OOS ---")
    d7_rows = []
    for _, r in d7_confirmed.iterrows():
        tf, c1, c2 = r['tf'], r['c1'], r['c2']
        regime_min, regime_max = r['regime_min'], r['regime_max']
        is_r_min = r['d2_r_min']
        is_r_max = r['d2_r_max']

        col1 = feature_column_for(c1, tf)
        col2 = feature_column_for(c2, tf)
        if col1 not in full_oos.columns or col2 not in full_oos.columns:
            continue
        x_oos = full_oos[col1].values.astype(np.float64)
        y_oos = full_oos[col2].values.astype(np.float64)

        def calc_corr_in_regime(regime):
            m = (regimes_oos == regime) & ~np.isnan(x_oos) & ~np.isnan(y_oos)
            if m.sum() < args.min_cell_n:
                return float('nan'), 0
            xv, yv = x_oos[m], y_oos[m]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                return float('nan'), int(m.sum())
            return float(np.corrcoef(xv, yv)[0, 1]), int(m.sum())

        oos_r_min, n_oos_min = calc_corr_in_regime(regime_min)
        oos_r_max, n_oos_max = calc_corr_in_regime(regime_max)
        if np.isnan(oos_r_min) or np.isnan(oos_r_max):
            continue
        # Survives if: same sign as IS in BOTH regimes
        sign_min_ok = np.sign(oos_r_min) == np.sign(is_r_min) and abs(oos_r_min) > 0.05
        sign_max_ok = np.sign(oos_r_max) == np.sign(is_r_max) and abs(oos_r_max) > 0.05
        regime_diff_holds = (np.sign(oos_r_max) != np.sign(oos_r_min)
                                  and abs(oos_r_max - oos_r_min) > 0.20)
        survives = sign_min_ok and sign_max_ok and regime_diff_holds

        d7_rows.append({
            'tf': tf,
            'c1': c1,
            'c2': c2,
            'regime_min': regime_min,
            'regime_max': regime_max,
            'is_r_min': is_r_min,
            'is_r_max': is_r_max,
            'oos_r_min': oos_r_min,
            'oos_r_max': oos_r_max,
            'n_oos_min': n_oos_min,
            'n_oos_max': n_oos_max,
            'sign_min_ok': sign_min_ok,
            'sign_max_ok': sign_max_ok,
            'regime_diff_holds': regime_diff_holds,
            'survives': survives,
        })
    d7_oos = pd.DataFrame(d7_rows)
    d7_oos.to_csv(os.path.join(args.output_dir, 'd7_pair_oos.csv'),
                    index=False)
    n_d7 = len(d7_oos)
    n_surv_d7 = int(d7_oos['survives'].sum())
    print(f"  D7 pairs evaluated: {n_d7}")
    print(f"  D7 pairs SURVIVING OOS: {n_surv_d7} / {n_d7} "
          f"({100.0*n_surv_d7/max(n_d7,1):.1f}%)")

    print(f"\n  Top 25 D7 pairs by IS magnitude — OOS results:")
    d7_oos['min_is_mag'] = d7_oos[['is_r_min', 'is_r_max']].abs().min(axis=1)
    d7_sorted = d7_oos.sort_values('min_is_mag', ascending=False)
    print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'IS_min':>7}  {'IS_max':>7}  "
          f"{'OOS_min':>7}  {'OOS_max':>7}  {'surv':>4}")
    for _, r in d7_sorted.head(25).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['is_r_min']:>+7.3f}  {r['is_r_max']:>+7.3f}  "
              f"{r['oos_r_min']:>+7.3f}  {r['oos_r_max']:>+7.3f}  "
              f"{'YES' if r['survives'] else 'no':>4}")

    # ---- (B) D8 TRIPLET OOS validation ----
    # For each top-K D8 triplet, recompute corr(X, Y) per Z-bin on OOS
    # using IS-derived Z quantile boundaries.
    print(f"\n--- (B) Validating D8 top-{args.top_d8} contextualizers on OOS ---")
    d8_rows = []
    Q = args.quantiles
    for _, r in d8_top.iterrows():
        tf, X, Y, Z = r['tf'], r['X'], r['Y'], r['Z']
        is_r_min = r['r_min']
        is_r_max = r['r_max']
        is_lift = r['lift']

        cX = feature_column_for(X, tf)
        cY = feature_column_for(Y, tf)
        cZ = feature_column_for(Z, tf)
        if any(c not in full_oos.columns for c in [cX, cY, cZ]):
            continue
        if any(c not in full_is.columns for c in [cZ]):
            continue
        x_oos = full_oos[cX].values.astype(np.float64)
        y_oos = full_oos[cY].values.astype(np.float64)
        z_oos = full_oos[cZ].values.astype(np.float64)

        # IS-derived quantile boundaries for Z (across all IS, not regime-stratified
        # since D8 used global Z bins)
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
            if m.sum() < args.min_cell_n:
                oos_corrs.append(float('nan'))
                continue
            xv, yv = x_oos[m], y_oos[m]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                oos_corrs.append(float('nan'))
                continue
            oos_corrs.append(float(np.corrcoef(xv, yv)[0, 1]))

        valid_oos = [c for c in oos_corrs if not np.isnan(c)]
        if len(valid_oos) < 2:
            continue
        oos_r_min = min(valid_oos)
        oos_r_max = max(valid_oos)
        oos_lift = oos_r_max - oos_r_min

        # Sign flip on OOS too?
        oos_sign_flip = (oos_r_min < -0.05 and oos_r_max > 0.05) or \
                              (oos_r_min < 0.05 and oos_r_max > 0.05 and
                                np.sign(oos_r_min) != np.sign(oos_r_max))
        # Strict: OOS direction matches IS, lift >= 0.5
        sign_match = (np.sign(oos_r_min) == np.sign(is_r_min)
                            and np.sign(oos_r_max) == np.sign(is_r_max))
        survives = sign_match and oos_lift >= 0.50

        d8_rows.append({
            'tf': tf, 'X': X, 'Y': Y, 'Z': Z,
            'is_r_min': is_r_min, 'is_r_max': is_r_max, 'is_lift': is_lift,
            'oos_r_min': oos_r_min, 'oos_r_max': oos_r_max,
            'oos_lift': oos_lift,
            'sign_match': sign_match,
            'oos_sign_flip': oos_sign_flip,
            'survives': survives,
        })
    d8_oos = pd.DataFrame(d8_rows)
    d8_oos.to_csv(os.path.join(args.output_dir, 'd8_triplet_oos.csv'),
                    index=False)
    n_d8 = len(d8_oos)
    n_surv_d8 = int(d8_oos['survives'].sum())
    print(f"  D8 triplets evaluated: {n_d8}")
    print(f"  D8 triplets SURVIVING OOS (lift >= 0.5, signs match): "
          f"{n_surv_d8} / {n_d8} "
          f"({100.0*n_surv_d8/max(n_d8,1):.1f}%)")

    print(f"\n  Top 25 D8 triplets — IS vs OOS:")
    d8_oos_sorted = d8_oos.sort_values('is_lift', ascending=False)
    print(f"    {'tf':>4}  {'X':>22}  {'Y':>22}  {'Z':>22}  "
          f"{'IS_lift':>7}  {'OOS_lift':>8}  {'surv':>4}")
    for _, r in d8_oos_sorted.head(25).iterrows():
        print(f"    {r['tf']:>4}  {r['X']:>22}  {r['Y']:>22}  {r['Z']:>22}  "
              f"{r['is_lift']:>7.3f}  {r['oos_lift']:>8.3f}  "
              f"{'YES' if r['survives'] else 'no':>4}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Within-TF feature×feature OOS validation (Layer D9) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"## D7 confirmed regime-flip pairs OOS\n\n")
        f.write(f"- IS confirmed: {len(d7_confirmed)}\n")
        f.write(f"- Evaluated on OOS: {n_d7}\n")
        f.write(f"- **Survive OOS**: {n_surv_d7} ({100.0*n_surv_d7/max(n_d7,1):.1f}%)\n\n")
        f.write(d7_sorted.head(50).to_string(index=False))
        f.write(f"\n\n## D8 top-{args.top_d8} contextualizers OOS\n\n")
        f.write(f"- Evaluated: {n_d8}\n")
        f.write(f"- **Survive OOS** (lift >= 0.5, sign match): "
                f"{n_surv_d8} ({100.0*n_surv_d8/max(n_d8,1):.1f}%)\n\n")
        f.write(d8_oos_sorted.head(50).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
