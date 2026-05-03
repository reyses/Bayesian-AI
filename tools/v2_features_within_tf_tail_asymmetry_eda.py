"""
v2_features_within_tf_tail_asymmetry_eda.py — Layer D6: feature x feature
within-TF JOINT TAIL asymmetry (no price target).

Measures the joint distribution shape that Pearson can't capture:
when X is in its top quintile (Q4) and Y is in its top quintile,
do they co-occur more or less than chance? Asymmetric tail dependence.

Pearson treats the entire distribution uniformly. Two pairs can have
identical r yet very different tail behavior. A pair where (Q4_X, Q4_Y)
co-occurs at 2x chance but (Q0_X, Q0_Y) at 0.5x chance has stronger
upper-tail coupling than lower-tail.

For each (X, Y, TF, regime), compute:
  P(Q0_X, Q0_Y)  empirical joint prob in (bottom, bottom)
  P(Q4_X, Q4_Y)  empirical joint prob in (top, top)
  P(Q0_X, Q4_Y)  cross-corner
  P(Q4_X, Q0_Y)  cross-corner
  Compare to chance (1/Q^2 = 0.04 for Q=5).

Key derived metrics:
  upper_tail_lift  = P(Q4,Q4) / chance        > 1 -> co-extreme together
  lower_tail_lift  = P(Q0,Q0) / chance
  tail_asymmetry   = upper_tail_lift - lower_tail_lift  (signed)
  joint_extreme    = (P(Q4,Q4) + P(Q0,Q0)) / chance
  cross_extreme    = (P(Q0,Q4) + P(Q4,Q0)) / chance

Symmetric pair (e.g. body, velocity_1b): upper ~ lower lift, both >> 1.
Asymmetric pair (e.g. range vs volume in DOWN_SMOOTH): upper >> lower
or lower >> upper.

Outputs:
  reports/findings/v2_features_within_tf_tail_asymmetry/
    tail_summary.csv      (X, Y, TF, regime, P(Q4Q4), P(Q0Q0), lifts, asym)
    asymmetric_pairs.csv  pairs with |tail_asymmetry| > threshold
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


CONCEPTS = [
    'price_velocity_1b', 'price_accel_1b',
    'vol_velocity_1b',   'vol_accel_1b',
    'bar_range', 'body',
    'price_velocity_w', 'price_accel_w',
    'vol_velocity_w',   'vol_accel_w',
    'price_mean_w', 'price_sigma_w',
    'vol_mean_w',   'vol_sigma_w',
    'vwap_w',
    'z_se_w', 'z_high_w', 'z_low_w',
    'SE_high_w', 'SE_low_w',
    'hurst_w', 'reversion_prob_w', 'swing_noise_w',
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
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--asymmetry-threshold', type=float, default=0.5,
                        help='|tail_asymmetry| above this = flagged')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_tail_asymmetry')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    Q = args.quantiles
    chance = 1.0 / (Q * Q)
    print(f"{'='*70}")
    print(f"  V2 features within-TF tail asymmetry (Layer D6)")
    print(f"  TFs: {args.tfs}  Q: {Q}  chance per cell: {chance:.4f}")
    print(f"  Asymmetry threshold: {args.asymmetry_threshold}")
    print(f"{'='*70}")

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
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    regimes = full['regime_2d'].values.astype(str)

    rows = []
    print(f"\n--- Sweeping ---")
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in CONCEPTS]
        present = [(c, col) for c, col in zip(CONCEPTS, cols)
                       if col in full.columns]
        if len(present) < 5:
            continue
        tf_arrs = {c: full[col].values.astype(np.float64) for c, col in present}

        for c1, c2 in itertools.combinations([c for c, _ in present], 2):
            x = tf_arrs[c1]
            y = tf_arrs[c2]
            for regime in REGIME_2D_ORDER:
                rmask = (regimes == regime)
                m = rmask & ~np.isnan(x) & ~np.isnan(y)
                if m.sum() < args.min_cell_n * 5:  # need enough for Q-bins
                    continue
                xv, yv = x[m], y[m]
                # quantile bins WITHIN regime
                qx = np.quantile(xv, np.linspace(0, 1, Q + 1))
                qy = np.quantile(yv, np.linspace(0, 1, Q + 1))
                qx[0] -= 1e-9
                qx[-1] += 1e-9
                qy[0] -= 1e-9
                qy[-1] += 1e-9
                bx = np.digitize(xv, qx[1:-1])
                by = np.digitize(yv, qy[1:-1])

                n = len(xv)
                # joint counts per (bx, by) cell
                p_qq = float(((bx == Q-1) & (by == Q-1)).sum() / n)
                p_00 = float(((bx == 0) & (by == 0)).sum() / n)
                p_q0 = float(((bx == Q-1) & (by == 0)).sum() / n)
                p_0q = float(((bx == 0) & (by == Q-1)).sum() / n)

                upper_lift = p_qq / chance
                lower_lift = p_00 / chance
                cross_q0_lift = p_q0 / chance
                cross_0q_lift = p_0q / chance
                tail_asym = upper_lift - lower_lift
                joint_extreme = upper_lift + lower_lift
                cross_extreme = cross_q0_lift + cross_0q_lift

                pearson = float(np.corrcoef(xv, yv)[0, 1])

                rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'regime_2d': regime,
                    'n': n,
                    'pearson': pearson,
                    'p_q4q4': p_qq,
                    'p_q0q0': p_00,
                    'p_q4q0': p_q0,
                    'p_q0q4': p_0q,
                    'upper_lift': upper_lift,
                    'lower_lift': lower_lift,
                    'tail_asymmetry': tail_asym,
                    'joint_extreme': joint_extreme,
                    'cross_extreme': cross_extreme,
                })
        print(f"  {tf}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'tail_summary.csv'), index=False)
    print(f"  [saved] tail_summary.csv ({len(df)} rows)")

    # Strongest tail asymmetries
    df['abs_asym'] = df['tail_asymmetry'].abs()
    asym_df = df[df['abs_asym'] >= args.asymmetry_threshold].copy()
    asym_df = asym_df.sort_values('abs_asym', ascending=False)
    asym_df.to_csv(os.path.join(args.output_dir, 'asymmetric_pairs.csv'),
                      index=False)
    print(f"\n  Asymmetric pairs (|tail_asym| >= {args.asymmetry_threshold}): {len(asym_df)}")
    print(f"\n  Top 30 by |tail_asymmetry|:")
    print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'regime':>14}  "
          f"{'pearson':>7}  {'p_qq':>5}  {'p_00':>5}  {'up':>5}  {'low':>5}  {'asym':>5}")
    for _, r in asym_df.head(30).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  {r['pearson']:>+7.3f}  "
              f"{r['p_q4q4']:>5.3f}  {r['p_q0q0']:>5.3f}  "
              f"{r['upper_lift']:>5.2f}  {r['lower_lift']:>5.2f}  "
              f"{r['tail_asymmetry']:>+5.2f}")

    # Pairs with HIGH joint_extreme but near-zero pearson
    # (Pearson misses these — pure tail dependence)
    pearson_blind = df[(df['joint_extreme'] > 5)
                            & (df['pearson'].abs() < 0.2)].copy()
    pearson_blind = pearson_blind.sort_values('joint_extreme', ascending=False)
    pearson_blind.to_csv(
        os.path.join(args.output_dir, 'pearson_blind_tail_dep.csv'),
        index=False)
    print(f"\n  PEARSON-BLIND tail dependence (joint_extreme>5 but |pearson|<0.2):")
    print(f"  Cases where corner co-extremes >> chance but Pearson reads near-zero")
    print(f"  Total: {len(pearson_blind)}")
    for _, r in pearson_blind.head(20).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  pearson={r['pearson']:>+5.3f}  "
              f"upper={r['upper_lift']:>4.2f}x  lower={r['lower_lift']:>4.2f}x")

    # CROSS-tail dominance: cross_extreme > joint_extreme means
    # negatively-coupled tails (X high when Y low and vice versa)
    cross_dom = df[df['cross_extreme'] > df['joint_extreme'] + 1].copy()
    cross_dom = cross_dom.sort_values('cross_extreme', ascending=False)
    cross_dom.to_csv(os.path.join(args.output_dir, 'cross_dominant_pairs.csv'),
                       index=False)
    print(f"\n  CROSS-TAIL DOMINANT pairs ({len(cross_dom)}): cross_extreme > joint_extreme + 1")
    print(f"  Pairs that anti-couple at extremes (X high <-> Y low):")
    for _, r in cross_dom.head(15).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  pearson={r['pearson']:>+5.3f}  "
              f"cross={r['cross_extreme']:>4.2f}  joint={r['joint_extreme']:>4.2f}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-TF tail asymmetry (Layer D6) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Q**: {Q}  **chance per cell**: {chance:.4f}  "
                f"**Total rows**: {len(df)}\n\n")
        f.write(f"## Top 50 tail-asymmetric pairs\n\n")
        f.write(asym_df.head(50).to_string(index=False))
        f.write(f"\n\n## Pearson-blind tail dependence (joint extreme > 5 but |Pearson| < 0.2)\n\n")
        f.write(pearson_blind.head(50).to_string(index=False))
        f.write(f"\n\n## Cross-tail dominant pairs (anti-couple at extremes)\n\n")
        f.write(cross_dom.head(30).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
