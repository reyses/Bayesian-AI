"""
v2_features_within_tf_contextualizer_eda.py — Layer D8: feature x feature
WITHIN-TF contextualization (no price target).

For each pair (X, Y) at TF T, ask: does a third feature Z modify the
corr(X, Y) when stratified by Z's quantile? This is the analog of the
price-track contextualization layer — but here the target is corr(X, Y),
not corr(target, fwd_return).

If corr(X, Y) is +0.6 when Z is in Q0 but -0.2 when Z is in Q4, then Z
is a STRONG contextualizer of (X, Y) — it flips the relationship.

Algorithm:
  C(23, 2) = 253 pairs (X, Y)
  21 candidate modifiers Z per pair (any feature not in the pair)
  5 TFs * 5 Z-quantile bins per pair-modifier
  -> 5,313 (X, Y, Z) triplets * 5 TFs = 26,565 stratified corrs

Outputs:
  reports/findings/v2_features_within_tf_contextualizer/
    contextualizer_summary.csv  (X, Y, Z, TF, q_z, corr_xy, n)
    top_contextualizers.csv     per (X, Y, TF): top Z by corr-range across q_z
    sign_flippers.csv           triplets where corr(X,Y) flips sign by Z bin
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
from tools.atlas_regime_labeler_2d import load_regime_labels
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
    parser.add_argument('--lift-threshold', type=float, default=0.30,
                        help='|max_r - min_r| across Z bins to flag')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_contextualizer')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    Q = args.quantiles
    print(f"{'='*70}")
    print(f"  V2 features within-TF contextualizer (Layer D8)")
    print(f"  TFs: {args.tfs}  Q (Z bins): {Q}")
    print(f"  Lift threshold: {args.lift_threshold}")
    n_pairs = 23 * 22 // 2
    n_triplets = n_pairs * 21
    print(f"  {n_triplets} (X, Y, Z) triplets * {len(args.tfs)} TFs * "
          f"{Q} Z-bins = {n_triplets * len(args.tfs) * Q} corrs")
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

    rows = []
    print(f"\n--- Sweeping ---")
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in CONCEPTS]
        present_concepts = [c for c, col in zip(CONCEPTS, cols)
                                 if col in full.columns]
        tf_arrs = {c: full[feature_column_for(c, tf)].values.astype(np.float64)
                       for c in present_concepts}

        # Pre-compute Z-bin assignments for every concept at this TF
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
                # corr(x, y) per z bin
                bin_corrs = []
                for q in range(Q):
                    m = valid_xy & (z_b == q)
                    if m.sum() < args.min_cell_n:
                        bin_corrs.append((q, float('nan'), 0))
                        continue
                    xv, yv = x[m], y[m]
                    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                        bin_corrs.append((q, float('nan'), int(m.sum())))
                        continue
                    r = float(np.corrcoef(xv, yv)[0, 1])
                    bin_corrs.append((q, r, int(m.sum())))
                # filter to bins with valid corrs
                valid_bins = [(q, r, n) for q, r, n in bin_corrs
                               if not np.isnan(r)]
                if len(valid_bins) < 2:
                    continue
                rs = np.array([r for _, r, _ in valid_bins])
                r_min, r_max = float(rs.min()), float(rs.max())
                lift = r_max - r_min
                # sign flip check
                pos = rs[rs > 0.05]
                neg = rs[rs < -0.05]
                sign_flip = len(pos) > 0 and len(neg) > 0
                # store summary row only (per-bin too verbose otherwise)
                for q, r, n in valid_bins:
                    rows.append({
                        'tf': tf,
                        'X': c1,
                        'Y': c2,
                        'Z': z,
                        'q_z': q,
                        'n': n,
                        'corr_xy': r,
                    })
        print(f"  {tf}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'contextualizer_summary.csv'),
                index=False)
    print(f"  [saved] contextualizer_summary.csv ({len(df)} rows)")

    # Per (X, Y, Z, TF) compute lift = max - min corr across Z bins
    print(f"\n--- Computing per-triplet lift ---")
    grouped = df.groupby(['tf', 'X', 'Y', 'Z'])
    lift_rows = []
    for (tf, X, Y, Z), g in grouped:
        rs = g['corr_xy'].dropna().values
        if len(rs) < 2:
            continue
        r_min, r_max = float(rs.min()), float(rs.max())
        lift = r_max - r_min
        pos = rs[rs > 0.05]
        neg = rs[rs < -0.05]
        sign_flip = len(pos) > 0 and len(neg) > 0
        lift_rows.append({
            'tf': tf,
            'X': X,
            'Y': Y,
            'Z': Z,
            'r_min': r_min,
            'r_max': r_max,
            'lift': lift,
            'n_bins': len(rs),
            'sign_flip': sign_flip,
        })
    lift_df = pd.DataFrame(lift_rows).sort_values('lift', ascending=False)
    lift_df.to_csv(os.path.join(args.output_dir, 'top_contextualizers.csv'),
                       index=False)
    print(f"  [saved] top_contextualizers.csv ({len(lift_df)} triplets)")

    print(f"\n  Top 30 contextualizers by lift across Z bins:")
    print(f"    {'tf':>4}  {'X':>22}  {'Y':>22}  {'Z':>22}  "
          f"{'r_min':>7}  {'r_max':>7}  {'lift':>5}  {'flip':>4}")
    for _, r in lift_df.head(30).iterrows():
        print(f"    {r['tf']:>4}  {r['X']:>22}  {r['Y']:>22}  {r['Z']:>22}  "
              f"{r['r_min']:>+7.3f}  {r['r_max']:>+7.3f}  {r['lift']:>5.3f}  "
              f"{'YES' if r['sign_flip'] else 'no':>4}")

    flip_df = lift_df[lift_df['sign_flip']].copy()
    flip_df.to_csv(os.path.join(args.output_dir, 'sign_flippers.csv'),
                      index=False)
    print(f"\n  Triplets where corr(X,Y) FLIPS SIGN by Z-bin: {len(flip_df)} of {len(lift_df)} "
          f"({100.0*len(flip_df)/max(len(lift_df),1):.1f}%)")
    print(f"  Top 25 sign-flippers (Z modifier toggles X-Y correlation polarity):")
    print(f"    {'tf':>4}  {'X':>22}  {'Y':>22}  {'Z':>22}  "
          f"{'r_min':>7}  {'r_max':>7}  {'lift':>5}")
    for _, r in flip_df.head(25).iterrows():
        print(f"    {r['tf']:>4}  {r['X']:>22}  {r['Y']:>22}  {r['Z']:>22}  "
              f"{r['r_min']:>+7.3f}  {r['r_max']:>+7.3f}  {r['lift']:>5.3f}")

    # Most influential modifier features (Z that contextualizes the most pairs)
    z_influence = (lift_df.groupby('Z')['lift']
                       .agg(['mean', 'max', 'count'])
                       .reset_index()
                       .sort_values('mean', ascending=False))
    z_influence.columns = ['Z', 'mean_lift', 'max_lift', 'n_triplets']
    z_influence.to_csv(os.path.join(args.output_dir, 'modifier_influence.csv'),
                          index=False)
    print(f"\n  Modifier-feature influence ranking (mean lift across all triplets):")
    for _, r in z_influence.iterrows():
        print(f"    {r['Z']:>22}  mean={r['mean_lift']:>5.3f}  "
              f"max={r['max_lift']:>5.3f}  n={int(r['n_triplets'])}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-TF contextualizer (Layer D8) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"## Top 50 contextualizer triplets by lift\n\n")
        f.write(lift_df.head(50).to_string(index=False))
        f.write(f"\n\n## Sign-flippers ({len(flip_df)} triplets)\n\n")
        f.write(flip_df.head(50).to_string(index=False))
        f.write(f"\n\n## Modifier-feature influence ranking\n\n")
        f.write(z_influence.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
