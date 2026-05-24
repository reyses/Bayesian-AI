"""
v2_features_within_tf_drift_eda.py — Layer D7: feature x feature within-TF
calendar drift (no price target).

For each (X, Y, TF, regime) cell measured in D2, split IS into halves
by date and re-measure the corr in each half. Asks: is the regime-
conditional sign-flip we found in D2 STABLE across time, or did the
relationship itself drift?

A trustworthy regime-conditional rule must be both regime-different AND
time-stable. A relationship that is regime-different but drifts across
IS halves is unreliable.

Outputs:
  reports/findings/v2_features_within_tf_drift/
    drift_summary.csv      (X, Y, TF, regime, r_h1, r_h2, delta, sign_flip)
    sign_flip_drift.csv    pairs that flip sign between halves WITHIN regime
    stable_decoupling.csv  pairs whose D2 regime-flip ALSO holds in both halves
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
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_drift')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features within-TF drift (Layer D7)")
    print(f"  TFs: {args.tfs}")
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

    unique_dates = sorted(merged['date'].unique())
    half = len(unique_dates) // 2
    h1_dates = set(unique_dates[:half])
    h2_dates = set(unique_dates[half:])
    print(f"  {len(unique_dates)} days; "
          f"H1: {len(h1_dates)} ({unique_dates[0]} -> {unique_dates[half-1]}); "
          f"H2: {len(h2_dates)} ({unique_dates[half]} -> {unique_dates[-1]})")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    regimes = full['regime_2d'].values.astype(str)
    dates = full['date'].values.astype(str)
    h1_mask = np.array([d in h1_dates for d in dates])
    h2_mask = np.array([d in h2_dates for d in dates])

    rows = []
    print(f"\n--- Sweeping ---")
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in CONCEPTS]
        present = [(c, col) for c, col in zip(CONCEPTS, cols)
                       if col in full.columns]
        if len(present) < 5:
            continue
        # Pre-extract per-TF concept arrays
        tf_arrs = {c: full[col].values.astype(np.float64) for c, col in present}

        for c1, c2 in itertools.combinations([c for c, _ in present], 2):
            x = tf_arrs[c1]
            y = tf_arrs[c2]
            for regime in REGIME_2D_ORDER:
                rmask = (regimes == regime)

                def calc_corr(half_mask):
                    m = rmask & half_mask & ~np.isnan(x) & ~np.isnan(y)
                    if m.sum() < args.min_cell_n:
                        return float('nan'), 0
                    xv, yv = x[m], y[m]
                    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                        return float('nan'), int(m.sum())
                    return float(np.corrcoef(xv, yv)[0, 1]), int(m.sum())

                r1, n1 = calc_corr(h1_mask)
                r2, n2 = calc_corr(h2_mask)
                if np.isnan(r1) or np.isnan(r2):
                    continue
                delta = r2 - r1
                sign_flip = (np.sign(r1) != np.sign(r2)) and abs(r1) > 0.05 and abs(r2) > 0.05
                rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'regime_2d': regime,
                    'r_h1': r1,
                    'r_h2': r2,
                    'delta': delta,
                    'abs_delta': abs(delta),
                    'sign_flip': sign_flip,
                    'n_h1': n1,
                    'n_h2': n2,
                })
        print(f"  {tf}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'drift_summary.csv'), index=False)
    print(f"  [saved] drift_summary.csv ({len(df)} cells)")

    # Statistics
    n_total = len(df)
    n_flips = int(df['sign_flip'].sum())
    print(f"\n  Sign flips (corr changes sign across halves WITHIN regime): "
          f"{n_flips} / {n_total} ({100.0*n_flips/max(n_total,1):.1f}%)")

    flips = df[df['sign_flip']].copy()
    flips['abs_max'] = flips[['r_h1', 'r_h2']].abs().max(axis=1)
    flips = flips.sort_values('abs_max', ascending=False)
    flips.to_csv(os.path.join(args.output_dir, 'sign_flip_drift.csv'),
                    index=False)

    print(f"\n  Top 25 sign-flip drifters (most magnitude in either half):")
    print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'regime':>14}  "
          f"{'r_h1':>7}  {'r_h2':>7}  {'delta':>7}")
    for _, r in flips.head(25).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  {r['r_h1']:>+7.3f}  {r['r_h2']:>+7.3f}  "
              f"{r['delta']:>+7.3f}")

    # Stable: |delta| < 0.10 AND both halves same sign with |r| > 0.10
    stable = df[(df['abs_delta'] < 0.10)
                  & (df['r_h1'].abs() > 0.10)
                  & (np.sign(df['r_h1']) == np.sign(df['r_h2']))].copy()
    stable['abs_mean'] = stable[['r_h1', 'r_h2']].abs().mean(axis=1)
    stable = stable.sort_values('abs_mean', ascending=False)
    stable.to_csv(os.path.join(args.output_dir, 'stable_pairs.csv'), index=False)
    print(f"\n  Stable pairs (|delta|<0.1, same sign, |r|>0.1): {len(stable)}")
    print(f"  Top 25 most-stable HIGH-MAGNITUDE pairs:")
    print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'regime':>14}  "
          f"{'r_h1':>7}  {'r_h2':>7}")
    for _, r in stable.head(25).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  {r['r_h1']:>+7.3f}  {r['r_h2']:>+7.3f}")

    # Cross-reference with D2 sign-flips: which D2 regime-conditional flips
    # ALSO hold in both halves (i.e., the regime difference is stable, not
    # an artifact of one half)?
    d2_path = 'reports/findings/v2_features_within_tf_regime/regime_sign_flips.csv'
    if os.path.exists(d2_path):
        d2 = pd.read_csv(d2_path)
        # For each D2 sign-flip pair (tf, pair), check whether the regime_min
        # corr is consistent in both halves AND regime_max corr is consistent
        # in both halves (so the regime difference is real and stable).
        confirmed_rows = []
        for _, fr in d2.iterrows():
            tf, c1, c2 = fr['tf'], fr['c1'], fr['c2']
            r_min, r_max = fr['r_min'], fr['r_max']
            regime_min, regime_max = fr['regime_min'], fr['regime_max']
            # find drift entries
            h1h2_min = df[(df['tf'] == tf) & (df['c1'] == c1) & (df['c2'] == c2)
                              & (df['regime_2d'] == regime_min)]
            h1h2_max = df[(df['tf'] == tf) & (df['c1'] == c1) & (df['c2'] == c2)
                              & (df['regime_2d'] == regime_max)]
            if h1h2_min.empty or h1h2_max.empty:
                continue
            # In regime_min: both halves should be NEGATIVE-ish
            min_h1 = h1h2_min['r_h1'].iloc[0]
            min_h2 = h1h2_min['r_h2'].iloc[0]
            # In regime_max: both halves should be POSITIVE-ish
            max_h1 = h1h2_max['r_h1'].iloc[0]
            max_h2 = h1h2_max['r_h2'].iloc[0]
            # Confirmed if: same sign as D2 in both halves for both regimes,
            # and |r| > 0.1 in each
            confirmed = (np.sign(min_h1) == np.sign(r_min)
                              and np.sign(min_h2) == np.sign(r_min)
                              and np.sign(max_h1) == np.sign(r_max)
                              and np.sign(max_h2) == np.sign(r_max)
                              and abs(min_h1) > 0.1 and abs(min_h2) > 0.1
                              and abs(max_h1) > 0.1 and abs(max_h2) > 0.1)
            confirmed_rows.append({
                'tf': tf,
                'c1': c1,
                'c2': c2,
                'regime_min': regime_min,
                'regime_max': regime_max,
                'd2_r_min': r_min,
                'd2_r_max': r_max,
                'min_h1': min_h1,
                'min_h2': min_h2,
                'max_h1': max_h1,
                'max_h2': max_h2,
                'confirmed': confirmed,
            })
        cr_df = pd.DataFrame(confirmed_rows)
        cr_df.to_csv(os.path.join(args.output_dir, 'd2_flip_confirmation.csv'),
                       index=False)
        n_conf = int(cr_df['confirmed'].sum())
        n_total_d2 = len(cr_df)
        print(f"\n  D2 sign-flip pairs CONFIRMED in both IS halves: "
              f"{n_conf} / {n_total_d2} "
              f"({100.0*n_conf/max(n_total_d2,1):.1f}%)")

        confirmed_pairs = cr_df[cr_df['confirmed']].copy()
        confirmed_pairs['min_mag'] = (confirmed_pairs[['min_h1', 'min_h2',
                                                                  'max_h1', 'max_h2']]
                                            .abs().min(axis=1))
        confirmed_pairs = confirmed_pairs.sort_values('min_mag', ascending=False)
        print(f"\n  Top 25 CONFIRMED regime-flip pairs (regime difference holds in both halves):")
        print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'reg-':>12}  {'reg+':>12}  "
              f"{'min_h1':>7}  {'min_h2':>7}  {'max_h1':>7}  {'max_h2':>7}")
        for _, r in confirmed_pairs.head(25).iterrows():
            print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
                  f"{r['regime_min']:>12}  {r['regime_max']:>12}  "
                  f"{r['min_h1']:>+7.3f}  {r['min_h2']:>+7.3f}  "
                  f"{r['max_h1']:>+7.3f}  {r['max_h2']:>+7.3f}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-TF drift (Layer D7) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"Half-1 IS dates: {sorted(h1_dates)[0]} -> {sorted(h1_dates)[-1]}  "
                f"({len(h1_dates)} days)\n\n")
        f.write(f"Half-2 IS dates: {sorted(h2_dates)[0]} -> {sorted(h2_dates)[-1]}  "
                f"({len(h2_dates)} days)\n\n")
        f.write(f"Sign flips between halves (within regime): {n_flips}/{n_total} "
                f"({100.0*n_flips/max(n_total,1):.1f}%)\n\n")
        f.write(f"## Top sign-flip drifters\n\n")
        f.write(flips.head(50).to_string(index=False))
        f.write(f"\n\n## Top stable high-magnitude pairs\n\n")
        f.write(stable.head(50).to_string(index=False))
        if 'cr_df' in locals():
            f.write(f"\n\n## D2 sign-flip confirmation\n\n")
            f.write(f"Of {n_total_d2} D2 regime-flip pairs, **{n_conf}** "
                     f"({100.0*n_conf/max(n_total_d2,1):.1f}%) have the regime "
                     f"difference confirmed in BOTH IS halves.\n\n")
            f.write(confirmed_pairs.head(50).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
