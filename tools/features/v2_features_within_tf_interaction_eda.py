"""
v2_features_within_tf_interaction_eda.py — Layer D1: feature x feature
within-TF interaction map (NO price target).

Drilldown asks: at a single timeframe, how do the 23 v2 features
co-vary with each other? Independent of forward price return — pure
feature-feature structure.

Built around physical hypotheses (the user's questions):
  - Does velocity slow when volume dries?  (|velocity_w| ~ vol_mean_w +)
  - Does high velocity have low variation? (|velocity_w| ~ price_sigma_w ?)
  - Range vs volume                         (bar_range ~ vol_mean_w +)
  - Body vs velocity                        (body ~ velocity_1b +)
  - Hurst vs reversion_prob                 (hurst ~ reversion_prob -)
  - VWAP vs price_mean redundancy           (vwap ~ price_mean +near 1)
  - Vol-mean vs vol-sigma redundancy        (vol_mean ~ vol_sigma + audit)
  - SE_high vs SE_low                       (+ audit)
  - swing_noise vs bar_range                (+ audit)
  - Extreme z -> wide bars                  (|z_se| ~ bar_range +)
  - Acceleration vs velocity                (accel ~ velocity ?)
  - Volume kinetic -> price kinetic?        (vol_velocity ~ price_velocity ?)

For each TF, computes:
  - Pearson AND Spearman correlation matrix (23 x 23) for all
    concept-feature pairs
  - Hypothesis test results (predicted sign vs measured)
  - Pairs whose corr SIGN flips across TFs (TF-dependent relationships)

Outputs:
  reports/findings/v2_features_within_tf_interaction/
    corr_matrix_<tf>.csv
    heatmap_corr_<tf>.png
    hypothesis_tests.csv     (hyp, tf, n, pearson, spearman, predicted, agrees)
    hypothesis_summary.csv   (per hypothesis: agreement count, sign flips)
    redundancy_top.csv       (top |corr| pairs across all TFs)
    sign_flips.csv           (pairs that flip sign across TFs)
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels
from tools.v2_features_tf_sweep_eda import feature_column_for


# Canonical 23 feature concepts in feature_column_for-compatible naming
# (L1 ends in _1b OR bar_range/body bare; L2/L3 end in _w)
CONCEPTS = [
    # L1 (6) — bar primitives
    'price_velocity_1b', 'price_accel_1b',
    'vol_velocity_1b',   'vol_accel_1b',
    'bar_range', 'body',
    # L2 (9) — rolling-window stats
    'price_velocity_w', 'price_accel_w',
    'vol_velocity_w',   'vol_accel_w',
    'price_mean_w', 'price_sigma_w',
    'vol_mean_w',   'vol_sigma_w',
    'vwap_w',
    # L3 (8)
    'z_se_w', 'z_high_w', 'z_low_w',
    'SE_high_w', 'SE_low_w',
    'hurst_w', 'reversion_prob_w', 'swing_noise_w',
]

DEFAULT_TFS = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']


# Hypothesis schema:
#   ('name', x_concept_or_'abs:concept', y_concept_or_'abs:concept',
#    predicted_sign in {+, -, 0, ?}, interpretation)
HYPOTHESES = [
    ('velocity_needs_volume',
        'abs:price_velocity_w', 'vol_mean_w', '+',
        'directional moves require volume'),
    ('velocity_kills_variation',
        'abs:price_velocity_w', 'price_sigma_w', '?',
        'high velocity vs sigma — user predicts negative'),
    ('chop_vs_trend',
        'hurst_w', 'reversion_prob_w', '-',
        'Hurst trending vs reversion mean-revert (definitional)'),
    ('range_tracks_volume',
        'bar_range', 'vol_mean_w', '+',
        'busy bars are big bars'),
    ('body_is_velocity',
        'body', 'price_velocity_1b', '+',
        'body and bar-velocity are the same primitive'),
    ('vwap_vs_price_mean',
        'vwap_w', 'price_mean_w', '+',
        'VWAP and unweighted mean redundancy audit'),
    ('vol_mean_vs_sigma',
        'vol_mean_w', 'vol_sigma_w', '+',
        'are volume mean and volume sigma redundant'),
    ('swing_noise_vs_range',
        'swing_noise_w', 'bar_range', '+',
        'are swing_noise and bar_range redundant'),
    ('SE_symmetry',
        'SE_high_w', 'SE_low_w', '+',
        'upper and lower standard errors symmetric'),
    ('z_extreme_wide_bars',
        'abs:z_se_w', 'bar_range', '+',
        'extreme regression dislocation comes with wide bars'),
    ('z_extreme_high_sigma',
        'abs:z_se_w', 'price_sigma_w', '+',
        'extreme z = high local variation'),
    ('accel_vs_velocity',
        'price_accel_w', 'price_velocity_w', '?',
        'acceleration vs velocity within window'),
    ('vol_kinetic_drives_price',
        'vol_velocity_w', 'price_velocity_w', '?',
        'do volume swings drive price swings?'),
    ('abs_vol_velocity_vs_range',
        'abs:vol_velocity_w', 'bar_range', '+',
        'rapid volume change = bigger bars'),
    ('reversion_vs_swing_noise',
        'reversion_prob_w', 'swing_noise_w', '?',
        'is reversion higher when path is noisier'),
    ('z_high_low_symmetry',
        'z_high_w', 'z_low_w', '+',
        'upper-band and lower-band z symmetric'),
    ('hurst_vs_swing_noise',
        'hurst_w', 'swing_noise_w', '-',
        'trending market has less path noise'),
    ('vol_accel_vs_vol_velocity',
        'vol_accel_w', 'vol_velocity_w', '?',
        'volume acceleration leads volume velocity'),
]


def parse_feature_spec(spec: str) -> tuple[str, bool]:
    """Parse 'concept' or 'abs:concept' -> (concept, take_abs)."""
    if spec.startswith('abs:'):
        return spec[4:], True
    return spec, False


def get_series(full_df: pd.DataFrame, concept: str, tf: str) -> np.ndarray:
    col = feature_column_for(concept, tf)
    if col not in full_df.columns:
        return None
    return full_df[col].values.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_interaction')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features within-TF interaction (Layer D1)")
    print(f"  TFs: {args.tfs}")
    print(f"  Concepts: {len(CONCEPTS)}")
    print(f"  Hypotheses: {len(HYPOTHESES)}")
    print(f"{'='*70}")

    # Load + merge
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

    # ---- Per-TF correlation matrices ----
    print(f"\n--- Computing per-TF correlation matrices ---")
    corr_matrices = {}  # tf -> 23x23 corr DataFrame
    redundancy_rows = []
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in CONCEPTS]
        present = [c for c in cols if c in full.columns]
        if len(present) < len(CONCEPTS):
            missing = [c for c in cols if c not in full.columns]
            print(f"  {tf}: missing {len(missing)} cols, will drop them")
        sub = full[present].copy()
        # drop rows with any NaN
        sub = sub.dropna()
        if len(sub) < args.min_n:
            print(f"  {tf}: only {len(sub)} valid rows — skipping")
            continue
        # rename to short concept names for readability
        rename_map = {feature_column_for(c, tf): c for c in CONCEPTS
                       if feature_column_for(c, tf) in present}
        sub = sub.rename(columns=rename_map)
        cm = sub.corr(method='pearson')
        corr_matrices[tf] = cm
        cm.to_csv(os.path.join(args.output_dir, f'corr_matrix_{tf}.csv'))
        # build redundancy rows for this TF
        for i, c1 in enumerate(cm.columns):
            for j, c2 in enumerate(cm.columns):
                if j <= i:
                    continue
                r = cm.iloc[i, j]
                if pd.isna(r):
                    continue
                redundancy_rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'pearson': float(r),
                })
        print(f"  {tf}: {sub.shape[0]:,} rows, {sub.shape[1]} features")

    # heatmaps
    print(f"\n--- Plotting per-TF heatmaps ---")
    for tf, cm in corr_matrices.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels(cm.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index, fontsize=8)
        for i in range(len(cm.index)):
            for j in range(len(cm.columns)):
                v = cm.iloc[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                          fontsize=6,
                          color='white' if abs(v) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='Pearson corr')
        ax.set_title(f'Within-TF feature correlation @ {tf}')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'heatmap_corr_{tf}.png'),
                     dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"  [saved] {len(corr_matrices)} heatmaps")

    # ---- Hypothesis tests ----
    print(f"\n--- Hypothesis tests ---")
    hyp_rows = []
    for h_name, x_spec, y_spec, predicted, interp in HYPOTHESES:
        x_concept, x_abs = parse_feature_spec(x_spec)
        y_concept, y_abs = parse_feature_spec(y_spec)
        for tf in args.tfs:
            x = get_series(full, x_concept, tf)
            y = get_series(full, y_concept, tf)
            if x is None or y is None:
                continue
            if x_abs:
                x = np.abs(x)
            if y_abs:
                y = np.abs(y)
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() < args.min_n:
                continue
            xv, yv = x[valid], y[valid]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                continue
            pearson = float(np.corrcoef(xv, yv)[0, 1])
            sp_r, _ = spearmanr(xv, yv)
            sp = float(sp_r)

            # measure agreement with predicted sign
            sign_actual = '+' if pearson > 0.05 else ('-' if pearson < -0.05
                                                            else '0')
            if predicted == '?':
                agrees = 'unpredicted'
            else:
                agrees = 'YES' if sign_actual == predicted else 'no'

            hyp_rows.append({
                'hypothesis': h_name,
                'x': x_spec,
                'y': y_spec,
                'tf': tf,
                'predicted': predicted,
                'pearson': pearson,
                'spearman': sp,
                'sign_actual': sign_actual,
                'agrees': agrees,
                'n': int(valid.sum()),
                'interpretation': interp,
            })

    hyp_df = pd.DataFrame(hyp_rows)
    hyp_df.to_csv(os.path.join(args.output_dir, 'hypothesis_tests.csv'),
                    index=False)
    print(f"  [saved] hypothesis_tests.csv ({len(hyp_df)} rows)")

    # Per-hypothesis summary across TFs
    summary_rows = []
    for h_name in [h[0] for h in HYPOTHESES]:
        sub = hyp_df[hyp_df['hypothesis'] == h_name]
        if sub.empty:
            continue
        signs = sub['sign_actual'].values
        n_pos = int((signs == '+').sum())
        n_neg = int((signs == '-').sum())
        n_zero = int((signs == '0').sum())
        # corr stats across TFs
        p_min = float(sub['pearson'].min())
        p_max = float(sub['pearson'].max())
        p_mean = float(sub['pearson'].mean())
        # Sign-flip detection
        unique_signs = set(signs) - {'0'}
        sign_flip = len(unique_signs) > 1
        agree_count = int((sub['agrees'] == 'YES').sum())
        n_tfs = len(sub)
        summary_rows.append({
            'hypothesis': h_name,
            'predicted': sub['predicted'].iloc[0],
            'pearson_min': p_min,
            'pearson_max': p_max,
            'pearson_mean': p_mean,
            'n_pos_tf': n_pos,
            'n_neg_tf': n_neg,
            'n_zero_tf': n_zero,
            'sign_flip_across_tfs': sign_flip,
            'agree_count': agree_count,
            'n_tfs': n_tfs,
            'interpretation': sub['interpretation'].iloc[0],
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(args.output_dir, 'hypothesis_summary.csv'),
                    index=False)
    print(f"  [saved] hypothesis_summary.csv ({len(sum_df)} rows)")

    # Print summary
    print(f"\n  Hypothesis summary (TF range of Pearson r):")
    print(f"    {'hypothesis':>30}  {'pred':>4}  {'r_min':>7}  {'r_max':>7}  "
          f"{'r_mean':>7}  {'+/-/0':>9}  {'agree':>5}  {'flip':>4}")
    for _, r in sum_df.iterrows():
        signs = f"{r['n_pos_tf']}/{r['n_neg_tf']}/{r['n_zero_tf']}"
        print(f"    {r['hypothesis']:>30}  {r['predicted']:>4}  "
              f"{r['pearson_min']:>+7.3f}  {r['pearson_max']:>+7.3f}  "
              f"{r['pearson_mean']:>+7.3f}  {signs:>9}  "
              f"{r['agree_count']:>5}  "
              f"{'YES' if r['sign_flip_across_tfs'] else 'no':>4}")

    # ---- Redundancy: top high-|corr| pairs averaged across TFs ----
    red_df = pd.DataFrame(redundancy_rows)
    if len(red_df) > 0:
        red_df['abs_pearson'] = red_df['pearson'].abs()
        # Average per (c1, c2) pair across TFs
        red_avg = (red_df.groupby(['c1', 'c2'])
                          .agg(mean_abs_corr=('abs_pearson', 'mean'),
                                mean_corr=('pearson', 'mean'),
                                min_corr=('pearson', 'min'),
                                max_corr=('pearson', 'max'),
                                n_tfs=('pearson', 'count'))
                          .reset_index()
                          .sort_values('mean_abs_corr', ascending=False))
        red_avg['sign_flip'] = (np.sign(red_avg['min_corr'])
                                    != np.sign(red_avg['max_corr']))
        red_avg.to_csv(os.path.join(args.output_dir, 'redundancy_top.csv'),
                         index=False)

        sign_flip_pairs = red_avg[red_avg['sign_flip']].sort_values(
            'mean_abs_corr', ascending=False)
        sign_flip_pairs.to_csv(os.path.join(args.output_dir, 'sign_flips.csv'),
                                  index=False)

        print(f"\n  Top 25 most-redundant pairs (mean |r| across TFs):")
        print(f"    {'feature_1':>22}  {'feature_2':>22}  "
              f"{'mean_r':>7}  {'min':>7}  {'max':>7}  {'flip':>4}")
        for _, r in red_avg.head(25).iterrows():
            print(f"    {r['c1']:>22}  {r['c2']:>22}  "
                  f"{r['mean_corr']:>+7.3f}  {r['min_corr']:>+7.3f}  "
                  f"{r['max_corr']:>+7.3f}  "
                  f"{'YES' if r['sign_flip'] else 'no':>4}")

        print(f"\n  Pairs that FLIP sign across TFs ({len(sign_flip_pairs)} pairs):")
        for _, r in sign_flip_pairs.head(15).iterrows():
            print(f"    {r['c1']:>22}  {r['c2']:>22}  "
                  f"min={r['min_corr']:>+6.3f}  max={r['max_corr']:>+6.3f}")

    # ---- SURPRISE SCANNER ----
    # 4 categories:
    #   AGREES_AS_PREDICTED  : in hypothesis list, sign matches, |r|>=0.10
    #   PREDICTED_SIGN_WRONG : in hypothesis list, sign FLIPPED from prediction
    #   PREDICTED_BUT_WEAK   : in hypothesis list, |r|<0.10 (weaker than expected)
    #   UNPREDICTED_HIGH     : NOT in hypothesis list, |r|>=0.30
    print(f"\n--- Surprise scanner ---")

    # Build set of (c1, c2) pairs we explicitly hypothesized about.
    # Note: pair is unordered (c1,c2)==(c2,c1).
    predicted_pairs = {}  # frozenset({c1, c2}) -> (h_name, predicted_sign)
    for h_name, x_spec, y_spec, predicted, _ in HYPOTHESES:
        x_concept, _ = parse_feature_spec(x_spec)
        y_concept, _ = parse_feature_spec(y_spec)
        # We bracket abs() vs raw separately - the corr_matrix uses raw values
        # so for matched pair tracking we use the concept names regardless
        key = frozenset({x_concept, y_concept})
        predicted_pairs[key] = (h_name, predicted, x_spec, y_spec)

    # Walk all per-TF pair correlations from redundancy_rows
    surprise_rows = []
    high_unpredicted_threshold = 0.30
    weak_predicted_threshold = 0.10
    for row in redundancy_rows:
        tf = row['tf']
        c1, c2 = row['c1'], row['c2']
        r = row['pearson']
        key = frozenset({c1, c2})
        if key in predicted_pairs:
            h_name, predicted, x_spec, y_spec = predicted_pairs[key]
            # Only test sign match for predicted +/- (skip "?")
            if predicted in ('+', '-'):
                expected_sign = 1 if predicted == '+' else -1
                actual_sign = 1 if r > 0.05 else (-1 if r < -0.05 else 0)
                if actual_sign == 0:
                    continue  # near-zero, ambiguous
                if actual_sign == expected_sign and abs(r) >= weak_predicted_threshold:
                    cat = 'AGREES_AS_PREDICTED'
                elif actual_sign != expected_sign:
                    cat = 'PREDICTED_SIGN_WRONG'
                else:
                    cat = 'PREDICTED_BUT_WEAK'
                surprise_rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'pearson': r,
                    'category': cat,
                    'predicted_sign': predicted,
                    'hypothesis': h_name,
                })
            elif predicted == '?':
                surprise_rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'pearson': r,
                    'category': 'PREDICTED_UNKNOWN_SIGN',
                    'predicted_sign': predicted,
                    'hypothesis': h_name,
                })
        else:
            if abs(r) >= high_unpredicted_threshold:
                surprise_rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'pearson': r,
                    'category': 'UNPREDICTED_HIGH',
                    'predicted_sign': '',
                    'hypothesis': '',
                })

    surprise_df = pd.DataFrame(surprise_rows)
    surprise_df.to_csv(os.path.join(args.output_dir, 'surprises.csv'),
                         index=False)
    print(f"  [saved] surprises.csv ({len(surprise_df)} entries)")

    # Per-category summary
    cat_counts = surprise_df['category'].value_counts()
    print(f"\n  Surprise category distribution:")
    for c, n in cat_counts.items():
        print(f"    {c:>26}: {n}")

    # Show worst predicted sign-wrongs
    sign_wrongs = surprise_df[surprise_df['category'] == 'PREDICTED_SIGN_WRONG']
    if len(sign_wrongs) > 0:
        print(f"\n  PREDICTED_SIGN_WRONG (my hypothesis was wrong on this pair at this TF):")
        print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'r':>7}  {'pred':>4}  {'hypothesis':>30}")
        for _, r in sign_wrongs.iterrows():
            print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
                  f"{r['pearson']:>+7.3f}  {r['predicted_sign']:>4}  "
                  f"{r['hypothesis']:>30}")

    # Show predicted-but-weak (predicted sign correct but |r| small)
    weaks = surprise_df[surprise_df['category'] == 'PREDICTED_BUT_WEAK']
    if len(weaks) > 0:
        print(f"\n  PREDICTED_BUT_WEAK ({len(weaks)} entries — sign right but corr weaker than expected):")
        for _, r in weaks.head(15).iterrows():
            print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
                  f"{r['pearson']:>+7.3f}  {r['hypothesis']:>30}")

    # Show top unpredicted-high (the genuine surprises — pairs I didn't think about)
    unpred = surprise_df[surprise_df['category'] == 'UNPREDICTED_HIGH'].copy()
    if len(unpred) > 0:
        unpred['abs_r'] = unpred['pearson'].abs()
        print(f"\n  TOP UNPREDICTED_HIGH ({len(unpred)} entries — pair I didn't hypothesize but |r|>={high_unpredicted_threshold}):")
        for _, r in unpred.sort_values('abs_r', ascending=False).head(30).iterrows():
            print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
                  f"{r['pearson']:>+7.3f}")

    # Aggregate UNPREDICTED_HIGH per pair across TFs (consistent surprises)
    if len(unpred) > 0:
        consistent_surprises = (unpred.groupby(['c1', 'c2'])
                                       .agg(n_tf_high=('pearson', 'count'),
                                             mean_r=('pearson', 'mean'),
                                             min_r=('pearson', 'min'),
                                             max_r=('pearson', 'max'))
                                       .reset_index())
        consistent_surprises = consistent_surprises[
            consistent_surprises['n_tf_high'] >= 4]  # at >= 4 TFs (50%)
        consistent_surprises = consistent_surprises.sort_values(
            'mean_r', key=lambda s: s.abs(), ascending=False)
        consistent_surprises.to_csv(
            os.path.join(args.output_dir, 'consistent_unpredicted_surprises.csv'),
            index=False)
        print(f"\n  CONSISTENT UNPREDICTED SURPRISES (pair has |r|>={high_unpredicted_threshold} at >=4 TFs):")
        print(f"    {'c1':>22}  {'c2':>22}  {'n_tfs':>5}  {'mean_r':>7}  {'min':>7}  {'max':>7}")
        for _, r in consistent_surprises.head(20).iterrows():
            print(f"    {r['c1']:>22}  {r['c2']:>22}  {int(r['n_tf_high']):>5}  "
                  f"{r['mean_r']:>+7.3f}  {r['min_r']:>+7.3f}  {r['max_r']:>+7.3f}")

    # ---- Markdown summary ----
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-TF interaction (Layer D1) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {len(CONCEPTS)} v2 features per TF\n\n")
        f.write(f"**TFs:** {args.tfs}\n\n")
        f.write(f"**Split:** {args.split}\n\n")
        f.write(f"## Hypothesis test summary\n\n")
        f.write(sum_df.to_string(index=False))
        f.write(f"\n\n## Top 30 redundant pairs across TFs\n\n")
        if len(red_avg) > 0:
            f.write(red_avg.head(30).to_string(index=False))
        f.write(f"\n\n## Pairs that flip sign by TF\n\n")
        if len(sign_flip_pairs) > 0:
            f.write(sign_flip_pairs.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
