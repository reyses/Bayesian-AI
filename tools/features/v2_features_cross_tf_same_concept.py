"""
v2_features_cross_tf_same_concept.py — Cross-TF Layer 1: same-concept
across TFs.

For each of the 23 concepts, compute the 8x8 correlation matrix across
TFs. Asks: how strongly does (e.g.) price_velocity_w at 5s correlate
with price_velocity_w at 1h? This is the "term structure" of each
concept.

Smooth term structure (e.g. monotonic decay with TF distance) means
the concept is fundamentally one signal observed at different
resolutions. Discontinuous term structure (sudden drops between
adjacent TFs) means short-window vs long-window measurements capture
different things.

Stratified by regime to test whether term structure is regime-stable.

Outputs:
  reports/findings/v2_features_cross_tf_same_concept/
    cross_tf_corr.csv         (concept, tf1, tf2, regime, n, pearson)
    term_structure_<concept>.png  per concept: 8x8 heatmap
    smoothness_ranking.csv    per concept: how smooth is the term
                                structure (mean adj-TF corr - mean far-TF corr)
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
TF_ORDER = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_cross_tf_same_concept')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  Cross-TF same-concept (Layer Cross-TF 1)")
    print(f"  Concepts: {len(CONCEPTS)}  TFs: {len(TF_ORDER)}")
    print(f"  Pairs per concept: C(8,2) = 28")
    print(f"  Total: 23 * 28 = 644 (concept, tf-pair) cells per regime")
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
    for concept in CONCEPTS:
        # Get all TF columns for this concept
        tf_cols = {}
        for tf in TF_ORDER:
            col = feature_column_for(concept, tf)
            if col in full.columns:
                tf_cols[tf] = full[col].values.astype(np.float64)
        if len(tf_cols) < 2:
            continue
        # Pairwise corr across TFs (and overall, no regime stratification)
        for tf1, tf2 in itertools.combinations(tf_cols.keys(), 2):
            x = tf_cols[tf1]
            y = tf_cols[tf2]
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() < args.min_cell_n:
                continue
            xv, yv = x[valid], y[valid]
            if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                continue
            r = float(np.corrcoef(xv, yv)[0, 1])
            rows.append({
                'concept': concept,
                'tf1': tf1,
                'tf2': tf2,
                'regime_2d': 'ALL',
                'n': int(valid.sum()),
                'pearson': r,
            })
            # Stratified by regime
            for regime in REGIME_2D_ORDER:
                rmask = (regimes == regime)
                m = rmask & valid
                if m.sum() < args.min_cell_n:
                    continue
                xv, yv = x[m], y[m]
                if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                    continue
                r = float(np.corrcoef(xv, yv)[0, 1])
                rows.append({
                    'concept': concept,
                    'tf1': tf1, 'tf2': tf2,
                    'regime_2d': regime,
                    'n': int(m.sum()),
                    'pearson': r,
                })
        print(f"  {concept}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'cross_tf_corr.csv'), index=False)
    print(f"\n  [saved] cross_tf_corr.csv ({len(df)} rows)")

    # ALL-regime view: build 8x8 matrix per concept
    plotted = 0
    smoothness_rows = []
    for concept in CONCEPTS:
        sub = df[(df['concept'] == concept) & (df['regime_2d'] == 'ALL')]
        if sub.empty:
            continue
        # Build symmetric 8x8 matrix
        present_tfs = sorted(set(sub['tf1']) | set(sub['tf2']),
                                key=lambda t: TF_ORDER.index(t))
        mat = pd.DataFrame(np.eye(len(present_tfs)),
                              index=present_tfs, columns=present_tfs)
        for _, r in sub.iterrows():
            mat.loc[r['tf1'], r['tf2']] = r['pearson']
            mat.loc[r['tf2'], r['tf1']] = r['pearson']
        # Plot
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                v = mat.iloc[i, j]
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                          fontsize=8,
                          color='white' if abs(v) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='Pearson')
        ax.set_title(f'{concept} — cross-TF term structure')
        fig.tight_layout()
        png = os.path.join(args.output_dir, f'term_structure_{concept}.png')
        fig.savefig(png, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plotted += 1

        # Smoothness metric: mean adjacent-TF corr vs mean far-TF corr
        adj_corrs = []
        far_corrs = []
        for i in range(len(present_tfs)):
            for j in range(i + 1, len(present_tfs)):
                d = j - i
                if d == 1:
                    adj_corrs.append(mat.iloc[i, j])
                elif d >= 4:
                    far_corrs.append(mat.iloc[i, j])
        adj_mean = float(np.mean(adj_corrs)) if adj_corrs else float('nan')
        far_mean = float(np.mean(far_corrs)) if far_corrs else float('nan')
        # Sign flips
        all_offdiag = []
        for i in range(len(present_tfs)):
            for j in range(i + 1, len(present_tfs)):
                all_offdiag.append(mat.iloc[i, j])
        n_pos = int(sum(1 for c in all_offdiag if c > 0.05))
        n_neg = int(sum(1 for c in all_offdiag if c < -0.05))
        n_zero = int(sum(1 for c in all_offdiag if abs(c) <= 0.05))
        smoothness_rows.append({
            'concept': concept,
            'mean_adjacent_TF_corr': adj_mean,
            'mean_far_TF_corr': far_mean,
            'decay': adj_mean - far_mean,
            'n_pos_offdiag': n_pos,
            'n_neg_offdiag': n_neg,
            'n_zero_offdiag': n_zero,
            'has_sign_flips': n_pos > 0 and n_neg > 0,
        })

    sm_df = pd.DataFrame(smoothness_rows).sort_values('decay',
                                                                ascending=False)
    sm_df.to_csv(os.path.join(args.output_dir, 'smoothness_ranking.csv'),
                    index=False)
    print(f"  [saved] smoothness_ranking.csv ({len(sm_df)} concepts)")
    print(f"  [saved] {plotted} term-structure heatmaps")

    print(f"\n  Concept term-structure smoothness (adjacent-TF vs far-TF corr):")
    print(f"    {'concept':>22}  {'adj_corr':>8}  {'far_corr':>8}  {'decay':>6}  {'+':>3}  {'-':>3}  {'flips':>5}")
    for _, r in sm_df.iterrows():
        print(f"    {r['concept']:>22}  {r['mean_adjacent_TF_corr']:>+8.3f}  "
              f"{r['mean_far_TF_corr']:>+8.3f}  {r['decay']:>+6.3f}  "
              f"{int(r['n_pos_offdiag']):>3}  {int(r['n_neg_offdiag']):>3}  "
              f"{'YES' if r['has_sign_flips'] else 'no':>5}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Cross-TF same-concept (Layer Cross-TF 1) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"## Term-structure smoothness ranking\n\n")
        f.write(sm_df.to_string(index=False))
        f.write(f"\n\n**Decay** = mean_adjacent_TF_corr - mean_far_TF_corr.\n")
        f.write(f"High decay = smooth term structure (each TF only relates "
                f"to its neighbors).\n")
        f.write(f"Low decay = rigid (the concept is consistent across all "
                f"TF distances).\n")
        f.write(f"Has_sign_flips = some TF pairs have opposite-sign correlation "
                f"to others — TF inversions exist.\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
