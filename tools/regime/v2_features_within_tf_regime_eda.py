"""
v2_features_within_tf_regime_eda.py — Layer D2: within-TF feature x
feature interactions stratified by regime_2d (no price target).

Mirrors the regime-stratified TF sweep on the price track. Asks:
does corr(X, Y) at TF T change between regimes? If yes, the
relationship is regime-conditional. If no, it's structural.

Output structure:
  - For each (concept_pair, TF, regime): corr value + n
  - REGIME SIGN-FLIP pairs: same (pair, TF) flips sign across regimes
  - REGIME DECOUPLING pairs: |corr| changes most across regimes (e.g.
    +0.5 in UP_SMOOTH, +0.05 in DOWN_CHOPPY -> coupling weakens)
  - SIDED REGIME LOCKING: the few pairs whose corr is regime-locked
    (only one regime has non-zero corr)

Outputs:
  reports/findings/v2_features_within_tf_regime/
    pair_summary.csv      (concept1, concept2, tf, regime, n, pearson)
    pivot_<tf>.csv        per-TF: pair x regime pivot of corr
    regime_sign_flips.csv pairs whose corr SIGN flips across regimes (within
                           same TF)
    regime_decoupling.csv pairs whose |corr| changes most across regimes
    summary.md
    plot_<tf>_<concept_pair>.png  bar chart of corr by regime, per top pair
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


# Same canonical 23 concepts as D1
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

DEFAULT_TFS = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--top-plots', type=int, default=15)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features within-TF interaction x regime (Layer D2)")
    print(f"  TFs: {args.tfs}")
    print(f"  Concepts: {len(CONCEPTS)}")
    print(f"  Pairs per TF: C(23,2) = {23*22//2}")
    print(f"  Total cells: {len(args.tfs)} TFs * 253 pairs * 6 regimes = "
          f"{len(args.tfs) * 253 * 6}")
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

    # ---- Per-(TF, regime) correlation matrices ----
    print(f"\n--- Computing per-(TF, regime) correlation matrices ---")
    rows = []
    for tf in args.tfs:
        cols = [feature_column_for(c, tf) for c in CONCEPTS]
        present_concepts = [c for c, col in zip(CONCEPTS, cols)
                                 if col in full.columns]
        present_cols = [feature_column_for(c, tf) for c in present_concepts]
        if len(present_concepts) < 5:
            print(f"  {tf}: too few features present, skipping")
            continue

        for regime in REGIME_2D_ORDER:
            r_mask = (regimes == regime)
            if r_mask.sum() < args.min_cell_n:
                continue
            sub = full.loc[r_mask, present_cols].copy()
            sub = sub.dropna()
            if len(sub) < args.min_cell_n:
                continue
            # rename for readability
            rename_map = {feature_column_for(c, tf): c for c in present_concepts}
            sub = sub.rename(columns=rename_map)
            cm = sub.corr(method='pearson')
            for i, c1 in enumerate(cm.columns):
                for j, c2 in enumerate(cm.columns):
                    if j <= i:
                        continue
                    r = cm.iloc[i, j]
                    if pd.isna(r):
                        continue
                    rows.append({
                        'tf': tf,
                        'c1': c1,
                        'c2': c2,
                        'regime_2d': regime,
                        'n': len(sub),
                        'pearson': float(r),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'pair_summary.csv'), index=False)
    print(f"  [saved] pair_summary.csv ({len(df)} rows)")

    # ---- Per-TF pivot: pair x regime ----
    print(f"\n--- Per-TF pivots ---")
    for tf in args.tfs:
        sub = df[df['tf'] == tf]
        if sub.empty:
            continue
        sub = sub.copy()
        sub['pair'] = sub['c1'] + '__' + sub['c2']
        pv = sub.pivot(index='pair', columns='regime_2d', values='pearson')
        pv = pv.reindex(columns=[r for r in REGIME_2D_ORDER if r in pv.columns])
        pv.to_csv(os.path.join(args.output_dir, f'pivot_{tf}.csv'))
    print(f"  [saved] {len(args.tfs)} per-TF pivots")

    # ---- REGIME SIGN-FLIP pairs (same pair-TF, opposite signs across regimes) ----
    df['pair'] = df['c1'] + '__' + df['c2']
    grouped = df.groupby(['tf', 'pair'])
    flip_rows = []
    decoupling_rows = []
    for (tf, pair), g in grouped:
        if len(g) < 2:
            continue
        rs = g['pearson'].values
        # require at least one + and one - (and both above 0.05 to avoid noise)
        pos = rs[rs > 0.05]
        neg = rs[rs < -0.05]
        regime_max = g.loc[g['pearson'].idxmax()]
        regime_min = g.loc[g['pearson'].idxmin()]
        if len(pos) > 0 and len(neg) > 0:
            flip_rows.append({
                'tf': tf,
                'pair': pair,
                'c1': g['c1'].iloc[0],
                'c2': g['c2'].iloc[0],
                'r_min': float(rs.min()),
                'r_max': float(rs.max()),
                'regime_min': regime_min['regime_2d'],
                'regime_max': regime_max['regime_2d'],
                'n_regimes': len(g),
                'spread': float(rs.max() - rs.min()),
            })
        # Decoupling: |corr| changes a lot
        if len(rs) >= 2:
            spread_abs = float(np.abs(rs).max() - np.abs(rs).min())
            decoupling_rows.append({
                'tf': tf,
                'pair': pair,
                'c1': g['c1'].iloc[0],
                'c2': g['c2'].iloc[0],
                'abs_min': float(np.abs(rs).min()),
                'abs_max': float(np.abs(rs).max()),
                'spread_abs': spread_abs,
                'r_min': float(rs.min()),
                'r_max': float(rs.max()),
                'sign_flip': bool(len(pos) > 0 and len(neg) > 0),
            })

    flips_df = pd.DataFrame(flip_rows).sort_values('spread', ascending=False)
    flips_df.to_csv(os.path.join(args.output_dir, 'regime_sign_flips.csv'),
                       index=False)
    print(f"\n  Regime SIGN flips: {len(flips_df)} pairs (within same TF, "
          f"corr sign changes across regimes)")

    decoup_df = pd.DataFrame(decoupling_rows).sort_values(
        'spread_abs', ascending=False)
    decoup_df.to_csv(os.path.join(args.output_dir, 'regime_decoupling.csv'),
                       index=False)

    if len(flips_df) > 0:
        print(f"\n  Top 30 sign-flip pairs:")
        print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  "
              f"{'r_min':>7}  {'regime_min':>14}  {'r_max':>7}  {'regime_max':>14}")
        for _, r in flips_df.head(30).iterrows():
            print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
                  f"{r['r_min']:>+7.3f}  {r['regime_min']:>14}  "
                  f"{r['r_max']:>+7.3f}  {r['regime_max']:>14}")

    print(f"\n  Top 30 regime-decoupling pairs (largest |corr| spread):")
    print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  "
          f"{'absmin':>6}  {'absmax':>6}  {'r_min':>7}  {'r_max':>7}  flip")
    for _, r in decoup_df.head(30).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['abs_min']:>6.3f}  {r['abs_max']:>6.3f}  "
              f"{r['r_min']:>+7.3f}  {r['r_max']:>+7.3f}  "
              f"{'YES' if r['sign_flip'] else 'no'}")

    # Stable pairs: low spread → universal/structural redundancies
    stable_df = decoup_df[decoup_df['spread_abs'] < 0.05].copy()
    stable_df = stable_df.sort_values('abs_max', ascending=False)
    stable_df.to_csv(os.path.join(args.output_dir, 'stable_pairs.csv'),
                       index=False)
    print(f"\n  Top 20 STABLE high-corr pairs (corr unchanged across regimes — structural):")
    for _, r in stable_df.head(20).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"absmax={r['abs_max']:>6.3f} spread={r['spread_abs']:>5.3f}")

    # ---- Plot top regime-flip pairs (bar chart per regime) ----
    plotted = 0
    if len(flips_df) > 0:
        for _, fr in flips_df.head(args.top_plots).iterrows():
            tf = fr['tf']
            pair = fr['pair']
            sub = df[(df['tf'] == tf) & (df['pair'] == pair)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            sub_sorted = sub.copy()
            sub_sorted['regime_2d'] = pd.Categorical(
                sub_sorted['regime_2d'], categories=REGIME_2D_ORDER, ordered=True)
            sub_sorted = sub_sorted.sort_values('regime_2d')
            colors = ['red' if x < 0 else 'blue' for x in sub_sorted['pearson']]
            ax.bar(range(len(sub_sorted)), sub_sorted['pearson'], color=colors,
                     alpha=0.85)
            ax.axhline(0, color='black', alpha=0.4)
            ax.set_xticks(range(len(sub_sorted)))
            ax.set_xticklabels(sub_sorted['regime_2d'], rotation=30, ha='right')
            ax.set_ylabel('Pearson corr')
            ax.set_title(f'{fr["c1"]} vs {fr["c2"]} @ {tf} — by regime')
            ax.grid(alpha=0.3, axis='y')
            for i, v in enumerate(sub_sorted['pearson'].values):
                ax.text(i, v + (0.02 if v > 0 else -0.04),
                          f'{v:+.2f}', ha='center', fontsize=9)
            fig.tight_layout()
            png_path = os.path.join(args.output_dir,
                                       f'plot_{tf}_{pair}.png')
            fig.savefig(png_path, dpi=120, bbox_inches='tight',
                          facecolor='white')
            plt.close(fig)
            plotted += 1
    print(f"\n  [saved] {plotted} regime-flip plots")

    # ---- Markdown summary ----
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-TF interaction x regime (Layer D2) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Pairs per TF**: 253 (C(23,2))  **TFs**: {len(args.tfs)}  "
                f"**Regimes**: 6  **Total cells**: {len(df)}\n\n")
        f.write(f"## Sign-flip pairs ({len(flips_df)})\n\n")
        f.write(f"Pairs whose corr changes SIGN across regimes within the "
                f"same TF. Strongest regime-conditional couplings.\n\n")
        if len(flips_df) > 0:
            f.write(flips_df.head(50).to_string(index=False))
        f.write(f"\n\n## Top decoupling pairs (largest |corr| spread)\n\n")
        f.write(decoup_df.head(50).to_string(index=False))
        f.write(f"\n\n## Stable high-corr pairs (universal across regimes)\n\n")
        f.write(stable_df.head(30).to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
