"""
v2_features_cross_tf_anchor_pair_drill.py — Cross-TF Layer 2:
cross-concept cross-TF correlations for D9 OOS-confirmed anchor pairs.

For each anchor (X concept, Y concept) drawn from D9-survivor pairs:
  - Compute the FULL 8x8 cross-TF matrix of corr(X_at_TFi, Y_at_TFj)
  - Stratify by regime (does the within-TF regime sign-flip extend
    to cross-TF cells?)
  - Find OFF-DIAGONAL cells where cross-TF gives stronger and/or
    differently-signed correlation than within-TF (diagonal)

If `bar_range_5s x body_1h` has higher |corr| than `bar_range_1h x
body_1h`, cross-TF gives additional information beyond within-TF.

Stratification: per regime, 8x8 matrices.

Outputs:
  reports/findings/v2_features_cross_tf_anchor_pair/
    cross_tf_pair.csv          (X, Y, tfX, tfY, regime, n, pearson)
    diagonal_vs_offdiag.csv    per (X, Y, regime): max diagonal vs max
                                 off-diagonal corr, and lift
    plot_<X>__<Y>__<regime>.png  per (X, Y, regime) 8x8 heatmap
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


TF_ORDER = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']

# Top anchor pairs from D9 OOS-confirmed (pulled from
# v2_features_within_tf_oos/d7_pair_oos.csv)
ANCHOR_PAIRS = [
    ('price_velocity_w', 'price_sigma_w'),
    ('bar_range', 'body'),
    ('price_velocity_1b', 'bar_range'),
    ('price_velocity_w', 'swing_noise_w'),
    ('price_velocity_w', 'SE_low_w'),
    ('price_velocity_1b', 'reversion_prob_w'),
    ('body', 'reversion_prob_w'),
    ('price_velocity_w', 'vol_mean_w'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_cross_tf_anchor_pair')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  Cross-TF anchor-pair drill (Layer Cross-TF 2)")
    print(f"  Anchor pairs: {len(ANCHOR_PAIRS)}")
    print(f"  TFs: {len(TF_ORDER)}")
    print(f"  Per pair: 8*8=64 cross-TF cells * {len(REGIME_2D_ORDER)} regimes = "
          f"{64*len(REGIME_2D_ORDER)} cells")
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
    print(f"\n--- Sweeping anchor pairs ---")
    for X, Y in ANCHOR_PAIRS:
        # Pre-extract X at each TF and Y at each TF
        x_tfs = {}
        y_tfs = {}
        for tf in TF_ORDER:
            cx = feature_column_for(X, tf)
            cy = feature_column_for(Y, tf)
            if cx in full.columns:
                x_tfs[tf] = full[cx].values.astype(np.float64)
            if cy in full.columns:
                y_tfs[tf] = full[cy].values.astype(np.float64)
        if len(x_tfs) < 2 or len(y_tfs) < 2:
            continue

        # 8x8 cross-TF, stratified by regime AND aggregated (ALL)
        for tfX in x_tfs:
            for tfY in y_tfs:
                x = x_tfs[tfX]
                y = y_tfs[tfY]

                def calc(mask):
                    m = mask & ~np.isnan(x) & ~np.isnan(y)
                    if m.sum() < args.min_cell_n:
                        return float('nan'), 0
                    xv, yv = x[m], y[m]
                    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                        return float('nan'), int(m.sum())
                    return float(np.corrcoef(xv, yv)[0, 1]), int(m.sum())

                # ALL regimes
                r_all, n_all = calc(np.ones(len(x), dtype=bool))
                rows.append({
                    'X': X, 'Y': Y,
                    'tfX': tfX, 'tfY': tfY,
                    'regime_2d': 'ALL', 'n': n_all, 'pearson': r_all,
                })
                # per regime
                for regime in REGIME_2D_ORDER:
                    rmask = (regimes == regime)
                    r, n = calc(rmask)
                    if np.isnan(r):
                        continue
                    rows.append({
                        'X': X, 'Y': Y,
                        'tfX': tfX, 'tfY': tfY,
                        'regime_2d': regime, 'n': n, 'pearson': r,
                    })
        print(f"  {X} x {Y}: done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'cross_tf_pair.csv'), index=False)
    print(f"\n  [saved] cross_tf_pair.csv ({len(df)} rows)")

    # ---- Diagonal vs off-diagonal analysis ----
    diag_rows = []
    for (X, Y, regime), g in df.groupby(['X', 'Y', 'regime_2d']):
        diag = g[g['tfX'] == g['tfY']].copy()
        offdiag = g[g['tfX'] != g['tfY']].copy()
        if diag.empty or offdiag.empty:
            continue
        # max abs corr in diagonal
        diag['abs_r'] = diag['pearson'].abs()
        offdiag['abs_r'] = offdiag['pearson'].abs()
        max_d = diag.loc[diag['abs_r'].idxmax()]
        max_o = offdiag.loc[offdiag['abs_r'].idxmax()]
        diag_rows.append({
            'X': X, 'Y': Y, 'regime_2d': regime,
            'max_diag_tf': max_d['tfX'],
            'max_diag_r': max_d['pearson'],
            'max_offdiag_tfX': max_o['tfX'],
            'max_offdiag_tfY': max_o['tfY'],
            'max_offdiag_r': max_o['pearson'],
            'offdiag_lift': abs(max_o['pearson']) - abs(max_d['pearson']),
            'sign_match': np.sign(max_o['pearson']) == np.sign(max_d['pearson']),
        })
    diag_df = pd.DataFrame(diag_rows).sort_values(
        'offdiag_lift', ascending=False)
    diag_df.to_csv(os.path.join(args.output_dir, 'diagonal_vs_offdiag.csv'),
                      index=False)
    print(f"  [saved] diagonal_vs_offdiag.csv ({len(diag_df)} rows)")

    print(f"\n  Top off-diagonal lifts (cross-TF beats within-TF):")
    print(f"    {'X':>22}  {'Y':>22}  {'regime':>14}  "
          f"{'best_diag':>10}  {'r':>7}  {'best_offdiag':>14}  {'r':>7}  {'lift':>5}  match")
    for _, r in diag_df.head(25).iterrows():
        diag_lab = f"({r['max_diag_tf']},{r['max_diag_tf']})"
        off_lab = f"({r['max_offdiag_tfX']},{r['max_offdiag_tfY']})"
        print(f"    {r['X']:>22}  {r['Y']:>22}  {r['regime_2d']:>14}  "
              f"{diag_lab:>10}  {r['max_diag_r']:>+7.3f}  "
              f"{off_lab:>14}  {r['max_offdiag_r']:>+7.3f}  "
              f"{r['offdiag_lift']:>+5.3f}  "
              f"{'YES' if r['sign_match'] else 'no'}")

    # cross-TF beats within-TF count
    n_lift = int((diag_df['offdiag_lift'] > 0).sum())
    print(f"\n  Off-diagonal exceeds best diagonal: {n_lift} of {len(diag_df)} "
          f"cells ({100*n_lift/max(len(diag_df),1):.1f}%)")
    n_strong = int((diag_df['offdiag_lift'] > 0.05).sum())
    print(f"  Lift > 0.05: {n_strong} of {len(diag_df)} "
          f"({100*n_strong/max(len(diag_df),1):.1f}%)")

    # ---- Plot 8x8 heatmaps for each (X, Y, regime), focus on UP_SMOOTH and DOWN_SMOOTH ----
    plotted = 0
    for (X, Y) in ANCHOR_PAIRS:
        for regime in ['ALL', 'UP_SMOOTH', 'DOWN_SMOOTH']:
            sub = df[(df['X'] == X) & (df['Y'] == Y) & (df['regime_2d'] == regime)]
            if sub.empty:
                continue
            present_tfX = sorted(set(sub['tfX']),
                                       key=lambda t: TF_ORDER.index(t))
            present_tfY = sorted(set(sub['tfY']),
                                       key=lambda t: TF_ORDER.index(t))
            mat = pd.DataFrame(np.full((len(present_tfX), len(present_tfY)),
                                              np.nan),
                                  index=present_tfX, columns=present_tfY)
            for _, r in sub.iterrows():
                mat.loc[r['tfX'], r['tfY']] = r['pearson']
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1, vmax=1,
                            aspect='auto')
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns)
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index)
            ax.set_xlabel(f'tfY ({Y})')
            ax.set_ylabel(f'tfX ({X})')
            for i in range(len(mat.index)):
                for j in range(len(mat.columns)):
                    v = mat.iloc[i, j]
                    if pd.isna(v):
                        continue
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                              fontsize=8,
                              color='white' if abs(v) > 0.5 else 'black')
            plt.colorbar(im, ax=ax, label='Pearson')
            ax.set_title(f'{X} x {Y} — cross-TF, regime={regime}')
            fig.tight_layout()
            png = os.path.join(args.output_dir, f'plot_{X}__{Y}__{regime}.png')
            fig.savefig(png, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plotted += 1
    print(f"\n  [saved] {plotted} cross-TF heatmaps")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Cross-TF anchor-pair drill (Layer Cross-TF 2) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"Anchor pairs: {ANCHOR_PAIRS}\n\n")
        f.write(f"## Diagonal vs off-diagonal — top lifts\n\n")
        f.write(diag_df.head(50).to_string(index=False))
        f.write(f"\n\nOff-diagonal exceeds diagonal: {n_lift}/{len(diag_df)} "
                f"({100*n_lift/max(len(diag_df),1):.1f}%); strong (>0.05): "
                f"{n_strong} ({100*n_strong/max(len(diag_df),1):.1f}%).\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
