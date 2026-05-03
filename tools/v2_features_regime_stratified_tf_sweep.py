"""
v2_features_regime_stratified_tf_sweep.py — Layer "TF sweep × regime".

User hypothesis: "a chop in 1s is a trend or a flat in the adjacent higher TF".
Stratify the TF sweep by the daily regime_2d label. For each (concept, TF,
regime_2d) cell, compute distribution + correlation metrics. Reveals:

  - How each (concept, TF)'s price relationship CHANGES across the 6 daily
    regimes
  - Whether a TF "looks chop-y" inside one regime but "trend-y" inside another
  - Cross-regime sign flips per concept × TF (specific to each regime context)

For each cell:
  n                           bars in this (regime, concept, tf) cell
  feat_mean, feat_std         distribution within regime
  corr_concurrent_signed      vs (close - vwap_5m)
  corr_lookback_return        vs close[t] - close[t-N]
  corr_forward_return         vs close[t+N] - close[t]
  trend_strength_local        std of feature normalized by regime baseline
                              (proxy for "is this TF carrying directional info
                              within this regime?")
  chop_strength_local         spread between SMOOTH and CHOPPY days within
                              this directional regime (computed only for the
                              direction axes UP_*, DOWN_*, FLAT_*)

Outputs:
  reports/findings/v2_features_regime_tf/
    cell_summary.csv             — long form (concept, tf, regime_2d, metrics)
    pivot_corr_fwd_<regime>.csv  — per-regime concept × TF correlation pivot
    cross_regime_inversions.csv  — concepts/TFs where corr_fwd flips sign across regimes
    multiscale_character.csv     — per regime, per TF: trend_strength vs chop_strength
    summary.md                   — narrative
    heatmap_<regime>.png         — concept × TF heatmap per regime

Usage:
  python tools/v2_features_regime_stratified_tf_sweep.py
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
    FEATURE_NAMES_V2, load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_tf_sweep_eda import (
    feature_column_for, cohen_d, safe_corr, TF_ORDER_SMALL_TO_LARGE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=TF_ORDER_SMALL_TO_LARGE)
    parser.add_argument('--lookback-n', type=int, default=12)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--concept-list', nargs='*', default=None)
    parser.add_argument('--min-cell-n', type=int, default=200)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_regime_tf')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features: TF sweep stratified by daily regime")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  TFs: {args.tfs}")
    print(f"  Min cell n: {args.min_cell_n}")
    print(f"{'='*70}")

    concepts = args.concept_list or FEATURE_NAMES_V2
    concepts = [c for c in concepts if c in FEATURE_NAMES_V2]

    # Load
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

    close = full['close'].values.astype(np.float64)
    n = len(close)

    vwap5_col = f'L2_{args.base_tf}_vwap_w'
    vwap5 = full[vwap5_col].values.astype(np.float64) \
        if vwap5_col in full.columns else close.copy()
    concurrent_signed = close - vwap5

    lookback_ret = np.full(n, np.nan)
    if n > args.lookback_n:
        lookback_ret[args.lookback_n:] = close[args.lookback_n:] - close[:-args.lookback_n]
    forward_ret = np.full(n, np.nan)
    if n > args.forward_n:
        forward_ret[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]

    regimes = full['regime_2d'].values.astype(str)

    # Sweep
    print(f"\n--- Sweeping {len(concepts)} concepts × {len(args.tfs)} TFs × "
          f"{len(REGIME_2D_ORDER)} regimes ---")
    rows = []
    for concept in concepts:
        for tf in args.tfs:
            col = feature_column_for(concept, tf)
            if col not in full.columns:
                continue
            v = full[col].values.astype(np.float64)
            for regime in REGIME_2D_ORDER:
                mask = (regimes == regime) & ~np.isnan(v)
                if mask.sum() < args.min_cell_n:
                    continue
                v_r = v[mask]
                cs_r = concurrent_signed[mask]
                lb_r = lookback_ret[mask]
                fw_r = forward_ret[mask]
                rows.append({
                    'concept': concept,
                    'tf': tf,
                    'tf_idx': args.tfs.index(tf),
                    'regime_2d': regime,
                    'n': int(mask.sum()),
                    'feat_mean': float(np.nanmean(v_r)),
                    'feat_std': float(np.nanstd(v_r)),
                    'corr_concurrent_signed': safe_corr(v_r, cs_r),
                    'corr_lookback_return': safe_corr(v_r, lb_r),
                    'corr_forward_return': safe_corr(v_r, fw_r),
                })

    df = pd.DataFrame(rows)
    cell_path = os.path.join(args.output_dir, 'cell_summary.csv')
    df.to_csv(cell_path, index=False)
    print(f"  [saved] {cell_path} ({len(df)} cells)")

    # Per-regime concept × TF pivot of corr_forward_return
    print(f"\n--- Per-regime concept × TF pivot of corr_forward_return ---")
    for regime in REGIME_2D_ORDER:
        sub = df[df['regime_2d'] == regime]
        if len(sub) == 0:
            continue
        pv = sub.pivot(index='concept', columns='tf', values='corr_forward_return')
        pv = pv.reindex(columns=[t for t in args.tfs if t in pv.columns])
        pv_path = os.path.join(args.output_dir, f'pivot_corr_fwd_{regime}.csv')
        pv.to_csv(pv_path)
        # Heatmap
        if pv.size > 0 and not np.all(np.isnan(pv.values)):
            fig, ax = plt.subplots(figsize=(11, 8))
            vmax = float(np.nanmax(np.abs(pv.values)))
            im = ax.imshow(pv.values, cmap='RdBu_r', aspect='auto',
                            vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(pv.columns)))
            ax.set_xticklabels(pv.columns)
            ax.set_yticks(range(len(pv.index)))
            ax.set_yticklabels(pv.index, fontsize=8)
            for i in range(len(pv.index)):
                for j in range(len(pv.columns)):
                    val = pv.iloc[i, j]
                    if pd.isna(val):
                        continue
                    ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                              fontsize=7,
                              color='white' if abs(val) > vmax / 2 else 'black')
            ax.set_xlabel('Timeframe (small -> large)')
            ax.set_ylabel('Concept')
            ax.set_title(f'corr(feature, forward_return) per (concept, TF) — '
                          f'regime={regime}\nvmax={vmax:.3f}')
            plt.colorbar(im, ax=ax, label='corr_forward_return')
            fig.tight_layout()
            heat_path = os.path.join(args.output_dir, f'heatmap_{regime}.png')
            fig.savefig(heat_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)

    # Cross-regime inversions: per (concept, TF), do the corr_fwd values
    # differ in SIGN across regimes?
    print(f"\n--- Cross-regime inversion analysis ---")
    inv_rows = []
    for concept in concepts:
        for tf in args.tfs:
            sub = df[(df['concept'] == concept) & (df['tf'] == tf)].copy()
            if len(sub) < 2:
                continue
            corrs = sub['corr_forward_return'].dropna().values
            if len(corrs) < 2:
                continue
            signs = np.sign(corrs)
            sign_counts = pd.Series(signs).value_counts()
            n_pos = int(sign_counts.get(1, 0))
            n_neg = int(sign_counts.get(-1, 0))
            inv_rows.append({
                'concept': concept,
                'tf': tf,
                'n_regimes': len(sub),
                'n_pos': n_pos,
                'n_neg': n_neg,
                'has_inversion': bool(n_pos > 0 and n_neg > 0),
                'min_corr': float(corrs.min()),
                'max_corr': float(corrs.max()),
                'corr_range': float(corrs.max() - corrs.min()),
            })
    inv_df = pd.DataFrame(inv_rows).sort_values('corr_range', ascending=False)
    inv_path = os.path.join(args.output_dir, 'cross_regime_inversions.csv')
    inv_df.to_csv(inv_path, index=False)
    print(f"  [saved] {inv_path}")

    print(f"\n  Top 15 (concept, TF) with biggest corr_fwd RANGE across regimes:")
    print(f"    {'concept':>30}  {'tf':>4}  {'min':>8} {'max':>8} {'range':>8} "
          f"{'inv':>4}")
    for _, r in inv_df.head(15).iterrows():
        print(f"    {r['concept']:>30}  {r['tf']:>4}  {r['min_corr']:>+8.3f} "
              f"{r['max_corr']:>+8.3f} {r['corr_range']:>8.3f} "
              f"{'YES' if r['has_inversion'] else 'no':>4}")

    # Multiscale character: per regime, per TF, characterize
    # trend_strength (std of price_velocity_w within regime, normalized)
    # chop_strength (std of swing_noise_w within regime, normalized)
    print(f"\n--- Multiscale character per (regime, TF) ---")
    char_rows = []
    for regime in REGIME_2D_ORDER:
        for tf in args.tfs:
            row = {'regime_2d': regime, 'tf': tf}
            # Trend strength proxy: std of price_velocity_w within regime
            vel_col = feature_column_for('price_velocity_w', tf)
            chop_col = feature_column_for('swing_noise_w', tf)
            range_col = feature_column_for('bar_range', tf)
            mask = (regimes == regime)
            if mask.sum() < args.min_cell_n:
                continue
            if vel_col in full.columns:
                row['vel_w_std'] = float(np.nanstd(full[vel_col].values[mask]))
                row['vel_w_abs_mean'] = float(np.nanmean(np.abs(full[vel_col].values[mask])))
            if chop_col in full.columns:
                row['swing_noise_mean'] = float(np.nanmean(full[chop_col].values[mask]))
            if range_col in full.columns:
                row['bar_range_mean'] = float(np.nanmean(full[range_col].values[mask]))
            row['n'] = int(mask.sum())
            char_rows.append(row)
    char_df = pd.DataFrame(char_rows)
    char_path = os.path.join(args.output_dir, 'multiscale_character.csv')
    char_df.to_csv(char_path, index=False)
    print(f"  [saved] {char_path}")

    # Print: for each daily regime, show per-TF trend / chop character
    if 'vel_w_abs_mean' in char_df.columns:
        print(f"\n  Per regime, mean |price_velocity_w| across TFs (trend strength proxy):")
        pv1 = char_df.pivot(index='regime_2d', columns='tf', values='vel_w_abs_mean')
        pv1 = pv1.reindex(index=[r for r in REGIME_2D_ORDER if r in pv1.index],
                            columns=[t for t in args.tfs if t in pv1.columns])
        print(pv1.round(3).to_string())

    if 'swing_noise_mean' in char_df.columns:
        print(f"\n  Per regime, mean swing_noise_w across TFs (chop strength proxy):")
        pv2 = char_df.pivot(index='regime_2d', columns='tf', values='swing_noise_mean')
        pv2 = pv2.reindex(index=[r for r in REGIME_2D_ORDER if r in pv2.index],
                            columns=[t for t in args.tfs if t in pv2.columns])
        print(pv2.round(3).to_string())

    if 'bar_range_mean' in char_df.columns:
        print(f"\n  Per regime, mean bar_range across TFs (variation proxy):")
        pv3 = char_df.pivot(index='regime_2d', columns='tf', values='bar_range_mean')
        pv3 = pv3.reindex(index=[r for r in REGIME_2D_ORDER if r in pv3.index],
                            columns=[t for t in args.tfs if t in pv3.columns])
        print(pv3.round(3).to_string())

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features TF sweep × regime — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`  **Split:** `{args.split}`  "
                f"**Min cell n:** {args.min_cell_n}\n\n")
        f.write("## Top 30 (concept, TF) by corr_fwd range across regimes\n\n")
        f.write(inv_df.head(30).to_string(index=False))
        f.write("\n\n## Multiscale character\n\n")
        if 'vel_w_abs_mean' in char_df.columns:
            f.write("### Mean |price_velocity_w| per (regime, TF) — trend strength proxy\n\n")
            f.write(pv1.round(3).to_string())
            f.write("\n\n")
        if 'swing_noise_mean' in char_df.columns:
            f.write("### Mean swing_noise_w per (regime, TF) — chop strength proxy\n\n")
            f.write(pv2.round(3).to_string())
            f.write("\n\n")
        if 'bar_range_mean' in char_df.columns:
            f.write("### Mean bar_range per (regime, TF) — variation proxy\n\n")
            f.write(pv3.round(3).to_string())
            f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
