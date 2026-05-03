"""
v2_features_tf_sweep_eda.py — TF-progressive EDA pre-amble.

Walks each TF (5s, 15s, 1m, 5m, 15m, 1h, 4h, 1D) and characterizes how
EACH of the 23 per-TF concepts relates to price/regime. Then surfaces:

  - which concepts hold consistent relationships across TFs
  - which invert at some TF threshold (sign flip)
  - which only carry signal at specific TFs

For each (concept, tf), computes:
  mean, std, p10, p90       distribution stats
  cohen_d_up_vs_down         Cohen d for UP_SMOOTH vs DOWN_SMOOTH (regime separating power)
  cohen_d_smooth_vs_choppy   Cohen d for *_SMOOTH vs *_CHOPPY (variation separating power)
  corr_concurrent_signed     Pearson r vs (close - vwap_5m)
  corr_concurrent_abs        vs |close - vwap_5m|
  corr_lookback_return       vs close[t] - close[t-N]
  corr_forward_return        vs close[t+N] - close[t]

Output: long-form CSV (rows = concept × tf) + wide pivots showing each
metric across TFs, plus an "inversion" finding: concepts whose
cohen_d_up_vs_down flips sign across the TF axis (= relationship
character changes with timescale).

Usage:
  python tools/v2_features_tf_sweep_eda.py
  python tools/v2_features_tf_sweep_eda.py --concept-list price_velocity_w body
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
    TF_HIERARCHY_V2, FEATURE_NAMES_V2,
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels


DEFAULT_BASE_TF = '5m'
DEFAULT_LOOKBACK_N = 12
DEFAULT_FORWARD_N = 12
TF_ORDER_SMALL_TO_LARGE = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']


def feature_column_for(concept: str, tf: str) -> str:
    """Construct the v2 column name for a (concept, tf) given FEATURE_NAMES_V2 layout.

    Mirrors the lookup logic in features_v2.py.
    """
    if concept.endswith('_1b') or concept in ('bar_range', 'body'):
        return f'L1_{tf}_{concept}'
    if concept.endswith('_w'):
        # Could be L2 or L3. L3: z_se_w, z_high_w, z_low_w, SE_high_w, SE_low_w,
        #                     hurst_w, reversion_prob_w, swing_noise_w
        l3_set = {'z_se_w', 'z_high_w', 'z_low_w', 'SE_high_w', 'SE_low_w',
                   'hurst_w', 'reversion_prob_w', 'swing_noise_w'}
        if concept in l3_set:
            return f'L3_{tf}_{concept}'
        return f'L2_{tf}_{concept}'
    return concept  # L0 or unknown


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    s_a, s_b = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * s_a ** 2 + (len(b) - 1) * s_b ** 2) /
                       (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 30:
        return float('nan')
    a = a[mask]; b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=TF_ORDER_SMALL_TO_LARGE)
    parser.add_argument('--lookback-n', type=int, default=DEFAULT_LOOKBACK_N)
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--concept-list', nargs='*', default=None,
                        help='Restrict to these concepts; default = all 23')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_tf_sweep')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features TF-Sweep EDA")
    print(f"  Base TF (anchor): {args.base_tf}")
    print(f"  TFs to sweep: {args.tfs}")
    print(f"  Split: {args.split}")
    print(f"{'='*70}")

    # Pick concepts (default = all 23 per-TF concepts)
    concepts = args.concept_list or FEATURE_NAMES_V2
    concepts = [c for c in concepts if c in FEATURE_NAMES_V2]
    print(f"\n  Concepts: {len(concepts)}")
    for c in concepts:
        print(f"    {c}")

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
    print(f"  {args.base_tf}: {len(base_df):,} bars")

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'direction_axis', 'variation_axis', 'regime_2d', 'split']],
        on='date', how='inner')
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    print(f"  v2 features: {len(features_5s):,} 5s rows")
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    close = full['close'].values.astype(np.float64)
    n = len(close)

    # Anchor: dislocation vs vwap_5m
    vwap5_col = f'L2_{args.base_tf}_vwap_w'
    vwap5 = full[vwap5_col].values.astype(np.float64) \
        if vwap5_col in full.columns else close.copy()
    concurrent_signed = close - vwap5
    concurrent_abs = np.abs(concurrent_signed)

    # Returns
    lookback_ret = np.full(n, np.nan)
    if n > args.lookback_n:
        lookback_ret[args.lookback_n:] = close[args.lookback_n:] - close[:-args.lookback_n]
    forward_ret = np.full(n, np.nan)
    if n > args.forward_n:
        forward_ret[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]

    # Regime masks
    is_up_smooth = (full['regime_2d'].values == 'UP_SMOOTH')
    is_dn_smooth = (full['regime_2d'].values == 'DOWN_SMOOTH')
    is_smooth = (full['variation_axis'].values == 'SMOOTH')
    is_choppy = (full['variation_axis'].values == 'CHOPPY')

    # Sweep: for each (concept, tf), compute metrics
    print(f"\n--- Sweeping {len(concepts)} concepts × {len(args.tfs)} TFs ---")
    rows = []
    for concept in concepts:
        for tf in args.tfs:
            col = feature_column_for(concept, tf)
            if col not in full.columns:
                continue
            v = full[col].values.astype(np.float64)
            valid = ~np.isnan(v)
            if valid.sum() < 200:
                continue
            row = {
                'concept': concept,
                'tf': tf,
                'tf_idx': args.tfs.index(tf),
                'n': int(valid.sum()),
                'mean': float(np.nanmean(v)),
                'std': float(np.nanstd(v)),
                'p10': float(np.nanpercentile(v, 10)),
                'p90': float(np.nanpercentile(v, 90)),
                'cohen_d_up_vs_down': cohen_d(v[is_up_smooth], v[is_dn_smooth]),
                'cohen_d_smooth_vs_choppy': cohen_d(v[is_smooth], v[is_choppy]),
                'corr_concurrent_signed': safe_corr(v, concurrent_signed),
                'corr_concurrent_abs': safe_corr(v, concurrent_abs),
                'corr_lookback_return': safe_corr(v, lookback_ret),
                'corr_forward_return': safe_corr(v, forward_ret),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    long_path = os.path.join(args.output_dir, 'concept_by_tf_long.csv')
    df.to_csv(long_path, index=False)
    print(f"  [saved] {long_path} ({len(df)} rows)")

    # Wide pivots: concept (rows) × tf (cols), one matrix per metric
    metrics = ['cohen_d_up_vs_down', 'cohen_d_smooth_vs_choppy',
                'corr_concurrent_signed', 'corr_concurrent_abs',
                'corr_lookback_return', 'corr_forward_return',
                'std']
    pivots = {}
    for m in metrics:
        pv = df.pivot(index='concept', columns='tf', values=m)
        # Ensure column order = TF order
        pv = pv.reindex(columns=[t for t in args.tfs if t in pv.columns])
        pivots[m] = pv
        wide_path = os.path.join(args.output_dir, f'pivot_{m}.csv')
        pv.to_csv(wide_path)
        print(f"  [saved] {wide_path}")

    # Inversion analysis: per concept, does cohen_d_up_vs_down flip sign across TFs?
    print(f"\n--- Inversion analysis (cohen_d_up_vs_down across TFs) ---")
    inv_rows = []
    for concept in concepts:
        sub = df[df['concept'] == concept].sort_values('tf_idx')
        if len(sub) < 2:
            continue
        signs = np.sign(sub['cohen_d_up_vs_down'].values)
        signs = signs[~np.isnan(signs)]
        if len(signs) < 2:
            continue
        # Count sign changes
        flips = int((signs[:-1] * signs[1:] < 0).sum())
        max_d = float(np.nanmax(np.abs(sub['cohen_d_up_vs_down'].values)))
        # Pattern: monotonic up, monotonic down, or zigzag
        d_vals = sub['cohen_d_up_vs_down'].values
        d_increasing = bool(np.all(np.diff(d_vals[~np.isnan(d_vals)]) > 0))
        d_decreasing = bool(np.all(np.diff(d_vals[~np.isnan(d_vals)]) < 0))
        inv_rows.append({
            'concept': concept,
            'tfs_present': len(sub),
            'sign_flips': flips,
            'max_abs_d': max_d,
            'monotone_increasing': d_increasing,
            'monotone_decreasing': d_decreasing,
            'd_5s': sub[sub['tf'] == '5s']['cohen_d_up_vs_down'].values[0]
                if (sub['tf'] == '5s').any() else float('nan'),
            'd_5m': sub[sub['tf'] == '5m']['cohen_d_up_vs_down'].values[0]
                if (sub['tf'] == '5m').any() else float('nan'),
            'd_1h': sub[sub['tf'] == '1h']['cohen_d_up_vs_down'].values[0]
                if (sub['tf'] == '1h').any() else float('nan'),
            'd_1D': sub[sub['tf'] == '1D']['cohen_d_up_vs_down'].values[0]
                if (sub['tf'] == '1D').any() else float('nan'),
        })

    inv_df = pd.DataFrame(inv_rows).sort_values('sign_flips', ascending=False)
    inv_path = os.path.join(args.output_dir, 'inversion_summary.csv')
    inv_df.to_csv(inv_path, index=False)
    print(f"  [saved] {inv_path}")

    print(f"\n  Concepts that FLIP sign across TFs (regime relationship inverts):")
    for _, r in inv_df[inv_df['sign_flips'] > 0].head(10).iterrows():
        print(f"    {r['concept']:>30}  flips={int(r['sign_flips'])}  "
              f"max|d|={r['max_abs_d']:.2f}  "
              f"5s={r['d_5s']:+.2f}  5m={r['d_5m']:+.2f}  "
              f"1h={r['d_1h']:+.2f}  1D={r['d_1D']:+.2f}")

    print(f"\n  Concepts that hold consistently across TFs (no sign flip):")
    consistent = inv_df[inv_df['sign_flips'] == 0].sort_values('max_abs_d', ascending=False)
    for _, r in consistent.head(10).iterrows():
        print(f"    {r['concept']:>30}  max|d|={r['max_abs_d']:.2f}  "
              f"5s={r['d_5s']:+.2f}  5m={r['d_5m']:+.2f}  "
              f"1h={r['d_1h']:+.2f}  1D={r['d_1D']:+.2f}")

    # Visualization: heatmap of cohen_d_up_vs_down per (concept, tf)
    pv = pivots['cohen_d_up_vs_down']
    if len(pv) > 0:
        fig, ax = plt.subplots(figsize=(11, 9))
        vmax = float(np.nanmax(np.abs(pv.values))) if pv.size else 1.0
        im = ax.imshow(pv.values, cmap='RdBu_r', aspect='auto',
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index, fontsize=8)
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                v = pv.iloc[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                          fontsize=7,
                          color='white' if abs(v) > vmax / 2 else 'black')
        ax.set_xlabel('Timeframe (small → large)')
        ax.set_ylabel('Concept')
        ax.set_title('Cohen d: UP_SMOOTH vs DOWN_SMOOTH per (concept, TF)\n'
                      'Positive = concept higher in UP days; Negative = higher in DOWN days')
        plt.colorbar(im, ax=ax, label='Cohen d')
        fig.tight_layout()
        heat_path = os.path.join(args.output_dir, 'heatmap_cohen_d_up_vs_down.png')
        fig.savefig(heat_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"\n  [saved] {heat_path}")

    # Same for variation axis
    pv2 = pivots['cohen_d_smooth_vs_choppy']
    if len(pv2) > 0:
        fig, ax = plt.subplots(figsize=(11, 9))
        vmax = float(np.nanmax(np.abs(pv2.values))) if pv2.size else 1.0
        im = ax.imshow(pv2.values, cmap='RdBu_r', aspect='auto',
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pv2.columns)))
        ax.set_xticklabels(pv2.columns)
        ax.set_yticks(range(len(pv2.index)))
        ax.set_yticklabels(pv2.index, fontsize=8)
        for i in range(len(pv2.index)):
            for j in range(len(pv2.columns)):
                v = pv2.iloc[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                          fontsize=7,
                          color='white' if abs(v) > vmax / 2 else 'black')
        ax.set_xlabel('Timeframe (small → large)')
        ax.set_ylabel('Concept')
        ax.set_title('Cohen d: SMOOTH vs CHOPPY per (concept, TF)')
        plt.colorbar(im, ax=ax, label='Cohen d')
        fig.tight_layout()
        heat2_path = os.path.join(args.output_dir, 'heatmap_cohen_d_smooth_vs_choppy.png')
        fig.savefig(heat2_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [saved] {heat2_path}")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features TF-sweep EDA — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {len(concepts)}  "
                f"**TFs:** {args.tfs}  "
                f"**Split:** {args.split}\n\n")
        f.write("## Concepts that FLIP sign across TFs (regime relationship character changes)\n\n")
        f.write(inv_df[inv_df['sign_flips'] > 0].to_string(index=False))
        f.write("\n\n## Concepts that hold consistently across TFs\n\n")
        f.write(consistent.head(15).to_string(index=False))
        f.write("\n\n## Cohen-d UP_SMOOTH vs DOWN_SMOOTH pivot\n\n")
        f.write(pivots['cohen_d_up_vs_down'].round(2).to_string())
        f.write("\n\n## Cohen-d SMOOTH vs CHOPPY pivot\n\n")
        f.write(pivots['cohen_d_smooth_vs_choppy'].round(2).to_string())
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
