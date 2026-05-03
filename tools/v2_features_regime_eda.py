"""
v2_features_regime_eda.py — How does each v2 feature behave across regimes?

Descriptive analysis (NOT prediction). For every one of the 185 v2 feature
columns, characterize:

  1. Distribution per regime cell
       mean, std, median, p10, p90 in each of UP_SMOOTH/UP_CHOPPY/
       DOWN_SMOOTH/DOWN_CHOPPY/FLAT_SMOOTH/FLAT_CHOPPY.

  2. Cross-regime separation
       Cohen's d between every regime pair → "which features separate
       which regimes most cleanly?".

  3. Correlation with price phenomena
       a. Concurrent: corr(feature_t, |close_t - vwap_5m_w|)
          → does the feature track magnitude of current dislocation?
       b. Concurrent signed: corr(feature_t, close_t - vwap_5m_w)
          → does the feature track direction of dislocation?
       c. Lookback return: corr(feature_t, close_t - close_{t-N})
          → does the feature reflect the recent move that just happened?
       d. Forward return: corr(feature_t, close_{t+N} - close_t)
          → does the feature lead future moves? (lead/lag analysis)
       N defaults to 12 bars (1 hour at 5m base).

  4. Behavior at regime transitions
       Aggregate feature change around regime flips. Tells us which
       features "warn" of regime shifts.

Outputs:
  reports/findings/v2_features_regime_eda/
    per_regime_distributions.csv   — feature × regime → mean/std/median/p10/p90
    regime_separation.csv          — feature × regime_pair → cohen_d, abs(d)
    price_correlations.csv         — feature → corr_concurrent_abs, signed,
                                      lookback, forward
    top_separators.md              — top 30 (feature, regime_pair) by |d|
    top_price_correlators.md       — top 30 features by each price-corr type
    feature_class_summary.png      — box plots grouped by L1/L2/L3 layer
    regime_separation_heatmap.png  — heatmap of mean |d| per regime pair
"""

from __future__ import annotations
import argparse
import os
import sys
from itertools import combinations
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
    load_regime_labels, REGIME_2D_ORDER, REGIME_2D_COLORS,
)


DEFAULT_BASE_TF = '5m'
DEFAULT_LOOKBACK_N = 12   # 12 × 5m = 1 hour
DEFAULT_FORWARD_N = 12    # symmetric default


# ── Helpers ──────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two arrays. Pooled std denominator."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    m_a, m_b = a.mean(), b.mean()
    s_a, s_b = a.std(ddof=1), b.std(ddof=1)
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
    if pooled < 1e-12:
        return 0.0
    return float((m_a - m_b) / pooled)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr, NaN-safe."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float('nan')
    a = a[mask]
    b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def feature_class(col: str) -> str:
    """Group features by layer prefix (L0/L1/L2/L3) for box plots."""
    if col.startswith('L0_'):
        return 'L0'
    if col.startswith('L1_'):
        return 'L1'
    if col.startswith('L2_'):
        return 'L2'
    if col.startswith('L3_'):
        return 'L3'
    return 'other'


# ── Main analyses ────────────────────────────────────────────────────────

def per_regime_distributions(merged: pd.DataFrame, feature_cols: list[str],
                              verbose: bool = True) -> pd.DataFrame:
    """For each feature × regime, compute mean/std/median/p10/p90."""
    rows = []
    if verbose:
        print(f"  Computing distributions across {len(feature_cols)} features × "
              f"{merged['regime_2d'].nunique()} regimes...")
    for feat in feature_cols:
        for r2d, group in merged.groupby('regime_2d'):
            v = group[feat].dropna().values
            if len(v) == 0:
                continue
            rows.append({
                'feature': feat,
                'regime_2d': r2d,
                'n': len(v),
                'mean': float(np.mean(v)),
                'std': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                'median': float(np.median(v)),
                'p10': float(np.percentile(v, 10)),
                'p90': float(np.percentile(v, 90)),
            })
    return pd.DataFrame(rows)


def regime_separation(merged: pd.DataFrame, feature_cols: list[str],
                       verbose: bool = True) -> pd.DataFrame:
    """For each feature × regime-pair, compute Cohen's d."""
    regimes = sorted(merged['regime_2d'].unique())
    pairs = list(combinations(regimes, 2))
    rows = []
    if verbose:
        print(f"  Computing Cohen's d for {len(feature_cols)} features × "
              f"{len(pairs)} regime pairs...")
    by_regime = {r: merged[merged['regime_2d'] == r] for r in regimes}
    for feat in feature_cols:
        for r_a, r_b in pairs:
            a = by_regime[r_a][feat].values
            b = by_regime[r_b][feat].values
            d = cohen_d(a, b)
            rows.append({
                'feature': feat,
                'regime_a': r_a,
                'regime_b': r_b,
                'pair': f'{r_a}__vs__{r_b}',
                'cohen_d': d,
                'abs_d': abs(d) if not np.isnan(d) else float('nan'),
            })
    return pd.DataFrame(rows)


def price_correlations(merged: pd.DataFrame, feature_cols: list[str],
                        base_close: np.ndarray, vwap_5m: np.ndarray,
                        lookback_n: int, forward_n: int,
                        verbose: bool = True) -> pd.DataFrame:
    """For each feature, compute correlations with several price phenomena."""
    rows = []
    n = len(merged)
    if verbose:
        print(f"  Computing price correlations for {len(feature_cols)} features...")

    # Concurrent dislocation (signed and abs)
    concurrent_signed = base_close - vwap_5m
    concurrent_abs = np.abs(concurrent_signed)

    # Lookback return (close[t] - close[t - N])
    lookback_ret = np.full(n, np.nan)
    if n > lookback_n:
        lookback_ret[lookback_n:] = base_close[lookback_n:] - base_close[:-lookback_n]

    # Forward return (close[t + N] - close[t])
    forward_ret = np.full(n, np.nan)
    if n > forward_n:
        forward_ret[:-forward_n] = base_close[forward_n:] - base_close[:-forward_n]

    for feat in feature_cols:
        v = merged[feat].values
        rows.append({
            'feature': feat,
            'corr_concurrent_signed': safe_corr(v, concurrent_signed),
            'corr_concurrent_abs': safe_corr(v, concurrent_abs),
            'corr_lookback_return': safe_corr(v, lookback_ret),
            'corr_forward_return': safe_corr(v, forward_ret),
        })
    return pd.DataFrame(rows)


def regime_transition_behavior(merged: pd.DataFrame, feature_cols: list[str],
                                window: int = 24, verbose: bool = True
                                ) -> pd.DataFrame:
    """Average feature value in `window` bars BEFORE vs AFTER regime flips.

    Returns one row per feature with: pre_mean, post_mean, delta, n_transitions.
    """
    if verbose:
        print(f"  Computing regime-transition deltas (window={window} bars)...")

    # Detect transitions: rows where regime_2d != previous regime_2d
    flip = (merged['regime_2d'] != merged['regime_2d'].shift(1)).values
    flip[0] = False  # first row has no prior regime
    flip_idx = np.where(flip)[0]

    rows = []
    for feat in feature_cols:
        v = merged[feat].values
        pre_vals = []
        post_vals = []
        for i in flip_idx:
            pre = v[max(0, i - window): i]
            post = v[i: min(len(v), i + window)]
            pre = pre[~np.isnan(pre)]
            post = post[~np.isnan(post)]
            if len(pre) > 0 and len(post) > 0:
                pre_vals.append(float(np.mean(pre)))
                post_vals.append(float(np.mean(post)))
        if not pre_vals:
            rows.append({
                'feature': feat,
                'n_transitions': 0,
                'pre_mean': float('nan'),
                'post_mean': float('nan'),
                'delta_mean': float('nan'),
                'abs_delta_mean': float('nan'),
            })
            continue
        pre_arr = np.array(pre_vals)
        post_arr = np.array(post_vals)
        delta = post_arr - pre_arr
        rows.append({
            'feature': feat,
            'n_transitions': len(pre_arr),
            'pre_mean': float(pre_arr.mean()),
            'post_mean': float(post_arr.mean()),
            'delta_mean': float(delta.mean()),
            'abs_delta_mean': float(np.abs(delta).mean()),
        })
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_regime_separation_heatmap(sep_df: pd.DataFrame, out_path: str):
    """Heatmap of mean |d| per regime pair (averaged across features)."""
    pivot = sep_df.groupby(['regime_a', 'regime_b'])['abs_d'].mean().reset_index()
    regimes = sorted(set(pivot['regime_a']).union(pivot['regime_b']))
    mat = np.full((len(regimes), len(regimes)), np.nan)
    for _, r in pivot.iterrows():
        i = regimes.index(r['regime_a'])
        j = regimes.index(r['regime_b'])
        mat[i, j] = r['abs_d']
        mat[j, i] = r['abs_d']
    np.fill_diagonal(mat, 0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(regimes)))
    ax.set_yticks(range(len(regimes)))
    ax.set_xticklabels(regimes, rotation=45, ha='right')
    ax.set_yticklabels(regimes)
    for i in range(len(regimes)):
        for j in range(len(regimes)):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                    color='white' if mat[i, j] < mat[~np.isnan(mat)].mean() else 'black',
                    fontsize=9)
    ax.set_title('Mean |Cohen d| across all features (regime pair separation)')
    plt.colorbar(im, ax=ax, label='Mean |d|')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_feature_class_boxes(merged: pd.DataFrame, feature_cols: list[str],
                              out_path: str):
    """Box plots of feature values per regime, faceted by L0/L1/L2/L3."""
    classes = sorted({feature_class(c) for c in feature_cols})
    fig, axes = plt.subplots(len(classes), 1, figsize=(14, 3.5 * len(classes)),
                              squeeze=False)
    for k, cls in enumerate(classes):
        ax = axes[k, 0]
        cols = [c for c in feature_cols if feature_class(c) == cls]
        # Z-score each feature globally so they sit on the same scale
        zs = []
        labels = []
        for r2d in REGIME_2D_ORDER:
            sub = merged[merged['regime_2d'] == r2d]
            if len(sub) == 0:
                continue
            block = sub[cols].values.astype(np.float64)
            # Center each col by global mean / std
            for j, col in enumerate(cols):
                v = merged[col].values
                mu, sd = np.nanmean(v), np.nanstd(v)
                if sd < 1e-12:
                    continue
                z = (block[:, j] - mu) / sd
                z = z[~np.isnan(z)]
                if len(z) == 0:
                    continue
                zs.append(z)
                labels.append(f'{r2d}\n{col.split("_")[-1] if cls != "L0" else col}')
        # Sample for plot tractability
        if len(zs) > 60:
            sample_idx = np.linspace(0, len(zs) - 1, 60, dtype=int)
            zs = [zs[i] for i in sample_idx]
            labels = [labels[i] for i in sample_idx]
        ax.boxplot(zs, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{cls} features (z-scored), distributions per regime')
        ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--lookback-n', type=int, default=DEFAULT_LOOKBACK_N)
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--transition-window', type=int, default=24)
    parser.add_argument('--split', default='IS',
                        help='Use only this split (IS/VAL/OOS/ALL). Default IS — keeps OOS clean.')
    parser.add_argument('--top-n', type=int, default=30,
                        help='Top-N features in each "top" markdown summary')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_regime_eda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features × Regime EDA (descriptive, IS-only by default)")
    print(f"  Base TF: {args.base_tf}")
    print(f"  Split: {args.split}")
    print(f"  Lookback/Forward N: {args.lookback_n}/{args.forward_n} bars")
    print(f"{'='*70}")

    # Load + merge
    print(f"\n--- Loading data ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if base_df.empty:
        print(f"ERROR: no OHLC for {args.base_tf}")
        return
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int

    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base_df['date'] = dt_la.dt.date.astype(str)
    print(f"  {args.base_tf}: {len(base_df):,} bars")

    labels_df = load_regime_labels(args.labels_csv)
    labels_df = labels_df.copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    keep = ['date', 'direction_axis', 'variation_axis', 'regime_2d', 'split']
    merged = base_df.merge(labels_df[keep], on='date', how='inner')
    print(f"  Bars after regime merge: {len(merged):,}")

    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split} filter: {len(merged):,} bars "
          f"({merged['date'].nunique()} days)")

    # v2 features
    ts_int = merged['ts_int'].values.astype(np.int64)
    ts_min, ts_max = int(ts_int.min()), int(ts_int.max())
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    print(f"  v2 features: {len(features_5s):,} 5s rows")
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    feature_cols = [c for c in aligned.columns if c != 'timestamp']
    print(f"  Aligned features: {len(feature_cols)} cols")

    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    # Anchor signals (for price_correlations)
    base_close = full['close'].values.astype(np.float64)
    vwap_col = f'L2_{args.base_tf}_vwap_w'
    if vwap_col in full.columns:
        vwap = full[vwap_col].values.astype(np.float64)
    else:
        # Fallback: use base_close itself (so concurrent_signed = 0)
        vwap = base_close.copy()
        print(f"  WARNING: no {vwap_col} in features, concurrent corr will be 0")

    # ── 1. Distributions ──
    print(f"\n--- Distributions per regime ---")
    dist_df = per_regime_distributions(full, feature_cols, verbose=True)
    dist_path = os.path.join(args.output_dir, 'per_regime_distributions.csv')
    dist_df.to_csv(dist_path, index=False)
    print(f"  [saved] {dist_path} ({len(dist_df)} rows)")

    # ── 2. Regime separation (Cohen d) ──
    print(f"\n--- Regime separation (Cohen d) ---")
    sep_df = regime_separation(full, feature_cols, verbose=True)
    sep_path = os.path.join(args.output_dir, 'regime_separation.csv')
    sep_df.to_csv(sep_path, index=False)
    print(f"  [saved] {sep_path}")

    # Top separators
    top_sep = sep_df.sort_values('abs_d', ascending=False).head(args.top_n)
    print(f"\n  Top {args.top_n} (feature, regime_pair) by |Cohen d|:")
    for _, r in top_sep.iterrows():
        print(f"    {r['feature']:>40}  {r['pair']:<32}  d={r['cohen_d']:+.2f}")

    # ── 3. Price correlations ──
    print(f"\n--- Price correlations ---")
    corr_df = price_correlations(full, feature_cols, base_close, vwap,
                                   args.lookback_n, args.forward_n, verbose=True)
    corr_path = os.path.join(args.output_dir, 'price_correlations.csv')
    corr_df.to_csv(corr_path, index=False)
    print(f"  [saved] {corr_path}")

    print(f"\n  Top {args.top_n} features by |corr_lookback_return|:")
    top_lb = corr_df.assign(abs_lb=corr_df['corr_lookback_return'].abs()) \
        .sort_values('abs_lb', ascending=False).head(args.top_n)
    for _, r in top_lb.iterrows():
        print(f"    {r['feature']:>40}  lookback r={r['corr_lookback_return']:+.3f}  "
              f"forward r={r['corr_forward_return']:+.3f}  "
              f"concurrent_abs r={r['corr_concurrent_abs']:+.3f}")

    print(f"\n  Top {args.top_n} features by |corr_forward_return| (lead/lag):")
    top_fw = corr_df.assign(abs_fw=corr_df['corr_forward_return'].abs()) \
        .sort_values('abs_fw', ascending=False).head(args.top_n)
    for _, r in top_fw.iterrows():
        print(f"    {r['feature']:>40}  forward r={r['corr_forward_return']:+.3f}  "
              f"lookback r={r['corr_lookback_return']:+.3f}")

    # ── 4. Regime transitions ──
    print(f"\n--- Regime transition behavior ---")
    trans_df = regime_transition_behavior(full, feature_cols,
                                            window=args.transition_window,
                                            verbose=True)
    trans_path = os.path.join(args.output_dir, 'regime_transitions.csv')
    trans_df.to_csv(trans_path, index=False)
    print(f"  [saved] {trans_path}")

    print(f"\n  Top {args.top_n} features by |delta_mean| at regime transitions:")
    top_tr = trans_df.sort_values('abs_delta_mean', ascending=False).head(args.top_n)
    for _, r in top_tr.iterrows():
        print(f"    {r['feature']:>40}  pre={r['pre_mean']:+.3g}  "
              f"post={r['post_mean']:+.3g}  delta={r['delta_mean']:+.3g}")

    # ── Plots ──
    print(f"\n--- Plotting ---")
    heatmap_path = os.path.join(args.output_dir, 'regime_separation_heatmap.png')
    plot_regime_separation_heatmap(sep_df, heatmap_path)
    print(f"  [saved] {heatmap_path}")

    box_path = os.path.join(args.output_dir, 'feature_class_boxes.png')
    plot_feature_class_boxes(full, feature_cols, box_path)
    print(f"  [saved] {box_path}")

    # ── Markdown summaries ──
    print(f"\n--- Summary md ---")
    md_path = os.path.join(args.output_dir, 'top_separators.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Top regime separators (|Cohen d|)\n\n")
        f.write(f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"Split: {args.split}\n\n")
        f.write(top_sep.to_string(index=False))
    print(f"  [saved] {md_path}")

    md2_path = os.path.join(args.output_dir, 'top_price_correlators.md')
    with open(md2_path, 'w', encoding='utf-8') as f:
        f.write(f"# Top features by price correlation\n\n")
        f.write(f"Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"Split: {args.split} | Lookback/Forward N: {args.lookback_n}/{args.forward_n}\n\n")
        f.write(f"## By |lookback return| corr (does feature reflect past N-bar move?)\n\n")
        f.write(top_lb.head(args.top_n).to_string(index=False))
        f.write("\n\n")
        f.write(f"## By |forward return| corr (does feature lead future N-bar move?)\n\n")
        f.write(top_fw.head(args.top_n).to_string(index=False))
        f.write("\n\n")
        f.write(f"## Top features by |corr_concurrent_abs| (track current dislocation magnitude)\n\n")
        top_ca = corr_df.assign(abs_ca=corr_df['corr_concurrent_abs'].abs()) \
            .sort_values('abs_ca', ascending=False).head(args.top_n)
        f.write(top_ca.to_string(index=False))
    print(f"  [saved] {md2_path}")

    print(f"\nDone. Outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
