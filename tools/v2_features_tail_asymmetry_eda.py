"""
v2_features_tail_asymmetry_eda.py — Tail asymmetry (Step #4).

For each (feature, TF, regime), compare Q0 (bottom quintile) vs Q4 (top
quintile) of the feature's regime-local distribution. Three flavors of
asymmetry:

(1) MAGNITUDE asymmetry
    |mean_fwd at Q0| vs |mean_fwd at Q4|
    If Q4 mean_fwd is +30 ticks but Q0 mean_fwd is -3 ticks, the feature
    only matters at the high end. One-sided trigger, not continuous.

(2) DISPERSION asymmetry
    std_fwd at Q0 vs std_fwd at Q4
    If Q0 has 50% wider dispersion than Q4, low values represent a
    riskier state regardless of mean direction.

(3) SIGN asymmetry
    sign(mean_fwd Q0) vs sign(mean_fwd Q4)
    Symmetric: opposite signs (high feature -> long, low feature -> short).
    Asymmetric same-sign: both push the same way (e.g. both volatility
    extremes are bearish).

Outputs:
  reports/findings/v2_features_tail_asymmetry/
    tail_summary.csv      (concept, tf, regime, q0_mean, q4_mean, q0_std, q4_std, ...)
    asymmetry_classes.csv per cell: classification + magnitude_ratio + dispersion_ratio
    summary.md
    plot_q0q4_<concept>_<tf>.png  scatter Q0 mean vs Q4 mean colored by regime
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


DEFAULT_CONCEPTS = [
    'price_sigma_w', 'bar_range', 'vol_mean_w', 'vol_velocity_w',
    'price_velocity_w', 'body', 'price_velocity_1b', 'swing_noise_w',
    'z_se_w', 'reversion_prob_w', 'hurst_w',
]
DEFAULT_TFS = ['5s', '1m', '5m', '15m', '1h']


def classify(q0_mean: float, q4_mean: float,
              ratio_threshold: float = 2.0) -> str:
    """Classify the tail-asymmetry pattern."""
    if np.isnan(q0_mean) or np.isnan(q4_mean):
        return 'unknown'
    s0, s4 = np.sign(q0_mean), np.sign(q4_mean)
    abs0, abs4 = abs(q0_mean), abs(q4_mean)

    # near-zero in either tail
    if abs0 < 1.0 and abs4 < 1.0:
        return 'flat_no_signal'

    if s0 != s4:
        if abs0 < 1.0:
            return 'top_only_one_sided'
        if abs4 < 1.0:
            return 'bottom_only_one_sided'
        ratio = max(abs0, abs4) / max(min(abs0, abs4), 1e-9)
        if ratio < ratio_threshold:
            return 'symmetric_opposite'
        return 'asymmetric_opposite'

    # same-sign tails
    return 'same_sign_both_tails'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--concepts', nargs='+', default=DEFAULT_CONCEPTS)
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=50)
    parser.add_argument('--top-plots', type=int, default=10)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_tail_asymmetry')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features tail asymmetry (Step #4)")
    print(f"  Concepts: {args.concepts}")
    print(f"  TFs: {args.tfs}")
    print(f"  Quantiles: {args.quantiles}; comparing Q0 vs Q{args.quantiles - 1}")
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

    close = full['close'].values.astype(np.float64)
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    regimes = full['regime_2d'].values.astype(str)

    top_q = args.quantiles - 1

    print(f"\n--- Sweeping {len(args.concepts)} concepts x {len(args.tfs)} TFs "
          f"x {len(REGIME_2D_ORDER)} regimes ---")

    rows = []
    for concept in args.concepts:
        for tf in args.tfs:
            col = feature_column_for(concept, tf)
            if col not in full.columns:
                continue
            v = full[col].values.astype(np.float64)
            for regime in REGIME_2D_ORDER:
                regime_mask = (regimes == regime)
                if regime_mask.sum() < 200:
                    continue
                v_r = v[regime_mask]
                valid = ~np.isnan(v_r)
                if valid.sum() < args.quantiles * 5:
                    continue
                qs = np.quantile(v_r[valid], np.linspace(0, 1, args.quantiles + 1))
                qs[0] -= 1e-9
                qs[-1] += 1e-9
                bin_idx = np.digitize(v, qs[1:-1])

                q0_mask = regime_mask & (bin_idx == 0) & ~np.isnan(v) & ~np.isnan(fwd)
                q4_mask = regime_mask & (bin_idx == top_q) & ~np.isnan(v) & ~np.isnan(fwd)
                if q0_mask.sum() < args.min_cell_n or q4_mask.sum() < args.min_cell_n:
                    continue

                f0 = fwd[q0_mask]
                f4 = fwd[q4_mask]
                q0_mean = float(f0.mean())
                q4_mean = float(f4.mean())
                q0_std = float(f0.std(ddof=1))
                q4_std = float(f4.std(ddof=1))
                q0_wr = float((f0 > 0).mean())
                q4_wr = float((f4 > 0).mean())

                # ratios
                mag_ratio = max(abs(q0_mean), abs(q4_mean)) / max(
                    min(abs(q0_mean), abs(q4_mean)), 1e-9)
                disp_ratio = max(q0_std, q4_std) / max(min(q0_std, q4_std), 1e-9)
                disp_skew = (q0_std - q4_std) / max(q0_std + q4_std, 1e-9)

                # significance: t-test difference of means
                # se = sqrt(s0^2/n0 + s4^2/n4)
                n0 = int(q0_mask.sum())
                n4 = int(q4_mask.sum())
                se = np.sqrt(q0_std ** 2 / n0 + q4_std ** 2 / n4)
                t_stat = (q4_mean - q0_mean) / max(se, 1e-9)

                cls = classify(q0_mean, q4_mean)

                rows.append({
                    'concept': concept,
                    'tf': tf,
                    'regime_2d': regime,
                    'n_q0': n0,
                    'n_q4': n4,
                    'q0_mean': q0_mean,
                    'q4_mean': q4_mean,
                    'spread': q4_mean - q0_mean,
                    'q0_std': q0_std,
                    'q4_std': q4_std,
                    'q0_wr': q0_wr,
                    'q4_wr': q4_wr,
                    'mag_ratio': mag_ratio,
                    'disp_ratio': disp_ratio,
                    'disp_skew_q0_minus_q4_norm': disp_skew,
                    't_stat': t_stat,
                    'class': cls,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'tail_summary.csv'), index=False)
    print(f"  [saved] tail_summary.csv ({len(df)} cells)")

    cls_counts = df['class'].value_counts()
    print(f"\n  Asymmetry class distribution ({len(df)} cells):")
    for c, n in cls_counts.items():
        pct = 100.0 * n / len(df)
        print(f"    {c:>26}: {n:>4}  ({pct:>5.1f}%)")

    # Show: top 20 by spread (q4_mean - q0_mean) magnitude
    df_sorted = df.copy()
    df_sorted['abs_spread'] = df_sorted['spread'].abs()
    top_spread = df_sorted.sort_values('abs_spread', ascending=False).head(25)
    print(f"\n  Top 25 by Q4-Q0 spread (continuous separation):")
    print(f"    {'concept':>20}  {'tf':>4}  {'regime':>14}  "
          f"{'q0':>7} {'q4':>7} {'spread':>8} {'magR':>5} {'class':>22}")
    for _, r in top_spread.iterrows():
        print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['q0_mean']:>7.2f} {r['q4_mean']:>7.2f} {r['spread']:>+8.2f} "
              f"{r['mag_ratio']:>5.1f} {r['class']:>22}")

    # Show: top one-sided cells (high Q4, near-0 Q0 OR high Q0, near-0 Q4)
    one_sided = df_sorted[df_sorted['class'].isin(
        ['top_only_one_sided', 'bottom_only_one_sided'])]
    print(f"\n  One-sided trigger cells (only one tail carries signal): "
          f"{len(one_sided)}")
    for _, r in one_sided.sort_values('abs_spread', ascending=False).head(15).iterrows():
        print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['q0_mean']:>7.2f} {r['q4_mean']:>7.2f} {r['class']:>22}")

    # Show: same-sign both tails — features where extremes BOTH push same way
    same_sign = df_sorted[df_sorted['class'] == 'same_sign_both_tails']
    print(f"\n  Same-sign both tails (extremes BOTH push same direction): "
          f"{len(same_sign)}")
    for _, r in same_sign.sort_values('abs_spread', ascending=False).head(15).iterrows():
        print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"q0={r['q0_mean']:>+7.2f} q4={r['q4_mean']:>+7.2f}  "
              f"both {'+' if r['q0_mean'] > 0 else '-'}")

    # Plot Q0 vs Q4 scatter for each (concept, tf)
    plotted = 0
    for concept in args.concepts:
        for tf in args.tfs:
            sub = df[(df['concept'] == concept) & (df['tf'] == tf)]
            if sub.empty or len(sub) < 3:
                continue
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(REGIME_2D_ORDER)))
            for ri, regime in enumerate(REGIME_2D_ORDER):
                ssub = sub[sub['regime_2d'] == regime]
                if ssub.empty:
                    continue
                ax.scatter(ssub['q0_mean'], ssub['q4_mean'], s=120,
                            c=[colors[ri]], label=regime, alpha=0.85,
                            edgecolors='black', linewidth=0.7)
            lim = max(abs(sub['q0_mean']).max(), abs(sub['q4_mean']).max()) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k:', alpha=0.4, label='symmetry y=x')
            ax.plot([-lim, lim], [lim, -lim], 'k:', alpha=0.2,
                     label='anti-symmetry y=-x')
            ax.axhline(0, color='black', alpha=0.3)
            ax.axvline(0, color='black', alpha=0.3)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel('Q0 mean_fwd (bottom quintile)')
            ax.set_ylabel(f'Q{top_q} mean_fwd (top quintile)')
            ax.set_title(f'{concept} {tf} — Q0 vs Q{top_q} by regime')
            ax.legend(fontsize=7, loc='best')
            ax.grid(alpha=0.3)
            fig.tight_layout()
            png_path = os.path.join(args.output_dir,
                                       f'plot_q0q4_{concept}_{tf}.png')
            fig.savefig(png_path, dpi=120, bbox_inches='tight',
                          facecolor='white')
            plt.close(fig)
            plotted += 1
            if plotted >= args.top_plots:
                break
        if plotted >= args.top_plots:
            break
    print(f"\n  [saved] {plotted} Q0/Q4 scatter plots")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features tail asymmetry (Step #4) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {args.concepts}\n\n")
        f.write(f"**TFs:** {args.tfs}\n\n")
        f.write(f"**Quantiles:** {args.quantiles} (compare Q0 vs Q{top_q})\n\n")
        f.write("## Asymmetry class distribution\n\n")
        for c, n in cls_counts.items():
            pct = 100.0 * n / len(df)
            f.write(f"- **{c}**: {n} ({pct:.1f}%)\n")
        f.write("\n## Top 30 by Q0-Q4 spread\n\n")
        f.write(top_spread.head(30).to_string(index=False))
        f.write("\n\n## One-sided trigger cells\n\n")
        f.write(one_sided.head(20).to_string(index=False))
        f.write("\n\n## Same-sign both-tails cells\n\n")
        f.write(same_sign.head(20).to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
