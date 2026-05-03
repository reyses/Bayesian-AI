"""
v2_features_triplet_regime_eda.py — Layer C1: regime-stratified triplet
forward-return EDA (anchored on leading features).

The single-feature drill-down (Steps 1-5) established three principles:
  - Regime carries DIRECTION; feature carries AMPLITUDE.
  - 89.7% of features LAG; only 8.8% lead. Real predictive content
    lives in 1h-window STRUCTURE features (hurst_w, bar_range_1h,
    vol_mean_w_1h, etc.).
  - Q4 signals are RTH-localized and SMOOTH-regime stable.

Therefore a triplet that earns its keep must:
  (a) Anchor on a LEADING feature (s>0 in lead-lag analysis).
  (b) Compose with at least one AMPLITUDE/MOMENTUM feature for sizing.
  (c) Optionally compose with a CONTEXTUALIZER for polarity.
  (d) Be evaluated WITHIN a regime (direction is a regime property).

For each triplet of the form (ANCHOR, AMPLITUDE_X, AMPLITUDE_Y/CONTEXT):
  - Bin each feature into 3 quantiles within regime
  - 3*3*3 = 27 cells per (triplet, regime); 6 regimes -> 162 cells
  - Filter to cells with n >= min_cell_n
  - Find cells where mean_fwd substantially exceeds the regime baseline
    (so the triplet ADDS to regime info rather than just inheriting it)

A "lift" cell is one where:
    cell_mean_fwd - regime_mean_fwd  has |delta| > threshold
                                     and significance is plausible (n large)

Outputs:
  reports/findings/v2_features_triplet_regime/
    triplet_summary.csv      (anchor, X, Y, regime, q_a, q_x, q_y, n,
                              cell_mean, regime_mean, delta, wr)
    top_lift_cells.csv       cells whose lift exceeds threshold
    summary.md
    plot_<anchor>_<X>_<Y>.png  per regime: 3x3 heatmap of mean_fwd
                                with anchor fixed at top quantile
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


# Anchors from Step 3 (lead-lag) — features whose peak corr is at s>0:
DEFAULT_ANCHORS = [
    ('bar_range', '1h'),       # UP_CHOPPY corr +0.19 at s=+12 (strongest)
    ('vol_mean_w', '1h'),      # DOWN_SMOOTH corr +0.10 at s=+12
    ('hurst_w', '1h'),         # UP_SMOOTH corr +0.07 at s=+12
    ('hurst_w', '15m'),
    ('hurst_w', '5m'),
    ('hurst_w', '1m'),
]

# Amplitude/momentum companions (the strong magnitude carriers):
DEFAULT_COMPANIONS = [
    ('price_velocity_w', '5m'),
    ('price_sigma_w', '5m'),
    ('swing_noise_w', '1m'),
    ('bar_range', '5m'),
    ('body', '5m'),
    ('vol_velocity_w', '15m'),
    ('vol_mean_w', '5m'),
    ('reversion_prob_w', '15m'),
    ('z_se_w', '15m'),
    ('price_velocity_1b', '15m'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--quantiles', type=int, default=3)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=80)
    parser.add_argument('--lift-threshold', type=float, default=10.0,
                        help='Min |cell_mean - regime_mean| in ticks')
    parser.add_argument('--top-plots', type=int, default=15)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_triplet_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features triplet regime-stratified (Layer C1)")
    print(f"  Anchors: {DEFAULT_ANCHORS}")
    print(f"  Companions: {len(DEFAULT_COMPANIONS)}")
    print(f"  Triplets to evaluate: {len(DEFAULT_ANCHORS)} * "
          f"C({len(DEFAULT_COMPANIONS)},2) = "
          f"{len(DEFAULT_ANCHORS) * (len(DEFAULT_COMPANIONS) * (len(DEFAULT_COMPANIONS)-1) // 2)}")
    print(f"  Quantiles per feature: {args.quantiles}")
    print(f"  Cells per triplet per regime: {args.quantiles**3}")
    print(f"  Lift threshold: {args.lift_threshold} ticks")
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

    # Pre-compute regime baselines (mean_fwd of all bars in each regime)
    regime_baseline = {}
    for regime in REGIME_2D_ORDER:
        mask = (regimes == regime) & ~np.isnan(fwd)
        if mask.sum() < 50:
            continue
        regime_baseline[regime] = float(fwd[mask].mean())
    print(f"\n  Regime baselines (mean_fwd):")
    for r, m in regime_baseline.items():
        print(f"    {r:>14}: {m:>+8.2f}")

    # Pre-compute regime-local quantile bins for every (concept, tf) used
    print(f"\n--- Computing per-regime quantile bins ---")
    all_features = list(set(DEFAULT_ANCHORS + DEFAULT_COMPANIONS))
    bin_arrays = {}  # (concept, tf, regime) -> integer bin index for every bar
    for (concept, tf) in all_features:
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
            bin_idx = np.full(n, -1, dtype=np.int32)
            valid_full = ~np.isnan(v)
            bin_idx[valid_full] = np.digitize(v[valid_full], qs[1:-1])
            bin_arrays[(concept, tf, regime)] = bin_idx

    # Sweep triplets
    print(f"\n--- Sweeping triplets ---")
    rows = []

    n_triplets = 0
    for anchor in DEFAULT_ANCHORS:
        # pair companions
        for x, y in itertools.combinations(DEFAULT_COMPANIONS, 2):
            # avoid trivial: if anchor == x or anchor == y, skip
            if anchor in (x, y):
                continue
            n_triplets += 1
            for regime in REGIME_2D_ORDER:
                if regime not in regime_baseline:
                    continue
                bin_a = bin_arrays.get((anchor[0], anchor[1], regime))
                bin_x = bin_arrays.get((x[0], x[1], regime))
                bin_y = bin_arrays.get((y[0], y[1], regime))
                if bin_a is None or bin_x is None or bin_y is None:
                    continue
                regime_mask = (regimes == regime)
                # iterate cells
                for qa in range(args.quantiles):
                    for qx in range(args.quantiles):
                        for qy in range(args.quantiles):
                            cell_mask = (regime_mask
                                              & (bin_a == qa) & (bin_x == qx)
                                              & (bin_y == qy)
                                              & ~np.isnan(fwd))
                            n_cell = int(cell_mask.sum())
                            if n_cell < args.min_cell_n:
                                continue
                            f = fwd[cell_mask]
                            cm = float(f.mean())
                            wr = float((f > 0).mean())
                            rb = regime_baseline[regime]
                            rows.append({
                                'anchor_concept': anchor[0],
                                'anchor_tf': anchor[1],
                                'x_concept': x[0],
                                'x_tf': x[1],
                                'y_concept': y[0],
                                'y_tf': y[1],
                                'regime_2d': regime,
                                'q_anchor': qa,
                                'q_x': qx,
                                'q_y': qy,
                                'n': n_cell,
                                'cell_mean': cm,
                                'regime_mean': rb,
                                'lift': cm - rb,
                                'win_rate': wr,
                                'cell_std': float(f.std(ddof=1)),
                            })

    print(f"  Evaluated {n_triplets} triplets x 6 regimes x "
          f"{args.quantiles**3} cells")
    df = pd.DataFrame(rows)
    df['triplet_id'] = (df['anchor_concept'] + '_' + df['anchor_tf'] + '|'
                            + df['x_concept'] + '_' + df['x_tf'] + '|'
                            + df['y_concept'] + '_' + df['y_tf'])
    df.to_csv(os.path.join(args.output_dir, 'triplet_summary.csv'),
              index=False)
    print(f"  [saved] triplet_summary.csv ({len(df)} cells with n>={args.min_cell_n})")

    # Top lift cells
    df['abs_lift'] = df['lift'].abs()
    high_lift = df[df['abs_lift'] >= args.lift_threshold].sort_values(
        'abs_lift', ascending=False)
    high_lift.to_csv(os.path.join(args.output_dir, 'top_lift_cells.csv'),
                       index=False)
    print(f"  [saved] top_lift_cells.csv ({len(high_lift)} cells with "
          f"|lift| >= {args.lift_threshold})")

    print(f"\n  TOP 30 high-lift cells:")
    print(f"    {'anchor':>20}  {'X':>20}  {'Y':>20}  {'regime':>14}  "
          f"{'cell':>4}  {'n':>5}  {'mean':>7}  {'base':>7}  {'lift':>7}  "
          f"{'wr':>5}")
    for _, r in high_lift.head(30).iterrows():
        a = f"{r['anchor_concept']}_{r['anchor_tf']}"
        x = f"{r['x_concept']}_{r['x_tf']}"
        y = f"{r['y_concept']}_{r['y_tf']}"
        cell = f"{r['q_anchor']},{r['q_x']},{r['q_y']}"
        print(f"    {a:>20}  {x:>20}  {y:>20}  {r['regime_2d']:>14}  "
              f"{cell:>4}  {int(r['n']):>5}  {r['cell_mean']:>+7.2f}  "
              f"{r['regime_mean']:>+7.2f}  {r['lift']:>+7.2f}  "
              f"{r['win_rate']:>5.2f}")

    # Aggregate: best triplet (a, x, y) per regime by max |cell lift|
    print(f"\n  Best (anchor, x, y) per regime by max |cell lift|:")
    for regime in REGIME_2D_ORDER:
        if regime not in regime_baseline:
            continue
        sub = df[df['regime_2d'] == regime]
        if sub.empty:
            continue
        # best triplet by max |cell lift|
        peaks = sub.loc[sub.groupby('triplet_id')['abs_lift'].idxmax()]
        top5 = peaks.nlargest(5, 'abs_lift')
        print(f"\n    --- {regime} (baseline {regime_baseline[regime]:+.2f}) ---")
        print(f"    {'anchor':>20}  {'X':>20}  {'Y':>20}  "
              f"{'cell':>4}  {'n':>5}  {'mean':>7}  {'lift':>7}  {'wr':>5}")
        for _, r in top5.iterrows():
            a = f"{r['anchor_concept']}_{r['anchor_tf']}"
            x = f"{r['x_concept']}_{r['x_tf']}"
            y = f"{r['y_concept']}_{r['y_tf']}"
            cell = f"{r['q_anchor']},{r['q_x']},{r['q_y']}"
            print(f"    {a:>20}  {x:>20}  {y:>20}  "
                  f"{cell:>4}  {int(r['n']):>5}  {r['cell_mean']:>+7.2f}  "
                  f"{r['lift']:>+7.2f}  {r['win_rate']:>5.2f}")

    # Plot top triplets (one figure per triplet, 6 subplots = 6 regimes)
    plotted = 0
    if len(high_lift) > 0:
        # Pick top triplets (by sum of |lift| of their high-lift cells)
        top_triplet_ids = (high_lift.groupby('triplet_id')['abs_lift']
                                .sum().sort_values(ascending=False)
                                .head(args.top_plots).index.tolist())

        for triplet_id in top_triplet_ids:
            sub = df[df['triplet_id'] == triplet_id]
            if sub.empty:
                continue
            anchor_concept = sub.iloc[0]['anchor_concept']
            anchor_tf = sub.iloc[0]['anchor_tf']
            x_concept = sub.iloc[0]['x_concept']
            x_tf = sub.iloc[0]['x_tf']
            y_concept = sub.iloc[0]['y_concept']
            y_tf = sub.iloc[0]['y_tf']
            # 6 panels: one per regime
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            axes = axes.flatten()
            for ri, regime in enumerate(REGIME_2D_ORDER):
                ax = axes[ri]
                rs = sub[sub['regime_2d'] == regime]
                if rs.empty:
                    ax.set_title(f'{regime} (no data)')
                    ax.axis('off')
                    continue
                # heatmap: rows=q_x, cols=q_y for each q_anchor (3 stacked tiers)
                # Show average over q_anchor first (or mark anchor=top only)
                # pivot: q_x x q_y, value=cell_mean (top anchor only)
                top_anchor = rs[rs['q_anchor'] == args.quantiles - 1]
                if top_anchor.empty:
                    top_anchor = rs
                pivot = top_anchor.pivot_table(
                    index='q_x', columns='q_y', values='cell_mean')
                if pivot.empty:
                    ax.axis('off')
                    continue
                vmax = float(np.nanmax(np.abs(pivot.values)))
                im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto',
                                vmin=-vmax, vmax=vmax)
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels([f'q{q}' for q in pivot.columns])
                ax.set_yticks(range(pivot.shape[0]))
                ax.set_yticklabels([f'q{q}' for q in pivot.index])
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        v = pivot.iloc[i, j]
                        if pd.isna(v):
                            continue
                        ax.text(j, i, f'{v:+.0f}', ha='center', va='center',
                                  fontsize=9, color='white' if abs(v) > vmax/2
                                                                else 'black')
                ax.set_xlabel(f'{y_concept}_{y_tf}')
                ax.set_ylabel(f'{x_concept}_{x_tf}')
                ax.set_title(f'{regime} (anchor={anchor_concept}_{anchor_tf} top)')
            fig.suptitle(f'Triplet: {anchor_concept}_{anchor_tf} | '
                            f'{x_concept}_{x_tf} | {y_concept}_{y_tf}',
                            fontsize=11)
            fig.tight_layout()
            png_path = os.path.join(args.output_dir,
                                       f'plot_{anchor_concept}_{anchor_tf}__'
                                       f'{x_concept}_{x_tf}__'
                                       f'{y_concept}_{y_tf}.png')
            fig.savefig(png_path, dpi=120, bbox_inches='tight',
                          facecolor='white')
            plt.close(fig)
            plotted += 1
    print(f"\n  [saved] {plotted} triplet heatmaps")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features triplet regime-stratified (Layer C1) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Triplet design**: anchor (leading feature) x companion x "
                f"companion. Each feature binned into {args.quantiles} "
                f"regime-local quantiles. {args.quantiles**3} cells per "
                f"(triplet, regime). Min cell n = {args.min_cell_n}.\n\n")
        f.write(f"**Lift threshold**: |cell_mean - regime_baseline| >= "
                f"{args.lift_threshold} ticks\n\n")
        f.write(f"**Anchors used**: {DEFAULT_ANCHORS}\n\n")
        f.write(f"**Companions used**: {DEFAULT_COMPANIONS}\n\n")
        f.write("## Regime baselines\n\n")
        for r, m in regime_baseline.items():
            f.write(f"- **{r}**: {m:+.2f} ticks\n")
        f.write(f"\n## Top 50 high-lift triplet cells\n\n")
        f.write(high_lift.head(50).to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
