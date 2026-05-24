"""
v2_features_quantile_regime_eda.py — Single-feature drill-down: where in
each feature's distribution does the signal live, and does that location
shift by regime?

The regime-stratified TF sweep showed corr_fwd flips sign across regimes
for the top features. But correlation collapses the entire distribution
into one number. This tool drills deeper:

For each (feature, TF, regime, quantile_bin):
  n          bars in this cell
  mean_fwd   mean forward return at t+N
  win_rate   sign(fwd) > 0 fraction
  std_fwd    dispersion within bin

Bins computed WITHIN the regime (not globally) so cells reflect regime-
relative position. This lets us see, e.g., "in UP_SMOOTH, top-quintile
of bar_range gives +X; in DOWN_CHOPPY, top-quintile gives −Y".

Per (feature, TF, regime), we also compute the SHAPE of the quantile-
to-fwd_return curve:
  monotonic_increasing   each quantile's mean_fwd is higher than the prior
  monotonic_decreasing   each quantile's mean_fwd is lower than the prior
  u_shape                tails (Q0, Qmax) higher than middle
  inverted_u_shape       middle higher than tails
  noisy                  no clear pattern
  signal_strength        max|mean_fwd| − min|mean_fwd| across quantiles

Outputs:
  reports/findings/v2_features_quantile_regime/
    quantile_summary.csv    long form (feature, tf, regime, q, mean_fwd, ...)
    shape_summary.csv       per (feature, tf, regime): shape label, signal_strength
    pivot_signal_strength.csv  rows=concept_TF, cols=regime → signal_strength
    top_shape_changers.csv  features whose quantile-curve SHAPE changes by regime
    summary.md
    heatmap_<regime>.png    per regime: feature × quantile heatmap of mean_fwd
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
    feature_column_for, TF_ORDER_SMALL_TO_LARGE,
)


def quantile_bins_within(values: np.ndarray, q: int) -> np.ndarray:
    valid = ~np.isnan(values)
    out = np.full(len(values), -1, dtype=np.int8)
    if valid.sum() < q * 5:
        return out
    qs = np.quantile(values[valid], np.linspace(0, 1, q + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_idx = np.digitize(values[valid], qs[1:-1])
    out[valid] = bin_idx.astype(np.int8)
    return out


def classify_shape(quantile_means: list[float], min_step: float = 0.5) -> str:
    """Label the curve as monotonic / u / inverted_u / noisy.

    min_step: minimum mean_fwd difference between adjacent quantiles to call
    it a "step" (in tick units, ~0.5 ticks = $0.25).
    """
    if len(quantile_means) < 3 or any(np.isnan(m) for m in quantile_means):
        return 'noisy'
    diffs = np.diff(quantile_means)
    n_up = int((diffs > min_step).sum())
    n_dn = int((diffs < -min_step).sum())
    n_flat = int((np.abs(diffs) <= min_step).sum())

    if n_up == len(diffs):
        return 'monotonic_increasing'
    if n_dn == len(diffs):
        return 'monotonic_decreasing'
    # U-shape: first diffs negative, then positive
    if len(diffs) >= 2:
        mid = len(diffs) // 2
        first_half = diffs[:mid]
        second_half = diffs[mid:]
        if (first_half < -min_step).all() and (second_half > min_step).all():
            return 'u_shape'
        if (first_half > min_step).all() and (second_half < -min_step).all():
            return 'inverted_u_shape'
    return 'noisy'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+',
                        default=['5s', '1m', '5m', '15m', '1h', '4h'])
    parser.add_argument('--concepts', nargs='*', default=None,
                        help='Default: all 23 concepts')
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=50)
    parser.add_argument('--shape-min-step', type=float, default=0.5,
                        help='Minimum mean_fwd diff (ticks) between adjacent '
                             'quantiles to count as a step')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_quantile_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features: per-feature × per-regime × per-quantile drill-down")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  TFs: {args.tfs}  Quantiles: {args.quantiles}")
    print(f"{'='*70}")

    concepts = args.concepts or FEATURE_NAMES_V2
    concepts = [c for c in concepts if c in FEATURE_NAMES_V2]

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

    close = full['close'].values.astype(np.float64)
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    regimes = full['regime_2d'].values.astype(str)

    # Sweep
    print(f"\n--- Sweeping {len(concepts)} concepts × {len(args.tfs)} TFs × "
          f"{len(REGIME_2D_ORDER)} regimes × {args.quantiles} quantiles ---")
    cell_rows = []
    shape_rows = []
    for concept in concepts:
        for tf in args.tfs:
            col = feature_column_for(concept, tf)
            if col not in full.columns:
                continue
            v = full[col].values.astype(np.float64)
            for regime in REGIME_2D_ORDER:
                mask = regimes == regime
                if mask.sum() < args.quantiles * args.min_cell_n:
                    continue
                v_r = v[mask]
                fwd_r = fwd[mask]
                bins = quantile_bins_within(v_r, args.quantiles)
                quantile_means = []
                quantile_wrs = []
                for q in range(args.quantiles):
                    cell_mask = (bins == q) & ~np.isnan(fwd_r)
                    n_cell = int(cell_mask.sum())
                    if n_cell < args.min_cell_n:
                        quantile_means.append(float('nan'))
                        quantile_wrs.append(float('nan'))
                        continue
                    f_c = fwd_r[cell_mask]
                    mean_fwd = float(f_c.mean())
                    wr = float((f_c > 0).mean())
                    cell_rows.append({
                        'concept': concept,
                        'tf': tf,
                        'regime_2d': regime,
                        'quantile': q,
                        'n': n_cell,
                        'feat_mean': float(np.mean(v_r[cell_mask])),
                        'mean_fwd': mean_fwd,
                        'std_fwd': float(f_c.std(ddof=1)) if n_cell > 1 else 0.0,
                        'win_rate': wr,
                    })
                    quantile_means.append(mean_fwd)
                    quantile_wrs.append(wr)
                # Shape classification
                shape = classify_shape(quantile_means, min_step=args.shape_min_step)
                valid_means = [m for m in quantile_means if not np.isnan(m)]
                signal_strength = (max(valid_means) - min(valid_means)) \
                    if len(valid_means) >= 2 else float('nan')
                shape_rows.append({
                    'concept': concept,
                    'tf': tf,
                    'regime_2d': regime,
                    'shape': shape,
                    'signal_strength': signal_strength,
                    'q0_fwd': quantile_means[0] if len(quantile_means) > 0 else float('nan'),
                    f'q{args.quantiles - 1}_fwd':
                        quantile_means[args.quantiles - 1]
                        if len(quantile_means) >= args.quantiles else float('nan'),
                    'wr_range': (max(w for w in quantile_wrs if not np.isnan(w)) -
                                  min(w for w in quantile_wrs if not np.isnan(w)))
                        if len([w for w in quantile_wrs if not np.isnan(w)]) >= 2
                        else float('nan'),
                })

    cell_df = pd.DataFrame(cell_rows)
    shape_df = pd.DataFrame(shape_rows)

    cell_path = os.path.join(args.output_dir, 'quantile_summary.csv')
    cell_df.to_csv(cell_path, index=False)
    print(f"  [saved] {cell_path} ({len(cell_df)} cells)")

    shape_path = os.path.join(args.output_dir, 'shape_summary.csv')
    shape_df.to_csv(shape_path, index=False)
    print(f"  [saved] {shape_path}")

    # Pivot: concept_TF × regime → signal_strength
    shape_df['concept_tf'] = shape_df['concept'] + '_' + shape_df['tf']
    pv = shape_df.pivot(index='concept_tf', columns='regime_2d',
                          values='signal_strength')
    pv = pv.reindex(columns=[r for r in REGIME_2D_ORDER if r in pv.columns])
    pv_path = os.path.join(args.output_dir, 'pivot_signal_strength.csv')
    pv.to_csv(pv_path)
    print(f"  [saved] {pv_path}")

    # Top features by max signal_strength across regimes
    pv['max_strength'] = pv.max(axis=1)
    top_strength = pv.sort_values('max_strength', ascending=False).head(20)
    print(f"\n  Top 20 (concept_tf) by max signal_strength across regimes "
          f"(quantile-curve range in ticks):")
    print(f"    {'concept_tf':>40}  " +
            "  ".join(f"{r:>12}" for r in REGIME_2D_ORDER))
    for cn_tf, row in top_strength.iterrows():
        line = f"    {cn_tf:>40}  "
        for r in REGIME_2D_ORDER:
            v = row.get(r, np.nan)
            line += f"{v:>+12.2f}  " if not pd.isna(v) else f"{'-':>12}  "
        print(line)

    # Top "shape changers": concept_tf where shape DIFFERS across regimes
    print(f"\n--- Top shape changers: concept_tf where curve shape differs by regime ---")
    shape_changes = []
    for (concept, tf), sub in shape_df.groupby(['concept', 'tf']):
        shapes = sub['shape'].unique().tolist()
        n_distinct_shapes = len(set(shapes))
        # Filter out trivial "all noisy"
        if n_distinct_shapes <= 1:
            continue
        shape_changes.append({
            'concept': concept,
            'tf': tf,
            'n_distinct_shapes': n_distinct_shapes,
            'shape_list': ','.join(sub.sort_values('regime_2d')['shape'].tolist()),
            'max_strength': float(sub['signal_strength'].max()),
        })
    sc_df = pd.DataFrame(shape_changes).sort_values(
        ['n_distinct_shapes', 'max_strength'], ascending=[False, False])
    sc_path = os.path.join(args.output_dir, 'top_shape_changers.csv')
    sc_df.to_csv(sc_path, index=False)
    print(f"  [saved] {sc_path}")
    print(f"  Top 20 shape-changers:")
    for _, r in sc_df.head(20).iterrows():
        print(f"    {r['concept']:>22}  {r['tf']:>4}  "
              f"distinct={int(r['n_distinct_shapes'])}  "
              f"max_strength={r['max_strength']:.2f}  "
              f"shapes={r['shape_list']}")

    # Drilldown: top 3 by max signal_strength — show their per-regime curves
    print(f"\n--- Top 3 by signal strength: per-regime quantile curves ---")
    for cn_tf, row in top_strength.head(3).iterrows():
        # Find concept and tf
        concept_tf = cn_tf
        # Reconstruct: concept could have underscores; try matching
        for tf in args.tfs:
            if concept_tf.endswith('_' + tf):
                concept = concept_tf[:-len('_' + tf)]
                break
        else:
            continue
        sub_cells = cell_df[(cell_df['concept'] == concept) &
                              (cell_df['tf'] == tf)]
        if len(sub_cells) == 0:
            continue
        print(f"\n  {concept_tf}:")
        print(f"    {'regime':>14}  " +
                "  ".join(f"Q{q}".rjust(8) for q in range(args.quantiles)))
        for regime in REGIME_2D_ORDER:
            sub_r = sub_cells[sub_cells['regime_2d'] == regime].sort_values('quantile')
            if len(sub_r) == 0:
                continue
            qs = sub_r['quantile'].tolist()
            means = sub_r['mean_fwd'].tolist()
            line = f"    {regime:>14}  "
            for q in range(args.quantiles):
                if q in qs:
                    idx = qs.index(q)
                    line += f"{means[idx]:>+8.2f}  "
                else:
                    line += f"{'-':>8}  "
            print(line)

    # Per-regime heatmap: concept × quantile
    for regime in REGIME_2D_ORDER:
        sub = cell_df[cell_df['regime_2d'] == regime]
        if len(sub) == 0:
            continue
        # Average across TFs to get a concept × quantile heatmap
        sub['concept_tf'] = sub['concept'] + '_' + sub['tf']
        pivot_h = sub.pivot(index='concept_tf', columns='quantile',
                              values='mean_fwd')
        if pivot_h.size == 0 or np.all(np.isnan(pivot_h.values)):
            continue
        # Sort rows by row-range for readability
        pivot_h['_range'] = pivot_h.max(axis=1) - pivot_h.min(axis=1)
        pivot_h = pivot_h.sort_values('_range', ascending=False).head(40)
        pivot_h = pivot_h.drop(columns='_range')
        fig, ax = plt.subplots(figsize=(10, 12))
        vmax = float(np.nanmax(np.abs(pivot_h.values)))
        im = ax.imshow(pivot_h.values, cmap='RdBu_r', aspect='auto',
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(args.quantiles))
        ax.set_xticklabels([f'Q{q}' for q in range(args.quantiles)])
        ax.set_yticks(range(len(pivot_h.index)))
        ax.set_yticklabels(pivot_h.index, fontsize=7)
        for i in range(len(pivot_h.index)):
            for j in range(args.quantiles):
                v = pivot_h.iloc[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=6,
                          color='white' if abs(v) > vmax / 2 else 'black')
        ax.set_xlabel('Quantile (within regime)')
        ax.set_ylabel('Concept_TF (top 40 by range)')
        ax.set_title(f'Mean forward return per quantile — regime={regime}\n'
                      f'vmax={vmax:.2f} ticks')
        plt.colorbar(im, ax=ax, label='mean_fwd (ticks)')
        fig.tight_layout()
        heat_path = os.path.join(args.output_dir, f'heatmap_{regime}.png')
        fig.savefig(heat_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # Markdown
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features: per-feature × per-regime × per-quantile — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**TFs:** {args.tfs}  **Quantiles:** {args.quantiles}  "
                f"**Split:** {args.split}\n\n")
        f.write("## Top 30 (concept_tf) by max signal_strength across regimes\n\n")
        f.write(top_strength.head(30).round(2).to_string())
        f.write("\n\n## Top 30 shape changers\n\n")
        f.write(sc_df.head(30).to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
