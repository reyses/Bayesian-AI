"""
v2_features_triplet_oos_validate.py — OOS validation of Layer C1 triplet
cells.

Layer C1 found cells with |lift| up to ~$100 ticks above regime baseline
on IS data (208 days, 47k bars). Quantile-binned cells ARE selection-
biased on IS (we picked the bin where mean_fwd was extreme).

This tool tests honest survival on OOS:
  1. For each top-K (concept, tf, regime, q_a, q_x, q_y) cell from IS:
     - Reuse the IS regime-local quantile boundaries (NOT recompute on OOS)
     - Apply to OOS data
     - Measure cell-mean, win rate, n on OOS
  2. Compare IS lift vs OOS lift, IS WR vs OOS WR, IS n vs OOS n.
  3. Flag cells whose OOS lift is in the same direction as IS (sign holds)
     and at least 30% of IS magnitude (rough survival threshold).
  4. Bootstrap CI on OOS cell-mean.

A cell survives OOS if: sign(OOS_lift) == sign(IS_lift) AND |OOS_lift| >=
0.30 * |IS_lift| AND n_oos >= 30 AND WR_oos in (0.45, 0.85) for sane
trade frequency.

Outputs:
  reports/findings/v2_features_triplet_oos/
    oos_validation.csv     (full table: IS metrics + OOS metrics + survival)
    surviving_cells.csv    cells that pass survival rule
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

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_tf_sweep_eda import feature_column_for


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int = 2000,
                        ci_level: float = 0.95) -> tuple[float, float]:
    if len(values) < 5:
        return float('nan'), float('nan')
    rng = np.random.default_rng(42)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(values), len(values))
        boots[i] = values[idx].mean()
    lo = float(np.quantile(boots, (1 - ci_level) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci_level) / 2))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--is-summary',
                        default='reports/findings/v2_features_triplet_regime/'
                                  'top_lift_cells.csv')
    parser.add_argument('--top-k-per-regime', type=int, default=20,
                        help='Validate top K IS cells per regime')
    parser.add_argument('--quantiles', type=int, default=3)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--min-oos-n', type=int, default=20)
    parser.add_argument('--survival-frac', type=float, default=0.30,
                        help='OOS lift must be >= survival_frac * IS lift '
                             'in same direction')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_triplet_oos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  Layer C1 triplet OOS validation")
    print(f"  IS top-K per regime: {args.top_k_per_regime}")
    print(f"  Survival rule: same sign + |OOS lift| >= {args.survival_frac} * |IS lift|")
    print(f"{'='*70}")

    # Load IS top-lift cells
    print(f"\n--- Loading IS top-lift cells ---")
    is_df = pd.read_csv(args.is_summary)
    print(f"  Loaded {len(is_df)} IS cells from {args.is_summary}")

    # Take top K per regime
    is_top = (is_df.sort_values('abs_lift', ascending=False)
                    .groupby('regime_2d').head(args.top_k_per_regime)
                    .reset_index(drop=True))
    print(f"  After top-{args.top_k_per_regime} per regime: {len(is_top)} cells")

    # Load IS data + OOS data
    print(f"\n--- Loading data ---")

    def load_for_split(data_root: str, cache_dir: str,
                         labels_df: pd.DataFrame, split_filter: str | None):
        base_df = load_atlas_tf(data_root, args.base_tf)
        if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
            ts_int = base_df['timestamp'].astype('int64') // 10**9
        else:
            ts_int = base_df['timestamp'].astype(np.int64)
        base_df = base_df.copy()
        base_df['ts_int'] = ts_int
        dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert(
            'America/Los_Angeles')
        base_df['date'] = dt_la.dt.date.astype(str)
        if split_filter is None:
            merged = base_df.merge(
                labels_df[['date', 'regime_2d']],
                on='date', how='inner')
        else:
            merged = base_df.merge(
                labels_df[['date', 'regime_2d', 'split']],
                on='date', how='inner')
            merged = merged[merged['split'] == split_filter].reset_index(drop=True)
        ts_int = merged['ts_int'].values.astype(np.int64)
        feats = load_v2_features(
            v2_dir=cache_dir, atlas_root=data_root, day_strs=None,
            ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False)
        aligned = align_v2_to_base_tf(feats, ts_int)
        full = pd.concat([merged.reset_index(drop=True),
                           aligned.reset_index(drop=True)], axis=1)
        return full

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]

    # IS - to recompute the regime-local quantile bin edges
    print(f"\n  Loading IS for quantile boundary recomputation...")
    is_full = load_for_split(args.data, args.cache, labels_df, 'IS')
    print(f"  IS bars: {len(is_full):,}")

    # OOS - apply IS quantile boundaries (subset of same data root via split)
    print(f"\n  Loading OOS (subset of DATA/ATLAS by split column)...")
    oos_full = load_for_split(args.data, args.cache, labels_df, 'OOS')
    print(f"  OOS bars: {len(oos_full):,}")

    # OOS regime baselines
    oos_close = oos_full['close'].values.astype(np.float64)
    n_oos = len(oos_close)
    fwd_oos = np.full(n_oos, np.nan)
    if n_oos > args.forward_n:
        fwd_oos[:-args.forward_n] = (oos_close[args.forward_n:]
                                          - oos_close[:-args.forward_n])
    regimes_oos = oos_full['regime_2d'].values.astype(str)

    oos_baseline = {}
    for regime in REGIME_2D_ORDER:
        mask = (regimes_oos == regime) & ~np.isnan(fwd_oos)
        if mask.sum() < 30:
            continue
        oos_baseline[regime] = float(fwd_oos[mask].mean())

    print(f"\n  OOS regime baselines:")
    for r, m in oos_baseline.items():
        n_r = int((regimes_oos == r).sum())
        print(f"    {r:>14}: {m:>+8.2f}  (n_bars={n_r})")

    # IS regime baselines (for reference; we use IS lift directly from CSV)
    is_close = is_full['close'].values.astype(np.float64)
    n_is = len(is_close)
    fwd_is = np.full(n_is, np.nan)
    if n_is > args.forward_n:
        fwd_is[:-args.forward_n] = is_close[args.forward_n:] - is_close[:-args.forward_n]
    regimes_is = is_full['regime_2d'].values.astype(str)

    # For each unique (concept, tf, regime) in IS, compute regime-local
    # quantile boundaries from IS, then use them on OOS.
    print(f"\n--- Validating cells on OOS ---")

    needed_cols = set()
    for _, r in is_top.iterrows():
        needed_cols.add((r['anchor_concept'], r['anchor_tf']))
        needed_cols.add((r['x_concept'], r['x_tf']))
        needed_cols.add((r['y_concept'], r['y_tf']))

    quantile_edges = {}  # (concept, tf, regime) -> array of boundaries
    for (concept, tf) in needed_cols:
        col = feature_column_for(concept, tf)
        if col not in is_full.columns:
            print(f"  Warning: column {col} missing from IS")
            continue
        v_is = is_full[col].values.astype(np.float64)
        for regime in REGIME_2D_ORDER:
            regime_mask_is = (regimes_is == regime)
            if regime_mask_is.sum() < 200:
                continue
            v_r = v_is[regime_mask_is]
            valid = ~np.isnan(v_r)
            if valid.sum() < args.quantiles * 5:
                continue
            qs = np.quantile(v_r[valid], np.linspace(0, 1, args.quantiles + 1))
            qs[0] -= 1e-9
            qs[-1] += 1e-9
            quantile_edges[(concept, tf, regime)] = qs

    # Per-cell OOS measurement
    rows = []
    for idx, r in is_top.iterrows():
        regime = r['regime_2d']
        if regime not in oos_baseline:
            continue
        a_concept, a_tf = r['anchor_concept'], r['anchor_tf']
        x_concept, x_tf = r['x_concept'], r['x_tf']
        y_concept, y_tf = r['y_concept'], r['y_tf']
        qa, qx, qy = int(r['q_anchor']), int(r['q_x']), int(r['q_y'])

        a_col = feature_column_for(a_concept, a_tf)
        x_col = feature_column_for(x_concept, x_tf)
        y_col = feature_column_for(y_concept, y_tf)
        if any(c not in oos_full.columns for c in [a_col, x_col, y_col]):
            continue

        a_edges = quantile_edges.get((a_concept, a_tf, regime))
        x_edges = quantile_edges.get((x_concept, x_tf, regime))
        y_edges = quantile_edges.get((y_concept, y_tf, regime))
        if a_edges is None or x_edges is None or y_edges is None:
            continue

        v_a = oos_full[a_col].values.astype(np.float64)
        v_x = oos_full[x_col].values.astype(np.float64)
        v_y = oos_full[y_col].values.astype(np.float64)

        # bin OOS using IS edges
        bin_a = np.digitize(v_a, a_edges[1:-1])
        bin_x = np.digitize(v_x, x_edges[1:-1])
        bin_y = np.digitize(v_y, y_edges[1:-1])

        cell_mask = ((regimes_oos == regime) & (bin_a == qa)
                          & (bin_x == qx) & (bin_y == qy)
                          & ~np.isnan(v_a) & ~np.isnan(v_x) & ~np.isnan(v_y)
                          & ~np.isnan(fwd_oos))
        n_oos_cell = int(cell_mask.sum())
        if n_oos_cell < args.min_oos_n:
            rows.append({
                'anchor_concept': a_concept, 'anchor_tf': a_tf,
                'x_concept': x_concept, 'x_tf': x_tf,
                'y_concept': y_concept, 'y_tf': y_tf,
                'regime_2d': regime,
                'cell': f'{qa},{qx},{qy}',
                'is_n': int(r['n']),
                'is_mean': float(r['cell_mean']),
                'is_lift': float(r['lift']),
                'is_wr': float(r['win_rate']),
                'oos_n': n_oos_cell,
                'oos_mean': float('nan'),
                'oos_lift': float('nan'),
                'oos_wr': float('nan'),
                'oos_ci_lo': float('nan'),
                'oos_ci_hi': float('nan'),
                'oos_baseline': oos_baseline[regime],
                'survives': False,
                'reason': f'n_oos < {args.min_oos_n}',
            })
            continue

        f = fwd_oos[cell_mask]
        oos_mean = float(f.mean())
        oos_wr = float((f > 0).mean())
        oos_lift = oos_mean - oos_baseline[regime]
        ci_lo, ci_hi = bootstrap_mean_ci(f)

        is_lift = float(r['lift'])
        same_sign = np.sign(oos_lift) == np.sign(is_lift)
        magnitude_ok = abs(oos_lift) >= args.survival_frac * abs(is_lift)
        survives = same_sign and magnitude_ok and n_oos_cell >= args.min_oos_n
        reason = 'survives' if survives else (
            'sign_flip' if not same_sign else 'mag_too_low')

        rows.append({
            'anchor_concept': a_concept, 'anchor_tf': a_tf,
            'x_concept': x_concept, 'x_tf': x_tf,
            'y_concept': y_concept, 'y_tf': y_tf,
            'regime_2d': regime,
            'cell': f'{qa},{qx},{qy}',
            'is_n': int(r['n']),
            'is_mean': float(r['cell_mean']),
            'is_lift': is_lift,
            'is_wr': float(r['win_rate']),
            'oos_n': n_oos_cell,
            'oos_mean': oos_mean,
            'oos_lift': oos_lift,
            'oos_wr': oos_wr,
            'oos_ci_lo': ci_lo,
            'oos_ci_hi': ci_hi,
            'oos_baseline': oos_baseline[regime],
            'survives': survives,
            'reason': reason,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(args.output_dir, 'oos_validation.csv'),
                    index=False)
    print(f"\n  [saved] oos_validation.csv ({len(out_df)} cells)")

    # Summary stats
    n_cells = len(out_df)
    n_with_data = (out_df['oos_n'] >= args.min_oos_n).sum()
    n_survives = out_df['survives'].sum()
    n_sign_flip = (out_df['reason'] == 'sign_flip').sum()
    n_mag_low = (out_df['reason'] == 'mag_too_low').sum()
    n_no_data = (out_df['reason'] == f'n_oos < {args.min_oos_n}').sum()

    print(f"\n  Survival summary:")
    print(f"    Total IS top cells:    {n_cells}")
    print(f"    With OOS data:         {n_with_data}")
    print(f"    SURVIVE OOS:           {n_survives} ({100.0 * n_survives / max(n_cells, 1):.1f}%)")
    print(f"    sign flips:            {n_sign_flip}")
    print(f"    magnitude too low:     {n_mag_low}")
    print(f"    no OOS data:           {n_no_data}")

    surv = out_df[out_df['survives']]
    surv.to_csv(os.path.join(args.output_dir, 'surviving_cells.csv'),
                  index=False)

    print(f"\n  Top 30 by IS lift, with OOS results:")
    print(f"    {'anchor':>20}  {'X':>20}  {'Y':>20}  {'regime':>14}  "
          f"{'cell':>5}  {'is_n':>4} {'is_lift':>7}  "
          f"{'oos_n':>5} {'oos_lift':>8} {'oos_ci':>14} {'surv':>4}")
    sorted_df = out_df.copy()
    sorted_df['abs_is_lift'] = sorted_df['is_lift'].abs()
    for _, r in sorted_df.sort_values('abs_is_lift',
                                            ascending=False).head(30).iterrows():
        a = f"{r['anchor_concept']}_{r['anchor_tf']}"
        x = f"{r['x_concept']}_{r['x_tf']}"
        y = f"{r['y_concept']}_{r['y_tf']}"
        cell = r['cell']
        oos_lift_s = f"{r['oos_lift']:>+8.2f}" if not np.isnan(r['oos_lift']) else f"{'nan':>8}"
        ci_s = (f"[{r['oos_ci_lo']:>+5.1f},{r['oos_ci_hi']:>+5.1f}]"
                  if not np.isnan(r['oos_ci_lo']) else f"{'no data':>14}")
        print(f"    {a:>20}  {x:>20}  {y:>20}  {r['regime_2d']:>14}  "
              f"{cell:>5}  {int(r['is_n']):>4} {r['is_lift']:>+7.2f}  "
              f"{int(r['oos_n']):>5} {oos_lift_s} {ci_s} "
              f"{'YES' if r['survives'] else 'no':>4}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Layer C1 triplet OOS validation - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Method:** for each top-K IS triplet cell per regime,\n"
                f"reuse IS regime-local quantile boundaries on OOS data,\n"
                f"measure cell-mean / win-rate / n on OOS, compute lift "
                f"vs OOS regime baseline.\n\n")
        f.write(f"**Survival rule:** sign(OOS lift) == sign(IS lift) AND "
                f"|OOS lift| >= {args.survival_frac} * |IS lift| "
                f"AND n_oos >= {args.min_oos_n}.\n\n")
        f.write(f"## Survival summary\n\n")
        f.write(f"- Total IS top cells: **{n_cells}**\n")
        f.write(f"- With OOS data: **{n_with_data}**\n")
        f.write(f"- **Survive OOS: {n_survives} ({100.0 * n_survives / max(n_cells, 1):.1f}%)**\n")
        f.write(f"- Sign flips: {n_sign_flip}\n")
        f.write(f"- Magnitude too low: {n_mag_low}\n")
        f.write(f"- No OOS data: {n_no_data}\n\n")
        f.write(f"## OOS regime baselines\n\n")
        for r, m in oos_baseline.items():
            f.write(f"- {r}: {m:+.2f}\n")
        f.write(f"\n## Top 30 IS cells with OOS validation\n\n")
        f.write(sorted_df.sort_values('abs_is_lift',
                                            ascending=False).head(30).to_string(index=False))
        f.write(f"\n\n## Surviving cells\n\n")
        f.write(surv.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
