"""
v2_features_drift_eda.py — Regime drift / calendar stability (Step #5).

Does (feature, TF, regime) behave the same in early IS vs late IS?
If signals shift over time, prior layer findings could be averaging
across non-stationary regimes.

For each (concept, TF, regime), split IS into halves by date and compute:
  - Q4 mean_fwd in early half
  - Q4 mean_fwd in late half
  - delta = late - early
  - sign_stable: same sign in both halves?
  - magnitude_stable: |delta| / |max(|early|, |late|)| < 0.5

Also computes correlation drift:
  - corr(feature_t, fwd) in early half
  - corr(feature_t, fwd) in late half
  - corr_delta

Stratified by regime — drift could be regime-specific.

Outputs:
  reports/findings/v2_features_drift/
    drift_summary.csv   (concept, tf, regime, q4_early, q4_late, delta, ...)
    sign_flips.csv      cells where sign FLIPS between halves
    summary.md
    plot_drift_<concept>_<tf>.png  early vs late scatter colored by regime
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
    parser.add_argument('--quantile', type=int, default=4)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--n-halves', type=int, default=2)
    parser.add_argument('--min-cell-n', type=int, default=30)
    parser.add_argument('--top-plots', type=int, default=10)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_drift')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features regime drift (Step #5)")
    print(f"  Concepts: {args.concepts}")
    print(f"  TFs: {args.tfs}")
    print(f"  Halves: {args.n_halves}; tracking Q{args.quantile}")
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

    # Date ordering
    unique_dates = sorted(merged['date'].unique())
    print(f"  After split={args.split}: {len(merged):,} bars over "
          f"{len(unique_dates)} days")
    half_size = len(unique_dates) // args.n_halves
    halves = []
    for h in range(args.n_halves):
        start_idx = h * half_size
        end_idx = (h + 1) * half_size if h < args.n_halves - 1 else len(unique_dates)
        half_dates = set(unique_dates[start_idx:end_idx])
        halves.append((f'half_{h+1}', half_dates))
        print(f"    {halves[-1][0]}: {len(half_dates)} days "
              f"({unique_dates[start_idx]} -> {unique_dates[end_idx-1]})")

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
    dates = full['date'].values.astype(str)

    # Pre-compute half masks
    half_masks = {}
    for hname, hdates in halves:
        half_masks[hname] = np.array([d in hdates for d in dates])

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

                # Compute global (regime-wide) quantile bounds for this feature
                v_r = v[regime_mask]
                valid = ~np.isnan(v_r)
                if valid.sum() < args.quantiles * 5:
                    continue
                qs = np.quantile(v_r[valid], np.linspace(0, 1, args.quantiles + 1))
                qs[0] -= 1e-9
                qs[-1] += 1e-9
                bin_idx = np.digitize(v, qs[1:-1])

                target_mask = (regime_mask & (bin_idx == args.quantile)
                                  & ~np.isnan(v) & ~np.isnan(fwd))

                # Per-half stats
                row = {
                    'concept': concept,
                    'tf': tf,
                    'regime_2d': regime,
                }
                half_results = {}
                for hname, hmask in half_masks.items():
                    hcell = target_mask & hmask
                    n_cell = int(hcell.sum())
                    valid_corr = (regime_mask & hmask & ~np.isnan(v)
                                      & ~np.isnan(fwd))
                    if n_cell < args.min_cell_n or valid_corr.sum() < 100:
                        half_results[hname] = None
                        continue
                    f = fwd[hcell]
                    half_results[hname] = {
                        'n': n_cell,
                        'mean_fwd': float(f.mean()),
                        'std_fwd': float(f.std(ddof=1)),
                        'win_rate': float((f > 0).mean()),
                        'corr': float(np.corrcoef(
                            v[valid_corr], fwd[valid_corr])[0, 1]),
                    }

                # only emit if both halves have data
                if any(v is None for v in half_results.values()):
                    continue
                if len(half_masks) != 2:
                    continue
                h1, h2 = halves[0][0], halves[1][0]
                e = half_results[h1]
                l = half_results[h2]
                row.update({
                    'q4_early_n': e['n'],
                    'q4_early_mean': e['mean_fwd'],
                    'q4_early_wr': e['win_rate'],
                    'q4_late_n': l['n'],
                    'q4_late_mean': l['mean_fwd'],
                    'q4_late_wr': l['win_rate'],
                    'delta_mean': l['mean_fwd'] - e['mean_fwd'],
                    'sign_flip': (np.sign(e['mean_fwd']) != np.sign(l['mean_fwd'])
                                       and abs(e['mean_fwd']) > 1.0
                                       and abs(l['mean_fwd']) > 1.0),
                    'corr_early': e['corr'],
                    'corr_late': l['corr'],
                    'corr_delta': l['corr'] - e['corr'],
                    'corr_sign_flip': (np.sign(e['corr']) != np.sign(l['corr'])
                                            and abs(e['corr']) > 0.02
                                            and abs(l['corr']) > 0.02),
                })
                # magnitude stability: |delta| / max(|early|, |late|)
                max_mag = max(abs(e['mean_fwd']), abs(l['mean_fwd']), 1e-9)
                row['mag_stability'] = 1.0 - abs(row['delta_mean']) / max_mag
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'drift_summary.csv'), index=False)
    print(f"  [saved] drift_summary.csv ({len(df)} cells)")

    # Sign flips
    flips = df[df['sign_flip']].copy()
    flips.to_csv(os.path.join(args.output_dir, 'sign_flips.csv'), index=False)
    print(f"  [saved] sign_flips.csv ({len(flips)} cells)")

    print(f"\n  Sign-flip summary: {len(flips)} of {len(df)} "
          f"({100.0 * len(flips) / max(len(df), 1):.1f}%) cells flip sign "
          f"between halves")

    if len(flips) > 0:
        flips_sorted = flips.copy()
        flips_sorted['abs_max'] = flips_sorted[['q4_early_mean',
                                                       'q4_late_mean']].abs().max(axis=1)
        flips_sorted = flips_sorted.sort_values('abs_max', ascending=False)
        print(f"\n  Top 20 sign flips (by max magnitude):")
        print(f"    {'concept':>20}  {'tf':>4}  {'regime':>14}  "
              f"{'early':>8} {'late':>8} {'delta':>8}")
        for _, r in flips_sorted.head(20).iterrows():
            print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
                  f"{r['q4_early_mean']:>+8.2f} {r['q4_late_mean']:>+8.2f} "
                  f"{r['delta_mean']:>+8.2f}")

    # Most-stable cells: large magnitude AND high mag_stability
    df['min_abs_mean'] = df[['q4_early_mean', 'q4_late_mean']].abs().min(axis=1)
    stable = df[(df['min_abs_mean'] > 5.0) & (df['mag_stability'] > 0.6)
                  & ~df['sign_flip']].sort_values(
        'min_abs_mean', ascending=False)
    print(f"\n  Most STABLE high-magnitude cells: {len(stable)}")
    print(f"    {'concept':>20}  {'tf':>4}  {'regime':>14}  "
          f"{'early':>8} {'late':>8} {'stab':>5} {'corr_e':>6} {'corr_l':>6}")
    for _, r in stable.head(20).iterrows():
        print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['q4_early_mean']:>+8.2f} {r['q4_late_mean']:>+8.2f} "
              f"{r['mag_stability']:>5.2f} {r['corr_early']:>+6.3f} "
              f"{r['corr_late']:>+6.3f}")

    # Correlation sign flips
    corr_flips = df[df['corr_sign_flip']]
    print(f"\n  Correlation sign flips: {len(corr_flips)} cells "
          f"({100.0 * len(corr_flips) / max(len(df), 1):.1f}%)")
    if len(corr_flips) > 0:
        cf_sorted = corr_flips.copy()
        cf_sorted['abs_corr_max'] = cf_sorted[['corr_early',
                                                   'corr_late']].abs().max(axis=1)
        cf_sorted = cf_sorted.sort_values('abs_corr_max', ascending=False)
        for _, r in cf_sorted.head(15).iterrows():
            print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
                  f"corr_e={r['corr_early']:>+6.3f}  corr_l={r['corr_late']:>+6.3f}")

    # Plot top concept_tfs: early vs late
    plotted = 0
    df['concept_tf'] = df['concept'] + '_' + df['tf']
    top_concept_tfs = (df.groupby('concept_tf')
                          .apply(lambda x: x[['q4_early_mean',
                                                'q4_late_mean']].abs().max().max())
                          .sort_values(ascending=False)
                          .head(args.top_plots))
    for cn_tf in top_concept_tfs.index:
        sub = df[df['concept_tf'] == cn_tf]
        if sub.empty or len(sub) < 3:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(REGIME_2D_ORDER)))
        for ri, regime in enumerate(REGIME_2D_ORDER):
            ssub = sub[sub['regime_2d'] == regime]
            if ssub.empty:
                continue
            ax.scatter(ssub['q4_early_mean'], ssub['q4_late_mean'], s=120,
                        c=[colors[ri]], label=regime, alpha=0.85,
                        edgecolors='black', linewidth=0.7)
        lim = max(abs(sub['q4_early_mean']).max(),
                    abs(sub['q4_late_mean']).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k:', alpha=0.4, label='stable y=x')
        ax.axhline(0, color='black', alpha=0.3)
        ax.axvline(0, color='black', alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'Q{args.quantile} mean_fwd in EARLY IS')
        ax.set_ylabel(f'Q{args.quantile} mean_fwd in LATE IS')
        ax.set_title(f'{cn_tf} — early vs late IS by regime')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(args.output_dir,
                                   f'plot_drift_{cn_tf}.png')
        fig.savefig(png_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plotted += 1
    print(f"\n  [saved] {plotted} drift plots")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features regime drift (Step #5) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {args.concepts}\n\n")
        f.write(f"**TFs:** {args.tfs}\n\n")
        f.write(f"**Half boundaries:**\n")
        for hname, hdates in halves:
            sd = sorted(list(hdates))
            f.write(f"- {hname}: {len(hdates)} days ({sd[0]} -> {sd[-1]})\n")
        f.write(f"\n**Sign-flip rate:** {len(flips)}/{len(df)} "
                f"({100.0 * len(flips) / max(len(df), 1):.1f}%)\n\n")
        f.write(f"**Correlation sign-flip rate:** {len(corr_flips)}/{len(df)} "
                f"({100.0 * len(corr_flips) / max(len(df), 1):.1f}%)\n\n")
        f.write("## Top sign-flip cells\n\n")
        if len(flips) > 0:
            f.write(flips_sorted.head(30).to_string(index=False))
        f.write("\n\n## Most-stable high-magnitude cells\n\n")
        f.write(stable.head(30).to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
