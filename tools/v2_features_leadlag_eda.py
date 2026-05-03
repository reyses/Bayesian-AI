"""
v2_features_leadlag_eda.py — Lead-lag analysis (Step #3).

Does the feature LEAD price (predictive) or LAG price (confirmation)?

For each (feature, TF, regime), compute Pearson corr between
  feature value at bar t  vs  realized return over [t+s, t+s+window]
for s in {-12, -8, -4, -2, -1, 0, +1, +2, +4, +8, +12} bars.

  s < 0  : feature TODAY correlates with PRICE MOVE before now (lagging)
  s = 0  : contemporaneous
  s > 0  : feature TODAY correlates with FUTURE price move (leading)

A feature with peak |corr| at s>0 is genuinely predictive.
A feature with peak |corr| at s<0 is a momentum-confirmation indicator,
not a forecaster.
A feature with |corr| highest at s=0 captures concurrent state.

Stratified by regime — a feature can lead in trending regimes but lag
in chop, or vice versa.

Outputs:
  reports/findings/v2_features_leadlag/
    leadlag_corr.csv        (concept, tf, regime, shift_s, n, corr)
    peak_lag.csv            per (concept, tf, regime): which shift wins
    summary.md
    plot_leadlag_<concept>_<tf>.png  per regime: corr vs shift
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

# shifts relative to feature_t. negative = past, positive = future.
DEFAULT_SHIFTS = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]


def signed_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation that survives flat columns and small samples."""
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 30:
        return float('nan')
    xv, yv = x[valid], y[valid]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return float('nan')
    return float(np.corrcoef(xv, yv)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--concepts', nargs='+', default=DEFAULT_CONCEPTS)
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--shifts', nargs='+', type=int, default=DEFAULT_SHIFTS)
    parser.add_argument('--window', type=int, default=12,
                        help='Return window length (in base bars). Default '
                             '12 = 1h forward when base=5m')
    parser.add_argument('--split', default='IS')
    parser.add_argument('--top-plots', type=int, default=15)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_leadlag')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features lead-lag (Step #3)")
    print(f"  Concepts: {args.concepts}")
    print(f"  TFs: {args.tfs}")
    print(f"  Shifts (base bars): {args.shifts}")
    print(f"  Window: {args.window} base bars")
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
    dates = full['date'].values.astype(str)
    regimes = full['regime_2d'].values.astype(str)

    # Pre-compute returns at every shift: ret_at_shift_s[i] is the return
    # over bars [i+s, i+s+window]
    print(f"\n--- Computing shifted returns ---")
    ret_by_shift = {}
    for s in args.shifts:
        ret = np.full(n, np.nan)
        for i in range(n):
            j = i + s
            k = j + args.window
            if j < 0 or k >= n:
                continue
            # block return only if all bars within window are same date
            # (otherwise we'd straddle session boundaries)
            if dates[j] != dates[k]:
                continue
            ret[i] = close[k] - close[j]
        ret_by_shift[s] = ret

    # Sweep
    print(f"\n--- Sweeping {len(args.concepts)} concepts x {len(args.tfs)} TFs "
          f"x {len(REGIME_2D_ORDER)} regimes x {len(args.shifts)} shifts ---")

    rows = []
    peak_rows = []
    series_cache = {}  # (concept, tf, regime) -> {s: corr}

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
                shift_corrs = {}
                for s in args.shifts:
                    ret = ret_by_shift[s]
                    valid = regime_mask & ~np.isnan(v) & ~np.isnan(ret)
                    if valid.sum() < 100:
                        continue
                    c = signed_correlation(v[valid], ret[valid])
                    rows.append({
                        'concept': concept,
                        'tf': tf,
                        'regime_2d': regime,
                        'shift_s': s,
                        'n': int(valid.sum()),
                        'corr': c,
                    })
                    shift_corrs[s] = c

                if not shift_corrs:
                    continue
                series_cache[(concept, tf, regime)] = shift_corrs

                # find peak |corr|
                valid_pairs = [(s, c) for s, c in shift_corrs.items()
                                  if not np.isnan(c)]
                if not valid_pairs:
                    continue
                peak_s, peak_c = max(valid_pairs, key=lambda kv: abs(kv[1]))
                contemp_c = shift_corrs.get(0, float('nan'))

                if peak_s > 0:
                    role = 'leading'
                elif peak_s < 0:
                    role = 'lagging'
                else:
                    role = 'contemporaneous'

                peak_rows.append({
                    'concept': concept,
                    'tf': tf,
                    'regime_2d': regime,
                    'peak_shift': peak_s,
                    'peak_corr': peak_c,
                    'contemp_corr': contemp_c,
                    'role': role,
                    'lift_vs_contemp': (abs(peak_c) - abs(contemp_c) if
                                         not np.isnan(contemp_c) else float('nan')),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'leadlag_corr.csv'), index=False)
    print(f"  [saved] leadlag_corr.csv ({len(df)} rows)")

    peak_df = pd.DataFrame(peak_rows).sort_values(
        'peak_corr', key=lambda s: s.abs(), ascending=False)
    peak_df.to_csv(os.path.join(args.output_dir, 'peak_lag.csv'), index=False)
    print(f"  [saved] peak_lag.csv")

    # Summary tables
    print(f"\n  Top 25 (concept, TF, regime) by |peak_corr|:")
    print(f"    {'concept':>20}  {'tf':>4}  {'regime':>14}  {'shift':>6} "
          f"{'corr':>6} {'contemp':>7} {'role':>14}")
    for _, r in peak_df.head(25).iterrows():
        print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['peak_shift']:>+6} {r['peak_corr']:>6.3f} "
              f"{r['contemp_corr']:>7.3f} {r['role']:>14}")

    # Distribution of role across (concept, TF, regime)
    role_counts = peak_df['role'].value_counts()
    print(f"\n  Role distribution across {len(peak_df)} (concept, tf, regime) cells:")
    for r, c in role_counts.items():
        pct = 100.0 * c / len(peak_df)
        print(f"    {r:>16}: {c:>4}  ({pct:>5.1f}%)")

    # Genuinely-LEADING (s>0) cells with strong corr
    lead_strong = peak_df[(peak_df['role'] == 'leading')
                                & (peak_df['peak_corr'].abs() > 0.05)]
    print(f"\n  Genuinely-leading cells (s>0, |corr| > 0.05): {len(lead_strong)}")
    print(f"    {'concept':>20}  {'tf':>4}  {'regime':>14}  {'shift':>6} "
          f"{'corr':>6} {'contemp':>7}")
    for _, r in lead_strong.head(15).iterrows():
        print(f"    {r['concept']:>20}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['peak_shift']:>+6} {r['peak_corr']:>6.3f} "
              f"{r['contemp_corr']:>7.3f}")

    # Plot top concept_tf curves: one panel per regime
    plotted = 0
    df['concept_tf'] = df['concept'] + '_' + df['tf']
    top_concept_tfs = (peak_df.groupby(['concept', 'tf'])['peak_corr']
                          .apply(lambda x: x.abs().max())
                          .sort_values(ascending=False)
                          .head(args.top_plots))
    for (concept, tf), _ in top_concept_tfs.items():
        sub = df[(df['concept'] == concept) & (df['tf'] == tf)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for regime in REGIME_2D_ORDER:
            ssub = sub[sub['regime_2d'] == regime]
            if ssub.empty:
                continue
            ax.plot(ssub['shift_s'], ssub['corr'], marker='o',
                     label=regime, alpha=0.85)
        ax.axhline(0, color='black', linestyle='-', alpha=0.4)
        ax.axvline(0, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('shift s (base bars; <0 = past, >0 = future)')
        ax.set_ylabel(f'corr(feature_t, return over [t+s, t+s+{args.window}])')
        ax.set_title(f'{concept} {tf} — lead-lag corr by regime')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(args.output_dir,
                                   f'plot_leadlag_{concept}_{tf}.png')
        fig.savefig(png_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plotted += 1
    print(f"\n  [saved] {plotted} lead-lag plots")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features lead-lag (Step #3) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {args.concepts}\n\n")
        f.write(f"**TFs:** {args.tfs}\n\n")
        f.write(f"**Shifts (base bars):** {args.shifts}\n\n")
        f.write(f"**Window:** {args.window} base bars\n\n")
        f.write("## Role distribution\n\n")
        for r, c in role_counts.items():
            pct = 100.0 * c / len(peak_df)
            f.write(f"- **{r}**: {c} ({pct:.1f}%)\n")
        f.write("\n## Top 30 (concept, TF, regime) by |peak_corr|\n\n")
        f.write(peak_df.head(30).to_string(index=False))
        f.write("\n\n## Genuinely-leading cells (s>0)\n\n")
        f.write(lead_strong.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
