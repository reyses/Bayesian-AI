"""
v2_features_within_tf_leadlag_eda.py — Layer D5: feature x feature
within-TF LEAD-LAG (no price target).

Mirror of the price-track lead-lag: does feature X at time t correlate
with feature Y at time t+s? If peak |corr| at s>0, X LEADS Y. If peak
at s<0, X LAGS Y. If at s=0, they are contemporaneous.

This exposes asymmetric/forward pass structure that symmetric pearson misses.
For example, if vol_velocity_w leads price_velocity_w by N bars, then
volume changes precede price changes — actionable.

For each (X, Y, TF) and shift s in {-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12},
compute Pearson corr(X_t, Y_{t+s}) within IS. Take peak |corr|, peak shift,
classify role.

Stratified by regime — a pair can lead in UP_SMOOTH and lag in DOWN_SMOOTH.

Outputs:
  reports/findings/v2_features_within_tf_leadlag/
    leadlag_corr.csv     (X, Y, TF, regime, shift_s, corr, n)
    peak_lag.csv         per (X, Y, TF, regime): peak shift + role
    summary.md
    plot_<tf>_<XY>.png   per top pair: corr vs shift, one line per regime
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


CONCEPTS = [
    'price_velocity_1b', 'price_accel_1b',
    'vol_velocity_1b',   'vol_accel_1b',
    'bar_range', 'body',
    'price_velocity_w', 'price_accel_w',
    'vol_velocity_w',   'vol_accel_w',
    'price_mean_w', 'price_sigma_w',
    'vol_mean_w',   'vol_sigma_w',
    'vwap_w',
    'z_se_w', 'z_high_w', 'z_low_w',
    'SE_high_w', 'SE_low_w',
    'hurst_w', 'reversion_prob_w', 'swing_noise_w',
]
DEFAULT_TFS = ['5s', '1m', '5m', '15m', '1h']
DEFAULT_SHIFTS = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]


def safe_corr(x, y):
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 100:
        return float('nan'), 0
    xv, yv = x[valid], y[valid]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return float('nan'), int(valid.sum())
    return float(np.corrcoef(xv, yv)[0, 1]), int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--shifts', nargs='+', type=int, default=DEFAULT_SHIFTS)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--top-plots', type=int, default=15)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_within_tf_leadlag')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features within-TF LEAD-LAG x regime (Layer D5)")
    print(f"  TFs: {args.tfs}  Shifts: {args.shifts}")
    n_pairs = 23 * 22 // 2
    print(f"  {n_pairs} pairs * {len(args.tfs)} TFs * {len(REGIME_2D_ORDER)} "
          f"regimes * {len(args.shifts)} shifts = "
          f"{n_pairs * len(args.tfs) * len(REGIME_2D_ORDER) * len(args.shifts)} "
          f"corr computations")
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
    dates = full['date'].values.astype(str)
    n_total = len(full)

    rows = []
    peak_rows = []

    print(f"\n--- Sweeping shifts ---")
    for tf in args.tfs:
        # Pre-extract per-TF concept arrays
        tf_arrs = {}
        for c in CONCEPTS:
            col = feature_column_for(c, tf)
            if col in full.columns:
                tf_arrs[c] = full[col].values.astype(np.float64)

        present = list(tf_arrs.keys())
        if len(present) < 5:
            print(f"  {tf}: skipping ({len(present)} features)")
            continue

        for c1, c2 in itertools.combinations(present, 2):
            x = tf_arrs[c1]
            y = tf_arrs[c2]
            for regime in REGIME_2D_ORDER:
                regime_mask = (regimes == regime)
                if regime_mask.sum() < 200:
                    continue
                shift_corrs = {}
                for s in args.shifts:
                    if s == 0:
                        rmask = regime_mask
                        a, b = x, y
                    elif s > 0:
                        # corr(x_t, y_{t+s}) — pair (i, i+s) require same date
                        rmask = np.zeros(n_total, dtype=bool)
                        end = n_total - s
                        rmask[:end] = regime_mask[:end] & (dates[:end] == dates[s:s+end])
                        a, b = x, np.concatenate([y[s:], np.full(s, np.nan)])
                    else:  # s < 0
                        s_abs = -s
                        rmask = np.zeros(n_total, dtype=bool)
                        rmask[s_abs:] = regime_mask[s_abs:] & (dates[s_abs:] == dates[:n_total - s_abs])
                        a, b = x, np.concatenate([np.full(s_abs, np.nan), y[:n_total - s_abs]])
                    a_r, b_r = a[rmask], b[rmask]
                    r, n = safe_corr(a_r, b_r)
                    if np.isnan(r):
                        continue
                    rows.append({
                        'tf': tf,
                        'c1': c1,
                        'c2': c2,
                        'regime_2d': regime,
                        'shift_s': s,
                        'corr': r,
                        'n': n,
                    })
                    shift_corrs[s] = (r, n)

                if not shift_corrs:
                    continue
                # find peak |corr|
                valid = [(s, r) for s, (r, _) in shift_corrs.items()
                          if not np.isnan(r)]
                if not valid:
                    continue
                peak_s, peak_c = max(valid, key=lambda kv: abs(kv[1]))
                contemp_c = shift_corrs.get(0, (float('nan'), 0))[0]
                if peak_s > 0:
                    role = 'X_leads_Y'
                elif peak_s < 0:
                    role = 'X_lags_Y'
                else:
                    role = 'contemporaneous'
                peak_rows.append({
                    'tf': tf,
                    'c1': c1,
                    'c2': c2,
                    'regime_2d': regime,
                    'peak_shift': peak_s,
                    'peak_corr': peak_c,
                    'contemp_corr': contemp_c,
                    'role': role,
                })
        print(f"  {tf}: done ({len([r for r in rows if r['tf']==tf])} cells)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'leadlag_corr.csv'), index=False)
    print(f"  [saved] leadlag_corr.csv ({len(df)} rows)")

    peak_df = pd.DataFrame(peak_rows)
    peak_df['abs_peak'] = peak_df['peak_corr'].abs()
    peak_df = peak_df.sort_values('abs_peak', ascending=False)
    peak_df.to_csv(os.path.join(args.output_dir, 'peak_lag.csv'), index=False)
    print(f"  [saved] peak_lag.csv ({len(peak_df)} rows)")

    # Role distribution
    role_counts = peak_df['role'].value_counts()
    print(f"\n  Role distribution across {len(peak_df)} (X, Y, TF, regime):")
    for r, c in role_counts.items():
        pct = 100.0 * c / len(peak_df)
        print(f"    {r:>20}: {c:>5}  ({pct:>5.1f}%)")

    # Top 30 by |peak|
    print(f"\n  Top 30 by |peak corr|:")
    print(f"    {'tf':>4}  {'c1':>22}  {'c2':>22}  {'regime':>14}  "
          f"{'shift':>6}  {'peak':>7}  {'contemp':>8}  {'role':>16}")
    for _, r in peak_df.head(30).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  {r['peak_shift']:>+6}  "
              f"{r['peak_corr']:>+7.3f}  {r['contemp_corr']:>+8.3f}  "
              f"{r['role']:>16}")

    # Asymmetry — pairs where peak shift != 0 (genuine lead-lag)
    asym = peak_df[peak_df['peak_shift'] != 0].copy()
    print(f"\n  Genuinely asymmetric pairs (peak_shift != 0): {len(asym)} of {len(peak_df)} "
          f"({100.0*len(asym)/max(len(peak_df),1):.1f}%)")

    # Strongest leaders (peak_s > 0, |peak| > contemp by margin)
    leaders = peak_df[(peak_df['peak_shift'] > 0)
                          & (peak_df['abs_peak'] > peak_df['contemp_corr'].abs() * 1.1)].copy()
    print(f"\n  Genuine leaders (X leads Y, peak |corr| > contemp by >10%): {len(leaders)}")
    print(f"    {'tf':>4}  {'c1 (leads)':>22}  {'c2 (follows)':>22}  "
          f"{'regime':>14}  {'shift':>6}  {'peak':>7}")
    for _, r in leaders.sort_values('abs_peak', ascending=False).head(20).iterrows():
        print(f"    {r['tf']:>4}  {r['c1']:>22}  {r['c2']:>22}  "
              f"{r['regime_2d']:>14}  {r['peak_shift']:>+6}  "
              f"{r['peak_corr']:>+7.3f}")

    # Plot top pairs (one per (c1, c2, tf), all 6 regime traces)
    plotted = 0
    df_pair_strength = (peak_df.groupby(['c1', 'c2', 'tf'])['abs_peak']
                          .max().sort_values(ascending=False)
                          .head(args.top_plots))
    for (c1, c2, tf), _ in df_pair_strength.items():
        sub = df[(df['c1'] == c1) & (df['c2'] == c2) & (df['tf'] == tf)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for regime in REGIME_2D_ORDER:
            ssub = sub[sub['regime_2d'] == regime]
            if ssub.empty:
                continue
            ax.plot(ssub['shift_s'], ssub['corr'], marker='o',
                     label=regime, alpha=0.85)
        ax.axhline(0, color='black', alpha=0.4)
        ax.axvline(0, color='black', alpha=0.4, linestyle=':')
        ax.set_xlabel('shift s (base bars)  s>0: X leads Y;  s<0: X lags Y')
        ax.set_ylabel('Pearson corr(X_t, Y_{t+s})')
        ax.set_title(f'{c1} vs {c2} @ {tf} — lead-lag by regime')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(args.output_dir, f'plot_{tf}_{c1}__{c2}.png')
        fig.savefig(png, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plotted += 1
    print(f"\n  [saved] {plotted} lead-lag plots")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-TF lead-lag (Layer D5) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Pairs**: {n_pairs}  **TFs**: {len(args.tfs)}  "
                f"**Regimes**: 6  **Shifts**: {args.shifts}\n\n")
        f.write(f"## Role distribution\n\n")
        for r, c in role_counts.items():
            pct = 100.0 * c / len(peak_df)
            f.write(f"- {r}: {c} ({pct:.1f}%)\n")
        f.write(f"\n## Top 50 by |peak corr|\n\n")
        f.write(peak_df.head(50).to_string(index=False))
        f.write(f"\n\n## Genuine leaders (peak_shift > 0)\n\n")
        f.write(leaders.head(50).to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
