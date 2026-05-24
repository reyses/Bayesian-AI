"""
v2_features_tod_eda.py — Time-of-day × regime × feature drill-down.

Does each feature's regime-conditional signal hold throughout the
trading session, or concentrate in specific hours?

For each (feature, TF, regime, hour_bucket), compute mean forward return
and win_rate. Highlights:
  - Hours where signal is strongest per regime
  - Hours where signal INVERTS vs the regime average

Default hour buckets (in UTC, mapped from America/Los_Angeles to keep
sessions stable):
  pre_market   18:00-23:00 PT (00:00-06:00 UTC) — Asian session
  asia_close   23:00-01:00 PT (06:00-09:00 UTC)
  eu_open      01:00-05:00 PT (09:00-12:00 UTC)
  us_pre       05:00-06:30 PT (12:00-13:30 UTC)
  us_open      06:30-08:00 PT (13:30-15:00 UTC) — most volatile
  us_morning   08:00-10:00 PT (15:00-17:00 UTC)
  us_lunch     10:00-11:30 PT (17:00-18:30 UTC) — drift / chop
  us_pm        11:30-13:00 PT (18:30-20:00 UTC) — close
  after_close  13:00-18:00 PT (20:00-01:00 UTC) — overnight thin

Outputs:
  reports/findings/v2_features_tod/
    tod_summary.csv        (feat, tf, regime, hour_bucket, n, mean_fwd, wr)
    tod_concentration.csv  per (feat, tf, regime): which hour carries max signal
    summary.md
    heatmap_<concept_tf>.png  rows=hour_bucket, cols=regime, color=mean_fwd
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


# Hour buckets (PT) keyed by (start_pt_hour, end_pt_hour); end exclusive.
# Use floats since some boundaries are :30
HOUR_BUCKETS = [
    ('pre_market',   18.0, 23.0),
    ('asia_close',   23.0, 1.0),    # wraps midnight
    ('eu_open',      1.0,  5.0),
    ('us_pre',       5.0,  6.5),
    ('us_open',      6.5,  8.0),
    ('us_morning',   8.0,  10.0),
    ('us_lunch',     10.0, 11.5),
    ('us_pm',        11.5, 13.0),
    ('after_close',  13.0, 18.0),
]
BUCKET_ORDER = [b[0] for b in HOUR_BUCKETS]


def assign_bucket(pt_hour: float) -> str:
    for name, lo, hi in HOUR_BUCKETS:
        if lo <= hi:
            if lo <= pt_hour < hi:
                return name
        else:  # wraps midnight
            if pt_hour >= lo or pt_hour < hi:
                return name
    return 'unknown'


# Default focus concepts (from earlier layers — top signal carriers)
DEFAULT_CONCEPTS = [
    'price_sigma_w', 'bar_range', 'vol_mean_w', 'vol_velocity_w',
    'price_velocity_w', 'body', 'price_velocity_1b', 'swing_noise_w',
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
    parser.add_argument('--split', default='IS')
    parser.add_argument('--quantile', type=int, default=4,
                        help='Use top quantile (Q4 of 5) bars for the signal '
                             'tracking — focused on extreme-tail behavior')
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--min-cell-n', type=int, default=30)
    parser.add_argument('--top-pairs-to-plot', type=int, default=12)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_tod')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features × time-of-day × regime")
    print(f"  Concepts: {args.concepts}")
    print(f"  TFs: {args.tfs}")
    print(f"  Tracking quantile Q{args.quantile} of {args.quantiles}")
    print(f"  Hour buckets (PT): {BUCKET_ORDER}")
    print(f"{'='*70}")

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
    base_df['pt_hour'] = dt_la.dt.hour + dt_la.dt.minute / 60.0
    base_df['hour_bucket'] = base_df['pt_hour'].apply(assign_bucket)

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    # Bucket distribution
    print(f"\n  Bars per bucket:")
    bk_counts = merged['hour_bucket'].value_counts()
    for b in BUCKET_ORDER:
        print(f"    {b:>14}: {int(bk_counts.get(b, 0)):>7}")

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
    buckets = full['hour_bucket'].values.astype(str)

    # Sweep
    print(f"\n--- Sweeping {len(args.concepts)} concepts × {len(args.tfs)} TFs × "
          f"{len(REGIME_2D_ORDER)} regimes × {len(BUCKET_ORDER)} buckets ---")

    rows = []
    concentration_rows = []
    cell_data = {}  # (concept, tf, regime, bucket) -> mean_fwd

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
                # Compute regime-local quantile of v
                v_r = v[regime_mask]
                valid = ~np.isnan(v_r)
                if valid.sum() < args.quantiles * 5:
                    continue
                qs = np.quantile(v_r[valid], np.linspace(0, 1, args.quantiles + 1))
                qs[0] -= 1e-9
                qs[-1] += 1e-9
                # For ALL bars (not just regime), compute their quantile relative
                # to the regime distribution
                bin_idx_full = np.digitize(v, qs[1:-1])
                # Mark only bars in this regime AND in target quantile
                target_mask = regime_mask & (bin_idx_full == args.quantile) & ~np.isnan(v) & ~np.isnan(fwd)

                if target_mask.sum() < args.min_cell_n:
                    continue

                # Per-bucket stats within the (regime, target_quantile)
                regime_quantile_n = int(target_mask.sum())
                bucket_means = {}
                for bk in BUCKET_ORDER:
                    bk_mask = target_mask & (buckets == bk)
                    n_cell = int(bk_mask.sum())
                    if n_cell < args.min_cell_n:
                        bucket_means[bk] = float('nan')
                        continue
                    f = fwd[bk_mask]
                    mean_fwd = float(f.mean())
                    wr = float((f > 0).mean())
                    rows.append({
                        'concept': concept,
                        'tf': tf,
                        'regime_2d': regime,
                        'hour_bucket': bk,
                        'n': n_cell,
                        'mean_fwd': mean_fwd,
                        'win_rate': wr,
                    })
                    bucket_means[bk] = mean_fwd

                # Concentration: which bucket has max |mean_fwd|?
                valid_means = {bk: m for bk, m in bucket_means.items()
                                if not np.isnan(m)}
                if not valid_means:
                    continue
                max_bk = max(valid_means.items(), key=lambda kv: abs(kv[1]))
                min_bk = min(valid_means.items(), key=lambda kv: abs(kv[1]))
                concentration_rows.append({
                    'concept': concept,
                    'tf': tf,
                    'regime_2d': regime,
                    'n_target': regime_quantile_n,
                    'n_buckets_with_data': len(valid_means),
                    'best_bucket': max_bk[0],
                    'best_bucket_fwd': max_bk[1],
                    'worst_bucket': min_bk[0],
                    'worst_bucket_fwd': min_bk[1],
                    'inversion': bool(np.sign(max_bk[1]) != np.sign(min_bk[1])
                                       and abs(max_bk[1]) > 1.0
                                       and abs(min_bk[1]) > 1.0),
                })
                cell_data[(concept, tf, regime)] = bucket_means

    df = pd.DataFrame(rows)
    df_path = os.path.join(args.output_dir, 'tod_summary.csv')
    df.to_csv(df_path, index=False)
    print(f"  [saved] {df_path} ({len(df)} cells)")

    conc_df = pd.DataFrame(concentration_rows).sort_values(
        'best_bucket_fwd', key=lambda s: s.abs(), ascending=False)
    conc_path = os.path.join(args.output_dir, 'tod_concentration.csv')
    conc_df.to_csv(conc_path, index=False)
    print(f"  [saved] {conc_path}")

    # Print: top 20 (concept, tf, regime) by |best_bucket_fwd|
    print(f"\n  Top 20 (concept, TF, regime) by best-bucket forward return:")
    print(f"    {'concept':>22}  {'tf':>4}  {'regime':>14}  {'best':>14} "
          f"{'fwd':>8} {'worst':>14} {'fwd':>8}  {'inv':>4}")
    for _, r in conc_df.head(20).iterrows():
        print(f"    {r['concept']:>22}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['best_bucket']:>14} {r['best_bucket_fwd']:>+8.2f} "
              f"{r['worst_bucket']:>14} {r['worst_bucket_fwd']:>+8.2f}  "
              f"{'YES' if r['inversion'] else 'no':>4}")

    print(f"\n  TOD INVERSIONS: same (concept, TF, regime, Q4) gives opposite "
          f"signs in different hour buckets")
    inv_df = conc_df[conc_df['inversion']]
    print(f"  Total: {len(inv_df)}")
    for _, r in inv_df.head(15).iterrows():
        print(f"    {r['concept']:>22}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['best_bucket']:>14} {r['best_bucket_fwd']:>+8.2f}  "
              f"{r['worst_bucket']:>14} {r['worst_bucket_fwd']:>+8.2f}")

    # Per concept_tf, plot heatmap of mean_fwd per (regime × hour_bucket)
    plotted = 0
    df['concept_tf'] = df['concept'] + '_' + df['tf']
    top_concept_tfs = (df.groupby('concept_tf')['mean_fwd']
                          .apply(lambda x: x.abs().max())
                          .sort_values(ascending=False)
                          .head(args.top_pairs_to_plot))
    for cn_tf in top_concept_tfs.index:
        sub = df[df['concept_tf'] == cn_tf]
        pivot_h = sub.pivot(index='hour_bucket', columns='regime_2d',
                              values='mean_fwd')
        pivot_h = pivot_h.reindex(
            index=[b for b in BUCKET_ORDER if b in pivot_h.index],
            columns=[r for r in REGIME_2D_ORDER if r in pivot_h.columns],
        )
        if pivot_h.size == 0 or np.all(np.isnan(pivot_h.values)):
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        vmax = float(np.nanmax(np.abs(pivot_h.values)))
        im = ax.imshow(pivot_h.values, cmap='RdBu_r', aspect='auto',
                        vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot_h.columns)))
        ax.set_xticklabels(pivot_h.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_h.index)))
        ax.set_yticklabels(pivot_h.index)
        for i in range(len(pivot_h.index)):
            for j in range(len(pivot_h.columns)):
                v = pivot_h.iloc[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=8,
                          color='white' if abs(v) > vmax / 2 else 'black')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Hour bucket (PT)')
        ax.set_title(f'{cn_tf} — Q{args.quantile} mean_fwd by (regime, hour)\n'
                      f'vmax={vmax:.2f}')
        plt.colorbar(im, ax=ax, label='mean_fwd (ticks)')
        fig.tight_layout()
        png_path = os.path.join(args.output_dir, f'heatmap_{cn_tf}.png')
        fig.savefig(png_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plotted += 1
    print(f"\n  [saved] {plotted} heatmaps")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features × time-of-day × regime — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {args.concepts}\n\n")
        f.write(f"**TFs:** {args.tfs}\n\n")
        f.write(f"**Tracking quantile:** Q{args.quantile} of {args.quantiles}\n\n")
        f.write(f"**Buckets (PT):** {BUCKET_ORDER}\n\n")
        f.write("## Top 30 (concept, TF, regime) by best-bucket forward return\n\n")
        f.write(conc_df.head(30).to_string(index=False))
        f.write("\n\n## TOD inversions (same Q4-cell, opposite signs in different hours)\n\n")
        f.write(inv_df.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
