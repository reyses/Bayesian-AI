"""
v2_features_rolling_corr_regime_drill.py — Drill Surprise #2: can the
rolling sign of regime-conditional feature pairs serve as a real-time
regime classifier?

D2/D7 found that ~83 pairs flip correlation sign by regime, with the
regime difference confirmed in both IS halves and on OOS (D9, 95%
survival). The classic example:
  price_velocity_w x price_sigma_w @ 15m:
    UP_SMOOTH r ~ +0.85 ; DOWN_SMOOTH r ~ -0.78
  bar_range x body @ 1h:
    UP_SMOOTH r ~ +0.71 ; DOWN_SMOOTH r ~ -0.59

If we compute corr(X, Y) over a ROLLING window (e.g. last 200 bars),
the sign should track the prevailing regime. This drill tests that
hypothesis.

For each candidate pair:
  1. Compute rolling-window Pearson corr at every bar
  2. Build daily aggregate (mean of rolling corr per day)
  3. Compare daily-aggregate sign to that day's regime_2d label
  4. Compute classification accuracy

If accuracy > 70%, the pair is a usable real-time regime detector.

Outputs:
  reports/findings/v2_drill_rolling_corr_regime/
    rolling_corr_per_day.csv     (date, pair, tf, mean_rolling_corr, regime_2d)
    classification_accuracy.csv  per (pair, tf): UP_acc, DOWN_acc
    confusion_matrices.csv
    summary.md
    plot_<pair>_<tf>.png         time series of rolling corr vs regime
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


# Top candidates from D7 confirmed regime-flips
CANDIDATE_PAIRS = [
    ('price_velocity_w', 'price_sigma_w', '15m'),
    ('price_velocity_w', 'price_sigma_w', '1h'),
    ('price_velocity_w', 'price_sigma_w', '5m'),
    ('bar_range', 'body', '1h'),
    ('bar_range', 'body', '15m'),
    ('price_velocity_1b', 'bar_range', '1h'),
    ('price_velocity_1b', 'bar_range', '15m'),
    ('price_velocity_w', 'swing_noise_w', '1h'),
    ('price_velocity_w', 'SE_low_w', '15m'),
    ('price_velocity_w', 'SE_high_w', '15m'),
    ('price_velocity_w', 'vol_mean_w', '1h'),
    ('price_velocity_1b', 'reversion_prob_w', '1h'),
]


def rolling_corr(x, y, window):
    """Pandas rolling pearson corr, NaN-tolerant."""
    s_x = pd.Series(x)
    s_y = pd.Series(y)
    return s_x.rolling(window, min_periods=max(30, window // 2)).corr(s_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--window', type=int, default=200,
                        help='Rolling window in 5m bars (200 ~ 16 hours)')
    parser.add_argument('--split', default='ALL')
    parser.add_argument('--threshold', type=float, default=0.20,
                        help='|corr| above which we classify regime')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_drill_rolling_corr_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  Drill: rolling corr as regime classifier")
    print(f"  Window: {args.window} 5m bars (~{args.window*5/60:.1f}h)")
    print(f"  Threshold: |corr| > {args.threshold}")
    print(f"  Pairs: {len(CANDIDATE_PAIRS)}")
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

    daily_rows = []
    for X, Y, tf in CANDIDATE_PAIRS:
        cx = feature_column_for(X, tf)
        cy = feature_column_for(Y, tf)
        if cx not in full.columns or cy not in full.columns:
            continue
        x = full[cx].values
        y = full[cy].values
        rc = rolling_corr(x, y, args.window).values
        # Aggregate per day: mean rolling corr (skipping nan)
        df_t = pd.DataFrame({
            'date': full['date'],
            'rolling_corr': rc,
            'regime_2d': full['regime_2d'],
        })
        per_day = (df_t.groupby('date')
                          .agg(mean_rolling_corr=('rolling_corr', 'mean'),
                                regime_2d=('regime_2d', 'first'))
                          .reset_index()
                          .dropna())
        per_day['pair'] = f'{X}__{Y}'
        per_day['tf'] = tf
        daily_rows.append(per_day)

    if not daily_rows:
        print("No data — bail")
        return
    big = pd.concat(daily_rows, ignore_index=True)
    big.to_csv(os.path.join(args.output_dir, 'rolling_corr_per_day.csv'),
                  index=False)
    print(f"  [saved] rolling_corr_per_day.csv ({len(big)} rows)")

    # Classification: predict regime from rolling corr sign
    # For each pair: if rolling > +threshold -> "UP-like"; if < -threshold -> "DOWN-like"; else "FLAT-like"
    # Compare to actual regime_2d
    print(f"\n--- Classification per pair ---")
    cls_rows = []
    for pair_tf in big[['pair', 'tf']].drop_duplicates().values:
        pair_str, tf_str = pair_tf[0], pair_tf[1]
        sub = big[(big['pair'] == pair_str) & (big['tf'] == tf_str)].copy()
        sub['pred'] = 'FLAT'
        sub.loc[sub['mean_rolling_corr'] > args.threshold, 'pred'] = 'UP'
        sub.loc[sub['mean_rolling_corr'] < -args.threshold, 'pred'] = 'DOWN'
        # actual aggregated to UP/DOWN/FLAT
        sub['actual'] = sub['regime_2d'].apply(
            lambda r: 'UP' if r.startswith('UP') else
                       ('DOWN' if r.startswith('DOWN') else 'FLAT'))
        # accuracy
        n_total = len(sub)
        if n_total == 0:
            continue
        n_correct = int((sub['pred'] == sub['actual']).sum())
        acc = n_correct / n_total
        # per-actual-class accuracy
        for actual in ('UP', 'DOWN', 'FLAT'):
            sub_act = sub[sub['actual'] == actual]
            if len(sub_act) > 0:
                act_correct = int((sub_act['pred'] == sub_act['actual']).sum())
                act_acc = act_correct / len(sub_act)
            else:
                act_acc = float('nan')
        # Confusion matrix
        cm = pd.crosstab(sub['actual'], sub['pred'], dropna=False)
        cm = cm.reindex(index=['UP', 'DOWN', 'FLAT'],
                          columns=['UP', 'DOWN', 'FLAT'], fill_value=0)
        cls_rows.append({
            'pair': pair_str,
            'tf': tf_str,
            'n_days': n_total,
            'overall_acc': acc,
            'up_acc':   int(cm.loc['UP', 'UP']) / max(int(cm.loc['UP'].sum()), 1),
            'down_acc': int(cm.loc['DOWN', 'DOWN']) / max(int(cm.loc['DOWN'].sum()), 1),
            'flat_acc': int(cm.loc['FLAT', 'FLAT']) / max(int(cm.loc['FLAT'].sum()), 1),
            'cm_up_to_up':   int(cm.loc['UP', 'UP']),
            'cm_up_to_down': int(cm.loc['UP', 'DOWN']),
            'cm_up_to_flat': int(cm.loc['UP', 'FLAT']),
            'cm_down_to_up':   int(cm.loc['DOWN', 'UP']),
            'cm_down_to_down': int(cm.loc['DOWN', 'DOWN']),
            'cm_down_to_flat': int(cm.loc['DOWN', 'FLAT']),
            'cm_flat_to_up':   int(cm.loc['FLAT', 'UP']),
            'cm_flat_to_down': int(cm.loc['FLAT', 'DOWN']),
            'cm_flat_to_flat': int(cm.loc['FLAT', 'FLAT']),
        })

    cls_df = pd.DataFrame(cls_rows).sort_values('overall_acc', ascending=False)
    cls_df.to_csv(os.path.join(args.output_dir, 'classification_accuracy.csv'),
                     index=False)
    print(f"  [saved] classification_accuracy.csv\n")

    print(f"  Pair-by-pair regime classification accuracy:")
    print(f"    {'pair':>40}  {'tf':>4}  {'n_days':>6}  "
          f"{'overall':>7}  {'UP_acc':>6}  {'DOWN_acc':>8}  {'FLAT_acc':>8}")
    for _, r in cls_df.iterrows():
        print(f"    {r['pair']:>40}  {r['tf']:>4}  {int(r['n_days']):>6}  "
              f"{r['overall_acc']:>7.3f}  {r['up_acc']:>6.3f}  "
              f"{r['down_acc']:>8.3f}  {r['flat_acc']:>8.3f}")

    # Baseline: regime distribution overall
    actual_dist = big['regime_2d'].apply(
        lambda r: 'UP' if r.startswith('UP') else
                   ('DOWN' if r.startswith('DOWN') else 'FLAT')).value_counts(normalize=True)
    print(f"\n  Baseline (regime distribution): "
          f"{dict(actual_dist.round(3))}")

    # Build plot per pair: time series of mean_rolling_corr colored by regime
    plotted = 0
    for pair_tf in big[['pair', 'tf']].drop_duplicates().head(8).values:
        pair_str, tf_str = pair_tf[0], pair_tf[1]
        sub = big[(big['pair'] == pair_str) & (big['tf'] == tf_str)].copy()
        sub = sub.sort_values('date').reset_index(drop=True)
        sub['date'] = pd.to_datetime(sub['date'])

        fig, ax = plt.subplots(figsize=(13, 5))
        # color by regime
        regime_colors = {
            'UP_SMOOTH': '#0066ff',
            'UP_CHOPPY': '#66aaff',
            'DOWN_SMOOTH': '#ff3333',
            'DOWN_CHOPPY': '#ff9999',
            'FLAT_SMOOTH': '#999999',
            'FLAT_CHOPPY': '#cccccc',
        }
        for regime, color in regime_colors.items():
            srows = sub[sub['regime_2d'] == regime]
            if srows.empty:
                continue
            ax.scatter(srows['date'], srows['mean_rolling_corr'], c=color,
                        s=12, alpha=0.85, label=regime)
        ax.axhline(args.threshold, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(-args.threshold, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.4)
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Rolling-{args.window}bar corr')
        ax.set_title(f'{pair_str} @ {tf_str} — daily mean of rolling corr, colored by regime')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        png = os.path.join(args.output_dir, f'plot_{pair_str}_{tf_str}.png')
        fig.savefig(png, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plotted += 1
    print(f"\n  [saved] {plotted} time-series plots")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Drill: rolling corr as regime classifier - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Window**: {args.window} 5m bars (~{args.window*5/60:.1f}h)\n\n")
        f.write(f"**Threshold**: |corr| > {args.threshold} for UP/DOWN, else FLAT\n\n")
        f.write(f"**Baseline regime distribution**: "
                f"{dict(actual_dist.round(3))}\n\n")
        f.write("## Pair classification accuracy\n\n")
        f.write(cls_df.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
