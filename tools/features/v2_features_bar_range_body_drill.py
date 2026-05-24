"""
v2_features_bar_range_body_drill.py — Drill: bar_range x body as a
JOINT-extreme regime detector.

The pair shows up in many top D8 triplets and survives D7+D9 (95% OOS).
Physical reading: corr(bar_range, body) measures DIRECTIONAL ASYMMETRY
OF BAR SIZE — when big bars happen, are they UP-biased or DOWN-biased?

Drill 2 showed rolling-corr classifiers don't work as standalones.
This drill takes a different angle: use the JOINT EXTREME state as a
high-conviction trigger.

For each TF:
  - Bin bar_range into 5 quantiles (focus on Q4)
  - Sign of body: positive (+) or negative (-)
  - Cell definitions:
      BIG+POS:  bar_range Q4 AND body > 0
      BIG+NEG:  bar_range Q4 AND body < 0
      ELSE:     other states
  - For each cell, compute regime distribution AND day-by-day frequency
  - Test as classifier: BIG+POS -> UP, BIG+NEG -> DOWN, else -> abstain

Outputs:
  reports/findings/v2_drill_bar_range_body/
    cell_regime_dist.csv         per (cell, TF) regime distribution
    daily_signal_count.csv       per day: count of BIG+POS / BIG+NEG bars
    classification_perf.csv      precision per TF
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


DEFAULT_TFS = ['5s', '1m', '5m', '15m', '1h']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--split', default='ALL')
    parser.add_argument('--bar-range-extreme-q', type=int, default=4)
    parser.add_argument('--day-min-signal-count', type=int, default=3,
                        help='Min count of BIG+POS or BIG+NEG bars in a day '
                             'to issue a daily prediction')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_drill_bar_range_body')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  Drill: bar_range x body joint-extreme regime detector")
    print(f"  TFs: {args.tfs}  Q: {args.quantiles}")
    print(f"  Extreme quantile for bar_range: Q{args.bar_range_extreme_q}")
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
    dates_all = full['date'].values.astype(str)

    cell_rows = []
    daily_rows = []
    perf_rows = []

    for tf in args.tfs:
        col_br = feature_column_for('bar_range', tf)
        col_body = feature_column_for('body', tf)
        if col_br not in full.columns or col_body not in full.columns:
            continue
        br = full[col_br].values.astype(np.float64)
        body = full[col_body].values.astype(np.float64)

        valid = ~np.isnan(br) & ~np.isnan(body)
        if valid.sum() < 200:
            continue

        # bin bar_range into Q quantiles
        qs = np.quantile(br[valid], np.linspace(0, 1, args.quantiles + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        br_bin = np.digitize(br, qs[1:-1])

        # cell labels
        cells = np.full(len(full), 'OTHER', dtype=object)
        big = (br_bin == args.bar_range_extreme_q) & valid
        cells[big & (body > 0)] = 'BIG+POS'
        cells[big & (body < 0)] = 'BIG+NEG'
        full[f'cell_{tf}'] = cells

        # cell -> regime distribution
        for cell_label in ('BIG+POS', 'BIG+NEG', 'OTHER'):
            mask = (cells == cell_label) & valid
            if mask.sum() == 0:
                continue
            rs = regimes[mask]
            for regime in REGIME_2D_ORDER:
                cell_rows.append({
                    'tf': tf,
                    'cell': cell_label,
                    'regime_2d': regime,
                    'n': int((rs == regime).sum()),
                    'pct': 100.0 * (rs == regime).sum() / max(mask.sum(), 1),
                })
            print(f"  TF={tf}, cell={cell_label}: n={int(mask.sum())}, "
                  f"regime distribution: ", end='')
            for regime in REGIME_2D_ORDER:
                pct = 100.0 * (rs == regime).sum() / max(mask.sum(), 1)
                print(f"{regime}={pct:.1f}%, ", end='')
            print()

        # daily counts of BIG+POS and BIG+NEG, and daily prediction
        df_t = pd.DataFrame({
            'date': dates_all,
            'cell': cells,
            'regime_2d': regimes,
        })
        per_day = (df_t.groupby('date')
                          .agg(n_bigpos=('cell',
                                              lambda c: int((c == 'BIG+POS').sum())),
                                n_bigneg=('cell',
                                              lambda c: int((c == 'BIG+NEG').sum())),
                                n_total=('cell', 'count'),
                                regime_2d=('regime_2d', 'first'))
                          .reset_index())
        per_day['tf'] = tf
        per_day['actual'] = per_day['regime_2d'].apply(
            lambda r: 'UP' if r.startswith('UP') else
                       ('DOWN' if r.startswith('DOWN') else 'FLAT'))
        # daily prediction: more BIG+POS than BIG+NEG by margin >= day_min -> UP, opposite -> DOWN
        def predict(row):
            margin = row['n_bigpos'] - row['n_bigneg']
            if abs(margin) < args.day_min_signal_count:
                return 'ABSTAIN'
            return 'UP' if margin > 0 else 'DOWN'
        per_day['pred'] = per_day.apply(predict, axis=1)
        daily_rows.append(per_day)

        # Performance: precision when prediction is UP or DOWN
        non_abstain = per_day[per_day['pred'] != 'ABSTAIN']
        n_up_pred = int((non_abstain['pred'] == 'UP').sum())
        n_down_pred = int((non_abstain['pred'] == 'DOWN').sum())
        n_up_correct = int(((non_abstain['pred'] == 'UP') &
                                  (non_abstain['actual'] == 'UP')).sum())
        n_down_correct = int(((non_abstain['pred'] == 'DOWN') &
                                    (non_abstain['actual'] == 'DOWN')).sum())
        up_precision = n_up_correct / max(n_up_pred, 1)
        down_precision = n_down_correct / max(n_down_pred, 1)
        coverage = len(non_abstain) / max(len(per_day), 1)
        # Base rates
        up_base = (per_day['actual'] == 'UP').mean()
        down_base = (per_day['actual'] == 'DOWN').mean()
        perf_rows.append({
            'tf': tf,
            'n_days': len(per_day),
            'n_up_pred': n_up_pred,
            'up_precision': up_precision,
            'up_lift_vs_baserate': up_precision - up_base,
            'n_down_pred': n_down_pred,
            'down_precision': down_precision,
            'down_lift_vs_baserate': down_precision - down_base,
            'coverage': coverage,
            'abstain_pct': 1.0 - coverage,
        })

    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(os.path.join(args.output_dir, 'cell_regime_dist.csv'),
                      index=False)
    print(f"  [saved] cell_regime_dist.csv")

    if daily_rows:
        daily_df = pd.concat(daily_rows, ignore_index=True)
        daily_df.to_csv(os.path.join(args.output_dir, 'daily_signal_count.csv'),
                          index=False)
        print(f"  [saved] daily_signal_count.csv ({len(daily_df)} day-tf rows)")

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(args.output_dir, 'classification_perf.csv'),
                      index=False)
    print(f"  [saved] classification_perf.csv\n")

    print(f"  Daily classifier performance per TF:")
    print(f"    {'tf':>4}  {'n_days':>6}  {'up_pred':>7}  {'up_prec':>7}  {'up_lift':>7}  "
          f"{'down_pred':>9}  {'down_prec':>9}  {'down_lift':>9}  {'coverage':>8}")
    for _, r in perf_df.iterrows():
        print(f"    {r['tf']:>4}  {int(r['n_days']):>6}  "
              f"{int(r['n_up_pred']):>7}  {r['up_precision']:>7.3f}  "
              f"{r['up_lift_vs_baserate']:>+7.3f}  "
              f"{int(r['n_down_pred']):>9}  {r['down_precision']:>9.3f}  "
              f"{r['down_lift_vs_baserate']:>+9.3f}  {r['coverage']:>8.3f}")

    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Drill: bar_range x body joint-extreme regime detector - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Trigger**: bar_range Q{args.bar_range_extreme_q} + sign(body)\n\n")
        f.write(f"**Daily prediction rule**: |n_bigpos - n_bigneg| >= "
                f"{args.day_min_signal_count} margin = UP or DOWN; else ABSTAIN\n\n")
        f.write(f"## Cell -> regime distribution\n\n")
        for tf in args.tfs:
            sub = cell_df[cell_df['tf'] == tf]
            if sub.empty:
                continue
            pv = sub.pivot(index='cell', columns='regime_2d', values='pct').fillna(0)
            pv = pv.reindex(columns=[r for r in REGIME_2D_ORDER if r in pv.columns])
            f.write(f"### TF={tf}\n\n")
            f.write(pv.round(1).to_string())
            f.write("\n\n")
        f.write(f"## Daily classifier performance\n\n")
        f.write(perf_df.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
