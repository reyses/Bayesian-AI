"""
v2_regress_trend_direction.py — Predict the day's signed net move
(trend direction × strength) from 5m bar features.

Where Analysis L predicted signed MFE at a regime-determined direction
(magnitude × regime direction), this tool predicts the DAY'S NET MOVE
directly. The target's sign IS the trend direction (UP / DOWN / FLAT
per `direction_axis` from regime_labels_2d).

Why this framing:
  - Analysis L's target was "magnitude * regime direction" — circular
    if regime direction is what we want to predict.
  - net_move is the cleanest "trend direction" target: its sign IS
    direction (positive = UP day, negative = DOWN day, near-zero = FLAT).
    Magnitude correlates with SMOOTH/CHOPPY (large = trended cleanly).
  - One target per day, repeated across the day's bars. Training uses
    DAY-level splits to avoid intraday leakage.

Pipeline:
  1. Load 5m OHLC + v2 features (185D), reindex onto 5m bars.
  2. Load DATA/ATLAS/regime_labels_2d.csv. For each 5m bar, attach the
     day's net_move and regime_2d label.
  3. Day-level 60/20/20 train/val/test split (per regime_labels_2d.split
     by default — preserves the same split as other v2 work).
  4. Fit OLS on bar features → net_move target.
  5. Score:
     - Bar-level: sign(pred) vs sign(actual day net_move) at each bar
     - Day-level: aggregate bar predictions per day (mean), sign vs sign
     - Per-regime breakdown: accuracy in each 2D regime cell

Outputs:
  reports/findings/v2_regress_trend_direction/
    summary.md             — narrative
    per_regime.csv         — accuracy stratified by regime_2d on test split
    per_bar_predictions.csv — timestamp, day_date, regime_2d, pred, actual_net_move

Usage:
    python tools/v2_regress_trend_direction.py
    python tools/v2_regress_trend_direction.py --base-tf 1m  # higher density
    python tools/v2_regress_trend_direction.py --binarize    # Y = sign instead of magnitude
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels


DEFAULT_BASE_TF = '5m'


def attach_day_targets(base_df: pd.DataFrame, base_tf: str,
                        labels_df: pd.DataFrame,
                        binarize: bool = False) -> pd.DataFrame:
    """For each base-TF bar, attach the day's net_move (target) + regime + split.

    `base_df` must have a 'timestamp' column in unix seconds. Days are
    derived in the same TZ used by atlas_regime_labeler (America/Los_Angeles
    per its source) — but regime_labels uses date column; we'll match by
    converting bar timestamps to the same date string.
    """
    base = base_df.copy()
    if pd.api.types.is_datetime64_any_dtype(base['timestamp']):
        ts_int = base['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base['timestamp'].astype(np.int64)
    base['ts_int'] = ts_int

    # Convert to America/Los_Angeles date to match labeler
    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base['date'] = dt_la.dt.date.astype(str)

    labels_df = labels_df.copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    keep = ['date', 'net_move', 'directional_strength', 'efficiency_ratio',
            'range_expansion', 'direction_axis', 'variation_axis',
            'regime_2d', 'split']
    keep = [c for c in keep if c in labels_df.columns]

    merged = base.merge(labels_df[keep], on='date', how='inner')
    if binarize:
        merged['target'] = np.where(merged['net_move'] > 0, 1.0,
                                      np.where(merged['net_move'] < 0, -1.0, 0.0))
    else:
        merged['target'] = merged['net_move']
    return merged


def fit_and_score(merged: pd.DataFrame, feature_cols: list[str],
                   ridge_alpha: float = 0.0,
                   verbose: bool = True) -> dict:
    """Day-level split, fit OLS/Ridge, score bar-level and day-level."""
    sub = merged.dropna(subset=feature_cols + ['target', 'split']).reset_index(drop=True)
    if verbose:
        print(f"  Total bars: {len(sub):,} ({sub['date'].nunique()} unique days)")
        print(f"  Split distribution:\n{sub['split'].value_counts().to_string()}")

    train = sub[sub['split'] == 'IS']
    val = sub[sub['split'] == 'VAL']
    test = sub[sub['split'] == 'OOS']

    if verbose:
        print(f"  Train: {len(train):,} bars / {train['date'].nunique()} days")
        print(f"  Test:  {len(test):,} bars / {test['date'].nunique()} days")

    X_tr = train[feature_cols].values.astype(np.float64)
    X_te = test[feature_cols].values.astype(np.float64)
    y_tr = train['target'].values.astype(np.float64)
    y_te = test['target'].values.astype(np.float64)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    if ridge_alpha > 0:
        model = Ridge(alpha=ridge_alpha)
    else:
        model = LinearRegression()
    model.fit(X_tr_sc, y_tr)
    pred_te = model.predict(X_te_sc)

    r2_te = float(model.score(X_te_sc, y_te))

    # Bar-level direction accuracy
    actual_dir = np.sign(y_te)
    pred_dir = np.sign(pred_te)
    valid = actual_dir != 0
    bar_acc = float((actual_dir[valid] == pred_dir[valid]).mean()) if valid.any() else float('nan')
    bar_baseline = float(max((actual_dir[valid] == 1).mean(),
                               (actual_dir[valid] == -1).mean())) if valid.any() else 0.5

    # Day-level: aggregate bar predictions to day, vote sign
    test_with_pred = test.assign(_pred=pred_te).copy()
    day_agg = test_with_pred.groupby('date').agg(
        pred_mean=('_pred', 'mean'),
        actual_net=('target', 'first'),
        regime_2d=('regime_2d', 'first'),
        n_bars=('_pred', 'size'),
    ).reset_index()
    day_actual_dir = np.sign(day_agg['actual_net'].values)
    day_pred_dir = np.sign(day_agg['pred_mean'].values)
    valid_d = day_actual_dir != 0
    day_acc = float((day_actual_dir[valid_d] == day_pred_dir[valid_d]).mean()) \
        if valid_d.any() else float('nan')
    day_baseline = float(max((day_actual_dir[valid_d] == 1).mean(),
                               (day_actual_dir[valid_d] == -1).mean())) \
        if valid_d.any() else 0.5

    # Per-regime stratification (bar-level)
    per_regime_rows = []
    for r2d, group in test_with_pred.groupby('regime_2d'):
        a = np.sign(group['target'].values)
        p = np.sign(group['_pred'].values)
        m = a != 0
        if m.sum() == 0:
            continue
        sub_acc = float((a[m] == p[m]).mean())
        per_regime_rows.append({
            'regime_2d': r2d,
            'n_bars': int(len(group)),
            'n_days': int(group['date'].nunique()),
            'accuracy_bar': sub_acc,
            'mean_pred': float(group['_pred'].mean()),
            'mean_actual': float(group['target'].mean()),
        })

    if verbose:
        print(f"\n  Test R²: {r2_te:.4f}")
        print(f"  Bar-level direction acc: {bar_acc:.1%} "
              f"(baseline {bar_baseline:.1%}, lift {bar_acc-bar_baseline:+.1%})")
        print(f"  Day-level direction acc: {day_acc:.1%} on {valid_d.sum()} days "
              f"(baseline {day_baseline:.1%}, lift {day_acc-day_baseline:+.1%})")
        print(f"\n  Per-regime (bar-level on test):")
        for r in per_regime_rows:
            print(f"    {r['regime_2d']:<14} n={r['n_bars']:>5} ({r['n_days']:>3} days)  "
                  f"acc={r['accuracy_bar']:.1%}  mean_pred={r['mean_pred']:+8.1f}  "
                  f"mean_actual={r['mean_actual']:+8.1f}")

    return {
        'r2_test': r2_te,
        'bar_acc': bar_acc,
        'bar_baseline': bar_baseline,
        'day_acc': day_acc,
        'day_baseline': day_baseline,
        'n_test_bars': len(test),
        'n_test_days': int(valid_d.sum()),
        'per_regime': per_regime_rows,
        'test_pred': pred_te,
        'test_target': y_te,
        'test_dates': test['date'].values,
        'test_regimes': test['regime_2d'].values,
        'day_agg': day_agg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--ridge-alpha', type=float, default=0.0,
                        help='Ridge regularization (0 = plain OLS)')
    parser.add_argument('--binarize', action='store_true',
                        help='Use sign(net_move) as target instead of magnitude')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_regress_trend_direction')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Regress — trend direction (day net_move target)")
    print(f"  Base TF: {args.base_tf}")
    print(f"  Target: {'sign(net_move)' if args.binarize else 'net_move (signed magnitude)'}")
    print(f"  Ridge alpha: {args.ridge_alpha}")
    print(f"{'='*70}")

    # Load base TF OHLC
    print(f"\n--- Step 1: load base {args.base_tf} OHLC ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if base_df.empty:
        print(f"ERROR: no OHLC for {args.base_tf}")
        return
    print(f"  {len(base_df):,} bars")

    # Build base TS ints + day labels
    print(f"\n--- Step 2: load regime labels + merge ---")
    labels_df = load_regime_labels(args.labels_csv)
    print(f"  Labels: {len(labels_df)} days")

    base_with_targets = attach_day_targets(base_df, args.base_tf, labels_df,
                                             binarize=args.binarize)
    print(f"  Bars after merge: {len(base_with_targets):,}")

    # Load v2 features
    print(f"\n--- Step 3: load v2 features ---")
    ts_int = base_with_targets['ts_int'].values.astype(np.int64)
    ts_min = int(ts_int.min())
    ts_max = int(ts_int.max())
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    print(f"  v2 features: {len(features_5s):,} 5s rows")

    aligned = align_v2_to_base_tf(features_5s, ts_int)
    feature_cols = [c for c in aligned.columns if c != 'timestamp']
    print(f"  Aligned features: {aligned.shape}, {len(feature_cols)} cols")

    # Concat: bars (with targets) + features
    full = pd.concat([base_with_targets.reset_index(drop=True), aligned.reset_index(drop=True)],
                      axis=1)

    # Fit + score
    print(f"\n--- Step 4: fit + score ---")
    result = fit_and_score(full, feature_cols, ridge_alpha=args.ridge_alpha,
                            verbose=True)

    # Persist per-regime CSV
    pr_path = os.path.join(args.output_dir, 'per_regime.csv')
    pd.DataFrame(result['per_regime']).to_csv(pr_path, index=False)
    print(f"\n  [saved] {pr_path}")

    # Per-bar predictions (test only) — sample if very large
    pred_df = pd.DataFrame({
        'date': result['test_dates'],
        'regime_2d': result['test_regimes'],
        'actual_target': result['test_target'],
        'pred_target': result['test_pred'],
        'actual_dir': np.sign(result['test_target']),
        'pred_dir': np.sign(result['test_pred']),
    })
    pred_path = os.path.join(args.output_dir, 'per_bar_predictions.csv')
    if len(pred_df) > 200_000:
        pred_df.iloc[::5].to_csv(pred_path, index=False)  # downsample 5x
        print(f"  [saved] {pred_path} (downsampled 5x, "
              f"{len(pred_df)//5} of {len(pred_df)} rows)")
    else:
        pred_df.to_csv(pred_path, index=False)
        print(f"  [saved] {pred_path} ({len(pred_df)} rows)")

    # Day-level aggregate
    day_path = os.path.join(args.output_dir, 'per_day_predictions.csv')
    result['day_agg'].to_csv(day_path, index=False)
    print(f"  [saved] {day_path}")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 Trend Direction Regression — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`\n")
        f.write(f"**Target:** `{'sign(net_move)' if args.binarize else 'net_move (signed)'}`\n")
        f.write(f"**Ridge alpha:** {args.ridge_alpha}\n")
        f.write(f"**Split:** day-level IS/VAL/OOS from `regime_labels_2d.csv`\n\n")

        f.write(f"## Test scores\n\n")
        f.write(f"- R²: {result['r2_test']:.4f}\n")
        f.write(f"- Bar-level direction accuracy: {result['bar_acc']:.1%} "
                f"(baseline {result['bar_baseline']:.1%}, "
                f"lift {result['bar_acc']-result['bar_baseline']:+.1%})\n")
        f.write(f"- Day-level direction accuracy: {result['day_acc']:.1%} "
                f"on {result['n_test_days']} days "
                f"(baseline {result['day_baseline']:.1%}, "
                f"lift {result['day_acc']-result['day_baseline']:+.1%})\n\n")

        f.write(f"## Per-regime accuracy (bar-level, test split)\n\n")
        f.write(pd.DataFrame(result['per_regime']).to_string(index=False))
        f.write("\n\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
