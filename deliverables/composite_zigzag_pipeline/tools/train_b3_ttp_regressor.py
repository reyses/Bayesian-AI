"""B3 — Time-to-pivot regressor.

Predicts seconds-to-next-zigzag-pivot per 1m bar (continuous target).
Built in parallel to B1 (binary "pivot within K") and B2 (per-pivot fakeout) —
does NOT replace them. Goal: directly estimate countdown.

User hypothesis (2026-05-17): if the model can predict "pivot in 5 min"
with some skill, it should predict "pivot in 10 seconds" with MORE skill —
because the V2 features right before a pivot (5s/15s volume spikes,
z-extremes) directly encode the imminent reversal.

Empirical test: does MAE shrink at small actual-TTN buckets?

Labels: seconds_to_next_pivot per bar (capped at MAX_TTN_S = 3600s).
Model:  HistGradientBoostingRegressor(loss='absolute_error') — MAE-optimal,
        robust to long-leg outliers.
Train:  full IS (2025, 277 days)
Eval:   NT8 OOS (32 days)
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


MAX_TTN_S = 3600   # cap target at 1h — beyond this we just say "far"


def pivot_centroid_ts(day_df: pd.DataFrame) -> np.ndarray:
    """Same as B1: collapse is_pivot==1 runs to centroid timestamps."""
    piv_rows = day_df[day_df['is_pivot'] == 1].sort_values('timestamp')
    if len(piv_rows) == 0:
        return np.array([], dtype=np.int64)
    ts = piv_rows['timestamp'].values.astype(np.int64)
    groups = [[ts[0]]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([ts[i]])
        else:
            groups[-1].append(ts[i])
    return np.array([int(np.median(g)) for g in groups], dtype=np.int64)


def build_ttp_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'ttn_seconds' column = time-to-next-pivot in seconds.
    Capped at MAX_TTN_S. Bars after the last pivot get capped value."""
    df = df.sort_values(['day', 'timestamp']).reset_index(drop=True)
    out = np.full(len(df), MAX_TTN_S, dtype=np.float64)
    for day, g in df.groupby('day'):
        pivots = pivot_centroid_ts(g)
        if len(pivots) == 0:
            continue
        bar_ts = g['timestamp'].values.astype(np.int64)
        idx = np.searchsorted(pivots, bar_ts, side='left')
        for i, k in enumerate(idx):
            if k < len(pivots):
                dt = pivots[k] - bar_ts[i]
                out[g.index[i]] = min(float(dt), MAX_TTN_S)
    df['ttn_seconds'] = out
    return df


def bucket_label(ttn_s):
    """Map TTN seconds to a bucket label for stratified analysis."""
    if ttn_s < 30: return '[0, 30s)'
    if ttn_s < 60: return '[30s, 1m)'
    if ttn_s < 120: return '[1m, 2m)'
    if ttn_s < 300: return '[2m, 5m)'
    if ttn_s < 600: return '[5m, 10m)'
    if ttn_s < 1200: return '[10m, 20m)'
    return '[20m+)'


BUCKET_ORDER = ['[0, 30s)', '[30s, 1m)', '[1m, 2m)', '[2m, 5m)',
                '[5m, 10m)', '[10m, 20m)', '[20m+)']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    ap.add_argument('--oos-dataset',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-pkl',
                    default='reports/findings/regret_oracle/b3_ttp_regressor.pkl')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/b3_ttp_regressor.txt')
    ap.add_argument('--out-cache',
                    default='reports/findings/regret_oracle/b3_ttp_OOS_NT8.parquet')
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading IS:', args.is_dataset)
    is_df = pd.read_parquet(args.is_dataset)
    print(f'  {len(is_df)} bars / {is_df["day"].nunique()} days')
    print('Loading OOS:', args.oos_dataset)
    oos_df = pd.read_parquet(args.oos_dataset)
    print(f'  {len(oos_df)} bars / {oos_df["day"].nunique()} days')

    print('Building TTN labels (IS)...')
    is_df = build_ttp_target(is_df)
    print('Building TTN labels (OOS)...')
    oos_df = build_ttp_target(oos_df)

    print(f'IS  TTN stats: median {is_df["ttn_seconds"].median():.0f}s   '
          f'p25 {is_df["ttn_seconds"].quantile(0.25):.0f}s   '
          f'p75 {is_df["ttn_seconds"].quantile(0.75):.0f}s')
    print(f'OOS TTN stats: median {oos_df["ttn_seconds"].median():.0f}s   '
          f'p25 {oos_df["ttn_seconds"].quantile(0.25):.0f}s   '
          f'p75 {oos_df["ttn_seconds"].quantile(0.75):.0f}s')

    v2_cols = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'V2 features: {len(v2_cols)}')

    X_is  = is_df[v2_cols].fillna(0.0).values.astype(np.float32)
    y_is  = is_df['ttn_seconds'].values.astype(np.float32)
    X_oos = oos_df[v2_cols].fillna(0.0).values.astype(np.float32)
    y_oos = oos_df['ttn_seconds'].values.astype(np.float32)

    print('Training HistGradientBoostingRegressor (MAE objective)...')
    model = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=args.max_iter, learning_rate=0.05,
        max_depth=6, min_samples_leaf=50,
        l2_regularization=0.5,
        random_state=args.seed,
    )
    model.fit(X_is, y_is)

    # Predict
    print('Predicting...')
    p_is  = model.predict(X_is)
    p_oos = model.predict(X_oos)
    p_is  = np.clip(p_is, 0, MAX_TTN_S)
    p_oos = np.clip(p_oos, 0, MAX_TTN_S)

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out(f'B3 TIME-TO-PIVOT REGRESSOR — target = sec-to-next-zigzag-pivot')
    out('=' * 78)
    out(f'IS rows: {len(is_df):,}   OOS rows: {len(oos_df):,}   features: {len(v2_cols)}')
    out('')

    # Overall MAE in minutes
    mae_is_min  = float(np.mean(np.abs(p_is  - y_is))  / 60)
    mae_oos_min = float(np.mean(np.abs(p_oos - y_oos)) / 60)
    out(f'Overall MAE:  IS {mae_is_min:.2f} min   OOS {mae_oos_min:.2f} min')

    # Baseline: predict median target
    median_pred = float(np.median(y_is))
    mae_baseline = float(np.mean(np.abs(median_pred - y_oos)) / 60)
    out(f'Baseline (predict {median_pred/60:.1f}min always): MAE {mae_baseline:.2f} min')
    out(f'Lift over baseline:  {(mae_baseline - mae_oos_min):.2f} min   '
        f'({(1 - mae_oos_min/mae_baseline)*100:.1f}% reduction)')
    out('')

    # === MAE conditional on ACTUAL TTN bucket — the user's hypothesis test ===
    out('--- MAE BY ACTUAL TTN BUCKET (the key test) ---')
    out('  Tests user hypothesis: "predicting 10s out should be easier')
    out('  than 5min out".  If true, MAE shrinks for small actual TTN.')
    out('')
    out(f'  {"bucket":<14}  {"n":>8}  {"actual_mean":>11}  {"pred_mean":>10}  '
        f'{"MAE_sec":>9}  {"MAE_min":>8}')
    oos_df['pred_ttn'] = p_oos
    oos_df['actual_bucket'] = pd.Categorical(
        [bucket_label(t) for t in y_oos],
        categories=BUCKET_ORDER, ordered=True,
    )
    for bk in BUCKET_ORDER:
        sub = oos_df[oos_df['actual_bucket'] == bk]
        if len(sub) == 0:
            out(f'  {bk:<14}  {"-":>8}'); continue
        actual = sub['ttn_seconds'].values
        pred = sub['pred_ttn'].values
        mae = np.mean(np.abs(pred - actual))
        out(f'  {bk:<14}  {len(sub):>8,}  {actual.mean():>10.1f}s  '
            f'{pred.mean():>9.1f}s  {mae:>8.1f}s  {mae/60:>7.2f}m')

    out('')
    out('--- CALIBRATION: PREDICTED bucket → actual TTN distribution ---')
    out(f'  {"pred_bucket":<14}  {"n":>8}  {"actual_med":>11}  {"actual_p25":>11}  '
        f'{"actual_p75":>11}  {"% w/ pivot<30s":>15}')
    pred_bucket = pd.Categorical(
        [bucket_label(t) for t in p_oos],
        categories=BUCKET_ORDER, ordered=True,
    )
    oos_df['pred_bucket'] = pred_bucket
    for bk in BUCKET_ORDER:
        sub = oos_df[oos_df['pred_bucket'] == bk]
        if len(sub) == 0:
            out(f'  {bk:<14}  {"-":>8}'); continue
        actual = sub['ttn_seconds'].values
        pct_30s = float((actual < 30).mean() * 100)
        out(f'  {bk:<14}  {len(sub):>8,}  {np.median(actual):>10.1f}s  '
            f'{np.percentile(actual,25):>10.1f}s  '
            f'{np.percentile(actual,75):>10.1f}s  {pct_30s:>14.2f}%')

    # === Game-changer test: when model says "pivot in <30s", what's reality? ===
    out('')
    out('--- HEADLINE: when B3 predicts pivot < 30s, what actually happens? ---')
    short_mask = p_oos < 30
    n_short = int(short_mask.sum())
    if n_short > 0:
        actual_short = y_oos[short_mask]
        out(f'  predictions: {n_short:,} ({n_short/len(p_oos)*100:.2f}% of bars)')
        out(f'  actual TTN:  median {np.median(actual_short):.1f}s   '
            f'mean {actual_short.mean():.1f}s   p75 {np.percentile(actual_short,75):.1f}s')
        out(f'  % truly within 30s: {(actual_short < 30).mean()*100:.1f}%')
        out(f'  % truly within 60s: {(actual_short < 60).mean()*100:.1f}%')
        out(f'  % truly within 120s: {(actual_short < 120).mean()*100:.1f}%')
    else:
        out('  No predictions below 30s — model conservative.')

    # Even narrower: predicted < 60s
    out('')
    out('--- When B3 predicts pivot < 60s ---')
    mid_mask = p_oos < 60
    n_mid = int(mid_mask.sum())
    if n_mid > 0:
        actual_mid = y_oos[mid_mask]
        out(f'  predictions: {n_mid:,} ({n_mid/len(p_oos)*100:.2f}% of bars)')
        out(f'  actual TTN:  median {np.median(actual_mid):.1f}s   '
            f'mean {actual_mid.mean():.1f}s   p75 {np.percentile(actual_mid,75):.1f}s')
        out(f'  % within 60s:  {(actual_mid < 60).mean()*100:.1f}%')
        out(f'  % within 120s: {(actual_mid < 120).mean()*100:.1f}%')
        out(f'  % within 300s: {(actual_mid < 300).mean()*100:.1f}%')

    # Save model + per-bar predictions
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump({'model': model, 'v2_cols': v2_cols, 'max_ttn_s': MAX_TTN_S},
                    f)
    # Per-bar cache for inspector
    cache = oos_df[['timestamp', 'day']].copy()
    cache['ttn_pred_seconds'] = p_oos
    cache['ttn_actual_seconds'] = y_oos
    cache.to_parquet(args.out_cache, index=False)

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_pkl}')
    print(f'Wrote: {args.out_cache}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
