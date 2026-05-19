"""DRS Path A — Step 1: apply trained B-models to IS data.

Generates the per-bar / per-pivot / per-leg prediction caches for IS days
that the deliverable's composite forward pass needs but never created.

Applies (READ-ONLY) the trained pickles:
  - B1 (pivot-imminent)  -> per-1m-bar P(pivot in K min), K in {1,3,5,10}
  - B2 (fakeout)         -> per-pivot-event P(fakeout in K min), K in {3,5,10}
  - B4 (pivot-region)    -> per-1m-bar P(in +/- W of pivot), W in {30,60,120,300}
  - B5 (leg-phase)       -> per-in-leg-1m-bar P(EARLY/MID/LATE)
  - B6 (directional)     -> per-1m-bar P(NO/LONG-PIVOT/SHORT-PIVOT in K)
  - B8 (hour-risk)       -> per-1m-bar E[forward 60-min total leg P&L]

B7 IS predictions are produced separately by drs_b7_predict_is.py.

Inputs:
  deliverables/composite_zigzag_pipeline/models/*.pkl
  deliverables/composite_zigzag_pipeline/caches/zigzag_pivot_dataset_IS_atr4.parquet

Outputs (in DATA/CROSS_DAY/predictions_IS/):
  b1_proba_IS.parquet
  b2_proba_IS.parquet
  b4_proba_IS.parquet
  b5_proba_IS.parquet
  b6_proba_IS.parquet
  b8_proba_IS.parquet

Run:
  python tools/sourcing/drs_a_step1_predict_b_models_is.py

Expected runtime: ~5-10 min on CPU (each model ~30-60s of inference over 282k bars).
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DELIVER = Path('deliverables/composite_zigzag_pipeline')
MODELS  = DELIVER / 'models'
IS_DATASET = DELIVER / 'caches' / 'zigzag_pivot_dataset_IS_atr4.parquet'

OUT_DIR = Path('DATA/CROSS_DAY/predictions_IS')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def derive_pivot_centroids(day_df: pd.DataFrame, time_col='timestamp'):
    """Same group-by-90s clustering as deliverable's pivot_centroid_events.
    Returns (centroid_indices, centroid_timestamps, pivot_dirs, pivot_prices)."""
    piv = day_df[day_df['is_pivot'] == 1].sort_values(time_col).reset_index()
    if len(piv) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), [], []
    ts = piv[time_col].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    pp_ = piv['pivot_price'].values
    idx0 = piv['index'].values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i])
        else:
            groups[-1].append(i)
    out_idx, out_ts, out_dir, out_px = [], [], [], []
    for grp in groups:
        mid = grp[len(grp) // 2]
        out_idx.append(int(idx0[mid]))
        out_ts.append(int(np.median(ts[grp])))
        vals, counts = np.unique(pd_[grp], return_counts=True)
        out_dir.append(str(vals[np.argmax(counts)]))
        out_px.append(float(np.mean(pp_[grp])))
    return (np.array(out_idx, dtype=np.int64),
            np.array(out_ts, dtype=np.int64),
            out_dir, out_px)


def main():
    print(f'Loading IS dataset: {IS_DATASET}')
    is_df = pd.read_parquet(IS_DATASET)
    print(f'  {len(is_df):,} bars / {is_df["day"].nunique()} days')

    v2_cols_all = [c for c in is_df.columns if c.startswith(('L1_', 'L2_', 'L3_'))]
    print(f'  V2 features in dataset: {len(v2_cols_all)}')

    # --- B1: per-1m-bar pivot-imminent ---
    print('\n[B1] per-1m-bar pivot-imminent...')
    with open(MODELS / 'b1_pivot_imminent.pkl', 'rb') as f:
        b1 = pickle.load(f)
    cols_b1 = b1[list(b1.keys())[0]]['v2_cols']
    X = is_df[cols_b1].fillna(0.0).values.astype(np.float32)
    out_b1 = is_df[['timestamp', 'day']].copy()
    for K in sorted(b1.keys()):
        out_b1[f'p_pivot_{K}m'] = b1[K]['model'].predict_proba(X)[:, 1]
    p1 = OUT_DIR / 'b1_proba_IS.parquet'
    out_b1.to_parquet(p1, index=False)
    print(f'  -> {p1}  ({len(out_b1):,} rows)')

    # --- B2: per-pivot fakeout ---
    print('\n[B2] per-pivot-event fakeout...')
    with open(MODELS / 'b2_fakeout.pkl', 'rb') as f:
        b2 = pickle.load(f)
    cols_b2 = b2[list(b2.keys())[0]]['v2_cols']
    # Build per-pivot dataset by centroid extraction
    rows = []
    for day in tqdm(sorted(is_df['day'].unique()), desc='B2 days', file=sys.stdout):
        day_df = is_df[is_df['day'] == day].reset_index(drop=True)
        idx, ts, _, _ = derive_pivot_centroids(day_df)
        if len(idx) == 0:
            continue
        sub = day_df.iloc[idx][['timestamp', 'day'] + cols_b2].copy()
        sub['centroid_ts'] = ts
        rows.append(sub)
    if rows:
        piv_df = pd.concat(rows, ignore_index=True)
        X_piv = piv_df[cols_b2].fillna(0.0).values.astype(np.float32)
        out_b2 = piv_df[['timestamp', 'day']].copy()
        for K in sorted(b2.keys()):
            out_b2[f'p_fakeout_{K}m'] = b2[K]['model'].predict_proba(X_piv)[:, 1]
        p2 = OUT_DIR / 'b2_proba_IS.parquet'
        out_b2.to_parquet(p2, index=False)
        print(f'  -> {p2}  ({len(out_b2):,} pivot rows)')

    # --- B4: per-1m-bar pivot-region ---
    print('\n[B4] per-1m-bar pivot-region...')
    with open(MODELS / 'b4_pivot_region.pkl', 'rb') as f:
        b4 = pickle.load(f)
    cols_b4 = b4[list(b4.keys())[0]]['v2_cols']
    X = is_df[cols_b4].fillna(0.0).values.astype(np.float32)
    out_b4 = is_df[['timestamp', 'day']].copy()
    for W in sorted(b4.keys()):
        out_b4[f'p_region_{W}s'] = b4[W]['model'].predict_proba(X)[:, 1]
    p4 = OUT_DIR / 'b4_proba_IS.parquet'
    out_b4.to_parquet(p4, index=False)
    print(f'  -> {p4}  ({len(out_b4):,} rows)')

    # --- B5: leg-phase 3-class ---
    print('\n[B5] per-in-leg-bar leg-phase...')
    with open(MODELS / 'b5_leg_phase.pkl', 'rb') as f:
        b5 = pickle.load(f)
    cols_b5 = b5['v2_cols']
    classes = list(b5['classes'])
    X = is_df[cols_b5].fillna(0.0).values.astype(np.float32)
    proba = b5['model'].predict_proba(X)
    out_b5 = is_df[['timestamp', 'day']].copy()
    for k, cls in enumerate(classes):
        out_b5[f'p_{cls}'] = proba[:, k]
    p5 = OUT_DIR / 'b5_proba_IS.parquet'
    out_b5.to_parquet(p5, index=False)
    print(f'  -> {p5}  ({len(out_b5):,} rows)')

    # --- B6: directional pivot ---
    print('\n[B6] per-1m-bar directional pivot...')
    with open(MODELS / 'b6_directional_pivot.pkl', 'rb') as f:
        b6 = pickle.load(f)
    le = b6['label_encoder']
    classes = list(le.classes_)
    out_b6 = is_df[['timestamp', 'day']].copy()
    for K in sorted(b6['models'].keys()):
        sub = b6['models'][K]
        cols_k = sub['v2_cols']
        X = is_df[cols_k].fillna(0.0).values.astype(np.float32)
        proba = sub['model'].predict_proba(X)
        for k, cls in enumerate(classes):
            out_b6[f'p_{cls}_{K}m'] = proba[:, k]
    p6 = OUT_DIR / 'b6_proba_IS.parquet'
    out_b6.to_parquet(p6, index=False)
    print(f'  -> {p6}  ({len(out_b6):,} rows)')

    # --- B8: hour-risk regression ---
    print('\n[B8] per-1m-bar hour-risk...')
    with open(MODELS / 'b8_hour_risk.pkl', 'rb') as f:
        b8 = pickle.load(f)
    feat_cols = b8['feat_cols']
    # Need to inject the time features (hour_sin, hour_cos, minute_of_session)
    is_aug = is_df.copy()
    dt = pd.to_datetime(is_aug['timestamp'], unit='s', utc=True)
    is_aug['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    is_aug['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    is_aug['minute_of_session'] = (is_aug.groupby('day')['timestamp']
                                    .transform(lambda s: (s - s.min()).astype('int64') / 60))
    X = is_aug[feat_cols].fillna(0.0).values.astype(np.float32)
    out_b8 = is_aug[['timestamp', 'day']].copy()
    out_b8['pred_hour_pnl'] = b8['model'].predict(X)
    p8 = OUT_DIR / 'b8_proba_IS.parquet'
    out_b8.to_parquet(p8, index=False)
    print(f'  -> {p8}  ({len(out_b8):,} rows)')

    print('\n=== ALL B-model IS predictions DONE ===')
    print(f'Output dir: {OUT_DIR}')


if __name__ == '__main__':
    main()
