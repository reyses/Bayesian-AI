"""CNN dataset builder.

Builds (X_grid, X_tod, X_regime, y) tuples by:
  1. Loading V2 features for a list of days via core_v2.features.load_features.
  2. Loading 1m OHLCV per day for forward-return computation.
  3. Sampling at the requested cadence (5m by default).
  4. Computing the forward return at horizon `forward_minutes` after each
     sample bar, in TICKS.
  5. Labeling: SHORT (0) if fwd < -tick_threshold, LONG (2) if fwd > +tick_threshold,
     FLAT (1) otherwise.

Output is held in numpy arrays so it fits comfortably in RAM:
  X_grid : (N, 8, 23)   float32
  X_tod  : (N, 1)       float32
  X_reg  : (N,)         int64
  y      : (N,)         int64

Forward-return horizon and tick threshold are tunable. Defaults from EDA:
  - forward_minutes = 5  (5m bar = the cadence MA-align EDA was run at)
  - tick_threshold  = 4 (1 point on MNQ; smaller than that is noise)
"""
from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from core_v2.features import load_features, FEATURE_NAMES, DEFAULT_FEATURES_ROOT
from training_iso_v2.state import regime_to_idx
from training_iso_v2.cnn.model import GRID_FLAT_IDX, L0_IDX, GRID_H, GRID_W


ATLAS_ROOT = 'DATA/ATLAS'
LABELS_CSV = 'DATA/ATLAS/regime_labels_2d.csv'
TICK = 0.25


def _load_regime_lookup(labels_csv: str = LABELS_CSV) -> dict:
    df = pd.read_csv(labels_csv)
    df['date'] = df['date'].astype(str).str[:10]
    return dict(zip(df['date'], df['regime_2d']))


def _resolve_days(target: str = 'is') -> List[str]:
    l0_dir = os.path.join(DEFAULT_FEATURES_ROOT, 'L0')
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    if target == 'is':
        return [d for d in days if d.startswith('2025_')]
    if target == 'oos':
        return [d for d in days if d.startswith('2026_')]
    return days


def _build_one_day(day: str, regime_lookup: dict,
                        atlas_root: str, features_root: str,
                        cadence_s: int = 300,
                        forward_s: int = 300,
                        tick_threshold: float = 4.0,
                        ) -> Optional[Tuple[np.ndarray, ...]]:
    """Build per-day dataset rows. Returns None if data missing."""
    feats = load_features(days=[day], root=features_root, require_all=False)
    if feats.empty:
        return None
    feats = feats.sort_values('timestamp').reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(feats['timestamp']):
        feats['timestamp'] = (feats['timestamp'].astype('int64') // 10**9)
    ts = feats['timestamp'].values.astype(np.int64)

    # 1m OHLCV for forward-return computation
    ohlcv_path = os.path.join(atlas_root, '1m', f'{day}.parquet')
    if not os.path.exists(ohlcv_path):
        return None
    ohlcv = pd.read_parquet(ohlcv_path).sort_values('timestamp').reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
        ohlcv['timestamp'] = (ohlcv['timestamp'].astype('int64') // 10**9)
    o_ts = ohlcv['timestamp'].values.astype(np.int64)
    o_close = ohlcv['close'].values.astype(np.float64)

    # Sample bar mask: bars at the cadence boundary
    sample_mask = (ts % cadence_s) == 0
    sample_idx = np.where(sample_mask)[0]
    if len(sample_idx) == 0:
        return None

    # Build feature matrix once
    v2_matrix = np.zeros((len(feats), len(FEATURE_NAMES)), dtype=np.float32)
    feat_cols = set(feats.columns)
    for j, name in enumerate(FEATURE_NAMES):
        if name in feat_cols:
            v2_matrix[:, j] = feats[name].values.astype(np.float32)

    # Forward returns: at each sample bar, find price now + forward_s
    sample_ts = ts[sample_idx]
    now_idx = np.searchsorted(o_ts, sample_ts, side='right') - 1
    fwd_idx = np.searchsorted(o_ts, sample_ts + forward_s, side='right') - 1
    valid = (now_idx >= 0) & (fwd_idx >= 0) & (fwd_idx < len(o_close))
    sample_idx = sample_idx[valid]
    now_idx = now_idx[valid]
    fwd_idx = fwd_idx[valid]
    if len(sample_idx) == 0:
        return None

    now_price = o_close[now_idx]
    fwd_price = o_close[fwd_idx]
    fwd_ticks = (fwd_price - now_price) / TICK

    # Labels
    y = np.full(len(sample_idx), 1, dtype=np.int64)  # FLAT default
    y[fwd_ticks > tick_threshold] = 2                  # LONG
    y[fwd_ticks < -tick_threshold] = 0                 # SHORT

    # Features at sample bars: extract grid + tod
    sample_v2 = v2_matrix[sample_idx]                  # (n, 185)
    grids = sample_v2[:, GRID_FLAT_IDX].reshape(-1, GRID_H, GRID_W)
    tods = sample_v2[:, L0_IDX].reshape(-1, 1)

    iso = day.replace('_', '-')
    regime_2d = regime_lookup.get(iso, 'UNKNOWN')
    regime_idx = regime_to_idx(regime_2d)
    regs = np.full(len(sample_idx), regime_idx, dtype=np.int64)

    return grids.astype(np.float32), tods.astype(np.float32), regs, y


def build_dataset(target: str = 'is',
                       cadence_s: int = 300,
                       forward_s: int = 300,
                       tick_threshold: float = 4.0,
                       atlas_root: str = ATLAS_ROOT,
                       features_root: str = DEFAULT_FEATURES_ROOT,
                       labels_csv: str = LABELS_CSV,
                       days: Optional[List[str]] = None,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build full dataset across a target's days."""
    regime_lookup = _load_regime_lookup(labels_csv)
    if days is None:
        days = _resolve_days(target)

    grids_l, tods_l, regs_l, ys_l = [], [], [], []
    for day in tqdm(days, desc=f'CNN dataset {target}'):
        out = _build_one_day(day, regime_lookup, atlas_root, features_root,
                                  cadence_s=cadence_s, forward_s=forward_s,
                                  tick_threshold=tick_threshold)
        if out is None:
            continue
        g, t, r, y = out
        grids_l.append(g)
        tods_l.append(t)
        regs_l.append(r)
        ys_l.append(y)

    if not grids_l:
        return (np.zeros((0, GRID_H, GRID_W), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64))

    X_grid = np.concatenate(grids_l, axis=0)
    X_tod = np.concatenate(tods_l, axis=0)
    X_reg = np.concatenate(regs_l, axis=0)
    y = np.concatenate(ys_l, axis=0)
    return X_grid, X_tod, X_reg, y
