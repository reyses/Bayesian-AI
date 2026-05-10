"""
core_v2/v2_to_v1_inmemory.py — build a V1-shape (91D + extension signals)
DataFrame in MEMORY from V2 layered features + raw OHLCV, on a per-day
basis. Same logic as tools/build_v2_to_v1_compat_cache.py but returns
a DataFrame instead of writing parquet — so the engine can read V2
features directly without a pre-built compat cache.

Goal: eliminate the on-disk compat cache. The compat cache was a
transitional artifact; this module makes it unnecessary.

Used by training_v2/sfe_ticker.py (V2NativeTicker class).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_v2 import v1_compat
from tools.research.features_v2 import load_v2_features


# V1 91D layout (same as build_v2_to_v1_compat_cache.py)
V1_TFS = ['15s', '1m', '5m', '15m', '1h', '1D']
V1_CORE = ['z_se', 'dmi_diff', 'variance_ratio', 'velocity', 'acceleration',
              'vol_rel', 'bar_range', 'hurst', 'reversion_prob', 'p_at_center',
              'z_high', 'z_low']
V1_HELPERS = ['dmi_gap', 'dir_vol', 'wick_ratio']
V2_EXTENSION_HELPERS = ['upper_wick', 'lower_wick', 'body', 'swing_noise',
                            'vwap_dist', 'vol_velocity']
V2_4H_SIGNALS = ['4h_z_se', '4h_velocity', '4h_body', '4h_swing_noise']

V1_TO_V2_DIRECT = {
    'z_se': lambda tf: f'L3_{tf}_z_se_w',
    'velocity': lambda tf: f'L2_{tf}_price_velocity_w',
    'acceleration': lambda tf: f'L2_{tf}_price_accel_w',
    'bar_range': lambda tf: f'L1_{tf}_bar_range',
    'hurst': lambda tf: f'L3_{tf}_hurst_w',
    'reversion_prob': lambda tf: f'L3_{tf}_reversion_prob_w',
    'z_high': lambda tf: f'L3_{tf}_z_high_w',
    'z_low': lambda tf: f'L3_{tf}_z_low_w',
}

TICK = 0.25
V1_UNIT_CONVERSIONS = {'bar_range': 1.0 / TICK}
TF_PERIOD_S = {'15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1D': 86400}


_REGIME_LOOKUP: Optional[dict] = None


def _lookup_regime_for_day(day_yyyy_mm_dd: str,
                              labels_csv: str = 'DATA/ATLAS/regime_labels_2d.csv') -> str:
    global _REGIME_LOOKUP
    if _REGIME_LOOKUP is None:
        try:
            from tools.atlas_regime_labeler_2d import load_regime_labels
            df = load_regime_labels(labels_csv).copy()
            df['date'] = df['date'].astype(str).str[:10]
            _REGIME_LOOKUP = dict(zip(df['date'], df['regime_2d']))
        except Exception:
            _REGIME_LOOKUP = {}
    iso = day_yyyy_mm_dd.replace('_', '-')
    return _REGIME_LOOKUP.get(iso, 'UNKNOWN')


def _load_ohlcv_with_history(atlas_root: str, tf: str, day: str,
                                  history_days: int = 4) -> Optional[pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(atlas_root, tf, '*.parquet')))
    target_basename = f'{day}.parquet'
    target_idx = None
    for i, f in enumerate(files):
        if os.path.basename(f) == target_basename:
            target_idx = i
            break
    if target_idx is None:
        return None
    take = files[max(target_idx - history_days, 0):target_idx + 1]
    dfs = [pd.read_parquet(p) for p in take]
    df = pd.concat(dfs, ignore_index=True)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df


def build_v1_shape_for_day(atlas_root: str, day: str,
                                labels_csv: str = 'DATA/ATLAS/regime_labels_2d.csv',
                                return_v2: bool = False):
    """Build V1-shape (91D + V2 extension columns + regime) DataFrame for one
    day, in memory. Returns None if data missing.

    Same output schema as tools/build_v2_to_v1_compat_cache.py but no disk write.

    If return_v2=True, returns (compat_df, v2_df) tuple where v2_df is the raw
    V2 layered features (185D) aligned 1:1 with compat_df rows.
    """
    raw_5s_path = os.path.join(atlas_root, '5s', f'{day}.parquet')
    if not os.path.exists(raw_5s_path):
        return None
    raw_5s = pd.read_parquet(raw_5s_path)
    if pd.api.types.is_datetime64_any_dtype(raw_5s['timestamp']):
        raw_5s = raw_5s.copy()
        raw_5s['timestamp'] = (raw_5s['timestamp'].astype('int64') // 10**9)
    ts = raw_5s['timestamp'].values.astype(np.int64)
    if len(ts) == 0:
        return None
    ts_min, ts_max = int(ts.min()), int(ts.max())

    v2 = load_v2_features(
        v2_dir=os.path.join(atlas_root, 'FEATURES_5s_v2'),
        atlas_root=atlas_root, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    v2 = v2[v2['timestamp'].between(ts_min, ts_max)].reset_index(drop=True)
    if len(v2) == 0:
        return None

    n = len(v2)
    out = pd.DataFrame({'timestamp': v2['timestamp'].values})

    for tf in V1_TFS:
        ohlcv = _load_ohlcv_with_history(atlas_root, tf, day)
        if ohlcv is None:
            for c in V1_CORE:
                out[f'{tf}_{c}'] = np.zeros(n) if c != 'variance_ratio' else np.ones(n)
            for c in V1_HELPERS + V2_EXTENSION_HELPERS:
                out[f'{tf}_{c}'] = np.zeros(n)
            continue

        # Direct V2 mappings with unit conversion
        for c in ('z_se', 'velocity', 'acceleration', 'bar_range', 'hurst',
                    'reversion_prob', 'z_high', 'z_low'):
            v2_col = V1_TO_V2_DIRECT[c](tf)
            if v2_col in v2.columns:
                vals = v2[v2_col].values
                if c in V1_UNIT_CONVERSIONS:
                    vals = vals * V1_UNIT_CONVERSIONS[c]
                out[f'{tf}_{c}'] = vals
            else:
                out[f'{tf}_{c}'] = np.zeros(n)

        velocity = out[f'{tf}_velocity'].values
        out[f'{tf}_dmi_diff'] = np.where(velocity > 0, 5.0,
                                              np.where(velocity < 0, -5.0, 0.0))
        out[f'{tf}_dmi_gap'] = np.abs(out[f'{tf}_dmi_diff'].values)

        period = TF_PERIOD_S[tf]
        body_col = f'L1_{tf}_body'
        bar_range_col = f'L1_{tf}_bar_range'
        vel_col = f'L2_{tf}_price_velocity_w'
        z_se_col = f'L3_{tf}_z_se_w'
        if all(c in v2.columns for c in (body_col, bar_range_col, vel_col, z_se_col)):
            v2_view = v2[['timestamp', body_col, bar_range_col, vel_col, z_se_col]].copy()
            derived = v1_compat.derive_v1_concepts_batch(v2_view, ohlcv, tf, period)
            out[f'{tf}_variance_ratio'] = derived['variance_ratio'].values
            out[f'{tf}_vol_rel'] = derived['vol_rel'].values
            out[f'{tf}_p_at_center'] = derived['p_at_center'].values
            out[f'{tf}_dir_vol'] = derived['dir_vol'].values
            out[f'{tf}_wick_ratio'] = derived['wick_ratio'].values
        else:
            out[f'{tf}_variance_ratio'] = np.ones(n)
            out[f'{tf}_vol_rel'] = np.ones(n)
            out[f'{tf}_p_at_center'] = np.full(n, 0.5)
            out[f'{tf}_dir_vol'] = np.zeros(n)
            out[f'{tf}_wick_ratio'] = np.zeros(n)

        # V2 extensions (directional wicks + body + swing_noise + vwap_dist + vol_velocity)
        ohlcv_ts = ohlcv['timestamp'].values.astype(np.int64)
        opens = ohlcv['open'].values.astype(np.float64)
        highs = ohlcv['high'].values.astype(np.float64)
        lows = ohlcv['low'].values.astype(np.float64)
        closes = ohlcv['close'].values.astype(np.float64)
        upper_native, lower_native = v1_compat.directional_wicks_batch(
            opens, highs, lows, closes)
        anchor_ts = v2['timestamp'].values.astype(np.int64)
        last_closed = np.searchsorted(ohlcv_ts, anchor_ts - period, side='right') - 1
        valid = (last_closed >= 0) & (last_closed < len(ohlcv_ts))
        safe_idx = np.clip(last_closed, 0, len(ohlcv_ts) - 1)
        out[f'{tf}_upper_wick'] = np.where(valid, upper_native[safe_idx], 0.0)
        out[f'{tf}_lower_wick'] = np.where(valid, lower_native[safe_idx], 0.0)

        out[f'{tf}_body'] = (v2[body_col].values
                              if body_col in v2.columns else np.zeros(n))
        swing_col_v2 = f'L3_{tf}_swing_noise_w'
        out[f'{tf}_swing_noise'] = (v2[swing_col_v2].values
                                          if swing_col_v2 in v2.columns else np.zeros(n))
        vwap_col_v2 = f'L2_{tf}_vwap_w'
        if vwap_col_v2 in v2.columns:
            tf_close = np.where(valid, closes[safe_idx], 0.0)
            out[f'{tf}_vwap_dist'] = tf_close - v2[vwap_col_v2].values
        else:
            out[f'{tf}_vwap_dist'] = np.zeros(n)
        vol_vel_col_v2 = f'L2_{tf}_vol_velocity_w'
        out[f'{tf}_vol_velocity'] = (v2[vol_vel_col_v2].values
                                            if vol_vel_col_v2 in v2.columns else np.zeros(n))

    out['time_of_day'] = (v2['L0_time_of_day'].values
                              if 'L0_time_of_day' in v2.columns
                              else (out['timestamp'] % 86400) / 86400.0)

    for v2_col, out_col in [
        ('L3_4h_z_se_w', '4h_z_se'),
        ('L2_4h_price_velocity_w', '4h_velocity'),
        ('L1_4h_body', '4h_body'),
        ('L3_4h_swing_noise_w', '4h_swing_noise'),
    ]:
        out[out_col] = (v2[v2_col].values if v2_col in v2.columns else np.zeros(n))

    out['regime_2d'] = _lookup_regime_for_day(day, labels_csv)

    cols = ['timestamp']
    for tf in V1_TFS:
        for c in V1_CORE:
            cols.append(f'{tf}_{c}')
    for tf in V1_TFS:
        for c in V1_HELPERS:
            cols.append(f'{tf}_{c}')
    cols.append('time_of_day')
    for tf in V1_TFS:
        for c in V2_EXTENSION_HELPERS:
            cols.append(f'{tf}_{c}')
    for c in V2_4H_SIGNALS:
        cols.append(c)
    cols.append('regime_2d')
    out_compat = out[cols]
    if return_v2:
        return out_compat, v2.reset_index(drop=True)
    return out_compat
