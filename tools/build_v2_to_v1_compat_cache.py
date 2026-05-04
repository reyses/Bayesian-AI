"""
build_v2_to_v1_compat_cache.py — generate V1-shape (91D flat) per-day
parquets from V2 (184D layered) features.

This is the bridge that lets the legacy rule-based tier engine consume
V2 features WITHOUT engine modifications. It honors V2 spec Principle 7
(V2 cache stays clean — no V1 concepts stored in V2 layout) by computing
the V1 concepts ON THE FLY and writing them ONLY to this compatibility
cache, which is meant to be regenerated whenever V2 changes.

For each day:
  1. Load V2 layered features (L0 + L1/L2/L3 × 8 TFs)
  2. Load raw OHLCV at each V1 TF (15s, 1m, 5m, 15m, 1h, 1D) — 4 days
     of history for variance_ratio and vol_rel windows
  3. Construct 91D V1-shape DataFrame with V1 column ordering:
       Per TF (15s, 1m, 5m, 15m, 1h, 1D):
         z_se, dmi_diff, variance_ratio, velocity, acceleration,
         vol_rel, bar_range, hurst, reversion_prob, p_at_center,
         z_high, z_low
       Helpers per TF: dmi_gap, dir_vol, wick_ratio
       Global: time_of_day
       Total: 6×12 + 6×3 + 1 = 91 columns + timestamp
  4. Write DATA/ATLAS/FEATURES_5s_v2_as_v1/<day>.parquet

The output cache is consumed by training_v2/ (see training_v2/run.py
where FEATURES_DIR_5S points at this cache).

Usage:
  python tools/build_v2_to_v1_compat_cache.py --fresh
  python tools/build_v2_to_v1_compat_cache.py --start 2025-06-01 --end 2025-09-30

Per-day runtime: ~5-15 sec on RTX 3060 / 16GB RAM. Full 345 days: ~30-50 min.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_v2 import v1_compat
from tools.research.features_v2 import load_v2_features
from tools.v2_features_tf_sweep_eda import feature_column_for


# V1 91D layout
V1_TFS = ['15s', '1m', '5m', '15m', '1h', '1D']
V1_CORE = ['z_se', 'dmi_diff', 'variance_ratio', 'velocity', 'acceleration',
              'vol_rel', 'bar_range', 'hurst', 'reversion_prob', 'p_at_center',
              'z_high', 'z_low']
V1_HELPERS = ['dmi_gap', 'dir_vol', 'wick_ratio']

# V2-extension columns (NEW for training_v2 — directional wick info that V1
# never carried). These let KILL_SHOT/CASCADE classify direction by wick side
# (lower wick → support bounce → LONG; upper wick → ceiling rejection → SHORT)
# instead of relying on z_se sign.
V2_EXTENSION_HELPERS = ['upper_wick', 'lower_wick']

# V2 column lookup for each V1 concept that maps directly
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

# Per-feature unit conversion V2 -> V1.
# V1 stores bar_range in TICK units (= (high - low) / TICK).
# V2 stores bar_range in PRICE units. Convert by dividing by TICK = 0.25
# (i.e. multiplying by 4).
TICK = 0.25
V1_UNIT_CONVERSIONS = {
    'bar_range': 1.0 / TICK,  # V2 price units -> V1 ticks
    # velocity, acceleration, z_se: same formula family, different
    # implementations (V1 cuda OLS slope vs V2 numpy MA derivative).
    # No simple conversion — accept the SFE drift. Tier-engine thresholds
    # must be re-calibrated; see reports/findings/v1_v2_threshold_calibration/.
}

TF_PERIOD_S = {'15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1D': 86400}


def load_ohlcv_with_history(atlas_root: str, tf: str, day: str,
                                  history_days: int = 4) -> pd.DataFrame | None:
    """Load native-TF OHLCV for `day` plus N prior days of history.

    Needed for variance_ratio's 60-bar window and vol_rel's 30-bar window
    at higher TFs (1h needs 60 hours = 2.5 days of history).
    """
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


def build_one_day(atlas_root: str, day: str, output_dir: str,
                    overwrite: bool = False) -> bool:
    """Generate one day's V1-shape parquet from V2 cache + raw OHLCV.

    Returns True if written, False if skipped.
    """
    out_path = os.path.join(output_dir, f'{day}.parquet')
    if not overwrite and os.path.exists(out_path):
        return False

    # Load V2 features for this day's anchor timestamps
    # We need to know the day's ts range — read V1 cache OR derive from raw 5s
    # Use raw 5s as the anchor source since V1 cache may not exist yet
    raw_5s_path = os.path.join(atlas_root, '5s', f'{day}.parquet')
    if not os.path.exists(raw_5s_path):
        return False
    raw_5s = pd.read_parquet(raw_5s_path)
    if pd.api.types.is_datetime64_any_dtype(raw_5s['timestamp']):
        raw_5s = raw_5s.copy()
        raw_5s['timestamp'] = (raw_5s['timestamp'].astype('int64') // 10**9)
    ts = raw_5s['timestamp'].values.astype(np.int64)
    if len(ts) == 0:
        return False
    ts_min, ts_max = int(ts.min()), int(ts.max())

    v2 = load_v2_features(
        v2_dir=os.path.join(atlas_root, 'FEATURES_5s_v2'),
        atlas_root=atlas_root, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    # Filter V2 to exact ts range (load_v2_features may pull adjacent days)
    v2 = v2[v2['timestamp'].between(ts_min, ts_max)].reset_index(drop=True)
    if len(v2) == 0:
        return False

    n = len(v2)
    out = pd.DataFrame({'timestamp': v2['timestamp'].values})

    # Build per-TF concept arrays in V1 order
    for tf in V1_TFS:
        # Load native OHLCV for VR / vol_rel
        ohlcv = load_ohlcv_with_history(atlas_root, tf, day)
        if ohlcv is None:
            # Day's TF has no data — fill with neutral values
            for c in V1_CORE:
                out[f'{tf}_{c}'] = np.zeros(n) if c != 'variance_ratio' else np.ones(n)
            for c in V1_HELPERS:
                out[f'{tf}_{c}'] = np.zeros(n)
            continue

        # Direct V2 mappings (with unit conversion where needed)
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

        # DMI substitute: sign(velocity) * 5
        velocity = out[f'{tf}_velocity'].values
        out[f'{tf}_dmi_diff'] = np.where(velocity > 0, 5.0,
                                              np.where(velocity < 0, -5.0, 0.0))

        # variance_ratio + vol_rel via shim batch
        # Need to align ohlcv to v2 anchor timestamps using last-closed-bar logic
        # The shim's derive_v1_concepts_batch handles alignment internally.
        # But it requires v2_df to have specific columns; build a minimal v2 view.
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
            # Missing V2 columns — fill neutral
            out[f'{tf}_variance_ratio'] = np.ones(n)
            out[f'{tf}_vol_rel'] = np.ones(n)
            out[f'{tf}_p_at_center'] = np.full(n, 0.5)
            out[f'{tf}_dir_vol'] = np.zeros(n)
            out[f'{tf}_wick_ratio'] = np.zeros(n)

        # ── V2-extension: upper_wick + lower_wick (directional wick info) ──
        # Compute from raw OHLC at native TF cadence, then step-fill to anchor.
        ohlcv_ts = ohlcv['timestamp'].values.astype(np.int64)
        if pd.api.types.is_datetime64_any_dtype(ohlcv['timestamp']):
            ohlcv_ts = (ohlcv['timestamp'].astype('int64') // 10**9).values
        opens = ohlcv['open'].values.astype(np.float64)
        highs = ohlcv['high'].values.astype(np.float64)
        lows = ohlcv['low'].values.astype(np.float64)
        closes = ohlcv['close'].values.astype(np.float64)
        upper_native, lower_native = v1_compat.directional_wicks_batch(
            opens, highs, lows, closes)
        # Align to 5s anchor: last-closed-bar lookup
        anchor_ts = v2['timestamp'].values.astype(np.int64)
        last_closed = np.searchsorted(ohlcv_ts, anchor_ts - period, side='right') - 1
        valid = (last_closed >= 0) & (last_closed < len(ohlcv_ts))
        safe_idx = np.clip(last_closed, 0, len(ohlcv_ts) - 1)
        out[f'{tf}_upper_wick'] = np.where(valid, upper_native[safe_idx], 0.0)
        out[f'{tf}_lower_wick'] = np.where(valid, lower_native[safe_idx], 0.0)

        # dmi_gap = abs(dmi_diff)
        out[f'{tf}_dmi_gap'] = np.abs(out[f'{tf}_dmi_diff'].values)

    # Global
    if 'L0_time_of_day' in v2.columns:
        out['time_of_day'] = v2['L0_time_of_day'].values
    else:
        out['time_of_day'] = (out['timestamp'] % 86400) / 86400.0

    # Reorder columns to match V1 cache (91D), with V2-extension columns
    # appended after time_of_day:
    #   timestamp, V1_CORE×6TFs, V1_HELPERS×6TFs, time_of_day,
    #   V2_EXTENSION_HELPERS×6TFs (NEW)
    cols = ['timestamp']
    for tf in V1_TFS:
        for c in V1_CORE:
            cols.append(f'{tf}_{c}')
    for tf in V1_TFS:
        for c in V1_HELPERS:
            cols.append(f'{tf}_{c}')
    cols.append('time_of_day')
    # NEW: directional wicks (12 cols: 6 TFs × upper/lower)
    for tf in V1_TFS:
        for c in V2_EXTENSION_HELPERS:
            cols.append(f'{tf}_{c}')
    out = out[cols]

    os.makedirs(output_dir, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas-root', default='DATA/ATLAS')
    parser.add_argument('--output-dir', default='DATA/ATLAS/FEATURES_5s_v2_as_v1')
    parser.add_argument('--fresh', action='store_true',
                        help='Rebuild all days even if output exists')
    parser.add_argument('--start', help='YYYY-MM-DD start date (inclusive)')
    parser.add_argument('--end', help='YYYY-MM-DD end date (inclusive)')
    parser.add_argument('--days', nargs='+', help='specific days like 2025_06_02')
    args = parser.parse_args()

    raw_5s_files = sorted(glob.glob(os.path.join(args.atlas_root, '5s', '*.parquet')))
    all_days = [os.path.basename(p).replace('.parquet', '') for p in raw_5s_files]

    if args.days:
        days = args.days
    else:
        days = all_days
        if args.start:
            start_key = args.start.replace('-', '_')
            days = [d for d in days if d >= start_key]
        if args.end:
            end_key = args.end.replace('-', '_')
            days = [d for d in days if d <= end_key]

    print(f"V2 -> V1 compat cache build")
    print(f"  Output: {args.output_dir}")
    print(f"  Days to process: {len(days)}")
    print(f"  Fresh: {args.fresh}")

    n_built = 0
    n_skipped = 0
    n_failed = 0
    for day in tqdm(days, desc='days'):
        try:
            ok = build_one_day(args.atlas_root, day, args.output_dir,
                                  overwrite=args.fresh)
            if ok:
                n_built += 1
            else:
                n_skipped += 1
        except Exception as e:
            tqdm.write(f"  {day}: FAILED — {e}")
            n_failed += 1

    print(f"\n  Built: {n_built}  Skipped (existing): {n_skipped}  Failed: {n_failed}")
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
