"""
v2 feature loading helpers for tools/standalone_research.py.

Reads precomputed feature parquets from DATA/ATLAS/FEATURES_5s_v2/, joining
across the 25 layer-family directories (L0, L1_<TF>, L2_<TF>, L3_<TF> for
8 TFs). Falls back to live compute via core_v2.StatisticalFieldEngine when
a day is not yet built.

Schema reference: docs/JULES_standalone_research_v2.md and
core_v2/statistical_field_engine.py.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Constants (also surfaced via tools/research/data.py) ──────────────────

from core_v2.features import (
    TF_HIERARCHY_V2,
    FEATURE_NAMES_V2,
    N_TFS_V2,
    N_FEATURES_PER_TF_V2,
    N_FLAT_FEATURES_V2,
    assemble_v2_grid
)

TF_LABELS_V2 = ['d0_1D', 'd1_4h', 'd2_1h', 'd3_15m', 'd4_5m', 'd5_1m', 'd6_15s', 'd7_5s']

# Per-TF L2/L3 window suffix used in raw column names. Matches core_v2 N_BASE.
N_BASE_V2 = {
    '5s':  9,  '15s': 12, '1m':  15,
    '5m':  9,  '15m': 12, '1h':  12,
    '4h':  18, '1D':  5,
}

# Hurst window multiplier (per core_v2.N_HURST_MULT)
N_HURST_MULT = 8


# ── Detection ────────────────────────────────────────────────────────────

def detect_v2_cache(path: str) -> bool:
    """True iff `path` is a v2 features directory.

    Heuristic: directory exists and contains both L0/ and at least one L1_<TF>/.
    """
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isdir(os.path.join(path, 'L0')):
        return False
    for tf in TF_HIERARCHY_V2:
        if os.path.isdir(os.path.join(path, f'L1_{tf}')):
            return True
    return False


# ── Per-day load (precomputed primary path) ──────────────────────────────

def _strip_window_suffix(col: str, tf: str, window: int) -> str:
    """Strip _<W> suffix from column names so they line up across TFs.

    Examples:
        'L2_15m_price_velocity_12'      -> 'L2_15m_price_velocity_w'
        'L3_5s_hurst_72'                -> 'L3_5s_hurst_w'   (hurst uses W*8)
        'L1_1m_price_velocity_1b'       -> unchanged (L1 uses _1b literal)
    """
    # L1 columns end in _1b or are bare (bar_range, body) — never have window suffix
    if col.startswith('L1_'):
        return col
    if col == 'timestamp':
        return col
    # L2/L3: strip trailing _<int>
    parts = col.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f'{parts[0]}_w'
    return col


def load_v2_features_for_day(v2_dir: str, day_str: str):
    """Load and inner-join all 25 layer-dir parquets for one day.

    Args:
        v2_dir: e.g., 'DATA/ATLAS/FEATURES_5s_v2'
        day_str: 'YYYY_MM_DD'

    Returns:
        DataFrame at 5s cadence with columns: timestamp, L0_time_of_day,
        and {L1,L2,L3}_{tf}_<feature_w_or_1b> for 8 TFs (185 cols total).
        Returns None if any required dir/file is missing.
    """
    layer_dirs = ['L0'] + [f'{layer}_{tf}' for layer in ('L1', 'L2', 'L3') for tf in TF_HIERARCHY_V2]
    out = None
    for d in layer_dirs:
        f = os.path.join(v2_dir, d, f'{day_str}.parquet')
        if not os.path.exists(f):
            return None  # Trigger fallback for this day
        df = pd.read_parquet(f)
        if 'timestamp' not in df.columns:
            return None

        # Strip window suffixes from L2/L3 cols so cross-TF names align
        if d != 'L0':
            tf = d.split('_', 1)[1]
            window = N_BASE_V2.get(tf, 0)
            df = df.rename(columns={c: _strip_window_suffix(c, tf, window) for c in df.columns})

        if out is None:
            out = df
        else:
            out = out.merge(df, on='timestamp', how='inner')
    return out


# ── Per-day live compute (fallback) ──────────────────────────────────────

def compute_v2_features_live_for_day(atlas_root: str, day_str: str):
    """Fallback when precomputed parquet is missing.

    Loads raw OHLC for the day from {atlas_root}/{tf}/{day_str}.parquet for
    each of 8 TFs, runs core_v2 SFE compute_L0/L1/L2/L3, joins on timestamp.

    Returns DataFrame at 5s cadence matching load_v2_features_for_day schema.
    """
    from core_v2.statistical_field_engine import StatisticalFieldEngine

    sfe = StatisticalFieldEngine()

    # Base = 5s
    base_path = os.path.join(atlas_root, '5s', f'{day_str}.parquet')
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Cannot live-compute v2 features for {day_str}: "
            f"base 5s OHLC missing at {base_path}"
        )
    base_5s = pd.read_parquet(base_path).sort_values('timestamp').reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(base_5s['timestamp']):
        base_5s['timestamp'] = base_5s['timestamp'].astype('int64') // 10**9

    # L0 from base 5s
    out = sfe.compute_L0(base_5s).copy()
    out.insert(0, 'timestamp', base_5s['timestamp'].values)

    # For each TF: load OHLC, compute L1/L2/L3, reindex onto 5s base ts
    for tf in TF_HIERARCHY_V2:
        tf_path = os.path.join(atlas_root, tf, f'{day_str}.parquet')
        if not os.path.exists(tf_path):
            raise FileNotFoundError(
                f"Cannot live-compute v2 features for {day_str}/{tf}: "
                f"OHLC missing at {tf_path}"
            )
        tf_df = pd.read_parquet(tf_path).sort_values('timestamp').reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(tf_df['timestamp']):
            tf_df['timestamp'] = tf_df['timestamp'].astype('int64') // 10**9

        l1 = sfe.compute_L1(tf_df, tf=tf)
        l2 = sfe.compute_L2(tf_df, tf=tf)
        l3 = sfe.compute_L3(tf_df, tf=tf)
        tf_feats = pd.concat([l1, l2, l3], axis=1)
        tf_feats.insert(0, 'timestamp', tf_df['timestamp'].values)

        # Strip window suffixes
        window = N_BASE_V2.get(tf, 0)
        tf_feats = tf_feats.rename(
            columns={c: _strip_window_suffix(c, tf, window) for c in tf_feats.columns}
        )

        # Reindex onto base 5s timestamps via merge_asof (forward-fill from TF bar)
        tf_feats = tf_feats.sort_values('timestamp').reset_index(drop=True)
        out = pd.merge_asof(
            out.sort_values('timestamp'),
            tf_feats,
            on='timestamp',
            direction='backward',
        )

    return out


# ── Top-level loader ─────────────────────────────────────────────────────

def _ts_to_day_str(ts: int) -> str:
    """Convert unix seconds to 'YYYY_MM_DD' string (UTC)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y_%m_%d')


def list_available_days(v2_dir: str) -> list[str]:
    """List YYYY_MM_DD strings for which the L0 parquet exists."""
    files = sorted(glob.glob(os.path.join(v2_dir, 'L0', '*.parquet')))
    return [os.path.basename(f).replace('.parquet', '') for f in files]


def load_v2_features(v2_dir: str, atlas_root: str | None,
                     day_strs: list[str] | None = None,
                     ts_range: tuple[int, int] | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Load v2 features across multiple days.

    Args:
        v2_dir: path to FEATURES_5s_v2/ directory
        atlas_root: ATLAS root for fallback OHLC loading (e.g., 'DATA/ATLAS').
            Pass None to disable fallback (strict precomputed-only).
        day_strs: optional explicit list of YYYY_MM_DD strings. If None, use
            all days available in v2_dir.
        ts_range: optional (ts_min, ts_max) unix-seconds tuple to filter
            day_strs. Days whose midnight UTC is outside the range are skipped.
        verbose: print per-day status.

    Returns:
        DataFrame at 5s cadence, sorted by timestamp.
    """
    if day_strs is None:
        day_strs = list_available_days(v2_dir)
        if not day_strs and atlas_root is None:
            raise FileNotFoundError(f"No precomputed days in {v2_dir} and atlas_root=None")

    if ts_range is not None:
        ts_min, ts_max = ts_range
        from datetime import datetime, timezone
        def _day_in_range(d):
            dt = datetime.strptime(d, '%Y_%m_%d').replace(tzinfo=timezone.utc)
            day_start = int(dt.timestamp())
            return day_start <= ts_max and (day_start + 86400) >= ts_min
        day_strs = [d for d in day_strs if _day_in_range(d)]

    if not day_strs:
        raise ValueError(f"No days to load (v2_dir={v2_dir}, ts_range={ts_range})")

    frames = []
    n_precomp = 0
    n_live = 0
    n_skip = 0
    if verbose:
        print(f"  Loading v2 features for {len(day_strs)} days from {v2_dir}")
    from tqdm import tqdm
    for day in tqdm(day_strs, desc="v2 days", unit="day", ascii=True, dynamic_ncols=True):
        df = load_v2_features_for_day(v2_dir, day)
        if df is not None:
            frames.append(df)
            n_precomp += 1
            continue
        if atlas_root is None:
            n_skip += 1
            continue
        try:
            df = compute_v2_features_live_for_day(atlas_root, day)
            frames.append(df)
            n_live += 1
        except FileNotFoundError as e:
            if verbose:
                print(f"  [skip] {day}: {e}")
            n_skip += 1

    if verbose:
        print(f"  v2 load summary: precomputed={n_precomp}, live-computed={n_live}, skipped={n_skip}")

    if not frames:
        raise RuntimeError("No v2 features loaded for any requested day")

    out = pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return out


# ── Reindex onto base TF timestamps ─────────────────────────────────────

def align_v2_to_base_tf(features_5s: pd.DataFrame, base_ts: np.ndarray) -> pd.DataFrame:
    """For each base_ts, return the most recent feature row at or before that timestamp.

    Uses searchsorted(side='right') - 1 against the sorted 5s feature timestamps.
    Any base_ts before the first feature ts gets a row of NaN (and the caller
    is expected to filter those out via `dropna()`).

    Args:
        features_5s: DataFrame from load_v2_features(), sorted by timestamp.
        base_ts: array of base TF unix-second timestamps.

    Returns:
        DataFrame indexed positionally with one row per base_ts.
        Columns: same as features_5s minus 'timestamp'.
    """
    feat_ts = features_5s['timestamp'].values.astype(np.int64)
    base_ts_arr = np.asarray(base_ts, dtype=np.int64)
    pos = np.searchsorted(feat_ts, base_ts_arr, side='right') - 1
    valid = pos >= 0
    out_cols = [c for c in features_5s.columns if c != 'timestamp']
    out = pd.DataFrame(np.nan, index=np.arange(len(base_ts_arr)), columns=out_cols)
    safe_pos = np.where(valid, pos, 0)
    out.iloc[valid] = features_5s.iloc[safe_pos[valid]][out_cols].values
    return out


# ── Reshape to per-TF stack ──────────────────────────────────────────────

def extract_per_tf_block(row: pd.Series, tf: str) -> np.ndarray:
    """Pull the 23 features for one TF out of a joined-feature row.

    Returns a (23,) numpy array in FEATURE_NAMES_V2 order.
    Missing columns get 0.0 (NaN -> 0 to keep matrix usable).
    """
    out = np.zeros(N_FEATURES_PER_TF_V2, dtype=np.float64)
    for i, fname in enumerate(FEATURE_NAMES_V2):
        # L1 features have _1b suffix; L2 have _w suffix; L3 are bare.
        if fname.endswith('_1b'):
            col = f'L1_{tf}_{fname}'
        elif fname == 'bar_range' or fname == 'body':
            col = f'L1_{tf}_{fname}'
        elif fname.endswith('_w'):
            col = f'L2_{tf}_{fname}'
        else:
            col = f'L3_{tf}_{fname}_w'  # L3 also got window-suffix-stripped to _w
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                out[i] = float(v)
    return out


def reshape_v2_to_stack(features_aligned: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reshape the aligned v2 features into a stacked-TF matrix per row.

    Args:
        features_aligned: DataFrame from align_v2_to_base_tf(). One row per
            base TF bar. Columns: L0_time_of_day + L{1,2,3}_{tf}_<feat_w_or_1b>.

    Returns:
        stack: (N, N_TFS_V2, N_FEATURES_PER_TF_V2) array
        l0_global: (N,) array of L0_time_of_day values
    """
    n = len(features_aligned)
    
    # We must construct a flat matrix using core_v2.features.FEATURE_NAMES
    # order so we can pass it to assemble_v2_grid, OR we can just implement the same mapping here.
    # Actually, the user specifically wants single source of truth.
    # Let's import the full canonical list from core_v2.features
    from core_v2.features import FEATURE_NAMES, N_FEATURES
    flat_matrix = np.zeros((n, N_FEATURES), dtype=np.float32)
    
    col_map = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for col in features_aligned.columns:
        if col in col_map:
            vals = features_aligned[col].values
            flat_matrix[:, col_map[col]] = np.nan_to_num(vals, nan=0.0)
            
    stack = assemble_v2_grid(flat_matrix)
    
    if 'L0_time_of_day' in features_aligned.columns:
        l0_global = np.nan_to_num(features_aligned['L0_time_of_day'].values, nan=0.0)
    else:
        l0_global = np.zeros(n, dtype=np.float64)

    return stack, l0_global


# ── Smoke test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    v2_dir = 'DATA/ATLAS/FEATURES_5s_v2'
    print(f"detect_v2_cache('{v2_dir}'): {detect_v2_cache(v2_dir)}")
    print(f"detect_v2_cache('foo.npz'): {detect_v2_cache('foo.npz')}")

    days = list_available_days(v2_dir)
    print(f"Available days: {len(days)}, range: {days[0]} -> {days[-1]}")

    # Load 1 day
    print("\n--- Single day load ---")
    df = load_v2_features_for_day(v2_dir, '2025_01_07')
    print(f"  Shape: {df.shape}")
    print(f"  First 5 cols: {list(df.columns[:5])}")
    print(f"  Last 5 cols: {list(df.columns[-5:])}")

    # Reindex onto 1m timestamps
    print("\n--- Reindex onto 1m base ---")
    base_1m = pd.read_parquet('DATA/ATLAS/1m/2025_01_07.parquet')
    base_ts = base_1m['timestamp'].values.astype(np.int64)
    if base_1m['timestamp'].dtype == 'datetime64[ns]':
        base_ts = base_ts // 10**9
    aligned = align_v2_to_base_tf(df, base_ts)
    print(f"  Aligned shape: {aligned.shape}")

    # Reshape to stack
    print("\n--- Reshape to (N, 8, 23) ---")
    stack, l0 = reshape_v2_to_stack(aligned)
    print(f"  Stack shape: {stack.shape}")
    print(f"  L0 shape: {l0.shape}")
    print(f"  Stack[0, :, 0] (price_velocity_1b for each TF): {stack[0, :, 0]}")
    print(f"  Sample of nonzero stack values: {stack[stack != 0][:5]}")
