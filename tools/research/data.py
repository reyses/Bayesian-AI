"""
Research data loading and physics extraction utilities.

Provides functions for loading ATLAS parquet data, computing per-TF physics
via StatisticalFieldEngine, extracting 16D feature vectors, and building
stacked (12, 16) multi-TF state matrices with oracle MFE/MAE labels.
"""

import sys, os, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core_v2.statistical_field_engine import StatisticalFieldEngine
from config.oracle_config import ORACLE_LOOKAHEAD_BARS


# -- TF hierarchy: 12 levels from macro (1W) to micro (15s) -------------------
# Drops 1s and 5s (too noisy, too much data for screening)
TF_HIERARCHY = ['1W', '1D', '4h', '1h', '30m', '15m', '5m', '3m', '2m', '1m', '30s', '15s']

# Seconds per bar for each TF (used for tf_scale feature = log2(seconds))
TF_SECONDS = {
    '1W': 604800, '1D': 86400, '4h': 14400, '1h': 3600,
    '30m': 1800, '15m': 900, '5m': 300, '3m': 180, '2m': 120,
    '1m': 60, '30s': 30, '15s': 15,
}

# TF labels for column naming (matches standalone_research.py convention)
TF_LABELS = [
    'd0_1W', 'd1_1D', 'd2_4h', 'd3_1h', 'd4_30m', 'd5_15m',
    'd6_5m', 'd7_3m', 'd8_2m', 'd9_1m', 'd10_30s', 'd11_15s'
]

# 16D feature names
FEATURE_NAMES = [
    'z_score', 'log1p_vol', 'log1p_mom', 'entropy_normalized', 'tf_scale', 'depth',
    'parent_ctx', 'self_adx', 'self_hurst', 'self_dmi_diff',
    'parent_z', 'parent_dmi_diff', 'root_is_roche', 'tf_alignment',
    'self_pid', 'osc_coh'
]


def load_atlas_tf(data_dir, tf_name, months=None):
    """Load ATLAS parquet files for a single timeframe.

    Args:
        data_dir: Root ATLAS directory (e.g., 'DATA/ATLAS' or 'DATA/ATLAS_1DAY')
        tf_name: Timeframe string (e.g., '15m', '1h')
        months: Optional list of month strings (e.g., ['2025_01']). If None, load all.

    Returns:
        pd.DataFrame with [timestamp, open, high, low, close, volume], sorted by timestamp.
        Returns empty DataFrame if TF directory doesn't exist.
    """
    tf_dir = os.path.join(data_dir, tf_name)
    if not os.path.isdir(tf_dir):
        return pd.DataFrame()

    if months:
        files = [os.path.join(tf_dir, f'{m}.parquet') for m in months]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))

    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure timestamp is numeric (seconds)
    if 'timestamp' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def compute_tf_physics(tf_name, df):
    """Run StatisticalFieldEngine on a single TF's data.

    Returns:
        dict mapping timestamp (int) -> MarketState
    """
    if df.empty or len(df) < 21:
        return {}

    engine = StatisticalFieldEngine(regression_period=21)
    results = engine.batch_compute_states(df, use_cuda=True)

    states = {}
    for r in results:
        state = r['state']
        ts = int(state.timestamp) if hasattr(state, 'timestamp') and state.timestamp else 0
        if ts > 0:
            states[ts] = state

    return states


def extract_16d(state, tf_name):
    """Build 16D feature vector from a MarketState.

    Matches the feature layout in fractal_clustering.py:extract_features()
    but without ancestry (features 5-6, 10-13 set to 0).
    """
    z = state.z_score
    v = abs(state.velocity) if hasattr(state, 'velocity') else 0.0
    m = abs(state.momentum_strength) if hasattr(state, 'momentum_strength') else 0.0
    c = state.entropy_normalized if hasattr(state, 'entropy_normalized') else 0.0

    tf_scale = math.log2(max(TF_SECONDS.get(tf_name, 60), 1))

    adx = (state.adx_strength * 0.01) if hasattr(state, 'adx_strength') else 0.0
    hurst = state.hurst_exponent if hasattr(state, 'hurst_exponent') else 0.5
    dmi_diff = ((state.dmi_plus - state.dmi_minus) * 0.01) if hasattr(state, 'dmi_plus') else 0.0
    pid = state.term_pid if hasattr(state, 'term_pid') else 0.0
    osc = state.oscillation_entropy_normalized if hasattr(state, 'oscillation_entropy_normalized') else 0.0

    return [
        z,                          # [0]  signed z-score
        math.log1p(v),              # [1]  log1p(velocity)
        math.log1p(m),              # [2]  log1p(momentum)
        c,                          # [3]  coherence
        tf_scale,                   # [4]  tf_scale = log2(seconds)
        0.0,                        # [5]  depth (no ancestry)
        0.0,                        # [6]  parent_ctx (no ancestry)
        adx,                        # [7]  self_adx / 100
        hurst,                      # [8]  self_hurst
        dmi_diff,                   # [9]  self_dmi_diff / 100
        0.0,                        # [10] parent_z (no ancestry)
        0.0,                        # [11] parent_dmi_diff (no ancestry)
        0.0,                        # [12] root_is_roche (no ancestry)
        0.0,                        # [13] tf_alignment (no ancestry)
        pid,                        # [14] self_pid
        osc,                        # [15] osc_coh
    ]


def build_stacked_matrices(all_tf_states, base_tf, base_df,
                           context_days=21, analysis_days=7):
    """Build (12, 16) multi-TF state matrices from stacked TF physics.

    For each bar in the base TF's analysis window:
      - Find the most recent state at each of the 12 TFs
      - Stack into (12, 16) matrix
      - Compute oracle MFE/MAE from the base TF's future bars

    Uses vectorized numpy operations for the TF alignment step (~4-6x faster
    than per-bar Python loop on multi-core CPUs).

    Args:
        all_tf_states: dict {tf_name: {timestamp: MarketState}}
        base_tf: Base timeframe string (e.g., '15m')
        base_df: DataFrame for the base TF (for MFE/MAE lookahead)
        context_days: Days of warmup before analysis window
        analysis_days: Days of analysis window (0 = use all remaining)

    Returns:
        matrices: list of (12, 16) numpy arrays
        mfes: numpy array of MFE values
        maes: numpy array of MAE values
        meta: list of dicts with timestamp, dmi_diff, etc.
    """
    if base_tf not in all_tf_states or not all_tf_states[base_tf]:
        print("ERROR: No states computed for base TF")
        return [], np.array([]), np.array([]), []

    # Get sorted timestamps for the base TF
    base_states = all_tf_states[base_tf]
    base_timestamps = sorted(base_states.keys())

    if not base_timestamps:
        return [], np.array([]), np.array([]), []

    t_min = base_timestamps[0]
    t_max = base_timestamps[-1]

    # Define analysis window
    from datetime import datetime, timezone
    data_span_days = (t_max - t_min) / 86400

    # Auto-adjust if data is shorter than context window
    if context_days > 0 and data_span_days < context_days + 1:
        old_ctx = context_days
        context_days = max(0, int(data_span_days * 0.3))
        print(f"  Auto-adjusted context: {old_ctx}d -> {context_days}d "
              f"(data span is only {data_span_days:.1f}d)")

    t_warmup_end = t_min + context_days * 86400
    if analysis_days > 0:
        t_analysis_end = t_warmup_end + analysis_days * 86400
    else:
        t_analysis_end = t_max + 1

    print(f"  Data range:     {datetime.fromtimestamp(t_min, tz=timezone.utc):%Y-%m-%d %H:%M} to "
          f"{datetime.fromtimestamp(t_max, tz=timezone.utc):%Y-%m-%d %H:%M}")
    print(f"  Data span:      {data_span_days:.1f} days")
    print(f"  Warmup:         {context_days}d -> analysis starts "
          f"{datetime.fromtimestamp(t_warmup_end, tz=timezone.utc):%Y-%m-%d %H:%M}")
    print(f"  Analysis end:   {datetime.fromtimestamp(min(t_analysis_end, t_max), tz=timezone.utc):%Y-%m-%d %H:%M}")

    # Filter base timestamps to analysis window
    analysis_ts = [t for t in base_timestamps if t_warmup_end <= t < t_analysis_end]
    print(f"  Analysis bars:  {len(analysis_ts)} ({base_tf} bars in window)")

    if not analysis_ts:
        print("  WARNING: No bars in analysis window. Try reducing context_days.")
        return [], np.array([]), np.array([]), []

    # ---- Pre-extract: MarketState -> numpy arrays (eliminates object access in hot loop) ----
    print("  Pre-extracting TF features into numpy arrays...")
    base_secs = TF_SECONDS.get(base_tf, 900)
    tf_ts_arrays = {}   # tf -> np.array of sorted timestamps
    tf_feat_arrays = {}  # tf -> np.array of shape (N, 16) aligned to tf_ts_arrays

    for tf in TF_HIERARCHY:
        if tf not in all_tf_states or not all_tf_states[tf]:
            continue
        sorted_ts = sorted(all_tf_states[tf].keys())
        feats = np.array([extract_16d(all_tf_states[tf][t], tf) for t in sorted_ts])
        tf_ts_arrays[tf] = np.array(sorted_ts, dtype=np.int64)
        tf_feat_arrays[tf] = feats

    # Pre-extract base TF z-scores and dmi_diff for all analysis bars
    base_z = np.array([base_states[t].z_score for t in analysis_ts])
    base_dmi = np.array([
        (base_states[t].dmi_plus - base_states[t].dmi_minus)
        if hasattr(base_states[t], 'dmi_plus') else 0.0
        for t in analysis_ts
    ])
    base_adx = np.array([
        base_states[t].adx_strength if hasattr(base_states[t], 'adx_strength') else 0.0
        for t in analysis_ts
    ])

    # ---- Vectorized TF alignment: for each TF, compute aligned indices for ALL bars at once ----
    print("  Vectorized TF alignment...")
    analysis_ts_arr = np.array(analysis_ts, dtype=np.int64)
    n_bars = len(analysis_ts_arr)

    # Pre-compute aligned feature index for each (bar, tf) pair
    # Result: all_mats[bar_idx, depth_idx, :] = 16D features
    all_mats = np.zeros((n_bars, 12, 16), dtype=np.float64)
    has_data_counts = np.zeros(n_bars, dtype=np.int32)

    for depth_idx, tf in enumerate(TF_HIERARCHY):
        if tf not in tf_ts_arrays:
            continue

        tf_ts = tf_ts_arrays[tf]
        tf_feats = tf_feat_arrays[tf]
        tf_secs = TF_SECONDS.get(tf, 60)

        # Vectorized searchsorted: find aligned index for ALL bars at once
        raw_idx = np.searchsorted(tf_ts, analysis_ts_arr, side='right')
        if tf_secs > base_secs:
            raw_idx -= 2  # N-1 for slow TFs
        else:
            raw_idx -= 1  # current completed bar for fast TFs

        # Mask valid indices
        valid = raw_idx >= 0
        # Clip to valid range for safe indexing (invalid ones masked out below)
        clipped_idx = np.clip(raw_idx, 0, len(tf_ts) - 1)

        # Bulk assign features
        all_mats[valid, depth_idx, :] = tf_feats[clipped_idx[valid]]
        has_data_counts[valid] += 1

    # ---- Vectorized oracle MFE/MAE computation ----
    print("  Computing oracle MFE/MAE...")
    if 'timestamp' in base_df.columns:
        ts_col = base_df['timestamp'].values
    else:
        ts_col = np.arange(len(base_df))

    ts_to_idx = {}
    for i, t in enumerate(ts_col):
        ts_to_idx[int(t)] = i

    lookahead = ORACLE_LOOKAHEAD_BARS.get(base_tf, 16)
    closes = base_df['close'].values.astype(np.float64)
    highs = base_df['high'].values.astype(np.float64)
    lows = base_df['low'].values.astype(np.float64)
    n_df = len(base_df)

    # Pre-compute bar indices for analysis timestamps
    bar_indices = np.array([ts_to_idx.get(int(t), -1) for t in analysis_ts], dtype=np.int64)

    # Build result arrays
    matrices = []
    mfes = []
    maes = []
    meta = []
    _n_long = 0
    _n_short = 0
    _n_skip = 0

    _pbar = tqdm(range(n_bars), desc="Building matrices", unit="bar",
                 ascii=True, dynamic_ncols=True, mininterval=0.3,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}, {rate_fmt}] "
                            "{postfix}")
    for i in _pbar:
        if has_data_counts[i] < 3:
            _n_skip += 1
            continue

        bar_idx = bar_indices[i]
        if bar_idx < 0 or bar_idx + lookahead >= n_df:
            _n_skip += 1
            continue

        entry_price = closes[bar_idx]
        future_hi = highs[bar_idx + 1: bar_idx + 1 + lookahead]
        future_lo = lows[bar_idx + 1: bar_idx + 1 + lookahead]

        if len(future_hi) == 0:
            _n_skip += 1
            continue

        max_up = float(future_hi.max() - entry_price)
        max_down = float(entry_price - future_lo.min())

        if max_up == 0 and max_down == 0:
            _n_skip += 1
            continue

        z = base_z[i]
        if z < 0:  # LONG
            mfe_val, mae_val = max_up, max_down
            _n_long += 1
        else:      # SHORT
            mfe_val, mae_val = max_down, max_up
            _n_short += 1

        matrices.append(all_mats[i])
        mfes.append(mfe_val)
        maes.append(mae_val)

        adx_bin = int(min(base_adx[i] // 25, 3))
        meta.append({
            'tid': f'adx_q{adx_bin}',
            'idx': len(matrices) - 1,
            'depth': 11,
            'ts': analysis_ts[i],
            'dmi_diff': float(base_dmi[i]),
            'z_score': float(z),
        })

        if len(matrices) % 200 == 0 or len(matrices) == 1:
            _avg_mfe = np.mean(mfes) if mfes else 0
            _pbar.set_postfix_str(
                f"ok={len(matrices)} skip={_n_skip} "
                f"L={_n_long} S={_n_short} "
                f"avgMFE={_avg_mfe:.1f}",
                refresh=True
            )

    _pbar.close()
    print(f"  Built {len(matrices)} multi-TF state matrices with oracle labels")
    return matrices, np.array(mfes), np.array(maes), meta
