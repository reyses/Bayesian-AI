"""
Research data loading and physics extraction utilities.

Provides functions for loading ATLAS parquet data, computing per-TF physics
via StatisticalFieldEngine, extracting 16D feature vectors, and building
stacked (12, 16) hypervolume matrices with oracle MFE/MAE labels.
"""

import sys, os, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.quantum_field_engine import StatisticalFieldEngine
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
    results = engine.batch_compute_states(df, use_cuda=False)

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
    """Build (12, 16) hypervolume matrices from stacked TF physics.

    For each bar in the base TF's analysis window:
      - Find the most recent state at each of the 12 TFs
      - Stack into (12, 16) matrix
      - Compute oracle MFE/MAE from the base TF's future bars

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
        # Use first half for warmup, rest for analysis
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

    # Pre-sort timestamps for each TF (for binary search alignment)
    tf_sorted_ts = {}
    for tf in TF_HIERARCHY:
        if tf in all_tf_states and all_tf_states[tf]:
            tf_sorted_ts[tf] = sorted(all_tf_states[tf].keys())

    # Build timestamp->index mapping for base_df (for MFE/MAE computation)
    if 'timestamp' in base_df.columns:
        ts_col = base_df['timestamp'].values
    else:
        ts_col = np.arange(len(base_df))

    ts_to_idx = {}
    for i, t in enumerate(ts_col):
        ts_to_idx[int(t)] = i

    # Oracle lookahead for base TF
    lookahead = ORACLE_LOOKAHEAD_BARS.get(base_tf, 16)

    matrices = []
    mfes = []
    maes = []
    meta = []
    _n_long = 0
    _n_short = 0
    _n_skip = 0

    _pbar = tqdm(analysis_ts, desc="Hypervolumes", unit="bar",
                 ascii=True, dynamic_ncols=True, mininterval=0.3,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}, {rate_fmt}] "
                            "{postfix}")
    for t in _pbar:
        # --- Stack 16D across all 12 TFs ---
        mat = np.zeros((12, 16))
        has_data = 0

        base_secs = TF_SECONDS.get(base_tf, 900)

        for depth_idx, tf in enumerate(TF_HIERARCHY):
            if tf not in tf_sorted_ts:
                continue  # TF not available, leave as zeros

            tf_ts_list = tf_sorted_ts[tf]
            tf_secs = TF_SECONDS.get(tf, 60)

            # For TFs longer than base: bar containing t is incomplete,
            # so use N-1 (last FULLY COMPLETED bar) to avoid look-ahead.
            # For TFs <= base: bar completes within base bar window, no leak.
            if tf_secs > base_secs:
                idx = np.searchsorted(tf_ts_list, t, side='right') - 2
            else:
                idx = np.searchsorted(tf_ts_list, t, side='right') - 1
            if idx < 0:
                continue  # No completed data before this timestamp

            nearest_ts = tf_ts_list[idx]
            state = all_tf_states[tf][nearest_ts]
            mat[depth_idx, :] = extract_16d(state, tf)
            has_data += 1

        if has_data < 3:
            _n_skip += 1
            continue  # Need at least 3 TFs with data

        # --- Compute oracle MFE/MAE from base TF future bars ---
        if t not in ts_to_idx:
            _n_skip += 1
            continue

        bar_idx = ts_to_idx[t]
        if bar_idx + lookahead >= len(base_df):
            _n_skip += 1
            continue  # Not enough future data

        entry_price = float(base_df.iloc[bar_idx]['close'])
        future = base_df.iloc[bar_idx + 1 : bar_idx + 1 + lookahead]

        if future.empty:
            _n_skip += 1
            continue

        max_up = float(future['high'].max() - entry_price)
        max_down = float(entry_price - future['low'].min())

        if max_up == 0 and max_down == 0:
            _n_skip += 1
            continue

        # Direction from z-score sign at base TF
        base_state = base_states[t]
        z = base_state.z_score
        dmi_diff = (base_state.dmi_plus - base_state.dmi_minus) \
            if hasattr(base_state, 'dmi_plus') else 0.0

        # MFE/MAE assignment based on direction
        # z < 0 -> LONG setup (MFE = up, MAE = down)
        # z > 0 -> SHORT setup (MFE = down, MAE = up)
        if z < 0:  # LONG
            mfe_val = max_up
            mae_val = max_down
            _n_long += 1
        else:      # SHORT
            mfe_val = max_down
            mae_val = max_up
            _n_short += 1

        matrices.append(mat)
        mfes.append(mfe_val)
        maes.append(mae_val)

        # ADX quartile for segmentation (proxy for template_id)
        adx = base_state.adx_strength if hasattr(base_state, 'adx_strength') else 0.0
        adx_bin = int(min(adx // 25, 3))  # 0-3 quartiles

        meta.append({
            'tid': f'adx_q{adx_bin}',
            'idx': len(matrices) - 1,
            'depth': 11,  # always full depth in standalone
            'ts': t,
            'dmi_diff': dmi_diff,
            'z_score': z,
        })

        # Update live stats every 50 bars to avoid I/O overhead
        if len(matrices) % 50 == 0 or len(matrices) == 1:
            _avg_mfe = np.mean(mfes) if mfes else 0
            _pbar.set_postfix_str(
                f"ok={len(matrices)} skip={_n_skip} "
                f"L={_n_long} S={_n_short} "
                f"avgMFE={_avg_mfe:.1f}",
                refresh=True
            )

    _pbar.close()
    print(f"  Built {len(matrices)} hypervolume matrices with oracle labels")
    return matrices, np.array(mfes), np.array(maes), meta
