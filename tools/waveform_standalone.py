"""
Standalone Waveform Screening Tool
====================================
Price-first I-MR analysis from raw ATLAS parquet data.
No templates.pkl, no checkpoints, no training dependency.

Default mode (price I-MR):
  1. Load base TF (15m) close prices
  2. I-MR chart: I = close price, MR = |bar-to-bar change|
  3. Detect regimes from MR UCL breaks (natural price segments)
  4. Oracle MFE/MAE with regime-based direction
  5. Charts: price I-MR + regime summary

Full mode (--full):
  6-20. Load all 12 TFs, compute 16D physics, build hypervolumes,
        fractal screening with regime-based segmentation

Usage:
    python tools/waveform_standalone.py --data DATA/ATLAS_1WEEK --base-tf 15m
    python tools/waveform_standalone.py --data DATA/ATLAS_1WEEK --base-tf 15m --full
    python tools/waveform_standalone.py --data DATA/ATLAS --context-days 30 --analysis-days 7

Output: tools/standalone_report.txt + tools/plots/standalone/
"""

import sys, os, io, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.quantum_field_engine import QuantumFieldEngine
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

# TF labels for column naming (matches waveform_screening.py convention)
TF_LABELS = [
    'd0_1W', 'd1_1D', 'd2_4h', 'd3_1h', 'd4_30m', 'd5_15m',
    'd6_5m', 'd7_3m', 'd8_2m', 'd9_1m', 'd10_30s', 'd11_15s'
]

# 16D feature names
FEATURE_NAMES = [
    'z_score', 'log1p_vol', 'log1p_mom', 'coherence', 'tf_scale', 'depth',
    'parent_ctx', 'self_adx', 'self_hurst', 'self_dmi_diff',
    'parent_z', 'parent_dmi_diff', 'root_is_roche', 'tf_alignment',
    'self_pid', 'osc_coh'
]


class _Tee:
    """Write to both stdout and a StringIO buffer simultaneously."""
    def __init__(self, stream, buffer):
        self._stream = stream
        self._buffer = buffer

    def write(self, data):
        self._stream.write(data)
        self._buffer.write(data)

    def flush(self):
        self._stream.flush()


# =============================================================================
#  DATA PIPELINE: Raw ATLAS → Physics → 16D → Hypervolume Matrices
# =============================================================================

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
    """Run QuantumFieldEngine on a single TF's data.

    Returns:
        dict mapping timestamp (int) → ThreeBodyQuantumState
    """
    if df.empty or len(df) < 21:
        return {}

    engine = QuantumFieldEngine(regression_period=21)
    results = engine.batch_compute_states(df, use_cuda=False)

    states = {}
    for r in results:
        state = r['state']
        ts = int(state.timestamp) if hasattr(state, 'timestamp') and state.timestamp else 0
        if ts > 0:
            states[ts] = state

    return states


def extract_16d(state, tf_name):
    """Build 16D feature vector from a ThreeBodyQuantumState.

    Matches the feature layout in fractal_clustering.py:extract_features()
    but without ancestry (features 5-6, 10-13 set to 0).
    """
    z = state.z_score
    v = abs(state.particle_velocity) if hasattr(state, 'particle_velocity') else 0.0
    m = abs(state.momentum_strength) if hasattr(state, 'momentum_strength') else 0.0
    c = state.coherence if hasattr(state, 'coherence') else 0.0

    tf_scale = math.log2(max(TF_SECONDS.get(tf_name, 60), 1))

    adx = (state.adx_strength * 0.01) if hasattr(state, 'adx_strength') else 0.0
    hurst = state.hurst_exponent if hasattr(state, 'hurst_exponent') else 0.5
    dmi_diff = ((state.dmi_plus - state.dmi_minus) * 0.01) if hasattr(state, 'dmi_plus') else 0.0
    pid = state.term_pid if hasattr(state, 'term_pid') else 0.0
    osc = state.oscillation_coherence if hasattr(state, 'oscillation_coherence') else 0.0

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
        all_tf_states: dict {tf_name: {timestamp: ThreeBodyQuantumState}}
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

    # Build timestamp→index mapping for base_df (for MFE/MAE computation)
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

    for t in tqdm(analysis_ts, desc="Building hypervolumes", unit="bar"):
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
            continue  # Need at least 3 TFs with data

        # --- Compute oracle MFE/MAE from base TF future bars ---
        if t not in ts_to_idx:
            continue

        bar_idx = ts_to_idx[t]
        if bar_idx + lookahead >= len(base_df):
            continue  # Not enough future data

        entry_price = float(base_df.iloc[bar_idx]['close'])
        future = base_df.iloc[bar_idx + 1 : bar_idx + 1 + lookahead]

        if future.empty:
            continue

        max_up = float(future['high'].max() - entry_price)
        max_down = float(entry_price - future['low'].min())

        if max_up == 0 and max_down == 0:
            continue

        # Direction from z-score sign at base TF
        base_state = base_states[t]
        z = base_state.z_score
        dmi_diff = (base_state.dmi_plus - base_state.dmi_minus) \
            if hasattr(base_state, 'dmi_plus') else 0.0

        # MFE/MAE assignment based on direction
        # z < 0 → LONG setup (MFE = up, MAE = down)
        # z > 0 → SHORT setup (MFE = down, MAE = up)
        if z < 0:  # LONG
            mfe_val = max_up
            mae_val = max_down
        else:      # SHORT
            mfe_val = max_down
            mae_val = max_up

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

    print(f"  Built {len(matrices)} hypervolume matrices with oracle labels")
    return matrices, np.array(mfes), np.array(maes), meta


# =============================================================================
#  PRICE-FIRST I-MR: Pure price/time analysis (no physics)
# =============================================================================

D4 = 3.267   # SPC constant for n=2 subgroup
E2 = 2.660   # SPC constant for I-chart limits from MR

def compute_price_imr(base_df, context_days=21, analysis_days=7):
    """Pure price-based I-MR chart from 15m close prices.

    I chart = raw close price per bar.
    MR = |close[t] - close[t-1]| (bar-to-bar price movement).
    Control limits calibrated from the warmup (context) period.

    Returns dict with all arrays and control limit values.
    """
    close = base_df['close'].values.astype(float)
    timestamps = base_df['timestamp'].values.astype(float)
    n = len(close)

    # MR: bar-to-bar signed price change (tracks direction)
    mr_signed = np.diff(close)
    mr_signed = np.concatenate([[0.0], mr_signed])  # pad first bar with 0
    mr_abs = np.abs(mr_signed)

    # Determine warmup boundary (in bar count)
    t_min, t_max = timestamps[0], timestamps[-1]
    data_span_days = (t_max - t_min) / 86400

    # Auto-adjust if data is shorter than context window
    if context_days > 0 and data_span_days < context_days + 1:
        old_ctx = context_days
        context_days = max(0, int(data_span_days * 0.3))
        print(f"  Auto-adjusted context: {old_ctx}d -> {context_days}d "
              f"(data span is only {data_span_days:.1f}d)")

    t_warmup_end = t_min + context_days * 86400

    warmup_mask = timestamps < t_warmup_end
    warmup_end_idx = int(warmup_mask.sum())

    if warmup_end_idx < 20:
        # Not enough warmup — use first 30% of data
        warmup_end_idx = max(20, int(n * 0.3))

    # Control limits from warmup period (use absolute MR for limits)
    warmup_close = close[:warmup_end_idx]
    warmup_mr_abs = mr_abs[1:warmup_end_idx]  # skip first MR (=0)

    center = float(np.mean(warmup_close))
    mr_bar = float(np.mean(warmup_mr_abs)) if len(warmup_mr_abs) > 0 else 1.0

    ucl_mr = D4 * mr_bar
    ucl_i = center + E2 * mr_bar
    lcl_i = center - E2 * mr_bar

    # Analysis window
    if analysis_days > 0:
        t_analysis_end = t_warmup_end + analysis_days * 86400
        analysis_mask = (timestamps >= t_warmup_end) & (timestamps < t_analysis_end)
    else:
        analysis_mask = timestamps >= t_warmup_end

    print(f"  Price I-MR: {n} bars, warmup={warmup_end_idx}, "
          f"analysis={int(analysis_mask.sum())}")
    print(f"  Center={center:.2f}, MR_bar={mr_bar:.2f}, "
          f"UCL_MR={ucl_mr:.2f}, UCL_I={ucl_i:.2f}, LCL_I={lcl_i:.2f}")

    return {
        'close': close,
        'mr': mr_signed,
        'mr_abs': mr_abs,
        'timestamps': timestamps,
        'center': center,
        'mr_bar': mr_bar,
        'ucl_mr': ucl_mr,
        'ucl_i': ucl_i,
        'lcl_i': lcl_i,
        'warmup_end_idx': warmup_end_idx,
        'analysis_mask': analysis_mask,
    }


def detect_regimes(price_imr, min_regime_bars=8):
    """Detect natural price regimes from MR UCL breaks.

    A new regime starts when MR > UCL_MR (price behavior changed character).
    Tiny regimes (< min_regime_bars) get merged into their larger neighbor.

    Returns:
        regime_ids: array of regime IDs per bar (full length, -1 for warmup)
        regime_meta: list of dicts with regime stats
    """
    close = price_imr['close']
    mr_abs = price_imr['mr_abs']
    ucl_mr = price_imr['ucl_mr']
    warmup_end = price_imr['warmup_end_idx']
    analysis_mask = price_imr['analysis_mask']
    n = len(close)

    # Initialize all bars to -1 (warmup/excluded)
    regime_ids = np.full(n, -1, dtype=int)

    # Find analysis bar indices
    analysis_indices = np.where(analysis_mask)[0]
    if len(analysis_indices) == 0:
        return regime_ids, []

    # Assign regime IDs: new regime at each |MR| > UCL break
    current_regime = 0
    for i, idx in enumerate(analysis_indices):
        if i > 0 and mr_abs[idx] > ucl_mr:
            current_regime += 1
        regime_ids[idx] = current_regime

    n_raw = current_regime + 1

    # Merge tiny regimes into neighbors
    merge_pass = 0
    while True:
        unique_ids = [r for r in np.unique(regime_ids) if r >= 0]
        sizes = {r: int((regime_ids == r).sum()) for r in unique_ids}
        tiny = [r for r in unique_ids if sizes[r] < min_regime_bars]
        if not tiny:
            break

        r = tiny[0]
        r_pos = unique_ids.index(r)
        if r_pos > 0 and r_pos < len(unique_ids) - 1:
            left, right = unique_ids[r_pos - 1], unique_ids[r_pos + 1]
            target = left if sizes.get(left, 0) >= sizes.get(right, 0) else right
        elif r_pos > 0:
            target = unique_ids[r_pos - 1]
        elif len(unique_ids) > 1:
            target = unique_ids[r_pos + 1]
        else:
            break
        regime_ids[regime_ids == r] = target
        merge_pass += 1
        if merge_pass > n:
            break

    # Re-compact to 0-based contiguous
    unique_ids = sorted([r for r in np.unique(regime_ids) if r >= 0])
    remap = {old: new for new, old in enumerate(unique_ids)}
    for i in range(n):
        if regime_ids[i] >= 0:
            regime_ids[i] = remap[regime_ids[i]]

    n_regimes = len(unique_ids)

    # Build regime metadata
    regime_meta = []
    for rid in range(n_regimes):
        mask = regime_ids == rid
        indices = np.where(mask)[0]
        r_close = close[mask]
        r_mr = mr_abs[mask]

        regime_meta.append({
            'regime_id': rid,
            'start_idx': int(indices[0]),
            'end_idx': int(indices[-1]),
            'n_bars': int(mask.sum()),
            'mean_price': float(np.mean(r_close)),
            'volatility': float(np.mean(r_mr)),
            'price_change': float(r_close[-1] - r_close[0]) if len(r_close) > 1 else 0.0,
            'direction': 'LONG' if (r_close[-1] > r_close[0]) else 'SHORT',
        })

    print(f"  Regimes: {n_raw} raw -> {n_regimes} after merge "
          f"(min_bars={min_regime_bars})")
    for rm in regime_meta:
        print(f"    R{rm['regime_id']}: {rm['n_bars']:>4} bars, "
              f"price={rm['mean_price']:.1f}, vol={rm['volatility']:.2f}, "
              f"dir={rm['direction']}, chg={rm['price_change']:+.1f}")

    return regime_ids, regime_meta


def compute_regime_oracle(base_df, regime_ids, regime_meta, lookahead=16):
    """Compute oracle MFE/MAE per analysis bar using regime-based direction.

    Direction comes from the regime's price trend (not z-score).

    Returns:
        bar_indices: array of bar indices in base_df
        mfes: array of MFE values
        maes: array of MAE values
        directions: array of 'LONG'/'SHORT' strings
    """
    close = base_df['close'].values.astype(float)
    high = base_df['high'].values.astype(float)
    low = base_df['low'].values.astype(float)
    n = len(base_df)

    # Build regime direction lookup
    regime_dir = {}
    for rm in regime_meta:
        regime_dir[rm['regime_id']] = rm['direction']

    analysis_indices = np.where(regime_ids >= 0)[0]

    bar_indices = []
    mfes = []
    maes = []
    directions = []

    for idx in analysis_indices:
        if idx + lookahead >= n:
            continue

        entry = close[idx]
        future_high = high[idx + 1: idx + 1 + lookahead]
        future_low = low[idx + 1: idx + 1 + lookahead]

        if len(future_high) == 0:
            continue

        max_up = float(future_high.max() - entry)
        max_down = float(entry - future_low.min())

        rid = regime_ids[idx]
        direction = regime_dir.get(rid, 'LONG')

        if direction == 'LONG':
            mfe_val = max_up
            mae_val = max_down
        else:
            mfe_val = max_down
            mae_val = max_up

        bar_indices.append(idx)
        mfes.append(mfe_val)
        maes.append(mae_val)
        directions.append(direction)

    print(f"  Oracle: {len(mfes)} bars with MFE/MAE "
          f"(lookahead={lookahead})")

    return (np.array(bar_indices), np.array(mfes),
            np.array(maes), np.array(directions))


# =============================================================================
#  I-MR CHART PLOTS (matplotlib)
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'standalone')


def _resolve_plots_dir(data_path):
    """Set PLOTS_DIR subfolder based on data path.
    ATLAS_1DAY -> 1d, ATLAS_1WEEK -> 1w, ATLAS_OOS (4mo) -> 1y, ATLAS (10mo) -> 1y."""
    global PLOTS_DIR
    base = os.path.join(os.path.dirname(__file__), 'plots', 'standalone')
    dp = data_path.upper().replace('\\', '/')
    if '1DAY' in dp or '1_DAY' in dp:
        sub = '1d'
    elif '1WEEK' in dp or '1_WEEK' in dp:
        sub = '1w'
    elif 'OOS' in dp:
        sub = '1y'  # 4 months ≈ treat as long-horizon
    else:
        # Full ATLAS (10 months)
        sub = '1y'
    PLOTS_DIR = os.path.join(base, sub)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return sub


# Regime color palette (up to 20 distinct regimes)
_REGIME_COLORS = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#E91E63', '#8BC34A', '#FF5722', '#3F51B5',
    '#CDDC39', '#795548', '#607D8B', '#009688', '#FFC107',
    '#673AB7', '#03A9F4', '#FFEB3B', '#FF6F00', '#1B5E20',
]


def plot_price_imr(price_imr, regime_ids, regime_meta, base_df):
    """Plot the foundational price I-MR chart with regime coloring.

    4 panels: Price, I chart, MR chart, Regime map.
    Saves to tools/plots/standalone/0_price_imr.png
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    close = price_imr['close']
    mr = price_imr['mr']
    timestamps = price_imr['timestamps']
    center = price_imr['center']
    ucl_i = price_imr['ucl_i']
    lcl_i = price_imr['lcl_i']
    mr_bar = price_imr['mr_bar']
    ucl_mr = price_imr['ucl_mr']
    warmup_end = price_imr['warmup_end_idx']

    n = len(close)
    x = np.arange(n)

    # Convert timestamps to readable dates for x-axis
    from datetime import datetime, timezone
    date_labels = []
    date_positions = []
    prev_day = None
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day = dt.strftime('%m/%d')
        if day != prev_day:
            date_labels.append(day)
            date_positions.append(i)
            prev_day = day

    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True,
                              gridspec_kw={'height_ratios': [3, 2, 2, 1]})
    fig.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    n_regimes = len(regime_meta)

    # --- Panel 1: Price colored by regime ---
    ax = axes[0]
    # Draw warmup in gray
    if warmup_end > 1:
        ax.plot(x[:warmup_end], close[:warmup_end], color='#BBBBBB',
                linewidth=0.8, alpha=0.6)

    # Draw each regime segment with reference line at mean price
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.plot(x[s:e+1], close[s:e+1], color=color, linewidth=1.2,
                label=f"R{rm['regime_id']} ({rm['direction']})")
        # Regime mean price reference line
        ax.hlines(y=rm['mean_price'], xmin=s, xmax=e, color=color,
                  linestyle='--', linewidth=0.8, alpha=0.5)

    # Regime boundary vertical lines (on all panels)
    for rm in regime_meta[1:]:
        for a in axes:
            a.axvline(x=rm['start_idx'], color='#888888', linestyle=':',
                      linewidth=0.7, alpha=0.5)

    # Warmup boundary (on all panels)
    for a in axes:
        a.axvline(x=warmup_end, color='#333333', linestyle='--',
                  linewidth=1, alpha=0.6)
    ax.plot([], [], color='#333333', linestyle='--', label='Warmup end')

    ax.set_title('Price (15m Close) — Colored by Regime (dashed = mean price)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(fontsize=7, loc='upper left', ncol=min(n_regimes + 1, 6))
    ax.grid(True, alpha=0.15)

    # --- Panel 2: I chart (close with control limits + data points) ---
    ax = axes[1]
    ax.plot(x, close, color='#333333', linewidth=0.6, alpha=0.5)
    # Individual data point for every bar
    inside = (close <= ucl_i) & (close >= lcl_i)
    ax.scatter(x[inside], close[inside], color='#333333', s=6,
               zorder=4, alpha=0.6, label='Data point')
    # Highlight points outside control limits in red
    outside = ~inside
    if outside.any():
        ax.scatter(x[outside], close[outside], color='#F44336', s=12,
                   zorder=5, label='Outside limits')

    ax.axhline(y=center, color='#888888', linestyle='-', linewidth=1.5,
               label=f'Center={center:.1f}')
    ax.axhline(y=ucl_i, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'UCL={ucl_i:.1f}')
    ax.axhline(y=lcl_i, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'LCL={lcl_i:.1f}')

    ax.set_title('I Chart — Individual Close Prices (each dot = 1 bar)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Close', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.15)

    # --- Panel 3: MR chart (signed) ---
    ax = axes[2]
    mr_abs = price_imr['mr_abs']

    # Color bars by sign: green=up, red=down
    colors_mr = np.where(mr >= 0, '#4CAF50', '#F44336')
    ax.bar(x, mr, color=colors_mr, width=1.0, alpha=0.7, edgecolor='none')

    # UCL / LCL (symmetric for signed MR)
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
    ax.axhline(y=ucl_mr, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'+UCL={ucl_mr:.2f}')
    ax.axhline(y=-ucl_mr, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'-UCL={-ucl_mr:.2f}')
    ax.axhline(y=mr_bar, color='#888888', linestyle=':', linewidth=1,
               alpha=0.5, label=f'MR_bar={mr_bar:.2f}')
    ax.axhline(y=-mr_bar, color='#888888', linestyle=':', linewidth=1,
               alpha=0.5)

    # Mark UCL breaks
    mr_break = mr_abs > ucl_mr
    if mr_break.any():
        ax.scatter(x[mr_break], mr[mr_break], color='black', s=15,
                   zorder=5, marker='x', linewidths=1, label='UCL break')

    ax.set_title('MR Chart — Signed Moving Range (green=up, red=down)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Price Change', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.15)

    # --- Panel 4: Regime map ---
    ax = axes[3]
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.barh(0, e - s + 1, left=s, height=0.8, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        # Label in center
        mid = (s + e) / 2
        ax.text(mid, 0, f"R{rm['regime_id']}\n{rm['direction']}\n{rm['n_bars']}b",
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white')

    ax.set_yticks([])
    ax.set_title('Regime Map — Natural price segments from MR UCL breaks',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Bar index', fontsize=10)

    # Date tick labels
    if date_positions:
        step = max(1, len(date_positions) // 15)
        ax.set_xticks([date_positions[i] for i in range(0, len(date_positions), step)])
        ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), step)],
                           rotation=45, ha='right', fontsize=8)

    fig.suptitle(f'PRICE I-MR CHART — {n} bars, {n_regimes} regimes\n'
                 f'Warmup: {warmup_end} bars | UCL_MR={ucl_mr:.2f} '
                 f'(breaks trigger new regime)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '0_price_imr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_regime_summary(regime_meta, mfes, maes, bar_indices, regime_ids):
    """2x2 regime dashboard.

    Saves to tools/plots/standalone/0b_regime_summary.png
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.set_facecolor('white')
    for row in axes:
        for ax in row:
            ax.set_facecolor('white')

    n_regimes = len(regime_meta)
    regime_labels = [f"R{rm['regime_id']}" for rm in regime_meta]
    colors = [_REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
              for rm in regime_meta]

    # Map bar_indices to their regime IDs
    bar_regime = regime_ids[bar_indices]

    # --- Top-left: MFE distribution by regime (box plot) ---
    ax = axes[0, 0]
    mfe_by_regime = []
    labels_used = []
    for rm in regime_meta:
        mask = bar_regime == rm['regime_id']
        if mask.any():
            mfe_by_regime.append(mfes[mask])
            labels_used.append(f"R{rm['regime_id']}")
    if mfe_by_regime:
        bp = ax.boxplot(mfe_by_regime, labels=labels_used, patch_artist=True)
        for patch, rm in zip(bp['boxes'], regime_meta):
            patch.set_facecolor(_REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)])
            patch.set_alpha(0.6)
    ax.set_title('MFE Distribution by Regime', fontsize=11, fontweight='bold')
    ax.set_ylabel('MFE (points)', fontsize=9)
    ax.grid(True, alpha=0.15)

    # --- Top-right: Regime volatility vs mean MFE (scatter) ---
    ax = axes[0, 1]
    for rm in regime_meta:
        mask = bar_regime == rm['regime_id']
        mean_mfe = float(mfes[mask].mean()) if mask.any() else 0
        c = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.scatter(rm['volatility'], mean_mfe, color=c, s=rm['n_bars'] * 2,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f"R{rm['regime_id']}", (rm['volatility'], mean_mfe),
                    fontsize=8, ha='left', va='bottom')
    ax.set_title('Regime Volatility vs Mean MFE', fontsize=11, fontweight='bold')
    ax.set_xlabel('Volatility (mean MR)', fontsize=9)
    ax.set_ylabel('Mean MFE', fontsize=9)
    ax.grid(True, alpha=0.15)

    # --- Bottom-left: Regime duration histogram ---
    ax = axes[1, 0]
    durations = [rm['n_bars'] for rm in regime_meta]
    ax.bar(range(n_regimes), durations, color=colors, edgecolor='white',
           linewidth=0.5)
    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels(regime_labels, fontsize=8)
    ax.set_title('Regime Duration (bars)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of bars', fontsize=9)
    ax.grid(True, alpha=0.15)

    # --- Bottom-right: Win rate by regime ---
    ax = axes[1, 1]
    win_rates = []
    for rm in regime_meta:
        mask = bar_regime == rm['regime_id']
        if mask.any():
            wins = (mfes[mask] > maes[mask]).sum()
            wr = float(wins) / float(mask.sum()) * 100
        else:
            wr = 0.0
        win_rates.append(wr)
    ax.bar(range(n_regimes), win_rates, color=colors, edgecolor='white',
           linewidth=0.5)
    ax.axhline(y=50, color='#888888', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels(regime_labels, fontsize=8)
    ax.set_title('Win Rate by Regime (MFE > MAE)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Win Rate %', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'REGIME SUMMARY — {n_regimes} regimes, '
                 f'{len(mfes)} analysis bars',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '0b_regime_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_imr_charts(padded, mfes):
    """Generate Minitab-style I-MR charts for key features.

    Saves to tools/plots/standalone/:
      1_imr_key_features.png  — 6 key features, I + MR panel each
      2_i_heatmap.png         — full 12×16 I-chart heatmap
      3_mr_heatmap.png        — full 11×16 MR heatmap
      4_imr_correlation.png   — r(MFE) heatmap for I and MR
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n, n_depths, n_feat = padded.shape
    D4 = 3.267

    mr = np.diff(padded, axis=1)       # (n, 11, 16)
    mr_abs = np.abs(mr)

    tf_labels_short = [TF_HIERARCHY[d] for d in range(n_depths)]
    trans_labels = [f"{TF_HIERARCHY[d]}>{TF_HIERARCHY[d+1]}" for d in range(n_depths - 1)]

    key_feats = [
        (0,  'z_score',    'Signed z (fair value distance)'),
        (9,  'dmi_diff',   'DMI diff (directional bias)'),
        (14, 'self_pid',   'PID control force'),
        (3,  'coherence',  'Wave coherence'),
        (7,  'self_adx',   'ADX (trend strength)'),
        (8,  'self_hurst', 'Hurst exponent'),
    ]

    # ── PLOT 1: Key features I-MR panels ──
    fig = plt.figure(figsize=(20, 24))
    outer = gridspec.GridSpec(6, 1, hspace=0.35, figure=fig)

    for row, (f_idx, f_name, f_desc) in enumerate(key_feats):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row],
                                                 wspace=0.25)
        ax_i = fig.add_subplot(inner[0])
        ax_mr = fig.add_subplot(inner[1])

        i_vals = padded[:, :, f_idx]
        mr_vals_f = mr[:, :, f_idx]
        mr_abs_f = mr_abs[:, :, f_idx]

        i_mean = i_vals.mean(axis=0)
        i_std = i_vals.std(axis=0)
        center = float(i_mean.mean())

        mr_abs_mean = mr_abs_f.mean(axis=0)
        mr_bar = float(mr_abs_f.mean())
        ucl_val = D4 * mr_bar

        # Correlation with MFE at each depth
        corr_i = np.zeros(n_depths)
        for d in range(n_depths):
            col = i_vals[:, d]
            if np.std(col) > 1e-12:
                c = np.corrcoef(col, mfes)[0, 1]
                corr_i[d] = c if not np.isnan(c) else 0.0

        # ── I chart ──
        x = np.arange(n_depths)
        colors_i = ['#F44336' if abs(c) > 0.15 else '#2196F3' for c in corr_i]

        ax_i.bar(x, i_mean, color=colors_i, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_i.errorbar(x, i_mean, yerr=i_std, fmt='none', ecolor='gray',
                       capsize=3, linewidth=1)
        ax_i.axhline(y=center, color='green', linestyle='-', linewidth=1.5,
                      label=f'Center={center:.3f}')
        ax_i.axhline(y=center + 3 * float(i_std.mean()), color='red',
                      linestyle='--', linewidth=1, alpha=0.7, label='UCL/LCL')
        ax_i.axhline(y=center - 3 * float(i_std.mean()), color='red',
                      linestyle='--', linewidth=1, alpha=0.7)
        ax_i.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)

        ax_i.set_xticks(x)
        ax_i.set_xticklabels(tf_labels_short, rotation=45, ha='right', fontsize=8)
        ax_i.set_title(f'I Chart: {f_name}\n{f_desc}', fontsize=10, fontweight='bold')
        ax_i.set_ylabel('Mean value', fontsize=9)
        ax_i.legend(fontsize=7, loc='best')
        ax_i.grid(True, alpha=0.2)

        # Annotate r(MFE) on bars
        for d in range(n_depths):
            if abs(corr_i[d]) > 0.05:
                ax_i.text(d, i_mean[d], f'r={corr_i[d]:+.2f}', ha='center',
                         va='bottom' if i_mean[d] >= 0 else 'top',
                         fontsize=6, color='#333')

        # ── MR chart ──
        x_mr = np.arange(n_depths - 1)
        breaks_pct = (mr_abs_f > ucl_val).mean(axis=0) * 100
        colors_mr = ['#FF5722' if bp > 5 else '#4CAF50' for bp in breaks_pct]

        ax_mr.bar(x_mr, mr_abs_mean, color=colors_mr, alpha=0.7,
                  edgecolor='white', linewidth=0.5)
        ax_mr.axhline(y=mr_bar, color='green', linestyle='-', linewidth=1.5,
                      label=f'MR_bar={mr_bar:.4f}')
        ax_mr.axhline(y=ucl_val, color='red', linestyle='--', linewidth=1.5,
                      label=f'UCL={ucl_val:.4f}')

        ax_mr.set_xticks(x_mr)
        ax_mr.set_xticklabels(trans_labels, rotation=45, ha='right', fontsize=7)
        ax_mr.set_title(f'MR Chart: {f_name}\nUCL breaks shown in red', fontsize=10,
                        fontweight='bold')
        ax_mr.set_ylabel('Mean |MR|', fontsize=9)
        ax_mr.legend(fontsize=7, loc='best')
        ax_mr.grid(True, alpha=0.2)

        # Annotate break %
        for d in range(n_depths - 1):
            if breaks_pct[d] > 1:
                ax_mr.text(d, mr_abs_mean[d], f'{breaks_pct[d]:.0f}%',
                          ha='center', va='bottom', fontsize=6, color='#333')

    fig.suptitle(f'FRACTAL I-MR CHART — {n} data points × {n_depths} TF depths\n'
                 f'I = feature value at each TF | MR = TF-to-TF transition\n'
                 f'Red I bars = |r(MFE)| > 0.15 | Red MR bars = >5% UCL breaks',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(PLOTS_DIR, '1_imr_key_features.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 1_imr_key_features.png")

    # ── PLOT 2: Full I-chart heatmap (12 TF × 16 features) ──
    i_matrix = padded.mean(axis=0)  # (12, 16)

    fig, ax = plt.subplots(figsize=(16, 8))
    vmax = max(abs(i_matrix.min()), abs(i_matrix.max()), 0.5)
    im = ax.imshow(i_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_depths))
    ax.set_yticklabels([f'{TF_LABELS[d]} ({TF_HIERARCHY[d]})' for d in range(n_depths)],
                        fontsize=9)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('TF Depth (macro → micro)', fontsize=12)
    ax.set_title(f'I-CHART HEATMAP: Mean feature value at each TF depth\n'
                 f'{n} data points | Red = positive, Blue = negative',
                 fontsize=13, fontweight='bold')

    for d in range(n_depths):
        for f in range(n_feat):
            val = i_matrix[d, f]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean value', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '2_i_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 2_i_heatmap.png")

    # ── PLOT 3: MR heatmap (11 transitions × 16 features) ──
    mr_matrix = mr_abs.mean(axis=0)  # (11, 16)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(mr_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_depths - 1))
    ax.set_yticklabels(trans_labels, fontsize=8)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('TF Transition', fontsize=12)
    ax.set_title(f'MR HEATMAP: Mean |MR| at each TF transition\n'
                 f'Higher = bigger regime change between adjacent TFs',
                 fontsize=13, fontweight='bold')

    for d in range(n_depths - 1):
        for f in range(n_feat):
            val = mr_matrix[d, f]
            if val > 0.01:
                color = 'white' if val > mr_matrix.max() * 0.5 else 'black'
                ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean |MR|', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '3_mr_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 3_mr_heatmap.png")

    # ── PLOT 4: Correlation with MFE heatmap (I + MR side by side) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # I correlations
    corr_i_matrix = np.zeros((n_depths, n_feat))
    for d in range(n_depths):
        for f in range(n_feat):
            col = padded[:, d, f]
            if np.std(col) > 1e-12:
                c = np.corrcoef(col, mfes)[0, 1]
                corr_i_matrix[d, f] = c if not np.isnan(c) else 0.0

    im1 = ax1.imshow(corr_i_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax1.set_xticks(range(n_feat))
    ax1.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(n_depths))
    ax1.set_yticklabels([f'{TF_HIERARCHY[d]}' for d in range(n_depths)], fontsize=9)
    ax1.set_title('I values: r(MFE)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('TF Depth', fontsize=10)

    for d in range(n_depths):
        for f in range(n_feat):
            val = corr_i_matrix[d, f]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > 0.15 else 'black'
                ax1.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # MR correlations
    corr_mr_matrix = np.zeros((n_depths - 1, n_feat))
    for d in range(n_depths - 1):
        for f in range(n_feat):
            col = mr[:, d, f]
            if np.std(col) > 1e-12:
                c = np.corrcoef(col, mfes)[0, 1]
                corr_mr_matrix[d, f] = c if not np.isnan(c) else 0.0

    im2 = ax2.imshow(corr_mr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax2.set_xticks(range(n_feat))
    ax2.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(n_depths - 1))
    ax2.set_yticklabels(trans_labels, fontsize=8)
    ax2.set_title('MR values: r(MFE)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('TF Transition', fontsize=10)

    for d in range(n_depths - 1):
        for f in range(n_feat):
            val = corr_mr_matrix[d, f]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > 0.15 else 'black'
                ax2.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(f'CORRELATION WITH MFE: Which (TF × Feature) dimensions predict outcome?\n'
                 f'Red = higher value → higher MFE | Blue = higher value → lower MFE',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '4_imr_correlation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 4_imr_correlation.png")


def plot_segmented_imr(padded, mfes, maes, meta, tids, long_mask,
                       keep_segs, split_segs, drop_segs, base_df=None):
    """Plot 15m-anchored I-MR chart with price, z_score, and fission segments.

    Top panel = actual price through time (the thing we're trading).
    Then z_score I-chart, MR jump magnitude, MFE/MAE outcome, fission strip.
    Background bands = fission class. Line color/thickness = jump size.

    Saves to tools/plots/standalone/:
      5_segmented_imr_15m.png  — price + I-chart + MR with fission colors
      6_segmented_heatmap.png  — full fractal heatmap colored by segment
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = len(mfes)
    base_tf_depth = TF_HIERARCHY.index('15m')  # d5

    # Build fission class per data point
    keep_tids = set((s['tid'], s['dir']) for s in keep_segs)
    split_tids = set((s['tid'], s['dir']) for s in split_segs)

    fission_class = np.full(n, 2, dtype=int)  # 0=KEEP, 1=SPLIT, 2=DROP
    for i in range(n):
        d = 'LONG' if long_mask[i] else 'SHORT'
        tid = tids[i]
        if (tid, d) in keep_tids:
            fission_class[i] = 0
        elif (tid, d) in split_tids:
            fission_class[i] = 1

    fission_colors = {0: '#4CAF50', 1: '#FFC107', 2: '#F44336'}  # green/yellow/red
    fission_labels = {0: 'KEEP', 1: 'SPLIT', 2: 'DROP'}
    point_colors = [fission_colors[c] for c in fission_class]

    # Timestamps for x-axis
    from datetime import datetime, timezone as tz
    timestamps = [m['ts'] for m in meta]
    dt_labels = [datetime.fromtimestamp(t, tz=tz.utc) for t in timestamps]

    # Key features to plot on I-chart
    z_vals = padded[:, base_tf_depth, 0]       # z_score at 15m
    pid_vals = padded[:, base_tf_depth, 14]    # PID at 15m
    adx_vals = padded[:, base_tf_depth, 7]     # ADX at 15m
    dmi_vals = padded[:, base_tf_depth, 9]     # DMI diff at 15m

    # MR (bar-to-bar difference at 15m)
    z_mr = np.abs(np.diff(z_vals))
    D4 = 3.267
    mr_bar = float(z_mr.mean())
    ucl = D4 * mr_bar

    # ── PLOT 5: Segmented I-MR on 15m anchor ──
    # White background. Line color = fission class only.
    # Green (#2E7D32) = KEEP/trade, Yellow (#F9A825) = SPLIT/mixed, Red (#C62828) = DROP/no trade
    from matplotlib.collections import LineCollection

    # Extract actual close prices aligned to each analysis point
    prices = np.zeros(n)
    if base_df is not None and 'close' in base_df.columns:
        ts_col = base_df['timestamp'].values if 'timestamp' in base_df.columns else np.arange(len(base_df))
        ts_to_idx = {}
        for idx_i, t in enumerate(ts_col):
            ts_to_idx[int(t)] = idx_i
        for i, m in enumerate(meta):
            bar_idx = ts_to_idx.get(int(m['ts']), -1)
            if bar_idx >= 0:
                prices[i] = float(base_df.iloc[bar_idx]['close'])
    has_price = prices.sum() > 0

    n_panels = 5 if has_price else 4
    ratios = [3, 3, 2, 2, 1] if has_price else [3, 2, 2, 1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 22 if has_price else 16),
                              sharex=True, gridspec_kw={'height_ratios': ratios})
    fig.patch.set_facecolor('white')

    x = np.arange(n)
    seg_colors = [fission_colors[c] for c in fission_class[:-1]]

    def _add_legend(ax):
        for cls in [0, 1, 2]:
            n_cls = (fission_class == cls).sum()
            wr_cls = float((mfes[fission_class == cls] > maes[fission_class == cls]).mean()) \
                if n_cls > 0 else 0
            ax.plot([], [], color=fission_colors[cls], linewidth=3,
                    label=f'{fission_labels[cls]} (n={n_cls}, WR={wr_cls:.0%})')

    panel_idx = 0

    # Panel 0: PRICE through time
    if has_price:
        ax0 = axes[panel_idx]
        ax0.set_facecolor('white')
        panel_idx += 1

        points_p = np.column_stack([x, prices]).reshape(-1, 1, 2)
        segments_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
        lc_p = LineCollection(segments_p, colors=seg_colors, linewidths=1.5)
        ax0.add_collection(lc_p)
        ax0.set_xlim(0, n - 1)
        p_range = prices.max() - prices.min()
        ax0.set_ylim(prices.min() - p_range * 0.05, prices.max() + p_range * 0.05)

        _add_legend(ax0)
        ax0.legend(fontsize=8, loc='upper right', ncol=3)
        ax0.set_ylabel('Price (15m close)', fontsize=10)
        ax0.set_title('PRICE — Green=trade, Yellow=mixed, Red=no trade',
                      fontsize=12, fontweight='bold')
        ax0.grid(True, alpha=0.2)

    # Panel 1: I-chart (z_score)
    ax1 = axes[panel_idx]
    ax1.set_facecolor('white')
    panel_idx += 1

    points = np.column_stack([x, z_vals]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
    ax1.add_collection(lc)
    ax1.set_xlim(0, n - 1)
    ax1.set_ylim(z_vals.min() - 0.5, z_vals.max() + 0.5)

    center = float(np.mean(z_vals))
    std_z = float(np.std(z_vals))
    ax1.axhline(y=center, color='#888888', linewidth=1, linestyle='-', alpha=0.5, label=f'Center={center:.3f}')
    ax1.axhline(y=center + 3 * std_z, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4, label='UCL/LCL')
    ax1.axhline(y=center - 3 * std_z, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle=':', alpha=0.3)

    _add_legend(ax1)
    ax1.legend(fontsize=7, loc='upper right', ncol=3)
    ax1.set_ylabel('z_score (15m)', fontsize=10)
    ax1.set_title('I-CHART: 15m z_score', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Panel 2: MR chart
    ax2 = axes[panel_idx]
    ax2.set_facecolor('white')
    panel_idx += 1

    seg_colors_mr = [fission_colors[c] for c in fission_class[1:-1]]
    x_mr = np.arange(len(z_mr))
    points_mr = np.column_stack([x_mr, z_mr]).reshape(-1, 1, 2)
    segments_mr = np.concatenate([points_mr[:-1], points_mr[1:]], axis=1)
    lc_mr = LineCollection(segments_mr, colors=seg_colors_mr, linewidths=1.2)
    ax2.add_collection(lc_mr)
    ax2.set_xlim(0, len(z_mr) - 1)
    ax2.set_ylim(0, min(z_mr.max() * 1.2, ucl * 2))
    ax2.axhline(y=mr_bar, color='#888888', linewidth=1, alpha=0.5, label=f'MR_bar={mr_bar:.4f}')
    ax2.axhline(y=ucl, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4, label=f'UCL={ucl:.4f}')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylabel('|MR| z_score', fontsize=10)
    ax2.set_title('MR CHART: Bar-to-bar jump magnitude', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    # Panel 3: MFE/MAE
    ax3 = axes[panel_idx]
    ax3.set_facecolor('white')
    panel_idx += 1

    points_mfe = np.column_stack([x, mfes]).reshape(-1, 1, 2)
    seg_mfe = np.concatenate([points_mfe[:-1], points_mfe[1:]], axis=1)
    lc_mfe = LineCollection(seg_mfe, colors=seg_colors, linewidths=1.2)
    ax3.add_collection(lc_mfe)

    points_mae = np.column_stack([x, -maes]).reshape(-1, 1, 2)
    seg_mae = np.concatenate([points_mae[:-1], points_mae[1:]], axis=1)
    lc_mae = LineCollection(seg_mae, colors=seg_colors, linewidths=0.8, alpha=0.5)
    ax3.add_collection(lc_mae)
    ax3.set_xlim(0, n - 1)
    ax3.set_ylim(-maes.max() * 1.1, mfes.max() * 1.1)
    ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax3.axhline(y=float(np.mean(mfes)), color='#2196F3', linewidth=0.8, linestyle='--',
                alpha=0.5, label=f'Mean MFE={np.mean(mfes):.0f}')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_ylabel('MFE / -MAE (ticks)', fontsize=10)
    ax3.set_title('ORACLE OUTCOME: MFE (solid) and -MAE (faded)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.2)

    # Panel 4: Fission class strip
    ax4 = axes[panel_idx]
    ax4.set_facecolor('white')
    for i in range(n):
        ax4.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i]], alpha=0.8)
    ax4.set_yticks([])
    ax4.set_ylabel('Class', fontsize=10)
    ax4.set_xlabel('Analysis bar index (15m)', fontsize=10)

    # X-axis timestamps
    n_ticks = min(20, n)
    tick_positions = np.linspace(0, n - 1, n_ticks, dtype=int)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([dt_labels[i].strftime('%m/%d %H:%M') for i in tick_positions],
                        rotation=45, ha='right', fontsize=7)

    fig.suptitle(f'SEGMENTED I-MR CHART — 15m Anchor ({n} data points)\n'
                 f'Green=KEEP (trade), Yellow=SPLIT (mixed), Red=DROP (no trade)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '5_segmented_imr_15m.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 5_segmented_imr_15m.png")

    # ── PLOT 6: Fractal heatmap — mean feature value per (TF depth, feature), split by fission ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax_idx, (cls, cls_label) in enumerate([(0, 'KEEP'), (1, 'SPLIT'), (2, 'DROP')]):
        ax = axes[ax_idx]
        cls_mask = fission_class == cls
        n_cls = cls_mask.sum()

        if n_cls < 2:
            ax.set_title(f'{cls_label} (n={n_cls})\n(too few)', fontsize=11)
            ax.axis('off')
            continue

        cls_padded = padded[cls_mask]
        cls_mean = cls_padded.mean(axis=0)  # (12, 16)

        # Correlation with MFE for this class
        cls_mfes = mfes[cls_mask]
        cls_maes = maes[cls_mask]
        cls_wr = float((cls_mfes > cls_maes).mean())

        vmax = max(abs(cls_mean.min()), abs(cls_mean.max()), 0.5)
        im = ax.imshow(cls_mean, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(16))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f'{TF_HIERARCHY[d]}' for d in range(12)], fontsize=8)

        if ax_idx == 0:
            ax.set_ylabel('TF Depth (macro → micro)', fontsize=10)

        ax.set_title(f'{cls_label} (n={n_cls}, WR={cls_wr:.0%})',
                    fontsize=11, fontweight='bold',
                    color=fission_colors[cls])

        # Annotate cells with values
        for d in range(12):
            for f in range(16):
                val = cls_mean[d, f]
                if abs(val) > 0.01:
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                            fontsize=5, color=color)

        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f'FRACTAL FINGERPRINT BY FISSION CLASS\n'
                 f'Mean (TF × Feature) value — what does each class look like?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '6_segmented_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 6_segmented_heatmap.png")


# =============================================================================
#  ANALYSIS PIPELINE (reused from waveform_screening.py)
# =============================================================================

def pad_to_fixed_depth(matrices, max_depth=12):
    """Pad variable-depth matrices to fixed (max_depth, 16) with zeros."""
    n = len(matrices)
    padded = np.zeros((n, max_depth, 16))
    for i, mat in enumerate(matrices):
        d = min(mat.shape[0], max_depth)
        padded[i, :d, :] = mat[:d, :]
    return padded


def compute_moving_range(padded):
    """Compute I-MR segmentation features from (n, 12, 16) hypervolume.

    Returns: mr_flat (n, 448), mr_col_names
    """
    n, n_depths, n_feat = padded.shape

    # MR: depth-to-depth differences
    mr = np.diff(padded, axis=1)  # (n, 11, 16)

    # UCL per feature column (D4=3.267 for n=2 subgroup)
    D4 = 3.267
    mr_abs = np.abs(mr)
    mr_bar_global = mr_abs.mean(axis=(0, 1))  # (16,)
    ucl = D4 * mr_bar_global
    ucl_flags = (mr_abs > ucl[None, None, :]).astype(float)

    # Column summaries
    slopes = np.zeros((n, n_feat))
    mr_bar_local = np.zeros((n, n_feat))
    n_breaks = np.zeros((n, n_feat))

    depth_x = np.arange(n_depths, dtype=float)
    depth_x_centered = depth_x - depth_x.mean()
    denom = (depth_x_centered ** 2).sum()

    for f in range(n_feat):
        col_vals = padded[:, :, f]
        slopes[:, f] = (col_vals * depth_x_centered[None, :]).sum(axis=1) / max(denom, 1e-12)
        mr_bar_local[:, f] = mr_abs[:, :, f].mean(axis=1)
        n_breaks[:, f] = ucl_flags[:, :, f].sum(axis=1)

    # Flatten
    mr_flat_parts = []
    mr_col_names = []

    # MR values (11 × 16 = 176)
    mr_flat_parts.append(mr.reshape(n, -1))
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"MR_{d_from}>{d_to}__{f_lbl}")

    # UCL flags (11 × 16 = 176)
    mr_flat_parts.append(ucl_flags.reshape(n, -1))
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"UCL_{d_from}>{d_to}__{f_lbl}")

    # Column summaries (3 × 16 = 48)
    mr_flat_parts.append(slopes)
    for f in range(n_feat):
        mr_col_names.append(f"slope__{FEATURE_NAMES[f]}")
    mr_flat_parts.append(mr_bar_local)
    for f in range(n_feat):
        mr_col_names.append(f"mr_bar__{FEATURE_NAMES[f]}")
    mr_flat_parts.append(n_breaks)
    for f in range(n_feat):
        mr_col_names.append(f"n_breaks__{FEATURE_NAMES[f]}")

    mr_flat = np.hstack(mr_flat_parts)
    print(f"  MR features: {mr_flat.shape[1]} columns "
          f"(176 MR + 176 UCL + 48 summaries)")
    return mr_flat, mr_col_names


def flatten_matrices(padded):
    """Flatten (n, 12, 16) -> (n, 192) with named columns."""
    n = padded.shape[0]
    flat = padded.reshape(n, -1)

    col_names = []
    for d in range(padded.shape[1]):
        tf_lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        for f in range(padded.shape[2]):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            col_names.append(f"{tf_lbl}__{f_lbl}")

    return flat, col_names


def screen_factors(flat, col_names, mfes):
    """Correlate each column with MFE, return sorted by |corr|."""
    results = []
    for j, name in enumerate(col_names):
        col = flat[:, j]
        if np.std(col) < 1e-12:
            results.append((name, 0.0, 0.0))
            continue
        corr = float(np.corrcoef(col, mfes)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        results.append((name, corr, abs(corr)))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def regression_r2(flat, col_names, mfes, top_k=20, return_model=False):
    """Stepwise OLS on top-K factors, report adj-R².
    If return_model=True, also returns (model, scaler, top_indices) for the final step."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    screening = screen_factors(flat, col_names, mfes)
    top_names = [s[0] for s in screening[:top_k]]
    top_indices = [col_names.index(n) for n in top_names]

    print(f"\n{'='*70}")
    print(f"  STEPWISE REGRESSION (top {top_k} factors -> MFE)")
    print(f"{'='*70}")
    print(f"  {'Step':>4}  {'Factor':<35} {'R2':>8}  {'dR2':>8}  {'adj-R2':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}")

    scaler = StandardScaler()
    prev_r2 = 0.0
    steps = []

    for step, idx in enumerate(top_indices, 1):
        selected = top_indices[:step]
        X = scaler.fit_transform(flat[:, selected])
        reg = LinearRegression().fit(X, mfes)
        r2 = reg.score(X, mfes)
        n, k = X.shape
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)
        delta = r2 - prev_r2
        print(f"  {step:>4}  {col_names[idx]:<35} {r2:>8.4f}  {delta:>+8.4f}  {adj_r2:>8.4f}")
        steps.append((col_names[idx], r2, delta, adj_r2))
        prev_r2 = r2

    if return_model:
        # Refit final model cleanly for reuse
        final_scaler = StandardScaler()
        X_final = final_scaler.fit_transform(flat[:, top_indices])
        final_model = LinearRegression().fit(X_final, mfes)
        return steps, (final_model, final_scaler, top_indices)

    return steps


def print_screening_report(results, mfes, maes, meta, top_n=30):
    """Print the screening report."""
    print(f"\n{'='*70}")
    print(f"  STANDALONE WAVEFORM SCREENING REPORT")
    print(f"{'='*70}")
    print(f"  Data points: {len(mfes):,}")
    print(f"  MFE: mean={np.mean(mfes):.2f}, std={np.std(mfes):.2f}")
    print(f"  MAE: mean={np.mean(maes):.2f}, std={np.std(maes):.2f}")
    print(f"  Win rate (MFE > MAE): {(mfes > maes).mean():.1%}")

    # Top correlations
    print(f"\n  TOP {top_n} FACTORS (correlation with MFE):")
    print(f"  {'Rank':>4}  {'Factor':<35} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results[:top_n], 1):
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

    # Dead factors
    dead = [r for r in results if r[2] < 0.01]
    print(f"\n  Dead factors (|corr| < 0.01): {len(dead)} / {len(results)}")

    # Group by TF depth
    print(f"\n  FACTOR IMPORTANCE BY TIMEFRAME DEPTH:")
    print(f"  {'Depth':<12} {'TF':>6} {'Mean |corr|':>12}  {'Max |corr|':>12}  {'# active':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*12}  {'-'*12}  {'-'*10}")
    for d in range(12):
        prefix = TF_LABELS[d]
        depth_factors = [(n, c, a) for n, c, a in results if n.startswith(prefix + '__')]
        if depth_factors:
            abs_corrs = [a for _, _, a in depth_factors]
            active = sum(1 for a in abs_corrs if a >= 0.01)
            print(f"  {prefix:<12} {TF_HIERARCHY[d]:>6} {np.mean(abs_corrs):>12.4f}  "
                  f"{max(abs_corrs):>12.4f}  {active:>10}")

    # Group by feature
    print(f"\n  FACTOR IMPORTANCE BY FEATURE:")
    print(f"  {'Feature':<20} {'Mean |corr|':>12}  {'Max |corr|':>12}  {'Best TF':<12}")
    print(f"  {'-'*20} {'-'*12}  {'-'*12}  {'-'*12}")
    for f_name in FEATURE_NAMES:
        feat_factors = [(n, c, a) for n, c, a in results if n.endswith(f'__{f_name}')]
        if feat_factors:
            abs_corrs = [a for _, _, a in feat_factors]
            best_idx = np.argmax(abs_corrs)
            best_depth = feat_factors[best_idx][0].split('__')[0]
            print(f"  {f_name:<20} {np.mean(abs_corrs):>12.4f}  {max(abs_corrs):>12.4f}  "
                  f"{best_depth:<12}")


# =============================================================================
#  SEED PRIMITIVE LIBRARY (Analysis I)
#
#  12 orthogonal mathematical shapes, normalized 0-1.
#  Categories 1 & 2 get _UP / _DOWN variants (x2) = 16
#  Category 3 (symmetrical) = 4
#  Total: 20 shapes in the dictionary.
# =============================================================================

class SeedPrimitiveLibrary:
    """Library of 20 normalized seed shapes for trajectory classification."""

    CORR_THRESHOLD = 0.75  # minimum Pearson r to classify (below = NOISE)

    def __init__(self, N=16):
        self.N = N
        self.shapes = {}
        self._build(N)

    def _norm01(self, arr):
        """Normalize array to [0, 1]. Returns zeros if flat."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _build(self, N):
        x = np.linspace(0, 1, N)  # normalized time axis

        # --- Category 1: Directional ---
        cat1 = {
            'LINEAR':      x,
            'EXPONENTIAL': x ** 2,
            'LOGARITHMIC': np.log(x + 1),
            'STEP':        np.where(np.arange(N) < N / 2, 0.0, 1.0),
        }

        # --- Category 2: Reversals ---
        cat2 = {
            'SYMMETRIC_V':  np.abs(np.arange(N) - N / 2),
            'ROUNDED_U':    (np.arange(N) - N / 2) ** 2,
            'FRONT_SKEWED': np.exp(-4 * x) - np.exp(-8 * x),
            'BACK_SKEWED':  np.exp(4 * (x - 1)) - np.exp(8 * (x - 1)),
        }

        # Normalize all Cat1 & Cat2 to 0-1, then create UP/DOWN variants
        for name, arr in {**cat1, **cat2}.items():
            normed = self._norm01(arr.astype(float))
            self.shapes[f'{name}_UP'] = normed
            self.shapes[f'{name}_DOWN'] = 1.0 - normed

        # --- Category 3: Volatility (symmetrical, no inversion) ---
        t_cyc = np.linspace(0, 2 * np.pi, N)
        cat3 = {
            'SINE_WAVE':          np.sin(t_cyc),
            'DAMPED_OSCILLATOR':  np.exp(-2 * x) * np.sin(t_cyc),
            'EXPAND_OSCILLATOR':  np.exp(2 * x) * np.sin(t_cyc),
            'FLATLINE':           np.ones(N),
        }
        for name, arr in cat3.items():
            self.shapes[name] = self._norm01(arr.astype(float))

    def classify_trajectory(self, price_segment):
        """Classify a price segment against the 20 seed primitives.

        Args:
            price_segment: raw prices (not pre-normalized)

        Returns:
            (best_shape_name, correlation) or ('NOISE', best_corr)
        """
        seg = np.asarray(price_segment, dtype=float)
        if len(seg) != self.N:
            return 'NOISE', 0.0

        # Normalize input to 0-1
        mn, mx = seg.min(), seg.max()
        if mx - mn < 1e-12:
            return 'FLATLINE', 1.0  # truly flat -> direct match

        normed = (seg - mn) / (mx - mn)

        # Pearson correlation against all 20 shapes
        best_name = 'NOISE'
        best_corr = -999.0

        for name, template in self.shapes.items():
            # Skip zero-variance templates (FLATLINE → all zeros after norm)
            if template.std() < 1e-12:
                continue
            r = np.corrcoef(normed, template)[0, 1]
            if np.isnan(r):
                continue
            if r > best_corr:
                best_corr = r
                best_name = name

        if best_corr < self.CORR_THRESHOLD:
            return 'NOISE', best_corr

        return best_name, best_corr


def _detect_inflections(centroid):
    """Detect inflection points on a centroid (raw ticks or normalized).

    An inflection = where the bar-to-bar direction flips sign.
    Returns list of (bar_idx, level) for each inflection point,
    plus segment descriptors between them.
    """
    d = np.diff(centroid)  # bar-to-bar changes
    inflections = [(0, centroid[0])]  # start point always included

    for i in range(1, len(d)):
        # Sign flip: direction changed
        if d[i] * d[i - 1] < 0:
            inflections.append((i, centroid[i]))

    inflections.append((len(centroid) - 1, centroid[-1]))  # end point

    # Build segment descriptors between inflection points
    segments = []
    for k in range(len(inflections) - 1):
        b0, v0 = inflections[k]
        b1, v1 = inflections[k + 1]
        delta = v1 - v0
        if abs(delta) < 1e-6:
            label = 'HOLD'
        elif delta > 0:
            label = 'RISE'
        else:
            label = 'DROP'
        segments.append({'start': b0, 'end': b1, 'v0': v0, 'v1': v1, 'label': label})

    return inflections, segments


def _adaptive_split(deltas, r2_target=0.80, min_n=2, max_k=48):
    """Find optimal k where all sub-types hit shape R² >= target.

    Clustering uses raw deltas (ticks) so magnitude matters for grouping.
    R² is computed on shape-normalized segments (0-1) so it measures shape
    consistency, not magnitude consistency.

    Tries k=1,2,3,...,max_k. Keeps the k with highest minimum R² across
    all clusters. Stops early if all clusters hit the target.

    Returns (labels, centroids, shape_r2s) — labels[i] = cluster id,
    centroids[k] = raw mean trace, shape_r2s[k] = shape-normalized R².
    """
    from sklearn.cluster import KMeans

    def _shape_r2(sub):
        """Mean Pearson r² between each segment (0-1 normed) and centroid.

        More robust than global R² for small clusters — each segment's
        shape agreement is measured independently then averaged.
        """
        n = len(sub)
        if n < 2:
            return 1.0
        normed_sub = np.zeros_like(sub)
        for i in range(n):
            mn, mx = sub[i].min(), sub[i].max()
            rng = mx - mn
            normed_sub[i] = (sub[i] - mn) / rng if rng > 1e-12 else 0.0
        centroid = normed_sub.mean(axis=0)
        if centroid.std() < 1e-12:
            return 0.0
        r2_vals = []
        for i in range(n):
            if normed_sub[i].std() < 1e-12:
                continue
            r = np.corrcoef(normed_sub[i], centroid)[0, 1]
            if not np.isnan(r):
                r2_vals.append(r ** 2)
        return np.mean(r2_vals) if r2_vals else 0.0

    n_total = len(deltas)

    # Build shape-normalized version for clustering (0-1 per segment)
    normed = np.zeros_like(deltas)
    for i in range(n_total):
        mn, mx = deltas[i].min(), deltas[i].max()
        rng = mx - mn
        normed[i] = (deltas[i] - mn) / rng if rng > 1e-12 else 0.0

    # k=1 baseline (no splitting)
    base_r2 = _shape_r2(deltas)
    best_labels = np.zeros(n_total, dtype=int)
    best_centroids = np.array([deltas.mean(axis=0)])
    best_r2s = np.array([base_r2])
    best_min_r2 = base_r2

    if base_r2 >= r2_target:
        return best_labels, best_centroids, best_r2s

    # Try increasing k — cluster on SHAPE (normalized), report in raw ticks
    k_limit = min(max_k, n_total // min_n)
    for k in range(2, k_limit + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(normed)  # cluster by shape, not magnitude

        # Check all clusters have min_n segments
        valid = True
        r2s = []
        raw_centroids = []
        for ci in range(k):
            mask = (labels == ci)
            n_ci = mask.sum()
            if n_ci < min_n:
                valid = False
                break
            r2s.append(_shape_r2(deltas[mask]))
            raw_centroids.append(deltas[mask].mean(axis=0))

        if not valid:
            continue

        min_r2 = min(r2s)
        if min_r2 > best_min_r2:
            best_labels = labels.copy()
            best_centroids = np.array(raw_centroids)
            best_r2s = np.array(r2s)
            best_min_r2 = min_r2

        if min_r2 >= r2_target:
            break  # all clusters meet target

    # --- Phase 2: targeted bisection of worst clusters ---
    # KMeans finds the best global k, but some clusters may still be below
    # target. Bisect those specifically until they meet R² or hit min_n.
    improved = True
    while improved:
        improved = False
        new_labels = best_labels.copy()
        new_centroids = list(best_centroids)
        new_r2s = list(best_r2s)

        # Find worst cluster that can be split
        worst_ci = -1
        worst_r2 = r2_target
        for ci in range(len(new_centroids)):
            if new_r2s[ci] < worst_r2:
                ci_mask = (new_labels == ci)
                if ci_mask.sum() >= 2 * min_n:
                    worst_ci = ci
                    worst_r2 = new_r2s[ci]

        if worst_ci < 0:
            break  # nothing to split

        ci_mask = (new_labels == worst_ci)
        ci_indices = np.where(ci_mask)[0]
        ci_normed = normed[ci_indices]

        km2 = KMeans(n_clusters=2, random_state=42, n_init=20)
        sub_labels = km2.fit_predict(ci_normed)

        idx_a = ci_indices[sub_labels == 0]
        idx_b = ci_indices[sub_labels == 1]

        if len(idx_a) < min_n or len(idx_b) < min_n:
            # Can't split — mark as final and stop trying this cluster
            break

        r2_a = _shape_r2(deltas[idx_a])
        r2_b = _shape_r2(deltas[idx_b])

        # Only accept if BOTH halves improve over the original
        if min(r2_a, r2_b) > worst_r2:
            new_id = len(new_centroids)
            new_labels[idx_a] = worst_ci
            new_labels[idx_b] = new_id
            new_centroids[worst_ci] = deltas[idx_a].mean(axis=0)
            new_centroids.append(deltas[idx_b].mean(axis=0))
            new_r2s[worst_ci] = r2_a
            new_r2s.append(r2_b)

            best_labels = new_labels
            best_centroids = np.array(new_centroids)
            best_r2s = np.array(new_r2s)
            best_min_r2 = min(best_r2s)
            improved = True

    return best_labels, best_centroids, best_r2s


# =============================================================================
#  MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Standalone waveform screening from raw ATLAS data')
    parser.add_argument('--data', default='DATA/ATLAS',
                        help='ATLAS data directory (default: DATA/ATLAS)')
    parser.add_argument('--months', nargs='+', default=None,
                        help='Specific months to load (e.g., 2025_01 2025_02)')
    parser.add_argument('--base-tf', default='15m',
                        help='Base timeframe for analysis points (default: 15m)')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup days before analysis window (default: 21)')
    parser.add_argument('--analysis-days', type=int, default=7,
                        help='Analysis window in days (0 = all remaining, default: 7)')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top factors to display')
    parser.add_argument('--full', action='store_true',
                        help='Run full 16D fractal pipeline (physics + hypervolumes)')
    args = parser.parse_args()

    # Resolve plots dir based on data path
    sample_label = _resolve_plots_dir(args.data)

    # Capture all output to report file
    _report_buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _report_buf)

    print(f"{'='*70}")
    print(f"  STANDALONE WAVEFORM SCREENING")
    print(f"  Data: {args.data}")
    print(f"  Base TF: {args.base_tf}")
    print(f"  Context: {args.context_days}d warmup, {args.analysis_days}d analysis")
    print(f"  Mode: {'FULL (16D fractal)' if args.full else 'PRICE I-MR (default)'}")
    print(f"{'='*70}")

    # =====================================================================
    #  STEP 1: Load base TF data
    # =====================================================================
    print(f"\n--- STEP 1: Loading base TF data ({args.base_tf}) ---")
    base_df = load_atlas_tf(args.data, args.base_tf, months=args.months)
    if base_df.empty:
        print(f"ERROR: Base TF '{args.base_tf}' has no data in {args.data}")
        sys.exit(1)
    print(f"  {args.base_tf}: {len(base_df):,} bars loaded")

    # =====================================================================
    #  STEP 2: Price I-MR chart (pure price/time, no physics)
    # =====================================================================
    print(f"\n--- STEP 2: Price I-MR Chart ---")
    price_imr = compute_price_imr(base_df, args.context_days, args.analysis_days)

    # =====================================================================
    #  STEP 3: Detect regimes from MR UCL breaks
    # =====================================================================
    print(f"\n--- STEP 3: Regime Detection ---")
    regime_ids, regime_meta = detect_regimes(price_imr)

    # =====================================================================
    #  STEP 4: Oracle MFE/MAE with regime-based direction
    # =====================================================================
    print(f"\n--- STEP 4: Oracle MFE/MAE ---")
    lookahead = ORACLE_LOOKAHEAD_BARS.get(args.base_tf, 16)
    bar_indices, mfes, maes, directions = compute_regime_oracle(
        base_df, regime_ids, regime_meta, lookahead=lookahead)

    if len(mfes) < 10:
        print(f"ERROR: Only {len(mfes)} oracle bars (need >= 10)")
        sys.exit(1)

    # =====================================================================
    #  STEP 5: Price I-MR plot + Regime summary
    # =====================================================================
    print(f"\n--- STEP 5: Generating charts ---")
    plot_price_imr(price_imr, regime_ids, regime_meta, base_df)
    plot_regime_summary(regime_meta, mfes, maes, bar_indices, regime_ids)

    # =====================================================================
    #  STEP 6: Print regime summary table
    # =====================================================================

    print(f"\n{'='*70}")
    print(f"  REGIME SUMMARY")
    print(f"{'='*70}")
    print(f"  Analysis bars: {len(mfes)}")
    print(f"  Regimes: {len(regime_meta)}")
    print(f"  MFE: mean={np.mean(mfes):.2f}, std={np.std(mfes):.2f}")
    print(f"  MAE: mean={np.mean(maes):.2f}, std={np.std(maes):.2f}")
    print(f"  Win rate (MFE > MAE): {(mfes > maes).mean():.1%}")

    print(f"\n  {'Regime':<10} {'Dir':<6} {'N':>5} {'WR':>7} {'MFE':>8} "
          f"{'MAE':>8} {'Vol':>6} {'Price':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*7} {'-'*8} "
          f"{'-'*8} {'-'*6} {'-'*10}")

    bar_regime = regime_ids[bar_indices]
    for rm in regime_meta:
        rid = rm['regime_id']
        mask = bar_regime == rid
        if mask.sum() == 0:
            continue
        wr = float((mfes[mask] > maes[mask]).mean())
        mean_mfe = float(np.mean(mfes[mask]))
        mean_mae = float(np.mean(maes[mask]))
        print(f"  R{rid:<9} {rm['direction']:<6} {mask.sum():>5} {wr:>7.1%} "
              f"{mean_mfe:>+8.1f} {mean_mae:>8.1f} {rm['volatility']:>6.2f} "
              f"{rm['mean_price']:>10.1f}")

    # Directional split
    long_mask = np.array([d == 'LONG' for d in directions])
    short_mask = ~long_mask
    n_long = long_mask.sum()
    n_short = short_mask.sum()
    wr_long = float((mfes[long_mask] > maes[long_mask]).mean()) if n_long > 0 else 0
    wr_short = float((mfes[short_mask] > maes[short_mask]).mean()) if n_short > 0 else 0

    print(f"\n  DIRECTIONAL SPLIT:")
    print(f"  LONG:  {n_long:>5} bars, WR={wr_long:.1%}")
    print(f"  SHORT: {n_short:>5} bars, WR={wr_short:.1%}")

    # =====================================================================
    #  STEP 7: Load all 12 TFs + compute fractal context
    # =====================================================================
    print(f"\n--- STEP 7: Loading all TF data + fractal context ---")
    all_dfs = {args.base_tf: base_df}
    for tf in TF_HIERARCHY:
        if tf == args.base_tf:
            continue
        df = load_atlas_tf(args.data, tf, months=args.months)
        if not df.empty:
            all_dfs[tf] = df
            print(f"  {tf:>4}: {len(df):>8,} bars")
        else:
            print(f"  {tf:>4}:   (not found)")

    print(f"\n  Computing physics per TF...")
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="Physics", unit="tf"):
        if tf not in all_dfs:
            continue
        states = compute_tf_physics(tf, all_dfs[tf])
        if states:
            all_tf_states[tf] = states
            print(f"  {tf:>4}: {len(states):>8,} states computed")

    # =====================================================================
    #  STEP 8: Build X (fractal context + current MR) for each bar
    #
    #  X = 192 fractal features at time t + signed MR[t] = 193 context features
    #  Two Y targets:
    #    Y_price     = close[t]              (can we explain the price?)
    #    Y_direction = sign(close[t+1]-close[t])  (can we explain the direction?)
    # =====================================================================
    print(f"\n--- STEP 8: Building context matrix (193 features per bar) ---")

    analysis_idx = np.where(regime_ids >= 0)[0]
    timestamps = base_df['timestamp'].values.astype(float)
    close = base_df['close'].values.astype(float)
    mr_signed = price_imr['mr']

    # Pre-sort timestamps for each TF (for binary search alignment)
    tf_sorted_ts = {}
    for tf in TF_HIERARCHY:
        if tf in all_tf_states and all_tf_states[tf]:
            tf_sorted_ts[tf] = np.array(sorted(all_tf_states[tf].keys()))

    X_rows = []
    X_delta_rows = []  # rate-of-change features
    Y_price = []
    Y_direction = []
    sample_ts = []
    base_secs = TF_SECONDS.get(args.base_tf, 900)

    def _build_mat(t):
        """Build (12,16) fractal fingerprint at timestamp t."""
        mat = np.zeros((12, 16))
        n = 0
        for depth_idx, tf in enumerate(TF_HIERARCHY):
            if tf not in tf_sorted_ts:
                continue
            tf_ts_list = tf_sorted_ts[tf]
            tf_secs = TF_SECONDS.get(tf, 60)
            if tf_secs > base_secs:
                pos = np.searchsorted(tf_ts_list, t, side='right') - 2
            else:
                pos = np.searchsorted(tf_ts_list, t, side='right') - 1
            if pos < 0:
                continue
            nearest_ts = tf_ts_list[pos]
            state = all_tf_states[tf][nearest_ts]
            mat[depth_idx, :] = extract_16d(state, tf)
            n += 1
        return mat, n

    n_bars = len(close)
    for idx in tqdm(analysis_idx, desc="Fractal context", unit="bar"):
        if idx + 1 >= n_bars or idx < 1:
            continue

        t = int(timestamps[idx])
        t_prev = int(timestamps[idx - 1])
        current_mr = mr_signed[idx]

        mat, has_data = _build_mat(t)
        if has_data < 3:
            continue

        # Build previous bar's matrix for rate-of-change
        mat_prev, has_prev = _build_mat(t_prev)
        if has_prev < 3:
            delta = np.zeros_like(mat)
        else:
            delta = mat - mat_prev

        x_row = np.concatenate([mat.flatten(), [current_mr]])  # 193 features
        x_delta = delta.flatten()  # 192 delta features
        X_rows.append(x_row)
        X_delta_rows.append(x_delta)
        Y_price.append(close[idx])
        next_change = close[idx + 1] - close[idx]
        Y_direction.append(1.0 if next_change > 0 else (-1.0 if next_change < 0 else 0.0))
        sample_ts.append(t)

    X = np.array(X_rows)
    X_delta = np.array(X_delta_rows)
    Y_p = np.array(Y_price)
    Y_d = np.array(Y_direction)
    print(f"  Samples: {len(Y_p)}, Level features: {X.shape[1] if len(X) > 0 else 0}, "
          f"Delta features: {X_delta.shape[1] if len(X_delta) > 0 else 0}")
    print(f"  X: 192 fractal + 1 current MR = 193 level features")
    print(f"  X_delta: 192 rate-of-change (feature[t] - feature[t-1])")

    # Build column names: 192 fractal + 1 current MR
    col_names = []
    for d in range(12):
        tf_lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        for f in range(16):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            col_names.append(f"{tf_lbl}__{f_lbl}")
    col_names.append("current_MR")

    # Delta column names
    delta_col_names = []
    for d in range(12):
        tf_lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        for f in range(16):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            delta_col_names.append(f"dt_{tf_lbl}__{f_lbl}")

    # =====================================================================
    #  ANALYSIS A: PRICE EXPLANATION (independent)
    #
    #  Y = close[t] -- the actual price level
    #  Question: does the fractal fingerprint describe WHERE price is?
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS A: PRICE EXPLANATION")
    print(f"  Y = close[t] at each bar")
    print(f"  Y range: {Y_p.min():.1f} to {Y_p.max():.1f}, "
          f"mean={Y_p.mean():.1f}, std={Y_p.std():.1f}")
    print(f"  Samples: {len(Y_p)}")
    print(f"{'='*70}")

    results_price = screen_factors(X, col_names, Y_p)

    print(f"\n  TOP 20 FACTORS (correlation with price):")
    print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results_price[:20], 1):
        bar = '#' * int(abs_corr * 50)
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

    print(f"\n  Stepwise regression: context -> price")
    result_a = regression_r2(X, col_names, Y_p, top_k=20, return_model=True)
    steps_price, (price_model, price_scaler, price_feat_idx) = result_a
    r2_p = steps_price[-1][3] if steps_price else 0
    print(f"\n  >> PRICE adj-R2 = {r2_p:.4f}")
    print(f"  >> Context explains {r2_p*100:.1f}% of price variance")

    # Price: BY TIMEFRAME
    print(f"\n  PRICE BY TIMEFRAME:")
    print(f"  {'Depth':<12} {'TF':>6} {'Mean |r|':>10} {'Max |r|':>10} {'Top Factor':<35}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*35}")
    for d in range(12):
        prefix = TF_LABELS[d]
        pf = [(n, c, a) for n, c, a in results_price if n.startswith(prefix + '__')]
        if pf:
            abs_vals = [a for _, _, a in pf]
            mp = np.mean(abs_vals)
            mx = max(abs_vals)
            best = max(pf, key=lambda x: x[2])
            print(f"  {prefix:<12} {TF_HIERARCHY[d]:>6} {mp:>10.4f} {mx:>10.4f} {best[0]:<35}")

    # Price: BY FEATURE
    print(f"\n  PRICE BY FEATURE:")
    print(f"  {'Feature':<20} {'Mean |r|':>10} {'Max |r|':>10} {'Best TF':<15}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15}")
    for f_name in FEATURE_NAMES:
        pf = [(n, c, a) for n, c, a in results_price if n.endswith(f'__{f_name}')]
        if pf:
            abs_vals = [a for _, _, a in pf]
            mp = np.mean(abs_vals)
            mx = max(abs_vals)
            best = max(pf, key=lambda x: x[2])
            best_tf = best[0].split('__')[0]
            print(f"  {f_name:<20} {mp:>10.4f} {mx:>10.4f} {best_tf:<15}")
    mr_r_p = next((a for n, c, a in results_price if n == 'current_MR'), 0)
    print(f"  {'current_MR':<20} {mr_r_p:>10.4f} {mr_r_p:>10.4f} {'(base TF)':<15}")

    # Price: CONCLUSION
    print(f"\n  PRICE CONCLUSION:")
    if r2_p > 0.80:
        print(f"  Strong: adj-R2 = {r2_p:.4f}. The fractal context reliably describes")
        print(f"  WHERE price is. The 193 features map to price level with high fidelity.")
    elif r2_p > 0.30:
        print(f"  Moderate: adj-R2 = {r2_p:.4f}. Context captures price structure")
        print(f"  but with meaningful residual noise.")
    else:
        print(f"  Weak: adj-R2 = {r2_p:.4f}. Context does not reliably explain price level.")

    # =====================================================================
    #  ANALYSIS B: DIRECTION EXPLANATION (independent)
    #
    #  Y = sign(close[t+1] - close[t]) -- will price go up or down?
    #  Question: does the fractal fingerprint tell us which way price moves?
    # =====================================================================
    # Build direction matrix: X + price anchor = 194 features
    X_dir = np.column_stack([X, Y_p])  # add close[t] as anchor
    col_names_dir = col_names + ['price_anchor']

    n_up = (Y_d > 0).sum()
    n_down = (Y_d < 0).sum()
    n_flat = (Y_d == 0).sum()

    print(f"\n{'='*70}")
    print(f"  ANALYSIS B: DIRECTION EXPLANATION (with price anchor)")
    print(f"  Y = sign(next change): +1=up, -1=down")
    print(f"  X = 192 fractal + current_MR + price[t] = {X_dir.shape[1]} features")
    print(f"  Distribution: {n_up} up ({n_up/len(Y_d):.0%}), "
          f"{n_down} down ({n_down/len(Y_d):.0%}), {n_flat} flat")
    print(f"  Samples: {len(Y_d)}")
    print(f"{'='*70}")

    results_dir = screen_factors(X_dir, col_names_dir, Y_d)

    print(f"\n  TOP 20 FACTORS (correlation with direction):")
    print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results_dir[:20], 1):
        bar = '#' * int(abs_corr * 50)
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

    print(f"\n  Stepwise regression: context + anchor -> direction")
    steps_dir = regression_r2(X_dir, col_names_dir, Y_d, top_k=20)
    r2_d = steps_dir[-1][3] if steps_dir else 0
    print(f"\n  >> DIRECTION adj-R2 = {r2_d:.4f}")
    print(f"  >> Context explains {r2_d*100:.1f}% of direction variance")

    # Direction: BY TIMEFRAME
    print(f"\n  DIRECTION BY TIMEFRAME:")
    print(f"  {'Depth':<12} {'TF':>6} {'Mean |r|':>10} {'Max |r|':>10} {'Top Factor':<35}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*35}")
    for d in range(12):
        prefix = TF_LABELS[d]
        df = [(n, c, a) for n, c, a in results_dir if n.startswith(prefix + '__')]
        if df:
            abs_vals = [a for _, _, a in df]
            md = np.mean(abs_vals)
            mx = max(abs_vals)
            best = max(df, key=lambda x: x[2])
            print(f"  {prefix:<12} {TF_HIERARCHY[d]:>6} {md:>10.4f} {mx:>10.4f} {best[0]:<35}")

    # Direction: BY FEATURE
    print(f"\n  DIRECTION BY FEATURE:")
    print(f"  {'Feature':<20} {'Mean |r|':>10} {'Max |r|':>10} {'Best TF':<15}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15}")
    for f_name in FEATURE_NAMES:
        df = [(n, c, a) for n, c, a in results_dir if n.endswith(f'__{f_name}')]
        if df:
            abs_vals = [a for _, _, a in df]
            md = np.mean(abs_vals)
            mx = max(abs_vals)
            best = max(df, key=lambda x: x[2])
            best_tf = best[0].split('__')[0]
            print(f"  {f_name:<20} {md:>10.4f} {mx:>10.4f} {best_tf:<15}")
    mr_r_d = next((a for n, c, a in results_dir if n == 'current_MR'), 0)
    print(f"  {'current_MR':<20} {mr_r_d:>10.4f} {mr_r_d:>10.4f} {'(base TF)':<15}")

    # Direction: sign analysis — are correlations mostly negative?
    top20_signs = [c for _, c, _ in results_dir[:20]]
    n_neg = sum(1 for s in top20_signs if s < 0)
    n_pos = sum(1 for s in top20_signs if s > 0)
    print(f"\n  SIGN PATTERN: {n_neg}/20 top factors have NEGATIVE correlation")
    if n_neg > 14:
        print(f"  Most direction factors point to MEAN REVERSION -- higher feature")
        print(f"  values predict DOWN moves, suggesting overbought/overextended states.")
    elif n_pos > 14:
        print(f"  Most direction factors point to TREND CONTINUATION -- higher")
        print(f"  feature values predict UP moves, suggesting momentum persistence.")
    else:
        print(f"  Mixed signs -- no dominant directional bias in the features.")

    # Direction: CONCLUSION
    print(f"\n  DIRECTION CONCLUSION:")
    if r2_d > 0.15:
        print(f"  Useful: adj-R2 = {r2_d:.4f}. The fractal context carries meaningful")
        print(f"  directional signal. Worth building a directional model from these features.")
    elif r2_d > 0.05:
        print(f"  Weak but present: adj-R2 = {r2_d:.4f}. Some directional signal exists")
        print(f"  but it is fragile. May need more data, different features, or")
        print(f"  non-linear methods to extract it reliably.")
    else:
        print(f"  Insufficient: adj-R2 = {r2_d:.4f}. The fractal context does not")
        print(f"  reliably explain direction. The next bar is essentially unpredictable")
        print(f"  from these features alone.")

    # =====================================================================
    #  ANALYSIS C: DIRECTION FROM PRICE MODEL (derived)
    #
    #  Since we can explain price (A=95%) but not direction standalone (B=8.7%),
    #  can we derive direction from consecutive price predictions?
    #  predicted_dir = sign( predict(features[t+1]) - predict(features[t]) )
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS C: DIRECTION DERIVED FROM PRICE MODEL")
    print(f"  If price model predicts price[t] and price[t+1],")
    print(f"  direction = sign(predicted[t+1] - predicted[t])")
    print(f"{'='*70}")

    # Predict price for every sample using the price model from Analysis A
    X_price_feat = price_scaler.transform(X[:, price_feat_idx])
    predicted_prices = price_model.predict(X_price_feat)

    # Build consecutive pairs: predicted[t] vs predicted[t+1]
    n_pairs = len(predicted_prices) - 1
    pred_dir = np.sign(predicted_prices[1:] - predicted_prices[:-1])
    actual_dir = Y_d[:-1]  # actual direction at each t (already sign of close[t+1]-close[t])

    # Filter out flat actuals
    mask = actual_dir != 0
    pred_dir_f = pred_dir[mask]
    actual_dir_f = actual_dir[mask]
    n_valid = mask.sum()

    correct = (pred_dir_f == actual_dir_f).sum()
    accuracy = correct / n_valid if n_valid > 0 else 0

    print(f"\n  Pairs: {n_pairs}, Valid (non-flat): {n_valid}")
    print(f"  Predicted direction accuracy: {correct}/{n_valid} = {accuracy:.1%}")

    # Breakdown by actual direction
    up_mask = actual_dir_f > 0
    down_mask = actual_dir_f < 0
    up_correct = (pred_dir_f[up_mask] > 0).sum() if up_mask.sum() > 0 else 0
    down_correct = (pred_dir_f[down_mask] < 0).sum() if down_mask.sum() > 0 else 0
    print(f"\n  When actual UP:   {up_correct}/{up_mask.sum()} = "
          f"{up_correct/up_mask.sum():.1%}" if up_mask.sum() > 0 else "")
    print(f"  When actual DOWN: {down_correct}/{down_mask.sum()} = "
          f"{down_correct/down_mask.sum():.1%}" if down_mask.sum() > 0 else "")

    # Residual analysis: how big are the prediction errors vs actual moves?
    residuals = predicted_prices - Y_p
    actual_moves = np.diff(Y_p)
    print(f"\n  Price model residuals: mean={residuals.mean():.2f}, std={residuals.std():.2f}")
    print(f"  Actual bar-to-bar moves: mean={np.mean(np.abs(actual_moves)):.2f}, "
          f"std={np.std(actual_moves):.2f}")
    snr = np.mean(np.abs(actual_moves)) / residuals.std() if residuals.std() > 0 else 0
    print(f"  Signal-to-noise ratio: {snr:.3f} "
          f"({'good' if snr > 1.5 else 'marginal' if snr > 0.8 else 'poor'}: "
          f"{'moves > noise' if snr > 1 else 'noise > moves'})")

    # Confidence: only count predictions where delta is large enough
    pred_deltas = predicted_prices[1:] - predicted_prices[:-1]
    for threshold in [0.0, 5.0, 10.0, 20.0]:
        conf_mask = (np.abs(pred_deltas) > threshold) & mask
        if conf_mask.sum() > 0:
            conf_correct = (pred_dir[conf_mask] == actual_dir[conf_mask]).sum()
            conf_acc = conf_correct / conf_mask.sum()
            print(f"  |predicted delta| > {threshold:>5.1f}: "
                  f"{conf_correct}/{conf_mask.sum()} = {conf_acc:.1%}")

    # CONCLUSION
    print(f"\n  DERIVED DIRECTION CONCLUSION:")
    if accuracy > 0.60:
        print(f"  Promising: {accuracy:.1%} accuracy. Deriving direction from consecutive")
        print(f"  price predictions works better than standalone direction modeling.")
        if snr > 1.0:
            print(f"  Signal-to-noise is favorable ({snr:.2f}) -- moves are larger than")
            print(f"  prediction residuals.")
        else:
            print(f"  However, signal-to-noise is low ({snr:.2f}) -- may improve with")
            print(f"  more data or better price features.")
    elif accuracy > 0.52:
        print(f"  Marginal: {accuracy:.1%} accuracy. Slightly better than chance but")
        print(f"  not reliable enough. The price model's residual noise ({residuals.std():.1f})")
        print(f"  is {'larger' if snr < 1 else 'comparable to'} the typical move ({np.mean(np.abs(actual_moves)):.1f}).")
    else:
        print(f"  No improvement: {accuracy:.1%}. The price model's residuals overwhelm")
        print(f"  the bar-to-bar signal. 95% R2 on level does not translate to")
        print(f"  directional accuracy at this resolution.")

    # =====================================================================
    #  ANALYSIS D: DOES RATE-OF-CHANGE IMPROVE PRICE & DIRECTION?
    #
    #  Add delta features (feature[t] - feature[t-1]) to test if
    #  temporal pattern recognition helps beyond the spatial snapshot.
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS D: RATE-OF-CHANGE (PATTERN RECOGNITION)")
    print(f"  Level = 193 features (snapshot at t)")
    print(f"  Delta = 192 features (change from t-1 to t)")
    print(f"  Combined = {193 + 192} features (level + delta)")
    print(f"{'='*70}")

    # Combine level + delta features
    X_combined = np.column_stack([X, X_delta])
    combined_col_names = col_names + delta_col_names

    # D1: Does delta help PRICE explanation?
    print(f"\n  D1: PRICE with level+delta features")
    print(f"  Stepwise regression: {X_combined.shape[1]} features -> price")
    steps_price_d = regression_r2(X_combined, combined_col_names, Y_p, top_k=20)
    r2_pd = steps_price_d[-1][3] if steps_price_d else 0
    print(f"\n  >> PRICE adj-R2 (level only):  {r2_p:.4f}")
    print(f"  >> PRICE adj-R2 (level+delta): {r2_pd:.4f}")
    print(f"  >> Delta contribution: {r2_pd - r2_p:+.4f} ({(r2_pd - r2_p)*100:+.1f}%)")

    # How many delta features made it into the model?
    n_delta_in_price = sum(1 for step in steps_price_d if step[0].startswith('dt_'))
    print(f"  >> Delta features in price model: {n_delta_in_price}/{len(steps_price_d)}")

    # Top delta features for price
    price_d_results = screen_factors(X_delta, delta_col_names, Y_p)
    print(f"\n  TOP 10 DELTA FACTORS for price:")
    print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(price_d_results[:10], 1):
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}")

    # D2: Does delta help DIRECTION explanation?
    print(f"\n  D2: DIRECTION with level+delta features")
    X_dir_combined = np.column_stack([X, X_delta, Y_p])  # add price anchor too
    dir_combined_names = col_names + delta_col_names + ['price_anchor']
    print(f"  Stepwise regression: {X_dir_combined.shape[1]} features -> direction")
    steps_dir_d = regression_r2(X_dir_combined, dir_combined_names, Y_d, top_k=20)
    r2_dd = steps_dir_d[-1][3] if steps_dir_d else 0
    print(f"\n  >> DIRECTION adj-R2 (level only):  {r2_d:.4f}")
    print(f"  >> DIRECTION adj-R2 (level+delta): {r2_dd:.4f}")
    print(f"  >> Delta contribution: {r2_dd - r2_d:+.4f} ({(r2_dd - r2_d)*100:+.1f}%)")

    n_delta_in_dir = sum(1 for step in steps_dir_d if step[0].startswith('dt_'))
    print(f"  >> Delta features in direction model: {n_delta_in_dir}/{len(steps_dir_d)}")

    # Top delta features for direction
    dir_d_results = screen_factors(X_delta, delta_col_names, Y_d)
    print(f"\n  TOP 10 DELTA FACTORS for direction:")
    print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(dir_d_results[:10], 1):
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}")

    # D3: Derived direction from combined price model
    print(f"\n  D3: DERIVED DIRECTION from level+delta price model")
    result_pd = regression_r2(X_combined, combined_col_names, Y_p, top_k=20, return_model=True)
    steps_pd, (pd_model, pd_scaler, pd_feat_idx) = result_pd
    X_pd_feat = pd_scaler.transform(X_combined[:, pd_feat_idx])
    pred_prices_d = pd_model.predict(X_pd_feat)

    pred_dir_d = np.sign(pred_prices_d[1:] - pred_prices_d[:-1])
    actual_dir_d = Y_d[:-1]
    mask_d = actual_dir_d != 0
    correct_d = (pred_dir_d[mask_d] == actual_dir_d[mask_d]).sum()
    n_valid_d = mask_d.sum()
    accuracy_d = correct_d / n_valid_d if n_valid_d > 0 else 0

    print(f"  Derived direction (level only):  {accuracy:.1%}")
    print(f"  Derived direction (level+delta): {accuracy_d:.1%}")
    print(f"  Delta contribution: {accuracy_d - accuracy:+.1%}")

    # Confidence gates for combined model
    pred_deltas_d = pred_prices_d[1:] - pred_prices_d[:-1]
    for threshold in [0.0, 5.0, 10.0, 20.0]:
        conf_mask = (np.abs(pred_deltas_d) > threshold) & mask_d
        if conf_mask.sum() > 0:
            conf_correct = (pred_dir_d[conf_mask] == actual_dir_d[conf_mask]).sum()
            conf_acc = conf_correct / conf_mask.sum()
            print(f"  |predicted delta| > {threshold:>5.1f}: "
                  f"{conf_correct}/{conf_mask.sum()} = {conf_acc:.1%}")

    # ANALYSIS D CONCLUSION
    print(f"\n  ANALYSIS D CONCLUSION:")
    price_gain = r2_pd - r2_p
    dir_gain = r2_dd - r2_d
    derived_gain = accuracy_d - accuracy
    if price_gain > 0.01 or dir_gain > 0.01 or derived_gain > 0.03:
        print(f"  Pattern recognition HELPS:")
        if price_gain > 0.01:
            print(f"    Price R2: {r2_p:.4f} -> {r2_pd:.4f} (+{price_gain:.4f})")
        if dir_gain > 0.01:
            print(f"    Direction R2: {r2_d:.4f} -> {r2_dd:.4f} (+{dir_gain:.4f})")
        if derived_gain > 0.03:
            print(f"    Derived accuracy: {accuracy:.1%} -> {accuracy_d:.1%} (+{derived_gain:.1%})")
    else:
        print(f"  Rate-of-change features do NOT meaningfully improve results.")
        print(f"    Price R2:  {r2_p:.4f} -> {r2_pd:.4f} ({price_gain:+.4f})")
        print(f"    Dir R2:    {r2_d:.4f} -> {r2_dd:.4f} ({dir_gain:+.4f})")
        print(f"    Derived:   {accuracy:.1%} -> {accuracy_d:.1%} ({derived_gain:+.1%})")
        print(f"  The spatial snapshot already captures what matters. Adding temporal")
        print(f"  deltas does not reveal hidden directional signal.")

    # =====================================================================
    #  ANALYSIS E: dP/dT-GROUPED DIRECTION (signal amplification)
    #
    #  Group bars by signed dP/dT (= signed MR = close[t] - close[t-1]).
    #  Within each group, bars have similar price behavior, preventing
    #  signal dilution from mixing different market characters.
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS E: dP/dT-GROUPED ANALYSIS (SIGNAL AMPLIFICATION)")
    print(f"  Group bars by signed price change rate, run Three Questions per group.")
    print(f"  Hypothesis: homogeneous groups prevent signal dilution.")
    print(f"{'='*70}")

    # current_MR is the last column of X (index -1), which is signed dP/dT
    sample_mr = X[:, -1]  # signed MR for each sample

    # Bin into groups by signed MR: DOWN / FLAT / UP (terciles)
    # Use 3 bins to keep groups large enough for regression
    bin_edges = np.percentile(sample_mr, [33, 67])
    bin_labels = ['DOWN', 'FLAT', 'UP']
    bin_ids = np.digitize(sample_mr, bin_edges)  # 0-2

    print(f"\n  dP/dT bins (quintiles of signed MR):")
    print(f"  {'Bin':<15} {'Range':>20} {'N':>6} {'Mean MR':>10}")
    print(f"  {'-'*15} {'-'*20} {'-'*6} {'-'*10}")
    n_bins = len(bin_labels)
    for b in range(n_bins):
        mask_b = bin_ids == b
        n_b = mask_b.sum()
        if n_b > 0:
            mr_b = sample_mr[mask_b]
            print(f"  {bin_labels[b]:<15} [{mr_b.min():>+8.1f}, {mr_b.max():>+8.1f}] {n_b:>6} {mr_b.mean():>+10.2f}")

    # Run Three Questions per bin
    print(f"\n  PER-GROUP RESULTS:")
    print(f"  {'Bin':<15} {'N':>5} {'Price R2':>10} {'Dir R2':>10} {'Derived':>10} {'Dir>20':>10}")
    print(f"  {'-'*15} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    group_results = []
    for b in range(n_bins):
        mask_b = bin_ids == b
        n_b = mask_b.sum()
        if n_b < 20:  # need minimum samples for regression
            print(f"  {bin_labels[b]:<15} {n_b:>5}   (too few samples)")
            group_results.append((bin_labels[b], n_b, 0, 0, 0, 0))
            continue

        X_b = X[mask_b]
        Y_p_b = Y_p[mask_b]
        Y_d_b = Y_d[mask_b]

        # A: Price R2 per group
        steps_p_b = regression_r2(X_b, col_names, Y_p_b, top_k=20)
        r2_p_b = steps_p_b[-1][3] if steps_p_b else 0

        # B: Direction R2 per group
        X_dir_b = np.column_stack([X_b, Y_p_b])
        steps_d_b = regression_r2(X_dir_b, col_names + ['price_anchor'], Y_d_b, top_k=20)
        r2_d_b = steps_d_b[-1][3] if steps_d_b else 0

        # C: Derived direction per group
        result_b = regression_r2(X_b, col_names, Y_p_b, top_k=20, return_model=True)
        steps_b, (model_b, scaler_b, feat_idx_b) = result_b
        X_feat_b = scaler_b.transform(X_b[:, feat_idx_b])
        pred_p_b = model_b.predict(X_feat_b)

        pred_dir_b = np.sign(pred_p_b[1:] - pred_p_b[:-1])
        actual_dir_b = Y_d_b[:-1]
        mask_nf = actual_dir_b != 0
        n_valid_b = mask_nf.sum()
        if n_valid_b > 0:
            correct_b = (pred_dir_b[mask_nf] == actual_dir_b[mask_nf]).sum()
            acc_b = correct_b / n_valid_b
        else:
            acc_b = 0

        # Confidence gate >20
        pred_deltas_b = pred_p_b[1:] - pred_p_b[:-1]
        conf20_mask = (np.abs(pred_deltas_b) > 20) & mask_nf
        if conf20_mask.sum() > 5:
            conf20_acc = (pred_dir_b[conf20_mask] == actual_dir_b[conf20_mask]).sum() / conf20_mask.sum()
            conf20_str = f"{conf20_acc:.1%}({conf20_mask.sum()})"
        else:
            conf20_acc = 0
            conf20_str = "n/a"

        print(f"  {bin_labels[b]:<15} {n_b:>5} {r2_p_b:>10.4f} {r2_d_b:>10.4f} {acc_b:>9.1%} {conf20_str:>10}")
        group_results.append((bin_labels[b], n_b, r2_p_b, r2_d_b, acc_b, conf20_acc))

    # Compare vs global
    print(f"\n  {'GLOBAL':<15} {len(Y_p):>5} {r2_p:>10.4f} {r2_d:>10.4f} {accuracy:>9.1%}")

    # Summary statistics
    valid_groups = [(lbl, n, rp, rd, da, c20) for lbl, n, rp, rd, da, c20 in group_results if n >= 20]
    if valid_groups:
        avg_dir_r2 = np.mean([rd for _, _, _, rd, _, _ in valid_groups])
        avg_derived = np.mean([da for _, _, _, _, da, _ in valid_groups])
        best_group = max(valid_groups, key=lambda x: x[3])
        worst_group = min(valid_groups, key=lambda x: x[3])

        print(f"\n  ANALYSIS E CONCLUSION:")
        print(f"  Average per-group direction R2: {avg_dir_r2:.4f} (vs global {r2_d:.4f})")
        print(f"  Average per-group derived dir:  {avg_derived:.1%} (vs global {accuracy:.1%})")
        print(f"  Best group:  {best_group[0]} (dir R2={best_group[3]:.4f}, derived={best_group[4]:.1%})")
        print(f"  Worst group: {worst_group[0]} (dir R2={worst_group[3]:.4f}, derived={worst_group[4]:.1%})")

        if avg_dir_r2 > r2_d * 1.5:
            print(f"\n  SIGNAL AMPLIFICATION CONFIRMED: grouping by dP/dT improves")
            print(f"  direction R2 by {avg_dir_r2/max(r2_d,0.001):.1f}x on average.")
            print(f"  Homogeneous groups preserve directional signal that drowns")
            print(f"  in the global model. This validates the clustering approach.")
        else:
            print(f"\n  Grouping by dP/dT does not significantly amplify the signal.")
            print(f"  The direction problem may be fundamental, not a grouping issue.")

    # =====================================================================
    #  ANALYSIS F: REGIME SIGNATURE PLOT
    #
    #  Like fractal dimension vs SNR plots: each regime gets ONE mean
    #  trajectory line on a shared chart. Shapes normalized to entry=0
    #  so we compare MOVEMENT, not price level. Separation between
    #  regime lines = clustering captures distinct behavior.
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS F: REGIME SIGNATURE PLOT")
    print(f"{'='*70}")

    lookback_bars = 8    # 8 bars before entry (2 hours at 15m)
    lookahead_bars = 16  # 16 bars after entry (4 hours at 15m)
    shape_len = lookback_bars + 1 + lookahead_bars  # 25 total points

    # Map sample timestamps back to base_df indices
    _ts_col = timestamps.astype(int)
    _ts_to_idx = {int(t): i for i, t in enumerate(_ts_col)}
    sample_indices = []
    for ts in sample_ts:
        if ts in _ts_to_idx:
            sample_indices.append(_ts_to_idx[ts])
        else:
            sample_indices.append(-1)

    # Collect shapes per regime (normalized to entry=0)
    shapes_by_regime = {rm['regime_id']: [] for rm in regime_meta}
    raw_shapes_by_regime = {rm['regime_id']: [] for rm in regime_meta}
    outcomes_by_regime = {rm['regime_id']: [] for rm in regime_meta}

    for i, bar_idx in enumerate(sample_indices):
        if bar_idx < 0 or bar_idx < lookback_bars:
            continue
        if bar_idx + lookahead_bars >= len(close):
            continue

        rid = regime_ids[bar_idx]
        if rid < 0:
            continue  # warmup bar

        # Extract price shape: lookback + entry + lookahead
        shape_raw = close[bar_idx - lookback_bars : bar_idx + lookahead_bars + 1]
        if len(shape_raw) != shape_len:
            continue

        entry_price = close[bar_idx]
        shape_norm = shape_raw - entry_price  # normalize: entry = 0

        shapes_by_regime[rid].append(shape_norm)
        raw_shapes_by_regime[rid].append(shape_raw)

        # Win/loss: MFE > MAE from oracle
        future = close[bar_idx + 1 : bar_idx + 1 + lookahead_bars]
        if len(future) > 0:
            max_up = future.max() - entry_price
            max_down = entry_price - future.min()
            outcomes_by_regime[rid].append(1 if max_up > max_down else 0)

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Filter to regimes with enough shapes (min 5)
    active_rids = [rm['regime_id'] for rm in regime_meta
                   if len(shapes_by_regime.get(rm['regime_id'], [])) >= 5]
    n_active = len(active_rids)

    if n_active == 0:
        print("  No regimes with enough shapes.")
    else:
        # Convert raw shapes to delta-from-entry for each regime
        delta_by_regime = {}
        for rid in active_rids:
            raw = np.array(shapes_by_regime[rid])
            entry_prices = raw[:, lookback_bars]  # price at bar 0
            delta_by_regime[rid] = raw - entry_prices[:, np.newaxis]

        # Distinct colors + line styles + markers (like fractal dim reference)
        cmap = plt.cm.tab10
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        markers     = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        regime_colors = {rid: cmap(k % 10) for k, rid in enumerate(active_rids)}
        x_axis = np.arange(-lookback_bars, lookahead_bars + 1)

        # ==== CHART 1: Signature overlay (all regime means, one plot) ====
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        legend_handles = []
        for k, rid in enumerate(active_rids):
            delta = delta_by_regime[rid]
            n_shapes = len(delta)
            color = regime_colors[rid]
            ls = line_styles[k % len(line_styles)]
            mk = markers[k % len(markers)]
            rm = next(m for m in regime_meta if m['regime_id'] == rid)
            wr = np.mean(outcomes_by_regime[rid]) * 100 if outcomes_by_regime[rid] else 0

            mean_d = delta.mean(axis=0)
            std_d = delta.std(axis=0)

            # Mean line with markers every 2 bars
            ax.plot(x_axis, mean_d, color=color, linewidth=2.5,
                    linestyle=ls, marker=mk, markevery=2, markersize=6)
            # +/- 1 std band
            ax.fill_between(x_axis, mean_d - std_d, mean_d + std_d,
                            color=color, alpha=0.08)

            label = f"R{rid} ({rm['direction']}, n={n_shapes}, WR={wr:.0f}%)"
            legend_handles.append(mlines.Line2D(
                [], [], color=color, linestyle=ls, marker=mk,
                markersize=6, linewidth=2.5, label=label))

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Bars from entry (15m)', fontsize=12)
        ax.set_ylabel('Price change from entry (ticks)', fontsize=12)
        ax.set_title('Regime Signatures: Mean Price Trajectory per I-MR Regime\n'
                      '(delta from entry, +/-1 std bands)', fontsize=14)
        ax.legend(handles=legend_handles, fontsize=10, loc='best',
                  framealpha=0.9, edgecolor='gray')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        sig_path = os.path.join(PLOTS_DIR, '0c_stacked_shapes.png')
        fig.savefig(sig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved signature overlay: {sig_path}")

        # ==== CHART 2: Per-regime spaghetti audit (delta, vertical stack) ====
        fig2, axes2 = plt.subplots(n_active, 1,
                                   figsize=(12, 3.0 * n_active),
                                   squeeze=False)

        for row, rid in enumerate(active_rids):
            ax2 = axes2[row, 0]
            delta = delta_by_regime[rid]
            n_shapes = len(delta)
            color = regime_colors[rid]
            rm = next(m for m in regime_meta if m['regime_id'] == rid)
            wr = np.mean(outcomes_by_regime[rid]) * 100 if outcomes_by_regime[rid] else 0

            # Individual traces
            max_to_plot = min(n_shapes, 300)
            alpha = max(0.05, min(0.25, 20.0 / max_to_plot))
            for j in range(max_to_plot):
                ax2.plot(x_axis, delta[j], color=color, alpha=alpha, linewidth=0.5)

            # Mean + std envelope
            mean_d = delta.mean(axis=0)
            std_d = delta.std(axis=0)
            ax2.plot(x_axis, mean_d, color=color, linewidth=3, label='Mean')
            ax2.fill_between(x_axis, mean_d - std_d, mean_d + std_d,
                             color=color, alpha=0.15)

            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            ax2.set_title(f"R{rid}: {n_shapes} shapes | dir={rm['direction']}, "
                          f"vol={rm['volatility']:.2f} | WR={wr:.0f}%",
                          fontsize=11, loc='left')
            ax2.set_ylabel('dPrice (ticks)')
            if row == n_active - 1:
                ax2.set_xlabel('Bars from entry (15m)')
            ax2.legend(fontsize=8, loc='upper right')
            ax2.grid(True, alpha=0.2)

        fig2.suptitle('Per-Regime Shape Audit (delta from entry, individual traces)',
                      fontsize=14, y=1.01)
        plt.tight_layout()
        audit_path = os.path.join(PLOTS_DIR, '0d_regime_audit.png')
        fig2.savefig(audit_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"  Saved per-regime audit: {audit_path}")

        # Summary table
        for rid in active_rids:
            n_s = len(shapes_by_regime[rid])
            rm = next(m for m in regime_meta if m['regime_id'] == rid)
            delta = delta_by_regime[rid]
            mean_end = delta[:, -1].mean()   # mean endpoint delta
            std_end = delta[:, -1].std()
            wr = np.mean(outcomes_by_regime[rid]) * 100 if outcomes_by_regime[rid] else 0
            print(f"  R{rid}: {n_s:>4} shapes, dir={rm['direction']:>5}, "
                  f"mean_end={mean_end:>+7.1f}, std_end={std_end:>6.1f}, "
                  f"WR={wr:.0f}%")

    # =====================================================================
    #  ANALYSIS G: LAPLACIAN SUB-SEGMENTATION
    #
    #  d2p/dt2 (curvature) = the missing acceleration layer.
    #  I-MR segments by velocity breaks. Laplacian segments by SHAPE
    #  changes: inflection points, momentum shifts, deceleration.
    #  Sub-segment each I-MR regime by curvature sign runs, then check
    #  if sub-segments produce tighter shape overlay (the Shi ideal).
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS G: LAPLACIAN SUB-SEGMENTATION (d2p/dt2)")
    print(f"{'='*70}")

    # --- G1: Compute curvature (discrete Laplacian) ---
    # d2p/dt2[t] = close[t+1] - 2*close[t] + close[t-1]
    curvature = np.zeros(len(close))
    curvature[1:-1] = close[2:] - 2 * close[1:-1] + close[:-2]
    curvature[0] = curvature[1]   # pad edges
    curvature[-1] = curvature[-2]

    # Curvature I-MR: same SPC approach on d2p/dt2
    curv_abs = np.abs(curvature)
    analysis_curv = curvature[price_imr['analysis_mask']]
    analysis_curv_abs = np.abs(analysis_curv)

    # MR of curvature (moving range of curvature)
    curv_mr = np.zeros(len(curvature))
    curv_mr[1:] = np.abs(curvature[1:] - curvature[:-1])

    analysis_curv_mr = curv_mr[price_imr['analysis_mask']]
    mr_bar_curv = np.mean(analysis_curv_mr[1:]) if len(analysis_curv_mr) > 1 else 1.0
    ucl_curv = 3.267 * mr_bar_curv  # D4 for n=2

    print(f"  Curvature stats: mean={np.mean(analysis_curv):.4f}, "
          f"std={np.std(analysis_curv):.4f}")
    print(f"  Curvature MR: mean={mr_bar_curv:.4f}, UCL={ucl_curv:.4f}")

    # --- G2: Sub-segment by curvature sign + UCL breaks ---
    # Within each I-MR regime, create sub-segments where:
    #   - curvature sign flips (convex <-> concave)
    #   - OR curvature MR exceeds UCL (acceleration shock)
    # Minimum sub-segment size: 4 bars
    MIN_SUB = 4

    sub_ids = np.full(len(close), -1, dtype=int)
    sub_meta = []
    current_sub = 0
    analysis_indices_g = np.where(price_imr['analysis_mask'])[0]

    for rm in regime_meta:
        # Bars in this regime
        r_mask = (regime_ids == rm['regime_id'])
        r_indices = np.where(r_mask)[0]
        if len(r_indices) < MIN_SUB:
            for idx in r_indices:
                sub_ids[idx] = current_sub
            current_sub += 1
            continue

        # Walk through regime bars, break on sign flip or UCL
        seg_start = 0
        prev_sign = 1 if curvature[r_indices[0]] >= 0 else -1

        for j in range(1, len(r_indices)):
            idx = r_indices[j]
            cur_sign = 1 if curvature[idx] >= 0 else -1
            mr_break = curv_mr[idx] > ucl_curv

            if (cur_sign != prev_sign or mr_break) and (j - seg_start >= MIN_SUB):
                # Close current sub-segment
                for k in range(seg_start, j):
                    sub_ids[r_indices[k]] = current_sub
                current_sub += 1
                seg_start = j

            prev_sign = cur_sign

        # Close final sub-segment
        for k in range(seg_start, len(r_indices)):
            sub_ids[r_indices[k]] = current_sub
        current_sub += 1

    # Merge tiny sub-segments
    n_subs_raw = current_sub
    for s in range(n_subs_raw):
        mask_s = (sub_ids == s)
        if 0 < mask_s.sum() < MIN_SUB:
            # Merge into previous or next
            idxs = np.where(mask_s)[0]
            if idxs[0] > 0 and sub_ids[idxs[0] - 1] >= 0:
                sub_ids[mask_s] = sub_ids[idxs[0] - 1]
            elif idxs[-1] < len(sub_ids) - 1 and sub_ids[idxs[-1] + 1] >= 0:
                sub_ids[mask_s] = sub_ids[idxs[-1] + 1]

    # Re-compact
    unique_subs = sorted([s for s in np.unique(sub_ids) if s >= 0])
    remap_s = {old: new for new, old in enumerate(unique_subs)}
    for i in range(len(sub_ids)):
        if sub_ids[i] >= 0:
            sub_ids[i] = remap_s[sub_ids[i]]
    n_subs = len(unique_subs)

    # Build sub-segment metadata
    for sid in range(n_subs):
        mask_s = (sub_ids == sid)
        indices_s = np.where(mask_s)[0]
        s_close = close[mask_s]
        s_curv = curvature[mask_s]
        parent_rid = regime_ids[indices_s[0]]

        sub_meta.append({
            'sub_id': sid,
            'parent_regime': int(parent_rid),
            'n_bars': int(mask_s.sum()),
            'mean_price': float(np.mean(s_close)),
            'mean_curvature': float(np.mean(s_curv)),
            'curv_sign': 'CONVEX' if np.mean(s_curv) >= 0 else 'CONCAVE',
            'price_change': float(s_close[-1] - s_close[0]) if len(s_close) > 1 else 0.0,
            'direction': 'LONG' if (s_close[-1] > s_close[0]) else 'SHORT',
            'start_idx': int(indices_s[0]),
            'end_idx': int(indices_s[-1]),
        })

    print(f"\n  I-MR regimes: {len(regime_meta)} -> Laplacian sub-segments: {n_subs}")
    print(f"  Avg sub-segment size: {np.mean([sm['n_bars'] for sm in sub_meta]):.1f} bars")
    print(f"\n  {'Sub':>4} {'Parent':>6} {'Bars':>5} {'Curv':>8} {'Shape':>8} "
          f"{'Dir':>6} {'Chg':>8}")
    print(f"  {'-'*4} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for sm in sub_meta:
        print(f"  S{sm['sub_id']:<3} R{sm['parent_regime']:<5} {sm['n_bars']:>5} "
              f"{sm['mean_curvature']:>+8.3f} {sm['curv_sign']:>8} "
              f"{sm['direction']:>6} {sm['price_change']:>+8.1f}")

    # --- G3: Collect shapes per sub-segment ---
    shapes_by_sub = {sm['sub_id']: [] for sm in sub_meta}
    outcomes_by_sub = {sm['sub_id']: [] for sm in sub_meta}

    for i, bar_idx in enumerate(sample_indices):
        if bar_idx < 0 or bar_idx < lookback_bars:
            continue
        if bar_idx + lookahead_bars >= len(close):
            continue

        sid = sub_ids[bar_idx]
        if sid < 0:
            continue

        shape_raw = close[bar_idx - lookback_bars : bar_idx + lookahead_bars + 1]
        if len(shape_raw) != shape_len:
            continue

        shapes_by_sub[sid].append(shape_raw)

        entry_price = close[bar_idx]
        future = close[bar_idx + 1 : bar_idx + 1 + lookahead_bars]
        if len(future) > 0:
            max_up = future.max() - entry_price
            max_down = entry_price - future.min()
            outcomes_by_sub[sid].append(1 if max_up > max_down else 0)

    # --- G4: Signature plot — sub-segments overlaid ---
    active_subs = [sm['sub_id'] for sm in sub_meta
                   if len(shapes_by_sub.get(sm['sub_id'], [])) >= 5]
    n_active_g = len(active_subs)

    print(f"\n  Sub-segments with >=5 shapes: {n_active_g}/{n_subs}")

    if n_active_g > 0:
        # Compute delta from entry
        delta_by_sub = {}
        for sid in active_subs:
            raw = np.array(shapes_by_sub[sid])
            entry_prices = raw[:, lookback_bars]
            delta_by_sub[sid] = raw - entry_prices[:, np.newaxis]

        # Color by parent regime, line style by curvature sign
        cmap_g = plt.cm.tab10
        x_axis_g = np.arange(-lookback_bars, lookahead_bars + 1)

        fig_g, ax_g = plt.subplots(1, 1, figsize=(14, 8))
        legend_handles_g = []

        for k, sid in enumerate(active_subs):
            sm = next(s for s in sub_meta if s['sub_id'] == sid)
            delta = delta_by_sub[sid]
            n_shapes = len(delta)
            color = cmap_g(sm['parent_regime'] % 10)
            ls = '-' if sm['curv_sign'] == 'CONVEX' else '--'
            mk = markers[k % len(markers)] if k < len(markers) else 'o'
            wr = np.mean(outcomes_by_sub[sid]) * 100 if outcomes_by_sub[sid] else 0

            mean_d = delta.mean(axis=0)
            std_d = delta.std(axis=0)

            ax_g.plot(x_axis_g, mean_d, color=color, linewidth=2.5,
                      linestyle=ls, marker=mk, markevery=2, markersize=5)
            ax_g.fill_between(x_axis_g, mean_d - std_d, mean_d + std_d,
                              color=color, alpha=0.06)

            label = (f"S{sid} (R{sm['parent_regime']},{sm['curv_sign'][:3]}, "
                     f"n={n_shapes}, WR={wr:.0f}%)")
            legend_handles_g.append(mlines.Line2D(
                [], [], color=color, linestyle=ls, marker=mk,
                markersize=5, linewidth=2.5, label=label))

        ax_g.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_g.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_g.set_xlabel('Bars from entry (15m)', fontsize=12)
        ax_g.set_ylabel('Price change from entry (ticks)', fontsize=12)
        ax_g.set_title('Laplacian Sub-Segment Signatures\n'
                        '(solid=CONVEX, dashed=CONCAVE, color=parent regime)',
                        fontsize=14)
        ax_g.legend(handles=legend_handles_g, fontsize=9, loc='best',
                    framealpha=0.9, edgecolor='gray')
        ax_g.grid(True, alpha=0.3)

        plt.tight_layout()
        g_sig_path = os.path.join(PLOTS_DIR, '0e_laplacian_signatures.png')
        fig_g.savefig(g_sig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_g)
        print(f"  Saved: {g_sig_path}")

        # --- G5: Coherence comparison: I-MR vs Laplacian ---
        # Compute avg std at endpoint for I-MR regimes vs Laplacian sub-segments
        imr_stds = []
        for rid in active_rids:
            if rid in delta_by_regime:
                imr_stds.append(delta_by_regime[rid][:, -1].std())

        lap_stds = []
        for sid in active_subs:
            lap_stds.append(delta_by_sub[sid][:, -1].std())

        avg_imr_std = np.mean(imr_stds) if imr_stds else 0
        avg_lap_std = np.mean(lap_stds) if lap_stds else 0

        print(f"\n  COHERENCE COMPARISON:")
        print(f"  I-MR regimes   -> avg endpoint std: {avg_imr_std:.1f} ticks "
              f"({len(active_rids)} regimes)")
        print(f"  Laplacian subs -> avg endpoint std: {avg_lap_std:.1f} ticks "
              f"({n_active_g} sub-segments)")
        if avg_imr_std > 0:
            improvement = (1 - avg_lap_std / avg_imr_std) * 100
            print(f"  Improvement: {improvement:+.1f}% "
                  f"({'TIGHTER' if improvement > 0 else 'WIDER'})")

        # Per sub-segment summary
        print(f"\n  {'Sub':>4} {'Parent':>6} {'Shape':>7} {'N':>4} "
              f"{'MeanEnd':>8} {'StdEnd':>7} {'WR':>5}")
        for sid in active_subs:
            sm = next(s for s in sub_meta if s['sub_id'] == sid)
            delta = delta_by_sub[sid]
            wr = np.mean(outcomes_by_sub[sid]) * 100 if outcomes_by_sub[sid] else 0
            print(f"  S{sid:<3} R{sm['parent_regime']:<5} {sm['curv_sign'][:3]:>7} "
                  f"{len(delta):>4} {delta[:,-1].mean():>+8.1f} "
                  f"{delta[:,-1].std():>7.1f} {wr:>4.0f}%")

    # =====================================================================
    #  ANALYSIS H: ITERATIVE SHAPE CLUSTERING (delta from entry)
    #
    #  Every segment starts at 0, values = cumulative price change.
    #  e.g. [0, +40, +20, +30, +20, +40] = the movement pattern.
    #  Grid-search over segment length and cluster count.
    #  Score by silhouette, auto-select best, show top 10 clusters.
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS H: ITERATIVE SHAPE CLUSTERING (delta from entry)")
    print(f"{'='*70}")

    from sklearn.cluster import KMeans
    from collections import Counter

    TOP_K = 10
    MIN_CLUSTER_SIZE = 5  # minimum members for a useful cluster

    analysis_idx_h = np.where(price_imr['analysis_mask'])[0]

    def _extract_segments(seg_len):
        """Cut segments, delta from entry (start=0, values=cumulative change)."""
        raws, feats, idxs = [], [], []
        for idx in analysis_idx_h:
            if idx + seg_len > len(close):
                continue
            seg = close[idx : idx + seg_len]
            if len(seg) != seg_len:
                continue
            feat = seg - seg[0]  # delta from entry: [0, +40, +20, ...]
            raws.append(seg)
            feats.append(feat)
            idxs.append(idx)
        if len(feats) == 0:
            return np.array([]), np.array([]), np.array([])
        return np.array(raws), np.array(feats), np.array(idxs)

    def _cluster_coherence(feats, labels, top_k=10):
        """Mean within-cluster std (lower = tighter overlays).
        Only considers top-k clusters by size with >= MIN_CLUSTER_SIZE."""
        counts = Counter(labels)
        top = [cid for cid, cnt in counts.most_common(top_k)
               if cnt >= MIN_CLUSTER_SIZE]
        if not top:
            return 999.0, 0
        stds = []
        for cid in top:
            mask = (labels == cid)
            stds.append(feats[mask].std(axis=0).mean())
        return np.mean(stds), len(top)

    # --- Phase 1: Find best segment length ---
    seg_lens = [8, 12, 16, 24]
    best_len_score = 999.0
    best_seg_len = 16

    print(f"\n  Phase 1: Find best segment length (k=20, delta mode)")
    for seg_len in seg_lens:
        raws, feats, idxs = _extract_segments(seg_len)
        n_seg = len(feats)
        if n_seg < 40:
            continue
        k_test = min(20, n_seg // 3)
        km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
        labels = km.fit_predict(feats)
        coh, n_valid = _cluster_coherence(feats, labels)
        is_best = coh < best_len_score
        if is_best:
            best_len_score = coh
            best_seg_len = seg_len
        marker = ' <--' if is_best else ''
        print(f"    len={seg_len:>2}: {n_seg} segs, k={k_test}, "
              f"coherence={coh:.2f} ({n_valid} valid clusters){marker}")

    print(f"  Best length: {best_seg_len}")

    # --- Phase 2: Iterate k upward until clusters are tight ---
    raws, feats, idxs = _extract_segments(best_seg_len)
    n_seg = len(feats)

    max_k = min(n_seg // MIN_CLUSTER_SIZE, 100)
    k_candidates = [k for k in [10, 15, 20, 30, 40, 50, 75, 100]
                    if k <= max_k and k < n_seg]
    if not k_candidates:
        k_candidates = [max(2, n_seg // MIN_CLUSTER_SIZE)]

    print(f"\n  Phase 2: Iterate k (len={best_seg_len}, {n_seg} segments)")
    print(f"  {'K':>4} {'Coherence':>10} {'ValidClusters':>14} "
          f"{'MinSize':>8} {'Best?':>5}")
    print(f"  {'-'*4} {'-'*10} {'-'*14} {'-'*8} {'-'*5}")

    best_coh = 999.0
    best_config = None
    prev_coh = None

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(feats)
        coh, n_valid = _cluster_coherence(feats, labels)

        counts = Counter(labels)
        top_sizes = [cnt for _, cnt in counts.most_common(TOP_K)]
        min_top = min(top_sizes) if top_sizes else 0

        is_best = coh < best_coh and n_valid >= min(TOP_K, k)
        if is_best:
            best_coh = coh
            best_config = {
                'seg_len': best_seg_len, 'k': k,
                'labels': labels, 'raws': raws, 'feats': feats,
                'idxs': idxs, 'coherence': coh,
            }

        marker = ' <--' if is_best else ''
        print(f"  {k:>4} {coh:>10.2f} {n_valid:>14} "
              f"{min_top:>8}{marker}")

        # Stop if coherence stopped improving (< 5% gain)
        if prev_coh is not None and coh > prev_coh * 0.95 and not is_best:
            if k > 30:  # only stop early after trying enough
                print(f"  (converged at k={k})")
                break
        prev_coh = coh

    if best_config is None:
        print("  No valid configuration found. Skipping.")
    else:
        bc = best_config
        print(f"\n  BEST: len={bc['seg_len']}, k={bc['k']}, "
              f"coherence={bc['coherence']:.2f}")

        # Build cluster stats
        labels = bc['labels']
        raws = bc['raws']
        feats = bc['feats']
        counts = Counter(labels)
        # Sort by size, filter to >= MIN_CLUSTER_SIZE
        top_clusters = [(cid, cnt) for cid, cnt in counts.most_common()
                        if cnt >= MIN_CLUSTER_SIZE][:TOP_K]

        print(f"\n  {'Clust':>5} {'N':>5} {'MeanChg':>8} {'StdChg':>7} "
              f"{'WR':>5} {'Coh':>6}")
        print(f"  {'-'*5} {'-'*5} {'-'*8} {'-'*7} {'-'*5} {'-'*6}")

        cluster_stats = []
        for cid, count in top_clusters:
            mask_c = (labels == cid)
            raw_c = raws[mask_c]
            feat_c = feats[mask_c]

            changes = raw_c[:, -1] - raw_c[:, 0]
            mean_chg = changes.mean()
            std_chg = changes.std()
            wr = (changes > 0).sum() / len(changes) * 100
            coherence = feat_c.std(axis=0).mean()

            cluster_stats.append({
                'cid': cid, 'count': count, 'mean_chg': mean_chg,
                'std_chg': std_chg, 'wr': wr, 'coherence': coherence,
                'feat': feat_c, 'raw': raw_c,
            })

            print(f"  C{cid:<4} {count:>5} {mean_chg:>+8.1f} {std_chg:>7.1f} "
                  f"{wr:>4.0f}% {coherence:>6.1f}")

        # Plot top 10 clusters
        n_plot = min(TOP_K, len(cluster_stats))
        n_cols = 5
        n_rows = (n_plot + n_cols - 1) // n_cols
        seg_len = bc['seg_len']
        x_seg = np.arange(seg_len)

        fig_h, axes_h = plt.subplots(n_rows, n_cols,
                                      figsize=(4 * n_cols, 3.5 * n_rows),
                                      squeeze=False)

        for k in range(n_plot):
            row, col = divmod(k, n_cols)
            ax = axes_h[row, col]
            cs = cluster_stats[k]
            feat_c = cs['feat']
            n_in = len(feat_c)
            color = cmap(k % 10)

            # Individual traces
            max_plot = min(n_in, 200)
            alpha = max(0.05, min(0.3, 20.0 / max_plot))
            for j in range(max_plot):
                ax.plot(x_seg, feat_c[j], color=color, alpha=alpha, linewidth=0.5)

            # Mean + std envelope
            mean_f = feat_c.mean(axis=0)
            std_f = feat_c.std(axis=0)
            ax.plot(x_seg, mean_f, color=color, linewidth=3)
            ax.fill_between(x_seg, mean_f - std_f, mean_f + std_f,
                            color=color, alpha=0.15)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            # Auto-fit y-axis
            all_v = feat_c.flatten()
            y_c = np.mean(all_v)
            y_s = max(np.std(all_v) * 3, 5)
            ax.set_ylim(y_c - y_s, y_c + y_s)

            ax.set_title(f"C{cs['cid']}: n={n_in}, WR={cs['wr']:.0f}%\n"
                         f"coh={cs['coherence']:.1f}, chg={cs['mean_chg']:+.0f}",
                         fontsize=9)
            if col == 0:
                ax.set_ylabel('Delta (ticks)')
            if row == n_rows - 1:
                ax.set_xlabel('Bar')
            ax.grid(True, alpha=0.2)

        for k in range(n_plot, n_rows * n_cols):
            row, col = divmod(k, n_cols)
            axes_h[row, col].set_visible(False)

        fig_h.suptitle(f'Top {n_plot} Shape Clusters (len={seg_len}, k={bc["k"]})\n'
                        f'Delta from entry, coherence={bc["coherence"]:.1f}, '
                        f'{len(feats)} segments',
                        fontsize=13)
        plt.tight_layout()
        h_path = os.path.join(PLOTS_DIR, '0f_shape_clusters.png')
        fig_h.savefig(h_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_h)
        print(f"\n  Saved: {h_path}")

    # =====================================================================
    #  ANALYSIS I: SEED PRIMITIVE SHAPE CLASSIFICATION
    #
    #  Classify every segment against 20 mathematical seed shapes
    #  using Pearson correlation. Threshold 0.85 → shape or NOISE.
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS I: SEED PRIMITIVE CLASSIFICATION (20 shapes)")
    print(f"{'='*70}")

    from collections import Counter as Counter_I

    # Use same segment length as Analysis H (best_seg_len), fallback 16
    seed_len = best_seg_len if 'best_seg_len' in dir() else 16
    library = SeedPrimitiveLibrary(N=seed_len)

    print(f"\n  Seed library: {len(library.shapes)} shapes, segment length={seed_len}")
    print(f"  Shapes: {', '.join(sorted(library.shapes.keys()))}")

    # Extract segments (same as Analysis H: every analysis bar)
    analysis_mask_i = price_imr['analysis_mask']
    analysis_idx_i = np.where(analysis_mask_i)[0]

    classifications = []  # (idx, shape_name, correlation, raw_segment)
    for idx in tqdm(analysis_idx_i, desc='  Classifying', ncols=80):
        if idx + seed_len > len(close):
            continue
        seg = close[idx : idx + seed_len]
        if len(seg) != seed_len:
            continue
        shape_name, corr = library.classify_trajectory(seg)
        classifications.append((idx, shape_name, corr, seg))

    n_total = len(classifications)
    if n_total == 0:
        print("  No segments to classify. Skipping Analysis I.")
    else:
        # Tally
        shape_counts = Counter_I(c[1] for c in classifications)
        n_noise = shape_counts.get('NOISE', 0)
        n_matched = n_total - n_noise
        noise_pct = n_noise / n_total * 100

        print(f"\n  Total segments: {n_total}")
        print(f"  Matched (corr >= {library.CORR_THRESHOLD}): {n_matched} ({100 - noise_pct:.1f}%)")
        print(f"  NOISE (corr < {library.CORR_THRESHOLD}):    {n_noise} ({noise_pct:.1f}%)")

        # Shape breakdown table
        print(f"\n  {'Shape':<25} {'Count':>6} {'%':>6} {'MeanCorr':>9} "
              f"{'MeanChg':>8} {'WR':>5}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*9} {'-'*8} {'-'*5}")

        shape_stats = []
        for shape_name in sorted(shape_counts.keys()):
            if shape_name == 'NOISE':
                continue
            entries = [(idx, corr, seg) for idx, sn, corr, seg in classifications
                       if sn == shape_name]
            count = len(entries)
            pct = count / n_total * 100
            mean_corr = np.mean([e[1] for e in entries])
            changes = np.array([e[2][-1] - e[2][0] for e in entries])
            mean_chg = changes.mean()
            wr = (changes > 0).sum() / len(changes) * 100 if len(changes) > 0 else 0

            shape_stats.append({
                'name': shape_name, 'count': count, 'pct': pct,
                'mean_corr': mean_corr, 'mean_chg': mean_chg, 'wr': wr,
                'entries': entries,
            })

            print(f"  {shape_name:<25} {count:>6} {pct:>5.1f}% {mean_corr:>9.3f} "
                  f"{mean_chg:>+8.1f} {wr:>4.0f}%")

        # NOISE deep-dive: what shapes were they closest to?
        if n_noise > 0:
            noise_entries = [(idx, corr, seg) for idx, sn, corr, seg in classifications
                             if sn == 'NOISE']
            noise_corrs = np.array([e[1] for e in noise_entries])
            noise_chgs = np.array([e[2][-1] - e[2][0] for e in noise_entries])
            noise_wr = (noise_chgs > 0).sum() / len(noise_chgs) * 100
            print(f"  {'NOISE':<25} {n_noise:>6} {noise_pct:>5.1f}% "
                  f"{np.mean(noise_corrs):>9.3f} {noise_chgs.mean():>+8.1f} "
                  f"{noise_wr:>4.0f}%")

            # --- NOISE BREAKDOWN: what is the best-match shape for each noise segment? ---
            print(f"\n  NOISE BREAKDOWN (best-match shape for segments below 0.85):")

            # Re-classify noise to get their best-match shape name
            noise_best_shapes = []
            for idx, corr, seg in noise_entries:
                mn, mx = seg.min(), seg.max()
                if mx - mn < 1e-12:
                    noise_best_shapes.append('FLATLINE')
                    continue
                normed = (seg - mn) / (mx - mn)
                best_n, best_r = 'UNKNOWN', -999.0
                for nm, tmpl in library.shapes.items():
                    if tmpl.std() < 1e-12:
                        continue
                    r = np.corrcoef(normed, tmpl)[0, 1]
                    if not np.isnan(r) and r > best_r:
                        best_r = r
                        best_n = nm
                noise_best_shapes.append(best_n)

            noise_shape_counts = Counter_I(noise_best_shapes)
            print(f"\n  {'Nearest Shape':<25} {'Count':>6} {'%ofNoise':>9} "
                  f"{'MeanCorr':>9} {'MeanChg':>8} {'WR':>5}")
            print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*9} {'-'*8} {'-'*5}")

            for ns_name, ns_cnt in noise_shape_counts.most_common():
                ns_mask = [i for i, s in enumerate(noise_best_shapes) if s == ns_name]
                ns_corrs = noise_corrs[ns_mask]
                ns_chgs = noise_chgs[ns_mask]
                ns_wr = (ns_chgs > 0).sum() / len(ns_chgs) * 100 if len(ns_chgs) > 0 else 0
                ns_pct = ns_cnt / n_noise * 100
                print(f"  {ns_name:<25} {ns_cnt:>6} {ns_pct:>8.1f}% "
                      f"{ns_corrs.mean():>9.3f} {ns_chgs.mean():>+8.1f} {ns_wr:>4.0f}%")

            # Correlation band breakdown
            print(f"\n  NOISE BY CORRELATION BAND:")
            print(f"  {'Band':<15} {'Count':>6} {'%':>6} {'MeanChg':>8} {'WR':>5} "
                  f"{'StdChg':>7}")
            print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*7}")
            thr = library.CORR_THRESHOLD
            bands = [(thr - 0.05, thr, f'{thr-0.05:.2f}-{thr:.2f}'),
                     (thr - 0.15, thr - 0.05, f'{thr-0.15:.2f}-{thr-0.05:.2f}'),
                     (thr - 0.35, thr - 0.15, f'{thr-0.35:.2f}-{thr-0.15:.2f}'),
                     (-1.0, thr - 0.35, f'<{thr-0.35:.2f}')]
            for lo, hi, label in bands:
                band_mask = (noise_corrs >= lo) & (noise_corrs < hi)
                bc = band_mask.sum()
                if bc == 0:
                    continue
                b_chgs = noise_chgs[band_mask]
                b_wr = (b_chgs > 0).sum() / bc * 100
                print(f"  {label:<15} {bc:>6} {bc/n_noise*100:>5.1f}% "
                      f"{b_chgs.mean():>+8.1f} {b_wr:>4.0f}% {b_chgs.std():>7.1f}")

        # =================================================================
        #  SUB-CLASSIFICATION: within each shape, cluster timing variants
        # =================================================================
        from sklearn.cluster import KMeans as KMeans_sub

        MIN_FOR_SUB = 10   # need at least this many to sub-cluster
        SUB_K = 3           # number of timing sub-types per shape
        x_plot = np.arange(seed_len)

        sub_shapes = [s for s in shape_stats if s['count'] >= MIN_FOR_SUB]

        if sub_shapes:
            print(f"\n  {'='*60}")
            print(f"  SUB-CLASSIFICATION (shapes with n >= {MIN_FOR_SUB})")
            print(f"  {'='*60}")

            all_sub_stats = []  # collect for plotting

            for ss in sub_shapes:
                entries = ss['entries']
                # Normalize each segment to 0-1 for shape clustering
                normed_segs = []
                raw_segs = []
                for idx, corr, seg in entries:
                    mn, mx = seg.min(), seg.max()
                    if mx - mn < 1e-12:
                        normed_segs.append(np.zeros(seed_len))
                    else:
                        normed_segs.append((seg - mn) / (mx - mn))
                    raw_segs.append(seg)
                normed_arr = np.array(normed_segs)
                raw_arr = np.array(raw_segs)

                k_use = min(SUB_K, len(entries) // 3)
                if k_use < 2:
                    k_use = 2

                km = KMeans_sub(n_clusters=k_use, random_state=42, n_init=10)
                sub_labels = km.fit_predict(normed_arr)

                print(f"\n  {ss['name']} (n={ss['count']}) -> {k_use} sub-types:")
                print(f"    {'Sub':<8} {'N':>4} {'MeanCorr':>9} {'MeanChg':>8} "
                      f"{'WR':>5} {'R2':>6} {'Timing':>20}")
                print(f"    {'-'*8} {'-'*4} {'-'*9} {'-'*8} {'-'*5} {'-'*6} {'-'*20}")

                shape_sub_stats = []
                for si in range(k_use):
                    mask = (sub_labels == si)
                    n_sub = mask.sum()
                    if n_sub == 0:
                        continue
                    sub_normed = normed_arr[mask]
                    sub_raw = raw_arr[mask]
                    sub_corrs = [entries[j][1] for j in range(len(entries)) if mask[j]]

                    # Timing: where does main movement happen?
                    # Measure cumulative change at 33% and 66% of segment
                    centroid = km.cluster_centers_[si]
                    third = seed_len // 3
                    two_third = 2 * seed_len // 3

                    # Movement in first/mid/last third
                    move_early = abs(centroid[third] - centroid[0])
                    move_mid = abs(centroid[two_third] - centroid[third])
                    move_late = abs(centroid[-1] - centroid[two_third])
                    total_move = move_early + move_mid + move_late

                    if total_move < 1e-12:
                        timing = "FLAT"
                    else:
                        pct_early = move_early / total_move
                        pct_late = move_late / total_move
                        if pct_early > 0.45:
                            timing = "EARLY (front-loaded)"
                        elif pct_late > 0.45:
                            timing = "LATE (back-loaded)"
                        else:
                            timing = "STEADY (even)"

                    sub_chgs = sub_raw[:, -1] - sub_raw[:, 0]
                    sub_wr = (sub_chgs > 0).sum() / n_sub * 100
                    sub_mean_chg = sub_chgs.mean()
                    sub_mean_corr = np.mean(sub_corrs)

                    # Goodness of fit: R² = 1 - SS_res / SS_tot
                    ss_res = np.sum((sub_normed - centroid[np.newaxis, :]) ** 2)
                    ss_tot = np.sum((sub_normed - sub_normed.mean()) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

                    label = f"{ss['name']}_{si}"
                    shape_sub_stats.append({
                        'label': label, 'n': n_sub, 'timing': timing,
                        'mean_corr': sub_mean_corr, 'mean_chg': sub_mean_chg,
                        'wr': sub_wr, 'r2': r2, 'centroid': centroid,
                        'normed': sub_normed, 'raw': sub_raw,
                    })

                    print(f"    {label:<8} {n_sub:>4} {sub_mean_corr:>9.3f} "
                          f"{sub_mean_chg:>+8.1f} {sub_wr:>4.0f}% {r2:>5.2f} "
                          f"{timing:>20}")

                all_sub_stats.append({
                    'parent': ss['name'], 'subs': shape_sub_stats,
                    'template': library.shapes[ss['name']],
                })

            # --- Sub-classification plot: one row per parent shape ---
            if all_sub_stats:
                n_parents = len(all_sub_stats)
                max_subs = max(len(a['subs']) for a in all_sub_stats)
                fig_sub, axes_sub = plt.subplots(
                    n_parents, max_subs,
                    figsize=(5 * max_subs, 4 * n_parents),
                    squeeze=False)

                sub_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']

                for row_i, parent_info in enumerate(all_sub_stats):
                    for col_i, sub in enumerate(parent_info['subs']):
                        ax = axes_sub[row_i, col_i]
                        normed = sub['normed']
                        n_s = len(normed)
                        max_show = min(n_s, 150)
                        alpha = max(0.08, min(0.4, 20.0 / max_show))
                        color = sub_colors[col_i % len(sub_colors)]

                        for j in range(max_show):
                            ax.plot(x_plot, normed[j], color=color,
                                    alpha=alpha, linewidth=0.5)

                        # Centroid
                        ax.plot(x_plot, sub['centroid'], color=color,
                                linewidth=3, label='Centroid')
                        # Template
                        ax.plot(x_plot, parent_info['template'], color='gray',
                                linewidth=2, linestyle='--', alpha=0.6,
                                label='Template')

                        ax.set_title(
                            f"{sub['label']}\n"
                            f"n={sub['n']}, WR={sub['wr']:.0f}%, "
                            f"chg={sub['mean_chg']:+.0f}\n"
                            f"{sub['timing']}",
                            fontsize=9)
                        ax.set_ylim(-0.05, 1.05)
                        ax.grid(True, alpha=0.15)
                        ax.axhline(y=0, color='gray', linewidth=0.3)
                        if col_i == 0:
                            ax.set_ylabel(parent_info['parent'], fontsize=9,
                                          fontweight='bold')
                        if row_i == 0 and col_i == 0:
                            ax.legend(fontsize=7, loc='best')

                    # Hide unused columns
                    for col_i in range(len(parent_info['subs']), max_subs):
                        axes_sub[row_i, col_i].set_visible(False)

                fig_sub.suptitle(
                    f'Sub-Classification: Timing Variants Within Shapes\n'
                    f'{len(sub_shapes)} shapes sub-clustered (k={SUB_K})',
                    fontsize=13)
                plt.tight_layout()
                sub_path = os.path.join(PLOTS_DIR, '0k_sub_classification.png')
                fig_sub.savefig(sub_path, dpi=150, bbox_inches='tight',
                                facecolor='white')
                plt.close(fig_sub)
                print(f"\n  Saved: {sub_path}")

        # --- Plot 1: Seed template gallery (4x5 grid of all 20 shapes) ---
        fig_seeds, axes_seeds = plt.subplots(4, 5, figsize=(20, 12), squeeze=False)
        sorted_shapes = sorted(library.shapes.keys())
        x_plot = np.arange(seed_len)

        for i, name in enumerate(sorted_shapes):
            row, col = divmod(i, 5)
            ax = axes_seeds[row, col]
            template = library.shapes[name]
            cnt = shape_counts.get(name, 0)
            ax.plot(x_plot, template, color='#2196F3', linewidth=2.5)
            ax.fill_between(x_plot, 0, template, alpha=0.15, color='#2196F3')
            ax.set_title(f'{name}\nn={cnt}', fontsize=9, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, seed_len - 1)
            ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axhline(y=1, color='gray', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.15)
            if col == 0:
                ax.set_ylabel('Normalized')
            if row == 3:
                ax.set_xlabel('Bar')

        fig_seeds.suptitle(
            f'Seed Primitive Library ({len(library.shapes)} shapes, N={seed_len})\n'
            f'Matched: {n_matched}/{n_total} ({100 - noise_pct:.1f}%), '
            f'NOISE: {n_noise} ({noise_pct:.1f}%)',
            fontsize=13)
        plt.tight_layout()
        seeds_path = os.path.join(PLOTS_DIR, '0g_seed_templates.png')
        fig_seeds.savefig(seeds_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_seeds)
        print(f"\n  Saved: {seeds_path}")

        # --- Plot 2: Top matched shapes with actual price overlays ---
        # Show up to 10 most-populated shapes (excluding NOISE)
        top_shapes = sorted(shape_stats, key=lambda s: s['count'], reverse=True)[:10]
        n_top = len(top_shapes)

        if n_top > 0:
            n_cols_t = 5
            n_rows_t = (n_top + n_cols_t - 1) // n_cols_t
            fig_match, axes_match = plt.subplots(
                n_rows_t, n_cols_t, figsize=(4 * n_cols_t, 3.5 * n_rows_t),
                squeeze=False)

            for k, ss in enumerate(top_shapes):
                row, col = divmod(k, n_cols_t)
                ax = axes_match[row, col]

                entries = ss['entries']
                # Normalize each segment to 0-1 and overlay
                max_show = min(len(entries), 200)
                alpha = max(0.05, min(0.3, 20.0 / max_show))

                all_normed = []
                for j in range(max_show):
                    seg = entries[j][2]
                    mn, mx = seg.min(), seg.max()
                    if mx - mn < 1e-12:
                        normed = np.zeros(seed_len)
                    else:
                        normed = (seg - mn) / (mx - mn)
                    all_normed.append(normed)
                    ax.plot(x_plot, normed, color='#1976D2', alpha=alpha,
                            linewidth=0.5)

                # Template
                template = library.shapes[ss['name']]
                ax.plot(x_plot, template, color='#F44336', linewidth=2.5,
                        label='Template')

                # Mean of actual segments
                if all_normed:
                    mean_n = np.mean(all_normed, axis=0)
                    ax.plot(x_plot, mean_n, color='#FF9800', linewidth=2,
                            linestyle='--', label='Mean actual')

                ax.set_title(f"{ss['name']}\nn={ss['count']}, "
                             f"r={ss['mean_corr']:.2f}, WR={ss['wr']:.0f}%",
                             fontsize=8)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.15)
                if col == 0:
                    ax.set_ylabel('Normalized')
                if row == n_rows_t - 1:
                    ax.set_xlabel('Bar')
                if k == 0:
                    ax.legend(fontsize=7, loc='best')

            # Hide unused
            for k in range(n_top, n_rows_t * n_cols_t):
                row, col = divmod(k, n_cols_t)
                axes_match[row, col].set_visible(False)

            fig_match.suptitle(
                f'Top {n_top} Matched Shapes — Actual Segments vs Templates\n'
                f'{n_total} total segments, {n_matched} matched, '
                f'{n_noise} NOISE ({noise_pct:.1f}%)',
                fontsize=13)
            plt.tight_layout()
            match_path = os.path.join(PLOTS_DIR, '0h_seed_matches.png')
            fig_match.savefig(match_path, dpi=150, bbox_inches='tight',
                              facecolor='white')
            plt.close(fig_match)
            print(f"  Saved: {match_path}")

        # --- Plot 3: Correlation distribution histogram ---
        all_corrs = [c[2] for c in classifications]
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        ax_hist.hist(all_corrs, bins=50, color='#2196F3', alpha=0.7,
                     edgecolor='white')
        ax_hist.axvline(x=library.CORR_THRESHOLD, color='#F44336', linewidth=2,
                        linestyle='--', label=f'Threshold ({library.CORR_THRESHOLD})')
        ax_hist.set_xlabel('Best Pearson Correlation')
        ax_hist.set_ylabel('Segment Count')
        ax_hist.set_title(f'Correlation Distribution (n={n_total})\n'
                          f'Above 0.85: {n_matched} ({100 - noise_pct:.1f}%), '
                          f'Below: {n_noise} ({noise_pct:.1f}%)')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.15)
        plt.tight_layout()
        hist_path = os.path.join(PLOTS_DIR, '0i_corr_distribution.png')
        fig_hist.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_hist)
        print(f"  Saved: {hist_path}")

        # --- Plot 4: NOISE audit — spaghetti by nearest shape ---
        if n_noise > 0 and 'noise_entries' in dir() and 'noise_best_shapes' in dir():
            # Group noise by nearest shape, show top 6
            noise_by_shape = {}
            for i, (idx, corr, seg) in enumerate(noise_entries):
                ns = noise_best_shapes[i]
                if ns not in noise_by_shape:
                    noise_by_shape[ns] = []
                noise_by_shape[ns].append((corr, seg))

            top_noise_shapes = sorted(noise_by_shape.keys(),
                                      key=lambda s: len(noise_by_shape[s]),
                                      reverse=True)[:6]
            n_ns = len(top_noise_shapes)

            if n_ns > 0:
                n_cols_ns = min(3, n_ns)
                n_rows_ns = (n_ns + n_cols_ns - 1) // n_cols_ns
                fig_noise, axes_noise = plt.subplots(
                    n_rows_ns, n_cols_ns,
                    figsize=(5 * n_cols_ns, 4 * n_rows_ns),
                    squeeze=False)

                for k, ns_name in enumerate(top_noise_shapes):
                    row, col = divmod(k, n_cols_ns)
                    ax = axes_noise[row, col]
                    ns_segs = noise_by_shape[ns_name]
                    ns_cnt = len(ns_segs)
                    ns_corrs_plot = [s[0] for s in ns_segs]
                    ns_raw = [s[1] for s in ns_segs]

                    # Delta from entry (not normalized — show raw movement)
                    max_show = min(ns_cnt, 150)
                    alpha_ns = max(0.08, min(0.4, 20.0 / max_show))

                    deltas = []
                    for j in range(max_show):
                        d = ns_raw[j] - ns_raw[j][0]
                        deltas.append(d)
                        ax.plot(x_plot, d, color='#9E9E9E', alpha=alpha_ns,
                                linewidth=0.5)

                    # Mean delta
                    if deltas:
                        mean_d = np.mean(deltas, axis=0)
                        std_d = np.std(deltas, axis=0)
                        ax.plot(x_plot, mean_d, color='#F44336', linewidth=2.5,
                                label='Mean')
                        ax.fill_between(x_plot, mean_d - std_d, mean_d + std_d,
                                        color='#F44336', alpha=0.12)

                    # Template overlay (normalized to match delta scale)
                    tmpl = library.shapes.get(ns_name)
                    if tmpl is not None and tmpl.std() > 1e-12 and deltas:
                        # Scale template to delta range for visual comparison
                        d_range = np.abs(mean_d).max()
                        if d_range > 0:
                            tmpl_scaled = (tmpl - tmpl.mean()) / tmpl.std() * np.std(deltas)
                            tmpl_scaled = tmpl_scaled - tmpl_scaled[0]
                            ax.plot(x_plot, tmpl_scaled, color='#2196F3',
                                    linewidth=2, linestyle='--', label='Template')

                    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                    ns_chgs = np.array([s[1][-1] - s[1][0] for s in ns_segs])
                    ns_wr = (ns_chgs > 0).sum() / len(ns_chgs) * 100
                    ax.set_title(
                        f'NOISE nearest: {ns_name}\n'
                        f'n={ns_cnt}, r={np.mean(ns_corrs_plot):.2f}, '
                        f'WR={ns_wr:.0f}%',
                        fontsize=9)
                    ax.grid(True, alpha=0.15)
                    if col == 0:
                        ax.set_ylabel('Delta (ticks)')
                    if row == n_rows_ns - 1:
                        ax.set_xlabel('Bar')
                    if k == 0:
                        ax.legend(fontsize=7, loc='best')

                for k in range(n_ns, n_rows_ns * n_cols_ns):
                    row, col = divmod(k, n_cols_ns)
                    axes_noise[row, col].set_visible(False)

                fig_noise.suptitle(
                    f'NOISE Segments by Nearest Shape (n={n_noise}, '
                    f'corr < {library.CORR_THRESHOLD})\nDelta from entry (raw ticks)',
                    fontsize=13)
                plt.tight_layout()
                noise_path = os.path.join(PLOTS_DIR, '0j_noise_audit.png')
                fig_noise.savefig(noise_path, dpi=150, bbox_inches='tight',
                                  facecolor='white')
                plt.close(fig_noise)
                print(f"  Saved: {noise_path}")

    # =====================================================================
    #  ANALYSIS J: RAW DELTA SUB-CLASSIFICATION (ADAPTIVE R² >= 0.80)
    #
    #  Recursive bisecting KMeans: keep splitting any sub-type with
    #  R² < 0.80 until it meets the target or runs out of segments.
    #  Raw delta from entry (ticks, not normalized).
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS J: ADAPTIVE RAW DELTA SUB-CLASSIFICATION (R\u00b2 >= 0.90)")
    print(f"{'='*70}")

    J_R2_TARGET = 0.90
    J_MIN_N = 3   # minimum segments per sub-type
    J_MIN_TOTAL = 10  # need at least this many segments to attempt
    x_j = np.arange(seed_len)
    shade_colors = {'RISE': '#4CAF50', 'DROP': '#F44336', 'HOLD': '#9E9E9E'}

    # Get all shapes with enough segments (sorted by count descending)
    j_shape_names = sorted(
        [sn for sn in set(c[1] for c in classifications) if sn != 'NOISE'],
        key=lambda sn: sum(1 for c in classifications if c[1] == sn),
        reverse=True)

    j_summary = []  # collect stats for console table
    subtype_map = {}  # idx → (shape_name, subtype_id) for Analysis K

    MISFIT_IQR_K = 1.0  # IQR multiplier for outlier detection (aggressive)

    def _norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if (mx - mn) > 1e-12 else np.zeros_like(arr)

    def _plot_subtype(ax, sub_deltas, centroid, r2_val, title_str, x_arr,
                      shade_map, r2_target):
        """Plot a single sub-type panel: spaghetti + centroid + inflections."""
        n_sub = len(sub_deltas)
        if n_sub == 0:
            ax.set_visible(False)
            return

        max_show = min(n_sub, 150)
        alpha = max(0.1, min(0.5, 15.0 / max_show))
        for j in range(max_show):
            ax.plot(x_arr, sub_deltas[j], color='#90CAF9', alpha=alpha,
                    linewidth=0.7)

        ax.plot(x_arr, centroid, color='black', linewidth=3, zorder=5)

        inflections, segs_desc = _detect_inflections(centroid)
        for sd in segs_desc:
            clr = shade_map.get(sd['label'], '#9E9E9E')
            ax.axvspan(sd['start'], sd['end'], alpha=0.08, color=clr)
        for bi, lvl in inflections:
            ax.plot(bi, lvl, 'o', color='#F44336', markersize=10, zorder=6)
            ax.annotate(f'({bi},{lvl:+.0f})',
                        xy=(bi, lvl), xytext=(5, 10),
                        textcoords='offset points', fontsize=8,
                        fontweight='bold', color='#D32F2F',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8))

        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.15)

        sub_chgs = sub_deltas[:, -1]
        sub_wr = (sub_chgs > 0).sum() / n_sub * 100 if n_sub > 0 else 0
        r2_color = '#2E7D32' if r2_val >= r2_target else '#C62828'
        ax.set_title(f'{title_str}\n'
                     f'n={n_sub}, WR={sub_wr:.0f}%, '
                     f'chg={sub_chgs.mean():+.0f}t, '
                     f'R\u00b2={r2_val:.2f}',
                     fontsize=10, fontweight='bold', color=r2_color)
        ax.set_xlabel('Bar')

    for target_shape in j_shape_names:
        entries_j = [(idx, corr, seg) for idx, sn, corr, seg in classifications
                     if sn == target_shape]
        n_seg_j = len(entries_j)

        if n_seg_j < J_MIN_TOTAL:
            continue

        deltas_j = np.array([e[2] - e[2][0] for e in entries_j])  # raw ticks

        # --- Pass 1: Adaptive split ---
        j_labels, j_centroids, j_r2s = _adaptive_split(
            deltas_j, r2_target=J_R2_TARGET, min_n=J_MIN_N)

        n_clusters = len(j_centroids)

        # --- Quality gate: IQR outlier detection in raw tick space ---
        # Segments with RMSE > Q3 + 1.5*IQR from centroid are misfits
        misfit_mask = np.zeros(len(deltas_j), dtype=bool)
        for si in range(n_clusters):
            cl_mask = (j_labels == si)
            cl_indices = np.where(cl_mask)[0]
            if len(cl_indices) < 4:
                continue  # need enough for IQR
            centroid = j_centroids[si]
            rmses = np.array([np.sqrt(np.mean((deltas_j[gi] - centroid) ** 2))
                              for gi in cl_indices])
            q1, q3 = np.percentile(rmses, [25, 75])
            iqr = q3 - q1
            fence = q3 + MISFIT_IQR_K * iqr
            for i, gi in enumerate(cl_indices):
                if rmses[i] > fence:
                    misfit_mask[gi] = True

        n_misfits = misfit_mask.sum()

        # Remove misfits from original clusters, recompute centroids/R²
        clean_labels = j_labels.copy()
        clean_labels[misfit_mask] = -1  # mark misfits

        # Rebuild clean centroids and R²s (shape-normalized)
        clean_centroids = []
        clean_r2s = []
        active_ids = []
        for si in range(n_clusters):
            cl_mask = (clean_labels == si)
            n_cl = cl_mask.sum()
            if n_cl < J_MIN_N:
                clean_labels[cl_mask] = -1  # too small after filtering
                continue
            sub = deltas_j[cl_mask]
            # Shape R²
            normed = np.array([_norm01(s) for s in sub])
            c_norm = normed.mean(axis=0)
            ss_res = np.sum((normed - c_norm[np.newaxis, :]) ** 2)
            ss_tot = np.sum((normed - normed.mean()) ** 2)
            sr2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            clean_centroids.append(sub.mean(axis=0))
            clean_r2s.append(sr2)
            active_ids.append(si)

        # --- Pass 2: Reclassify misfits ---
        all_misfit_idx = np.where(clean_labels == -1)[0]
        n_total_misfits = len(all_misfit_idx)
        reclass_centroids = []
        reclass_r2s = []
        reclass_labels = None
        unclass_deltas = None
        n_unclass = 0

        if n_total_misfits >= 2 * J_MIN_N:
            misfit_deltas = deltas_j[all_misfit_idx]
            r_labels, r_centroids, r_r2s = _adaptive_split(
                misfit_deltas, r2_target=J_R2_TARGET, min_n=J_MIN_N)
            reclass_labels = r_labels
            reclass_centroids = list(r_centroids)
            reclass_r2s = list(r_r2s)
        elif n_total_misfits > 0:
            # Too few to reclassify — all go to UNCLASSIFIED
            unclass_deltas = deltas_j[all_misfit_idx]
            n_unclass = len(unclass_deltas)

        # --- Save subtype mapping for Analysis K ---
        for ci_idx, si in enumerate(active_ids):
            cl_mask = (clean_labels == si)
            for li in np.where(cl_mask)[0]:
                orig_idx = entries_j[li][0]  # bar index in base_df
                subtype_map[orig_idx] = (target_shape, ci_idx)
        if reclass_labels is not None:
            n_clean_k = len(clean_centroids)
            for ri in range(len(reclass_centroids)):
                ri_mask = (reclass_labels == ri)
                for li in np.where(ri_mask)[0]:
                    orig_idx = entries_j[all_misfit_idx[li]][0]
                    subtype_map[orig_idx] = (target_shape, n_clean_k + ri)

        # --- Build combined plot ---
        n_clean = len(clean_centroids)
        n_reclass = len(reclass_centroids)
        has_unclass = n_unclass > 0 or (n_total_misfits > 0 and n_total_misfits < 2 * J_MIN_N)
        n_total_panels = n_clean + n_reclass + (1 if has_unclass else 0)

        if n_total_panels <= 4:
            n_cols = max(n_total_panels, 1)
            n_rows = 1
        else:
            n_cols = 4
            n_rows = (n_total_panels + 3) // 4

        fig_j, axes_j = plt.subplots(n_rows, n_cols,
                                      figsize=(6 * n_cols, 5 * n_rows),
                                      squeeze=False)

        panel_idx = 0

        # Plot clean sub-types
        for ci_idx, si in enumerate(active_ids):
            row, col = divmod(panel_idx, n_cols)
            ax = axes_j[row, col]
            cl_mask = (clean_labels == si)
            sub_d = deltas_j[cl_mask]
            _plot_subtype(ax, sub_d, clean_centroids[ci_idx],
                          clean_r2s[ci_idx],
                          f'{target_shape} sub-{ci_idx}',
                          x_j, shade_colors, J_R2_TARGET)
            if col == 0:
                ax.set_ylabel('Delta from entry (ticks)')
            panel_idx += 1

        # Plot reclassified sub-types
        if n_reclass > 0:
            misfit_deltas = deltas_j[all_misfit_idx]
            for ri in range(n_reclass):
                row, col = divmod(panel_idx, n_cols)
                ax = axes_j[row, col]
                ri_mask = (reclass_labels == ri)
                sub_d = misfit_deltas[ri_mask]
                _plot_subtype(ax, sub_d, reclass_centroids[ri],
                              reclass_r2s[ri],
                              f'{target_shape} reclass-{ri}',
                              x_j, shade_colors, J_R2_TARGET)
                # Orange border for reclassified panels
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF6F00')
                    spine.set_linewidth(3)
                if col == 0:
                    ax.set_ylabel('Delta from entry (ticks)')
                panel_idx += 1

        # Plot UNCLASSIFIED remainder
        if has_unclass:
            row, col = divmod(panel_idx, n_cols)
            ax = axes_j[row, col]
            if n_total_misfits > 0 and n_total_misfits < 2 * J_MIN_N:
                unc_d = deltas_j[all_misfit_idx]
            else:
                unc_d = unclass_deltas if unclass_deltas is not None else np.empty((0, seed_len))
            if len(unc_d) > 0:
                for j in range(min(len(unc_d), 50)):
                    ax.plot(x_j, unc_d[j], color='#FFAB91', alpha=0.5,
                            linewidth=0.8)
                ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                ax.grid(True, alpha=0.15)
                ax.set_title(f'UNCLASSIFIED\nn={len(unc_d)}',
                             fontsize=10, fontweight='bold', color='#BF360C')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#BF360C')
                    spine.set_linewidth(3)
                ax.set_xlabel('Bar')
                if col == 0:
                    ax.set_ylabel('Delta from entry (ticks)')
            panel_idx += 1

        # Hide unused axes
        for idx in range(panel_idx, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes_j[row, col].set_visible(False)

        all_r2 = clean_r2s + reclass_r2s
        min_r2 = min(all_r2) if all_r2 else 0.0
        met_target = all(r2 >= J_R2_TARGET for r2 in all_r2)
        status = f'ALL >= {J_R2_TARGET}' if met_target else f'min R\u00b2={min_r2:.2f}'

        fig_j.suptitle(
            f'Analysis J: {target_shape} (k={n_clean}+{n_reclass}r'
            f'{f"+{n_unclass}u" if has_unclass else ""})\n'
            f'{n_seg_j} segments, {n_total_misfits} filtered | {status}',
            fontsize=13,
            color='#2E7D32' if met_target else '#C62828')
        plt.tight_layout()
        fname = target_shape.lower().replace(' ', '_')
        j_path = os.path.join(PLOTS_DIR, f'0l_{fname}_raw.png')
        fig_j.savefig(j_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_j)

        j_summary.append((target_shape, n_seg_j, n_clean, n_reclass,
                          n_total_misfits, min_r2, met_target))
        print(f"  {target_shape:<25} n={n_seg_j:>3}  k={n_clean}+{n_reclass}r  "
              f"filt={n_total_misfits:>2}  min_R\u00b2={min_r2:.2f}  "
              f"{'OK' if met_target else 'BELOW'}")

    # Summary table
    print(f"\n  {'Shape':<25} {'N':>4} {'k':>3} {'rcl':>4} {'filt':>5} "
          f"{'minR2':>6} {'Status':>8}")
    print(f"  {'-'*25} {'-'*4} {'-'*3} {'-'*4} {'-'*5} {'-'*6} {'-'*8}")
    for sn, n, k, rc, fl, mr2, ok in j_summary:
        print(f"  {sn:<25} {n:>4} {k:>3} {rc:>4} {fl:>5} "
              f"{mr2:>6.2f} {'OK' if ok else 'BELOW':>8}")

    # =====================================================================
    #  ANALYSIS K: DIRECTION PREDICTION WITH FRACTAL CONTEXT
    #
    #  Blend 193D fractal properties at entry with shape classification
    #  to predict segment direction (UP/DOWN).
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS K: DIRECTION PREDICTION WITH FRACTAL CONTEXT")
    print(f"{'='*70}")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    # --- Step 1: Build direction dataset ---
    # Bridge segment bar indices to X rows via timestamp
    ts_to_xrow = {int(ts): i for i, ts in enumerate(sample_ts)}

    X_k_rows = []
    y_k = []
    shapes_k = []

    for idx, sn, corr, seg in classifications:
        if sn == 'NOISE':
            continue
        t = int(timestamps[idx])
        xrow_idx = ts_to_xrow.get(t, -1)
        if xrow_idx < 0 or xrow_idx >= len(X):
            continue
        direction = 1 if seg[-1] > seg[0] else 0
        X_k_rows.append(X[xrow_idx])
        y_k.append(direction)
        shapes_k.append(sn)

    X_k = np.array(X_k_rows)
    y_k = np.array(y_k)
    shapes_k = np.array(shapes_k)

    n_k = len(y_k)
    n_up = (y_k == 1).sum()
    n_down = (y_k == 0).sum()
    baseline = max(n_up, n_down) / n_k * 100 if n_k > 0 else 50.0

    print(f"\n  Dataset: {n_k} segments, {X_k.shape[1]} features")
    print(f"  UP: {n_up} ({n_up/n_k*100:.1f}%)  DOWN: {n_down} ({n_down/n_k*100:.1f}%)")
    print(f"  Baseline (majority class): {baseline:.1f}%")

    if n_k < 50:
        print(f"  SKIP: too few segments ({n_k}) for meaningful model")
    else:
        # --- Step 2: Train classifier ---
        X_train, X_test, y_train, y_test, sh_train, sh_test = train_test_split(
            X_k, y_k, shapes_k, test_size=0.30, random_state=42, stratify=y_k)

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        clf.fit(X_train, y_train)

        acc_train = accuracy_score(y_train, clf.predict(X_train)) * 100
        acc_test = accuracy_score(y_test, clf.predict(X_test)) * 100
        lift = acc_test - baseline

        print(f"\n  Model: GradientBoosting (200 trees, depth=4)")
        print(f"  Train accuracy: {acc_train:.1f}%")
        print(f"  Test accuracy:  {acc_test:.1f}%")
        print(f"  Lift vs baseline: {lift:+.1f}%")

        # --- Step 3: Per-shape breakdown ---
        y_pred_test = clf.predict(X_test)
        unique_shapes = sorted(set(shapes_k))

        print(f"\n  {'Shape':<25} {'N':>5} {'Base%':>7} {'Model%':>7} {'Lift':>7}")
        print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

        shape_stats_k = []
        for sn in unique_shapes:
            # Baseline from full dataset
            sn_mask_all = (shapes_k == sn)
            sn_n = sn_mask_all.sum()
            sn_base = (y_k[sn_mask_all] == 1).sum() / sn_n * 100
            sn_base = max(sn_base, 100 - sn_base)  # majority class

            # Model accuracy on test set
            sn_mask_test = (sh_test == sn)
            sn_n_test = sn_mask_test.sum()
            if sn_n_test >= 3:
                sn_acc = accuracy_score(y_test[sn_mask_test],
                                        y_pred_test[sn_mask_test]) * 100
            else:
                sn_acc = float('nan')

            sn_lift = sn_acc - sn_base if not np.isnan(sn_acc) else float('nan')
            shape_stats_k.append((sn, sn_n, sn_base, sn_acc, sn_lift))

            if not np.isnan(sn_acc):
                print(f"  {sn:<25} {sn_n:>5} {sn_base:>6.1f}% {sn_acc:>6.1f}% "
                      f"{sn_lift:>+6.1f}%")
            else:
                print(f"  {sn:<25} {sn_n:>5} {sn_base:>6.1f}%     n/a     n/a")

        # --- Step 4: Feature importance ---
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]

        print(f"\n  Top 20 fractal features for direction:")
        for rank, fi in enumerate(top_idx):
            fname = col_names[fi] if fi < len(col_names) else f'f{fi}'
            print(f"  {rank+1:>3}. {fname:<30} importance={importances[fi]:.4f}")

        # --- Step 5: Plot ---
        fig_k, axes_k = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Confusion matrix
        ax = axes_k[0, 0]
        cm = confusion_matrix(y_test, y_pred_test)
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['DOWN', 'UP'])
        ax.set_yticklabels(['DOWN', 'UP'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_title(f'Confusion Matrix\nAccuracy={acc_test:.1f}%, '
                     f'Baseline={baseline:.1f}%, Lift={lift:+.1f}%',
                     fontsize=11, fontweight='bold')

        # Panel 2: Per-shape accuracy (baseline vs model)
        ax = axes_k[0, 1]
        valid_stats = [(sn, n, b, a, l) for sn, n, b, a, l in shape_stats_k
                       if not np.isnan(a)]
        valid_stats.sort(key=lambda x: x[4], reverse=True)  # sort by lift
        if valid_stats:
            y_pos = np.arange(len(valid_stats))
            names = [s[0] for s in valid_stats]
            bases = [s[2] for s in valid_stats]
            accs = [s[3] for s in valid_stats]
            ax.barh(y_pos - 0.15, bases, 0.3, color='#90CAF9', label='Baseline')
            ax.barh(y_pos + 0.15, accs, 0.3, color='#2E7D32', label='Model')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Accuracy %')
            ax.axvline(x=50, color='gray', linewidth=0.5, linestyle='--')
            ax.legend(fontsize=9)
        ax.set_title('Per-Shape Direction Accuracy', fontsize=11, fontweight='bold')

        # Panel 3: Top 20 feature importance
        ax = axes_k[1, 0]
        top20_names = [col_names[i] if i < len(col_names) else f'f{i}'
                       for i in top_idx]
        top20_imp = importances[top_idx]
        y_pos = np.arange(20)
        ax.barh(y_pos, top20_imp[::-1], color='#FF6F00')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top20_names[::-1], fontsize=8)
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Fractal Features', fontsize=11, fontweight='bold')

        # Panel 4: Per-TF contribution
        ax = axes_k[1, 1]
        tf_contrib = np.zeros(12)
        for fi in range(192):
            tf_idx = fi // 16
            tf_contrib[tf_idx] += importances[fi]
        # Add current_MR separately
        mr_contrib = importances[192] if len(importances) > 192 else 0

        tf_labels_plot = TF_LABELS[:12] if len(TF_LABELS) >= 12 else \
            [f'TF{i}' for i in range(12)]
        x_pos = np.arange(13)
        bars = list(tf_contrib) + [mr_contrib]
        labels = list(tf_labels_plot) + ['MR']
        ax.bar(x_pos, bars, color='#1565C0')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Sum of Importances')
        ax.set_title('Per-Timeframe Contribution to Direction',
                     fontsize=11, fontweight='bold')

        fig_k.suptitle(
            f'Analysis K: Direction Prediction with Fractal Context\n'
            f'{n_k} segments, {X_k.shape[1]} features | '
            f'Accuracy={acc_test:.1f}%, Lift={lift:+.1f}%',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        k_path = os.path.join(PLOTS_DIR, '0m_direction_prediction.png')
        fig_k.savefig(k_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_k)
        print(f"\n  Saved: {k_path}")

    # =====================================================================
    #  ANALYSIS L: SIGNED MFE OLS (direction from price prediction)
    #
    #  Fit OLS: Y = signed_MFE = MFE * sign(direction)
    #  If we can predict signed MFE, sign gives direction and magnitude
    #  gives confidence. One model replaces direction + quality classifiers.
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS L: SIGNED MFE OLS (DIRECTION FROM PRICE PREDICTION)")
    print(f"  Y = MFE * sign(direction)  |  positive=UP, negative=DOWN")
    print(f"  sign(prediction) -> direction,  |prediction| -> confidence")
    print(f"{'='*70}")

    from sklearn.linear_model import LinearRegression as _LR_L
    from sklearn.preprocessing import StandardScaler as _SS_L
    from sklearn.model_selection import train_test_split as _split_L

    # Bridge oracle bars to X rows via timestamp
    base_ts = base_df['timestamp'].values
    _oracle_ts_set = {}
    for i, bi in enumerate(bar_indices):
        _oracle_ts_set[int(base_ts[bi])] = i

    _l_xrows = []
    _l_smfe = []
    for xi, ts_val in enumerate(sample_ts):
        oi = _oracle_ts_set.get(int(ts_val), -1)
        if oi >= 0:
            _l_xrows.append(xi)
            _sign = 1.0 if directions[oi] == 'LONG' else -1.0
            _l_smfe.append(float(mfes[oi]) * _sign)

    n_l = len(_l_smfe)
    print(f"\n  Matched samples: {n_l} (oracle bars with fractal context)")

    if n_l >= 50:
        X_l = X[_l_xrows]
        Y_l = np.array(_l_smfe)

        n_pos = (Y_l > 0).sum()
        n_neg = (Y_l < 0).sum()
        print(f"  UP (positive): {n_pos} ({n_pos/n_l*100:.1f}%)  "
              f"DOWN (negative): {n_neg} ({n_neg/n_l*100:.1f}%)")
        print(f"  Y range: [{Y_l.min():.1f}, {Y_l.max():.1f}], "
              f"mean={Y_l.mean():.2f}, std={Y_l.std():.2f}")

        # Train/test split
        X_tr, X_te, y_tr, y_te = _split_L(X_l, Y_l, test_size=0.30, random_state=42)

        sc_l = _SS_L()
        X_tr_sc = sc_l.fit_transform(X_tr)
        X_te_sc = sc_l.transform(X_te)

        ols_l = _LR_L().fit(X_tr_sc, y_tr)
        pred_tr = ols_l.predict(X_tr_sc)
        pred_te = ols_l.predict(X_te_sc)

        # R² on train and test
        r2_tr = ols_l.score(X_tr_sc, y_tr)
        r2_te = ols_l.score(X_te_sc, y_te)
        n_te, k_te = X_te_sc.shape
        adj_r2_te = 1.0 - (1.0 - r2_te) * (n_te - 1) / max(1, n_te - k_te - 1)

        print(f"\n  OLS Signed MFE:")
        print(f"    Train R2:     {r2_tr:.4f}")
        print(f"    Test R2:      {r2_te:.4f}")
        print(f"    Test adj-R2:  {adj_r2_te:.4f}")

        # Direction accuracy: sign(predicted) vs sign(actual)
        dir_pred = np.sign(pred_te)
        dir_actual = np.sign(y_te)
        _nz = dir_actual != 0
        if _nz.sum() > 0:
            dir_correct = (dir_pred[_nz] == dir_actual[_nz]).sum()
            dir_acc = dir_correct / _nz.sum()
            _baseline_l = max((dir_actual[_nz] > 0).sum(), (dir_actual[_nz] < 0).sum()) / _nz.sum()
            _lift_l = dir_acc - _baseline_l
            print(f"\n  Direction from sign(prediction):")
            print(f"    Accuracy: {dir_correct}/{_nz.sum()} = {dir_acc:.1%}")
            print(f"    Baseline (majority): {_baseline_l:.1%}")
            print(f"    Lift: {_lift_l:+.1%}")

            # Confidence gates: only predict when |predicted| > threshold
            print(f"\n  Confidence gates (|predicted signed MFE| > threshold):")
            print(f"  {'Threshold':>10} {'N':>6} {'Accuracy':>10} {'Lift':>8} {'% of data':>10}")
            print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
            for thr in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
                _cm = (np.abs(pred_te) > thr) & _nz
                if _cm.sum() > 0:
                    _cc = (dir_pred[_cm] == dir_actual[_cm]).sum()
                    _ca = _cc / _cm.sum()
                    _pct = _cm.sum() / _nz.sum() * 100
                    print(f"  {thr:>10.1f} {_cm.sum():>6} {_ca:>10.1%} "
                          f"{_ca - _baseline_l:>+8.1%} {_pct:>9.1f}%")

            # LONG vs SHORT breakdown
            _pred_long = dir_pred[_nz] > 0
            _actual_long = dir_actual[_nz] > 0
            _long_correct = (_pred_long & _actual_long).sum()
            _long_total = _actual_long.sum()
            _short_correct = (~_pred_long & ~_actual_long).sum()
            _short_total = (~_actual_long).sum()
            print(f"\n  When actual LONG:  {_long_correct}/{_long_total} = "
                  f"{_long_correct/_long_total:.1%}" if _long_total > 0 else "")
            print(f"  When actual SHORT: {_short_correct}/{_short_total} = "
                  f"{_short_correct/_short_total:.1%}" if _short_total > 0 else "")

        # Top features by coefficient magnitude
        coeff_abs = np.abs(ols_l.coef_)
        top_idx = np.argsort(coeff_abs)[::-1][:20]
        all_names = col_names  # 193 features
        print(f"\n  TOP 20 FEATURES (by |coefficient| in signed MFE OLS):")
        print(f"  {'Rank':>4}  {'Feature':<40} {'Coeff':>10} {'|Coeff|':>10}")
        print(f"  {'-'*4}  {'-'*40} {'-'*10} {'-'*10}")
        for rank, fi in enumerate(top_idx, 1):
            fn = all_names[fi] if fi < len(all_names) else f'f{fi}'
            print(f"  {rank:>4}  {fn:<40} {ols_l.coef_[fi]:>+10.4f} {coeff_abs[fi]:>10.4f}")

        # CONCLUSION
        print(f"\n  ANALYSIS L CONCLUSION:")
        if _nz.sum() > 0 and dir_acc > 0.55:
            print(f"  PROMISING: {dir_acc:.1%} direction accuracy from signed MFE OLS.")
            print(f"  The 16D fractal context can predict not just WHERE price is,")
            print(f"  but which WAY it's going and how FAR. One regression gives")
            print(f"  direction (sign) + confidence (magnitude) + TP target (|pred|).")
        elif _nz.sum() > 0 and dir_acc > 0.52:
            print(f"  MARGINAL: {dir_acc:.1%} accuracy, slight lift over baseline.")
            print(f"  May improve with importance weighting or feature selection.")
        else:
            print(f"  INSUFFICIENT: {dir_acc:.1%} accuracy. Signed MFE is not reliably")
            print(f"  predictable from the 192D snapshot. Fall back to balanced")
            print(f"  direction classifier or template DMI side.")

        # ── Plot: Signed MFE — Predicted vs Actual ──────────────────────
        fig_l, axes_l = plt.subplots(2, 2, figsize=(16, 12),
                                      facecolor='white')

        # (0,0) Scatter: predicted vs actual signed MFE, color = actual direction
        ax = axes_l[0, 0]
        _c_long  = '#2196F3'  # blue = LONG (up)
        _c_short = '#F44336'  # red  = SHORT (down)
        _colors_te = np.where(y_te > 0, _c_long, _c_short)
        ax.scatter(y_te, pred_te, c=_colors_te, alpha=0.5, s=20, edgecolors='none')
        _lim = max(abs(y_te).max(), abs(pred_te).max()) * 1.1
        ax.plot([-_lim, _lim], [-_lim, _lim], 'k--', alpha=0.3, lw=1)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
        # Shade quadrants
        ax.fill_between([-_lim, 0], -_lim, 0, color=_c_short, alpha=0.04)  # correct SHORT
        ax.fill_between([0, _lim], 0, _lim, color=_c_long, alpha=0.04)     # correct LONG
        ax.set_xlabel('Actual Signed MFE', fontsize=10)
        ax.set_ylabel('Predicted Signed MFE', fontsize=10)
        ax.set_title(f'Predicted vs Actual (R\u00b2={r2_te:.3f})', fontsize=11, fontweight='bold')
        # Legend
        from matplotlib.lines import Line2D
        _leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=_c_long, markersize=8, label='LONG (actual)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=_c_short, markersize=8, label='SHORT (actual)')]
        ax.legend(handles=_leg, loc='upper left', fontsize=9)

        # (0,1) Histogram: predicted signed MFE distribution, stacked by actual direction
        ax = axes_l[0, 1]
        _pred_long_vals  = pred_te[y_te > 0]
        _pred_short_vals = pred_te[y_te < 0]
        _bins = np.linspace(-_lim, _lim, 40)
        ax.hist(_pred_long_vals, bins=_bins, alpha=0.7, color=_c_long, label='Actual LONG', edgecolor='white', lw=0.5)
        ax.hist(_pred_short_vals, bins=_bins, alpha=0.7, color=_c_short, label='Actual SHORT', edgecolor='white', lw=0.5)
        ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
        ax.set_xlabel('Predicted Signed MFE', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Prediction Distribution by Actual Direction', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.text(0.02, 0.95, f'LEFT of 0 = model says SHORT\nRIGHT of 0 = model says LONG',
                transform=ax.transAxes, fontsize=8, va='top', color='gray')

        # (1,0) Confusion matrix as heatmap
        ax = axes_l[1, 0]
        if _nz.sum() > 0:
            _cm_labels = ['SHORT', 'LONG']
            _tp_short = (~_pred_long & ~_actual_long).sum()
            _fp_long  = (_pred_long & ~_actual_long).sum()
            _fn_long  = (~_pred_long & _actual_long).sum()
            _tp_long  = (_pred_long & _actual_long).sum()
            _cm = np.array([[_tp_short, _fp_long], [_fn_long, _tp_long]])
            _im = ax.imshow(_cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1]); ax.set_xticklabels(_cm_labels, fontsize=10)
            ax.set_yticks([0, 1]); ax.set_yticklabels(_cm_labels, fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
            for _ri in range(2):
                for _ci in range(2):
                    _val = _cm[_ri, _ci]
                    _clr = 'white' if _val > _cm.max() * 0.5 else 'black'
                    ax.text(_ci, _ri, str(_val), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=_clr)
            ax.set_title(f'Direction Confusion Matrix\nAccuracy={dir_acc:.1%}, Lift={_lift_l:+.1%}',
                        fontsize=11, fontweight='bold')

        # (1,1) Confidence gate curve
        ax = axes_l[1, 1]
        _thrs = np.linspace(0, np.percentile(np.abs(pred_te), 95), 30)
        _accs = []
        _ns = []
        for _t in _thrs:
            _m = (np.abs(pred_te) > _t) & _nz
            if _m.sum() >= 5:
                _accs.append((dir_pred[_m] == dir_actual[_m]).sum() / _m.sum() * 100)
                _ns.append(_m.sum() / _nz.sum() * 100)
            else:
                _accs.append(np.nan)
                _ns.append(0)
        ax.plot(_thrs, _accs, color='#2196F3', lw=2, label='Accuracy %')
        ax.axhline(_baseline_l * 100, color='gray', ls='--', lw=1, alpha=0.7, label=f'Baseline {_baseline_l:.0%}')
        ax.set_xlabel('|Predicted Signed MFE| Threshold', fontsize=10)
        ax.set_ylabel('Direction Accuracy %', fontsize=10)
        ax.set_title('Confidence Gate: Accuracy vs Threshold', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax2 = ax.twinx()
        ax2.fill_between(_thrs, 0, _ns, alpha=0.15, color='orange')
        ax2.set_ylabel('% of Data Remaining', fontsize=9, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        fig_l.suptitle(
            f'Analysis L: Signed MFE OLS — Direction from Price Prediction\n'
            f'{n_l} samples | Accuracy={dir_acc:.1%} | Lift={_lift_l:+.1%} | '
            f'LONG: {n_pos} ({n_pos/n_l*100:.0f}%)  SHORT: {n_neg} ({n_neg/n_l*100:.0f}%)',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        l_path = os.path.join(PLOTS_DIR, '0n_signed_mfe_direction.png')
        fig_l.savefig(l_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_l)
        print(f"\n  Saved: {l_path}")

        # ── Plot 2: Price chart with LONG/SHORT segment overlay ─────────
        # Predict signed MFE for ALL matched samples (not just test split)
        X_l_all_sc = sc_l.transform(X_l)
        pred_all = ols_l.predict(X_l_all_sc)
        pred_dir_all = np.sign(pred_all)

        # Map back to bar indices in base_df
        _matched_bar_idx = [bar_indices[_oracle_ts_set[int(sample_ts[xi])]]
                            for xi in _l_xrows]
        _matched_bar_idx = np.array(_matched_bar_idx)

        from datetime import datetime, timezone as _tz_l
        close_all = base_df['close'].values.astype(float)
        ts_all = base_df['timestamp'].values

        fig_p, ax_p = plt.subplots(1, 1, figsize=(20, 7), facecolor='white')

        # Plot full price line in gray
        _x_dates = [datetime.fromtimestamp(int(t), tz=_tz_l.utc) for t in ts_all]
        ax_p.plot(_x_dates, close_all, color='#BDBDBD', lw=0.8, alpha=0.6, zorder=1)

        # Overlay colored segments at each prediction point
        # Draw a short colored line segment around each prediction bar
        _seg_half = max(1, len(close_all) // 500)  # adaptive segment width
        for i, bi in enumerate(_matched_bar_idx):
            _s = max(0, bi - _seg_half)
            _e = min(len(close_all), bi + _seg_half + 1)
            _seg_x = _x_dates[_s:_e]
            _seg_y = close_all[_s:_e]
            if len(_seg_x) < 2:
                continue
            _color = '#2196F3' if pred_dir_all[i] > 0 else '#F44336'
            _alpha = min(1.0, 0.3 + abs(pred_all[i]) / 100.0)  # stronger prediction = more opaque
            ax_p.plot(_seg_x, _seg_y, color=_color, lw=2.0, alpha=_alpha, zorder=2)

        # Mark correct/wrong with small dots
        for i, bi in enumerate(_matched_bar_idx):
            _actual_sign = 1.0 if directions[_oracle_ts_set[int(sample_ts[_l_xrows[i]])]] == 'LONG' else -1.0
            _correct = (pred_dir_all[i] == _actual_sign)
            if not _correct:
                ax_p.plot(_x_dates[bi], close_all[bi], 'x', color='black',
                         markersize=4, alpha=0.5, zorder=3)

        ax_p.set_xlabel('Time', fontsize=10)
        ax_p.set_ylabel('Price', fontsize=10)
        ax_p.set_title(
            f'Price with Predicted Direction Overlay\n'
            f'Blue = LONG prediction | Red = SHORT prediction | X = wrong direction',
            fontsize=12, fontweight='bold')
        from matplotlib.lines import Line2D as _Line2D_p
        _leg_p = [_Line2D_p([0], [0], color='#2196F3', lw=2, label='Predicted LONG'),
                  _Line2D_p([0], [0], color='#F44336', lw=2, label='Predicted SHORT'),
                  _Line2D_p([0], [0], marker='x', color='black', lw=0, markersize=6, label='Wrong direction')]
        ax_p.legend(handles=_leg_p, loc='upper left', fontsize=9)
        fig_p.autofmt_xdate()
        plt.tight_layout()
        p_path = os.path.join(PLOTS_DIR, '0o_price_direction_overlay.png')
        fig_p.savefig(p_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_p)
        print(f"  Saved: {p_path}")

    else:
        print(f"  SKIP: too few matched samples ({n_l}) for meaningful analysis")

    # Save report and exit (default mode)
    if not args.full:
        sys.stdout = _orig_stdout
        report_dir = os.path.dirname(__file__)

        # Save combined report
        report_path = os.path.join(report_dir, 'standalone_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"STANDALONE WAVEFORM SCREENING REPORT\n")
            f.write(f"Data: {args.data}, Base TF: {args.base_tf}\n")
            f.write(f"Context: {args.context_days}d, Analysis: {args.analysis_days}d\n")
            f.write(f"Samples: {len(Y_p)}\n")
            f.write(f"Price adj-R2: {r2_p:.4f}, Direction adj-R2: {r2_d:.4f}\n")
            f.write(f"Derived direction accuracy: {accuracy:.1%}\n\n")
            f.write(_report_buf.getvalue())
        print(f"\n  Report saved: {report_path}")
        print(f"  Charts: {PLOTS_DIR}/")
        return

    # =====================================================================
    #  FULL MODE: 16D fractal pipeline (Steps 7-14)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  FULL 16D FRACTAL PIPELINE")
    print(f"{'='*70}")

    # --- 7. Load all TFs + compute physics ---
    print(f"\n--- STEP 7: Loading all TF data + physics ---")
    all_dfs = {args.base_tf: base_df}
    for tf in TF_HIERARCHY:
        if tf == args.base_tf:
            continue
        df = load_atlas_tf(args.data, tf, months=args.months)
        if not df.empty:
            all_dfs[tf] = df
            print(f"  {tf:>4}: {len(df):>8,} bars")
        else:
            print(f"  {tf:>4}:   (not found)")

    print(f"\n  Computing physics per TF...")
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="Physics", unit="tf"):
        if tf not in all_dfs:
            continue
        states = compute_tf_physics(tf, all_dfs[tf])
        if states:
            all_tf_states[tf] = states
            print(f"  {tf:>4}: {len(states):>8,} states computed")

    # --- 8. Build stacked hypervolume matrices (regime-based segmentation) ---
    print(f"\n--- STEP 8: Building fractal hypervolume matrices ---")
    matrices, mfes_16d, maes_16d, meta = build_stacked_matrices(
        all_tf_states, args.base_tf, all_dfs[args.base_tf],
        context_days=args.context_days,
        analysis_days=args.analysis_days
    )

    if len(matrices) < 20:
        print(f"ERROR: Only {len(matrices)} matrices built (need >= 20)")
        sys.exit(1)

    # Replace ADX quartile bins with regime IDs in meta
    ts_to_regime = {}
    timestamps = base_df['timestamp'].values.astype(float)
    for i, ts in enumerate(timestamps):
        if regime_ids[i] >= 0:
            ts_to_regime[int(ts)] = regime_ids[i]

    for m in meta:
        rid = ts_to_regime.get(int(m['ts']), -1)
        if rid >= 0:
            m['tid'] = f'regime_{rid}'
        # keep existing tid if no regime match

    # --- 9. Pad + I-MR chart plots ---
    print(f"\n--- STEP 9: Fractal I-MR Charts ---")
    padded = pad_to_fixed_depth(matrices, max_depth=12)
    plot_imr_charts(padded, mfes_16d)

    # Use 16D mfes/maes for the full pipeline from here on
    mfes = mfes_16d
    maes = maes_16d

    # --- 10. Flatten + MR segmentation ---
    print(f"\n--- STEP 10: MR segmentation ---")
    flat_i, col_names_i = flatten_matrices(padded)
    flat_mr, col_names_mr = compute_moving_range(padded)

    flat_z = np.hstack([flat_i, flat_mr])
    col_names_z = col_names_i + col_names_mr
    print(f"  Combined: {len(col_names_i)} I + {len(col_names_mr)} MR "
          f"= {len(col_names_z)} total features")

    # --- 11. Screen all three: I-only, MR-only, combined ---
    print(f"\n{'='*70}")
    print(f"  SCREENING X: Raw I values ({len(col_names_i)} features)")
    print(f"{'='*70}")
    results_i = screen_factors(flat_i, col_names_i, mfes)
    print_screening_report(results_i, mfes, maes, meta, top_n=args.top)
    steps_i = regression_r2(flat_i, col_names_i, mfes, top_k=20)

    print(f"\n{'='*70}")
    print(f"  SCREENING Y: MR Segmentation ({len(col_names_mr)} features)")
    print(f"{'='*70}")
    results_mr = screen_factors(flat_mr, col_names_mr, mfes)
    print(f"\n  TOP {args.top} MR FACTORS:")
    print(f"  {'Rank':>4}  {'Factor':<40} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*40} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results_mr[:args.top], 1):
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  {name:<40} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")
    steps_mr = regression_r2(flat_mr, col_names_mr, mfes, top_k=20)

    print(f"\n{'='*70}")
    print(f"  SCREENING Z: X + Y Combined ({len(col_names_z)} features)")
    print(f"{'='*70}")
    results_z = screen_factors(flat_z, col_names_z, mfes)
    print(f"\n  TOP {args.top} COMBINED FACTORS:")
    print(f"  {'Rank':>4}  {'Factor':<40} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*40} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results_z[:args.top], 1):
        src = 'I' if name in col_names_i else 'MR'
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  [{src}] {name:<37} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")
    steps_z = regression_r2(flat_z, col_names_z, mfes, top_k=20)

    # --- 12. Summary comparison ---
    r2_i = steps_i[-1][3] if steps_i else 0
    r2_mr = steps_mr[-1][3] if steps_mr else 0
    r2_z = steps_z[-1][3] if steps_z else 0
    print(f"\n{'='*70}")
    print(f"  COMPARISON: adj-R² @ 20 factors")
    print(f"{'='*70}")
    print(f"  X (I values only):     {r2_i:.4f}")
    print(f"  Y (MR segments only):  {r2_mr:.4f}")
    print(f"  Z (X + Y combined):    {r2_z:.4f}")
    print(f"  Lift from MR:          {r2_z - r2_i:+.4f}")

    # --- 13. Directional split (16D pipeline uses DMI from meta) ---
    dmi_float = np.array([float(m.get('dmi_diff', 0)) for m in meta])
    long_mask = dmi_float >= 0
    short_mask = ~long_mask

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    wr_long = float((mfes[long_mask] > maes[long_mask]).mean()) if n_long > 0 else 0
    wr_short = float((mfes[short_mask] > maes[short_mask]).mean()) if n_short > 0 else 0

    print(f"\n{'='*70}")
    print(f"  DIRECTIONAL SPLIT (16D)")
    print(f"{'='*70}")
    print(f"  LONG  (DMI >= 0): {n_long:>5} points, WR={wr_long:.1%}")
    print(f"  SHORT (DMI <  0): {n_short:>5} points, WR={wr_short:.1%}")
    print(f"  Mixed WR:         {float((mfes > maes).mean()):.1%}")

    # --- 14. Segmented screening ---
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from collections import Counter, defaultdict

    tids = np.array([m['tid'] for m in meta])
    unique_tids = sorted(set(tids))

    print(f"\n{'='*70}")
    print(f"  SEGMENTED SCREENING: {len(unique_tids)} segments x 2 directions")
    print(f"{'='*70}")

    global_mfe_mean = float(np.mean(mfes))
    global_mfe_std = float(np.std(mfes))

    seg_results = []
    for tid in unique_tids:
        for dir_name, dir_mask in [('LONG', long_mask), ('SHORT', short_mask)]:
            seg_mask = (tids == tid) & dir_mask
            n_seg = seg_mask.sum()
            if n_seg < 15:
                continue

            seg_mfes = mfes[seg_mask]
            seg_maes = maes[seg_mask]
            seg_flat = flat_z[seg_mask]

            seg_screening = screen_factors(seg_flat, col_names_z, seg_mfes)
            top1_name, top1_corr, top1_abs = seg_screening[0]

            top5_names = [s[0] for s in seg_screening[:5]]
            top5_idx = [col_names_z.index(n) for n in top5_names]
            scaler = StandardScaler()
            X = scaler.fit_transform(seg_flat[:, top5_idx])
            reg = LinearRegression().fit(X, seg_mfes)
            r2 = reg.score(X, seg_mfes)
            n, k = X.shape
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)

            # MR entry/exit signals
            mr_factors = [(nm, c, a) for nm, c, a in seg_screening
                          if nm.startswith('MR_') and a > 0.20]
            entry_signals = [(nm, c) for nm, c, a in mr_factors if c > 0]
            exit_signals = [(nm, c) for nm, c, a in mr_factors if c < 0]

            # Cpk / Ppk
            seg_mean = float(np.mean(seg_mfes))
            seg_std = float(np.std(seg_mfes))
            cpk = seg_mean / (3 * seg_std) if seg_std > 1e-6 else 0.0
            ppk = seg_mean / (3 * global_mfe_std) if global_mfe_std > 1e-6 else 0.0

            win_rate = float((seg_mfes > seg_maes).mean())
            p_positive = float((seg_mfes > 0).mean())
            p_good = float((seg_mfes > seg_mean).mean())

            median_mfe = float(np.median(seg_mfes))
            good_mask_seg = seg_mfes >= median_mfe
            bad_mask_seg = ~good_mask_seg

            good_bad_diff = []
            if good_mask_seg.sum() >= 5 and bad_mask_seg.sum() >= 5:
                for j, cname in enumerate(col_names_z):
                    col = seg_flat[:, j]
                    if np.std(col) < 1e-12:
                        continue
                    good_mean = float(np.mean(col[good_mask_seg]))
                    bad_mean = float(np.mean(col[bad_mask_seg]))
                    pooled_std = float(np.std(col))
                    if pooled_std > 1e-12:
                        effect_size = (good_mean - bad_mean) / pooled_std
                        good_bad_diff.append((cname, effect_size, abs(effect_size)))
                good_bad_diff.sort(key=lambda x: x[2], reverse=True)

            seg_results.append({
                'tid': tid, 'dir': dir_name, 'n': n_seg,
                'seg_id': f"{dir_name[0]}_{tid}",
                'mfe_mean': seg_mean, 'mfe_std': seg_std,
                'mae_mean': float(np.mean(seg_maes)),
                'top1': top1_name, 'top1_corr': top1_corr,
                'adj_r2_5': adj_r2,
                'top5': [(s[0], s[1]) for s in seg_screening[:5]],
                'entry_signals': entry_signals[:5],
                'exit_signals': exit_signals[:5],
                'cpk': cpk, 'ppk': ppk,
                'win_rate': win_rate, 'p_positive': p_positive, 'p_good': p_good,
                'good_bad_top5': good_bad_diff[:5],
                'median_mfe': median_mfe,
                'good_mfe_mean': float(np.mean(seg_mfes[good_mask_seg])),
                'bad_mfe_mean': float(np.mean(seg_mfes[bad_mask_seg])),
            })

    # Extract dominant context feature for each segment
    def _extract_feature(factor_name):
        parts = factor_name.split('__')
        return parts[-1] if parts else factor_name

    def _extract_depth(factor_name):
        if factor_name.startswith('MR_'):
            return factor_name.split('__')[0].replace('MR_', '')
        elif factor_name.startswith('UCL_'):
            return factor_name.split('__')[0].replace('UCL_', '')
        elif factor_name.startswith('slope__') or factor_name.startswith('mr_bar__') or factor_name.startswith('n_breaks__'):
            return 'all'
        else:
            return factor_name.split('__')[0]

    for s in seg_results:
        top5_features = [_extract_feature(fn) for fn, _ in s['top5']]
        feat_counts = Counter(top5_features)
        s['dominant_feature'] = feat_counts.most_common(1)[0][0]
        s['top1_feature'] = _extract_feature(s['top1'])
        s['top1_depth'] = _extract_depth(s['top1'])
        s['top1_src'] = 'I' if s['top1'] in col_names_i else 'MR'
        s['feature_profile'] = list(dict.fromkeys(top5_features))

    # --- 15. Model fission ---
    all_sorted = sorted(seg_results, key=lambda x: x['win_rate'], reverse=True)

    for s in all_sorted:
        s['snr'] = s['mfe_mean'] / s['mfe_std'] if s['mfe_std'] > 1e-6 else 0.0

    print(f"\n{'='*70}")
    print(f"  MODEL FISSION: Segment x Direction (sorted by P(success))")
    print(f"  P(win) = MFE > MAE.  SNR = mean/std.  Action = KEEP/SPLIT/DROP")
    print(f"{'='*70}")
    print(f"  {'Seg ID':<10} {'Ctx':<12} {'N':>4} {'P(win)':>7} {'P(>0)':>6} "
          f"{'SNR':>5} {'R2':>5} {'MFE':>6} {'MAE':>5} {'Action':<7}")
    print(f"  {'-'*10} {'-'*12} {'-'*4} {'-'*7} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*7}")

    keep_segs, split_segs, drop_segs = [], [], []
    for s in all_sorted:
        if s['win_rate'] >= 0.65 and s['snr'] >= 0.5:
            action = 'KEEP'
            keep_segs.append(s)
        elif s['win_rate'] >= 0.50:
            action = 'SPLIT'
            split_segs.append(s)
        else:
            action = 'DROP'
            drop_segs.append(s)

        wr_bar = '#' * int(s['win_rate'] * 20)
        print(f"  {s['seg_id']:<10} {s['dominant_feature']:<12} {s['n']:>4} "
              f"{s['win_rate']:>7.1%} {s['p_positive']:>6.0%} "
              f"{s['snr']:>5.2f} {s['adj_r2_5']:>5.2f} "
              f"{s['mfe_mean']:>+6.0f} {s['mae_mean']:>5.0f} "
              f"{action:<7} {wr_bar}")

    # KEEP segments detail
    if keep_segs:
        keep_n = sum(s['n'] for s in keep_segs)
        keep_wr = np.average([s['win_rate'] for s in keep_segs],
                             weights=[s['n'] for s in keep_segs])
        keep_mfe = np.average([s['mfe_mean'] for s in keep_segs],
                              weights=[s['n'] for s in keep_segs])
        print(f"\n  KEEP ({len(keep_segs)} segments, {keep_n} patterns, "
              f"WR={keep_wr:.1%}, avg MFE={keep_mfe:+.0f}):")
        for s in keep_segs:
            entry_str = ', '.join(
                f"{nm.split('__')[-1]}@{nm.split('__')[0].replace('MR_','')}"
                for nm, c in s['entry_signals'][:2]) or 'none'
            exit_str = ', '.join(
                f"{nm.split('__')[-1]}@{nm.split('__')[0].replace('MR_','')}"
                for nm, c in s['exit_signals'][:2]) or 'none'
            print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                  f"WR={s['win_rate']:.0%} MFE={s['mfe_mean']:+.0f}")
            print(f"      Entry: {entry_str}")
            print(f"      Exit:  {exit_str}")
            if s['good_bad_top5']:
                top_diff = s['good_bad_top5'][0]
                src = 'I' if top_diff[0] in col_names_i else 'MR'
                dir_str = 'higher' if top_diff[1] > 0 else 'lower'
                print(f"      Good vs Bad: {top_diff[0]} {dir_str} in winners (d={top_diff[1]:+.2f})")

    # SPLIT segments
    if split_segs:
        split_n = sum(s['n'] for s in split_segs)
        split_wr = np.average([s['win_rate'] for s in split_segs],
                              weights=[s['n'] for s in split_segs])
        print(f"\n  SPLIT ({len(split_segs)} segments, {split_n} patterns, "
              f"WR={split_wr:.1%}) -- signal exists but noisy, need finer cuts:")
        for s in split_segs:
            if s['good_bad_top5']:
                split_feature = s['good_bad_top5'][0][0]
                print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                      f"WR={s['win_rate']:.0%} -- split on: {split_feature}")
            else:
                print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                      f"WR={s['win_rate']:.0%}")

    # DROP segments
    if drop_segs:
        drop_n = sum(s['n'] for s in drop_segs)
        drop_mfe = np.average([s['mfe_mean'] for s in drop_segs],
                              weights=[s['n'] for s in drop_segs])
        print(f"\n  DROP ({len(drop_segs)} segments, {drop_n} patterns, "
              f"avg MFE={drop_mfe:+.0f}) -- net noise, remove from model:")
        for s in drop_segs:
            print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                  f"WR={s['win_rate']:.0%} MFE={s['mfe_mean']:+.0f} "
                  f"MAE={s['mae_mean']:.0f}")

    # --- 15b. Segmented I-MR plots ---
    print(f"\n--- Generating segmented I-MR plots ---")
    plot_segmented_imr(padded, mfes, maes, meta, tids, long_mask,
                       keep_segs, split_segs, drop_segs,
                       base_df=all_dfs.get(args.base_tf))

    # --- 16. Export gate config ---
    import json as _json

    _fission_map = {}
    for s in keep_segs:
        _dir = 'long' if s['seg_id'].startswith('L_') else 'short'
        _tid = s['seg_id'].split('_', 1)[1]
        _fission_map[f"{_tid}_{_dir}"] = 'KEEP'
    for s in split_segs:
        _dir = 'long' if s['seg_id'].startswith('L_') else 'short'
        _tid = s['seg_id'].split('_', 1)[1]
        _fission_map[f"{_tid}_{_dir}"] = 'SPLIT'

    _gate_config = {
        'fission_map': _fission_map,
        'good_hours_utc': [0, 5, 17, 18, 19, 20],
        'default_class': 'DROP',
    }
    _gate_path = os.path.join(os.path.dirname(__file__), 'screening_gates.json')
    with open(_gate_path, 'w') as _gf:
        _json.dump(_gate_config, _gf, indent=2)
    print(f"\n  >> Exported screening gates to {_gate_path}")
    print(f"     KEEP: {sum(1 for v in _fission_map.values() if v == 'KEEP')}, "
          f"SPLIT: {sum(1 for v in _fission_map.values() if v == 'SPLIT')}, "
          f"hours: {_gate_config['good_hours_utc']}")

    # --- 17. What-if impact ---
    total_n = sum(s['n'] for s in seg_results)
    total_wr = np.average([s['win_rate'] for s in seg_results],
                          weights=[s['n'] for s in seg_results]) if seg_results else 0
    total_mfe = np.average([s['mfe_mean'] for s in seg_results],
                           weights=[s['n'] for s in seg_results]) if seg_results else 0

    print(f"\n{'='*70}")
    print(f"  WHAT-IF: Fission Impact")
    print(f"{'='*70}")
    print(f"  CURRENT (all segments):")
    print(f"    Segments: {len(seg_results)}, Patterns: {total_n}, "
          f"WR: {total_wr:.1%}, MFE: {total_mfe:+.0f}")

    if keep_segs:
        keep_total_n = sum(s['n'] for s in keep_segs)
        keep_total_wr = np.average([s['win_rate'] for s in keep_segs],
                                   weights=[s['n'] for s in keep_segs])
        keep_total_mfe = np.average([s['mfe_mean'] for s in keep_segs],
                                    weights=[s['n'] for s in keep_segs])
        print(f"  KEEP ONLY:")
        print(f"    Segments: {len(keep_segs)}, Patterns: {keep_total_n}, "
              f"WR: {keep_total_wr:.1%}, MFE: {keep_total_mfe:+.0f}")
        print(f"    Dropped: {total_n - keep_total_n} patterns "
              f"({(total_n - keep_total_n)/total_n:.0%} of volume)")
        print(f"    WR lift: {keep_total_wr - total_wr:+.1%}")

    if keep_segs or split_segs:
        ks = keep_segs + split_segs
        ks_n = sum(s['n'] for s in ks)
        ks_wr = np.average([s['win_rate'] for s in ks], weights=[s['n'] for s in ks])
        ks_mfe = np.average([s['mfe_mean'] for s in ks], weights=[s['n'] for s in ks])
        print(f"  KEEP + SPLIT (before refining splits):")
        print(f"    Segments: {len(ks)}, Patterns: {ks_n}, "
              f"WR: {ks_wr:.1%}, MFE: {ks_mfe:+.0f}")

    # --- 18. PID drill-down ---
    pid_idx = FEATURE_NAMES.index('self_pid')  # 14

    print(f"\n{'='*70}")
    print(f"  PID DRILL-DOWN: I-MR x Direction")
    print(f"{'='*70}")

    # PID I-chart
    print(f"\n  PID I-CHART (mean value at each depth):")
    print(f"  {'Depth':<12} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'r(MFE)L':>9} {'r(MFE)S':>9}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    for d in range(12):
        pid_col = padded[:, d, pid_idx]
        l_mean = float(np.mean(pid_col[long_mask])) if long_mask.sum() > 0 else 0
        s_mean = float(np.mean(pid_col[short_mask])) if short_mask.sum() > 0 else 0
        corr_l = float(np.corrcoef(pid_col[long_mask], mfes[long_mask])[0, 1]) \
            if long_mask.sum() > 10 and np.std(pid_col[long_mask]) > 1e-12 else 0.0
        corr_s = float(np.corrcoef(pid_col[short_mask], mfes[short_mask])[0, 1]) \
            if short_mask.sum() > 10 and np.std(pid_col[short_mask]) > 1e-12 else 0.0
        if np.isnan(corr_l): corr_l = 0.0
        if np.isnan(corr_s): corr_s = 0.0
        lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        print(f"  {lbl:<12} {l_mean:>+8.2f} {s_mean:>+8.2f} {l_mean - s_mean:>+8.2f} "
              f"{corr_l:>+9.4f} {corr_s:>+9.4f}")

    # PID MR
    mr_pid = np.diff(padded[:, :, pid_idx], axis=1)  # (n, 11)

    print(f"\n  PID MR (depth-to-depth gradient):")
    print(f"  {'Transition':<16} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'r(MFE)L':>9} {'r(MFE)S':>9}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    pid_mr_key_transitions = []
    for d in range(11):
        mr_col = mr_pid[:, d]
        l_mean = float(np.mean(mr_col[long_mask])) if long_mask.sum() > 0 else 0
        s_mean = float(np.mean(mr_col[short_mask])) if short_mask.sum() > 0 else 0
        corr_l = float(np.corrcoef(mr_col[long_mask], mfes[long_mask])[0, 1]) \
            if long_mask.sum() > 10 and np.std(mr_col[long_mask]) > 1e-12 else 0.0
        corr_s = float(np.corrcoef(mr_col[short_mask], mfes[short_mask])[0, 1]) \
            if short_mask.sum() > 10 and np.std(mr_col[short_mask]) > 1e-12 else 0.0
        if np.isnan(corr_l): corr_l = 0.0
        if np.isnan(corr_s): corr_s = 0.0
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        tag = ''
        if abs(corr_l) > 0.15 or abs(corr_s) > 0.15:
            tag = ' ***'
            pid_mr_key_transitions.append((f"{d_from}>{d_to}", corr_l, corr_s))
        print(f"  {d_from}>{d_to:<9} {l_mean:>+8.3f} {s_mean:>+8.3f} "
              f"{l_mean - s_mean:>+8.3f} {corr_l:>+9.4f} {corr_s:>+9.4f}{tag}")

    # PID UCL breaks
    D4 = 3.267
    pid_mr_abs = np.abs(mr_pid)
    pid_mr_bar = float(pid_mr_abs.mean())
    pid_ucl = D4 * pid_mr_bar
    pid_ucl_breaks = (pid_mr_abs > pid_ucl).astype(float)

    print(f"\n  PID UCL BREAKS (% with control limit violation, UCL={pid_ucl:.3f}):")
    print(f"  {'Transition':<16} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'WR|brk L':>9} {'WR|brk S':>9}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    for d in range(11):
        brk = pid_ucl_breaks[:, d]
        l_pct = float(brk[long_mask].mean()) * 100 if long_mask.sum() > 0 else 0
        s_pct = float(brk[short_mask].mean()) * 100 if short_mask.sum() > 0 else 0
        l_brk_mask = long_mask & (brk > 0.5)
        s_brk_mask = short_mask & (brk > 0.5)
        wr_l_brk = float((mfes[l_brk_mask] > maes[l_brk_mask]).mean()) * 100 \
            if l_brk_mask.sum() > 5 else float('nan')
        wr_s_brk = float((mfes[s_brk_mask] > maes[s_brk_mask]).mean()) * 100 \
            if s_brk_mask.sum() > 5 else float('nan')
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        wr_l_str = f"{wr_l_brk:>7.1f}%" if not np.isnan(wr_l_brk) else "    n/a "
        wr_s_str = f"{wr_s_brk:>7.1f}%" if not np.isnan(wr_s_brk) else "    n/a "
        print(f"  {d_from}>{d_to:<9} {l_pct:>7.1f}% {s_pct:>7.1f}% "
              f"{l_pct - s_pct:>+7.1f}% {wr_l_str} {wr_s_str}")

    # PID profile by fission class
    print(f"\n  PID PROFILE BY FISSION CLASS:")
    for label, seg_list in [('KEEP', keep_segs), ('SPLIT', split_segs), ('DROP', drop_segs)]:
        if not seg_list:
            continue
        class_mask = np.zeros(len(mfes), dtype=bool)
        for s in seg_list:
            seg_m = (tids == s['tid']) & (long_mask if s['dir'] == 'LONG' else short_mask)
            class_mask |= seg_m
        if class_mask.sum() < 10:
            continue
        pid_vals = padded[class_mask, :, pid_idx]
        pid_mfes_class = mfes[class_mask]
        pid_maes_class = maes[class_mask]
        class_wr = float((pid_mfes_class > pid_maes_class).mean())

        print(f"\n  {label} ({class_mask.sum()} patterns, WR={class_wr:.1%}):")
        print(f"    {'Depth':<12} {'Mean PID':>10} {'Std':>8} {'r(MFE)':>8}")
        print(f"    {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
        for d in range(12):
            col = pid_vals[:, d]
            corr = float(np.corrcoef(col, pid_mfes_class)[0, 1]) \
                if np.std(col) > 1e-12 else 0.0
            if np.isnan(corr): corr = 0.0
            lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
            print(f"    {lbl:<12} {np.mean(col):>+10.3f} {np.std(col):>8.3f} {corr:>+8.4f}")

        pid_mr_class = np.diff(pid_vals, axis=1)
        print(f"    MR transitions:")
        print(f"    {'Transition':<16} {'Mean MR':>10} {'r(MFE)':>8}")
        print(f"    {'-'*16} {'-'*10} {'-'*8}")
        for d in range(11):
            mr_col = pid_mr_class[:, d]
            corr = float(np.corrcoef(mr_col, pid_mfes_class)[0, 1]) \
                if np.std(mr_col) > 1e-12 else 0.0
            if np.isnan(corr): corr = 0.0
            d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
            d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
            tag = ' ***' if abs(corr) > 0.15 else ''
            print(f"    {d_from}>{d_to:<9} {np.mean(mr_col):>+10.3f} {corr:>+8.4f}{tag}")

    # PID x direction confirmation
    print(f"\n  PID x DIRECTION CONFIRMATION:")
    print(f"  (Does PID sign at each depth agree with DMI direction?)")
    print(f"  {'Depth':<12} {'Agree%':>8} {'WR|agree':>10} {'WR|disagr':>10} {'Lift':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for d in range(12):
        pid_col = padded[:, d, pid_idx]
        agree = (long_mask & (pid_col > 0)) | (short_mask & (pid_col < 0))
        disagree = ~agree
        n_agree = agree.sum()
        n_disagree = disagree.sum()
        agree_pct = float(n_agree) / len(mfes) * 100
        wr_agree = float((mfes[agree] > maes[agree]).mean()) if n_agree > 10 else float('nan')
        wr_disagree = float((mfes[disagree] > maes[disagree]).mean()) if n_disagree > 10 else float('nan')
        lift = wr_agree - wr_disagree if not (np.isnan(wr_agree) or np.isnan(wr_disagree)) else 0
        lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        wr_a_str = f"{wr_agree:.1%}" if not np.isnan(wr_agree) else "n/a"
        wr_d_str = f"{wr_disagree:.1%}" if not np.isnan(wr_disagree) else "n/a"
        print(f"  {lbl:<12} {agree_pct:>7.1f}% {wr_a_str:>10} {wr_d_str:>10} {lift:>+7.1%}")

    # --- 19. Temporal special cause analysis ---
    from datetime import datetime, timezone

    ts_arr = np.array([m['ts'] for m in meta])
    valid_ts = ts_arr > 0
    n_valid = valid_ts.sum()

    print(f"\n{'='*70}")
    print(f"  TEMPORAL SPECIAL CAUSE ANALYSIS")
    print(f"  (Patterns with valid timestamps: {n_valid} / {len(meta)})")
    print(f"{'='*70}")

    if n_valid > 50:
        dts = np.array([
            datetime.fromtimestamp(t, tz=timezone.utc) if t > 0 else None
            for t in ts_arr
        ])
        hours_utc = np.array([dt.hour if dt else -1 for dt in dts])
        dow = np.array([dt.weekday() if dt else -1 for dt in dts])
        dom = np.array([dt.day if dt else -1 for dt in dts])

        def _session(h):
            if h >= 22 or h < 8:
                return 'ASIA'
            elif h < 14:
                return 'EUROPE'
            elif h < 21:
                return 'US_RTH'
            else:
                return 'US_CLOSE'

        sessions = np.array([_session(h) if h >= 0 else 'UNK' for h in hours_utc])

        # 1. Market sessions
        print(f"\n  1. MARKET SESSION:")
        print(f"  {'Session':<12} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'PID_d7':>8}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

        for sess in ['ASIA', 'EUROPE', 'US_RTH', 'US_CLOSE']:
            smask = (sessions == sess) & valid_ts
            n_s = smask.sum()
            if n_s < 10:
                continue
            wr = float((mfes[smask] > maes[smask]).mean())
            mfe_m = float(np.mean(mfes[smask]))
            mae_m = float(np.mean(maes[smask]))
            sl = smask & long_mask
            ss_m = smask & short_mask
            wr_l = float((mfes[sl] > maes[sl]).mean()) if sl.sum() > 5 else float('nan')
            wr_s = float((mfes[ss_m] > maes[ss_m]).mean()) if ss_m.sum() > 5 else float('nan')
            pid_d7 = float(np.mean(padded[smask, 7, pid_idx]))
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            print(f"  {sess:<12} {n_s:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {pid_d7:>+8.2f}")

        # 2. Hourly breakdown
        print(f"\n  2. HOURLY BREAKDOWN (UTC):")
        print(f"  {'Hour':>4} {'Session':<10} {'N':>5} {'WR':>7} {'MFE':>7} {'WR_L':>7} {'WR_S':>7}")
        print(f"  {'-'*4} {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

        for h in range(24):
            hmask = (hours_utc == h) & valid_ts
            n_h = hmask.sum()
            if n_h < 10:
                continue
            wr = float((mfes[hmask] > maes[hmask]).mean())
            mfe_m = float(np.mean(mfes[hmask]))
            hl = hmask & long_mask
            hs = hmask & short_mask
            wr_l = float((mfes[hl] > maes[hl]).mean()) if hl.sum() > 5 else float('nan')
            wr_s = float((mfes[hs] > maes[hs]).mean()) if hs.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            bar = '#' * int(wr * 20)
            print(f"  {h:>4} {_session(h):<10} {n_h:>5} {wr:>7.1%} {mfe_m:>+7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7}  {bar}")

        # 3. Day of week
        dow_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

        # Build KEEP mask for cross-reference
        keep_mask_all = np.zeros(len(mfes), dtype=bool)
        for s in keep_segs:
            seg_m = (tids == s['tid']) & (long_mask if s['dir'] == 'LONG' else short_mask)
            keep_mask_all |= seg_m

        print(f"\n  3. DAY OF WEEK:")
        print(f"  {'Day':<5} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

        for d_idx in range(7):
            dmask = (dow == d_idx) & valid_ts
            n_d = dmask.sum()
            if n_d < 10:
                continue
            wr = float((mfes[dmask] > maes[dmask]).mean())
            mfe_m = float(np.mean(mfes[dmask]))
            mae_m = float(np.mean(maes[dmask]))
            dl = dmask & long_mask
            ds = dmask & short_mask
            dk = dmask & keep_mask_all
            wr_l = float((mfes[dl] > maes[dl]).mean()) if dl.sum() > 5 else float('nan')
            wr_s = float((mfes[ds] > maes[ds]).mean()) if ds.sum() > 5 else float('nan')
            wr_k = float((mfes[dk] > maes[dk]).mean()) if dk.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            wr_k_str = f"{wr_k:.1%}" if not np.isnan(wr_k) else "n/a"
            print(f"  {dow_names[d_idx]:<5} {n_d:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {wr_k_str:>8}")

        # 4. Month position
        def _month_pos(day):
            if day <= 7:
                return 'FIRST_WK'
            elif day >= 23:
                return 'LAST_WK'
            else:
                return 'MID'

        month_pos = np.array([_month_pos(d) if d > 0 else 'UNK' for d in dom])

        print(f"\n  4. MONTH POSITION:")
        print(f"  {'Period':<10} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

        for pos in ['FIRST_WK', 'MID', 'LAST_WK']:
            pmask = (month_pos == pos) & valid_ts
            n_p = pmask.sum()
            if n_p < 10:
                continue
            wr = float((mfes[pmask] > maes[pmask]).mean())
            mfe_m = float(np.mean(mfes[pmask]))
            mae_m = float(np.mean(maes[pmask]))
            pl = pmask & long_mask
            ps = pmask & short_mask
            pk = pmask & keep_mask_all
            wr_l = float((mfes[pl] > maes[pl]).mean()) if pl.sum() > 5 else float('nan')
            wr_s = float((mfes[ps] > maes[ps]).mean()) if ps.sum() > 5 else float('nan')
            wr_k = float((mfes[pk] > maes[pk]).mean()) if pk.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            wr_k_str = f"{wr_k:.1%}" if not np.isnan(wr_k) else "n/a"
            print(f"  {pos:<10} {n_p:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {wr_k_str:>8}")

        # 5. Session open/close proximity
        print(f"\n  5. SESSION OPEN/CLOSE (first & last 30min):")
        print(f"  {'Marker':<20} {'N':>5} {'WR':>7} {'MFE':>7} {'vs Sess':>8}")
        print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*8}")

        minutes_utc = np.array([
            dt.hour * 60 + dt.minute if dt else -1 for dt in dts
        ])

        markers = [
            ('ASIA open',       22 * 60, 22 * 60 + 30, 'ASIA'),
            ('ASIA close',      7 * 60 + 30, 8 * 60, 'ASIA'),
            ('EUROPE open',     8 * 60, 8 * 60 + 30, 'EUROPE'),
            ('EUROPE close',    14 * 60, 14 * 60 + 30, 'EUROPE'),
            ('US RTH open',     14 * 60 + 30, 15 * 60, 'US_RTH'),
            ('US RTH close',    20 * 60 + 30, 21 * 60, 'US_RTH'),
        ]

        for marker_label, t_start, t_end, parent_sess in markers:
            if t_start < t_end:
                mmask = (minutes_utc >= t_start) & (minutes_utc < t_end) & valid_ts
            else:
                mmask = ((minutes_utc >= t_start) | (minutes_utc < t_end)) & valid_ts
            n_m = mmask.sum()
            if n_m < 5:
                continue
            wr = float((mfes[mmask] > maes[mmask]).mean())
            mfe_m = float(np.mean(mfes[mmask]))
            sess_mask = (sessions == parent_sess) & valid_ts
            sess_wr = float((mfes[sess_mask] > maes[sess_mask]).mean()) if sess_mask.sum() > 10 else wr
            delta = wr - sess_wr
            print(f"  {marker_label:<20} {n_m:>5} {wr:>7.1%} {mfe_m:>+7.0f} {delta:>+7.1%}")

        # 6. Week position
        print(f"\n  6. WEEK POSITION:")
        print(f"  {'Period':<12} {'N':>5} {'WR':>7} {'MFE':>7} {'WR_L':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

        week_pos = {
            'START(M-T)': (dow == 0) | (dow == 1),
            'MID(W)':     dow == 2,
            'END(T-F)':   (dow == 3) | (dow == 4),
        }
        for wlabel, wmask_base in week_pos.items():
            wmask = wmask_base & valid_ts
            n_w = wmask.sum()
            if n_w < 10:
                continue
            wr = float((mfes[wmask] > maes[wmask]).mean())
            mfe_m = float(np.mean(mfes[wmask]))
            wl = wmask & long_mask
            wk = wmask & keep_mask_all
            wr_l = float((mfes[wl] > maes[wl]).mean()) if wl.sum() > 5 else float('nan')
            wr_k = float((mfes[wk] > maes[wk]).mean()) if wk.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_k_str = f"{wr_k:.1%}" if not np.isnan(wr_k) else "n/a"
            print(f"  {wlabel:<12} {n_w:>5} {wr:>7.1%} {mfe_m:>+7.0f} "
                  f"{wr_l_str:>7} {wr_k_str:>8}")

        # 7. MR UCL breaks x Temporal
        mr_ucl_start = 11 * 16
        mr_ucl_end = mr_ucl_start + 11 * 16
        ucl_per_pattern = flat_mr[:, mr_ucl_start:mr_ucl_end].sum(axis=1)
        has_ucl = ucl_per_pattern > 0

        print(f"\n  7. MR UCL BREAKS x TEMPORAL:")
        print(f"  (Where do control limit violations cluster in time?)")
        print(f"  Patterns with any UCL break: {has_ucl.sum()} / {len(mfes)} "
              f"({has_ucl.mean():.1%})")

        # UCL breaks by session
        print(f"\n  UCL breaks by SESSION:")
        print(f"  {'Session':<12} {'N_brk':>6} {'%brk':>6} {'WR|brk':>8} {'WR|no':>8} {'Lift':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")

        for sess in ['ASIA', 'EUROPE', 'US_RTH', 'US_CLOSE']:
            smask = (sessions == sess) & valid_ts
            n_s = smask.sum()
            if n_s < 10:
                continue
            brk_in_sess = smask & has_ucl
            no_brk_in_sess = smask & ~has_ucl
            n_brk = brk_in_sess.sum()
            pct_brk = n_brk / max(n_s, 1)
            wr_brk = float((mfes[brk_in_sess] > maes[brk_in_sess]).mean()) if n_brk > 5 else float('nan')
            wr_no = float((mfes[no_brk_in_sess] > maes[no_brk_in_sess]).mean()) \
                if no_brk_in_sess.sum() > 5 else float('nan')
            lift = wr_brk - wr_no if not (np.isnan(wr_brk) or np.isnan(wr_no)) else 0
            wr_b_str = f"{wr_brk:.1%}" if not np.isnan(wr_brk) else "n/a"
            wr_n_str = f"{wr_no:.1%}" if not np.isnan(wr_no) else "n/a"
            print(f"  {sess:<12} {n_brk:>6} {pct_brk:>5.1%} {wr_b_str:>8} {wr_n_str:>8} {lift:>+6.1%}")

        # UCL breaks by hour
        print(f"\n  UCL breaks by HOUR (top hours with most breaks):")
        print(f"  {'Hour':>4} {'Session':<10} {'N_brk':>6} {'%brk':>6} {'WR|brk':>8} {'Lift':>7}")
        print(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")

        hour_data = []
        for h in range(24):
            hmask = (hours_utc == h) & valid_ts
            n_h = hmask.sum()
            if n_h < 10:
                continue
            brk_h = hmask & has_ucl
            no_brk_h = hmask & ~has_ucl
            n_brk = brk_h.sum()
            pct_brk = n_brk / max(n_h, 1)
            wr_brk = float((mfes[brk_h] > maes[brk_h]).mean()) if n_brk > 5 else float('nan')
            wr_no = float((mfes[no_brk_h] > maes[no_brk_h]).mean()) if no_brk_h.sum() > 5 else float('nan')
            lift = wr_brk - wr_no if not (np.isnan(wr_brk) or np.isnan(wr_no)) else 0
            hour_data.append((h, n_brk, pct_brk, wr_brk, lift))

        hour_data.sort(key=lambda x: x[1], reverse=True)
        for h, n_brk, pct_brk, wr_brk, lift in hour_data[:10]:
            wr_b_str = f"{wr_brk:.1%}" if not np.isnan(wr_brk) else "n/a"
            print(f"  {h:>4} {_session(h):<10} {n_brk:>6} {pct_brk:>5.1%} "
                  f"{wr_b_str:>8} {lift:>+6.1%}")

        # UCL breaks by day of week
        print(f"\n  UCL breaks by DAY OF WEEK:")
        print(f"  {'Day':<5} {'N_brk':>6} {'%brk':>6} {'WR|brk':>8} {'WR|no':>8} {'Lift':>7}")
        print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")

        for d_idx in range(7):
            dmask = (dow == d_idx) & valid_ts
            n_d = dmask.sum()
            if n_d < 10:
                continue
            brk_d = dmask & has_ucl
            no_brk_d = dmask & ~has_ucl
            n_brk = brk_d.sum()
            pct_brk = n_brk / max(n_d, 1)
            wr_brk = float((mfes[brk_d] > maes[brk_d]).mean()) if n_brk > 5 else float('nan')
            wr_no = float((mfes[no_brk_d] > maes[no_brk_d]).mean()) if no_brk_d.sum() > 5 else float('nan')
            lift = wr_brk - wr_no if not (np.isnan(wr_brk) or np.isnan(wr_no)) else 0
            wr_b_str = f"{wr_brk:.1%}" if not np.isnan(wr_brk) else "n/a"
            wr_n_str = f"{wr_no:.1%}" if not np.isnan(wr_no) else "n/a"
            print(f"  {dow_names[d_idx]:<5} {n_brk:>6} {pct_brk:>5.1%} "
                  f"{wr_b_str:>8} {wr_n_str:>8} {lift:>+6.1%}")

        # Top MR breaks per session
        print(f"\n  TOP MR BREAKS per SESSION (which features spike when):")
        mr_transitions = []
        for d in range(11):
            d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
            d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
            for f_i in range(16):
                f_lbl = FEATURE_NAMES[f_i] if f_i < len(FEATURE_NAMES) else f'f{f_i}'
                col_idx = mr_ucl_start + d * 16 + f_i
                mr_transitions.append((f"{d_from}>{d_to}__{f_lbl}", col_idx))

        for sess in ['ASIA', 'EUROPE', 'US_RTH']:
            smask = (sessions == sess) & valid_ts
            if smask.sum() < 20:
                continue
            break_counts = []
            for tr_name, col_idx in mr_transitions:
                if col_idx >= flat_mr.shape[1]:
                    continue
                col = flat_mr[smask, col_idx]
                n_brk = int((col > 0.5).sum())
                if n_brk > 0:
                    brk_mask_local = (col > 0.5)
                    wr_brk = float((mfes[smask][brk_mask_local] > maes[smask][brk_mask_local]).mean())
                    break_counts.append((tr_name, n_brk, wr_brk))
            break_counts.sort(key=lambda x: x[1], reverse=True)
            print(f"\n  {sess}:")
            for tr_name, n_brk, wr_brk in break_counts[:5]:
                print(f"    {tr_name:<30} breaks={n_brk:>4}, WR|brk={wr_brk:.1%}")

    else:
        print(f"  (Skipped — insufficient valid timestamps)")

    # --- 20. Stacked gate analysis ---
    print(f"\n{'='*70}")
    print(f"  STACKED GATE ANALYSIS: Compound Filters")
    print(f"  Each gate stacks on previous — progressive noise removal")
    print(f"{'='*70}")

    keep_tids = set()
    for s in keep_segs:
        keep_tids.add((s['tid'], s['dir']))

    keep_mask = np.zeros(len(mfes), dtype=bool)
    for i, m in enumerate(meta):
        d = 'LONG' if long_mask[i] else 'SHORT'
        if (m['tid'], d) in keep_tids:
            keep_mask[i] = True

    if n_valid > 50:
        europe_mask = np.array([s == 'EUROPE' for s in sessions]) & valid_ts

        session_open_mask = np.zeros(len(mfes), dtype=bool)
        for i, dt in enumerate(dts):
            if dt is None:
                continue
            h, mn = dt.hour, dt.minute
            if h == 14 and mn < 30:
                session_open_mask[i] = True
            elif h == 8 and mn < 30:
                session_open_mask[i] = True

        good_hours = {0, 5, 17, 18, 19, 20}
        good_hour_mask = np.array([h in good_hours for h in hours_utc]) & valid_ts

        pid_d7_vals = padded[:, 7, pid_idx]
        pid_contrarian = ((pid_d7_vals > 0) & short_mask) | ((pid_d7_vals < 0) & long_mask)

        good_dow = {1, 3}
        good_dow_mask = np.array([d in good_dow for d in dow]) & valid_ts

        # Progressive stacking
        gates = []
        gates.append(('ALL patterns', np.ones(len(mfes), dtype=bool)))
        gates.append(('+ KEEP segments', keep_mask))

        g2 = keep_mask & long_mask
        gates.append(('+ LONG direction', g2))

        g3 = g2 & ~session_open_mask
        gates.append(('+ Skip session opens', g3))

        g4 = g3 & ~europe_mask
        gates.append(('+ Skip Europe session', g4))

        g5 = g4 & good_hour_mask
        gates.append(('+ Best hours (17-20,0,5)', g5))

        g6 = g5 & pid_contrarian
        gates.append(('+ PID contrarian', g6))

        g7 = g5 & good_dow_mask
        gates.append(('+ Best DOW (TUE,THU)', g7))

        g8 = g5 & good_dow_mask & pid_contrarian
        gates.append(('FULL STACK (all gates)', g8))

        print(f"\n  {'Gate':<32} {'N':>5} {'%vol':>6} {'WR':>7} {'MFE':>7} "
              f"{'MAE':>6} {'$/trade':>8} {'Lift':>7}")
        print(f"  {'-'*32} {'-'*5} {'-'*6} {'-'*7} {'-'*7} "
              f"{'-'*6} {'-'*8} {'-'*7}")

        base_wr = float((mfes > maes).mean())
        total_patterns = len(mfes)

        for gate_label, gmask in gates:
            n_g = gmask.sum()
            if n_g < 5:
                print(f"  {gate_label:<32} {n_g:>5} {'<5':>6} {'n/a':>7}")
                continue
            wr_g = float((mfes[gmask] > maes[gmask]).mean())
            mfe_g = float(np.mean(mfes[gmask]))
            mae_g = float(np.mean(maes[gmask]))
            vol_pct = n_g / total_patterns
            avg_pnl = mfe_g - mae_g
            lift = wr_g - base_wr
            print(f"  {gate_label:<32} {n_g:>5} {vol_pct:>5.1%} {wr_g:>7.1%} "
                  f"{mfe_g:>+7.0f} {mae_g:>6.0f} {avg_pnl:>+8.0f} {lift:>+6.1%}")

        # Daily throughput
        t_min_ts = ts_arr[valid_ts].min()
        t_max_ts = ts_arr[valid_ts].max()
        days_span = max((t_max_ts - t_min_ts) / 86400, 1)

        print(f"\n  DAILY THROUGHPUT (over {days_span:.0f} calendar days):")
        for gate_label, gmask in gates:
            n_g = gmask.sum()
            if n_g < 5:
                continue
            per_day = n_g / days_span
            wr_g = float((mfes[gmask] > maes[gmask]).mean())
            print(f"  {gate_label:<32} {per_day:>6.1f}/day  WR={wr_g:.1%}")

        # Ride-the-wave summary
        best_gate = g5
        best_label = "KEEP+LONG+GoodHours"
        n_best = best_gate.sum()
        if n_best >= 5:
            wr_best = float((mfes[best_gate] > maes[best_gate]).mean())
            mfe_best = float(np.mean(mfes[best_gate]))
            mae_best = float(np.mean(maes[best_gate]))
            per_day = n_best / max(days_span, 1)

            print(f"\n  {'='*60}")
            print(f"  RIDE THE WAVE — Practical Gate Summary")
            print(f"  {'='*60}")
            print(f"  Filter: {best_label}")
            print(f"  Patterns:   {n_best} ({n_best/total_patterns:.1%} of volume)")
            print(f"  Win Rate:   {wr_best:.1%}")
            print(f"  Avg MFE:    {mfe_best:+.0f} ticks")
            print(f"  Avg MAE:    {mae_best:.0f} ticks")
            print(f"  $/trade:    {mfe_best - mae_best:+.0f} ticks net")
            print(f"  Throughput: {per_day:.1f} trades/day")
            print(f"  WR lift:    {wr_best - base_wr:+.1%} vs baseline")

            # MES contract scaling
            tick_val = 1.25
            net_per_trade = (mfe_best - mae_best) * tick_val
            daily_pnl_1 = net_per_trade * per_day
            print(f"\n  MES CONTRACT SCALING:")
            print(f"    1 contract:  ${net_per_trade:+.2f}/trade, "
                  f"${daily_pnl_1:+.0f}/day")
            for contracts in [2, 5, 10]:
                print(f"    {contracts} contracts: ${net_per_trade*contracts:+.2f}/trade, "
                      f"${daily_pnl_1*contracts:+.0f}/day")

            # SPLIT segments with temporal gates
            split_tids = set()
            for s in split_segs:
                split_tids.add((s['tid'], s['dir']))

            split_mask = np.zeros(len(mfes), dtype=bool)
            for i, m in enumerate(meta):
                d = 'LONG' if long_mask[i] else 'SHORT'
                if (m['tid'], d) in split_tids:
                    split_mask[i] = True

            n_split_raw = split_mask.sum()
            if n_split_raw >= 10:
                print(f"\n  {'='*60}")
                print(f"  SPLIT SEGMENTS — Temporal Gate Cleanup")
                print(f"  {'='*60}")

                split_gates = []
                split_gates.append(('SPLIT raw', split_mask))

                sp1 = split_mask & ~session_open_mask
                split_gates.append(('+ Skip session opens', sp1))

                sp2 = sp1 & ~europe_mask
                split_gates.append(('+ Skip Europe', sp2))

                sp3 = sp2 & good_hour_mask
                split_gates.append(('+ Best hours', sp3))

                sp4 = sp2 & good_dow_mask
                split_gates.append(('+ Best DOW (TUE,THU)', sp4))

                sp5 = sp3 & good_dow_mask
                split_gates.append(('+ Best hours + DOW', sp5))

                print(f"\n  {'Gate':<32} {'N':>5} {'WR':>7} {'MFE':>7} "
                      f"{'MAE':>6} {'net':>6} {'$/day':>8}")
                print(f"  {'-'*32} {'-'*5} {'-'*7} {'-'*7} "
                      f"{'-'*6} {'-'*6} {'-'*8}")

                for sp_label, gmask in split_gates:
                    n_g = gmask.sum()
                    if n_g < 5:
                        print(f"  {sp_label:<32} {n_g:>5}  (too few)")
                        continue
                    wr_g = float((mfes[gmask] > maes[gmask]).mean())
                    mfe_g = float(np.mean(mfes[gmask]))
                    mae_g = float(np.mean(maes[gmask]))
                    net_ticks = mfe_g - mae_g
                    per_day_g = n_g / max(days_span, 1)
                    daily_1mes = net_ticks * tick_val * per_day_g
                    print(f"  {sp_label:<32} {n_g:>5} {wr_g:>7.1%} {mfe_g:>+7.0f} "
                          f"{mae_g:>6.0f} {net_ticks:>+6.0f} ${daily_1mes:>+7.0f}")

                # Revenue model
                sp_best = sp3
                n_sp = sp_best.sum()

                k_net = (mfe_best - mae_best)
                k_per_day = n_best / max(days_span, 1)

                if n_sp >= 5:
                    sp_wr = float((mfes[sp_best] > maes[sp_best]).mean())
                    sp_mfe = float(np.mean(mfes[sp_best]))
                    sp_mae = float(np.mean(maes[sp_best]))
                    sp_net = sp_mfe - sp_mae
                    sp_per_day = n_sp / max(days_span, 1)
                else:
                    sp_wr, sp_net, sp_per_day = 0, 0, 0

                unified_mask = best_gate | sp_best
                n_unified = unified_mask.sum()
                if n_unified >= 5:
                    u_wr = float((mfes[unified_mask] > maes[unified_mask]).mean())
                    u_mfe = float(np.mean(mfes[unified_mask]))
                    u_mae = float(np.mean(maes[unified_mask]))
                    u_net = u_mfe - u_mae
                    u_per_day = n_unified / max(days_span, 1)
                    u_daily_1 = u_net * tick_val * u_per_day
                else:
                    u_wr, u_net, u_per_day, u_daily_1 = 0, 0, 0, 0

                print(f"\n  {'='*60}")
                print(f"  REVENUE MODEL — 1 Contract (KEEP + SPLIT unified)")
                print(f"  {'='*60}")

                print(f"\n  POOL BREAKDOWN (1 MES = $1.25/tick):")
                print(f"  {'Pool':<20} {'N':>5} {'WR':>7} {'net/t':>6} "
                      f"{'trades/d':>9} {'$/day':>8}")
                print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*6} "
                      f"{'-'*9} {'-'*8}")
                k_daily_1 = k_net * tick_val * k_per_day
                sp_daily_1 = sp_net * tick_val * sp_per_day
                print(f"  {'KEEP (best hrs)':<20} {n_best:>5} {wr_best:>7.1%} "
                      f"{k_net:>+6.0f} {k_per_day:>9.1f} ${k_daily_1:>7,.0f}")
                if n_sp >= 5:
                    print(f"  {'SPLIT (best hrs)':<20} {n_sp:>5} {sp_wr:>7.1%} "
                          f"{sp_net:>+6.0f} {sp_per_day:>9.1f} ${sp_daily_1:>7,.0f}")
                print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*6} "
                      f"{'-'*9} {'-'*8}")
                print(f"  {'UNIFIED':<20} {n_unified:>5} {u_wr:>7.1%} "
                      f"{u_net:>+6.0f} {u_per_day:>9.1f} ${u_daily_1:>7,.0f}")

                # OOS degradation scenarios
                if u_net > 0 and u_per_day > 0:
                    print(f"\n  OOS DEGRADATION SCENARIOS (1 MES contract):")
                    print(f"  IS baseline: {u_per_day:.1f} trades/day, "
                          f"+{u_net:.0f} ticks/trade, ${u_daily_1:,.0f}/day")
                    print(f"\n  {'Scenario':<25} {'net/t':>6} {'$/trade':>8} "
                          f"{'$/day':>8} {'$/month':>9} {'$800?':>6}")
                    print(f"  {'-'*25} {'-'*6} {'-'*8} "
                          f"{'-'*8} {'-'*9} {'-'*6}")

                    for pct, decay_label in [(0, 'IS (no decay)'),
                                       (10, '10% haircut'),
                                       (20, '20% haircut'),
                                       (30, '30% haircut'),
                                       (40, '40% haircut'),
                                       (50, '50% haircut')]:
                        decay = 1.0 - pct / 100
                        d_net = u_net * decay
                        d_trade = d_net * tick_val
                        d_daily = d_trade * u_per_day
                        d_monthly = d_daily * 20
                        hits = 'YES' if d_daily >= 800 else 'no'
                        print(f"  {decay_label:<25} {d_net:>+6.0f} ${d_trade:>7,.0f} "
                              f"${d_daily:>7,.0f} ${d_monthly:>8,.0f} {hits:>6}")

                    # Breakeven
                    min_net_800 = 800 / (u_per_day * tick_val) if u_per_day > 0 else 0
                    max_decay_800 = (1 - min_net_800 / u_net) * 100 if u_net > 0 else 0

                    print(f"\n  BREAKEVEN:")
                    print(f"    $800/day needs +{min_net_800:.0f} ticks/trade "
                          f"@ {u_per_day:.1f} trades/day")
                    print(f"    Max tolerable decay: {max_decay_800:.0f}% "
                          f"before dropping below $800")
                    print(f"    IS net: +{u_net:.0f}t -> "
                          f"buffer of {u_net - min_net_800:.0f} ticks "
                          f"({max_decay_800:.0f}% margin of safety)")

                    # Contract scaling
                    print(f"\n  CONTRACT SCALING (at IS rates, ${u_daily_1:,.0f}/day/MES):")
                    margin_1 = 1320
                    for cts in [1, 2, 3, 5]:
                        d_val = u_daily_1 * cts
                        m_val = margin_1 * cts
                        print(f"    {cts} MES: ${d_val:>8,.0f}/day, "
                              f"${d_val*20:>9,.0f}/month  (margin: ${m_val:>6,.0f})")

    else:
        print(f"  (Skipped — insufficient valid timestamps)")

    # --- Save report ---
    sys.stdout = _orig_stdout

    report_path = os.path.join(os.path.dirname(__file__), 'standalone_report.txt')
    header = f"STANDALONE WAVEFORM SCREENING REPORT (FULL 16D)\n"
    header += f"Data: {args.data}, Base TF: {args.base_tf}\n"
    header += f"Context: {args.context_days}d, Analysis: {args.analysis_days}d\n"
    header += f"Regimes: {len(regime_meta)}, Data points: {len(mfes)}\n"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(_report_buf.getvalue())

    print(f"\n  Full report saved: {report_path}")
    print(f"  Gates saved: {_gate_path}")
    print(f"  Charts: {PLOTS_DIR}/")
    print(f"  Done.")


if __name__ == '__main__':
    main()
