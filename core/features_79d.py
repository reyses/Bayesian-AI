"""
79D Unified Feature Vector — Single Source of Truth
====================================================
10 core features x 6 TFs + 3 helpers x 6 TFs + 1 global = 79D

Same 10 measurements at every timeframe. The fractal state in one array.
Each TF is just a different aggregation window on the same 5s atomic data.

Core features (per TF):
  [0] z_se             — position in regression band (SE units)
  [1] dmi_diff         — DI+ minus DI- (direction + strength)
  [2] variance_ratio   — short/long vol ratio (regime)
  [3] velocity         — price rate of change
  [4] acceleration     — velocity change (chop detector)
  [5] vol_rel          — volume vs 30-bar SMA (conviction)
  [6] bar_range        — (high-low)/tick (risk)
  [7] hurst            — persistence exponent
  [8] reversion_prob   — P(revert to center) from OU first-passage
  [9] p_at_center      — 3-class probability near mean

Helper features (per TF):
  [0] dmi_gap          — abs(dmi_diff)
  [1] dir_vol          — sign(velocity) * vol_rel
  [2] wick_ratio       — 1 - abs(close-open)/range

Global:
  time_of_day          — timestamp % 86400 / 86400

TF order: 15s, 1m, 5m, 15m, 1h, 1D
Layout: [15s_core(10), 1m_core(10), ..., 1D_core(10), 15s_help(3), ..., 1D_help(3), time_of_day]

Spec: docs/Active/FEATURE_VECTOR_79D_SPEC.md
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List

TICK = 0.25  # MNQ tick size

# Canonical TF order
TF_ORDER = ['15s', '1m', '5m', '15m', '1h', '1D']
N_TFS = len(TF_ORDER)

# Feature names per TF
CORE_FEATURE_NAMES = [
    'z_se', 'dmi_diff', 'variance_ratio', 'velocity', 'acceleration',
    'vol_rel', 'bar_range', 'hurst', 'reversion_prob', 'p_at_center',
]
HELPER_FEATURE_NAMES = ['dmi_gap', 'dir_vol', 'wick_ratio']
GLOBAL_FEATURE_NAMES = ['time_of_day']

N_CORE = len(CORE_FEATURE_NAMES)      # 10
N_HELPER = len(HELPER_FEATURE_NAMES)   # 3
N_GLOBAL = len(GLOBAL_FEATURE_NAMES)   # 1
N_FEATURES = N_CORE * N_TFS + N_HELPER * N_TFS + N_GLOBAL  # 79

# Full feature name list (for column headers, logging, etc.)
FEATURE_NAMES_79D = []
for tf in TF_ORDER:
    for feat in CORE_FEATURE_NAMES:
        FEATURE_NAMES_79D.append(f'{tf}_{feat}')
for tf in TF_ORDER:
    for feat in HELPER_FEATURE_NAMES:
        FEATURE_NAMES_79D.append(f'{tf}_{feat}')
FEATURE_NAMES_79D.extend(GLOBAL_FEATURE_NAMES)

# Index boundaries for slicing
CORE_START = 0
CORE_END = N_CORE * N_TFS                                    # 60
HELPER_START = CORE_END                                       # 60
HELPER_END = HELPER_START + N_HELPER * N_TFS                  # 78
GLOBAL_START = HELPER_END                                     # 78

# TF durations in seconds (for partial bar aggregation)
TF_SECONDS = {
    '15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1D': 86400,
}


# ═══════════════════════════════════════════════════════════════════════
# PARTIAL TF AGGREGATION — zero-lookahead higher TF bar construction
# ═══════════════════════════════════════════════════════════════════════
#
# SAFETY RULE: Higher TF bars are built from CLOSED anchor (1m) bars only.
# The current open 1m bar is NEVER included in any higher TF aggregation.
# This guarantees zero lookahead in testing and live.
#
# In training: can use pre-built TF parquets (labels come from the future
# anyway). Partial TFs are only needed for testing/live/forward pass.
# ═══════════════════════════════════════════════════════════════════════


def aggregate_partial_bar(closed_bars: pd.DataFrame, tf_seconds: int) -> pd.DataFrame:
    """Aggregate closed 1m bars into partial higher TF bars.

    Takes a history of CLOSED 1m bars and builds partial + completed bars
    for a target TF. The current incomplete TF bar is included as a partial.

    SAFETY: Only call this with bars that are FULLY CLOSED at the anchor TF.
    The caller is responsible for NOT passing the current open bar.

    Args:
        closed_bars: DataFrame of CLOSED 1m bars with [timestamp, open, high, low, close, volume].
                     Must be sorted by timestamp. ALL bars must be closed.
        tf_seconds:  Target TF in seconds (300 for 5m, 900 for 15m, etc.)

    Returns:
        DataFrame of aggregated bars (completed + current partial) with same columns.
        Last row is the current partial bar.
    """
    if closed_bars is None or len(closed_bars) == 0:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = closed_bars.copy()
    df['bar_ts'] = (df['timestamp'] // tf_seconds) * tf_seconds

    agg = df.groupby('bar_ts').agg(
        timestamp=('bar_ts', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index(drop=True).sort_values('timestamp')

    return agg


def build_partial_tfs(
    closed_1m_bars: pd.DataFrame,
    tfs: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Build partial higher TF bars from closed 1m bars.

    Zero-lookahead guarantee: only uses CLOSED 1m bars.
    Returns a dict of {tf_label: DataFrame} ready for SFE and feature extraction.

    Args:
        closed_1m_bars: DataFrame of CLOSED 1m bars, sorted by timestamp.
                        This must NOT include the current open 1m bar.
        tfs:            List of TFs to build. Default: all higher TFs.
                        '1m' is passed through as-is (no aggregation needed).
                        '15s' is NOT built here (comes from sub-anchor data).

    Returns:
        {tf_label: DataFrame} — one entry per requested TF.
    """
    if tfs is None:
        tfs = ['5m', '15m', '1h', '1D']

    result = {}

    for tf in tfs:
        if tf == '1m':
            result['1m'] = closed_1m_bars
        elif tf == '15s':
            # 15s comes from sub-anchor data, not aggregation upward
            continue
        elif tf in TF_SECONDS:
            tf_sec = TF_SECONDS[tf]
            if tf_sec <= 60:
                # TFs <= 1m shouldn't be aggregated from 1m
                continue
            result[tf] = aggregate_partial_bar(closed_1m_bars, tf_sec)
        else:
            raise ValueError(f'Unknown TF: {tf}')

    return result


def build_all_tf_ohlcv(
    closed_1m_bars: pd.DataFrame,
    closed_15s_bars: pd.DataFrame = None,
    historical_1m_bars: pd.DataFrame = None,
) -> Dict[str, pd.DataFrame]:
    """Build OHLCV for all 6 TFs from closed anchor + sub-anchor bars.

    Zero-lookahead guarantee: only uses CLOSED bars at each level.

    For 1h and 1D, a single day of 1m bars isn't enough (SFE needs 21+ bars).
    Pass historical_1m_bars (previous days) to provide context for higher TFs.
    The historical bars are prepended to today's closed bars before aggregation.

    Args:
        closed_1m_bars:     CLOSED 1m bars for today. Must not include current open bar.
        closed_15s_bars:    CLOSED 15s bars (execution TF). Optional.
        historical_1m_bars: CLOSED 1m bars from previous days (for 1h/1D context).
                            These are prepended before aggregation. Already closed = safe.

    Returns:
        {tf_label: DataFrame} for all 6 TFs in TF_ORDER.
    """
    # For higher TFs, prepend historical context
    if historical_1m_bars is not None and len(historical_1m_bars) > 0:
        full_1m = pd.concat([historical_1m_bars, closed_1m_bars], ignore_index=True)
        full_1m = full_1m.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)
    else:
        full_1m = closed_1m_bars

    result = build_partial_tfs(full_1m, tfs=['5m', '15m', '1h', '1D'])
    result['1m'] = full_1m  # include history for SFE regression warmup

    if closed_15s_bars is not None and len(closed_15s_bars) > 0:
        result['15s'] = closed_15s_bars

    return result


def extract_tf_features(state, ohlcv_df, prev_velocity: float = 0.0) -> tuple:
    """Extract 10 core + 3 helper features for a single TF.

    Args:
        state: MarketState from SFE batch_compute_states
               (or dict with 'state' key from SFE result list)
        ohlcv_df: DataFrame with [timestamp, open, high, low, close, volume]
                  for this TF. Used for variance_ratio, vol_rel, bar_range, wick_ratio.
                  Must be sorted by timestamp. Last row = current bar.
        prev_velocity: velocity from previous bar (for acceleration)

    Returns:
        (core: np.ndarray(10,), helper: np.ndarray(3,), velocity: float)
        velocity is returned so caller can track it for next bar's acceleration
    """
    # Unpack state
    if isinstance(state, dict):
        state = state['state']

    core = np.zeros(N_CORE, dtype=np.float32)
    helper = np.zeros(N_HELPER, dtype=np.float32)

    # --- From MarketState ---
    z_score = getattr(state, 'z_score', 0.0)
    vel = getattr(state, 'velocity', 0.0)
    dmi_p = getattr(state, 'dmi_plus', 0.0)
    dmi_m = getattr(state, 'dmi_minus', 0.0)
    hurst = getattr(state, 'hurst_exponent', 0.5)
    rev_prob = getattr(state, 'reversion_probability', 0.0)
    p_center = getattr(state, 'P_at_center', 0.0)
    sigma = getattr(state, 'regression_sigma', 0.0)

    # [0] z_se: z-score using standard error
    # MarketState.z_score uses sigma. We convert to SE by multiplying by sqrt(n)/1.
    # But SFE already computes z = (price - center) / sigma, and SE = sigma / sqrt(n).
    # So z_se = z * sqrt(n). However n (regression period) is internal to SFE.
    # For now: use z_score as-is. The SFE's z_score IS the normalized position.
    core[0] = z_score

    # [1] dmi_diff: DI+ - DI- (raw, not /100)
    dmi_diff = dmi_p - dmi_m
    core[1] = dmi_diff

    # --- From OHLCV ---
    if ohlcv_df is not None and len(ohlcv_df) >= 2:
        closes = ohlcv_df['close'].values
        volumes = ohlcv_df['volume'].values if 'volume' in ohlcv_df.columns else np.ones(len(ohlcv_df))
        highs = ohlcv_df['high'].values
        lows = ohlcv_df['low'].values
        opens = ohlcv_df['open'].values
        n = len(closes)

        # [2] variance_ratio: short_std / long_std
        if n >= 60:
            short_std = np.std(closes[-10:])
            long_std = np.std(closes[-60:])
            core[2] = short_std / long_std if long_std > 1e-8 else 1.0
        elif n >= 10:
            short_std = np.std(closes[-5:])
            long_std = np.std(closes)
            core[2] = short_std / long_std if long_std > 1e-8 else 1.0
        else:
            core[2] = 1.0

        # [3] velocity: from MarketState (SFE computed)
        core[3] = vel

        # [4] acceleration: velocity change from previous bar
        core[4] = vel - prev_velocity

        # [5] vol_rel: current volume / 30-bar volume SMA
        vol_window = volumes[-30:] if n >= 30 else volumes
        vol_avg = np.mean(vol_window)
        vol_avg = max(vol_avg, 1.0)
        core[5] = volumes[-1] / vol_avg

        # [6] bar_range: (high - low) / tick for current bar
        core[6] = (highs[-1] - lows[-1]) / TICK

        # [7] hurst: from MarketState
        core[7] = hurst

        # [8] reversion_prob: from MarketState (OU first-passage)
        core[8] = rev_prob

        # [9] p_at_center: from MarketState (3-class probability)
        core[9] = p_center

        # --- Helpers ---
        # [0] dmi_gap: abs(dmi_diff)
        helper[0] = abs(dmi_diff)

        # [1] dir_vol: sign(velocity) * vol_rel
        vel_sign = 1.0 if vel > 0 else -1.0 if vel < 0 else 0.0
        helper[1] = vel_sign * core[5]

        # [2] wick_ratio: 1 - abs(close-open)/range
        bar_rng = highs[-1] - lows[-1]
        if bar_rng > 0:
            helper[2] = 1.0 - abs(closes[-1] - opens[-1]) / bar_rng
        else:
            helper[2] = 0.0

    else:
        # Minimal: just MarketState fields, no OHLCV
        core[2] = 1.0   # variance_ratio default
        core[3] = vel
        core[4] = vel - prev_velocity
        core[5] = 1.0   # vol_rel default
        core[6] = 0.0   # bar_range unknown
        core[7] = hurst
        core[8] = rev_prob
        core[9] = p_center
        helper[0] = abs(dmi_diff)

    return core, helper, vel


def extract_79d(
    states_by_tf: Dict[str, object],
    ohlcv_by_tf: Dict[str, 'pd.DataFrame'],
    prev_velocities: Dict[str, float],
    timestamp: float = 0.0,
) -> tuple:
    """Compute the full 79D feature vector from multi-TF SFE states + OHLCV.

    Args:
        states_by_tf: {tf_label: MarketState} — SFE state for each TF.
                      Keys should be from TF_ORDER ('15s', '1m', '5m', '15m', '1h', '1D').
                      Missing TFs get zeros.
        ohlcv_by_tf:  {tf_label: DataFrame} — OHLCV data for each TF.
                      Each DataFrame has [timestamp, open, high, low, close, volume].
                      Last row = current bar. History needed for variance_ratio, vol_rel.
        prev_velocities: {tf_label: float} — velocity from previous bar per TF.
                         Used to compute acceleration. Updated in-place.
        timestamp:    Current bar timestamp (for time_of_day).

    Returns:
        (features: np.ndarray(79,), updated_velocities: dict)
        updated_velocities has current velocity per TF for next call.
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)
    new_velocities = {}

    # Core features: 10 per TF
    for tf_idx, tf in enumerate(TF_ORDER):
        state = states_by_tf.get(tf)
        ohlcv = ohlcv_by_tf.get(tf)
        prev_vel = prev_velocities.get(tf, 0.0)

        if state is not None:
            core, helper, vel = extract_tf_features(state, ohlcv, prev_vel)
            new_velocities[tf] = vel
        else:
            core = np.zeros(N_CORE, dtype=np.float32)
            core[2] = 1.0   # variance_ratio default
            core[7] = 0.5   # hurst default
            helper = np.zeros(N_HELPER, dtype=np.float32)
            new_velocities[tf] = 0.0

        # Write core features
        start = tf_idx * N_CORE
        features[start:start + N_CORE] = core

        # Write helper features
        h_start = HELPER_START + tf_idx * N_HELPER
        features[h_start:h_start + N_HELPER] = helper

    # Global: time_of_day
    if timestamp > 0:
        features[GLOBAL_START] = (timestamp % 86400) / 86400

    return features, new_velocities


def extract_79d_batch(
    states_list_by_tf: Dict[str, list],
    ohlcv_by_tf: Dict[str, 'pd.DataFrame'],
) -> np.ndarray:
    """Batch extract 79D features for all bars in a day.

    More efficient than calling extract_79d() per bar — processes each TF once.

    Args:
        states_list_by_tf: {tf_label: [state_dict, ...]} — list of SFE results per TF.
                           Each list is aligned by bar index within the anchor TF.
        ohlcv_by_tf:       {tf_label: DataFrame} — full day OHLCV per TF.

    Returns:
        np.ndarray of shape (n_bars, 79) where n_bars = length of anchor TF.
    """
    # Determine number of bars from the anchor TF (1m)
    anchor_tf = '1m'
    if anchor_tf not in states_list_by_tf:
        # Fallback: use whatever TF has data
        for tf in TF_ORDER:
            if tf in states_list_by_tf and len(states_list_by_tf[tf]) > 0:
                anchor_tf = tf
                break

    anchor_states = states_list_by_tf.get(anchor_tf, [])
    n_bars = len(anchor_states)
    if n_bars == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    # Get anchor timestamps for alignment
    anchor_ohlcv = ohlcv_by_tf.get(anchor_tf)
    if anchor_ohlcv is not None and 'timestamp' in anchor_ohlcv.columns:
        anchor_ts = anchor_ohlcv['timestamp'].values
    else:
        anchor_ts = np.arange(n_bars, dtype=np.float64)

    result = np.zeros((n_bars, N_FEATURES), dtype=np.float32)

    for tf_idx, tf in enumerate(TF_ORDER):
        tf_states = states_list_by_tf.get(tf, [])
        tf_ohlcv = ohlcv_by_tf.get(tf)

        if not tf_states or tf_ohlcv is None or len(tf_ohlcv) == 0:
            # Fill defaults for this TF
            core_start = tf_idx * N_CORE
            result[:, core_start + 2] = 1.0   # variance_ratio
            result[:, core_start + 7] = 0.5   # hurst
            continue

        tf_ts = tf_ohlcv['timestamp'].values if 'timestamp' in tf_ohlcv.columns else None

        # For each anchor bar, find the matching TF bar
        prev_vel = 0.0
        for bar_idx in range(n_bars):
            # Find the TF state for this anchor timestamp
            if tf == anchor_tf:
                # Direct 1:1 alignment
                if bar_idx < len(tf_states):
                    state = tf_states[bar_idx]
                else:
                    continue
                ohlcv_slice = tf_ohlcv.iloc[:bar_idx + 1] if bar_idx < len(tf_ohlcv) else tf_ohlcv
            else:
                # Align by timestamp: find latest TF bar <= anchor timestamp
                if tf_ts is None:
                    continue
                ts = anchor_ts[bar_idx] if bar_idx < len(anchor_ts) else 0
                tf_bar_idx = np.searchsorted(tf_ts, ts, side='right') - 1
                if tf_bar_idx < 0:
                    continue
                if tf_bar_idx < len(tf_states):
                    state = tf_states[tf_bar_idx]
                else:
                    state = tf_states[-1] if tf_states else None
                if state is None:
                    continue
                ohlcv_slice = tf_ohlcv.iloc[:tf_bar_idx + 1]

            core, helper, vel = extract_tf_features(state, ohlcv_slice, prev_vel)
            prev_vel = vel

            # Write core
            core_start = tf_idx * N_CORE
            result[bar_idx, core_start:core_start + N_CORE] = core

            # Write helper
            h_start = HELPER_START + tf_idx * N_HELPER
            result[bar_idx, h_start:h_start + N_HELPER] = helper

    # Global: time_of_day
    if len(anchor_ts) == n_bars:
        result[:, GLOBAL_START] = (anchor_ts % 86400) / 86400

    return result


# --- Utility functions ---

def get_tf_core_slice(tf: str) -> slice:
    """Get the slice for a TF's 10 core features in the 79D vector."""
    idx = TF_ORDER.index(tf)
    start = idx * N_CORE
    return slice(start, start + N_CORE)


def get_tf_helper_slice(tf: str) -> slice:
    """Get the slice for a TF's 3 helper features in the 79D vector."""
    idx = TF_ORDER.index(tf)
    start = HELPER_START + idx * N_HELPER
    return slice(start, start + N_HELPER)


def get_feature_index(tf: str, feature_name: str) -> int:
    """Get the index of a specific feature in the 79D vector.

    Example: get_feature_index('1h', 'z_se') → 40
    """
    tf_idx = TF_ORDER.index(tf)
    if feature_name in CORE_FEATURE_NAMES:
        feat_idx = CORE_FEATURE_NAMES.index(feature_name)
        return tf_idx * N_CORE + feat_idx
    elif feature_name in HELPER_FEATURE_NAMES:
        feat_idx = HELPER_FEATURE_NAMES.index(feature_name)
        return HELPER_START + tf_idx * N_HELPER + feat_idx
    elif feature_name in GLOBAL_FEATURE_NAMES:
        return GLOBAL_START + GLOBAL_FEATURE_NAMES.index(feature_name)
    else:
        raise ValueError(f'Unknown feature: {feature_name}')


def describe_79d(features: np.ndarray) -> str:
    """Human-readable summary of a 79D vector. For debugging/logging."""
    lines = []
    for tf_idx, tf in enumerate(TF_ORDER):
        start = tf_idx * N_CORE
        z = features[start]
        dmi = features[start + 1]
        vr = features[start + 2]
        vel = features[start + 3]
        accel = features[start + 4]
        vol = features[start + 5]
        rng = features[start + 6]
        hurst = features[start + 7]
        rev_p = features[start + 8]
        p_ctr = features[start + 9]

        lines.append(
            f'  {tf:>4}: z={z:+6.2f} dmi={dmi:+6.1f} vr={vr:.2f} '
            f'vel={vel:+6.1f} acc={accel:+5.1f} vol={vol:.1f} '
            f'rng={rng:5.0f} H={hurst:.2f} Prev={rev_p:.2f} Pctr={p_ctr:.2f}'
        )

    tod = features[GLOBAL_START]
    hour = int(tod * 24)
    minute = int((tod * 24 - hour) * 60)
    lines.append(f'  time: {hour:02d}:{minute:02d} UTC')

    return '\n'.join(lines)
