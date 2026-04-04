"""
Canonical Feature Vectors  -- Single Source of Truth
=====================================================
Both FractalClusteringEngine.extract_features() and
TimeframeBeliefNetwork.state_to_features() delegate here.

16D base layout (fractal/structural):
  [0]  abs(z_score)
  [1]  log1p(|velocity|)
  [2]  log1p(|momentum|)
  [3]  entropy_normalized
  [4]  log2(tf_seconds)          -- timeframe scale
  [5]  depth                     -- fractal depth
  [6]  parent_is_band_reversal   -- 1.0 if parent type is BAND_REVERSAL
  [7]  adx / 100                -- self regime ADX
  [8]  hurst_exponent            -- self regime Hurst
  [9]  dmi_diff / 100            -- self regime (DMI+ - DMI-)
  [10] parent_z                  -- immediate parent |z|
  [11] parent_dmi_diff           -- immediate parent DMI diff
  [12] root_is_roche             -- 1.0 if root ancestor is BAND_REVERSAL (historical name kept for checkpoint compat)
  [13] tf_alignment              -- sign(self_dmi) * sign(root_dmi)
  [14] term_pid                  -- PID control force
  [15] oscillation_entropy_normalized  -- oscillation coherence

13D grounded layout (nightmare protocol features, computed from OHLCV):
  [16] dmi_diff_raw              -- DI+ minus DI- (raw, not /100)
  [17] dmi_gap                   -- abs(DI+ - DI-)
  [18] vol_rel                   -- volume / 30-bar volume SMA
  [19] dir_vol                   -- directional volume (sign * vol_rel)
  [20] velocity_raw              -- price rate of change (raw)
  [21] z_se                      -- z-score using standard error
  [22] price_accel               -- velocity change (accel)
  [23] std_price                 -- rolling 30-bar price std
  [24] variance_ratio            -- short/long vol ratio (regime)
  [25] bar_range                 -- (high - low) / tick
  [26] wick_ratio                -- 1 - |close-open| / range
  [27] vwap_distance             -- (price - VWAP) / tick
  [28] time_of_day               -- timestamp % 86400 / 86400

29D = 16D base + 13D grounded (--grounded mode)

23D augmented layout (--cnn-augment mode):
  [0-15]  16D base (above)
  [16] cnn_pred_dmi_diff         -- predicted dmi_diff at t+5
  [17] cnn_pred_dmi_gap          -- predicted dmi_gap at t+5
  [18] cnn_pred_vol_rel          -- predicted vol_rel at t+5
  [19] cnn_pred_dir_vol          -- predicted dir_vol at t+5
  [20] cnn_pred_velocity         -- predicted velocity at t+5
  [21] cnn_pred_z_se             -- predicted z_se at t+5
  [22] cnn_pred_price_accel      -- predicted price_accel at t+5
"""

import numpy as np
import pandas as pd

TICK = 0.25  # MNQ tick size

# Grounded feature names (nightmare protocol)
GROUNDED_FEATURE_NAMES = [
    'dmi_diff_raw', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity_raw',
    'z_se', 'price_accel', 'std_price', 'variance_ratio',
    'bar_range', 'wick_ratio', 'vwap_distance', 'time_of_day',
]
GROUNDED_DIM = len(GROUNDED_FEATURE_NAMES)  # 13


def extract_grounded_features(state, window_data: pd.DataFrame, timestamp: float) -> list:
    """Compute 13D grounded features from MarketState + OHLCV window.

    These are the features proven to have edge in the nightmare ticker.
    Computed from raw OHLCV data — no lookahead, no physics metaphors.

    Args:
        state: MarketState with dmi_plus, dmi_minus, velocity
        window_data: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
        timestamp: current bar timestamp (for time_of_day)
    Returns:
        list of 13 floats
    """
    if window_data is None or len(window_data) < 2:
        return [0.0] * GROUNDED_DIM

    closes = window_data['close'].values
    volumes = window_data['volume'].values if 'volume' in window_data.columns else np.ones(len(window_data))
    highs = window_data['high'].values
    lows = window_data['low'].values
    opens = window_data['open'].values

    price = closes[-1]
    n = len(closes)

    # DMI features (from MarketState)
    dmi_p = getattr(state, 'dmi_plus', 0.0)
    dmi_m = getattr(state, 'dmi_minus', 0.0)
    dmi_diff_raw = dmi_p - dmi_m
    dmi_gap = abs(dmi_diff_raw)

    # Volume relative to 30-bar SMA
    vol_window = volumes[-30:] if n >= 30 else volumes
    vol_avg = np.mean(vol_window) if len(vol_window) > 0 else 1.0
    vol_avg = max(vol_avg, 1.0)
    vol_rel = volumes[-1] / vol_avg

    # Directional volume
    dir_sign = 1.0 if n >= 2 and price > closes[-2] else -1.0
    dir_vol = dir_sign * vol_rel

    # Velocity (raw from state)
    velocity_raw = getattr(state, 'velocity', 0.0)

    # Z-score using standard error
    z_se = 0.0
    if n >= 15:
        window = closes[-60:] if n >= 60 else closes
        mean_p = np.mean(window)
        std_p = np.std(window)
        se = std_p / (len(window) ** 0.5) if len(window) > 1 else std_p
        z_se = (price - mean_p) / se if se > 1e-8 else 0.0

    # Price acceleration
    prev_vel = getattr(state, 'velocity', 0.0)
    # Approximate: use velocity diff if we had previous, else 0
    # In batch context this gets computed properly by the caller
    price_accel = 0.0

    # Regime features
    std_price = np.std(closes[-30:]) if n >= 30 else np.std(closes)
    variance_ratio = 1.0
    if n >= 60:
        short_std = np.std(closes[-10:])
        long_std = np.std(closes[-60:])
        variance_ratio = short_std / long_std if long_std > 1e-8 else 1.0

    # Bar features
    bar_range = (highs[-1] - lows[-1]) / TICK
    wick_ratio = 0.0
    if highs[-1] - lows[-1] > 0:
        wick_ratio = 1.0 - abs(closes[-1] - opens[-1]) / (highs[-1] - lows[-1])

    # Context
    vwap_distance = 0.0
    if n >= 30 and len(vol_window) >= 30:
        p_arr = closes[-30:]
        v_arr = volumes[-30:]
        vwap = np.sum(p_arr * v_arr) / (np.sum(v_arr) + 1e-8)
        vwap_distance = (price - vwap) / TICK

    time_of_day = (timestamp % 86400) / 86400 if timestamp > 0 else 0.0

    return [
        dmi_diff_raw, dmi_gap, vol_rel, dir_vol, velocity_raw,
        z_se, price_accel, std_price, variance_ratio,
        bar_range, wick_ratio, vwap_distance, time_of_day,
    ]


def extract_feature_vector(
    z_score: float, velocity: float, momentum: float,
    entropy_normalized: float, tf_seconds: int, depth: float,
    parent_is_band_reversal: float,
    adx: float, hurst: float, dmi_diff: float,
    parent_z: float, parent_dmi_diff: float,
    root_is_roche: float, tf_alignment: float,
    pid: float, osc_coherence: float,
) -> list:
    """Canonical 16D feature vector. Single source of truth.

    All inputs are raw values  -- compression (log1p, log2, /100) is applied here.
    Callers must NOT pre-compress.
    """
    v_feat = np.log1p(abs(velocity))
    m_feat = np.log1p(abs(momentum))
    tf_scale = np.log2(max(1, tf_seconds))

    return [
        abs(z_score), v_feat, m_feat, entropy_normalized,
        tf_scale, depth, parent_is_band_reversal,
        adx, hurst, dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
        pid, osc_coherence,
    ]


# Number of CNN-predicted features appended in augmented mode
CNN_AUGMENT_DIM = 7


def extract_augmented_feature_vector(
    z_score: float, velocity: float, momentum: float,
    entropy_normalized: float, tf_seconds: int, depth: float,
    parent_is_band_reversal: float,
    adx: float, hurst: float, dmi_diff: float,
    parent_z: float, parent_dmi_diff: float,
    root_is_roche: float, tf_alignment: float,
    pid: float, osc_coherence: float,
    cnn_predicted_7d=None,
) -> list:
    """23D feature vector = 16D base + 7D CNN-predicted state at t+5.

    When cnn_predicted_7d is None or unavailable, pads with zeros.
    The 7D CNN features are: dmi_diff, dmi_gap, vol_rel, dir_vol,
    velocity, z_se, price_accel — all predicted at the t+5 horizon.
    """
    base = extract_feature_vector(
        z_score, velocity, momentum, entropy_normalized,
        tf_seconds, depth, parent_is_band_reversal,
        adx, hurst, dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
        pid, osc_coherence,
    )
    if cnn_predicted_7d is not None and len(cnn_predicted_7d) == CNN_AUGMENT_DIM:
        base.extend(list(cnn_predicted_7d))
    else:
        base.extend([0.0] * CNN_AUGMENT_DIM)
    return base


def extract_grounded_feature_vector(
    z_score: float, velocity: float, momentum: float,
    entropy_normalized: float, tf_seconds: int, depth: float,
    parent_is_band_reversal: float,
    adx: float, hurst: float, dmi_diff: float,
    parent_z: float, parent_dmi_diff: float,
    root_is_roche: float, tf_alignment: float,
    pid: float, osc_coherence: float,
    state=None, window_data=None, timestamp: float = 0.0,
) -> list:
    """29D feature vector = 16D base + 13D grounded (nightmare protocol).

    The 16D captures fractal structure (depth, parent chain, TF scale).
    The 13D captures the grounded features proven to have trading edge
    (z_se, variance_ratio, vol_rel, bar_range, etc.).

    Args:
        (first 16 args): same as extract_feature_vector
        state: MarketState (for dmi_plus/minus, velocity)
        window_data: OHLCV DataFrame around the pattern
        timestamp: current bar timestamp
    """
    base = extract_feature_vector(
        z_score, velocity, momentum, entropy_normalized,
        tf_seconds, depth, parent_is_band_reversal,
        adx, hurst, dmi_diff,
        parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
        pid, osc_coherence,
    )
    grounded = extract_grounded_features(state, window_data, timestamp)
    base.extend(grounded)
    return base
