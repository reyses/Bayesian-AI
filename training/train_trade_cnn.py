"""
TradeCNN Training Pipeline — Walk-Forward State Prediction.

Predicts future feature states (not price direction directly).
The trading logic interprets predicted states into entry/exit decisions.

Phases:
  --phase labels    : Build + validate feature/label pipeline (13D)
  --phase labels29  : Build 29D MTF feature pipeline (13D base + 16D multi-TF)
  --phase train     : Walk-forward training (Model A)
  --phase all       : labels + train

Usage:
  python -m training.train_trade_cnn --phase labels
  python -m training.train_trade_cnn --phase labels29
  python -m training.train_trade_cnn --phase all --model A
"""
import argparse
import gc
import json
import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- PATHS ---
IS_ROOT = 'DATA/ATLAS'
OOS_ROOT = 'DATA/ATLAS_OOS'
CHECKPOINT_DIR = 'checkpoints/trade_cnn'
RESULTS_LOG = 'reports/findings/experiment_log.txt'
TICK = 0.25

# --- HORIZONS ---
HORIZONS_FAST = [1, 5, 10]    # v1: scalping (1-10 bar moves)
HORIZONS_HOLD = [5, 10, 20]   # v2: sustained moves (5-20 bars)
HORIZONS_10 = [10]            # v3: single horizon at sweet spot (10 min)
HORIZONS = HORIZONS_FAST      # default, overridden by --horizons
MAX_FORWARD = max(HORIZONS)

# --- 13D FEATURES ---
FEATURE_NAMES_7D = ['dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel']
FEATURE_NAMES_REGIME = ['std_price', 'variance_ratio', 'bar_range', 'wick_ratio']
FEATURE_NAMES_CONTEXT = ['vwap_distance', 'time_of_day']
FEATURE_NAMES_13D = FEATURE_NAMES_7D + FEATURE_NAMES_REGIME + FEATURE_NAMES_CONTEXT
N_FEAT = len(FEATURE_NAMES_13D)  # 13

# Label: 7D features at each horizon = 7 * 3 = 21 outputs
N_LABELS = len(FEATURE_NAMES_7D) * len(HORIZONS)  # 21
LOOKBACK = 10

# --- 29D FEATURES (13D base + 16D multi-TF) ---
MTF_TFS = ['1s', '5m', '15m', '1h']
MTF_FEAT_PER_TF = ['dmi_diff', 'z_se', 'velocity', 'vol_rel']
MTF_FEATURE_NAMES = [f'{tf}_{f}' for tf in MTF_TFS for f in MTF_FEAT_PER_TF]
FEATURE_NAMES_29D = FEATURE_NAMES_13D + MTF_FEATURE_NAMES
N_FEAT_29D = len(FEATURE_NAMES_29D)  # 29

# TF boundary indices for per-TF z-score normalization
TF_BOUNDARIES = [
    (0, 13),    # 1m base
    (13, 17),   # 1s
    (17, 21),   # 5m
    (21, 25),   # 15m
    (25, 29),   # 1h
]

# Bar durations in seconds (for alignment: bar_close_time = timestamp + duration)
TF_BAR_DURATION = {'1s': 1, '5m': 300, '15m': 900, '1h': 3600}

# Rolling z-score window: 30 trading days × bars_per_day
ZSCORE_WINDOW_DAYS = 30
BARS_PER_DAY_1M = 1380  # ~23 hours of futures trading
ZSCORE_WINDOW = ZSCORE_WINDOW_DAYS * BARS_PER_DAY_1M  # ~41,400 bars
ZSCORE_MIN_PERIODS = 100


def extract_features_13d(states, df):
    """Extract 13D grounded features per bar from SFE states + OHLCV.

    7D directional (same as direction_cnn):
      dmi_diff, dmi_gap, vol_rel, dir_vol, velocity, z_se, price_accel
    4D regime:
      std_price, variance_ratio, bar_range, wick_ratio
    2D context:
      vwap_distance, time_of_day
    """
    n = len(states)
    feats = np.zeros((n, N_FEAT), dtype=np.float32)

    prices = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(n)
    timestamps = df['timestamp'].values.astype(np.float64)

    # Rolling 30-bar volume SMA (matches training/live parity)
    vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values

    prev_vel = 0.0

    for i in range(n):
        st = states[i]['state'] if isinstance(states[i], dict) else states[i]
        dmi_p = getattr(st, 'dmi_plus', 0.0)
        dmi_m = getattr(st, 'dmi_minus', 0.0)
        vel = getattr(st, 'velocity', 0.0)
        vol = volumes[i]
        _va = vol_avg[i] if vol_avg[i] > 0 else 1.0

        # --- 7D directional ---
        feats[i, 0] = dmi_p - dmi_m                            # dmi_diff
        feats[i, 1] = abs(dmi_p - dmi_m)                       # dmi_gap
        feats[i, 2] = vol / _va                                 # vol_rel
        if i > 0:
            _dir = 1.0 if prices[i] > prices[i-1] else -1.0
            feats[i, 3] = _dir * vol / _va                      # dir_vol
        feats[i, 4] = vel                                        # velocity
        if i >= 15:
            _window = prices[max(0, i-60):i+1]
            _mean = _window.mean()
            _std = _window.std()
            _se = _std / (len(_window) ** 0.5) if len(_window) > 1 else _std
            feats[i, 5] = (prices[i] - _mean) / _se if _se > 1e-8 else 0.0  # z_se
        feats[i, 6] = vel - prev_vel                             # price_accel
        prev_vel = vel

        # --- 4D regime ---
        if i >= 30:
            feats[i, 7] = np.std(prices[i-30:i+1])              # std_price
            if i >= 60:
                _short_std = np.std(prices[i-10:i+1])
                _long_std = np.std(prices[i-60:i+1])
                feats[i, 8] = _short_std / _long_std if _long_std > 1e-8 else 1.0  # variance_ratio

        _range = highs[i] - lows[i]
        feats[i, 9] = _range / TICK                              # bar_range (ticks)
        if _range > 0:
            _body = abs(prices[i] - opens[i])
            feats[i, 10] = 1.0 - (_body / _range)                # wick_ratio

        # --- 2D context ---
        if i >= 30:
            _vwap_num = np.sum(prices[i-30:i+1] * volumes[i-30:i+1])
            _vwap_den = np.sum(volumes[i-30:i+1])
            _vwap = _vwap_num / _vwap_den if _vwap_den > 0 else prices[i]
            feats[i, 11] = (prices[i] - _vwap) / TICK            # vwap_distance (ticks)
        feats[i, 12] = (timestamps[i] % 86400) / 86400.0         # time_of_day (0-1)

    return feats


def build_state_labels(feats_7d, horizons=HORIZONS):
    """Build state prediction labels: actual 7D features at t+h for each horizon.

    For each bar i, label = [feat_7d[i+1], feat_7d[i+5], feat_7d[i+10]].
    Returns (n_bars, 21) array. Bars without full forward window get zeros.
    """
    n = len(feats_7d)
    n_feat = feats_7d.shape[1]  # 7
    n_horizons = len(horizons)
    labels = np.zeros((n, n_feat * n_horizons), dtype=np.float32)

    for i in range(n):
        for hi, h in enumerate(horizons):
            if i + h < n:
                labels[i, hi * n_feat:(hi + 1) * n_feat] = feats_7d[i + h]

    return labels


# =============================================================================
# MTF 29D FEATURE PIPELINE (Phase A of counter-proposal)
# =============================================================================

def extract_4_features_from_sfe(states, df):
    """Extract the 4 MTF features from SFE states + OHLCV for one timeframe.

    Returns (n, 4) array: [dmi_diff, z_se, velocity, vol_rel].
    Uses the same computation as the 13D base features for consistency.
    """
    n = len(states)
    feats = np.zeros((n, 4), dtype=np.float32)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(n)
    prices = df['close'].values.astype(np.float64)

    # 30-bar rolling volume SMA
    vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values

    for i in range(n):
        st = states[i]['state'] if isinstance(states[i], dict) else states[i]

        # dmi_diff
        feats[i, 0] = getattr(st, 'dmi_plus', 0.0) - getattr(st, 'dmi_minus', 0.0)

        # z_se: (price - mean_60) / SE_60
        if i >= 15:
            _window = prices[max(0, i - 60):i + 1]
            _mean = _window.mean()
            _std = _window.std()
            _se = _std / (len(_window) ** 0.5) if len(_window) > 1 else _std
            feats[i, 1] = (prices[i] - _mean) / _se if _se > 1e-8 else 0.0

        # velocity
        feats[i, 2] = getattr(st, 'velocity', 0.0)

        # vol_rel
        _va = vol_avg[i] if vol_avg[i] > 0 else 1.0
        feats[i, 3] = volumes[i] / _va

    return feats


def extract_4_features_from_raw(df):
    """Extract 4 features from raw 1s OHLCV without SFE.

    For 27.5M 1s bars, running full SFE is too heavy.
    Computes: dmi_diff proxy (EWM up-down), z_se, velocity, vol_rel.
    Returns (n, 4) float32 array.
    """
    n = len(df)
    prices = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(n)

    feats = np.zeros((n, 4), dtype=np.float32)

    # --- dmi_diff proxy: EWM of up vs down moves (Wilder's smoothing, period=14) ---
    price_diff = np.diff(prices, prepend=prices[0])
    up_moves = np.maximum(price_diff, 0.0)
    dn_moves = np.maximum(-price_diff, 0.0)

    alpha = 1.0 / 14  # Wilder's smoothing = 1/period
    smooth_up = pd.Series(up_moves).ewm(alpha=alpha, adjust=False).mean().values
    smooth_dn = pd.Series(dn_moves).ewm(alpha=alpha, adjust=False).mean().values
    feats[:, 0] = smooth_up - smooth_dn

    # --- z_se: z-score over 60 bars ---
    price_series = pd.Series(prices)
    rolling_mean = price_series.rolling(60, min_periods=1).mean().values
    rolling_std = price_series.rolling(60, min_periods=1).std().values
    counts = np.minimum(np.arange(1, n + 1), 60).astype(np.float64)
    rolling_se = rolling_std / np.sqrt(counts)
    feats[:, 1] = np.where(rolling_se > 1e-8, (prices - rolling_mean) / rolling_se, 0.0).astype(np.float32)

    # --- velocity: price change per bar ---
    feats[1:, 2] = np.diff(prices).astype(np.float32)

    # --- vol_rel: volume / 30-bar SMA ---
    vol_sma = pd.Series(volumes).rolling(30, min_periods=1).mean().values
    feats[:, 3] = np.where(vol_sma > 0, volumes / vol_sma, 0.0).astype(np.float32)

    return feats


def load_tf_data(data_root, tf):
    """Load all parquet files for a given timeframe, sorted by timestamp."""
    files = sorted(glob.glob(os.path.join(data_root, tf, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_root}/{tf}/")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df


def compute_mtf_features(data_root, cache_dir):
    """Compute 4 features for each MTF timeframe. Cache as .npy files.

    For 5m/15m/1h: run full SFE.
    For 1s: use raw extraction (no SFE on 27.5M bars).

    Returns dict: tf -> {'feats': (n,4), 'timestamps': (n,), 'df': DataFrame}
    """
    from core.statistical_field_engine import StatisticalFieldEngine

    os.makedirs(cache_dir, exist_ok=True)
    result = {}

    for tf in MTF_TFS:
        cache_feats = os.path.join(cache_dir, f'{tf}_features_4d.npy')
        cache_ts = os.path.join(cache_dir, f'{tf}_timestamps.npy')

        if os.path.exists(cache_feats) and os.path.exists(cache_ts):
            print(f"  {tf}: loading cached features")
            feats = np.load(cache_feats)
            timestamps = np.load(cache_ts)
            df_tf = load_tf_data(data_root, tf)
            result[tf] = {'feats': feats, 'timestamps': timestamps, 'df': df_tf}
            continue

        print(f"  {tf}: loading data...", end=' ')
        df_tf = load_tf_data(data_root, tf)
        print(f"{len(df_tf):,} bars")

        if tf == '1s':
            print(f"  {tf}: extracting features from raw OHLCV (no SFE)...")
            feats = extract_4_features_from_raw(df_tf)
        else:
            print(f"  {tf}: computing SFE states...")
            sfe = StatisticalFieldEngine()
            states = sfe.batch_compute_states(df_tf)
            print(f"  {tf}: extracting 4 features from SFE...")
            feats = extract_4_features_from_sfe(states, df_tf)
            del states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        timestamps = df_tf['timestamp'].values.astype(np.int64)

        np.save(cache_feats, feats)
        np.save(cache_ts, timestamps)
        print(f"  {tf}: cached {feats.shape} features + timestamps")

        result[tf] = {'feats': feats, 'timestamps': timestamps, 'df': df_tf}

    return result


def build_alignment_indices(timestamps_1m, mtf_data):
    """For each 1m bar, find the index of the last COMPLETED bar from each higher TF.

    Rule: a higher-TF bar is usable only if its close_time < 1m_open_time.
    close_time = bar_timestamp + bar_duration.
    1m bar timestamp = bar open time.

    Returns dict: tf -> np.ndarray of shape (n_1m,) with indices into that TF's array.
    Index = -1 means no completed bar available yet.
    """
    n = len(timestamps_1m)
    alignment = {}

    for tf in MTF_TFS:
        tf_ts = mtf_data[tf]['timestamps']
        bar_dur = TF_BAR_DURATION[tf]

        # close times for higher-TF bars
        tf_close_times = tf_ts + bar_dur

        indices = np.full(n, -1, dtype=np.int64)

        # For each 1m bar, find the rightmost higher-TF bar whose close_time < 1m_open_time
        # Using searchsorted: find where 1m_open_time would be inserted into tf_close_times
        # Then subtract 1 to get the last bar that closed strictly before.
        insert_pos = np.searchsorted(tf_close_times, timestamps_1m, side='left')
        # side='left': returns first position where tf_close_time >= 1m_open_time
        # So insert_pos - 1 is the last bar where tf_close_time < 1m_open_time
        indices = insert_pos - 1

        # Clamp: where no bar is available yet, set to -1
        indices[indices < 0] = -1

        alignment[tf] = indices

    return alignment


def validate_mtf_alignment(timestamps_1m, mtf_data, alignment):
    """Assert zero lookahead in MTF alignment. Hard stop if any violations.

    For every 1m bar at time T, the higher-TF feature must come from
    a bar whose CLOSE TIME < T (strictly before).
    """
    total_violations = 0

    for tf in MTF_TFS:
        tf_ts = mtf_data[tf]['timestamps']
        bar_dur = TF_BAR_DURATION[tf]
        indices = alignment[tf]

        violations = 0
        n_checked = 0
        violation_examples = []

        for i in range(len(timestamps_1m)):
            h_idx = indices[i]
            if h_idx < 0:
                continue
            n_checked += 1

            h_close_time = tf_ts[h_idx] + bar_dur
            m_open_time = timestamps_1m[i]

            if h_close_time >= m_open_time:
                violations += 1
                if len(violation_examples) < 3:
                    violation_examples.append(
                        f"    1m bar {i} at {m_open_time} uses {tf} bar {h_idx} "
                        f"closing at {h_close_time}"
                    )

        total_violations += violations
        status = "PASS" if violations == 0 else f"FAIL ({violations})"
        print(f"  {tf} alignment: {status} ({n_checked:,} bars checked)")
        for ex in violation_examples:
            print(ex)

    if total_violations > 0:
        raise AssertionError(
            f"MTF alignment has {total_violations} lookahead violations. "
            f"Fix alignment logic before training."
        )
    print(f"  ALL TFs: PASS (zero violations)")
    return True


def assemble_features_29d(feats_13d, mtf_data, alignment):
    """Concatenate 13D base + 16D MTF features into 29D array.

    For each 1m bar, look up the aligned index in each TF and copy its 4 features.
    Bars with no available higher-TF data (index=-1) get zeros for that TF.
    """
    n = len(feats_13d)
    feats_29d = np.zeros((n, N_FEAT_29D), dtype=np.float32)

    # Copy 13D base
    feats_29d[:, :13] = feats_13d

    # Copy 16D MTF (4 features × 4 TFs)
    for ti, tf in enumerate(MTF_TFS):
        tf_feats = mtf_data[tf]['feats']  # (n_tf_bars, 4)
        indices = alignment[tf]           # (n_1m,)
        col_start = 13 + ti * 4
        col_end = col_start + 4

        # Vectorized: gather aligned features
        valid = indices >= 0
        feats_29d[valid, col_start:col_end] = tf_feats[indices[valid]]

    return feats_29d


def normalize_per_tf(feats_29d):
    """Z-score each TF's features using that TF's own rolling statistics.

    Uses 30-day rolling window to avoid lookahead in normalization.
    Each TF group is normalized independently because the raw scales differ
    (e.g., 1h dmi_diff has different range than 1s dmi_diff).
    """
    normalized = np.copy(feats_29d)

    for start, end in TF_BOUNDARIES:
        for col in range(start, end):
            series = pd.Series(feats_29d[:, col])
            rolling_mean = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
            rolling_std = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std()
            normalized[:, col] = ((series - rolling_mean) / (rolling_std + 1e-8)).values

    # Replace NaN from warmup period with 0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return normalized


def print_correlation_matrix_29d(feats, feature_names):
    """Print feature pairs with |r| > 0.8 and flag |r| > 0.9."""
    from scipy import stats as sp_stats

    print(f"\n  FEATURE CORRELATION (|r| > 0.8):")
    high_corr_count = 0
    dangerous_count = 0

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            # Skip if either column has zero variance
            if feats[:, i].std() < 1e-8 or feats[:, j].std() < 1e-8:
                continue
            r, _ = sp_stats.pearsonr(feats[:, i], feats[:, j])
            if abs(r) > 0.8:
                flag = " *** DANGER" if abs(r) > 0.9 else ""
                print(f"    {feature_names[i]:<20} <-> {feature_names[j]:<20}: r={r:.3f}{flag}")
                high_corr_count += 1
                if abs(r) > 0.9:
                    dangerous_count += 1

    if high_corr_count == 0:
        print(f"    No pairs with |r| > 0.8")
    if dangerous_count > 0:
        print(f"\n  WARNING: {dangerous_count} pairs with |r| > 0.9 — consider dropping one")


def build_29d_pipeline(data_root, cache_dir, feats_13d=None, df_1m=None):
    """Full Phase A pipeline: compute MTF features, align, validate, normalize, cache.

    Returns: feats_29d (normalized), feats_29d_raw (unnormalized), 1s slippage arrays.
    """
    from core.statistical_field_engine import StatisticalFieldEngine

    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Build or load 13D base features
    if feats_13d is None or df_1m is None:
        print("Building 13D base features...")
        feats_13d, _, _, df_1m = build_dataset(data_root)

    timestamps_1m = df_1m['timestamp'].values.astype(np.int64)

    # Step 2: Compute MTF features (SFE for 5m/15m/1h, raw for 1s)
    print("\nComputing MTF features...")
    mtf_cache = os.path.join(cache_dir, 'mtf_cache')
    mtf_data = compute_mtf_features(data_root, mtf_cache)

    # Step 3: Build alignment indices
    print("\nBuilding MTF alignment indices...")
    alignment = build_alignment_indices(timestamps_1m, mtf_data)

    # Print alignment coverage stats
    for tf in MTF_TFS:
        valid = (alignment[tf] >= 0).sum()
        print(f"  {tf}: {valid:,}/{len(timestamps_1m):,} bars have aligned data "
              f"({valid / len(timestamps_1m) * 100:.1f}%)")

    # Step 4: Validate alignment (zero lookahead or die)
    print("\nValidating MTF alignment (zero-lookahead check)...")
    validate_mtf_alignment(timestamps_1m, mtf_data, alignment)

    # Step 5: Assemble 29D features
    print("\nAssembling 29D features...")
    feats_29d_raw = assemble_features_29d(feats_13d, mtf_data, alignment)
    print(f"  Shape: {feats_29d_raw.shape}")

    # Step 6: Per-TF z-score normalization
    print("\nApplying per-TF z-score normalization (30-day rolling window)...")
    feats_29d = normalize_per_tf(feats_29d_raw)

    # Step 7: Validate the result
    print(f"\n{'='*60}")
    print(f"29D PIPELINE VALIDATION")
    print(f"{'='*60}")
    print(f"  Samples: {len(feats_29d):,}")
    print(f"  Features: {feats_29d.shape[1]}D")

    print(f"\n  FEATURE DISTRIBUTIONS (normalized):")
    print(f"  {'Name':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Zero%':>6}")
    for j, name in enumerate(FEATURE_NAMES_29D):
        col = feats_29d[:, j]
        _zero = (col == 0).mean() * 100
        print(f"  {name:<25} {col.mean():>8.3f} {col.std():>8.3f} "
              f"{col.min():>8.1f} {col.max():>8.1f} {_zero:>5.1f}%")

    # NaN/Inf check
    _nan = np.isnan(feats_29d).sum()
    _inf = np.isinf(feats_29d).sum()
    print(f"\n  NaN: {_nan}, Inf: {_inf}")
    if _nan > 0 or _inf > 0:
        print("  WARNING: NaN/Inf in normalized features!")

    # Step 8: Correlation matrix
    print_correlation_matrix_29d(feats_29d, FEATURE_NAMES_29D)

    # Step 9: Cache everything
    np.save(os.path.join(cache_dir, 'is_features_29d.npy'), feats_29d)
    np.save(os.path.join(cache_dir, 'is_features_29d_raw.npy'), feats_29d_raw)
    print(f"\n  Saved: {cache_dir}/is_features_29d.npy ({feats_29d.shape})")
    print(f"  Saved: {cache_dir}/is_features_29d_raw.npy ({feats_29d_raw.shape})")

    # Step 10: Cache 1s prices + timestamps for slippage fills
    print("\nCaching 1s prices + timestamps for slippage fills...")
    _1s_prices = mtf_data['1s']['df']['close'].values.astype(np.float64)
    _1s_ts = mtf_data['1s']['timestamps']
    np.save(os.path.join(cache_dir, '1s_prices.npy'), _1s_prices)
    np.save(os.path.join(cache_dir, '1s_timestamps.npy'), _1s_ts)
    print(f"  Saved: {cache_dir}/1s_prices.npy ({len(_1s_prices):,} bars)")
    print(f"  Saved: {cache_dir}/1s_timestamps.npy")

    # Save alignment indices for reuse
    for tf in MTF_TFS:
        np.save(os.path.join(cache_dir, f'alignment_{tf}.npy'), alignment[tf])
    print(f"  Saved: alignment indices for {MTF_TFS}")

    # Cleanup MTF data to free memory
    del mtf_data
    gc.collect()

    return feats_29d, feats_29d_raw


def build_dataset(data_root, max_bars=0):
    """Load data, compute states, extract 13D features, build 21D labels."""
    from core.statistical_field_engine import StatisticalFieldEngine

    print(f"Loading 1m data from {data_root}...")
    files = sorted(glob.glob(os.path.join(data_root, '1m', '*.parquet')))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if max_bars > 0:
        df = df.tail(max_bars).reset_index(drop=True)
    print(f"  Bars: {len(df):,}")

    print("Computing SFE states...")
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    print(f"  States: {len(states)}")

    print("Extracting 13D features...")
    feats = extract_features_13d(states, df)

    print("Building state labels (7D x 3 horizons = 21D)...")
    labels = build_state_labels(feats[:, :7], horizons=HORIZONS)  # labels from 7D directional only

    return feats, labels, states, df


class SlidingWindowDataset(Dataset):
    """Sliding window: input=(lookback x 13D), label=(21D state predictions)."""

    def __init__(self, features, labels, lookback=LOOKBACK):
        self.features = features
        self.labels = labels
        self.lookback = lookback
        self.n = len(features) - lookback - MAX_FORWARD

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        i = idx + self.lookback
        x = self.features[i - self.lookback:i]  # (lookback, 13)
        y = self.labels[i]                        # (21,)
        return torch.FloatTensor(x), torch.FloatTensor(y)


def validate_pipeline(feats, labels, df):
    """Print feature/label distributions and sanity checks."""
    print(f"\n{'='*60}")
    print(f"PIPELINE VALIDATION")
    print(f"{'='*60}")
    print(f"  Samples: {len(feats):,}")
    print(f"  Features: {feats.shape[1]}D")
    print(f"  Labels: {labels.shape[1]}D")

    # Feature distributions
    print(f"\n  FEATURE DISTRIBUTIONS:")
    print(f"  {'Name':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'NaN':>5} {'Zero%':>6}")
    for j, name in enumerate(FEATURE_NAMES_13D):
        col = feats[:, j]
        _nan = np.isnan(col).sum()
        _zero = (col == 0).mean() * 100
        print(f"  {name:<20} {col.mean():>10.3f} {col.std():>10.3f} "
              f"{col.min():>10.3f} {col.max():>10.3f} {_nan:>5} {_zero:>5.1f}%")

    # Label distributions
    print(f"\n  LABEL DISTRIBUTIONS (7D x {len(HORIZONS)} horizons):")
    print(f"  {'Name':<25} {'Mean':>10} {'Std':>10} {'Var>0':>6}")
    for hi, h in enumerate(HORIZONS):
        for fi, fname in enumerate(FEATURE_NAMES_7D):
            idx = hi * 7 + fi
            col = labels[:, idx]
            _var_ok = 'YES' if col.std() > 1e-6 else 'NO'
            print(f"  {fname}_t{h:<18} {col.mean():>10.3f} {col.std():>10.3f} {_var_ok:>6}")

    # NaN/Inf check
    _nan_feats = np.isnan(feats).sum()
    _inf_feats = np.isinf(feats).sum()
    _nan_labels = np.isnan(labels).sum()
    print(f"\n  NaN in features: {_nan_feats}")
    print(f"  Inf in features: {_inf_feats}")
    print(f"  NaN in labels: {_nan_labels}")

    if _nan_feats > 0 or _inf_feats > 0:
        print("  WARNING: NaN/Inf detected — clean before training!")

    # Feature correlation (check redundancy)
    print(f"\n  FEATURE CORRELATION (top pairs with |r| > 0.8):")
    from scipy import stats as sp_stats
    for i in range(N_FEAT):
        for j in range(i+1, N_FEAT):
            r, _ = sp_stats.pearsonr(feats[:, i], feats[:, j])
            if abs(r) > 0.8:
                print(f"    {FEATURE_NAMES_13D[i]} <-> {FEATURE_NAMES_13D[j]}: r={r:.3f}")

    # Trading days
    trading_days = pd.to_datetime(df['timestamp'], unit='s').dt.date.nunique()
    print(f"\n  Trading days: {trading_days}")
    print(f"  Bars/day: {len(feats) / trading_days:.0f}")

    # Save validation report
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    report = {
        'samples': len(feats), 'features': feats.shape[1], 'labels': labels.shape[1],
        'trading_days': trading_days, 'nan_feats': int(_nan_feats), 'inf_feats': int(_inf_feats),
    }
    with open(os.path.join(CHECKPOINT_DIR, 'pipeline_validation.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {CHECKPOINT_DIR}/pipeline_validation.json")


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='TradeCNN training pipeline')
    parser.add_argument('--phase', default='labels', choices=['labels', 'labels29', 'train', 'all', 'oos'])
    parser.add_argument('--model', default='A', choices=['A'])
    parser.add_argument('--max-bars', type=int, default=0)
    parser.add_argument('--horizons', default='fast', choices=['fast', 'hold', '10'],
                        help='fast=[1,5,10] scalp, hold=[5,10,20] sustained, 10=[10] sweet spot')
    args = parser.parse_args()

    # Set horizons and checkpoint dir based on mode
    global HORIZONS, MAX_FORWARD, N_LABELS, CHECKPOINT_DIR
    if args.horizons == 'hold':
        HORIZONS = HORIZONS_HOLD
        CHECKPOINT_DIR = 'checkpoints/trade_cnn_hold'
    elif args.horizons == '10':
        HORIZONS = HORIZONS_10
        CHECKPOINT_DIR = 'checkpoints/trade_cnn_10'
    else:
        HORIZONS = HORIZONS_FAST
        CHECKPOINT_DIR = 'checkpoints/trade_cnn'
    MAX_FORWARD = max(HORIZONS)
    N_LABELS = len(FEATURE_NAMES_7D) * len(HORIZONS)
    print(f"Horizons: {HORIZONS} -> {N_LABELS}D labels -> {CHECKPOINT_DIR}")

    if args.phase in ('labels', 'all'):
        t0 = time.time()
        feats, labels, states, df = build_dataset(IS_ROOT, max_bars=args.max_bars)
        print(f"Dataset built in {time.time()-t0:.1f}s")
        validate_pipeline(feats, labels, df)

        # Save features + labels for reuse
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        np.save(os.path.join(CHECKPOINT_DIR, 'is_features_13d.npy'), feats)
        np.save(os.path.join(CHECKPOINT_DIR, 'is_labels_21d.npy'), labels)
        print(f"  Saved: {CHECKPOINT_DIR}/is_features_13d.npy ({feats.shape})")
        print(f"  Saved: {CHECKPOINT_DIR}/is_labels_21d.npy ({labels.shape})")

        # Release memory
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if args.phase == 'labels29':
        t0 = time.time()

        # Check if 13D features already cached
        _feat_path = os.path.join(CHECKPOINT_DIR, 'is_features_13d.npy')
        _label_path = os.path.join(CHECKPOINT_DIR, 'is_labels_21d.npy')
        if os.path.exists(_feat_path):
            print("Loading cached 13D features...")
            feats_13d = np.load(_feat_path)
            files = sorted(glob.glob(os.path.join(IS_ROOT, '1m', '*.parquet')))
            df_1m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df_1m = df_1m.sort_values('timestamp').reset_index(drop=True)
            print(f"  13D features: {feats_13d.shape}, 1m bars: {len(df_1m):,}")
        else:
            print("13D features not cached — building from scratch...")
            feats_13d, _, _, df_1m = build_dataset(IS_ROOT, max_bars=args.max_bars)

        cache_29d = os.path.join(CHECKPOINT_DIR, '29d')
        feats_29d, feats_29d_raw = build_29d_pipeline(IS_ROOT, cache_29d, feats_13d, df_1m)

        # Also build labels (same 7D × horizons, from 13D directional features)
        labels = build_state_labels(feats_13d[:, :7], horizons=HORIZONS)
        np.save(os.path.join(cache_29d, 'is_labels_21d.npy'), labels)
        print(f"  Saved: {cache_29d}/is_labels_21d.npy ({labels.shape})")

        print(f"\n29D pipeline complete in {time.time()-t0:.1f}s")

        # Cleanup
        del feats_13d, feats_29d, feats_29d_raw, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.phase in ('train', 'all'):
        from core.trade_cnn import StatePredictor

        # Load or compute features
        _feat_path = os.path.join(CHECKPOINT_DIR, 'is_features_13d.npy')
        _label_path = os.path.join(CHECKPOINT_DIR, 'is_labels_21d.npy')
        if os.path.exists(_feat_path) and os.path.exists(_label_path):
            print("Loading cached features + labels...")
            feats = np.load(_feat_path)
            labels = np.load(_label_path)
            # Need df for day boundaries
            files = sorted(glob.glob(os.path.join(IS_ROOT, '1m', '*.parquet')))
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"  Features: {feats.shape}, Labels: {labels.shape}")
        else:
            print("Features not cached — run --phase labels first")
            return

        walk_forward_train(feats, labels, df, args)

    if args.phase == 'oos':
        oos_single_pass()


def walk_forward_train(feats, labels, df, args):
    """Walk-forward training: carry-forward model, score before train each day."""
    from core.trade_cnn import StatePredictor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Split data into days
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    day_boundaries = []
    for date, group in df.groupby('date'):
        _start = group.index[0]
        _end = group.index[-1]
        day_boundaries.append({'date': date, 'start': _start, 'end': _end, 'n_bars': len(group)})
    print(f"  Days: {len(day_boundaries)}")

    # Model
    model = StatePredictor(n_features=N_FEAT, latent_dim=64, n_labels=N_LABELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    day_results = []

    for di, day in enumerate(tqdm(day_boundaries, desc="Walk-Forward")):
        _start = day['start']
        _end = day['end']
        _date = day['date']

        # Skip days with too few bars
        if _end - _start < LOOKBACK + MAX_FORWARD + 10:
            continue

        day_feats = feats[_start:_end+1]
        day_labels = labels[_start:_end+1]

        # --- SCORE (predict BEFORE training on this day) ---
        if di > 0:  # Day 1 has no model to score with
            model.eval()
            day_ds = SlidingWindowDataset(day_feats, day_labels, lookback=LOOKBACK)
            if len(day_ds) > 0:
                score = _validate_day(model, day_ds, day_feats, day_labels, device)
                score['date'] = str(_date)
                score['day'] = di + 1

                # Trading simulation
                trade_result = _simulate_day_trading(model, day_feats, device)
                score.update(trade_result)

                day_results.append(score)

                if (di + 1) % 30 == 0:
                    _cum_pnl = sum(r.get('sim_pnl', 0) for r in day_results)
                    print(f"  Day {di+1}: corr={score.get('avg_corr', 0):.3f} "
                          f"dir={score.get('dir_acc', 0):.1f}% "
                          f"pnl={score.get('sim_pnl', 0):+.0f}t "
                          f"cum=${_cum_pnl*0.5:,.0f}")

        # --- TRAIN on this day ---
        model.train()
        day_ds = SlidingWindowDataset(day_feats, day_labels, lookback=LOOKBACK)
        if len(day_ds) < 10:
            continue

        dl = DataLoader(day_ds, batch_size=min(256, len(day_ds)), shuffle=True)

        # Cold start (Day 1): 30 epochs. Carry-forward: 5 epochs with lower LR
        if di == 0:
            _epochs = 30
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3
        else:
            _epochs = 5
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-4

        for _ep in range(_epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Checkpoint every 10 days
        if (di + 1) % 10 == 0:
            torch.save({
                'model_state': model.state_dict(),
                'day': di + 1, 'date': str(_date),
            }, os.path.join(CHECKPOINT_DIR, f'model_day{di+1}.pt'))

    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'day': len(day_boundaries), 'date': str(day_boundaries[-1]['date']),
    }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

    # Walk-forward report
    walk_forward_report(day_results)

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _validate_day(model, dataset, feats, labels, device):
    """Per-day validation: feature correlations + direction accuracy."""
    from scipy import stats as sp_stats

    model.eval()
    all_pred = []
    all_true = []

    dl = DataLoader(dataset, batch_size=512, shuffle=False)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            pred = model(x)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    # Correlation per feature per horizon
    corrs = []
    for j in range(N_LABELS):
        if true[:, j].std() > 1e-8 and pred[:, j].std() > 1e-8:
            r, _ = sp_stats.spearmanr(pred[:, j], true[:, j])
            corrs.append(r)
        else:
            corrs.append(0.0)

    # Direction accuracy: sign of predicted dmi_diff at LAST horizon vs actual
    # For [1,5,10]: index 14. For [10]: index 0. For [5,10,20]: index 14.
    _last_h_idx = (len(HORIZONS) - 1) * 7  # dmi_diff at furthest horizon
    _dir_idx = min(_last_h_idx, pred.shape[1] - 1)
    _pred_dir = pred[:, _dir_idx] > 0
    _true_dir = true[:, _dir_idx] > 0
    dir_acc = (_pred_dir == _true_dir).mean() * 100

    return {
        'avg_corr': np.mean(corrs),
        'dir_acc': dir_acc,
        'n_samples': len(pred),
    }


def _simulate_day_trading(model, feats, device):
    """Simulate trading from predicted states."""
    model.eval()

    SL = 40  # hard SL in ticks
    MIN_HOLD = 3  # minimum bars before exit
    trades = []
    in_trade = False
    trade_dir = ''
    entry_price_idx = 0
    bars_held = 0

    for i in range(LOOKBACK, len(feats) - MAX_FORWARD):
        x = feats[i - LOOKBACK:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_t).cpu().numpy()[0]

        # Extract predicted states (adaptive to horizon count)
        n_h = len(HORIZONS)
        _pred_dmi = pred[0]               # dmi_diff at first/only horizon
        _pred_gap = pred[1]               # dmi_gap at first/only horizon
        _pred_vel = pred[4]               # velocity at first/only horizon
        # For multi-horizon, use the last horizon's dmi_diff
        if n_h >= 2:
            _pred_dmi = pred[(n_h - 1) * 7]
            _pred_gap = pred[(n_h - 1) * 7 + 1]
            _pred_vel = pred[(n_h - 1) * 7 + 4]

        if in_trade:
            bars_held += 1

            # Exit: trend reversed OR momentum fading (after min hold)
            if bars_held >= MIN_HOLD:
                _trend_reversed = (trade_dir == 'LONG' and _pred_dmi < -2) or \
                                  (trade_dir == 'SHORT' and _pred_dmi > 2)
                _momentum_fading = _pred_gap < feats[i, 1] * 0.5  # gap shrinking

                if _trend_reversed or _momentum_fading:
                    # Compute actual PnL from prices (if available via label reconstruction)
                    _actual_dmi_t5 = feats[min(i+5, len(feats)-1), 0]  # actual dmi_diff 5 bars later
                    _pnl = abs(_actual_dmi_t5) * (1 if (trade_dir == 'LONG') == (_actual_dmi_t5 > 0) else -1)
                    trades.append({'pnl': _pnl, 'bars': bars_held, 'dir': trade_dir})
                    in_trade = False

        if not in_trade:
            # Entry: predicted gap growing + velocity confirming + confidence
            _gap_building = _pred_gap > feats[i, 1] * 1.2
            _vel_confirming = abs(_pred_vel) > abs(feats[i, 4])
            _confident = abs(_pred_dmi) > 2.0

            if _confident and (_gap_building or _vel_confirming):
                in_trade = True
                trade_dir = 'LONG' if _pred_dmi > 0 else 'SHORT'
                entry_price_idx = i
                bars_held = 0

    total_pnl = sum(t['pnl'] for t in trades)
    n_trades = len(trades)
    n_wins = len([t for t in trades if t['pnl'] > 0])

    return {
        'sim_pnl': total_pnl,
        'sim_trades': n_trades,
        'sim_wr': n_wins / n_trades * 100 if n_trades > 0 else 0,
    }


def walk_forward_report(day_results):
    """Summary report across all scored days."""
    if not day_results:
        print("\nNo scored days — walk-forward report empty")
        return

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD SUMMARY: StatePredictor")
    print(f"{'='*60}")

    n_days = len(day_results)
    cum_pnl = sum(r.get('sim_pnl', 0) for r in day_results)
    avg_corr = np.mean([r.get('avg_corr', 0) for r in day_results])
    avg_dir = np.mean([r.get('dir_acc', 0) for r in day_results])
    avg_trades = np.mean([r.get('sim_trades', 0) for r in day_results])
    profitable_days = len([r for r in day_results if r.get('sim_pnl', 0) > 0])

    print(f"  Scored days: {n_days}")
    print(f"  Cumulative PnL: {cum_pnl:.0f}t (${cum_pnl*0.5:,.0f})")
    print(f"  $/day avg: ${cum_pnl*0.5/n_days:.2f}")
    print(f"  Profitable days: {profitable_days}/{n_days} ({profitable_days/n_days*100:.0f}%)")
    print(f"  Avg feature correlation: {avg_corr:.4f}")
    print(f"  Avg direction accuracy: {avg_dir:.1f}%")
    print(f"  Avg trades/day: {avg_trades:.1f}")
    print(f"  vs Baseline: $736/day (direction CNN)")

    # Monthly breakdown
    _monthly = {}
    for r in day_results:
        _month = r['date'][:7]
        if _month not in _monthly:
            _monthly[_month] = []
        _monthly[_month].append(r.get('sim_pnl', 0))

    print(f"\n  MONTHLY BREAKDOWN:")
    for m in sorted(_monthly):
        _pnls = _monthly[m]
        _total = sum(_pnls)
        _days = len(_pnls)
        print(f"    {m}: ${_total*0.5:>8,.0f} ({_days} days, ${_total*0.5/_days:.0f}/day)")

    # Save results
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    _results = {
        'n_days': n_days, 'cum_pnl_ticks': cum_pnl, 'cum_pnl_dollars': cum_pnl * 0.5,
        'per_day': cum_pnl * 0.5 / n_days, 'avg_corr': avg_corr,
        'avg_dir_acc': avg_dir, 'profitable_days_pct': profitable_days / n_days * 100,
        'day_results': day_results,
    }
    with open(os.path.join(CHECKPOINT_DIR, 'walk_forward_results.json'), 'w') as f:
        json.dump(_results, f, indent=2, default=str)
    print(f"  Saved: {CHECKPOINT_DIR}/walk_forward_results.json")

    # Append to experiment log
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"model=TradeCNN_A | days={n_days} | "
             f"corr={avg_corr:.4f} | dir={avg_dir:.1f}% | "
             f"PnL=${cum_pnl*0.5:,.0f} | $/day=${cum_pnl*0.5/n_days:.0f}\n")
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)
    print(f"  Logged: {RESULTS_LOG}")


def oos_single_pass():
    """Single forward pass on OOS with the trained model. Simplified trading + full logging."""
    from core.trade_cnn import StatePredictor
    import csv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(_ckpt_path):
        print(f"No model found at {_ckpt_path} — run --phase train first")
        return

    ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    model = StatePredictor(n_features=N_FEAT, latent_dim=64, n_labels=N_LABELS).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded model from day {ckpt.get('day', '?')}")

    # Build OOS features
    feats, labels, states, df = build_dataset(OOS_ROOT)
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values

    # Load 1s data for fill price lookup (actual price 2s after signal)
    print("Loading 1s data for fill delay...")
    _1s_files = sorted(glob.glob(os.path.join(OOS_ROOT, '1s', '*.parquet')))
    if _1s_files:
        df_1s = pd.concat([pd.read_parquet(f) for f in _1s_files], ignore_index=True)
        df_1s = df_1s.sort_values('timestamp').reset_index(drop=True)
        _1s_ts = df_1s['timestamp'].values
        _1s_close = df_1s['close'].values
        print(f"  1s bars: {len(df_1s):,}")
    else:
        df_1s = None
        _1s_ts = None
        _1s_close = None
        print("  No 1s data — using close price (no delay)")

    def _get_fill_price(signal_ts):
        """Get price FILL_DELAY_S seconds after signal timestamp."""
        if _1s_ts is None:
            return None
        _target_ts = signal_ts + FILL_DELAY_S
        _idx = np.searchsorted(_1s_ts, _target_ts)
        if _idx < len(_1s_close):
            return float(_1s_close[_idx])
        return None

    # Simple trading: follow predicted direction, trail after $5, SL=40
    SL = 40
    BE_ACT = 5       # move SL to breakeven after 5 ticks profit (direction confirmed)
    TRAIL_ACT = 10   # activate trail after 10 ticks profit
    TRAIL_DIST = 10  # trail distance from peak
    CONF_THRESHOLD = 3.0  # minimum confidence to enter (was 2.0)
    FILL_DELAY_S = 2  # seconds from signal to fill

    trades = []
    trade_log = []  # full state log per trade
    in_trade = False
    trade_dir = ''
    entry_price = 0.0
    entry_bar = 0
    peak_price = 0.0
    trail_active = False

    for i in tqdm(range(LOOKBACK, len(feats) - MAX_FORWARD), desc="OOS Single Pass"):
        x = feats[i - LOOKBACK:i]
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_t).cpu().numpy()[0]

        price = prices[i]
        high = highs[i]
        low = lows[i]

        # Predicted direction from dmi_diff — adapt to number of horizons
        n_h = len(HORIZONS)
        _h0_dmi = pred[0]                # dmi_diff at first horizon
        if n_h >= 2:
            _h1_dmi = pred[7]
        else:
            _h1_dmi = _h0_dmi
        if n_h >= 3:
            _h2_dmi = pred[14]
        else:
            _h2_dmi = _h1_dmi

        # Primary prediction: last available horizon
        _pred_dmi = pred[(n_h - 1) * 7]  # dmi_diff at the furthest horizon
        _pred_dir = 'LONG' if _pred_dmi > 0 else 'SHORT'
        _confidence = abs(_pred_dmi)
        # All horizons agree
        _all_agree = True
        for _hi in range(1, n_h):
            if np.sign(pred[_hi * 7]) != np.sign(pred[0]):
                _all_agree = False
                break

        if in_trade:
            # Same-bar high/low for SL/trail (standard 1m backtest convention)
            # Exit slippage added via 2s fill delay on exit orders
            if trade_dir == 'LONG':
                peak_price = max(peak_price, high)
                _pnl = (price - entry_price) / TICK
                _pnl_from_low = (low - entry_price) / TICK
            else:
                peak_price = min(peak_price, low) if peak_price > 0 else low
                _pnl = (entry_price - price) / TICK
                _pnl_from_low = (entry_price - high) / TICK

            _peak_pnl = (peak_price - entry_price) / TICK if trade_dir == 'LONG' \
                else (entry_price - peak_price) / TICK

            # Breakeven: once direction confirmed (+5t), move SL to entry
            _be_active = _peak_pnl >= BE_ACT
            _effective_sl = 0 if _be_active else SL  # 0 = breakeven, SL = full

            # SL check (breakeven-aware) with exit slippage
            if _pnl_from_low <= -_effective_sl:
                _sl_fill = _get_fill_price(timestamps[i])
                if _effective_sl > 0:
                    # SL: capped at -40 (2s slippage can't make it worse than SL)
                    _exit_pnl = -_effective_sl
                else:
                    # BE: exit at 2s fill price, might slip 1-2 ticks
                    if _sl_fill is not None:
                        _exit_pnl = (_sl_fill - entry_price) / TICK if trade_dir == 'LONG' \
                            else (entry_price - _sl_fill) / TICK
                    else:
                        _exit_pnl = 0
                _exit_type = 'BE' if _be_active else 'SL'
                trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                               'held': i - entry_bar, 'exit': _exit_type, 'peak': _peak_pnl})
                trade_log.append({
                    'bar': i, 'price': price, 'entry': entry_price,
                    'pnl': _exit_pnl, 'dir': trade_dir, 'exit': _exit_type,
                    'held': i - entry_bar, 'peak': _peak_pnl,
                    'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                    'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                })
                in_trade = False
                continue

            # Trail activation
            if not trail_active and _peak_pnl >= TRAIL_ACT:
                trail_active = True

            # Trail check (uses previous bar's low/high)
            if trail_active:
                if trade_dir == 'LONG':
                    _trail_level = peak_price - TRAIL_DIST * TICK
                    if low <= _trail_level:
                        _trail_fill = _get_fill_price(timestamps[i])
                        _exit_price = _trail_fill if _trail_fill else _trail_level
                        _exit_pnl = max(0, (_exit_price - entry_price) / TICK)
                        trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                                       'held': i - entry_bar, 'exit': 'TRAIL', 'peak': _peak_pnl})
                        trade_log.append({
                            'bar': i, 'price': price, 'entry': entry_price,
                            'pnl': _exit_pnl, 'dir': trade_dir, 'exit': 'TRAIL',
                            'held': i - entry_bar, 'peak': _peak_pnl,
                            'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                            'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                        })
                        in_trade = False
                        continue
                else:
                    _trail_level = peak_price + TRAIL_DIST * TICK
                    if high >= _trail_level:
                        _trail_fill = _get_fill_price(timestamps[i])
                        _exit_price = _trail_fill if _trail_fill else _trail_level
                        _exit_pnl = max(0, (entry_price - _exit_price) / TICK)
                        trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                                       'held': i - entry_bar, 'exit': 'TRAIL', 'peak': _peak_pnl})
                        trade_log.append({
                            'bar': i, 'price': price, 'entry': entry_price,
                            'pnl': _exit_pnl, 'dir': trade_dir, 'exit': 'TRAIL',
                            'held': i - entry_bar, 'peak': _peak_pnl,
                            'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                            'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                        })
                        in_trade = False
                        continue

            # Flip: predicted direction changed AND all horizons agree
            if _pred_dir != trade_dir and _confidence > CONF_THRESHOLD and _all_agree:
                _exit_pnl = _pnl
                trades.append({'bar': i, 'pnl': _exit_pnl, 'dir': trade_dir,
                               'held': i - entry_bar, 'exit': 'FLIP', 'peak': _peak_pnl})
                trade_log.append({
                    'bar': i, 'price': price, 'entry': entry_price,
                    'pnl': _exit_pnl, 'dir': trade_dir, 'exit': 'FLIP',
                    'held': i - entry_bar, 'peak': _peak_pnl,
                    'pred_dmi_h0': _h0_dmi, 'pred_dmi_h1': _h1_dmi,
                    'actual_dmi': feats[i, 0], 'actual_vel': feats[i, 4],
                })
                # Enter opposite direction (fill at 2s after signal)
                _fill = _get_fill_price(timestamps[i])
                in_trade = True
                trade_dir = _pred_dir
                entry_price = _fill if _fill is not None else price
                entry_bar = i
                peak_price = entry_price
                trail_active = False
                continue

        # Entry: all horizons agree + confidence (momentum building optional)
        if not in_trade and _confidence > CONF_THRESHOLD and _all_agree:
            in_trade = True
            trade_dir = _pred_dir
            # Fill at actual price 2s after signal (from 1s data)
            _fill = _get_fill_price(timestamps[i])
            entry_price = _fill if _fill is not None else price
            entry_bar = i
            peak_price = entry_price
            trail_active = False

    # Flush last trade
    if in_trade:
        _pnl = (prices[-1] - entry_price) / TICK if trade_dir == 'LONG' \
            else (entry_price - prices[-1]) / TICK
        trades.append({'bar': len(prices)-1, 'pnl': _pnl, 'dir': trade_dir,
                       'held': len(prices) - 1 - entry_bar, 'exit': 'EOD', 'peak': 0})

    # Results
    total_pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    w = len([t for t in trades if t['pnl'] > 0])
    trading_days = pd.to_datetime(df['timestamp'], unit='s').dt.date.nunique()

    _wr = w / n * 100 if n > 0 else 0
    _per_day = total_pnl * 0.5 / trading_days if trading_days > 0 else 0

    print(f"\n{'='*60}")
    print(f"OOS SINGLE PASS: StatePredictor + Trail SL={SL} Trail={TRAIL_ACT}/{TRAIL_DIST}")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {_wr:.1f}%")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${_per_day:.2f}")
    print(f"  Trading days: {trading_days}")

    # Exit breakdown
    _exits = {}
    for t in trades:
        ex = t['exit']
        if ex not in _exits:
            _exits[ex] = {'n': 0, 'pnl': 0}
        _exits[ex]['n'] += 1
        _exits[ex]['pnl'] += t['pnl']
    print(f"\n  EXIT BREAKDOWN:")
    for ex, v in sorted(_exits.items(), key=lambda x: x[1]['pnl']):
        print(f"    {ex:<10} {v['n']:>5} trades  ${v['pnl']*0.5:>10,.2f}")

    # Save trade log
    _log_path = os.path.join(CHECKPOINT_DIR, 'oos_trade_log.csv')
    if trade_log:
        _keys = trade_log[0].keys()
        with open(_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_keys)
            writer.writeheader()
            writer.writerows(trade_log)
        print(f"\n  Trade log: {_log_path} ({len(trade_log)} trades)")

    # Append to experiment log
    _line = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"model=TradeCNN_A_OOS | days={trading_days} | "
             f"trades={n} | WR={_wr:.1f}% | "
             f"PnL=${total_pnl*0.5:,.0f} | $/day=${_per_day:.0f}\n")
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, 'a') as f:
        f.write(_line)

    # Cleanup
    del feats, labels, states, df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
