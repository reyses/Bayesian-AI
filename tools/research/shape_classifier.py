"""
Shape Classifier — shared waveform classification + quality scoring.

Extracted from seed_pattern_analyzer.py so both the analyzer (post-hoc)
and the builder (pre-filter) use the same classification logic.

Quality scoring is calibrated from human-marked seeds (255 seeds, Jan 5-7 2025).
Use --recalibrate to update priors from an expanding seed library.

Usage:
    from tools.research.shape_classifier import classify_shape, quality_score
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

TICK_SIZE = 0.25

# ═══════════════════════════════════════════════════════════════
# Shape classification
# ═══════════════════════════════════════════════════════════════

def classify_shape(prices: np.ndarray, entry_idx: int) -> Tuple[str, float, dict]:
    """Classify a waveform into shape primitives based on geometry.

    prices: raw close prices (lookback + segment)
    entry_idx: index of entry point within prices array

    Returns: (shape_name, confidence, features_dict)
    """
    if len(prices) < 3 or entry_idx >= len(prices) - 1:
        return 'UNKNOWN', 0.0, {}

    segment = prices[entry_idx:]
    lookback = prices[:entry_idx] if entry_idx > 0 else prices[:1]

    if len(segment) < 2:
        return 'UNKNOWN', 0.0, {}

    # Normalize to ticks from entry
    entry = segment[0]
    seg_ticks = (segment - entry) / TICK_SIZE
    lb_ticks = (lookback - entry) / TICK_SIZE

    net_move = seg_ticks[-1]
    direction = 1 if net_move > 0 else -1

    # Directional moves (favorable = in direction of net move)
    if direction > 0:
        fav = seg_ticks
    else:
        fav = -seg_ticks

    peak_idx = int(np.argmax(fav))
    trough_idx = int(np.argmin(fav))
    peak_val = float(fav[peak_idx])
    trough_val = float(fav[trough_idx])
    abs_net = abs(net_move)

    # Derived metrics
    n = len(segment)
    mid = n // 2

    # Path efficiency: how straight was the move? (net / total path length)
    path_length = float(np.sum(np.abs(np.diff(seg_ticks))))
    efficiency = abs_net / max(path_length, 0.01)

    # Retracement: how much did it pull back from peak?
    if peak_val > 0:
        retracement = float((peak_val - fav[-1]) / peak_val)
    else:
        retracement = 0.0

    # First half vs second half momentum
    first_half = fav[:mid + 1] if mid > 0 else fav[:1]
    second_half = fav[mid:]
    first_move = float(first_half[-1] - first_half[0]) if len(first_half) > 1 else 0
    second_move = float(second_half[-1] - second_half[0]) if len(second_half) > 1 else 0

    # Lookback trend (was there momentum before entry?)
    if len(lb_ticks) > 2:
        lb_trend = float(lb_ticks[-1] - lb_ticks[0])
        lb_slope = lb_trend / len(lb_ticks)
    else:
        lb_trend = 0.0
        lb_slope = 0.0

    # Maximum adverse excursion position (early vs late)
    mae_idx = trough_idx
    mae_early = mae_idx < n * 0.3  # MAE in first 30%
    mae_late = mae_idx > n * 0.7   # MAE in last 30%

    # Peak timing (early vs late)
    peak_early = peak_idx < n * 0.3
    peak_late = peak_idx > n * 0.7

    # Monotonicity: how many bars move in favorable direction?
    diffs = np.diff(fav)
    monotonic_frac = float(np.sum(diffs > 0)) / max(len(diffs), 1)

    features = {
        'net_ticks': float(net_move),
        'abs_net': abs_net,
        'peak_ticks': peak_val,
        'trough_ticks': trough_val,
        'efficiency': efficiency,
        'retracement': retracement,
        'first_half_move': first_move,
        'second_half_move': second_move,
        'lb_trend': lb_trend,
        'lb_slope': lb_slope,
        'peak_at_pct': float(peak_idx) / max(n - 1, 1),
        'mae_at_pct': float(mae_idx) / max(n - 1, 1),
        'monotonic_frac': monotonic_frac,
        'n_bars': n,
    }

    # ── Classification rules ──

    # V-REVERSAL: lookback trends one way, segment sharply reverses
    lb_dir = 1 if lb_slope > 0.5 else (-1 if lb_slope < -0.5 else 0)
    if lb_dir != 0 and lb_dir != direction and abs_net > 30 and efficiency > 0.25:
        return 'V_REVERSAL', min(1.0, abs(lb_slope) * efficiency * 2), features

    # IMPULSE: fast, efficient, monotonic move
    if efficiency > 0.45 and monotonic_frac > 0.6 and abs_net > 20:
        return 'IMPULSE', efficiency * monotonic_frac, features

    # RAMP: steady directional move, good efficiency but less monotonic
    if efficiency > 0.25 and abs_net > 30 and monotonic_frac > 0.45:
        return 'RAMP', efficiency, features

    # FAKEOUT: strong initial move then retracement > 60%
    if peak_early and retracement > 0.6 and peak_val > 15:
        return 'FAKEOUT', retracement, features

    # EXHAUSTION: peak late, then gives back
    if peak_late and retracement > 0.3 and peak_val > 15:
        return 'EXHAUSTION', retracement * (peak_val / max(abs_net, 1)), features

    # SIGMOID: slow start, fast middle, slow end (S-curve)
    if abs(first_move) < abs(second_move) * 0.5 and abs_net > 20:
        return 'SIGMOID', abs(second_move) / max(abs(first_move) + 1, 1), features

    # COMPRESSION: tight range then breakout (low efficiency but net move)
    range_first = float(np.max(fav[:mid+1]) - np.min(fav[:mid+1])) if mid > 0 else 0
    range_second = float(np.max(fav[mid:]) - np.min(fav[mid:])) if mid < n else 0
    if range_first < 15 and range_second > 30:
        return 'COMPRESSION', range_second / max(range_first + 1, 1), features

    # CHOP: low efficiency, low net, no clear pattern
    if efficiency < 0.15 and abs_net < 20:
        return 'CHOP', 1.0 - efficiency, features

    # TREND_CONTINUATION: lookback and segment same direction
    if lb_dir == direction and abs_net > 15:
        return 'TREND_CONTINUATION', efficiency, features

    return 'OTHER', 0.5, features


# ═══════════════════════════════════════════════════════════════
# Quality scoring — calibrated from human seeds
# ═══════════════════════════════════════════════════════════════

# Default priors from 255 human seeds (Jan 5-7 2025, 1h/15m/5m)
# Frequency-weighted: shapes humans mark more often = higher prior
DEFAULT_SHAPE_PRIORS = {
    'V_REVERSAL':         0.70,   # 56% of 15m, 34% of 5m — dominant human pattern
    'IMPULSE':            0.65,   # 30% of 5m, 21% of 15m — fast directional bursts
    'RAMP':               0.50,   # 13% of 5m, 8% of 15m — steady grinds
    'TREND_CONTINUATION': 0.40,   # 10% of 15m — prior move extends
    'SIGMOID':            0.35,   # slow start, fast finish
    'COMPRESSION':        0.25,   # 15% of 5m — coiling, some value but noisy
    'EXHAUSTION':         0.20,   # 1% of 5m — peak then giveback
    'FAKEOUT':            0.15,   # 3% of 15m — rare, hard to trade
    'CHOP':               0.05,   # never marked by human — pure noise
    'OTHER':              0.20,
    'UNKNOWN':            0.00,
}


def quality_score(shape: str, confidence: float, features: dict,
                  priors: Optional[Dict[str, float]] = None) -> float:
    """Score a classified waveform 0.0-1.0 based on human-derived quality metrics.

    Calibrated from 255 human-marked seeds across 1h/15m/5m (Jan 5-7 2025).
    Higher score = more similar to what the human would mark.

    Args:
        shape: shape label from classify_shape()
        confidence: classification confidence
        features: feature dict from classify_shape()
        priors: optional custom shape priors (from recalibration)
    """
    if priors is None:
        priors = DEFAULT_SHAPE_PRIORS

    eff = features.get('efficiency', 0)
    abs_net = features.get('abs_net', 0)
    mono = features.get('monotonic_frac', 0)
    retr = features.get('retracement', 0)
    peak_pct = features.get('peak_at_pct', 0.5)

    # Base score from shape type (human frequency = prior for quality)
    base = priors.get(shape, 0.2)

    # Efficiency bonus (human avg: 0.78 at 15m, 0.38 = chop threshold)
    eff_bonus = max(0.0, min(0.3, (eff - 0.2) * 0.5))

    # Net move bonus (filter out tiny wiggles — human min ~15t at 5m)
    net_bonus = max(0.0, min(0.15, (abs_net - 10) * 0.005))

    # Late MFE bonus (human V_REVs peak in last 30% — 72% efficient)
    late_mfe_bonus = 0.1 if peak_pct > 0.7 and shape == 'V_REVERSAL' else 0.0

    # Monotonicity bonus for directional shapes
    mono_bonus = 0.0
    if shape in ('IMPULSE', 'RAMP') and mono > 0.6:
        mono_bonus = (mono - 0.6) * 0.3

    # Penalty for high retracement (gave back the move)
    retr_penalty = 0.0
    if shape not in ('FAKEOUT', 'EXHAUSTION') and retr > 0.5:
        retr_penalty = (retr - 0.5) * 0.3

    score = base + eff_bonus + net_bonus + late_mfe_bonus + mono_bonus - retr_penalty
    return max(0.0, min(1.0, score))


def quality_tier(score: float) -> str:
    """Map quality score to tier label."""
    if score >= 0.7:
        return 'GOLD'
    elif score >= 0.5:
        return 'SILVER'
    elif score >= 0.3:
        return 'BRONZE'
    else:
        return 'NOISE'


# ═══════════════════════════════════════════════════════════════
# Recalibration from expanding seed library
# ═══════════════════════════════════════════════════════════════

CALIBRATION_PATH = 'checkpoints/shape_quality_calibration.json'


def calibrate_from_human_seeds(seeds_dir: str, data_dir: str,
                               output_path: str = CALIBRATION_PATH) -> Dict[str, float]:
    """Recompute shape priors from ALL available human seeds.

    Loads every human seed file, classifies each, then computes
    frequency-weighted priors. Saves to JSON for future use.

    Returns updated priors dict.
    """
    from collections import defaultdict
    from pathlib import Path
    from tools.research.data import load_atlas_tf

    seed_files = sorted(Path(seeds_dir).glob('seeds_*_multi.json'))
    if not seed_files:
        seed_files = sorted(Path(seeds_dir).glob('seeds_*.json'))
    if not seed_files:
        print("  No human seed files found for calibration.")
        return DEFAULT_SHAPE_PRIORS.copy()

    # Load price data cache
    tf_close_cache = {}
    shape_counts = defaultdict(int)
    total = 0

    for sf in seed_files:
        with open(sf) as f:
            data = json.load(f)

        file_tf = data.get('timeframe', '1m')

        for s in data.get('seeds', []):
            tf = s.get('timeframe', file_tf)

            if tf not in tf_close_cache:
                df = load_atlas_tf(data_dir, tf)
                if df.empty:
                    tf_close_cache[tf] = None
                    continue
                tf_close_cache[tf] = df['close'].values.astype(np.float64)
                del df

            close = tf_close_cache[tf]
            if close is None:
                continue

            si = s.get('regime_start_idx', s.get('start_idx', 0))
            ei = s.get('end_idx', si + 10)
            lb_start = s.get('lookback_start_idx', max(0, si - 10))

            if ei >= len(close) or si >= len(close):
                continue

            waveform = close[lb_start:ei + 1]
            entry_idx = si - lb_start

            shape, conf, feats = classify_shape(waveform, entry_idx)
            shape_counts[shape] += 1
            total += 1

    if total == 0:
        print("  No valid seeds for calibration.")
        return DEFAULT_SHAPE_PRIORS.copy()

    # Frequency -> prior: most-marked shape gets 0.8, least gets 0.1
    # Linear scale between min and max frequency
    freqs = {sh: cnt / total for sh, cnt in shape_counts.items()}
    max_freq = max(freqs.values())
    min_freq = min(freqs.values())
    freq_range = max(max_freq - min_freq, 0.01)

    priors = {}
    for sh in DEFAULT_SHAPE_PRIORS:
        if sh in freqs:
            normalized = (freqs[sh] - min_freq) / freq_range
            priors[sh] = 0.1 + 0.7 * normalized  # Range [0.1, 0.8]
        else:
            priors[sh] = 0.05  # Never marked = near-zero

    # Always keep CHOP/UNKNOWN low
    priors['CHOP'] = min(priors.get('CHOP', 0.05), 0.1)
    priors['UNKNOWN'] = 0.0

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    calibration = {
        'priors': priors,
        'shape_counts': dict(shape_counts),
        'total_seeds': total,
        'seed_files': [str(sf) for sf in seed_files],
    }
    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"  Calibration saved: {output_path} ({total} seeds, {len(shape_counts)} shapes)")

    return priors


def load_calibration(path: str = CALIBRATION_PATH) -> Optional[Dict[str, float]]:
    """Load previously saved calibration priors, or None if not found."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('priors')
