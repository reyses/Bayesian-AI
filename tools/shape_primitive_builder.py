#!/usr/bin/env python
"""
Shape Primitive Builder — UMAP + HDBSCAN waveform clustering.

Pipeline:
  Multi-TF ZigZag → quality filter → normalize + resample to 32pts + magnitude →
  UMAP 34D→2D → HDBSCAN (power-analyzed min_cluster_size) → bootstrap →
  shape primitives with 16D centroids.

Primitives feed into Phase 2 K-Means as informed initial centroids.

Usage:
    python tools/shape_primitive_builder.py                              # full ATLAS
    python tools/shape_primitive_builder.py --data DATA/ATLAS_1WEEK      # fast test
    python tools/shape_primitive_builder.py --plot                        # save UMAP scatter
    python tools/shape_primitive_builder.py --skip-physics                # skip 16D (fast)
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.auto_swing_marker import detect_swings, TICK_SIZE
from tools.research.data import load_atlas_tf, compute_tf_physics, extract_16d, TF_SECONDS
from tools.research.shape_classifier import classify_shape, quality_score, quality_tier, load_calibration

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# 7 active ZigZag TFs (3-to-5 ratio, excluding 2m/3m, 1s/5s too noisy)
SWING_TFS = ['15s', '30s', '1m', '5m', '15m', '30m', '1h']

LOOKBACK_BARS = 10
RESAMPLE_POINTS = 32
SEEDS_DIR = 'DATA/regime_seeds'

UMAP_DEFAULTS = dict(n_neighbors=30, min_dist=0.05, metric='euclidean', random_state=42)
HDBSCAN_DEFAULTS = dict(min_cluster_size=50, min_samples=10, cluster_selection_method='eom')

# Trading economics for power analysis
TICK_VALUE_USD = 0.50    # MNQ $0.50 per tick
CONFIDENCE_Z = 1.645     # z-score for 90% confidence (shape recognition, not MFE prediction)
MAGNITUDE_WEIGHT = 3.0   # Weight for magnitude features in UMAP input


# ═══════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class ShapePrimitive:
    primitive_id: int
    centroid_waveform: np.ndarray      # (32,) normalized shape
    centroid_16d: np.ndarray           # (16,) mean feature vector
    n_members: int
    dominant_tf: str                   # Most common TF among members
    tf_distribution: Dict[str, int]    # {tf: count}
    direction_bias: float              # Fraction LONG (0-1)
    mean_mfe_ticks: float
    mean_mae_ticks: float
    mean_duration_mins: float
    shape_r2: float                    # Tightness (r^2 vs centroid)
    umap_center: Tuple[float, float]   # 2D center for visualization
    member_indices: List[int]          # Into global seed array
    dominant_shape: str = ''           # Most common shape label among members
    shape_distribution: Dict[str, int] = field(default_factory=dict)  # {shape: count}
    mean_quality_score: float = 0.0    # Average quality score of members
    quality_tier_label: str = ''       # GOLD/SILVER/BRONZE based on mean score
    bootstrap_stable: bool = True      # Passed bootstrap stability test
    centroid_drift: float = 0.0        # Bootstrap centroid drift (lower = more stable)


@dataclass
class ShapePrimitiveLibrary:
    primitives: List[ShapePrimitive]
    created_at: str
    n_total_seeds: int
    n_clustered_seeds: int
    n_noise_seeds: int
    umap_params: Dict
    hdbscan_params: Dict
    tf_params: Dict[str, Dict]         # Per-TF ZigZag params
    version: str = '1.0'

    def get_centroids_for_tf(self, tf: str) -> Optional[np.ndarray]:
        """Return (K, 16) centroids for a TF bucket."""
        matching = [p for p in self.primitives
                    if p.dominant_tf == tf or tf in p.tf_distribution]
        return np.array([p.centroid_16d for p in matching]) if matching else None


# ═══════════════════════════════════════════════════════════════
# ZigZag parameter scaling
# ═══════════════════════════════════════════════════════════════

def get_zigzag_params(tf: str) -> dict:
    """Per-TF ZigZag parameters scaled from 1m baseline via sqrt(tf_secs/60)."""
    tf_secs = TF_SECONDS.get(tf, 60)
    sqrt_r = math.sqrt(tf_secs / 60.0)
    min_reversal = max(10, min(250, round(30 * sqrt_r)))

    if tf_secs <= 15:
        min_bars, max_bars = 8, 30
    elif tf_secs <= 30:
        min_bars, max_bars = 6, 20
    elif tf_secs <= 60:
        min_bars, max_bars = 5, 15
    elif tf_secs <= 300:
        min_bars, max_bars = 3, 8
    elif tf_secs <= 900:
        min_bars, max_bars = 2, 6
    elif tf_secs <= 1800:
        min_bars, max_bars = 2, 5
    else:
        min_bars, max_bars = 2, 4

    return {'min_reversal': min_reversal, 'min_bars': min_bars, 'max_bars': max_bars}


# ═══════════════════════════════════════════════════════════════
# Step 1: Multi-TF swing detection
# ═══════════════════════════════════════════════════════════════

def detect_multi_tf_swings(data_dir: str, burn_hours: int = 10) -> list:
    """Run ZigZag on all swing TFs. Returns unified list of seed dicts."""
    all_seeds = []

    # Determine burn cutoff from first timestamp in 1m data
    df_1m = load_atlas_tf(data_dir, '1m')
    if df_1m.empty:
        print("  ERROR: No 1m data for burn calculation")
        return []
    first_ts = float(df_1m['timestamp'].iloc[0])
    burn_until = first_ts + burn_hours * 3600
    print(f"  Burn cutoff: first {burn_hours}h -> ts < {burn_until:.0f}")
    del df_1m

    for tf in SWING_TFS:
        params = get_zigzag_params(tf)
        print(f"\n  [{tf}] Loading... (min_rev={params['min_reversal']}t, "
              f"min_bars={params['min_bars']}, max_bars={params['max_bars']})")

        df = load_atlas_tf(data_dir, tf)
        if df.empty:
            print(f"  [{tf}] No data, skipping")
            continue

        close = df['close'].values.astype(np.float64)
        timestamps = df['timestamp'].values.astype(np.float64)
        print(f"  [{tf}] {len(close):,} bars")

        # Run ZigZag on full continuous series
        pivots = detect_swings(close,
                               min_reversal=params['min_reversal'],
                               min_bars=params['min_bars'],
                               max_bars=params['max_bars'])

        n_segments = max(0, len(pivots) - 1)
        print(f"  [{tf}] {len(pivots)} pivots -> {n_segments} segments")

        tf_seeds = 0
        burned = 0
        short_lb = 0

        for j in range(len(pivots) - 1):
            si = pivots[j]
            ei = pivots[j + 1]
            if ei <= si:
                continue

            ts_start = timestamps[si]
            ts_end = timestamps[ei]

            if ts_start < burn_until:
                burned += 1
                continue

            lb_start = max(0, si - LOOKBACK_BARS)
            actual_lb = si - lb_start

            # Drop seeds with < 3 bars lookback (near data start)
            if actual_lb < 3:
                short_lb += 1
                continue

            direction = 'LONG' if close[ei] > close[si] else 'SHORT'
            change_ticks = (close[ei] - close[si]) / TICK_SIZE

            seed = {
                'tf': tf,
                'start_idx': int(si),
                'end_idx': int(ei),
                'lookback_start_idx': int(lb_start),
                'ts_start': float(ts_start),
                'ts_end': float(ts_end),
                'entry_price': float(close[si]),
                'exit_price': float(close[ei]),
                'direction': direction,
                'change_ticks': float(change_ticks),
                'duration_mins': float((ts_end - ts_start) / 60.0),
                'n_bars': ei - si,
                'lookback_bars': actual_lb,
                'waveform_close': close[lb_start:ei + 1].copy(),
                'source': 'auto_swing',
            }
            all_seeds.append(seed)
            tf_seeds += 1

        print(f"  [{tf}] {tf_seeds:,} seeds (burned={burned}, short_lb={short_lb})")
        del df, close, timestamps

    return all_seeds


# ═══════════════════════════════════════════════════════════════
# Step 1b: Load human seeds
# ═══════════════════════════════════════════════════════════════

def load_human_seeds(seeds_dir: str, data_dir: str, burn_until: float = 0) -> list:
    """Load human-marked seeds from JSON files and extract their waveforms."""
    if not os.path.exists(seeds_dir):
        return []

    # Prefer merged multi-TF files, fallback to individual
    seed_files = sorted(Path(seeds_dir).glob('seeds_*_multi.json'))
    if not seed_files:
        seed_files = sorted(Path(seeds_dir).glob('seeds_*.json'))
    if not seed_files:
        return []

    print(f"\n  Loading human seeds from {len(seed_files)} files...")

    tf_close_cache = {}  # tf -> close array
    human_seeds = []

    for sf in seed_files:
        with open(sf) as f:
            data = json.load(f)

        seeds_list = data.get('seeds', [])
        file_tf = data.get('timeframe', '1m')

        for s in seeds_list:
            tf = s.get('timeframe', file_tf)
            ts_start = s.get('ts_start', 0)

            if ts_start < burn_until:
                continue

            # Cache close prices per TF
            if tf not in tf_close_cache:
                df = load_atlas_tf(data_dir, tf)
                if df.empty:
                    tf_close_cache[tf] = None
                    continue
                tf_close_cache[tf] = (
                    df['close'].values.astype(np.float64),
                    df['timestamp'].values.astype(np.float64),
                )
                del df

            cached = tf_close_cache[tf]
            if cached is None:
                continue
            close, timestamps = cached

            si = s.get('regime_start_idx', s.get('start_idx', 0))
            ei = s.get('end_idx', si + s.get('n_bars', 10))

            if ei >= len(close) or si >= len(close):
                continue

            lb_start = s.get('lookback_start_idx', max(0, si - LOOKBACK_BARS))
            actual_lb = si - lb_start
            if actual_lb < 3:
                continue

            seed = {
                'tf': tf,
                'start_idx': int(si),
                'end_idx': int(ei),
                'lookback_start_idx': int(lb_start),
                'ts_start': float(ts_start),
                'ts_end': float(s.get('ts_end', ts_start + 60)),
                'entry_price': float(close[si]),
                'exit_price': float(close[ei]),
                'direction': s.get('direction', 'LONG'),
                'change_ticks': float(s.get('change_ticks', 0)),
                'duration_mins': float(s.get('duration_mins', 0)),
                'n_bars': ei - si,
                'lookback_bars': actual_lb,
                'waveform_close': close[lb_start:ei + 1].copy(),
                'source': 'human',
            }
            human_seeds.append(seed)

    print(f"  Loaded {len(human_seeds)} human seeds")
    return human_seeds


# ═══════════════════════════════════════════════════════════════
# Step 2: Waveform normalization
# ═══════════════════════════════════════════════════════════════

def normalize_waveforms(seeds: list, magnitude_weight: float = MAGNITUDE_WEIGHT) -> Tuple[np.ndarray, list]:
    """Normalize and resample segment waveforms to fixed-length vectors with magnitude.

    Clusters the SEGMENT (lookback + swing). In ZigZag, every lookback IS the
    previous segment, so segment shapes = lookback shapes = same vocabulary.
    Matching a completed swing to a primitive tells you both "what just happened"
    AND "the setup context for what comes next."

    Shape (32D): zero at entry price, divide by max(|range|), resample to 32pts.
    Magnitude (2D): log1p of segment range and net change in ticks, weighted.
    Direction-normalized: SHORT flipped so all shapes face favorable direction.

    Output: (N, 34) array — 32 shape + 2 magnitude dimensions.
    Left-pad with first available price if lookback < LOOKBACK_BARS.

    Returns: (N, 34) array, list of bool (valid mask).
    """
    waveforms = []
    valid_mask = []
    n_dims = RESAMPLE_POINTS + 2  # 32 shape + 2 magnitude

    for seed in tqdm(seeds, desc='Normalizing waveforms', unit='seed',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        raw = seed['waveform_close']
        lb = seed['lookback_bars']

        # Left-pad if lookback < LOOKBACK_BARS (flat extension)
        if lb < LOOKBACK_BARS:
            pad_n = LOOKBACK_BARS - lb
            raw = np.concatenate([np.full(pad_n, raw[0]), raw])
            lb = LOOKBACK_BARS

        entry_price = raw[lb]  # entry is at lookback boundary

        # Full waveform (lookback + segment)
        shifted = raw - entry_price
        max_range = np.max(np.abs(shifted))

        if max_range < 1e-10:
            valid_mask.append(False)
            waveforms.append(np.zeros(n_dims, dtype=np.float32))
            continue

        normalized = shifted / max_range  # [-1, 1]

        # Direction-normalize: flip SHORT so all shapes face "favorable" direction
        # This makes UMAP cluster by SHAPE, not LONG vs SHORT
        if seed.get('direction') == 'SHORT':
            normalized = -normalized

        # Resample to 32 points
        x_orig = np.linspace(0, 1, len(normalized))
        x_new = np.linspace(0, 1, RESAMPLE_POINTS)
        resampled = np.interp(x_new, x_orig, normalized)

        # Magnitude features: distinguish big moves from small moves
        # log scale keeps outliers from dominating; /6 normalizes to ~[0, 1]
        abs_change_ticks = abs(seed.get('change_ticks', 0))
        max_range_ticks = max_range / TICK_SIZE
        mag_change = np.log1p(abs_change_ticks) / 6.0 * magnitude_weight
        mag_range = np.log1p(max_range_ticks) / 6.0 * magnitude_weight

        vec = np.concatenate([resampled, [mag_change, mag_range]])
        waveforms.append(vec.astype(np.float32))
        valid_mask.append(True)

    return np.array(waveforms, dtype=np.float32), valid_mask


# ═══════════════════════════════════════════════════════════════
# Step 3: UMAP embedding
# ═══════════════════════════════════════════════════════════════

def embed_umap(waveforms: np.ndarray, params: dict) -> np.ndarray:
    """UMAP: 32D waveforms -> 2D embedding."""
    import umap

    print(f"\n  UMAP: {waveforms.shape[0]:,} waveforms, {waveforms.shape[1]}D "
          f"(32 shape + {waveforms.shape[1]-32} magnitude) -> 2D")
    print(f"  Params: {params}")

    reducer = umap.UMAP(n_components=2, **params)
    embedding = reducer.fit_transform(waveforms)
    print(f"  Done: {embedding.shape}")
    return embedding


# ═══════════════════════════════════════════════════════════════
# Step 3b: Power analysis — compute min_cluster_size from economics
# ═══════════════════════════════════════════════════════════════

def compute_min_cluster_size(seeds: list, relative_tolerance: float = 0.50,
                             confidence_z: float = CONFIDENCE_Z) -> Tuple[int, dict]:
    """Compute minimum cluster size via power analysis, per-TF.

    Uses RELATIVE tolerance: each primitive's mean MFE must be within
    ±relative_tolerance of the true mean at 90% confidence.

    The primitives are for SHAPE RECOGNITION, not MFE prediction.
    The exit engine (giveback, envelope, SL) handles risk management.
    So tolerance is generous — we just need to distinguish "good shape"
    from "noise shape", not predict exact tick outcomes.

    Default 50% = "this primitive's mean MFE is accurate within ±50%"
    → n = (z / relative_tolerance)² = (1.645 / 0.50)² ≈ 11 members

    Per-TF: computed from each TF's coefficient of variation (CV = σ/μ).
    TFs with higher CV need more members; low CV need fewer.

    Returns: (global_min_cluster_size, {tf: stats})
    """
    # Group seeds by TF
    tf_groups = defaultdict(list)
    for s in seeds:
        tf_groups[s['tf']].append(abs(s['change_ticks']))

    # Base n from relative tolerance (CV-independent)
    # n = (z / relative_tolerance)² when CV=1; scale by actual CV²
    base_n = (confidence_z / relative_tolerance) ** 2

    print(f"\n  Power analysis ({confidence_z:.3f}z / 90% CI, {relative_tolerance*100:.0f}% relative tolerance):")
    print(f"    Base n (CV=1): {base_n:.0f}")
    print(f"    Primitives identify SHAPE — exits handle risk")
    print(f"    {'TF':>5s}  {'Seeds':>7s}  {'u_mfe':>7s}  {'s_mfe':>7s}  {'CV':>6s}  {'n_req':>6s}")
    print(f"    {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")

    tf_stats = {}
    all_n_required = []

    for tf in sorted(tf_groups.keys()):
        mfe_vals = np.array(tf_groups[tf])
        mu = float(np.mean(mfe_vals)) if len(mfe_vals) > 0 else 50.0
        sigma = float(np.std(mfe_vals)) if len(mfe_vals) > 2 else 50.0
        if mu < 1.0:
            mu = 1.0

        cv = sigma / mu  # Coefficient of variation

        # n = (z * CV / relative_tolerance)² = base_n * CV²
        n_req = int(np.ceil(base_n * cv ** 2))
        n_req = max(15, min(500, n_req))

        tf_stats[tf] = {
            'n_required': n_req,
            'mean_mfe': mu,
            'sigma_mfe': sigma,
            'cv': cv,
            'n_seeds': len(mfe_vals),
        }

        all_n_required.append(n_req)

        print(f"    {tf:>5s}  {len(mfe_vals):7,d}  {mu:6.1f}t  {sigma:6.1f}t  {cv:6.2f}  {n_req:6d}")

    # Use median n_required (robust to outlier TFs), capped for small datasets
    median_n = int(np.median(all_n_required)) if all_n_required else 30
    global_min = max(15, min(median_n, len(seeds) // 20))

    print(f"\n    HDBSCAN min_cluster_size = {global_min} "
          f"(median across TFs, capped at N/20={len(seeds)//20})")

    return global_min, tf_stats


# ═══════════════════════════════════════════════════════════════
# Step 4: HDBSCAN clustering
# ═══════════════════════════════════════════════════════════════

def cluster_hdbscan(embedding: np.ndarray, params: dict) -> np.ndarray:
    """HDBSCAN clustering in 2D UMAP space."""
    import hdbscan

    print(f"\n  HDBSCAN: {embedding.shape[0]:,} points")
    print(f"  Params: {params}")

    clusterer = hdbscan.HDBSCAN(**params)
    labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"  Found {n_clusters} clusters, {n_noise:,} noise ({n_noise / len(labels) * 100:.1f}%)")

    return labels


# ═══════════════════════════════════════════════════════════════
# Step 4b: Bootstrap stability test
# ═══════════════════════════════════════════════════════════════

def bootstrap_stability_test(waveforms: np.ndarray, labels: np.ndarray,
                             n_bootstrap: int = 100, max_drift_frac: float = 0.15,
                             rng_seed: int = 42) -> dict:
    """Test each cluster's centroid stability via bootstrap resampling.

    For each cluster: resample members with replacement N times,
    recompute centroid each time, measure drift (RMSE vs full centroid).
    Reject clusters where drift > max_drift_frac of centroid range.

    Returns: {cluster_id: {'stable': bool, 'drift': float, 'n_members': int}}
    """
    rng = np.random.RandomState(rng_seed)
    cluster_ids = sorted(set(labels) - {-1})
    results = {}

    for cid in tqdm(cluster_ids, desc='Bootstrap stability',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        member_idx = np.where(labels == cid)[0]
        member_waves = waveforms[member_idx]
        n_members = len(member_idx)

        if n_members < 5:
            results[cid] = {'stable': False, 'drift': float('inf'), 'n_members': n_members}
            continue

        full_centroid = np.mean(member_waves, axis=0)
        centroid_range = float(np.max(full_centroid) - np.min(full_centroid))
        if centroid_range < 1e-6:
            centroid_range = 1.0

        drifts = []
        for _ in range(n_bootstrap):
            boot_idx = rng.choice(n_members, size=n_members, replace=True)
            boot_centroid = np.mean(member_waves[boot_idx], axis=0)
            rmse = float(np.sqrt(np.mean((boot_centroid - full_centroid) ** 2)))
            drifts.append(rmse)

        mean_drift = float(np.mean(drifts))
        drift_frac = mean_drift / centroid_range
        stable = drift_frac < max_drift_frac

        results[cid] = {
            'stable': stable,
            'drift': mean_drift,
            'drift_frac': drift_frac,
            'n_members': n_members,
        }

    n_stable = sum(1 for r in results.values() if r['stable'])
    n_total = len(results)
    print(f"\n  Bootstrap: {n_stable}/{n_total} clusters stable "
          f"(drift < {max_drift_frac*100:.0f}% of centroid range)")

    return results


# ═══════════════════════════════════════════════════════════════
# Step 5: 16D feature centroids (physics-based)
# ═══════════════════════════════════════════════════════════════

def compute_16d_centroids(seeds: list, labels: np.ndarray, data_dir: str) -> dict:
    """Compute 16D feature centroid per cluster.

    Groups seeds by (tf, month), runs physics once per group,
    then extracts 16D at each seed's entry timestamp.
    """
    # Group seed indices by (tf, month)
    tf_month_groups = defaultdict(list)
    for i, seed in enumerate(seeds):
        if labels[i] == -1:
            continue
        month = datetime.fromtimestamp(seed['ts_start'], tz=timezone.utc).strftime('%Y_%m')
        tf_month_groups[(seed['tf'], month)].append(i)

    n_groups = len(tf_month_groups)
    print(f"\n  Computing 16D across {n_groups} (tf, month) groups...")

    seed_features = {}  # seed_index -> 16D array

    for (tf, month), indices in tqdm(tf_month_groups.items(), desc='Physics',
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        df = load_atlas_tf(data_dir, tf, months=[month])
        if df.empty:
            continue

        states = compute_tf_physics(tf, df)
        if not states:
            del df
            continue

        ts_array = np.array(sorted(states.keys()))

        for idx in indices:
            ts = int(seeds[idx]['ts_start'])
            pos = np.searchsorted(ts_array, ts)
            if pos >= len(ts_array):
                pos = len(ts_array) - 1
            nearest = ts_array[pos]
            if pos > 0 and abs(ts_array[pos - 1] - ts) < abs(nearest - ts):
                nearest = ts_array[pos - 1]

            state = states.get(int(nearest))
            if state is not None:
                seed_features[idx] = np.array(extract_16d(state, tf), dtype=np.float32)

        del df, states

    # Per-cluster mean
    cluster_ids = set(labels) - {-1}
    centroids = {}

    for cid in sorted(cluster_ids):
        member_idx = np.where(labels == cid)[0]
        feats = [seed_features[i] for i in member_idx if i in seed_features]
        centroids[cid] = np.mean(feats, axis=0) if feats else np.zeros(16, dtype=np.float32)

    n_with_feat = len(seed_features)
    n_clustered = int(np.sum(labels != -1))
    print(f"  16D centroids for {len(centroids)} clusters "
          f"({n_with_feat}/{n_clustered} seeds with features)")

    return centroids


# ═══════════════════════════════════════════════════════════════
# Step 6: Build primitives
# ═══════════════════════════════════════════════════════════════

def build_primitives(seeds, waveforms, labels, embedding, centroids_16d) -> list:
    """Assemble ShapePrimitive objects from clustering results."""
    cluster_ids = sorted(set(labels) - {-1})
    primitives = []

    for cid in cluster_ids:
        member_idx = np.where(labels == cid)[0]
        member_seeds = [seeds[i] for i in member_idx]
        member_waves = waveforms[member_idx]
        member_embed = embedding[member_idx]

        # Centroid and R2 computed on shape dims only (first 32), not magnitude
        shape_waves = member_waves[:, :RESAMPLE_POINTS]
        centroid_wave = np.mean(shape_waves, axis=0)

        # Shape R^2
        if len(shape_waves) > 1:
            ss_res = np.sum((shape_waves - centroid_wave) ** 2)
            ss_tot = np.sum((shape_waves - np.mean(shape_waves)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        else:
            r2 = 1.0

        tf_counts = Counter(s['tf'] for s in member_seeds)
        dominant_tf = tf_counts.most_common(1)[0][0]

        n_long = sum(1 for s in member_seeds if s['direction'] == 'LONG')
        direction_bias = n_long / len(member_seeds)

        mfes = [abs(s['change_ticks']) for s in member_seeds]
        durations = [s['duration_mins'] for s in member_seeds]

        # Shape distribution from quality filter (if run)
        sh_counts = Counter(s.get('shape', 'UNKNOWN') for s in member_seeds)
        dom_shape = sh_counts.most_common(1)[0][0] if sh_counts else 'UNKNOWN'
        q_scores = [s.get('quality_score', 0.0) for s in member_seeds]
        mean_q = float(np.mean(q_scores)) if q_scores else 0.0

        prim = ShapePrimitive(
            primitive_id=cid,
            centroid_waveform=centroid_wave.astype(np.float32),
            centroid_16d=centroids_16d.get(cid, np.zeros(16, dtype=np.float32)),
            n_members=len(member_idx),
            dominant_tf=dominant_tf,
            tf_distribution=dict(tf_counts),
            direction_bias=direction_bias,
            mean_mfe_ticks=float(np.mean(mfes)),
            mean_mae_ticks=0.0,  # Not available from ZigZag alone
            mean_duration_mins=float(np.mean(durations)),
            shape_r2=float(r2),
            umap_center=(float(np.mean(member_embed[:, 0])),
                         float(np.mean(member_embed[:, 1]))),
            member_indices=member_idx.tolist(),
            dominant_shape=dom_shape,
            shape_distribution=dict(sh_counts),
            mean_quality_score=mean_q,
            quality_tier_label=quality_tier(mean_q),
        )
        primitives.append(prim)

    return primitives


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def save_umap_plot(embedding, labels, seeds, output_path):
    """Save UMAP scatter plot colored by cluster, with TF marker shapes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Left: colored by cluster
    ax = axes[0]
    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    if noise_mask.any():
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c='lightgray', s=1, alpha=0.2, label='noise')
    if cluster_mask.any():
        sc = ax.scatter(embedding[cluster_mask, 0], embedding[cluster_mask, 1],
                        c=labels[cluster_mask], cmap='tab20', s=3, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Cluster ID')

    n_clusters = len(set(labels) - {-1})
    ax.set_title(f'By Cluster ({n_clusters} clusters, {noise_mask.sum():,} noise)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    # Right: colored by TF
    ax = axes[1]
    tf_list = sorted(set(s['tf'] for s in seeds))
    tf_to_int = {tf: i for i, tf in enumerate(tf_list)}
    tf_colors = np.array([tf_to_int[s['tf']] for s in seeds])

    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=tf_colors, cmap='viridis', s=3, alpha=0.4)
    cbar = plt.colorbar(sc, ax=ax, label='Timeframe')
    cbar.set_ticks(range(len(tf_list)))
    cbar.set_ticklabels(tf_list)

    ax.set_title('By Timeframe')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.suptitle('Shape Primitive Builder — UMAP Embedding', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Shape Primitive Builder')
    parser.add_argument('--data', default='DATA/ATLAS', help='ATLAS data directory')
    parser.add_argument('--output', default='checkpoints/shape_primitives.pkl',
                        help='Output pickle path')
    parser.add_argument('--burn-hours', type=int, default=10,
                        help='Skip first N hours of dataset (regression warmup)')
    parser.add_argument('--seeds-dir', default=SEEDS_DIR,
                        help='Human seeds directory')
    parser.add_argument('--plot', action='store_true', help='Save UMAP scatter plot')
    parser.add_argument('--skip-physics', action='store_true',
                        help='Skip 16D computation (fast test)')
    parser.add_argument('--quality-threshold', type=float, default=0.3,
                        help='Minimum quality score to keep (default 0.3, 0=no filter)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable quality filter (cluster everything)')
    parser.add_argument('--recalibrate', action='store_true',
                        help='Recalibrate quality priors from human seeds before filtering')
    parser.add_argument('--relative-tolerance', type=float, default=0.50,
                        help='Relative MFE tolerance as fraction (default 0.50 = ±50%%)')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Confidence z-score for power analysis (default 1.645 = 90%%)')
    parser.add_argument('--magnitude-weight', type=float, default=MAGNITUDE_WEIGHT,
                        help='Weight for magnitude features in UMAP (default 3.0, 0=shape only)')
    parser.add_argument('--no-bootstrap', action='store_true',
                        help='Skip bootstrap stability test')
    parser.add_argument('--bootstrap-n', type=int, default=100,
                        help='Bootstrap resampling iterations (default 100)')
    parser.add_argument('--umap-neighbors', type=int, default=None)
    parser.add_argument('--umap-min-dist', type=float, default=None)
    parser.add_argument('--hdbscan-min-cluster', type=int, default=None)
    parser.add_argument('--hdbscan-min-samples', type=int, default=None)
    args = parser.parse_args()

    print('=' * 60)
    print('  SHAPE PRIMITIVE BUILDER')
    print('=' * 60)

    umap_params = UMAP_DEFAULTS.copy()
    if args.umap_neighbors:
        umap_params['n_neighbors'] = args.umap_neighbors
    if args.umap_min_dist is not None:
        umap_params['min_dist'] = args.umap_min_dist

    hdbscan_params = HDBSCAN_DEFAULTS.copy()
    # NOTE: min_cluster_size may be overridden by power analysis below
    if args.hdbscan_min_cluster:
        hdbscan_params['min_cluster_size'] = args.hdbscan_min_cluster
    if args.hdbscan_min_samples:
        hdbscan_params['min_samples'] = args.hdbscan_min_samples

    # ── Step 1: Multi-TF swing detection ──────────────────────
    print(f"\n{'=' * 60}")
    print('  STEP 1: Multi-TF Swing Detection')
    print(f"{'=' * 60}")
    seeds = detect_multi_tf_swings(args.data, burn_hours=args.burn_hours)

    # Step 1b: Human seeds
    if os.path.exists(args.seeds_dir):
        df_1m = load_atlas_tf(args.data, '1m')
        burn_until = float(df_1m['timestamp'].iloc[0]) + args.burn_hours * 3600
        del df_1m
        human = load_human_seeds(args.seeds_dir, args.data, burn_until=burn_until)
        seeds.extend(human)

    print(f"\n  Total seeds: {len(seeds):,}")
    if not seeds:
        print('  No seeds found. Exiting.')
        return

    tf_counts = Counter(s['tf'] for s in seeds)
    src_counts = Counter(s['source'] for s in seeds)
    print(f"  By TF:     {dict(sorted(tf_counts.items()))}")
    print(f"  By source: {dict(src_counts)}")

    # ── Step 1c: Shape quality filter ─────────────────────────
    if not args.no_filter and args.quality_threshold > 0:
        print(f"\n{'=' * 60}")
        print('  STEP 1c: Shape Quality Filter')
        print(f"{'=' * 60}")

        # Load calibration if available, or recalibrate
        q_priors = None
        if args.recalibrate:
            from tools.research.shape_classifier import calibrate_from_human_seeds
            q_priors = calibrate_from_human_seeds(args.seeds_dir, args.data)
        else:
            q_priors = load_calibration()

        if q_priors:
            print(f"  Using calibrated priors ({len(q_priors)} shapes)")
        else:
            print(f"  Using default priors (255 human seeds, Jan 5-7 2025)")

        print(f"  Classifying {len(seeds):,} raw swings (threshold={args.quality_threshold})...")

        filtered_seeds = []
        shape_counts = defaultdict(int)
        tier_counts = {'GOLD': 0, 'SILVER': 0, 'BRONZE': 0, 'NOISE': 0}

        for seed in tqdm(seeds, desc='Classifying',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            waveform = seed['waveform_close']
            entry_idx = seed['lookback_bars']
            shape, conf, features = classify_shape(waveform, entry_idx)

            score = quality_score(shape, conf, features, priors=q_priors)

            # Human seeds always pass as GOLD
            if seed.get('source') == 'human':
                score = max(score, 0.7)

            seed['shape'] = shape
            seed['shape_confidence'] = conf
            seed['quality_score'] = score

            shape_counts[shape] += 1
            tier = quality_tier(score)
            tier_counts[tier] += 1

            if score >= args.quality_threshold:
                filtered_seeds.append(seed)

        print(f"\n  Shape distribution:")
        for sh, cnt in sorted(shape_counts.items(), key=lambda x: -x[1]):
            print(f"    {sh:22s}: {cnt:6,d} ({cnt/len(seeds)*100:5.1f}%)")
        print(f"\n  Quality tiers:")
        for tier in ['GOLD', 'SILVER', 'BRONZE', 'NOISE']:
            cnt = tier_counts[tier]
            print(f"    {tier:8s}: {cnt:6,d} ({cnt/len(seeds)*100:5.1f}%)")
        print(f"\n  Kept: {len(filtered_seeds):,} / {len(seeds):,} "
              f"({len(filtered_seeds)/max(len(seeds),1)*100:.1f}%)")

        seeds = filtered_seeds

    # ═══════════════════════════════════════════════════════════
    # PER-TF CLUSTERING LOOP
    # Each TF gets its own UMAP + HDBSCAN so shapes aren't drowned
    # by higher-frequency TFs with more seeds.
    # ═══════════════════════════════════════════════════════════

    conf_z = args.confidence if args.confidence else CONFIDENCE_Z
    all_primitives = []
    total_clustered = 0
    total_noise = 0
    next_prim_id = 0

    # Group seeds by TF
    tf_seed_groups = defaultdict(list)
    for s in seeds:
        tf_seed_groups[s['tf']].append(s)

    active_tfs = sorted(tf_seed_groups.keys(),
                        key=lambda t: TF_SECONDS.get(t, 0))

    for tf in active_tfs:
        tf_seeds = tf_seed_groups[tf]
        print(f"\n{'=' * 60}")
        print(f"  TF: {tf} ({len(tf_seeds):,} seeds)")
        print(f"{'=' * 60}")

        if len(tf_seeds) < 30:
            print(f"  Too few seeds, skipping")
            continue

        # ── Power analysis for this TF ─────────────────────────
        if not args.hdbscan_min_cluster:
            tf_mfe = np.array([abs(s['change_ticks']) for s in tf_seeds])
            mu = float(np.mean(tf_mfe))
            sigma = float(np.std(tf_mfe))
            cv = sigma / max(mu, 1.0)
            base_n = (conf_z / args.relative_tolerance) ** 2
            n_req = max(15, min(500, int(np.ceil(base_n * cv ** 2))))
            tf_min_cluster = max(15, min(n_req, len(tf_seeds) // 20))
            tf_hdbscan = hdbscan_params.copy()
            tf_hdbscan['min_cluster_size'] = tf_min_cluster
            tf_hdbscan['min_samples'] = max(5, tf_min_cluster // 5)
            print(f"  Power: CV={cv:.2f}, n_req={n_req}, min_cluster={tf_min_cluster}")
        else:
            tf_hdbscan = hdbscan_params.copy()

        # ── Normalize ──────────────────────────────────────────
        waveforms, valid_mask = normalize_waveforms(tf_seeds,
                                                     magnitude_weight=args.magnitude_weight)
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        if len(valid_idx) < len(tf_seeds):
            print(f"  Dropped {len(tf_seeds) - len(valid_idx)} flat waveforms")
            tf_seeds = [tf_seeds[i] for i in valid_idx]
            waveforms = waveforms[valid_idx]

        if len(tf_seeds) < tf_hdbscan.get('min_cluster_size', 15) * 2:
            print(f"  Too few valid seeds for clustering, skipping")
            continue

        # ── UMAP ───────────────────────────────────────────────
        # Scale n_neighbors to pool size (can't exceed N-1)
        tf_umap = umap_params.copy()
        tf_umap['n_neighbors'] = min(tf_umap['n_neighbors'],
                                     max(5, len(tf_seeds) - 1))
        embedding = embed_umap(waveforms, tf_umap)

        # ── HDBSCAN ────────────────────────────────────────────
        labels = cluster_hdbscan(embedding, tf_hdbscan)

        # ── Bootstrap ──────────────────────────────────────────
        if not args.no_bootstrap:
            stability = bootstrap_stability_test(waveforms, labels,
                                                 n_bootstrap=args.bootstrap_n)
            n_rejected = 0
            for cid, info in stability.items():
                if not info['stable']:
                    labels[np.where(labels == cid)] = -1
                    n_rejected += 1
            if n_rejected > 0:
                n_remaining = len(set(labels) - {-1})
                print(f"  Rejected {n_rejected} unstable → {n_remaining} remain")

        # ── 16D centroids ──────────────────────────────────────
        if args.skip_physics:
            centroids_16d = {}
        else:
            centroids_16d = compute_16d_centroids(tf_seeds, labels, args.data)

        # ── Build primitives for this TF ───────────────────────
        for s in tf_seeds:
            s.pop('waveform_close', None)

        tf_prims = build_primitives(tf_seeds, waveforms, labels,
                                    embedding, centroids_16d)

        # Renumber primitive IDs globally
        for p in tf_prims:
            p.primitive_id = next_prim_id
            next_prim_id += 1

        tf_clustered = int(np.sum(labels != -1))
        tf_noise = int(np.sum(labels == -1))
        total_clustered += tf_clustered
        total_noise += tf_noise
        all_primitives.extend(tf_prims)

        print(f"  {tf}: {len(tf_prims)} primitives, "
              f"{tf_clustered:,} clustered, {tf_noise:,} noise")

    # ═══════════════════════════════════════════════════════════
    # Save library
    # ═══════════════════════════════════════════════════════════

    tf_zz_params = {tf: get_zigzag_params(tf) for tf in SWING_TFS}

    library = ShapePrimitiveLibrary(
        primitives=all_primitives,
        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        n_total_seeds=len(seeds),
        n_clustered_seeds=total_clustered,
        n_noise_seeds=total_noise,
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        tf_params=tf_zz_params,
    )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(library, f)
    print(f"\n  Saved: {args.output}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print('  SHAPE PRIMITIVE SUMMARY')
    print(f"{'=' * 60}")
    print(f"  Total seeds:     {len(seeds):,}")
    print(f"  Clustered:       {total_clustered:,} ({total_clustered / max(len(seeds),1) * 100:.1f}%)")
    print(f"  Noise:           {total_noise:,} ({total_noise / max(len(seeds),1) * 100:.1f}%)")
    print(f"  Primitives:      {len(all_primitives)}")
    print()

    # Per-TF summary
    tf_prim_counts = Counter(p.dominant_tf for p in all_primitives)
    for tf in active_tfs:
        cnt = tf_prim_counts.get(tf, 0)
        if cnt > 0:
            print(f"    {tf:>5s}: {cnt:4d} primitives")
    print()

    for p in sorted(all_primitives, key=lambda x: x.n_members, reverse=True)[:20]:
        bias = max(p.direction_bias, 1 - p.direction_bias)
        dir_label = 'LONG' if p.direction_bias > 0.5 else 'SHORT'
        print(f"  #{p.primitive_id:3d}: {p.n_members:6,d} members | {p.dominant_tf:4s} | "
              f"{p.dominant_shape:16s} | {p.quality_tier_label:6s} Q={p.mean_quality_score:.2f} | "
              f"{dir_label} {bias * 100:4.0f}% | R2={p.shape_r2:.3f} | "
              f"MFE={p.mean_mfe_ticks:5.0f}t | {p.mean_duration_mins:5.1f}m")


if __name__ == '__main__':
    main()
