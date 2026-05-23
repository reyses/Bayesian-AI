#!/usr/bin/env python
"""
Two-Stage Shape Primitive Builder — Entry + Exit primitives via UMAP + HDBSCAN.

Entry primitives: 10-bar lookback geometry + 192D context -> "what setup?"
Exit primitives:  segment shape + magnitude -> "how to exit?"

Pipeline (shared):
  Multi-TF ZigZag -> quality filter -> split into entry/exit paths

Entry path:
  Extract lookback geometry (6D) + 192D context @ entry bar
  -> per-TF UMAP 198D->2D -> HDBSCAN -> entry primitives
  -> checkpoints/entry_primitives.pkl

Exit path:
  Extract segment (post-entry) -> normalize + resample to 32pts + magnitude (34D)
  -> per-TF UMAP 34D->2D -> HDBSCAN -> calibrate exit params -> exit primitives
  -> checkpoints/exit_primitives.pkl

Usage:
    python tools/shape_primitive_builder.py --all                    # both (default)
    python tools/shape_primitive_builder.py --entry                  # entry only
    python tools/shape_primitive_builder.py --exit                   # exit only
    python tools/shape_primitive_builder.py --data DATA/ATLAS_1WEEK  # fast test
    python tools/shape_primitive_builder.py --plot                   # save UMAP scatter
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.auto_swing_marker import detect_swings, TICK_SIZE
from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    TF_SECONDS, TF_HIERARCHY
)
from tools.research.shape_classifier import (
    classify_shape, quality_score, quality_tier, load_calibration
)
from core_v2.shape_primitives import (
    EntryPrimitive, ExitPrimitive,
    EntryPrimitiveLibrary, ExitPrimitiveLibrary,
    extract_lookback_geometry,
    GEOMETRY_DIM, CONTEXT_DIM, ENTRY_DIM, SEGMENT_DIM, MAGNITUDE_DIM, EXIT_DIM,
)

# ===================================================================
# Constants
# ===================================================================

SWING_TFS = ['15s', '30s', '1m', '5m', '15m', '30m', '1h']
LOOKBACK_BARS = 10
RESAMPLE_POINTS = SEGMENT_DIM  # 32
SEEDS_DIR = 'DATA/regime_seeds'

UMAP_DEFAULTS = dict(n_neighbors=30, min_dist=0.05, metric='euclidean', random_state=42)
UMAP_ENTRY_DEFAULTS = dict(n_neighbors=50, min_dist=0.05, metric='euclidean', random_state=42)
HDBSCAN_DEFAULTS = dict(min_cluster_size=50, min_samples=10, cluster_selection_method='eom')

TICK_VALUE_USD = 0.50
CONFIDENCE_Z = 1.645     # 90% confidence
MAGNITUDE_WEIGHT = 3.0   # Weight for magnitude features in exit UMAP
GEOMETRY_WEIGHT = 5.0     # Weight for lookback geometry in entry UMAP (vs 192D context)


# ===================================================================
# ZigZag parameter scaling
# ===================================================================

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


# ===================================================================
# Step 1: Multi-TF swing detection
# ===================================================================

def detect_multi_tf_swings(data_dir: str, burn_hours: int = 10) -> list:
    """Run ZigZag on all swing TFs. Returns unified list of seed dicts."""
    all_seeds = []

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
                'lookback_close': close[lb_start:si + 1].copy(),   # lookback path (lb_start to entry inclusive)
                'segment_close': close[si:ei + 1].copy(),          # segment path (entry to exit inclusive)
                'waveform_close': close[lb_start:ei + 1].copy(),   # full path (for quality filter)
                'source': 'auto_swing',
            }
            all_seeds.append(seed)
            tf_seeds += 1

        print(f"  [{tf}] {tf_seeds:,} seeds (burned={burned}, short_lb={short_lb})")
        del df, close, timestamps

    return all_seeds


# ===================================================================
# Step 1b: Load human seeds
# ===================================================================

def load_human_seeds(seeds_dir: str, data_dir: str, burn_until: float = 0) -> list:
    """Load human-marked seeds from JSON files and extract their waveforms."""
    if not os.path.exists(seeds_dir):
        return []

    seed_files = sorted(Path(seeds_dir).glob('seeds_*_multi.json'))
    if not seed_files:
        seed_files = sorted(Path(seeds_dir).glob('seeds_*.json'))
    if not seed_files:
        return []

    print(f"\n  Loading human seeds from {len(seed_files)} files...")

    tf_close_cache = {}
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
                'lookback_close': close[lb_start:si + 1].copy(),
                'segment_close': close[si:ei + 1].copy(),
                'waveform_close': close[lb_start:ei + 1].copy(),
                'source': 'human',
            }
            human_seeds.append(seed)

    print(f"  Loaded {len(human_seeds)} human seeds")
    return human_seeds


# ===================================================================
# Step 2a: Exit waveform normalization (segment only)
# ===================================================================

def normalize_segments(seeds: list, magnitude_weight: float = MAGNITUDE_WEIGHT
                       ) -> Tuple[np.ndarray, list]:
    """Normalize and resample SEGMENT waveforms (post-entry) to 34D vectors.

    Shape (32D): zero at entry price, divide by max(|range|), resample to 32pts.
    Magnitude (2D): log1p of segment range and net change in ticks, weighted.
    Direction-normalized: SHORT flipped so UMAP clusters by shape, not direction.

    Returns: (N, 34) array, list of bool (valid mask).
    """
    waveforms = []
    valid_mask = []
    n_dims = RESAMPLE_POINTS + MAGNITUDE_DIM  # 34

    for seed in tqdm(seeds, desc='Normalizing segments', unit='seed',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        segment = seed['segment_close']

        if len(segment) < 3:
            valid_mask.append(False)
            waveforms.append(np.zeros(n_dims, dtype=np.float32))
            continue

        entry_price = segment[0]
        shifted = segment - entry_price
        max_range = np.max(np.abs(shifted))

        if max_range < 1e-10:
            valid_mask.append(False)
            waveforms.append(np.zeros(n_dims, dtype=np.float32))
            continue

        normalized = shifted / max_range  # [-1, 1]

        # Direction-normalize: flip SHORT so all shapes face "favorable" direction
        if seed.get('direction') == 'SHORT':
            normalized = -normalized

        # Resample to 32 points
        x_orig = np.linspace(0, 1, len(normalized))
        x_new = np.linspace(0, 1, RESAMPLE_POINTS)
        resampled = np.interp(x_new, x_orig, normalized)

        # Magnitude features
        abs_change_ticks = abs(seed.get('change_ticks', 0))
        max_range_ticks = max_range / TICK_SIZE
        mag_change = np.log1p(abs_change_ticks) / 6.0 * magnitude_weight
        mag_range = np.log1p(max_range_ticks) / 6.0 * magnitude_weight

        vec = np.concatenate([resampled, [mag_change, mag_range]])
        waveforms.append(vec.astype(np.float32))
        valid_mask.append(True)

    return np.array(waveforms, dtype=np.float32), valid_mask


# ===================================================================
# Step 2b: Entry feature extraction (6D geometry + 192D context)
# ===================================================================

def compute_entry_features(seeds: list, data_dir: str,
                           geometry_weight: float = GEOMETRY_WEIGHT
                           ) -> Tuple[np.ndarray, list]:
    """Extract 198D entry feature vectors: 6D lookback geometry + 192D context.

    Groups seeds by (tf, month), runs physics once per group for 192D extraction.
    Returns: (N, 198) array, list of bool (valid mask).
    """
    n_seeds = len(seeds)
    features = np.zeros((n_seeds, ENTRY_DIM), dtype=np.float32)
    valid_mask = [False] * n_seeds

    # Step A: Compute 6D lookback geometry for all seeds (fast, no I/O)
    print(f"  Extracting lookback geometry ({GEOMETRY_DIM}D) for {n_seeds:,} seeds...")
    for i, seed in enumerate(tqdm(seeds, desc='Lookback geometry',
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        lb_close = seed['lookback_close']
        # Left-pad if < LOOKBACK_BARS
        if len(lb_close) < LOOKBACK_BARS + 1:
            pad_n = (LOOKBACK_BARS + 1) - len(lb_close)
            lb_close = np.concatenate([np.full(pad_n, lb_close[0]), lb_close])
        geom = extract_lookback_geometry(lb_close)
        features[i, :GEOMETRY_DIM] = geom * geometry_weight

    # Step B: Compute 192D context at each seed's entry timestamp
    # Group seeds by (tf, month) for batch physics computation
    tf_month_groups = defaultdict(list)
    for i, seed in enumerate(seeds):
        month = datetime.fromtimestamp(seed['ts_start'], tz=timezone.utc).strftime('%Y_%m')
        tf_month_groups[(seed['tf'], month)].append(i)

    # We need ALL 12 TFs for 192D context, not just the seed's own TF
    # Strategy: for each seed timestamp, get the most recent state from each of the 12 TFs
    # Pre-compute physics for all (TF, month) combos needed

    # First, find all months present in seeds
    all_months = sorted(set(
        datetime.fromtimestamp(s['ts_start'], tz=timezone.utc).strftime('%Y_%m')
        for s in seeds
    ))

    # Build physics cache: (tf, month) -> {timestamp: MarketState}
    # Only load TFs that have data
    print(f"\n  Computing 192D context ({CONTEXT_DIM}D) across {len(all_months)} months x {len(TF_HIERARCHY)} TFs...")
    physics_cache = {}  # (tf, month) -> (ts_array, states_dict)

    for tf in tqdm(TF_HIERARCHY, desc='Loading TF physics',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        for month in all_months:
            df = load_atlas_tf(data_dir, tf, months=[month])
            if df.empty:
                continue
            states = compute_tf_physics(tf, df)
            if states:
                ts_arr = np.array(sorted(states.keys()))
                physics_cache[(tf, month)] = (ts_arr, states)
            del df

    # Now extract 192D for each seed
    print(f"  Extracting 192D per seed...")
    n_context_ok = 0

    for i, seed in enumerate(tqdm(seeds, desc='192D context',
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        ts = int(seed['ts_start'])
        month = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m')

        context_vec = np.zeros(CONTEXT_DIM, dtype=np.float32)
        n_tfs_found = 0

        for tf_idx, tf in enumerate(TF_HIERARCHY):
            cached = physics_cache.get((tf, month))
            if cached is None:
                continue

            ts_arr, states = cached
            pos = np.searchsorted(ts_arr, ts)
            # Use most recent completed bar (at or before entry)
            if pos > 0:
                nearest = ts_arr[pos - 1]
            elif pos < len(ts_arr):
                nearest = ts_arr[pos]
            else:
                continue

            state = states.get(int(nearest))
            if state is not None:
                feat_16d = extract_16d(state, tf)
                context_vec[tf_idx * 16:(tf_idx + 1) * 16] = feat_16d
                n_tfs_found += 1

        features[i, GEOMETRY_DIM:] = context_vec
        # Valid if we got at least 3 TFs of context
        valid_mask[i] = n_tfs_found >= 3
        if valid_mask[i]:
            n_context_ok += 1

    print(f"  {n_context_ok:,}/{n_seeds:,} seeds with valid 192D context (>=3 TFs)")

    # Clean up physics cache
    del physics_cache

    return features, valid_mask


# ===================================================================
# Step 3: UMAP embedding
# ===================================================================

def embed_umap(waveforms: np.ndarray, params: dict, label: str = '') -> np.ndarray:
    """UMAP: ND -> 2D embedding."""
    import umap

    print(f"\n  UMAP{' (' + label + ')' if label else ''}: "
          f"{waveforms.shape[0]:,} points, {waveforms.shape[1]}D -> 2D")
    print(f"  Params: {params}")

    reducer = umap.UMAP(n_components=2, **params)
    embedding = reducer.fit_transform(waveforms)
    print(f"  Done: {embedding.shape}")
    return embedding


# ===================================================================
# Step 3b: Power analysis
# ===================================================================

def compute_tf_power_analysis(tf_seeds: list, relative_tolerance: float = 0.50,
                              confidence_z: float = CONFIDENCE_Z) -> int:
    """Compute min_cluster_size for a single TF via power analysis."""
    mfe_vals = np.array([abs(s['change_ticks']) for s in tf_seeds])
    mu = max(float(np.mean(mfe_vals)), 1.0) if len(mfe_vals) > 0 else 50.0
    sigma = float(np.std(mfe_vals)) if len(mfe_vals) > 2 else 50.0
    cv = sigma / mu
    base_n = (confidence_z / relative_tolerance) ** 2
    n_req = int(np.ceil(base_n * cv ** 2))
    n_req = max(15, min(500, n_req))
    return max(15, min(n_req, len(tf_seeds) // 20))


# ===================================================================
# Step 4: HDBSCAN clustering
# ===================================================================

def cluster_hdbscan(embedding: np.ndarray, params: dict) -> np.ndarray:
    """HDBSCAN clustering in 2D UMAP space."""
    import hdbscan

    print(f"\n  HDBSCAN: {embedding.shape[0]:,} points")
    print(f"  Params: {params}")

    clusterer = hdbscan.HDBSCAN(**params)
    labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"  Found {n_clusters} clusters, {n_noise:,} noise ({n_noise / max(len(labels), 1) * 100:.1f}%)")

    return labels


# ===================================================================
# Step 4b: Bootstrap stability test
# ===================================================================

def bootstrap_stability_test(waveforms: np.ndarray, labels: np.ndarray,
                             n_bootstrap: int = 100, max_drift_frac: float = 0.15,
                             rng_seed: int = 42) -> dict:
    """Test each cluster's centroid stability via bootstrap resampling."""
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


# ===================================================================
# Step 5: 16D feature centroids (for K-Means init)
# ===================================================================

def compute_16d_centroids(seeds: list, labels: np.ndarray, data_dir: str) -> dict:
    """Compute 16D feature centroid per cluster."""
    tf_month_groups = defaultdict(list)
    for i, seed in enumerate(seeds):
        if labels[i] == -1:
            continue
        month = datetime.fromtimestamp(seed['ts_start'], tz=timezone.utc).strftime('%Y_%m')
        tf_month_groups[(seed['tf'], month)].append(i)

    n_groups = len(tf_month_groups)
    print(f"\n  Computing 16D across {n_groups} (tf, month) groups...")

    seed_features = {}

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


# ===================================================================
# Step 6a: Build ENTRY primitives from clusters
# ===================================================================

def build_entry_primitives_from_clusters(seeds, features, labels, embedding,
                                         centroids_16d, tf_label: str,
                                         bootstrap_results: dict = None) -> list:
    """Assemble EntryPrimitive objects from entry clustering results."""
    cluster_ids = sorted(set(labels) - {-1})
    primitives = []

    for cid in cluster_ids:
        member_idx = np.where(labels == cid)[0]
        member_seeds = [seeds[i] for i in member_idx]
        member_feats = features[member_idx]
        member_embed = embedding[member_idx]

        # Centroid geometry (first 6D after unweighting)
        centroid_geom = np.mean(member_feats[:, :GEOMETRY_DIM], axis=0) / GEOMETRY_WEIGHT
        # Centroid 192D (remaining dims)
        centroid_192d = np.mean(member_feats[:, GEOMETRY_DIM:], axis=0)

        # Shape R2 on geometry dims only (6D unweighted) — interpretable cluster tightness
        geom_feats = member_feats[:, :GEOMETRY_DIM] / GEOMETRY_WEIGHT
        geom_centroid = np.mean(geom_feats, axis=0)
        if len(geom_feats) > 1:
            ss_res = np.sum((geom_feats - geom_centroid) ** 2)
            # scalar mean for ss_tot (matches exit R2 pattern)
            ss_tot = np.sum((geom_feats - np.mean(geom_feats)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        else:
            r2 = 1.0

        n_long = sum(1 for s in member_seeds if s['direction'] == 'LONG')
        direction_bias = n_long / len(member_seeds)

        mfes = [abs(s['change_ticks']) for s in member_seeds]
        durations = [s['duration_mins'] for s in member_seeds]

        sh_counts = Counter(s.get('shape', 'UNKNOWN') for s in member_seeds)
        dom_shape = sh_counts.most_common(1)[0][0] if sh_counts else 'UNKNOWN'

        # Bootstrap info
        stable = True
        drift = 0.0
        if bootstrap_results and cid in bootstrap_results:
            stable = bootstrap_results[cid]['stable']
            drift = bootstrap_results[cid].get('drift', 0.0)

        prim = EntryPrimitive(
            primitive_id=cid,
            tf=tf_label,
            centroid_geometry=centroid_geom.astype(np.float32),
            centroid_192d=centroid_192d.astype(np.float32),
            centroid_16d=centroids_16d.get(cid, np.zeros(16, dtype=np.float32)),
            n_members=len(member_idx),
            direction_bias=direction_bias,
            mean_mfe_ticks=float(np.mean(mfes)),
            mean_mae_ticks=0.0,
            mean_duration_mins=float(np.mean(durations)),
            shape_r2=float(r2),
            dominant_shape=dom_shape,
            umap_center=(float(np.mean(member_embed[:, 0])),
                         float(np.mean(member_embed[:, 1]))),
            member_indices=member_idx.tolist(),
            bootstrap_stable=stable,
            centroid_drift=drift,
        )
        primitives.append(prim)

    return primitives


# ===================================================================
# Step 6b: Build EXIT primitives from clusters + calibrate exit params
# ===================================================================

def calibrate_exit_params(member_seeds: list) -> dict:
    """Compute calibrated exit parameters from member oracle outcomes.

    Returns dict with giveback_pct, giveback_delay_bars, envelope_halflife_mult,
    expected_peak_bar.
    """
    if not member_seeds:
        return {'giveback_pct': 0.55, 'giveback_delay_bars': 3,
                'envelope_halflife_mult': 1.0, 'expected_peak_bar': 0.5}

    # Giveback pct: derived from peak position + monotonicity, NOT efficiency.
    # ZigZag segments are clean swings (efficiency ≈ 1.0 always), so efficiency
    # can't differentiate shapes. Instead:
    #   - Early peak → tight giveback (sharp move reverses quickly)
    #   - Late peak → loose giveback (steady ramp, pullbacks are normal)
    #   - Low monotonicity → slightly tighter (choppy = unreliable)
    peak_positions = []
    monotonicities = []
    durations = []
    n_bars_list = []
    for s in member_seeds:
        seg = s.get('segment_close', None)
        if seg is not None and len(seg) > 2:
            entry = seg[0]
            direction = s.get('direction', 'LONG')
            if direction == 'LONG':
                shifted = seg - entry
            else:
                shifted = entry - seg
            peak_idx = np.argmax(shifted)
            peak_positions.append(peak_idx / max(len(seg) - 1, 1))
            # Monotonicity: fraction of bars moving in favorable direction
            diffs = np.diff(shifted)
            mono = np.sum(diffs > 0) / max(len(diffs), 1)
            monotonicities.append(mono)
        durations.append(s.get('duration_mins', 5.0))
        n_bars_list.append(s.get('n_bars', 5))

    if peak_positions:
        med_peak_pos = float(np.median(peak_positions))
        med_mono = float(np.median(monotonicities)) if monotonicities else 0.5
        # Peak position drives base giveback (linear: 0.30 at early peak → 0.80 at late peak)
        base_gb = 0.30 + med_peak_pos * 0.50
        # Monotonicity modulates: high mono → loosen, low mono → tighten (±0.05)
        mono_adj = (med_mono - 0.5) * 0.10
        giveback_pct = max(0.25, min(0.85, base_gb + mono_adj))
    else:
        giveback_pct = 0.55

    # Delay bars: approximate as half the segment length (peak in middle)
    if n_bars_list:
        median_bars = float(np.median(n_bars_list))
        delay_bars = max(1, min(20, int(median_bars * 0.4)))
    else:
        delay_bars = 3

    # Envelope halflife multiplier: based on duration relative to global median
    if durations:
        global_median_dur = 5.0  # approximate global median
        median_dur = float(np.median(durations))
        hl_mult = max(0.3, min(3.0, median_dur / max(global_median_dur, 0.1)))
    else:
        hl_mult = 1.0

    # Expected peak bar: fraction through segment where MFE peaks
    # Approximate from segment shape (find the extreme point)
    peak_fracs = []
    for s in member_seeds:
        seg = s.get('segment_close', None)
        if seg is not None and len(seg) > 2:
            entry = seg[0]
            shifted = seg - entry
            if s.get('direction') == 'SHORT':
                shifted = -shifted
            peak_idx = np.argmax(shifted)
            peak_fracs.append(peak_idx / max(len(seg) - 1, 1))

    expected_peak_bar = float(np.median(peak_fracs)) if peak_fracs else 0.5

    return {
        'giveback_pct': round(giveback_pct, 3),
        'giveback_delay_bars': delay_bars,
        'envelope_halflife_mult': round(hl_mult, 3),
        'expected_peak_bar': round(expected_peak_bar, 3),
    }


def build_exit_primitives_from_clusters(seeds, waveforms, labels, embedding,
                                        tf_label: str,
                                        bootstrap_results: dict = None) -> list:
    """Assemble ExitPrimitive objects with calibrated exit parameters."""
    cluster_ids = sorted(set(labels) - {-1})
    primitives = []

    for cid in cluster_ids:
        member_idx = np.where(labels == cid)[0]
        member_seeds = [seeds[i] for i in member_idx]
        member_waves = waveforms[member_idx]
        member_embed = embedding[member_idx]

        # Centroid and R2 on shape dims only (first 32)
        shape_waves = member_waves[:, :RESAMPLE_POINTS]
        centroid_wave = np.mean(shape_waves, axis=0)

        if len(shape_waves) > 1:
            ss_res = np.sum((shape_waves - centroid_wave) ** 2)
            ss_tot = np.sum((shape_waves - np.mean(shape_waves)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        else:
            r2 = 1.0

        n_long = sum(1 for s in member_seeds if s['direction'] == 'LONG')
        direction_bias = n_long / len(member_seeds)

        mfes = [abs(s['change_ticks']) for s in member_seeds]
        durations = [s['duration_mins'] for s in member_seeds]

        sh_counts = Counter(s.get('shape', 'UNKNOWN') for s in member_seeds)
        dom_shape = sh_counts.most_common(1)[0][0] if sh_counts else 'UNKNOWN'
        q_scores = [s.get('quality_score', 0.0) for s in member_seeds]
        mean_q = float(np.mean(q_scores)) if q_scores else 0.0

        # Exit calibration
        exit_params = calibrate_exit_params(member_seeds)

        # Bootstrap info
        stable = True
        drift = 0.0
        if bootstrap_results and cid in bootstrap_results:
            stable = bootstrap_results[cid]['stable']
            drift = bootstrap_results[cid].get('drift', 0.0)

        prim = ExitPrimitive(
            primitive_id=cid,
            tf=tf_label,
            centroid_waveform=centroid_wave.astype(np.float32),
            n_members=len(member_idx),
            dominant_shape=dom_shape,
            shape_distribution=dict(sh_counts),
            mean_quality_score=mean_q,
            direction_bias=direction_bias,
            mean_mfe_ticks=float(np.mean(mfes)),
            mean_mae_ticks=0.0,
            mean_duration_mins=float(np.mean(durations)),
            shape_r2=float(r2),
            umap_center=(float(np.mean(member_embed[:, 0])),
                         float(np.mean(member_embed[:, 1]))),
            member_indices=member_idx.tolist(),
            bootstrap_stable=stable,
            centroid_drift=drift,
            giveback_pct=exit_params['giveback_pct'],
            giveback_delay_bars=exit_params['giveback_delay_bars'],
            envelope_halflife_mult=exit_params['envelope_halflife_mult'],
            expected_peak_bar=exit_params['expected_peak_bar'],
        )
        primitives.append(prim)

    return primitives


# ===================================================================
# Visualization
# ===================================================================

def save_umap_plot(embedding, labels, seeds, output_path, title_prefix=''):
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

    prefix = f'{title_prefix} ' if title_prefix else ''
    plt.suptitle(f'{prefix}Shape Primitive Builder - UMAP Embedding', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


# ===================================================================
# Quality filter (shared by both paths)
# ===================================================================

def apply_quality_filter(seeds: list, threshold: float, recalibrate: bool,
                         seeds_dir: str, data_dir: str) -> list:
    """Classify shapes and filter by quality score."""
    q_priors = None
    if recalibrate:
        from tools.research.shape_classifier import calibrate_from_human_seeds
        q_priors = calibrate_from_human_seeds(seeds_dir, data_dir)
    else:
        q_priors = load_calibration()

    if q_priors:
        print(f"  Using calibrated priors ({len(q_priors)} shapes)")
    else:
        print(f"  Using default priors (255 human seeds, Jan 5-7 2025)")

    print(f"  Classifying {len(seeds):,} raw swings (threshold={threshold})...")

    filtered_seeds = []
    shape_counts = defaultdict(int)
    tier_counts = {'GOLD': 0, 'SILVER': 0, 'BRONZE': 0, 'NOISE': 0}

    for seed in tqdm(seeds, desc='Classifying',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        waveform = seed['waveform_close']
        entry_idx = seed['lookback_bars']
        shape, conf, features = classify_shape(waveform, entry_idx)

        score = quality_score(shape, conf, features, priors=q_priors)

        if seed.get('source') == 'human':
            score = max(score, 0.7)

        seed['shape'] = shape
        seed['shape_confidence'] = conf
        seed['quality_score'] = score

        shape_counts[shape] += 1
        tier = quality_tier(score)
        tier_counts[tier] += 1

        if score >= threshold:
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

    return filtered_seeds


# ===================================================================
# Per-TF clustering loop (shared logic)
# ===================================================================

def run_per_tf_clustering(seeds: list, features_or_waveforms: np.ndarray,
                          umap_params: dict, hdbscan_params: dict,
                          conf_z: float, rel_tol: float,
                          manual_min_cluster: int = 0,
                          do_bootstrap: bool = True, bootstrap_n: int = 100,
                          label: str = '') -> Tuple[dict, dict, dict]:
    """Run per-TF UMAP + HDBSCAN clustering.

    Returns:
        tf_labels: {tf: labels array}
        tf_embeddings: {tf: embedding array}
        tf_seeds_map: {tf: list of seeds}
        tf_features_map: {tf: features/waveforms array}
        tf_bootstrap: {tf: bootstrap_results}
    """
    tf_seed_groups = defaultdict(list)
    tf_seed_indices = defaultdict(list)
    for i, s in enumerate(seeds):
        tf_seed_groups[s['tf']].append(s)
        tf_seed_indices[s['tf']].append(i)

    active_tfs = sorted(tf_seed_groups.keys(),
                        key=lambda t: TF_SECONDS.get(t, 0))

    results = {}  # tf -> {seeds, features, labels, embedding, bootstrap}

    for tf in active_tfs:
        tf_seeds = tf_seed_groups[tf]
        tf_idx = tf_seed_indices[tf]
        tf_feats = features_or_waveforms[tf_idx]

        print(f"\n{'=' * 60}")
        print(f"  {label} TF: {tf} ({len(tf_seeds):,} seeds)")
        print(f"{'=' * 60}")

        if len(tf_seeds) < 30:
            print(f"  Too few seeds, skipping")
            continue

        # Power analysis
        if not manual_min_cluster:
            tf_min_cluster = compute_tf_power_analysis(tf_seeds, rel_tol, conf_z)
            tf_hdbscan = hdbscan_params.copy()
            tf_hdbscan['min_cluster_size'] = tf_min_cluster
            tf_hdbscan['min_samples'] = max(5, tf_min_cluster // 5)
            print(f"  Power: min_cluster={tf_min_cluster}")
        else:
            tf_hdbscan = hdbscan_params.copy()
            tf_hdbscan['min_cluster_size'] = manual_min_cluster

        # Filter out invalid features (all-zero rows)
        valid_mask = np.any(tf_feats != 0, axis=1)
        if not np.all(valid_mask):
            n_dropped = int(np.sum(~valid_mask))
            print(f"  Dropped {n_dropped} zero-feature seeds")
            valid_idx = np.where(valid_mask)[0]
            tf_seeds = [tf_seeds[i] for i in valid_idx]
            tf_feats = tf_feats[valid_idx]

        if len(tf_seeds) < tf_hdbscan.get('min_cluster_size', 15) * 2:
            print(f"  Too few valid seeds for clustering, skipping")
            continue

        # UMAP
        tf_umap = umap_params.copy()
        tf_umap['n_neighbors'] = min(tf_umap['n_neighbors'],
                                     max(5, len(tf_seeds) - 1))
        embedding = embed_umap(tf_feats, tf_umap, label=f'{label} {tf}')

        # HDBSCAN
        labels = cluster_hdbscan(embedding, tf_hdbscan)

        # Bootstrap
        bootstrap_res = {}
        if do_bootstrap:
            bootstrap_res = bootstrap_stability_test(tf_feats, labels,
                                                      n_bootstrap=bootstrap_n)
            n_rejected = 0
            for cid, info in bootstrap_res.items():
                if not info['stable']:
                    labels[np.where(labels == cid)] = -1
                    n_rejected += 1
            if n_rejected > 0:
                n_remaining = len(set(labels) - {-1})
                print(f"  Rejected {n_rejected} unstable -> {n_remaining} remain")

        n_clustered = int(np.sum(labels != -1))
        n_noise = int(np.sum(labels == -1))
        n_clusters = len(set(labels) - {-1})
        print(f"  {tf}: {n_clusters} clusters, {n_clustered:,} clustered, {n_noise:,} noise")

        results[tf] = {
            'seeds': tf_seeds,
            'features': tf_feats,
            'labels': labels,
            'embedding': embedding,
            'bootstrap': bootstrap_res,
        }

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Shape Primitive Builder')
    parser.add_argument('--data', default='DATA/ATLAS', help='ATLAS data directory')
    parser.add_argument('--burn-hours', type=int, default=10)
    parser.add_argument('--seeds-dir', default=SEEDS_DIR)

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--entry', action='store_true', help='Build entry primitives only')
    mode.add_argument('--exit', action='store_true', help='Build exit primitives only')
    mode.add_argument('--all', action='store_true', default=True,
                      help='Build both entry and exit primitives (default)')

    parser.add_argument('--plot', action='store_true', help='Save UMAP scatter plots')
    parser.add_argument('--skip-physics', action='store_true',
                        help='Skip 16D computation (fast test for exit path)')

    # Quality filter
    parser.add_argument('--quality-threshold', type=float, default=0.3)
    parser.add_argument('--no-filter', action='store_true')
    parser.add_argument('--recalibrate', action='store_true')

    # Clustering params
    parser.add_argument('--relative-tolerance', type=float, default=0.50)
    parser.add_argument('--confidence', type=float, default=None)
    parser.add_argument('--magnitude-weight', type=float, default=MAGNITUDE_WEIGHT)
    parser.add_argument('--geometry-weight', type=float, default=GEOMETRY_WEIGHT)
    parser.add_argument('--no-bootstrap', action='store_true')
    parser.add_argument('--bootstrap-n', type=int, default=100)

    # UMAP/HDBSCAN overrides
    parser.add_argument('--umap-neighbors', type=int, default=None)
    parser.add_argument('--umap-min-dist', type=float, default=None)
    parser.add_argument('--hdbscan-min-cluster', type=int, default=None)
    parser.add_argument('--hdbscan-min-samples', type=int, default=None)

    # Output paths
    parser.add_argument('--entry-output', default='checkpoints/entry_primitives.pkl')
    parser.add_argument('--exit-output', default='checkpoints/exit_primitives.pkl')

    args = parser.parse_args()

    # Resolve mode
    do_entry = args.entry or (not args.exit)
    do_exit = args.exit or (not args.entry)

    print('=' * 60)
    print('  TWO-STAGE SHAPE PRIMITIVE BUILDER')
    print(f'  Mode: {"ENTRY+EXIT" if (do_entry and do_exit) else "ENTRY" if do_entry else "EXIT"}')
    print('=' * 60)

    conf_z = args.confidence if args.confidence else CONFIDENCE_Z

    # Build UMAP/HDBSCAN param dicts
    exit_umap_params = UMAP_DEFAULTS.copy()
    entry_umap_params = UMAP_ENTRY_DEFAULTS.copy()
    if args.umap_neighbors:
        exit_umap_params['n_neighbors'] = args.umap_neighbors
        entry_umap_params['n_neighbors'] = args.umap_neighbors
    if args.umap_min_dist is not None:
        exit_umap_params['min_dist'] = args.umap_min_dist
        entry_umap_params['min_dist'] = args.umap_min_dist

    hdbscan_params = HDBSCAN_DEFAULTS.copy()
    manual_min_cluster = 0
    if args.hdbscan_min_cluster:
        hdbscan_params['min_cluster_size'] = args.hdbscan_min_cluster
        manual_min_cluster = args.hdbscan_min_cluster
    if args.hdbscan_min_samples:
        hdbscan_params['min_samples'] = args.hdbscan_min_samples

    # ── Step 1: Multi-TF swing detection (shared) ────────────────
    print(f"\n{'=' * 60}")
    print('  STEP 1: Multi-TF Swing Detection')
    print(f"{'=' * 60}")
    seeds = detect_multi_tf_swings(args.data, burn_hours=args.burn_hours)

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

    # ── Step 1c: Shape quality filter (shared) ───────────────────
    if not args.no_filter and args.quality_threshold > 0:
        print(f"\n{'=' * 60}")
        print('  STEP 1c: Shape Quality Filter')
        print(f"{'=' * 60}")
        seeds = apply_quality_filter(seeds, args.quality_threshold,
                                     args.recalibrate, args.seeds_dir, args.data)

    if not seeds:
        print('  No seeds after filtering. Exiting.')
        return

    # ==============================================================
    # ENTRY PRIMITIVES
    # ==============================================================

    if do_entry:
        print(f"\n{'=' * 60}")
        print('  ENTRY PRIMITIVE BUILDER')
        print(f"  Input: {GEOMETRY_DIM}D lookback geometry (x{args.geometry_weight}) + {CONTEXT_DIM}D context = {ENTRY_DIM}D")
        print(f"{'=' * 60}")

        # Step 2b: Compute entry features (6D geometry + 192D context)
        entry_features, entry_valid = compute_entry_features(
            seeds, args.data, geometry_weight=args.geometry_weight)

        # Filter invalid
        valid_idx = [i for i, v in enumerate(entry_valid) if v]
        entry_seeds = [seeds[i] for i in valid_idx]
        entry_feats = entry_features[valid_idx]
        print(f"  Valid entry seeds: {len(entry_seeds):,} / {len(seeds):,}")

        if len(entry_seeds) < 50:
            print("  Too few valid entry seeds. Skipping entry primitives.")
        else:
            # Per-TF clustering
            entry_results = run_per_tf_clustering(
                entry_seeds, entry_feats,
                entry_umap_params, hdbscan_params,
                conf_z, args.relative_tolerance,
                manual_min_cluster=manual_min_cluster,
                do_bootstrap=not args.no_bootstrap,
                bootstrap_n=args.bootstrap_n,
                label='ENTRY',
            )

            # Build entry primitives
            all_entry_prims = []
            total_entry_clustered = 0
            total_entry_noise = 0
            next_id = 0

            for tf, res in sorted(entry_results.items(),
                                   key=lambda x: TF_SECONDS.get(x[0], 0)):
                # 16D centroids for K-Means init
                if args.skip_physics:
                    centroids_16d = {}
                else:
                    centroids_16d = compute_16d_centroids(
                        res['seeds'], res['labels'], args.data)

                prims = build_entry_primitives_from_clusters(
                    res['seeds'], res['features'], res['labels'],
                    res['embedding'], centroids_16d, tf,
                    bootstrap_results=res['bootstrap'])

                for p in prims:
                    p.primitive_id = next_id
                    next_id += 1

                total_entry_clustered += int(np.sum(res['labels'] != -1))
                total_entry_noise += int(np.sum(res['labels'] == -1))
                all_entry_prims.extend(prims)

            # Fit StandardScaler on all valid entry features for runtime matching
            from sklearn.preprocessing import StandardScaler
            entry_scaler = StandardScaler()
            entry_scaler.fit(entry_feats)

            entry_library = EntryPrimitiveLibrary(
                primitives=all_entry_prims,
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                n_total_seeds=len(entry_seeds),
                n_clustered_seeds=total_entry_clustered,
                n_noise_seeds=total_entry_noise,
                umap_params=entry_umap_params,
                hdbscan_params=hdbscan_params,
                tf_params={tf: get_zigzag_params(tf) for tf in SWING_TFS},
                context_scaler=entry_scaler,
            )

            os.makedirs(os.path.dirname(args.entry_output) or '.', exist_ok=True)
            with open(args.entry_output, 'wb') as f:
                pickle.dump(entry_library, f)
            print(f"\n  Saved entry primitives: {args.entry_output}")

            # Summary
            print(f"\n{'=' * 60}")
            print('  ENTRY PRIMITIVE SUMMARY')
            print(f"{'=' * 60}")
            print(f"  Total seeds:     {len(entry_seeds):,}")
            print(f"  Clustered:       {total_entry_clustered:,}")
            print(f"  Noise:           {total_entry_noise:,}")
            print(f"  Primitives:      {len(all_entry_prims)}")

            tf_prim_counts = Counter(p.tf for p in all_entry_prims)
            for tf in sorted(tf_prim_counts.keys(),
                             key=lambda t: TF_SECONDS.get(t, 0)):
                print(f"    {tf:>5s}: {tf_prim_counts[tf]:4d} primitives")

            for p in sorted(all_entry_prims, key=lambda x: x.n_members, reverse=True)[:15]:
                bias = max(p.direction_bias, 1 - p.direction_bias)
                dir_label = 'LONG' if p.direction_bias > 0.5 else 'SHORT'
                print(f"  #{p.primitive_id:3d}: {p.n_members:6,d} members | {p.tf:4s} | "
                      f"{p.dominant_shape:16s} | "
                      f"{dir_label} {bias * 100:4.0f}% | R2={p.shape_r2:.3f} | "
                      f"MFE={p.mean_mfe_ticks:5.0f}t | {p.mean_duration_mins:5.1f}m")

            # Plot
            if args.plot:
                for tf, res in entry_results.items():
                    plot_path = f"checkpoints/umap_entry_{tf}.png"
                    save_umap_plot(res['embedding'], res['labels'],
                                   res['seeds'], plot_path,
                                   title_prefix=f'Entry {tf}')

    # ==============================================================
    # EXIT PRIMITIVES
    # ==============================================================

    if do_exit:
        print(f"\n{'=' * 60}")
        print('  EXIT PRIMITIVE BUILDER')
        print(f"  Input: {SEGMENT_DIM}D shape + {MAGNITUDE_DIM}D magnitude = {EXIT_DIM}D")
        print(f"{'=' * 60}")

        # Step 2a: Normalize segments
        exit_waveforms, exit_valid = normalize_segments(
            seeds, magnitude_weight=args.magnitude_weight)

        valid_idx = [i for i, v in enumerate(exit_valid) if v]
        exit_seeds = [seeds[i] for i in valid_idx]
        exit_waves = exit_waveforms[valid_idx]
        print(f"  Valid exit seeds: {len(exit_seeds):,} / {len(seeds):,}")

        if len(exit_seeds) < 50:
            print("  Too few valid exit seeds. Skipping exit primitives.")
        else:
            # Per-TF clustering
            exit_results = run_per_tf_clustering(
                exit_seeds, exit_waves,
                exit_umap_params, hdbscan_params,
                conf_z, args.relative_tolerance,
                manual_min_cluster=manual_min_cluster,
                do_bootstrap=not args.no_bootstrap,
                bootstrap_n=args.bootstrap_n,
                label='EXIT',
            )

            # Build exit primitives
            all_exit_prims = []
            total_exit_clustered = 0
            total_exit_noise = 0
            next_id = 0

            for tf, res in sorted(exit_results.items(),
                                   key=lambda x: TF_SECONDS.get(x[0], 0)):
                prims = build_exit_primitives_from_clusters(
                    res['seeds'], res['features'], res['labels'],
                    res['embedding'], tf,
                    bootstrap_results=res['bootstrap'])

                for p in prims:
                    p.primitive_id = next_id
                    next_id += 1

                total_exit_clustered += int(np.sum(res['labels'] != -1))
                total_exit_noise += int(np.sum(res['labels'] == -1))
                all_exit_prims.extend(prims)

            exit_library = ExitPrimitiveLibrary(
                primitives=all_exit_prims,
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                n_total_seeds=len(exit_seeds),
                n_clustered_seeds=total_exit_clustered,
                n_noise_seeds=total_exit_noise,
                umap_params=exit_umap_params,
                hdbscan_params=hdbscan_params,
                tf_params={tf: get_zigzag_params(tf) for tf in SWING_TFS},
            )

            os.makedirs(os.path.dirname(args.exit_output) or '.', exist_ok=True)
            with open(args.exit_output, 'wb') as f:
                pickle.dump(exit_library, f)
            print(f"\n  Saved exit primitives: {args.exit_output}")

            # Summary
            print(f"\n{'=' * 60}")
            print('  EXIT PRIMITIVE SUMMARY')
            print(f"{'=' * 60}")
            print(f"  Total seeds:     {len(exit_seeds):,}")
            print(f"  Clustered:       {total_exit_clustered:,}")
            print(f"  Noise:           {total_exit_noise:,}")
            print(f"  Primitives:      {len(all_exit_prims)}")

            tf_prim_counts = Counter(p.tf for p in all_exit_prims)
            for tf in sorted(tf_prim_counts.keys(),
                             key=lambda t: TF_SECONDS.get(t, 0)):
                print(f"    {tf:>5s}: {tf_prim_counts[tf]:4d} primitives")

            for p in sorted(all_exit_prims, key=lambda x: x.n_members, reverse=True)[:15]:
                bias = max(p.direction_bias, 1 - p.direction_bias)
                dir_label = 'LONG' if p.direction_bias > 0.5 else 'SHORT'
                print(f"  #{p.primitive_id:3d}: {p.n_members:6,d} members | {p.tf:4s} | "
                      f"{p.dominant_shape:16s} | "
                      f"{dir_label} {bias * 100:4.0f}% | R2={p.shape_r2:.3f} | "
                      f"MFE={p.mean_mfe_ticks:5.0f}t | {p.mean_duration_mins:5.1f}m | "
                      f"GB={p.giveback_pct:.0%} delay={p.giveback_delay_bars}b "
                      f"HL={p.envelope_halflife_mult:.2f}x peak@{p.expected_peak_bar:.0%}")

            # Plot
            if args.plot:
                for tf, res in exit_results.items():
                    plot_path = f"checkpoints/umap_exit_{tf}.png"
                    save_umap_plot(res['embedding'], res['labels'],
                                   res['seeds'], plot_path,
                                   title_prefix=f'Exit {tf}')

    print(f"\n{'=' * 60}")
    print('  DONE')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
