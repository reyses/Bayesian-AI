"""
Shared checkpoint loader  -- used by both trainer.py and live_engine.py.

Loads the common artifacts (library, scaler, tiers, depth weights, centroids)
from a checkpoint directory and returns a CheckpointBundle.

Quality filter: only templates with sufficient data (>=10 members),
proven edge (WR>=55%), and controlled variance (sigma<=10 ticks) are
included in valid_tids. Everything else is discarded at load time.
"""
import json
import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

logger = logging.getLogger(__name__)

# Quality gate thresholds  -- templates that fail ANY are excluded from trading
QUALITY_MIN_MEMBERS  = 10    # minimum cluster members to trust the statistics
QUALITY_MIN_WIN_RATE = 0.55  # minimum historical win rate
QUALITY_MAX_SIGMA    = 10.0  # maximum regression sigma in ticks


@dataclass
class CheckpointBundle:
    """All shared checkpoint artifacts needed by both trainer and live."""
    pattern_library: Dict
    scaler: object                             # StandardScaler
    valid_tids: List[int]
    centroids_scaled: np.ndarray
    template_tier_map: Dict = field(default_factory=dict)
    depth_score_adj: Dict[int, float] = field(default_factory=dict)
    depth_filter_out: Set[int] = field(default_factory=set)


def load_checkpoints(checkpoint_dir: str, *, verbose: bool = True) -> CheckpointBundle:
    """
    Load common training checkpoints from disk.

    Args:
        checkpoint_dir: Path to checkpoints/ directory.
        verbose: Print loading details (True for trainer CLI, False for quiet).

    Returns:
        CheckpointBundle with all shared artifacts.

    Raises:
        FileNotFoundError: If pattern_library.pkl is missing.
        ValueError: If no valid templates found.
    """
    _log = logger.info if not verbose else lambda msg: print(msg)

    # ── Pattern library ──────────────────────────────────────────────
    lib_path = os.path.join(checkpoint_dir, 'pattern_library.pkl')
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"pattern_library.pkl not found in {checkpoint_dir}. "
            f"Run with --fresh to build from scratch.")
    with open(lib_path, 'rb') as f:
        pattern_library = pickle.load(f)
    _log(f"  Loaded library: {len(pattern_library)} templates")

    # ── Clustering scaler ────────────────────────────────────────────
    scaler_path = os.path.join(checkpoint_dir, 'clustering_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        _n_feat = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else '?'
        _log(f"  Loaded scaler: {_n_feat} features")
        if isinstance(_n_feat, int) and _n_feat not in (16, 22):
            logger.warning(f"  [WARN] Unexpected scaler dimensionality: {_n_feat} (expected 16 or 22)")
        if isinstance(_n_feat, int) and _n_feat == 22:
            _log(f"  [info] Scaler expects 22D (--lookback mode)  -- features will be auto-padded if needed")
    else:
        # Fallback: fit on library centroids (identity-like transform)
        from sklearn.preprocessing import StandardScaler
        logger.warning("  clustering_scaler.pkl not found  -- reconstructing from centroids")
        _cents = [v['centroid'] for v in pattern_library.values() if 'centroid' in v]
        if _cents:
            scaler = StandardScaler().fit(np.array(_cents))
        else:
            raise FileNotFoundError("No centroids in library and no scaler")

    if not pattern_library:
        raise ValueError("Pattern library is empty  -- nothing to simulate")

    # ── Valid template IDs (must have centroids) ─────────────────────
    _all_tids = [tid for tid in pattern_library if 'centroid' in pattern_library[tid]]
    if not _all_tids:
        raise ValueError("No valid templates with centroids found")

    # ── Quality filter: hard gate on template statistics ─────────────
    # Only templates with proven data quality survive. The rest are noise.
    valid_tids = []
    _rejected = 0
    for tid in _all_tids:
        lib = pattern_library.get(tid, {})
        n = lib.get('member_count', 0)
        wr = lib.get('stats_win_rate', 0.0)
        sig = lib.get('regression_sigma_ticks', None)
        if (n >= QUALITY_MIN_MEMBERS
                and wr >= QUALITY_MIN_WIN_RATE
                and sig is not None
                and sig <= QUALITY_MAX_SIGMA):
            valid_tids.append(tid)
        else:
            _rejected += 1
    _log(f"  Quality filter: {len(valid_tids)} passed, {_rejected} rejected "
         f"(>={QUALITY_MIN_MEMBERS} members, WR>={QUALITY_MIN_WIN_RATE:.0%}, "
         f"sigma<={QUALITY_MAX_SIGMA} ticks)")
    if not valid_tids:
        raise ValueError("All templates rejected by quality filter  -- "
                         "lower thresholds or rebuild library with --fresh")

    # ── Template tiers ───────────────────────────────────────────────
    template_tier_map = {}
    tiers_path = os.path.join(checkpoint_dir, 'template_tiers.pkl')
    if os.path.exists(tiers_path):
        with open(tiers_path, 'rb') as f:
            template_tier_map = pickle.load(f)
        _log(f"  Loaded tier map: {len(template_tier_map)} templates "
             f"({sum(1 for v in template_tier_map.values() if v == 1)} Tier 1)")

    # ── Depth weights ────────────────────────────────────────────────
    depth_score_adj: Dict[int, float] = {}
    depth_filter_out: Set[int] = set()
    dw_path = os.path.join(checkpoint_dir, 'depth_weights.json')
    if os.path.exists(dw_path):
        with open(dw_path) as f:
            dw_data = json.load(f)
        depth_score_adj = {int(k): float(v.get('score_adj', 0.0))
                           for k, v in dw_data.items()}
        depth_filter_out = {int(k) for k, v in dw_data.items()
                            if v.get('filter_out', False)}
        _log(f"  Loaded depth weights: {len(depth_score_adj)} depths, "
             f"{len(depth_filter_out)} filtered out")

    # ── Centroids (scaled for L2 matching) ───────────────────────────
    raw_centroids = np.array([pattern_library[tid]['centroid']
                              for tid in valid_tids])
    centroids_scaled = scaler.transform(raw_centroids)
    _log(f"  Centroids: {len(valid_tids)} ready for matching")

    return CheckpointBundle(
        pattern_library=pattern_library,
        scaler=scaler,
        valid_tids=valid_tids,
        centroids_scaled=centroids_scaled,
        template_tier_map=template_tier_map,
        depth_score_adj=depth_score_adj,
        depth_filter_out=depth_filter_out,
    )
