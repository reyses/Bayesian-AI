"""
Two-Stage Shape Primitive Data Structures
==========================================
Entry primitives: 10-bar lookback geometry + 192D context -> "what setup is this?"
Exit primitives: 32-point segment shape -> "what move to expect? how to exit?"

Both built offline by tools/shape_primitive_builder.py, consumed at runtime by
ExecutionEngine (entry) and ExitEngine/giveback (exit).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ── Lookback geometry feature names (6D) ─────────────────────────────────
GEOMETRY_FEATURES = [
    'slope',          # linreg slope / range
    'curvature',      # mean 2nd derivative
    'efficiency',     # abs(net_change) / path_length
    'norm_range',     # (max-min) / entry_price * 10000
    'end_position',   # (entry - min) / (max - min)
    'monotonicity',   # fraction of bars in dominant direction
]

GEOMETRY_DIM = len(GEOMETRY_FEATURES)   # 6
CONTEXT_DIM = 192                        # 12 TFs x 16D
ENTRY_DIM = GEOMETRY_DIM + CONTEXT_DIM   # 198
SEGMENT_DIM = 32                         # resampled segment points
MAGNITUDE_DIM = 2                        # change_ticks, range_ticks
EXIT_DIM = SEGMENT_DIM + MAGNITUDE_DIM   # 34


# ── Entry Primitive ──────────────────────────────────────────────────────

@dataclass
class EntryPrimitive:
    """A cluster of similar lookback geometries + market contexts."""
    primitive_id: int
    tf: str                                  # TF this was clustered under
    centroid_geometry: np.ndarray             # (6,) mean lookback geometry
    centroid_192d: np.ndarray                # (192,) mean 12-TF x 16D context
    centroid_16d: np.ndarray                 # (16,) mean feature at entry TF (for K-Means init)
    n_members: int
    direction_bias: float                    # fraction LONG (0-1)
    mean_mfe_ticks: float
    mean_mae_ticks: float
    mean_duration_mins: float
    shape_r2: float                          # cluster tightness
    dominant_shape: str                      # most common shape class among members
    umap_center: Tuple[float, float] = (0.0, 0.0)
    member_indices: List[int] = field(default_factory=list)
    bootstrap_stable: bool = True
    centroid_drift: float = 0.0


# ── Exit Primitive ───────────────────────────────────────────────────────

@dataclass
class ExitPrimitive:
    """A cluster of similar segment shapes with calibrated exit parameters."""
    primitive_id: int
    tf: str
    centroid_waveform: np.ndarray             # (32,) normalized segment shape
    n_members: int
    dominant_shape: str                       # V_REVERSAL, RAMP, etc.
    shape_distribution: Dict[str, int] = field(default_factory=dict)
    mean_quality_score: float = 0.0
    direction_bias: float = 0.5
    mean_mfe_ticks: float = 0.0
    mean_mae_ticks: float = 0.0
    mean_duration_mins: float = 0.0
    shape_r2: float = 0.0
    umap_center: Tuple[float, float] = (0.0, 0.0)
    member_indices: List[int] = field(default_factory=list)
    bootstrap_stable: bool = True
    centroid_drift: float = 0.0
    # Exit calibration (data-derived from member outcomes)
    giveback_pct: float = 0.55               # median retracement fraction
    giveback_delay_bars: int = 3             # bars before giveback activates
    envelope_halflife_mult: float = 1.0      # multiplier on base halflife
    expected_peak_bar: float = 0.5           # when MFE typically peaks (0-1)


# ── Entry Primitive Library ──────────────────────────────────────────────

@dataclass
class EntryPrimitiveLibrary:
    """Collection of entry primitives with matching + K-Means init support."""
    primitives: List[EntryPrimitive]
    created_at: str
    n_total_seeds: int
    n_clustered_seeds: int
    n_noise_seeds: int
    umap_params: Dict = field(default_factory=dict)
    hdbscan_params: Dict = field(default_factory=dict)
    tf_params: Dict[str, Dict] = field(default_factory=dict)
    context_scaler: Any = None               # fitted StandardScaler for 198D
    version: str = '2.0'

    def match(self, lookback_geom: np.ndarray, context_192d: np.ndarray,
              tf: str) -> Tuple[Optional[int], float]:
        """Find nearest entry primitive for a given TF.

        Args:
            lookback_geom: (6,) lookback geometry features
            context_192d: (192,) multi-TF context vector
            tf: timeframe to match against

        Returns:
            (primitive_id, distance) or (None, inf) if no match
        """
        matching = [p for p in self.primitives if p.tf == tf]
        if not matching:
            return None, float('inf')

        # Build combined vector same as training
        from tools.shape_primitive_builder import GEOMETRY_WEIGHT
        combined = np.concatenate([
            lookback_geom * GEOMETRY_WEIGHT,
            context_192d,
        ])

        # Scale with fitted scaler
        if self.context_scaler is not None:
            combined = self.context_scaler.transform(combined.reshape(1, -1))[0]

        # Euclidean distance to each centroid
        best_id, best_dist = None, float('inf')
        for p in matching:
            centroid = np.concatenate([
                p.centroid_geometry * GEOMETRY_WEIGHT,
                p.centroid_192d,
            ])
            if self.context_scaler is not None:
                centroid = self.context_scaler.transform(centroid.reshape(1, -1))[0]
            dist = np.linalg.norm(combined - centroid)
            if dist < best_dist:
                best_id, best_dist = p.primitive_id, dist
        return best_id, best_dist

    def get_centroids_for_tf(self, tf: str) -> Optional[np.ndarray]:
        """Return (K, 16) centroids for Phase 2 K-Means init."""
        matching = [p for p in self.primitives if p.tf == tf]
        if not matching:
            return None
        return np.array([p.centroid_16d for p in matching])

    def get_primitive(self, primitive_id: int) -> Optional[EntryPrimitive]:
        """Lookup by ID."""
        for p in self.primitives:
            if p.primitive_id == primitive_id:
                return p
        return None


# ── Exit Primitive Library ───────────────────────────────────────────────

@dataclass
class ExitPrimitiveLibrary:
    """Collection of exit primitives with partial-match + exit param lookup."""
    primitives: List[ExitPrimitive]
    created_at: str
    n_total_seeds: int
    n_clustered_seeds: int
    n_noise_seeds: int
    umap_params: Dict = field(default_factory=dict)
    hdbscan_params: Dict = field(default_factory=dict)
    tf_params: Dict[str, Dict] = field(default_factory=dict)
    version: str = '2.0'

    def match_partial(self, partial_segment: np.ndarray, tf: str,
                      n_bars: int) -> Tuple[Optional[int], float]:
        """Match an unfolding segment to nearest exit primitive.

        Resamples the partial segment to 32 points and compares to centroids.
        Confidence increases with n_bars (more data = better match).

        Args:
            partial_segment: raw close prices from entry onward
            tf: timeframe for filtering
            n_bars: number of bars so far

        Returns:
            (primitive_id, confidence) where confidence in [0, 1]
        """
        matching = [p for p in self.primitives if p.tf == tf]
        if not matching or n_bars < 3:
            return None, 0.0

        # Normalize: zero at entry, unit range
        entry_price = partial_segment[0]
        normalized = (partial_segment - entry_price)
        rng = np.max(np.abs(normalized))
        if rng > 0:
            normalized = normalized / rng

        # Resample to 32 points
        x_old = np.linspace(0, 1, len(normalized))
        x_new = np.linspace(0, 1, SEGMENT_DIM)
        resampled = np.interp(x_new, x_old, normalized)

        # Find nearest centroid
        best_id, best_dist = None, float('inf')
        for p in matching:
            dist = np.linalg.norm(resampled - p.centroid_waveform)
            if dist < best_dist:
                best_id, best_dist = p.primitive_id, dist

        # Confidence: sigmoid based on n_bars (more bars = higher confidence)
        # At 3 bars: ~0.2, at 10 bars: ~0.7, at 20 bars: ~0.9
        raw_confidence = 1.0 / (1.0 + np.exp(-0.3 * (n_bars - 8)))
        # Penalize by distance (closer = higher confidence)
        dist_penalty = np.exp(-best_dist)
        confidence = raw_confidence * dist_penalty

        return best_id, float(np.clip(confidence, 0.0, 1.0))

    def get_exit_params(self, primitive_id: int) -> Optional[Dict]:
        """Return calibrated exit parameters for a primitive.

        Returns:
            {giveback_pct, delay_bars, hl_mult, peak_bar} or None
        """
        p = self.get_primitive(primitive_id)
        if p is None:
            return None
        return {
            'giveback_pct': p.giveback_pct,
            'delay_bars': p.giveback_delay_bars,
            'hl_mult': p.envelope_halflife_mult,
            'peak_bar': p.expected_peak_bar,
        }

    def get_primitive(self, primitive_id: int) -> Optional[ExitPrimitive]:
        """Lookup by ID."""
        for p in self.primitives:
            if p.primitive_id == primitive_id:
                return p
        return None


# ── Lookback Geometry Extraction ─────────────────────────────────────────

def extract_lookback_geometry(closes: np.ndarray) -> np.ndarray:
    """Extract 6D geometry features from a lookback price path.

    Args:
        closes: array of close prices (10 bars typically, minimum 3)

    Returns:
        (6,) array: [slope, curvature, efficiency, norm_range, end_position, monotonicity]
    """
    n = len(closes)
    if n < 3:
        return np.zeros(GEOMETRY_DIM)

    # 1. Slope: linear regression slope, normalized by range
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    c_mean = closes.mean()
    slope_raw = np.sum((x - x_mean) * (closes - c_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12)
    price_range = closes.max() - closes.min()
    slope = slope_raw / max(price_range, 1e-12) * n  # normalize to [-1, 1] ish

    # 2. Curvature: mean second derivative
    if n >= 3:
        d2 = np.diff(closes, n=2)
        curvature = d2.mean() / max(price_range, 1e-12)
    else:
        curvature = 0.0

    # 3. Efficiency ratio: abs(net change) / total path length
    net_change = abs(closes[-1] - closes[0])
    path_length = np.sum(np.abs(np.diff(closes)))
    efficiency = net_change / max(path_length, 1e-12)

    # 4. Normalized range: (max - min) / entry_price * 10000
    entry_price = closes[-1]  # entry is last bar of lookback
    norm_range = price_range / max(entry_price, 1e-12) * 10000

    # 5. End position: where entry sits in the lookback range
    end_position = (closes[-1] - closes.min()) / max(price_range, 1e-12)

    # 6. Monotonicity: fraction of bars moving in dominant direction
    diffs = np.diff(closes)
    if len(diffs) > 0:
        ups = np.sum(diffs > 0)
        downs = np.sum(diffs < 0)
        monotonicity = max(ups, downs) / len(diffs)
    else:
        monotonicity = 0.5

    return np.array([slope, curvature, efficiency, norm_range, end_position, monotonicity],
                    dtype=np.float64)
