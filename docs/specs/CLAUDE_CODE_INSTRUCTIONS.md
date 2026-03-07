# CLAUDE CODE INSTRUCTIONS: Level Detector Module
# Project: BayesianBridge MNQ Trading System
# Date: March 6, 2026

## CONTEXT

You are adding a new module to an existing Python trading system.
The codebase lives in a project with this structure:

```
core/                    # Physics engine, brain, state vectors
  quantum_field_engine.py  # QuantumFieldEngine — computes 3-body states
  three_body_state.py      # ThreeBodyQuantumState dataclass
  bayesian_brain.py        # Probability table
training/                # Training pipeline
  fractal_discovery_agent.py
  fractal_clustering.py
  timeframe_belief_network.py  # 11 TF workers
  wave_rider.py            # Position management
  orchestrator.py          # Main training loop
live/                    # Live trading connector
  live_engine.py
  bar_aggregator.py
config/                  # Settings, symbols
```

The system trades MNQ futures on NinjaTrader 8.
Tick size = 0.25, point value = $2.00.

---

## TASK: Build `core/level_detector.py`

Create a NEW file `core/level_detector.py` containing the complete
multi-timeframe level detection system. This is standalone — does NOT
modify any existing files. Integration comes later.

---

## FILE 1: `core/level_detector.py`

### Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3    # DBSCAN
matplotlib>=3.7
mplfinance>=0.12     # optional, for candlestick overlay
```

All are already in the project's requirements.txt.

### Module Structure (all in one file)

```python
"""
Multi-Timeframe Structural Level Detector
==========================================
Detects support/resistance levels using:
1. Manual range input (2 prices) → Fibonacci structural grid
2. Swing high/low detection across 4 timeframes
3. DBSCAN clustering of converging horizontals
4. Temporal windowing (recency decay)
5. Visual overlay for human supervision
6. JSON export for physics engine consumption

Usage:
    from core.level_detector import LevelDetector
    
    detector = LevelDetector.from_range_config('range_config.json')
    result = detector.run(data_dir='DATA/ATLAS')
    # result['overlay_path'] -> PNG for visual review
    # result['json_path']    -> price_levels.json for physics engine
"""
```

### Class: `LevelDetector`

One class, clean public API:

```python
class LevelDetector:
    """
    Main entry point. Orchestrates the full pipeline:
    fib generation → swing detection → aggregation → temporal window →
    DBSCAN clustering → fib reinforcement scoring → overlay → export.
    """
    
    def __init__(self, range_top: float, range_bottom: float,
                 tick_size: float = 0.25,
                 config: dict = None):
        """
        Args:
            range_top: Daily swing high (your upper line)
            range_bottom: Daily swing low (your lower line)
            tick_size: Instrument tick size (MNQ = 0.25)
            config: Override default params (eps_ticks, half_lives, etc.)
        """
    
    @classmethod
    def from_range_config(cls, config_path: str) -> 'LevelDetector':
        """Load from range_config.json"""
    
    def run(self, data_dir: str = None,
            ohlcv_data: dict = None,
            output_dir: str = './output/') -> dict:
        """
        Full pipeline. Accepts EITHER:
          - data_dir: path to ATLAS root (reads parquet files)
          - ohlcv_data: dict of {'daily': pd.DataFrame, '4h': df, '1h': df, '15m': df}
        
        Returns:
            {
                'fib_levels': List[FibLevel],
                'sub_levels': List[StructuralLevel],
                'all_levels': List[dict],  # unified, sorted by price
                'overlay_path': str,
                'json_path': str,
                'stats': dict
            }
        """
    
    def get_levels_for_price(self, price: float, n_nearest: int = 3) -> list:
        """
        Quick lookup: given current price, return N nearest levels
        above and below. For real-time use by physics engine.
        
        Returns list of dicts sorted by distance:
            [{'price': 25088.57, 'type': 'fib_anchor', 'label': 'Fib 61.8%',
              'distance_ticks': 5.3, 'direction': 'below', 'confidence': 0.92}, ...]
        """
```

### Internal Components (private methods or inner dataclasses)

#### 1. Fibonacci Generator

```python
@dataclass
class FibLevel:
    price: float
    ratio: float        # 0.0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0
    label: str          # "Fib 61.8%"
    tier: str = 'anchor'

FIB_RATIOS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0]

def _generate_fib_levels(range_top, range_bottom) -> List[FibLevel]:
    """
    fib_price = range_top - (range_top - range_bottom) * ratio
    
    For MNQ range 24544.25 → 25969.25:
      0%    = 25969.25 (top)
      23.6% = 25632.95
      38.2% = 25424.90
      50.0% = 25256.75
      61.8% = 25088.57  ← matches observed 25087 level
      76.4% = 24880.55
      100%  = 24544.25 (bottom)
    """
```

#### 2. Swing Detector

```python
@dataclass
class SwingPoint:
    price: float
    timestamp: float    # unix seconds
    type: str           # 'high' | 'low'
    timeframe: str      # 'daily' | '4h' | '1h' | '15m'
    strength: float     # 0-1
    volume_at_level: float

SWING_LOOKBACK = {
    'daily': 5,
    '4h': 6,
    '1h': 8,
    '15m': 10
}

def _detect_swings(ohlcv: pd.DataFrame, timeframe: str) -> List[SwingPoint]:
    """
    Fractional pivot detection:
    - Swing high = bar where High > High of N bars before AND N bars after
    - Swing low  = bar where Low < Low of N bars before AND N bars after
    
    IMPORTANT: Use vectorized numpy operations, not row-by-row loops.
    
    Algorithm:
        highs = ohlcv['high'].values
        lows = ohlcv['low'].values
        n = SWING_LOOKBACK[timeframe]
        
        For each bar i from n to len-n:
            if highs[i] == max(highs[i-n:i+n+1]):
                → swing high at price=highs[i]
            if lows[i] == min(lows[i-n:i+n+1]):
                → swing low at price=lows[i]
    
    Edge case: equal highs/lows both count (tighter cluster later).
    """
```

#### 3. Temporal Window

```python
WINDOW_CONFIG = {
    'daily': {'max_age_days': 180, 'half_life_days': 60},
    '4h':    {'max_age_days': 30,  'half_life_days': 10},
    '1h':    {'max_age_days': 7,   'half_life_days': 3},
    '15m':   {'max_age_days': 2,   'half_life_days': 0.5},
}

def _apply_temporal_window(lines, current_time) -> list:
    """
    recency_weight = exp(-0.693 * age_hours / (half_life_days * 24))
    Drop lines exceeding max_age_days entirely.
    Attach recency_weight to each surviving line.
    """
```

#### 4. DBSCAN Clusterer

```python
@dataclass
class StructuralLevel:
    price_center: float
    price_upper: float
    price_lower: float
    confidence: float       # 0-1 composite
    num_lines: int
    timeframes: List[str]
    dominant_type: str      # 'support' | 'resistance' | 'pivot'
    nearest_fib: dict       # {'label': 'Fib 61.8%', 'distance_ticks': 5.3}
    fib_reinforced: bool
    newest_touch: float     # timestamp
    oldest_touch: float

def _cluster_levels(lines, fib_levels, eps_ticks=8, min_samples=2):
    """
    Uses sklearn.cluster.DBSCAN on 1D price array.
    
    Distance metric: price distance in ticks (price_diff / tick_size).
    eps = eps_ticks * tick_size
    
    Confidence scoring per cluster:
        base = cluster_size / max_cluster_size
             * mean(recency_weights)
             * timeframe_diversity_bonus
             * fib_proximity_bonus
    
    Timeframe diversity bonus:
        1 TF → 1.0x, 2 TFs → 1.3x, 3 TFs → 1.6x, 4 TFs → 2.0x
    
    Fib proximity bonus (KEY — reinforcement scoring):
        Within 4 ticks of fib level → 1.5x
        Within 8 ticks of fib level → 1.25x
        Beyond 8 ticks → 1.0x (standalone sub-level)
    
    dominant_type:
        >66% swing lows → 'support'
        >66% swing highs → 'resistance'
        Mixed → 'pivot'
    
    CRITICAL: Use DBSCAN from sklearn, NOT KMeans.
    KMeans forces equal-size clusters. DBSCAN finds natural density clusters
    and marks isolated points as noise (which is what we want — isolated
    swing points that don't converge with anything get filtered out).
    
    from sklearn.cluster import DBSCAN
    prices_1d = np.array([line.price for line in lines]).reshape(-1, 1)
    db = DBSCAN(eps=eps_ticks * tick_size, min_samples=min_samples)
    labels = db.fit_predict(prices_1d)
    """
```

#### 5. Overlay Renderer

```python
def _render_overlay(ohlcv_15m, fib_levels, raw_lines, clustered_levels,
                    output_path, figsize=(20, 10), dpi=150):
    """
    Generates a PNG chart for visual supervision.
    
    USE matplotlib ONLY (no mplfinance dependency for core rendering).
    mplfinance is optional — fall back to OHLC line plot if not installed.
    
    Layers (back to front):
    1. Candlestick/OHLC base chart (15m bars, full dataset)
    2. Fib levels: thick dashed cyan lines, 50% in yellow
       Range top/bottom: thick dashed white
    3. Raw swing lines: thin, color-coded by TF, alpha by recency
       Daily = blue (alpha 0.9), 4H = green (0.7),
       1H = orange (0.5), 15m = red dotted (0.3)
    4. Clustered zones: shaded horizontal bands (axhspan)
       Blue = support, Red = resistance, Purple = pivot
       Fib-reinforced clusters get bold outline
    5. Right-side labels: confidence % + fib association
    
    Title: "Level Detection Overlay — {date}"
    Subtitle: "{instrument} | Range: {top} - {bottom}"
    """
```

#### 6. JSON Exporter

```python
def _export_levels(fib_levels, clustered_levels, range_config, output_path):
    """
    Writes price_levels.json with two-tier hierarchy:
    
    {
        "generated_at": "2026-03-06T...",
        "instrument": "MNQ_MAR26",
        "range": {"top": 25969.25, "bottom": 24544.25},
        "fib_anchors": [
            {"price": 25969.25, "ratio": 0.0, "label": "Range Top", "tier": "anchor"},
            {"price": 25632.95, "ratio": 0.236, "label": "Fib 23.6%", "tier": "anchor"},
            ...
        ],
        "sub_levels": [
            {
                "price_center": 25087.25,
                "price_upper": 25089.00,
                "price_lower": 25085.50,
                "confidence": 0.92,
                "type": "pivot",
                "tier": "sub_level",
                "timeframes": ["4h", "1h"],
                "nearest_fib": {"label": "Fib 61.8%", "distance_ticks": 5.3},
                "fib_reinforced": true
            }
        ]
    }
    """
```

### Data Loading (reads from existing ATLAS)

```python
def _load_atlas_data(data_dir: str) -> dict:
    """
    Reads OHLCV parquet files from the project's ATLAS directory structure.
    
    Expected layout:
        {data_dir}/1D/YYYY_MM.parquet    (or YYYYMMDD.parquet)
        {data_dir}/4h/YYYY_MM.parquet
        {data_dir}/1h/YYYY_MM.parquet
        {data_dir}/15m/YYYY_MM.parquet    (or '15min')
    
    Each parquet has columns: timestamp, open, high, low, close, volume
    Timestamp is float (unix seconds) or int64.
    
    TF directory name mapping (handle both conventions):
        'daily' or '1D' → daily
        '4h' → 4h
        '1h' → 1h  
        '15m' or '15min' → 15m
    
    Returns dict: {'daily': pd.DataFrame, '4h': df, '1h': df, '15m': df}
    Missing TFs return None (not error).
    
    IMPORTANT: Use glob to find parquet files. Concatenate all monthly
    files per TF. Sort by timestamp. Drop duplicates.
    """
```

---

## FILE 2: `range_config.json` (in project root)

```json
{
    "instrument": "MNQ_MAR26",
    "range_top": 25969.25,
    "range_bottom": 24544.25,
    "set_date": "2026-03-06",
    "notes": "Major swing range Sep 2025 - Mar 2026"
}
```

---

## FILE 3: `tests/test_level_detector.py`

### Test Cases Required

```python
"""
Tests for core/level_detector.py
Run: pytest tests/test_level_detector.py -v
"""

class TestFibGenerator:
    def test_known_fib_values(self):
        """Verify fib levels match hand-calculated values from the chart.
        Range: 24544.25 → 25969.25
        Expected 61.8% = 25969.25 - (25969.25 - 24544.25) * 0.618 = 25088.55
        Allow ±0.5 tolerance for float rounding."""
    
    def test_fib_count(self):
        """Should produce exactly 7 levels (including top and bottom)."""
    
    def test_fib_ordering(self):
        """Levels should be sorted descending (top → bottom)."""

class TestSwingDetector:
    def test_obvious_swing_high(self):
        """Create synthetic data with a clear peak. Detector must find it."""
    
    def test_obvious_swing_low(self):
        """Create synthetic data with a clear trough. Detector must find it."""
    
    def test_no_swings_in_trend(self):
        """Monotonically increasing prices should produce no swing highs
        (except possibly at boundaries)."""
    
    def test_lookback_scaling(self):
        """Daily (lookback=5) should find fewer swings than 15m (lookback=10)
        on the same data."""

class TestTemporalWindow:
    def test_fresh_lines_survive(self):
        """Lines from 1 hour ago should have recency_weight close to 1.0."""
    
    def test_old_lines_decay(self):
        """Lines at half_life should have recency_weight ≈ 0.5."""
    
    def test_expired_lines_dropped(self):
        """Lines beyond max_age_days should be removed entirely."""

class TestDBSCANClustering:
    def test_converging_lines_cluster(self):
        """3 lines within 2 ticks of each other should form 1 cluster."""
    
    def test_isolated_lines_filtered(self):
        """A single line 50 ticks from everything else should be noise."""
    
    def test_fib_proximity_bonus(self):
        """Cluster near a fib level should have higher confidence than
        identical cluster far from any fib level."""
    
    def test_multi_tf_bonus(self):
        """Cluster with lines from 3 timeframes should score higher than
        cluster with lines from 1 timeframe."""

class TestExporter:
    def test_json_schema(self):
        """Exported JSON must have 'fib_anchors' and 'sub_levels' keys."""
    
    def test_fib_anchors_never_empty(self):
        """Even with no OHLCV data, fib_anchors should contain 7 levels."""

class TestGetLevelsForPrice:
    def test_returns_nearest(self):
        """For price=25090, nearest fib should be 61.8% at 25088.57."""
    
    def test_direction_labeling(self):
        """Levels above price labeled 'above', below labeled 'below'."""
```

---

## FILE 4: `scripts/run_level_detector.py`

CLI wrapper for quick testing:

```python
"""
Quick-run script for level detection.

Usage:
    python scripts/run_level_detector.py
    python scripts/run_level_detector.py --data DATA/ATLAS --output output/
    python scripts/run_level_detector.py --top 26000 --bottom 24500
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.level_detector import LevelDetector

def main():
    parser = argparse.ArgumentParser(description='Run level detection')
    parser.add_argument('--config', default='range_config.json')
    parser.add_argument('--top', type=float, default=None)
    parser.add_argument('--bottom', type=float, default=None)
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--output', default='output/')
    parser.add_argument('--overlay-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    
    if args.top and args.bottom:
        detector = LevelDetector(range_top=args.top, range_bottom=args.bottom)
    else:
        detector = LevelDetector.from_range_config(args.config)
    
    result = detector.run(data_dir=args.data, output_dir=args.output)
    
    print(f"\nFib Anchors: {len(result['fib_levels'])}")
    print(f"Sub-Levels:  {len(result['sub_levels'])}")
    print(f"Overlay:     {result['overlay_path']}")
    print(f"JSON:        {result['json_path']}")
    
    # Print level table
    print(f"\n{'Price':>12} {'Type':>12} {'Confidence':>10} {'TFs':>10} {'Fib?':>6}")
    print("-" * 55)
    for level in result['all_levels']:
        print(f"{level['price']:>12.2f} {level['type']:>12} "
              f"{level.get('confidence', 1.0):>10.2f} "
              f"{','.join(level.get('timeframes', ['-'])):>10} "
              f"{'Y' if level.get('fib_reinforced') else '-':>6}")

if __name__ == '__main__':
    main()
```

---

## IMPLEMENTATION CONSTRAINTS

1. **ONE FILE for the module** (`core/level_detector.py`). All dataclasses,
   functions, and the main class in one file. No sub-package.

2. **DO NOT modify any existing files.** This module is additive only.
   No changes to `quantum_field_engine.py`, `three_body_state.py`,
   `timeframe_belief_network.py`, or any other existing file.

3. **Read parquet directly** using `pd.read_parquet()`. The ATLAS directory
   structure already has the data. No CSV conversion step.

4. **Vectorized numpy** for swing detection. No Python for-loops over bars.
   Use `np.lib.stride_tricks.sliding_window_view` or rolling max/min.

5. **DBSCAN from sklearn** (already installed). Not KMeans. 1D clustering
   on price values.

6. **matplotlib only** for overlay. No external chart libraries required.
   mplfinance is nice-to-have but wrap in try/except.

7. **Type hints** on all public methods. Docstrings on all classes.

8. **No GPU code.** This is pure CPU — numpy + sklearn + matplotlib.
   The datasets are small (hundreds of swing points, not millions of bars).

9. **Fail gracefully** if a timeframe directory doesn't exist in ATLAS.
   Log a warning, continue with available data.

10. **JSON output must be valid** and parseable by `json.load()`.
    Use `json.dump()` with `indent=2` for readability.

---

## IMPLEMENTATION ORDER

Build and test in this sequence:

```
Step 1: Dataclasses (FibLevel, SwingPoint, StructuralLevel)
Step 2: _generate_fib_levels() + test
Step 3: _detect_swings() + test (vectorized)
Step 4: _apply_temporal_window() + test
Step 5: _cluster_levels() with fib proximity bonus + test
Step 6: _load_atlas_data() (parquet reader)
Step 7: _export_levels() + test (JSON)
Step 8: _render_overlay() (matplotlib PNG)
Step 9: LevelDetector class (orchestrates all above)
Step 10: get_levels_for_price() (real-time lookup)
Step 11: scripts/run_level_detector.py (CLI)
Step 12: Run on real ATLAS data, visual validation
```

---

## VALIDATION CRITERIA

After implementation, run:

```bash
# Unit tests
pytest tests/test_level_detector.py -v

# Full pipeline on real data
python scripts/run_level_detector.py --data DATA/ATLAS --output output/

# Check outputs exist
ls output/level_overlay.png
ls output/price_levels.json

# Verify JSON is valid
python -c "import json; json.load(open('output/price_levels.json'))"
```

The overlay PNG should show:
- Candlestick chart with 15m bars
- 7 fib levels as dashed horizontal lines
- Clustered sub-levels as shaded bands
- Fib-reinforced clusters highlighted with bold outline
- Right-side confidence labels

The JSON should have:
- 7 entries in `fib_anchors`
- Variable number of `sub_levels` (depends on data)
- Every sub_level has `nearest_fib` and `fib_reinforced` fields

---

## AFTER IMPLEMENTATION (Phase 2 — separate task)

Once level_detector.py is validated visually:

1. Add `compute_state_anchored(levels)` to QuantumFieldEngine
2. Wire `LevelRelativeContext` into TimeframeBeliefNetwork workers  
3. Replace z_score sign fallback in direction cascade
4. Feature-flag: `use_structural_levels: bool = False`

DO NOT do Phase 2 yet. Get the detector working standalone first.
