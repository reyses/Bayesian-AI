# Level Detection & Overlay System — Implementation Spec

**Project:** BayesianBridge MNQ Trading System  
**Module:** Multi-Timeframe Level Detector with Visual Supervision  
**Target:** NinjaTrader 8 integration (C# downstream), Python analysis pipeline  
**Date:** March 5, 2026  

---

## 1. OBJECTIVE

Build a rule-based multi-timeframe level detection system that:
1. Takes TWO manual inputs: range top (swing high) and range bottom (swing low)
2. Auto-generates Fibonacci structural levels from the range
3. Detects sub-levels via swing detection across 4 lower timeframes
4. Clusters sub-levels and scores them (especially where they reinforce fib zones)
5. Applies temporal windowing to decay stale sub-levels
6. Outputs a visual overlay image for human supervision
7. Exports validated levels as JSON for the physics engine

**Human effort: Draw two lines on the daily chart. System fills in everything else.**

**This is NOT a CNN/ML module.** Pure algorithmic detection + clustering + Fibonacci derivation.

---

## 2. SYSTEM ARCHITECTURE

```
HUMAN INPUT: Range Top + Range Bottom (2 prices)
        ↓
┌─────────────────────────┐
│  FIB LEVEL GENERATOR    │  Compute 23.6%, 38.2%, 50%, 61.8%, 76.4%
│                         │  These are PERMANENT structural anchors
└──────────┬──────────────┘
           ↓
Market Data (OHLCV per timeframe)
        ↓
┌─────────────────────────┐
│  SWING DETECTOR         │  Per-timeframe swing high/low detection
│  (Daily → 4H → 1H → 15m)│
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  HORIZONTAL AGGREGATOR  │  Draw horizontal line at each swing point
│                         │  Tag with: price, timeframe, timestamp, type
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  TEMPORAL WINDOWER      │  Apply recency decay, remove expired sub-levels
│                         │  NOTE: Fib levels NEVER decay
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  DBSCAN CLUSTERER       │  Cluster converging lines into level zones
│                         │  Fib-proximity bonus: sub-levels near fib
│                         │  zones get confidence boost (reinforcement)
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  LEVEL MERGER           │  Merge fib anchors + clustered sub-levels
│                         │  into unified level hierarchy
└──────────┬──────────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
[OVERLAY PNG]  [levels.json]
```

---

## 3. MODULE SPECIFICATIONS

### 3.1 Fib Level Generator

**File:** `fib_generator.py`

**Purpose:** Given two prices (range top and bottom), compute Fibonacci structural levels that serve as permanent anchors for the entire system.

**Input:**
```python
def generate_fib_levels(
    range_top: float,          # e.g., 25969.25 (swing high)
    range_bottom: float,       # e.g., 24544.25 (swing low)
    custom_ratios: List[float] = None  # override defaults
) -> List[FibLevel]
```

**Default Fibonacci ratios:**
```python
FIB_RATIOS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0]
```

**Calculation:**
```
fib_price = range_bottom + (range_top - range_bottom) * (1 - ratio)
```
Note: Standard fib retracement measures FROM the top DOWN.

**Output:**
```python
@dataclass
class FibLevel:
    price: float               # computed fib price
    ratio: float               # 0.236, 0.382, etc.
    label: str                 # "23.6%", "38.2%", etc.
    tier: str                  # 'anchor' (always)
    type: str                  # 'range_top' | 'range_bottom' | 'fib_retracement'
```

**Example output for MNQ range 24544.25 → 25969.25:**

| Ratio | Price | Role |
|-------|-------|------|
| 0.0% | 25969.25 | Range top (your upper line) |
| 23.6% | 25632.95 | Fib resistance |
| 38.2% | 25424.90 | Fib level |
| 50.0% | 25256.75 | Mid-range pivot |
| 61.8% | 25088.57 | Key fib support (matches your 25087 level) |
| 76.4% | 24880.55 | Deep retracement |
| 100.0% | 24544.25 | Range bottom (your lower line) |

**Range config file:** `range_config.json`
```json
{
  "instrument": "MNQ_MAR26",
  "range_top": 25969.25,
  "range_bottom": 24544.25,
  "set_date": "2026-03-05",
  "notes": "Major swing range Sep 2025 - Mar 2026",
  "custom_ratios": null
}
```

**Update cycle:** You update `range_config.json` only when the macro range breaks (price closes beyond top or bottom on daily). Estimated frequency: quarterly or less.

---

### 3.2 Swing Detector

**File:** `swing_detector.py`

**Purpose:** Identify swing highs and swing lows per timeframe.

**Algorithm:** Fractional pivot detection
- A **swing high** = bar where High > High of N bars before AND N bars after
- A **swing low** = bar where Low < Low of N bars before AND N bars after
- Lookback `N` varies by timeframe:

| Timeframe | Lookback (N) | Approx Window |
|-----------|-------------|---------------|
| Daily     | 5 bars      | ~1 week       |
| 4H        | 6 bars      | ~1 day        |
| 1H        | 8 bars      | ~8 hours      |
| 15m       | 10 bars     | ~2.5 hours    |

**Input:**
```python
def detect_swings(
    ohlcv: pd.DataFrame,       # columns: open, high, low, close, volume, timestamp
    timeframe: str,            # 'daily' | '4h' | '1h' | '15m'
    lookback: int = None       # override default N
) -> List[SwingPoint]
```

**Output:**
```python
@dataclass
class SwingPoint:
    price: float               # exact price of the swing
    timestamp: datetime        # when it occurred
    type: str                  # 'high' | 'low'
    timeframe: str             # source timeframe
    strength: float            # 0-1, based on how many bars confirm it
    volume_at_level: float     # average volume of bars at this swing
```

**Edge cases:**
- Equal highs/lows: both count as swing points (creates tighter cluster later)
- Gaps: if a gap jumps over a potential swing, still detect the local extreme

---

### 3.3 Horizontal Aggregator

**File:** `horizontal_aggregator.py`

**Purpose:** Collect all swing points across timeframes into a flat list of horizontal lines.

```python
def aggregate_horizontals(
    swing_points: Dict[str, List[SwingPoint]]  # keyed by timeframe
) -> List[HorizontalLine]
```

**Output:**
```python
@dataclass
class HorizontalLine:
    price: float
    timeframe: str
    timestamp: datetime
    type: str                  # 'high' | 'low'
    strength: float
    volume: float
    age_hours: float           # hours since formation
```

No logic here — just flattening + computing `age_hours` relative to current time.

---

### 3.4 Temporal Windower

**File:** `temporal_window.py`

**Purpose:** Apply recency decay to horizontal lines. Old broken levels fade out.

**Parameters:**
```python
WINDOW_CONFIG = {
    'daily':  {'max_age_days': 180, 'half_life_days': 60},
    '4h':    {'max_age_days': 30,  'half_life_days': 10},
    '1h':    {'max_age_days': 7,   'half_life_days': 3},
    '15m':   {'max_age_days': 2,   'half_life_days': 0.5},
}
```

**Decay formula:**
```
recency_weight = exp(-0.693 * age_hours / (half_life_days * 24))
```
- Lines exceeding `max_age_days` are dropped entirely
- Lines within window get `recency_weight` ∈ (0, 1]

```python
def apply_temporal_window(
    lines: List[HorizontalLine],
    config: dict = WINDOW_CONFIG
) -> List[HorizontalLine]  # with recency_weight attached
```

---

### 3.5 DBSCAN Clusterer

**File:** `level_clusterer.py`

**Purpose:** Cluster converging horizontal lines into structural level zones.

**Algorithm:** DBSCAN with custom distance metric

**Distance metric:** Price distance in ticks (MNQ tick = 0.25)
```
eps = 8 ticks (2 points)   # lines within 8 ticks cluster together
min_samples = 2            # minimum 2 lines to form a level
```

**Cluster confidence scoring:**
```python
confidence = (
    cluster_size / max_cluster_size *          # density: 0-1
    mean(recency_weights) *                     # recency: 0-1
    timeframe_diversity_bonus *                 # multi-TF agreement: 1.0-2.0
    fib_proximity_bonus                         # near a fib level: 1.0-1.5
)
```

**Timeframe diversity bonus:**
```
1 timeframe  → 1.0x
2 timeframes → 1.3x
3 timeframes → 1.6x
4 timeframes → 2.0x
```

**Fib proximity bonus (NEW):**
```
Cluster centroid within 4 ticks of a fib level  → 1.5x (strong reinforcement)
Cluster centroid within 8 ticks of a fib level  → 1.25x (moderate reinforcement)
Cluster centroid >8 ticks from any fib level    → 1.0x (no bonus, standalone sub-level)
```
This is the key integration: sub-levels that independently converge near fib zones
get boosted confidence because two independent methods agree on the same price.

**Output:**
```python
@dataclass
class StructuralLevel:
    price_center: float        # cluster centroid
    price_upper: float         # highest line in cluster
    price_lower: float         # lowest line in cluster
    confidence: float          # 0-1 composite score
    num_lines: int             # how many horizontals converged
    timeframes: List[str]      # which timeframes contributed
    dominant_type: str         # 'support' | 'resistance' | 'pivot'
    newest_touch: datetime     # most recent line in cluster
    oldest_touch: datetime     # oldest line in cluster
```

**`dominant_type` logic:**
- >66% of lines are swing lows → `support`
- >66% of lines are swing highs → `resistance`
- Mixed → `pivot`

```python
def cluster_levels(
    lines: List[HorizontalLine],
    eps_ticks: float = 8,
    min_samples: int = 2
) -> List[StructuralLevel]
```

---

### 3.6 Overlay Renderer

**File:** `level_overlay.py`

**Purpose:** Generate a PNG overlay image for visual supervision.

**Library:** `matplotlib` with `mplfinance` for candlestick rendering

**Layout:**
```
┌─────────────────────────────────────────────┐
│  Title: "Level Detection Overlay — {date}"  │
│  Subtitle: "{instrument} | Range: Top-Bottom"│
├─────────────────────────────────────────────┤
│                                             │
│  Candlestick chart (15m bars, last 24hrs)   │
│                                             │
│  === Range Top (thick white, dashed) =======|  ← YOUR top line
│  === Fib 23.6% (thick cyan, dashed) ========|  ← Auto-generated
│  === Fib 38.2% (thick cyan, dashed) ========|
│  === Fib 50.0% (thick yellow, dashed) ======|
│  === Fib 61.8% (thick cyan, dashed) ========|
│  === Fib 76.4% (thick cyan, dashed) ========|
│  === Range Bottom (thick white, dashed) ====|  ← YOUR bottom line
│                                             │
│  --- 4H swing (medium green, alpha=0.7)  ---|  ← Detected sub-levels
│  --- 1H swing (thin orange, alpha=0.5)   ---|
│  ··· 15m swing (dotted red, alpha=0.3)   ···|
│                                             │
│  ████ Clustered zone (shaded band)     █████|
│  ████ Fib-reinforced zone (bold band)  █████|  ← Cluster near fib = highlighted
│                                             │
│  Confidence labels + fib % on right y-axis  │
│                                             │
├─────────────────────────────────────────────┤
│  Legend | Cluster count | Fib match count   │
└─────────────────────────────────────────────┘
```

**Visual encoding:**

| Element | Style | Meaning |
|---------|-------|---------|
| Thick dashed white line | `linewidth=3.0, linestyle='--', alpha=1.0` | Range top / bottom (YOUR lines) |
| Thick dashed cyan line | `linewidth=2.0, linestyle='--', alpha=0.8` | Fib retracement levels |
| Thick dashed yellow line | `linewidth=2.0, linestyle='--', alpha=0.9` | Fib 50% (mid-range pivot, highlighted) |
| Medium solid green line | `linewidth=1.8, alpha=0.7` | 4H sub-level |
| Thin solid orange line | `linewidth=1.2, alpha=0.5` | 1H sub-level |
| Dotted red line | `linewidth=0.8, linestyle=':', alpha=0.3` | 15m sub-level |
| Shaded horizontal band | `axhspan, alpha=0.15` | Clustered level zone (price_lower to price_upper) |
| Bold shaded band | `axhspan, alpha=0.25, edgecolor='white'` | Fib-reinforced cluster (sub-levels converge near fib) |
| Band color | Blue=support, Red=resistance, Purple=pivot | `dominant_type` |
| Band opacity | Proportional to confidence | Higher confidence = more visible |
| Right-side label | `f"{confidence:.0%} [Fib 61.8%]"` | Confidence + fib association if applicable |

**Decayed lines:** Lines with `recency_weight < 0.3` render with `alpha *= recency_weight` (visually fading)

```python
def render_overlay(
    ohlcv_15m: pd.DataFrame,          # 15m bars for candlestick base
    fib_levels: List[FibLevel],        # permanent fib anchors
    raw_lines: List[HorizontalLine],   # all detected horizontals (pre-cluster)
    levels: List[StructuralLevel],     # clustered output levels
    output_path: str = 'level_overlay.png',
    figsize: tuple = (20, 10),
    dpi: int = 150
) -> str  # returns output_path
```

**Additional overlay options (configurable):**
- `show_raw_lines: bool = True` — toggle individual swing lines
- `show_clusters_only: bool = False` — hide raw, show only zones
- `show_fibs: bool = True` — toggle fib level lines
- `highlight_active: bool = True` — bold outline on levels where price is within 20 ticks
- `annotate_touches: bool = False` — mark where price touched each level
- `highlight_fib_reinforced: bool = True` — bold clusters that align with fibs

---

### 3.7 JSON Exporter

**File:** `level_exporter.py`

**Purpose:** Export unified level hierarchy for consumption by the physics engine.

**Output format:** `price_levels.json`
```json
{
  "generated_at": "2026-03-05T21:30:00Z",
  "instrument": "MNQ_MAR26",
  "range": {
    "top": 25969.25,
    "bottom": 24544.25,
    "set_date": "2026-03-05"
  },
  "fib_anchors": [
    {"price": 25969.25, "ratio": 0.000, "label": "Range Top", "tier": "anchor"},
    {"price": 25632.95, "ratio": 0.236, "label": "Fib 23.6%", "tier": "anchor"},
    {"price": 25424.90, "ratio": 0.382, "label": "Fib 38.2%", "tier": "anchor"},
    {"price": 25256.75, "ratio": 0.500, "label": "Fib 50.0%", "tier": "anchor"},
    {"price": 25088.57, "ratio": 0.618, "label": "Fib 61.8%", "tier": "anchor"},
    {"price": 24880.55, "ratio": 0.764, "label": "Fib 76.4%", "tier": "anchor"},
    {"price": 24544.25, "ratio": 1.000, "label": "Range Bottom", "tier": "anchor"}
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
      "num_lines": 7,
      "nearest_fib": {"label": "Fib 61.8%", "distance_ticks": 5.3},
      "fib_reinforced": true,
      "newest_touch": "2026-03-05T20:15:00Z"
    },
    {
      "price_center": 25105.00,
      "price_upper": 25106.50,
      "price_lower": 25103.75,
      "confidence": 0.58,
      "type": "resistance",
      "tier": "sub_level",
      "timeframes": ["4h"],
      "num_lines": 3,
      "nearest_fib": {"label": "Fib 50.0%", "distance_ticks": 607.0},
      "fib_reinforced": false,
      "newest_touch": "2026-03-05T19:00:00Z"
    }
  ]
}
```

**Two-tier hierarchy:**
- `fib_anchors`: Permanent. Derived from your two lines. Never decay. Physics engine uses these as primary reference frame.
- `sub_levels`: Dynamic. Detected from lower timeframe swings. Decay over time. Provide fractal detail between anchors. `fib_reinforced: true` = this sub-level independently confirms a fib zone.

```python
def export_levels(
    fib_levels: List[FibLevel],
    clustered_levels: List[StructuralLevel],
    range_config: dict,
    output_path: str = 'price_levels.json'
) -> str
```

---

## 4. PIPELINE ORCHESTRATOR

**File:** `level_pipeline.py`

**Purpose:** Run the full fib generation → swing detection → cluster → render → export pipeline.

```python
def run_level_detection(
    range_config_path: str,            # range_config.json (your 2 lines)
    data_dir: str,                     # path to OHLCV CSVs per timeframe
    output_dir: str = './output/',
    config: dict = None                # override default params
) -> dict:
    """
    Returns:
        {
            'fib_levels': List[FibLevel],
            'sub_levels': List[StructuralLevel],
            'overlay_path': str,
            'json_path': str,
            'stats': {
                'total_swings_detected': int,
                'total_lines_after_window': int,
                'total_clusters': int,
                'fib_reinforced_count': int,
                'levels_by_type': {'support': N, 'resistance': N, 'pivot': N}
            }
        }
    """
```

**CLI interface:**
```bash
# Full pipeline (your 2 lines + auto everything else)
python level_pipeline.py --range ./range_config.json --data ./market_data/ --output ./output/

# Quick overlay (skip export)
python level_pipeline.py --range ./range_config.json --data ./market_data/ --overlay-only

# Export only (skip rendering)
python level_pipeline.py --range ./range_config.json --data ./market_data/ --export-only

# Update range (when macro range breaks)
python level_pipeline.py --set-range --top 26200.00 --bottom 24544.25
```

---

## 5. DATA FORMAT

### Input OHLCV CSVs

One file per timeframe in `data_dir/`:
```
data/
├── daily.csv
├── 4h.csv
├── 1h.csv
└── 15m.csv
```

CSV format:
```csv
timestamp,open,high,low,close,volume
2026-03-05T08:00:00,25065.25,25070.50,25058.00,25068.75,1523
```

### Range Config File (YOUR INPUT — just 2 prices)

`range_config.json`:
```json
{
  "instrument": "MNQ_MAR26",
  "range_top": 25969.25,
  "range_bottom": 24544.25,
  "set_date": "2026-03-05",
  "notes": "Major swing range Sep 2025 - Mar 2026"
}
```

**This is the only file you ever manually edit.** Update when the macro range breaks.

---

## 6. DEPENDENCIES

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3     # DBSCAN
matplotlib>=3.7
mplfinance>=0.12      # candlestick charts
```

No ML/deep learning frameworks needed.

---

## 7. CONFIGURATION

**File:** `config.yaml`
```yaml
instrument: MNQ_MAR26
tick_size: 0.25
range_config_path: ./range_config.json  # YOUR 2 lines

fibonacci:
  ratios: [0.0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0]

swing_detection:
  daily:  {lookback: 5}
  4h:     {lookback: 6}
  1h:     {lookback: 8}
  15m:    {lookback: 10}

temporal_window:
  daily:  {max_age_days: 180, half_life_days: 60}
  4h:     {max_age_days: 30,  half_life_days: 10}
  1h:     {max_age_days: 7,   half_life_days: 3}
  15m:    {max_age_days: 2,   half_life_days: 0.5}

clustering:
  eps_ticks: 8
  min_samples: 2
  timeframe_diversity_bonus: [1.0, 1.3, 1.6, 2.0]
  fib_proximity:
    strong_ticks: 4       # within 4 ticks = 1.5x bonus
    moderate_ticks: 8     # within 8 ticks = 1.25x bonus
    strong_bonus: 1.5
    moderate_bonus: 1.25

overlay:
  figsize: [20, 10]
  dpi: 150
  show_raw_lines: true
  show_fibs: true
  show_clusters_only: false
  highlight_active: true
  highlight_fib_reinforced: true
  active_proximity_ticks: 20
```

---

## 8. TESTING REQUIREMENTS

### Unit Tests
- `test_fib_generator.py` — Verify fib math against known values from your chart
- `test_swing_detector.py` — Known swing points on synthetic OHLCV data
- `test_temporal_window.py` — Verify decay math and expiration
- `test_clusterer.py` — Known clusters + fib-proximity bonus scoring
- `test_exporter.py` — JSON schema validation (two-tier structure)

### Integration Test
- Run full pipeline on 1 week of historical MNQ data
- Visually verify overlay matches known levels
- Confirm JSON output is consumable by physics engine

### Validation Criteria
- Detected levels should explain >80% of visible bounces/rejections on the chart
- Clustered zones should be within 4 ticks of manually identified levels
- No false levels in "empty space" where price passes through without interaction

---

## 9. FUTURE INTEGRATION POINTS

This module outputs `price_levels.json` which feeds into:

1. **QuantumFieldEngine** — `compute_state_anchored(levels)` method (to be built)
   - Z-scores computed relative to nearest level instead of regression bands
   - Forces anchored to structural levels

2. **Pattern CNN (Phase 2)** — Levels become spatial context input
   - Distance-to-level features for the geometric path
   - Level type (support/resistance/pivot) as categorical input

3. **WaveRider Execution** — Level proximity as trade filter
   - Suppress entries in empty space between levels
   - Boost confidence for entries at/near detected levels

**Do NOT build these integration points yet.** This module is standalone. Integration happens after visual validation confirms detection quality.

---

## 10. IMPLEMENTATION ORDER

```
Step 1: fib_generator.py + tests (trivial — just math)
Step 2: swing_detector.py + tests
Step 3: horizontal_aggregator.py
Step 4: temporal_window.py + tests
Step 5: level_clusterer.py + fib-proximity bonus + tests
Step 6: level_overlay.py (visual output with fib lines + sub-levels)
Step 7: level_exporter.py + JSON schema (two-tier output)
Step 8: level_pipeline.py (orchestrator + CLI)
Step 9: Integration test on real MNQ data
Step 10: Visual validation with you (human-in-the-loop)
```

**Human input required:** Create `range_config.json` with two prices before Step 9.

**Estimated effort:** ~500-700 lines of Python total. No ML complexity.
