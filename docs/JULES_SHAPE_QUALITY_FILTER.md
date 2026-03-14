# JULES Spec: Shape Quality Filter — Unifying Analysis & Building

## Problem

We have two sides of the same coin:
- **Analyzer** (`seed_pattern_analyzer.py`): classifies waveforms into shapes (V_REVERSAL, IMPULSE, etc.)
  and measures quality (efficiency, MFE timing, retracement, monotonicity)
- **Builder** (`shape_primitive_builder.py`): finds ~100K raw ZigZag swings across 7 TFs, clusters them all

Right now the builder clusters everything — good swings, noise, chop. The analyzer knows what
quality looks like (from studying 255 human seeds) but only runs post-hoc. The fix: make the
analyzer's classification logic a **pre-filter** in the builder pipeline so only human-grade
swings reach UMAP/HDBSCAN.

Human marks 255 seeds → analyzer learns what "good" looks like → builder finds 100K candidates →
classifier keeps only the ones that match human quality → UMAP/HDBSCAN clusters the good ones →
tighter, more meaningful primitives.

## Architecture

```
Builder pipeline (current):
  ZigZag (7 TFs) → 100K raw swings → normalize → UMAP → HDBSCAN → primitives

Builder pipeline (proposed):
  ZigZag (7 TFs) → 100K raw swings
    → classify_shape() on each → shape label + features
    → quality_score() on each → 0.0-1.0 score
    → FILTER: keep score >= threshold (default 0.3)
    → ~20K-40K quality swings
    → normalize → UMAP → HDBSCAN → primitives
```

## Phase 1: Extract Shared Shape Classifier

### New file: `tools/research/shape_classifier.py`

Extract from `seed_pattern_analyzer.py`:
- `classify_shape(prices, entry_idx)` → (shape_name, confidence, features_dict)
- `quality_score(shape, confidence, features)` → float 0.0-1.0 (**NEW**)

The classifier stays identical to what the analyzer uses. The quality scorer is new,
calibrated from human seed statistics.

### Quality Score Formula

Based on deep dive findings (255 human seeds):

```python
def quality_score(shape: str, confidence: float, features: dict) -> float:
    """Score a classified waveform 0.0-1.0 based on human-derived quality metrics.

    Calibrated from 255 human-marked seeds across 1h/15m/5m.
    """
    eff = features.get('efficiency', 0)
    abs_net = features.get('abs_net', 0)
    mono = features.get('monotonic_frac', 0)
    retr = features.get('retracement', 0)
    peak_pct = features.get('peak_at_pct', 0.5)

    # Base score from shape type (human frequency = prior for quality)
    shape_priors = {
        'V_REVERSAL':       0.7,   # 56% of human marks — dominant pattern
        'IMPULSE':          0.65,  # 30% of human marks — fast directional
        'RAMP':             0.5,   # 13% — steady grinds
        'TREND_CONTINUATION': 0.4, # continuation of existing
        'SIGMOID':          0.35,  # slow start, fast finish
        'COMPRESSION':      0.25,  # coiling — some value, but noisy
        'EXHAUSTION':       0.2,   # peak then giveback — marginal
        'FAKEOUT':          0.15,  # rare (3%), hard to trade
        'CHOP':             0.05,  # noise
        'OTHER':            0.2,
        'UNKNOWN':          0.0,
    }
    base = shape_priors.get(shape, 0.2)

    # Efficiency bonus (human avg: 0.78 at 15m, 0.38 = chop)
    eff_bonus = max(0, min(0.3, (eff - 0.2) * 0.5))

    # Net move bonus (filter out tiny wiggles)
    net_bonus = max(0, min(0.15, (abs_net - 10) * 0.005))

    # Late MFE bonus (human V_REVs have late peak — last 30%)
    late_mfe_bonus = 0.1 if peak_pct > 0.7 and shape == 'V_REVERSAL' else 0

    # Monotonicity bonus for IMPULSE/RAMP
    mono_bonus = 0
    if shape in ('IMPULSE', 'RAMP') and mono > 0.6:
        mono_bonus = (mono - 0.6) * 0.3

    # Penalty for high retracement (gave back the move)
    retr_penalty = max(0, (retr - 0.5) * 0.3) if shape not in ('FAKEOUT', 'EXHAUSTION') else 0

    score = base + eff_bonus + net_bonus + late_mfe_bonus + mono_bonus - retr_penalty
    return max(0.0, min(1.0, score))
```

### Quality Tiers

| Tier | Score Range | Expected % of Auto-Swings | Action |
|------|-------------|---------------------------|--------|
| GOLD | >= 0.7 | ~5-10% | Always keep, weight higher in HDBSCAN |
| SILVER | 0.5-0.7 | ~15-25% | Keep |
| BRONZE | 0.3-0.5 | ~20-30% | Keep (borderline) |
| NOISE | < 0.3 | ~40-50% | Drop before UMAP |

Default filter threshold: **0.3** (drop NOISE tier).
CLI flag: `--quality-threshold 0.3` to adjust.

## Phase 2: Wire Into Builder Pipeline

### Modify: `tools/shape_primitive_builder.py`

After ZigZag detects raw swings and before normalization:

```python
# Step 1c: Shape quality filter
from tools.research.shape_classifier import classify_shape, quality_score

print(f"  Classifying {len(seeds):,} raw swings...")
filtered_seeds = []
shape_counts = defaultdict(int)
tier_counts = {'GOLD': 0, 'SILVER': 0, 'BRONZE': 0, 'NOISE': 0}

for seed in tqdm(seeds, desc='Classifying'):
    waveform = seed['waveform_close']
    entry_idx = seed['lookback_bars']
    shape, conf, features = classify_shape(waveform, entry_idx)

    score = quality_score(shape, conf, features)
    seed['shape'] = shape
    seed['shape_confidence'] = conf
    seed['quality_score'] = score

    if score >= args.quality_threshold:
        filtered_seeds.append(seed)

    # Stats
    shape_counts[shape] += 1
    if score >= 0.7: tier_counts['GOLD'] += 1
    elif score >= 0.5: tier_counts['SILVER'] += 1
    elif score >= 0.3: tier_counts['BRONZE'] += 1
    else: tier_counts['NOISE'] += 1

print(f"  Shapes: {dict(shape_counts)}")
print(f"  Tiers: {dict(tier_counts)}")
print(f"  Kept: {len(filtered_seeds):,} / {len(seeds):,} "
      f"({len(filtered_seeds)/len(seeds)*100:.1f}%)")

seeds = filtered_seeds
```

Human seeds always pass (they are the quality reference):
```python
if seed.get('source') == 'human':
    score = max(score, 0.7)  # Human seeds always GOLD
```

### New CLI flags for builder:
- `--quality-threshold 0.3` — minimum quality score to keep (default 0.3)
- `--no-filter` — disable quality filter (cluster everything, legacy behavior)

## Phase 3: Enrich Primitives with Shape Labels

Each `ShapePrimitive` gains:

```python
@dataclass
class ShapePrimitive:
    # ... existing fields ...
    dominant_shape: str          # Most common shape label among members
    shape_distribution: Dict[str, int]  # {shape: count}
    mean_quality_score: float    # Average quality score of members
    quality_tier: str            # GOLD/SILVER/BRONZE based on mean score
```

This makes primitives self-describing: "Primitive #7 is a V_REVERSAL-dominant cluster
at 15m, GOLD tier, 0.82 mean quality."

## Phase 4: Feedback Loop — Calibrate From Expanding Seed Library

As the user marks more seeds (different dates, different regimes):
1. Run analyzer on new seeds → updated shape statistics
2. Optionally: `--recalibrate` flag on builder that recomputes `shape_priors` and
   score weights from ALL available human seeds (not hardcoded)
3. Save calibration to `checkpoints/shape_quality_calibration.json`

```python
def calibrate_from_human_seeds(seeds_dir, data_dir):
    """Compute shape priors + quality thresholds from human seeds."""
    # Load all human seeds, classify each
    # Count shape frequencies → shape_priors
    # Compute per-shape efficiency/net/mono distributions → threshold calibration
    # Save to JSON
```

This closes the loop: more human seeds → better quality filter → better primitives.

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `tools/research/shape_classifier.py` | **NEW** | ~120 |
| `tools/shape_primitive_builder.py` | **MODIFY** — add filter step, new CLI flags, enrich primitives | ~80 |
| `tools/seed_pattern_analyzer.py` | **MODIFY** — import from shared classifier instead of inline | ~10 |

## Implementation Order

1. Create `tools/research/shape_classifier.py` (extract + add quality_score)
2. Update `seed_pattern_analyzer.py` to import from shared module
3. Wire filter into `shape_primitive_builder.py`
4. Add shape labels to ShapePrimitive dataclass
5. Add `--recalibrate` for future seed library expansion

## Verification

1. Run analyzer before/after refactor → identical output (no regression)
2. Run builder on ATLAS_1WEEK with filter → fewer seeds, check tier distribution
3. Run builder on ATLAS_1WEEK without filter (`--no-filter`) → same as before
4. Compare UMAP plots: filtered should show tighter, more distinct clusters
5. Eventually: train with filtered primitives → compare IS/OOS to unfiltered baseline
