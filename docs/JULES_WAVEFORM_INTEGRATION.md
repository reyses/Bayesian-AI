# JULES: Waveform Analysis Integration into Orchestrator Workers

## Background

Analysis K (waveform_standalone.py) proved that 193D fractal features predict
segment direction at 70.6% accuracy (+16.5% lift over baseline). The key findings:

1. **4h and 1h timeframes dominate** direction prediction (35% of total importance)
2. **Hurst exponent** is the #1 feature (1h_hurst = 0.042, 4h_hurst = 0.035)
3. **osc_coherence at 4h** is #2 (0.040) — oscillation regularity matters
4. **15m base TF is barely top-10** — context > local state for direction
5. **Lift concentrates on ambiguous shapes** (V_SHAPE +16%, INVERTED_V +19%, FLAT +13%)

Current workers already use 16D features + logistic regression for direction.
This spec integrates the waveform findings to **re-weight features and add
shape-aware conviction modulation**.

---

## Part 1: TF Weight Recalibration

### File: `training/timeframe_belief_network.py`

### Current State
TF weights are hand-tuned constants (line ~922 in `get_belief()`):
```
1h: 4.0, 30m: 3.5, 15m: 3.0, 5m: 2.5, 3m: 2.0, 1m: 1.5, ...
```

### Change
Update TF weights to reflect Analysis K feature importance rankings:

| TF | Current Weight | New Weight | Rationale |
|----|---------------|------------|-----------|
| 4h | (not used) | 5.0 | Highest direction importance (0.182 total) |
| 1h | 4.0 | 4.5 | Second highest (0.168 total) |
| 30m | 3.5 | 3.0 | Moderate importance (0.087) |
| 15m | 3.0 | 3.0 | Anchor — keep unchanged |
| 5m | 2.5 | 2.0 | Low individual importance |
| 3m | 2.0 | 1.5 | Low importance |
| 1m | 1.5 | 1.0 | Minimal importance |
| 30s | 1.0 | 0.5 | Noise-dominated |
| 15s | 0.5 | 0.25 | Noise-dominated |
| 5s | 0.25 | 0.1 | Noise-dominated |
| 1s | 0.1 | 0.05 | Noise-dominated |

**Note**: If 4h worker doesn't exist yet, it needs to be added to the worker
pool. Check `TIMEFRAMES_SECONDS` — if 14400 (4h) is missing, add it.

### Validation
- Run IS forward pass before/after
- Path conviction distribution should shift (more weight on slower TFs)
- Direction accuracy on oracle trades should improve

---

## Part 2: Hurst-Weighted Conviction Modulation

### File: `training/timeframe_belief_network.py` → `TimeframeWorker._analyze()`

### Current State
Conviction = `|dir_prob - 0.5| * 2.0`, modulated by DNA agreement + price awareness.
Hurst exponent (feature[8]) is used in clustering but NOT in conviction scoring.

### Change
Add Hurst-based conviction scaling AFTER DNA modulation:

```python
# In _analyze(), after DNA modulation of conviction:
hurst = feat[8]  # self_hurst, already in [0, 1]

# High Hurst (>0.5) = trending = direction more reliable
# Low Hurst (<0.5) = mean-reverting = direction less reliable
# Scale: 0.7x at Hurst=0.3 to 1.3x at Hurst=0.7
hurst_scale = 0.7 + (hurst - 0.3) * (0.6 / 0.4)  # linear interpolation
hurst_scale = max(0.7, min(1.3, hurst_scale))       # clamp [0.7, 1.3]

conviction *= hurst_scale
```

### Rationale
Analysis K showed 1h_hurst and 4h_hurst are the #1 and #3 features for
direction prediction. When Hurst is high (trending), direction predictions
are more reliable → boost conviction. When Hurst is low (choppy), direction
is ambiguous → discount conviction.

### Validation
- Compare IS WR before/after
- Expect: fewer trades (low-Hurst signals discounted below threshold)
- Expect: higher WR on remaining trades (high-Hurst signals boosted)

---

## Part 3: osc_coherence Gate in Direction Blend

### File: `training/timeframe_belief_network.py` → `TimeframeWorker._analyze()`

### Current State
Direction blend: 50% ML logistic + 50% physics mean-reversion.
Playbook adds 30% if >= 5 samples.

### Change
Add oscillation coherence (feature[15]) as a blend weight modifier:

```python
osc_coh = feat[15]  # oscillation_coherence, [0, 1]

# High osc_coh = regular cycles = physics (mean-reversion) more reliable
# Low osc_coh = irregular = ML (learned patterns) more reliable
ml_weight = 0.5 - 0.15 * osc_coh    # 0.50 → 0.35 at high coherence
phys_weight = 0.5 + 0.15 * osc_coh  # 0.50 → 0.65 at high coherence

dir_prob = ml_weight * p_long_ml + phys_weight * p_long_phys
```

### Rationale
Analysis K showed 4h_osc_coh is the #2 feature (importance = 0.040). When
oscillation coherence is high, the market is cycling regularly — physics-based
mean-reversion signals are more trustworthy. When osc_coh is low, the market
is in a non-regular regime — lean more on learned ML patterns.

### Validation
- Check direction accuracy separately for high-osc vs low-osc segments
- Expect: physics-heavy blend better in high-osc, ML-heavy better in low-osc

---

## Part 4: Shape-Aware Direction Confidence (New Gate)

### Files:
- `training/orchestrator.py` — new gate between G4 and G3.5
- `tools/waveform_standalone.py` — export shape library (one-time)

### Concept
Analysis K showed that direction prediction lift is concentrated on
**ambiguous shapes** (V_SHAPE, INVERTED_V, FLAT, DOUBLE_DIP). For
**directional shapes** (LINEAR_UP, EXPONENTIAL_DOWN), the shape itself
determines direction — fractal features add nothing.

### Implementation

**Step A: Shape classifier at segment level**

During forward pass, when a new 15m segment forms (every 16 bars at base TF):
1. Extract last 16 bars of 15m close prices
2. Compute delta segment: `seg[i] = close[i] - close[0]`
3. Correlate against seed library (20 primitives, Pearson r)
4. If best |r| >= 0.75 → classified shape; else → NOISE

**Step B: Shape-aware conviction modulation**

```python
DIRECTIONAL_SHAPES = {
    'LINEAR_UP', 'LINEAR_DOWN', 'RAMP_UP', 'RAMP_DOWN',
    'EXPONENTIAL_UP', 'EXPONENTIAL_DOWN', 'LOGARITHMIC_UP', 'LOGARITHMIC_DOWN',
    'STEP_UP', 'STEP_DOWN'
}
AMBIGUOUS_SHAPES = {
    'V_SHAPE', 'INVERTED_V', 'FLAT', 'DOUBLE_DIP', 'DOUBLE_PEAK',
    'TRIANGLE_UP', 'TRIANGLE_DOWN', 'SIGMOID_UP', 'SIGMOID_DOWN'
}

if shape in DIRECTIONAL_SHAPES:
    # Shape determines direction — require AGREEMENT with fractal direction
    shape_dir = 1 if 'UP' in shape else 0
    if shape_dir != fractal_dir:
        conviction *= 0.5  # strong penalty for disagreement
    else:
        conviction *= 1.1  # mild boost for agreement

elif shape in AMBIGUOUS_SHAPES:
    # Shape is ambiguous — fractal features are the ONLY direction signal
    # Require higher conviction threshold for entry
    conviction *= 0.9  # slight discount (shape can't confirm)
    # BUT: this is where the model adds the most value (+16-19% lift)
    # So allow entry if fractal conviction is strong

else:  # NOISE
    conviction *= 0.8  # unclassified shape = less confident
```

**Step C: Export seed library**

Run once to export the 20 seed templates as a .npy file:
```bash
python tools/waveform_standalone.py --export-seeds seeds.npy
```

Or hardcode the templates (they're deterministic parametric functions):
```python
def _build_seed_library(n_bars=16):
    """Generate 20 seed primitives. Same as waveform_standalone.py."""
    t = np.linspace(0, 1, n_bars)
    seeds = {}
    seeds['LINEAR_UP'] = t
    seeds['LINEAR_DOWN'] = -t
    seeds['V_SHAPE'] = np.abs(t - 0.5) * 2 - 1
    # ... (20 total, all deterministic)
    return seeds
```

### Validation
- Tag each forward-pass trade with its shape classification
- Compare WR by shape: directional shapes should have higher WR when
  fractal direction agrees with shape direction
- Ambiguous shapes should show the same +16% lift seen in Analysis K

---

## Part 5: Feature Importance Pruning for Logistic Regression

### File: `training/fractal_clustering.py` → Phase 3 logistic fitting

### Current State
Logistic regression uses all 16D features with equal regularization.

### Change
Use Analysis K feature importance rankings to set per-feature regularization.
Features with higher importance get LESS regularization (allowed to contribute more):

```python
# Top features by Analysis K importance (mapped to 16D positions):
FEATURE_IMPORTANCE_WEIGHTS = {
    8:  1.0,   # hurst (top)
    15: 0.95,  # osc_coherence
    7:  0.90,  # adx
    9:  0.85,  # dmi_diff
    0:  0.80,  # z_score
    2:  0.75,  # momentum
    3:  0.70,  # coherence
    1:  0.65,  # velocity
    14: 0.60,  # term_pid
    13: 0.55,  # tf_alignment
    # remaining: 0.50 (default)
}
```

This can be implemented as a custom penalty in logistic regression, or more
practically, by scaling the features before fitting:
```python
# Scale important features UP so they have more influence
for dim, weight in FEATURE_IMPORTANCE_WEIGHTS.items():
    feat_scaled[:, dim] *= weight
```

### Validation
- Compare logistic regression R² before/after
- Direction accuracy should improve on test set
- Feature coefficients should align with Analysis K rankings

---

## Implementation Order

| Priority | Part | Effort | Risk |
|----------|------|--------|------|
| 1 | Part 2 (Hurst modulation) | ~20 lines | Low — additive, easy to revert |
| 2 | Part 1 (TF weights) | ~10 lines | Low — constant change |
| 3 | Part 3 (osc_coh blend) | ~15 lines | Low — modifies existing blend |
| 4 | Part 4 (Shape gate) | ~80 lines | Medium — new classification at runtime |
| 5 | Part 5 (Feature pruning) | ~30 lines | Medium — changes training pipeline |

**Total**: ~155 lines across 3 files.

**Recommended approach**: Implement Parts 1-3 first (low risk, ~45 lines),
run IS+OOS benchmark, then add Part 4 if direction accuracy improves.
Part 5 requires retraining, so do it last.

---

## Files Modified

| File | Parts | Changes |
|------|-------|---------|
| `training/timeframe_belief_network.py` | 1, 2, 3 | TF weights, Hurst scaling, osc_coh blend |
| `training/orchestrator.py` | 4 | Shape classification gate, seed library |
| `training/fractal_clustering.py` | 5 | Feature importance weighting in Phase 3 |

---

## Success Criteria

1. IS direction accuracy improves from 45.2% to >= 50%
2. WR improves from 37.5% to >= 40% (fewer wrong-direction trades)
3. Fewer total trades (low-conviction signals filtered by Hurst gate)
4. PnL improvement per trade (better direction = better entries)
5. Trail stop effectiveness improves (trending signals reach activation)

---

## Key Analysis K Numbers (for reference)

```
Full year: 15,800 segments, 193 features
Baseline:  54.1%  |  Model: 70.6%  |  Lift: +16.5%

Top features:
  1h_hurst      0.042
  4h_osc_coh    0.040
  4h_hurst      0.035
  1h_adx        0.032
  4h_dmi_diff   0.030
  1h_osc_coh    0.028

Per-TF importance:
  4h: 0.182  |  1h: 0.168  |  2h: 0.098  |  30m: 0.087  |  15m: 0.065
```
