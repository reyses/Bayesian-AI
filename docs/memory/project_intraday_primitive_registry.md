---
name: Intraday primitive registry — bar-level statistical building blocks
description: Catalog of bar-level, forward-pass-clean, statistical primitives available for use as chord axes. R^2_adjusted was the first; this enumerates the rest from V2 features + derivable scalars.
type: project
---

**Built 2026-05-09 evening** to answer "what primitives do we have at intraday scale?"
before extending the chord beyond the current 5 axes.

## Definition of a primitive (the rules a candidate must pass)

A bar-level primitive must be:
1. **Forward-pass clean**: computed only from data at-or-before bar t (no lookahead)
2. **Statistical** (per metaphor-translation rule): describes a measurable
   property, not a paradigm-loaded category
3. **Scalar or small-cardinality**: produces one number per bar (then bucketable)
4. **Stable to recompute live**: same inputs at-bar yield the same output
   in research and live (no global state mutation)
5. **Independent enough**: doesn't trivially equal another primitive
   (e.g., `vwap_w` ≡ `price_mean_w` per memory note 2026-05-09 morning)

## Currently in the chord (5 axes)

These are already wired into `tools/event_bucket_15m_crm.py`:

| primitive             | what it measures                                              | source                          | TF anchor    |
|-----------------------|---------------------------------------------------------------|---------------------------------|--------------|
| slope_15m             | sign + magnitude of 1h-lookback slope of M_close at 15m       | computed from `DATA/ATLAS/15m`  | 15m          |
| curvature_15m         | sign + magnitude of slope-of-slope                            | derived from slope              | 15m          |
| z_close_15m           | (5s_close − M_close_15m) / SE_close_15m                       | computed from `DATA/ATLAS/15m`  | 15m          |
| sigma_rank_15m        | rolling 60min percentile of SE_close_15m                      | derived                          | 15m          |
| r2adj_5m              | R^2_adjusted of 5min linear fit to 5s closes                  | vectorized rolling regression   | 5m window    |

## Available primitives — currently NOT in the chord

### Multi-anchor distance primitives (z-score family)

| primitive | computation | available at TFs | comment |
|-----------|-------------|------------------|---------|
| z_close at 1m  | (5s_close − M_close_1m) / SE_close_1m  | from `DATA/ATLAS/1m`  | tighter, higher-frequency mean |
| z_close at 5m  | analogous                                | already in V2 L3      | mid-frequency mean |
| z_close at 1h  | analogous                                | already in V2 L3      | strategic/macro distance |
| z_close at 4h  | analogous                                | already in V2 L3      | day-scale distance |
| z_high at TF   | (5s_close − M_high_TF) / SE_high_TF      | already in V2 L3      | distance from upper-band anchor |
| z_low at TF    | (5s_close − M_low_TF) / SE_low_TF        | already in V2 L3      | distance from lower-band anchor |

### Channel-width primitives

| primitive | computation | comment |
|-----------|-------------|---------|
| channel_width_15m | (M_high_15m − M_low_15m) / M_close_15m | normalized price-band width at 15m |
| channel_width_1h  | analogous                              | macro band width |

### Slope / curvature at OTHER TFs

The current chord uses 15m slope only. Same shape at:

| primitive | TF anchor |
|-----------|-----------|
| slope_5m  | 5m M_close, 15min lookback |
| slope_1h  | 1h M_close, 4h lookback   |
| slope_4h  | 4h M_close, 1D lookback   |

Each opens a TF-stacking primitive if computed jointly:
- `tf_alignment` = number of TFs in {5m,15m,1h,4h} whose slope sign agrees
  (0..4) — proven directional signal in 2026-05-01 MA-alignment work
  (cell `slope_alignment_4` carried 70% directional accuracy)

### R^2_adjusted at OTHER windows

Current chord uses 5min window. Available at:

| primitive   | window length |
|-------------|---------------|
| r2adj_1m    | 60s linear fit of 5s closes (12 bars) |
| r2adj_15m   | 15min linear fit (180 bars) |
| r2adj_60m   | 60min linear fit (720 bars) |

Different windows answer different questions:
- 1m: very-short-term smoothness (often noise-dominated; may not survive)
- 5m: current chord — local smoothness
- 15m: medium-frequency — most predictive of strategic regime
- 60m: macro smoothness

### Velocity / acceleration primitives (V2 L2)

| primitive          | source                       | comment                                  |
|--------------------|------------------------------|------------------------------------------|
| price_velocity_1m  | `L2_1m_price_velocity_15`    | 1m mean's slope                          |
| price_velocity_5m  | `L2_5m_price_velocity_9`     | 5m mean's slope                          |
| price_velocity_15m | `L2_15m_price_velocity_12`   | 15m mean's slope (≡ chord's slope_15m)   |
| price_velocity_1h  | `L2_1h_price_velocity_12`    | 1h mean's slope                          |
| price_accel_TF     | `L2_<tf>_price_accel_<w>`    | velocity-of-velocity                     |
| vol_velocity_TF    | `L2_<tf>_vol_velocity_<w>`   | LEADING pre-pivot signal (2026-05-09)    |
| vol_accel_TF       | `L2_<tf>_vol_accel_<w>`      | LEADING pre-pivot magnitude              |

### Hurst / swing_noise / reversion_prob (V2 L3)

| primitive          | source                       | comment                                  |
|--------------------|------------------------------|------------------------------------------|
| hurst_5m           | `L3_5m_hurst_9`              | Hurst exponent — random-walk vs persistent |
| hurst_15m          | `L3_15m_hurst_12`            | longer-window Hurst                      |
| swing_noise_5m     | `L3_5m_swing_noise_9`        | local oscillation measure                |
| reversion_prob_TF  | `L3_<tf>_reversion_prob_<w>` | probability of mean-reversion (V2 model) |

NOTE: `reversion_prob_w` was flagged as saturated by NMP entry-gate selection
bias (memory `project_useful_v2_signals.md`). Use cautiously; rank within IS
events rather than absolute value.

### Time / calendar primitives (V2 L0)

| primitive       | source                              | bucketing                       |
|-----------------|-------------------------------------|---------------------------------|
| tod_bucket      | `L0_time_of_day` quantized          | pre_market / us_open / morning / lunch / afternoon / close |
| dow             | derived from timestamp              | Mon..Fri (5)                    |
| cal_event       | external (FOMC/NFP/CPI list)        | binary or multi-class           |

### Path / sequence primitives (computable from 5s OHLCV)

| primitive            | computation                                                            |
|----------------------|------------------------------------------------------------------------|
| run_length_above_M   | bars since price last crossed below M_close (current dwell)            |
| run_length_below_M   | bars since price last crossed above M_close                            |
| monotonic_streak     | bars with same close-direction sign                                    |
| bars_since_pivot     | bars since last sign(slope_15m) flip                                   |
| bars_into_extreme    | bars since price entered current ±k σ band                             |

These are dwell-time primitives — useful for distinguishing "just entered the
state" from "stuck in the state for a while."

### Volume primitives

| primitive            | computation                                          |
|----------------------|------------------------------------------------------|
| vol_relative         | volume / mean(volume) over rolling N bars            |
| vol_rank_60m         | rolling 60min percentile of volume                   |
| vol_velocity_rank    | rolling 60min percentile of |L2_5m_vol_velocity|     |

## Already-rejected primitives (memory)

| primitive       | why rejected                                  |
|-----------------|-----------------------------------------------|
| L2_*_vwap_w     | identical to L2_*_price_mean_w (r=1.000)      |
| L2_4h_price_mean_18 | barely moves intraday — useless at intraday timescale |
| 1D regression  | step-function at TF cadence; no intraday signal |
| L1_1m_price_velocity_1b | too noisy on its own |
| L3_1m_hurst_15 | jitters around 0.5 boundary |
| L3_1m_reversion_prob_15 | saturated near 1.0 by NMP entry-gate selection bias |

## Selection criteria for chord extension

Before adding a primitive to the chord, it must pass:
1. **IS↔OOS sign stability**: marginal P(outcome | bin) at OOS within ±0.10 of IS
2. **Independence test**: not perfectly correlated with an existing chord axis
   (Pearson |r| < 0.7 with each existing axis on the event population)
3. **Outcome differentiation**: at least one outcome variable's distribution
   differs significantly across bins (Kruskal-Wallis p < 0.01 with N>=200/bin)
4. **Cell-population**: when added, the joint chord cells still average ≥ 5
   events per cell (otherwise hierarchical shrinkage swamps the new axis)

A primitive that fails 1 or 2 doesn't go in the chord. A primitive that fails 3
provides no information for a probability table. A primitive that fails 4 only
helps if it's used as a marginal axis (parent-of-cell) not a joint dimension.

## EXISTING FRACTURE MACHINERY (`tools/research/seeds.py`)

User: 'as part of the standalone tool there's already some work that even
fractures the primitives, but that work was done on trade primitives,
instead of day primitives'.

Three components, all reusable for chord work — apply to CRM trajectory
windows instead of trade-around-entry windows:

### 1. SeedPrimitiveLibrary — 20 normalized shape templates

```
Cat 1 (Directional, with UP/DOWN variants = 8):
  LINEAR, EXPONENTIAL, LOGARITHMIC, STEP

Cat 2 (Reversal, with UP/DOWN variants = 8):
  SYMMETRIC_V, ROUNDED_U, FRONT_SKEWED, BACK_SKEWED

Cat 3 (Oscillation, symmetric only = 4):
  SINE_WAVE, DAMPED_OSCILLATOR, EXPAND_OSCILLATOR, FLATLINE
```

`classify_trajectory(price_segment) -> (best_shape, pearson_r)` runs
Pearson correlation against all 20 normalized shapes. Returns NOISE
below CORR_THRESHOLD=0.75.

### 2. _detect_inflections(centroid)

Bar-to-bar sign-flip detection. Returns inflection points and
RISE/DROP/HOLD segments between them.

### 3. _adaptive_split(deltas, r2_target=0.80, max_k=48)

Recursive K-means that finds smallest K where every cluster's
shape-normalized R^2 >= target. Auto-fractures a long trajectory
into shape-coherent pieces.

### Trade primitives -> Day primitives

The original standalone_research applied these to TRADE trajectories
(price path around each trade entry). For chord work the SAME
machinery applies to DAY trajectories — at any bar t, classify the
recent CRM-line window:

```
Procedure at bar t:
  1. Take last N 15m-bars of M_close_15m  (e.g. N=12 = 3 hours)
  2. classify_trajectory(window) -> shape_class, correlation
  3. If correlation < 0.75:
       _adaptive_split(window) -> sub-trajectories
       classify each sub-trajectory
       use the LATEST sub-shape as the chord axis value
  4. shape_class becomes a CATEGORICAL chord axis
```

This adds a CATEGORICAL axis (up to 20 values) to the existing
continuous chord. Cardinality cost: chord size multiplies by ~20.
Manageable via hierarchical shrinkage (shape becomes a parent axis).

### What shape_class catches that continuous primitives miss

| continuous axis snapshot              | shape that fits          | strategy implication                         |
|---------------------------------------|-------------------------|----------------------------------------------|
| slope_pos + curv_pos                  | LINEAR_UP                | sustained drift — ride-friendly              |
| slope_pos + curv_pos                  | EXPONENTIAL_UP           | accelerating up — late-stage trend           |
| slope_pos + curv_neg                  | LOGARITHMIC_UP           | exhausting trend — fade-friendly setup       |
| slope_zero + high curv excursions     | SYMMETRIC_V              | sharp pivot, both legs steep                 |
| slope_zero + high curv excursions     | ROUNDED_U                | gentler pivot, both legs gradual             |
| slope_zero + high sigma_rank          | SINE_WAVE                | bounded oscillation — fade-friendly          |
| slope_zero + low sigma_rank           | FLATLINE                 | quiet — bow-out friendly                     |

The shape class disambiguates patterns that the continuous primitives
collapse into the same chord cell. Strong V0 candidate axis once
substrate proves out.

## Most promising additions to evaluate (ranked)

1. **tf_alignment** (slope-sign agreement across {5m,15m,1h,4h}) — proven 70%
   directional accuracy in the 2026-05-01 MA-alignment work; not yet a chord
   axis. Cardinality 5 (0..4 alignment count).
2. **z_close_1h** — strategic distance from macro mean. Different timescale
   than 15m z_close (currently in chord); should be conditionally independent.
3. **vol_velocity_rank_5m** — the LEADING pre-pivot signal we visually validated
   on 2026_02_12. Captures pre-cascade activity ramp.
4. **r2adj_15m** — broader-window smoothness. Likely complements r2adj_5m
   (different window catches different smoothness regimes).
5. **bars_since_pivot** — dwell-time primitive. Captures "fresh pivot" vs
   "stuck in state" distinction that pure-state primitives miss.

## What the chord might look like after extension

Conservative V1 (8 axes if all 5 promising additions pass selection):

```
chord_v1(t) = (
    slope_15m_q,        # in chord
    curvature_15m_q,    # in chord
    z_close_15m_q,      # in chord
    sigma_rank_15m_q,   # in chord
    r2adj_5m_q,         # in chord
    tf_alignment_q,     # NEW — TF stacking
    vol_velocity_rank_q,# NEW — pre-pivot leading
    bars_since_pivot_q, # NEW — dwell time
)
```

Cardinality estimate: 5 × 3 × 5 × 5 × 5 × 5 × 5 × 5 = 234,375 cells (way too thin).
Hierarchical shrinkage REQUIRED at this depth. Per-axis marginals stay populated
(5,340 events / 5 bins ≈ 1,068 events/bin). Joint cells go via shrinkage.

## NOISE motif structure (visualized 2026-05-10)

The NOISE shape_class accounts for ~40% of motifs (940 IS / 1,150 total). It
is NOT homogeneous — visualization revealed 4 distinct sub-populations:

| sub-pop          | trigger                                            | scale     | implication                                              |
|------------------|----------------------------------------------------|-----------|----------------------------------------------------------|
| NOISE_warmup     | M0 + pk_z >= 10 + ride=$0 + tiny mean_sigma        | ~35 (3%)  | DATA ARTIFACT: 15m regression warmup at start of day; SE near-zero produces inflated z; filter out |
| NOISE_calm       | pk_z < 2                                           | majority  | true rest; meta-router -> BOW                            |
| NOISE_eventful   | pk_z 2-5, length 30-180min, multi-leg              | many      | melody-level chord IS the structure here                 |
| NOISE_compound   | length > 3hr, no 15m inflection caught sub-patterns| smaller   | melody is the natural unit; macro is too coarse          |

After excluding warmup artifacts, true NOISE pk_z is median 2.42, q90 4.78
(vs. raw 2.49/207 — the artifacts inflate the upper tail dramatically).

### Three concrete fixes from this finding

1. **Warmup filter for segmenter**: skip first ~1hr of day OR require at
   least N completed TF bars before first motif starts. Eliminates the 35
   artifact motifs that confound NOISE statistics.

2. **NOISE-motif chord defers to melody multiset**: when shape_class=NOISE
   on a motif, the chord cell becomes "(NOISE, melody_shape_multiset)".
   A NOISE motif containing 3 LOGARITHMIC_DOWN melodies is a CRASH;
   one with mixed UP+DOWN melodies is a RANGE. Melody-level signal exists.

3. **Threshold sensitivity**: r >= 0.75 leaves borderline cases (r=0.70-0.75)
   in NOISE; tightening to r >= 0.70 would reclassify ~5-10% of NOISE
   motifs into named shapes. Worth A/B-comparing once V0 table is fitted.

## Pruning to V0

For V0 we should NOT add all 5 candidates at once. Build the table on the
existing 5-axis chord first, validate IS↔OOS, identify which axes survive.
THEN add 1-2 of the promising candidates and re-validate. Each addition gets
its own sign-stability check.

This avoids the 2026-05-03 quantile-cell-overfit pattern: more axes = more
cells = more chances for a top-|lift| cell to flip sign at OOS.

---

## DUAL-ANCHOR CHORD = THE GEOMETRIC-PRIMITIVE WORK, RIGHT SUBSTRATE

User connected this 2026-05-09 evening (verbatim): 'thats why i recalled
the primitive geometry work we did weeks ago'.

The earlier `tools/shape_primitive_builder.py` and `tools/level_shapes.py`
attempts to encode visual shapes as features failed because they used raw
price. But the visual structure they were trying to capture — micro-inside-
macro patterns at multiple TFs — IS what a multi-TF chord represents.

When the chord uses primitives at two anchors (5m AND 15m), the joint
chord vector encodes:

- macro state    (slope_15m_q, z_close_15m_q, sigma_rank_15m_q, r2adj_15m_q, ...)
- micro state    (slope_5m_q,  z_close_5m_q,  sigma_rank_5m_q,  r2adj_5m_q, ...)
- joint cell     micro x macro = WHICH micro-pattern is happening INSIDE which macro-pattern

Specific cells that become meaningful:

- micro_slope_pos x macro_slope_neg     micro pivoting against macro  (countertrend pop)
- micro_slope_neg x macro_slope_neg     micro confirming macro        (continuation)
- micro_sigma_q1  x macro_sigma_q5      compression inside expansion  (pre-resolution)
- micro_z_far_pos x macro_z_far_neg     extension against macro mean  (failed-fade setup)

These are PATTERN STRUCTURES recognizable to a human looking at the chart
(see the Thursday representative `chart/buckets/dow_representative.png` —
the 1m, 5m, 15m, 1h overlaid means make these patterns visible). But unlike
the geometric-primitive work that tried to detect them as visual shapes,
the chord encodes them as a JOINT BUCKET in feature space, queried by the
Bayesian table.

## CHORD AS FAILURE-MODE VOCABULARY (the unlock)

User: 'this way we can say well the trades failed, where? in a gentle slope
with high variability, or a crash with low variability'.

The chord is not just a lookup key — it is a SPEAKABLE VOCABULARY for the
failure-mode conversation. Each cell becomes a named scenario that maps
to plain-English description without losing math grounding (translation
table in `feedback_no_human_regime_terms.md`).

Examples once 5m+15m anchors are wired:

| chord cell                            | prose description                              | failure mode hypothesis                                                        |
|---------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------|
| slope_15m_q3 × r2adj_15m_q1           | gentle macro slope + high macro variation       | fades hit noise stops before mild drift plays out                              |
| slope_15m_q1 × r2adj_15m_q5           | strong negative macro trend + low macro var.    | fades fail in clean crashes — trend too coherent to revert                     |
| slope_15m_q3 × r2adj_15m_q5           | gentle macro slope + low macro variation        | 'happy' cell — fades work; orderly mean reversion                              |
| slope_15m_q1 × r2adj_15m_q1           | strong negative trend + high variation          | both adversarial — meta-router → BOW                                            |
| sign(slope_5m) ≠ sign(slope_15m)      | micro pivoting AGAINST macro                    | countertrend pop inside macro trend; fade-with-macro and ride-with-micro both risky |
| sigma_rank_5m_q1 × sigma_rank_15m_q5  | micro compression inside macro expansion        | pre-resolution setup — typically a transitional state                          |
| z_close_5m_far_pos × z_close_15m_far_neg | micro extended UP inside macro extended DOWN  | failed-fade setup — macro fade exhausted, micro reversal failing              |

The Bayesian table's per-cell numeric output ($/trade per tier, P_cascade,
expected MFE/MAE) gives the QUANTITATIVE backing. The chord cell name gives
the QUALITATIVE context. Together they enable the conversation:

  'trades fail in <chord cell>, where <prose description>, because
   <hypothesis>; mitigation is <gate this cell off via context filter
   OR route via meta-router>.'

That sentence is the deliverable of the failure-mode work. Without the
chord vocabulary, that sentence cannot be written.

## V0 chord proposal — 10 axes, dual anchor

Promote the backlog item 'Geometric primitives on CRM substrate' from
DEFERRED to ACTIVE. Same architecture, new name.

```
chord_v0(t) = (
    # macro context (15m-anchored)
    slope_15m_q,        # in chord
    curvature_15m_q,    # in chord
    z_close_15m_q,      # in chord
    sigma_rank_15m_q,   # in chord
    r2adj_15m_q,        # NEW (currently only r2adj_5m)

    # micro state (5m-anchored)
    slope_5m_q,         # NEW
    curvature_5m_q,     # NEW
    z_close_5m_q,       # NEW
    sigma_rank_5m_q,    # NEW
    r2adj_5m_q,         # in chord
)
```

Cardinality: 5×3×5×5×5 × 5×3×5×5×5 = 234,375 cells. With 5,340 IS events,
mean = 0.02 events/cell (essentially empty). Hierarchical shrinkage IS THE
ENGINE here:
- Cell level: at most ~5,340 unique chords actually seen
- Macro-marginal (collapse to 5 macro axes only): ~1,500 cells max
- Micro-marginal (collapse to 5 micro axes only): ~1,500 cells max
- Universal: 1 cell

Live lookup: query the cell, fall back to whichever marginal has enough N.

## What this means for the V0 build path

The proposed flow updates to:

```
1. Extend `tools/event_bucket_15m_crm.py` -> `tools/event_bucket_dual_crm.py`:
   compute the 10-axis chord at each event ENTRY bar
2. Sanity-check: rebuild the 5 axis charts at the new dual cardinality;
   visually confirm bucket panels show the expected micro/macro patterns
3. Oracle-label each event (lookahead OK per
   `feedback_oracle_vs_chord_lookahead.md`)
4. Diagnose: per joint chord cell, do oracle outcomes differ?
5. THEN build the Bayesian table fresh (not by extending the regime-keyed
   `training_iso_v2/bayesian_table.py`)
```
