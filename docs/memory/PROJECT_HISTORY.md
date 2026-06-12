# Legacy Project and Research History



## Source: project_5level_segmentation_substrate.md

---
name: 5-level hierarchical segmentation substrate
description: 2D-shape recursive segmenter at 15m/5m/1m/15s/5s with stepped EDA, hyper-volume regression, and Bayesian probability tables — the lookup substrate for the meta-router
type: project
---

# 5-LEVEL HIERARCHICAL SEGMENTATION SUBSTRATE (built 2026-05-09 evening)

## What it is

A recursive 2D-shape segmenter that decomposes every trading day into a
5-deep hierarchy of simple-shape primitives. Same library at every level
(13 primitives + NOISE); only the (TF anchor, min span, slope lookback)
tuple changes per level.

```
LEVEL       TF anchor       window           min span        slope lookback
phrase      15m M_close     12 bars (3hr)    60 5s (5min)    720 5s (1h)
motif       5m  M_close     9  bars (45min)  12 5s (1min)    360 5s (30min)
sub_motif   1m  M_close     15 bars (15min)  6  5s (30s)     90  5s (7.5min)
measure     15s M_close     12 bars (3min)   4  5s (20s)     30  5s (2.5min)
note        5s  M_close     6  bars (30s)    2  5s (10s)     12  5s (1min)
```

## Why it matters

This is the **Bayesian-table substrate** the meta-router needs to look
up `P_cascade(t)` at-bar with no lookahead. The 13-primitive simple-shape
library at every TF gives a small-cardinality categorical vector per
hierarchy node — perfect Bayesian-table key.

**Why:** the 2026-05-09 architectural lock established that the meta-router
needs a chord-keyed lookup (replacing the deterministic CRM detector v2
that over-suppressed slow-buildup macro pivots). The 5-level substrate
provides exactly that: at any 5s bar, the hierarchy gives `(phrase_shape,
motif_shape, sub_motif_shape, measure_shape, note_shape)` — a 5D
categorical chord that conditions all downstream probability lookups.

**How to apply:**
1. To compute P(fwd_up) at any bar, look up the (level, shape, parent_shape)
   row in the matching Bayesian table and apply Beta(1,1) Jeffreys posterior.
2. To detect "cascade in progress", combine note-level + measure-level
   shapes — large-n strong-signal cells like NOISE-after-STEEP_LINEAR_DOWN
   (35.5% UP, n=9,539, CI [0.345, 0.365]) are the structural cascade
   continuation markers.
3. To EXTEND the substrate, add an axis to the cell key — TOD bucket,
   calendar event, regime label. Cell granularity grows multiplicatively;
   pool sparse cells via the Beta hierarchy.

## Bulk-run results (345 days)

```
LEVEL          n         top-3 shapes
phrase     2,047    NOISE 21.3%  FLATLINE 17.0%  GENTLE_LINEAR_UP 13.8%
motif      6,276    FLATLINE 16.5%  NOISE 15.5%  GENTLE_CONCAVE_UP 9.4%
sub_motif 21,048    FLATLINE 15.4%  STEEP_LINEAR_DOWN 13.3%  STEEP_LINEAR_UP 12.9%
measure   70,782    STEEP_CONCAVE_UP 15.3%  NOISE 14.8%  STEEP_CONVEX_DOWN 14.8%
note     203,801    NOISE 31.6%  STEEP_LINEAR_UP 12.1%  STEEP_LINEAR_DOWN 12.0%
```

## Strongest cross-level findings (n>=50, |p−0.5|>=0.10)

```
LEVEL     SHAPE              | PARENT               n     P_up   [CI]
note      NOISE              | STEEP_LINEAR_DOWN  9539  0.355  [0.35, 0.36]   ★ tight
note      NOISE              | STEEP_LINEAR_UP    9793  0.622  [0.61, 0.63]   ★ tight
sub_mot   STEEP_CONCAVE_UP   | STEEP_LINEAR_UP    321   0.684  [0.63, 0.73]
sub_mot   FLATLINE           | STEEP_LINEAR_UP    306   0.646  [0.59, 0.70]
motif     FLATLINE           | STEEP_LINEAR_UP    57    0.746  [0.63, 0.85]
measure   NOISE              | STEEP_CONVEX_UP    134   0.728  [0.65, 0.80]
sub_mot   STEEP_CONVEX_UP    | GENTLE_CONCAVE_UP  51    0.245  [0.14, 0.37]
```

Full 60-finding table: `reports/findings/segments/bayes_tables/STRONG_findings_n50_e10.csv`.

Three structural patterns:
1. **Continuation dominates immediately after directional moves** (note level)
2. **Pause-after-rally is bullish** (motif/sub_motif FLATLINE-after-UP)
3. **Counter-trend curves fail** (small-shape inside larger-opposing)

## Tools (all parametric, all reusable)

```
tools/segment_simple_shapes.py            (5-level recursion + max_depth param)
tools/segment_simple_bulk_v2.py           (bulk runner; --threshold --max-depth)
tools/segment_stepped_eda.py              (per-level EDA: shape/length/skew/markov/parent-child)
tools/segment_stepped_surface_regression.py  (forward returns + 2D heatmaps + OLS)
tools/segment_bayes_tables.py             (Beta(1,1) Jeffreys posteriors at every conditioning)
tools/segment_chart_multilevel.py         (5-panel hierarchy visualization)
```

## Re-run pipeline

```
python tools/segment_simple_bulk_v2.py --threshold 0.85 --max-depth 5
python tools/segment_stepped_eda.py
python tools/segment_stepped_surface_regression.py
python tools/segment_bayes_tables.py
python tools/segment_chart_multilevel.py --extremes --random 5
```

## Output tree

```
reports/findings/segments/simple_bulk_v2/
    themes.csv  +  all_<level>s.csv (5)  +  per_day/<day>.json (345)

reports/findings/segments/stepped_eda/
    <level>_shape_dist.csv / length_dist.csv / skew_dist.csv / r_dist.csv
    <level>_split_dist.csv / markov.csv
    <level>_given_<parent>.csv
    <level>_overview.png
    overall_summary.csv

reports/findings/segments/stepped_surface_reg/
    <level>_marginal_by_shape.csv
    <level>_surface_shape_length.csv  + .png
    <level>_surface_shape_sigma.csv   + .png
    <level>_surface_parent_self.csv   + .png
    <level>_ols_coefs.csv  +  <level>_mfe_mae.csv

reports/findings/segments/bayes_tables/
    <level>_priors.csv  +  priors.png
    <level>_p_up_given_shape.csv  +  p_up.png
    <level>_p_up_given_shape_skew.csv
    <level>_p_up_given_shape_sigma.csv
    <level>_p_up_given_shape_parent.csv  +  p_up_parent_extremes.png
    STRONG_findings_n50_e10.csv      ★ 60 cross-level discoveries
```

## Caveats / open questions

- Tables currently pool IS+OOS (split CSVs exist but separate tables not yet built).
- "fwd_return > 0" is coarse; magnitude/MFE/MAE stored separately not joined.
- Forward horizons fixed per level (note=30s, ..., phrase=60min). No reason these are right.
- Some n=50 cells in the strong-findings list may collapse on OOS-only.
  **Do not trade off these without OOS validation.**

## Pending (next session)

1. Add **TOD-bucket axis** to every Bayesian cell (locked structural req).
2. Add **calendar-event axis** (FOMC/NFP/CPI flags).
3. **OOS sign-stability check** per cell — drop cells whose IS sign != OOS sign.
4. **Per-tier oracle PnL per primitive chord** (failure-mode identifier).
5. **V0 meta-router prototype** using `P_cascade(t)` from this substrate.

## Key journal cross-refs

- `docs/daily/2026-05-09.md` (evening section: "5-LEVEL HIERARCHICAL SEGMENTATION SUBSTRATE BUILT")
- `reports/findings/2026-05-09_5level_substrate.md` (focused finding doc)
- `memory/project_original_bayesian_brain_architecture.md` (original 8-step pipeline this substrate replaces step 3 of)
- `memory/feedback_no_human_regime_terms.md` (vocabulary translation table)
- `memory/feedback_quantile_selection_overfit.md` (the OOS-validation requirement that drives "pending #3")


## Source: project_9tier_discovery_v2.md

---
name: 9-tier discovery exercise (V2-native)
description: 2026-05-04 — recreate the legacy 9 ExNMP tiers in V2 by running NMP-only and finding signals that distinguish FADE_BETTER from FLIP_BETTER cohorts. Single-column V2 EDA found NO discriminating signal.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

The user requested recreating the legacy 9-tier discovery process in V2-native
form. Legacy methodology (per `docs/daily/2026-04-06.md` through `2026-04-18`):

1. Run NMP entry: `|z_1m| > 2.0 + variance_ratio < 1.0`
2. Find the splitter axis (legacy found: velocity at entry, CNN flip,
   1h alignment, wick rejection)
3. Sub-classify entries → measure $/trade per sub-tier
4. Add new entry types when EDA reveals signals NMP misses

The V2 equivalent of step 1 is REVERSION: `|z_se_w| ≥ 1.8 + reversion_prob ≥ 0.55`.

## Methodology used (training_v2/tier_discovery.py)

For each NMP-only IS trade, compute via regret labels:
- `fade_peak` = peak realized in original direction
- `flip_peak` = -mae_pnl (peak realized if direction had been opposite)

Classify each trade:
- **FADE_BETTER**: fade_peak > flip_peak + $15 AND fade_peak > $15
- **FLIP_BETTER**: flip_peak > fade_peak + $15 AND flip_peak > $15
- **CHOP_SKIP**: neither direction had a clear peak

For each of 185 V2 columns at entry, compute Cohen's d between
FADE_BETTER and FLIP_BETTER cohorts. Walk-forward validate on 70/30 split.

## Findings (2026-05-04, 19,106 NMP IS trades)

### The opportunity is real and large

| cohort | n | actual $/trade | fade peak | flip peak |
|---|---:|---:|---:|---:|
| FADE_BETTER | 8,243 | +$9.29 | $139 | $32 |
| FLIP_BETTER | 8,122 | -$9.20 | $33 | **$146** |
| CHOP_SKIP | 2,741 | +$0.56 | $42 | $42 |

Roughly 50/50 fade/flip split. A perfect oracle that flipped the
FLIP_BETTER cohort would convert -$74,710 → +~$120K (peak-side estimate),
turning the IS total from $3K → ~$152K.

### But single-column V2 features carry NO signal

- **Highest Cohen's d across all 185 V2 columns: 0.040** (negligible — small
  effect threshold is 0.2, medium is 0.5)
- **0 of 25** top IS features survive walk-forward validation (70/30 split)
- Top features are all volume-related: `L2_15m_vol_accel_12`,
  `L2_1m_vol_mean_15`, `L1_15s_price_accel_1b` — but signs flip on validation

This is the same OOS-overfit pattern as the 2026-05-03 quantile-cell lesson
(75% IS-best rules collapsed on OOS, 25.8% survival rate).

## Why this differs from the legacy CNN flip's 70.6% accuracy

The legacy CNN flip predictor saw a 6×13 V1 grid AND learned **cross-feature
patterns** (distillation showed it combined `15m_wick_ratio × 1h_z_align ×
5m_velocity`). Single-column linear EDA cannot see those.

Plus we are missing key features the legacy used:

| Legacy discriminator | In V2 entry vector? |
|---|---|
| `wick_ratio` (multi-TF) | NO — wick rejection requires OHLCV math, not in V2 |
| `1h_z_align` (sign agreement) | Yes-ish (`sign(L2_1h_price_velocity_w)`) but encoded as a magnitude (positive and negative cancel out) |
| `dmi_diff` extreme | NO — DMI was a V1 synthesis concept |
| Directional wicks (upper/lower) | NO — needs 1m/5m/15m OHLCV |

User noted that directional wicks WERE calculated during the run in the
legacy path (`directional_wicks_batch` in `core_v2/v1_compat.py`) but are
NOT yet in the V2-native entry feature vector. Adding them is path A
described below.

## Three honest paths forward

| Path | What | OOS-overfit risk |
|---|---|---|
| **A** | Add directional wicks per-TF to entry features and rerun discovery | Low — recovers known-good signal |
| **B** | Run chord-style joint-quantile search on 185D pairs (~17k pairs) | Medium-High — pair-quantile rules collapsed on OOS in 2026-05-03 |
| **C** | Train V2DirectionCNN on the FADE/FLIP/CHOP labels — let it learn cross-feature patterns | Medium — known to overfit; needs walk-forward validation |

The user's framing was "what signal can we use to TURN BAD INTO GOOD" — the
goal is direction-flipping FLIP_BETTER cohort, not skipping it.

## Code artifacts

- `training_v2/tier_discovery.py` — the discovery tool with Cohen's d + walk-forward
- `training_v2/output/regret_nmp.pkl` — regret labels for 19,106 NMP-only trades
- `training_v2/output/nmp_only.pkl` — the trade pickle
- `reports/findings/v2_tier_discovery.md` — markdown report

## Conclusion

V2-native single-column EDA at NMP entry produces no actionable splitter.
The 9-tier legacy methodology relied on cross-feature patterns the CNN
implicitly learned, OR features (directional wicks, DMI synthesis) we don't
currently have in the V2 entry vector. Plan: add directional wicks to the
entry vector first (path A), then escalate to CNN if single-column with
wicks still fails.


## Source: project_adx_chop_filter.md

---
name: ADX chop filter on peak entries
description: 1m ADX < 15 blocks peak entries in choppy markets. Feb 9 went from -$8,869 to -$24. Biggest single improvement to OOS.
type: project
---

ADX chop filter added 2026-03-19 as Layer 3 of peak entry gate.

**Why:** Peak detection fires on every price wiggle in chop. 94% of entries in low-ADX markets are noise. The system was entering and exiting repeatedly in sideways conditions, bleeding money.

**How to apply:** `_1m_confirms_peak()` in `bar_processor.py` checks 1m ADX. Below 15 = block. This is the single biggest improvement: Feb 9 crash went from -$8,869 (222 trades) to -$24 (1 trade). OOS PnL jumped from $6,143 to $14,336.

Three-layer gate: (1) 1m sensor opposition, (2) fake peak (vol+fm), (3) ADX chop. All in `_1m_confirms_peak()`.


## Source: project_auto_seeds_next.md

---
name: Auto seeds as template library (next major feature)
description: 31,605 auto seeds across 312 days with direction, MFE, MAE, duration. Replace HDBSCAN templates with seed matching.
type: project
---

Auto seeds are at `DATA/regime_seeds/auto_swing/auto_seeds_edited_20260313_212432.json`.

**Why:** HDBSCAN on full ATLAS produced only 3 templates. IS underperforms OOS ($0.97 vs $4.93/trade). Seeds are human-trained, pre-labeled with direction and expected outcomes. No clustering needed.

**How to apply:**
1. Augment seeds with peak detection + sensor data + 192D context
2. At runtime: peak fires -> extract 10-bar shape + context -> match nearest seed
3. Seed tells you: direction, expected MFE, MAE, duration
4. Then add quantum features as additional context dimensions
5. FibonacciPivots (S1-S3, R1-R3) as support/resistance context

Seed structure: `{trade_id, direction, entry_price, exit_price, mfe_ticks, mae_ticks, duration_mins, lookback_bars: 10, ...}`


## Source: project_bayesian_archetypes_pending.md

---
name: bayesian-archetypes-pending
description: Layer 3 of the regret-oracle arc — Bayesian Trade Archetypes via N-D Trajectory Clustering. Protocol LOCKED 2026-05-16 across 8 topics, build PENDING (5 phases). Spec at research/bayesian_archetypes/project.md.
metadata:
  type: project
---

**Status as of 2026-05-16 evening: protocol LOCKED, build PENDING.**

Full DMAIC spec: `research/bayesian_archetypes/project.md`.

## Locked decisions (8 topics, in discussion order)

1. **Clustering strategy**: peel extremes first (greedy iterative).
2. **Seed criterion**: highest remaining `mfe_velocity` ($ per minute of the
   trade's own outcome). Always positive, lookahead-OK for offline clustering.
3. **Feature pool**: ALL ~190 V2 features at
   `DATA/ATLAS/FEATURES_5s_v2/L{1,2,3}_{5s,15s,1m,5m,15m,1h,4h,1D}/`, joined
   per-bar over each trade's duration. (User flagged my initial ~19-feature
   listing as too narrow.)
4. **Trade signature**: PCA line in 190-D space (centroid + unit direction
   from SVD of the centered trajectory matrix). Per-feature 5%-of-seed-value
   matching was REJECTED because of zero-crossing degeneracy on signed
   features.
5. **Similarity criterion**: perpendicular distance from candidate trajectory
   points to seed's PCA line in z-scored units. Match if all (or quantile of)
   points within radius `r`.
6. **Hierarchical r-ladder**: coarse → fine (e.g., 0.5σ → 0.25σ → 0.125σ).
   Each peeled seed produces a TREE of nested clusters; Bayesian table is
   hierarchical per direction.
7. **Trade decay tracking** (free side-effect of trajectory storage):
   `d(t)` = time series of distance from trade's own trajectory points to
   its matched cluster's PCA line. Three patterns: stable / decaying /
   converging. This UNIFIES: entry classifier + exit signal + duration
   prediction + Bayesian online posterior update for direction.
8. **CUDA build** via PyTorch (batched SVD), matching codebase's CUDA-only
   convention.

**Level-1 stratification: direction (LONG/SHORT)** — clustering runs
independently within each pool.

## Deferred open questions (deliberately not answered)

1. **Decay metric definition**: raw r-distance `d(t)` vs normalized
   `d(t) / r_fine` (where 0=on-line, 1=fine edge, 2=medium edge, 4=coarse
   edge). DECIDE after Phase 4 IS sanity check visualizes actual decay
   curves.
2. **"Decayed" threshold**: hard-cross of `r_coarse` (binary state change)
   vs sustained positive slope of `d(t)` over K bars (gradual). DECIDE at
   Phase 5 when building live evaluator.
3. **Live applicability path**: lead-in-trajectory matching (cluster on past
   N bars too, correlate to forward archetype) vs single-bar entry-match
   with decay-update as bars arrive. DECIDE at Phase 5.

These do NOT block clustering itself; they block live deployment.

## Build phases (pending)

- **P1**: `tools/regret_join_v2_features.py` — daisy-chain CSV × V2 features
   joined per-bar over each trade's duration. Output: per-trade torch tensor.
- **P2**: `tools/regret_trade_signatures.py` (CUDA) — per-trade z-scoring +
   PCA SVD → save (centroid, direction unit-vec, magnitude, per-bar distance
   series). Edge: trades with T_bars < 5 are PCA-unstable; flag/skip.
- **P3**: `tools/regret_bayesian_table.py` (CUDA) — peel + hierarchical
   r-ladder. CLI parameters: `--r-coarse`, `--r-levels`, `--seed-feature`,
   `--direction-stratify`, `--min-cluster-size`, `--min-pool`,
   `--max-iterations`, `--remove-on-peel`. Outputs: hierarchical Bayesian
   table CSV + cluster assignments + decay curves npz.
- **P4**: IS sanity check on first 5 clusters per direction. Are they
   recognizable archetypes? Cluster-size distribution sane? Decay curve
   patterns visible?
- **P5**: `tools/regret_bayesian_live_eval.py` — match live bars to clusters,
   aggregate to Day WR + mode $/day per CLAUDE.md protocol, OOS-validate
   on 2026.

See [[regret-six-layer-architecture]], [[signed-mfe-pivot]],
[[kway-r2-saturation]].


## Source: project_bayes_table_filter_trail.md

---
name: Bayesian table = filter + trail stop for the 9 ExNMP tiers
description: Architectural lock 2026-05-10 — per-cell direction/duration/magnitude/decay tables become context filter + adaptive trail for all 9 tiers
type: project
---

# BAYESIAN TABLE → FILTER + TRAIL STOP FOR 9 TIERS

## Architectural lock (2026-05-10)

User: "and this becomes the filter and trailstop of the 9 tiers"

## 3-STATE simplification (2026-05-10, refining the lock)

User: "5s CRM has only 3 possible actions continue direction, flatline or
reverse, out of these 3 only one is adverse"

At any 5s bar the CRM has 3 next-state outcomes:
  CONTINUE  — same slope sign         → favorable to current position
  FLATLINE  — slope ≈ 0               → neutral, no PnL change
  REVERSE   — slope flips             → ONLY ADVERSE outcome

This collapses the filter+trail problem to a single per-cell statistic:
  P(reverse | current_cell, t_in_trade)

The duration_per_axis.csv table IS this survival function:
  P(reverse by t) = 1 - P(duration >= t)

The decay_per_axis.csv peak_horizon column = bar to start tightening trail.

No magnitude branching needed — REVERSE is the only adverse outcome to
defend against. Magnitude tables become input to position sizing, not exit
logic.

Structural EV note: with only 1/3 outcomes adverse, even uniform odds
give break-even. The Bayesian table tilts cells away from uniform — most
cells have P(continue) > P(reverse) by a margin that the table quantifies.

The Bayesian probability table built 2026-05-09→10 (per-primitive-cell direction
+ duration + magnitude + decay) is NOT a standalone strategy. It is:

1. **CONTEXT FILTER** — gates when each of the 9 ExNMP tiers fires
2. **ADAPTIVE TRAIL/STOP** — calibrates exits per cell

The 9 tiers (FADE_CALM/MOMENTUM/AGAINST, RIDE_CALM/MOMENTUM/AGAINST,
KILL_SHOT, CASCADE, FREIGHT_TRAIN) keep their existing entry rules. Each
gets the table as a wrapper that consults cell statistics at entry and
during the trade lifecycle.

## Composition (per 2026-05-09 context-filter-vs-tier lock)

```
Strategy = TIER_entry AND f1(state) AND f2(state) AND filter_bayes(at_bar_cell)
                                                                      AND
trail_bayes(cell, t_in_trade, current_PnL)
```

Multiplicative gate. The Bayesian filter is one of f_i. The trail logic
replaces (or overlays on) existing fixed-dollar trail stops.

## Filter logic

At each tier entry candidate:
1. Compute at-bar primitives (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q)
   from `tools/event_bucket_15m_crm.py:_compute_day_crm_features` — this is
   already lookahead-clean and reusable.
2. Look up cell in `BLEED_tier_x_sub_motif_x_measure.csv` for THIS tier.
   If cell is in the bleed-set: SKIP.
3. Look up cell in `p_up_per_axis.csv` for direction prior. If direction
   prior opposes the tier's intended direction: SKIP.
4. Else: PROCEED with the entry.

## Trail/stop logic

Once tier fires:
1. Initial hard stop: `entry_price ± stop_z * SE_anchor` where stop_z comes
   from `magnitude_per_axis.csv` (q90 of excess past trigger ≈ +2σ for
   most cells per user hypothesis).
2. Adaptive trail: query `decay_per_axis.csv` for cell's peak_horizon and
   decay_to_60m. Two archetypes:
   - SUSTAINED (peak=60m, decay=0): loose trail, ride to time-stop or
     band-mean cross
   - SPIKE_REVERT (peak=5-15m, decay<<0): tighten aggressively past
     peak_horizon, exit when expected remaining PnL approaches zero
3. Exit triggers:
   - hard stop hit
   - peak_horizon reached AND decay sign turning negative
   - price reverts to band-mean (M_close cross)
4. Optional: arm opposite-leg entry on exit (if exit-time cell has stable
   counter-edge in the same table)

## Loop architecture

```
Always-in-market (when both directions have stable cells):
  [tier ENTRY] → [bayes FILTER pass] → IN TRADE
       ↓
  [bayes TRAIL/STOP fires] → EXIT
       ↓
  [exit-bar primitive lookup] → opposite-direction cell stable?
       ↓                              ↓
       YES → arm opposite leg         NO → wait for next tier signal
```

The filter and trail share the same lookup table; the tier provides the
entry signal; the exit's primitive vector becomes the next entry decision.

## Required infrastructure

```
training_iso_v2/filters/                              (new dir)
    bayes_table_lookup.py        at-bar feature compute + cell lookup helper
    bayes_filter.py              SkipFilter wrapping any tier with the table

training_iso_v2/exits.py         (extend with bayes_trail exit class)
    BayesAdaptiveTrail           reads cell decay/magnitude, sets trail
                                  per archetype (sustained / spike_revert)

training_iso_v2/strategies/<each_tier>.py    (modify each)
    self.bayes_filter = BayesFilter(tier=self.name, table_dir=...)
    on_candidate(state): if not self.bayes_filter.allow(state): return None
    on_trade(state, trade): self.bayes_trail.update(state, trade)
```

## Tables consumed

```
reports/findings/segments/diagnostic_tier_bleed/
    BLEED_tier_x_sub_motif_x_measure.csv     per-tier skip-cell list

reports/findings/segments/bayes_table_v0_location/
    p_up_per_axis.csv                        direction prior per cell
    duration_per_axis.csv                    P(continues N min) per cell
    magnitude_per_axis.csv                   stop_z, target_z per cell
    decay_per_axis.csv                       peak_horizon, decay per cell
    actionable_ride_exit_table.csv           pre-joined fast-lookup
```

All tables keyed on (side, anchor, axis, bin) at min_n=10 IS samples.

## Validation path before deploy

1. Wire into ONE tier first (suggest NMP_FADE_RAW — biggest bleed savings
   $7,970 OOS uplift if filter works as expected)
2. Run full IS+OOS through `training_iso_v2.run_iso --tiers NMP_FADE_RAW`
   with and without bayes filter
3. Compare: $/day, day-WR, max DD, OOS gap from IS
4. If filter produces stable improvement (CI excludes 0): repeat for next
   tier in bleed-savings-rank order:
       NMP_FADE_RAW → FADE_AT_BAND → FADE_MOMENTUM → FADE_CALM → KILL_SHOT
       → CASCADE → FADE_AGAINST → ...
5. RIDE tiers come last — they have fewer bleed cells in current
   diagnostic, so the filter may have less room to help

## Caveats

- Bleed cells were SELECTED on data that includes OOS — even with
  IS<0 AND OOS<0 sign-stability, applying the skip rule to the same OOS
  is mildly optimistic. Honest test = re-fit bleed cells on IS only,
  apply to true held-out OOS. Likely shrinks the savings 20-40%.
- Trail/stop tables also use full IS+OOS. Same caveat for any trail
  parameter learned from these.
- Day-clustering not applied. Cells with n=89 may be 5-10 days; effective
  sample is much smaller.

## Related memories

- `feedback_5s_inherently_noise.md` — chord level used for diagnostic
  attachment, NOT for prediction at note level
- `project_5level_segmentation_substrate.md` — the chord substrate this
  builds on
- `project_context_filter_vs_tier.md` — the architectural rule this implements
- `feedback_quantile_selection_overfit.md` — the warning we keep ignoring
  about quantile-cell selection bias


## Source: project_b_c_naming_convention.md

---
name: project-b-c-naming-convention
description: "B-prefix for production candidates (validated/active in stack), C-prefix for failed candidates (research artifacts only, kept for audit but not in production). Numbering is contiguous within each prefix."
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

Naming convention for regret-oracle models, established 2026-05-17.

**B-prefix**: production candidates. Validated, integrated into the
forward pass / composite pipeline, or actively under validation.
Numbering contiguous (B1, B2, B3, ... no gaps).

**C-prefix**: failed candidates. Research artifacts kept for audit but
NOT in production. Marked as failed in their docstring/header so future
sessions don't accidentally promote them. Numbering can match the
original B-slot they once occupied (so C9 was once B9 before being
demoted).

**Rule**: when a B-model fails validation:
  1. Rename ALL its files (tool .py, cache .parquet/.pt, report .txt)
     using `git mv` from b{N}_* -> c{N}_*
  2. Update all internal references (paths, var names, class names,
     prose) to c{N} prefix
  3. Mark docstring with "(failed candidate)" suffix
  4. Free the B{N} slot for the next production candidate
  5. Save the rename in the daily journal + finding report

**Current state (2026-05-18)**:
  B1  pivot-imminent (HGB)                     ACTIVE
  B2  fakeout (HGB)                            ACTIVE
  B3  time-to-pivot (HGB regressor)            ACTIVE
  B4  pivot-region (HGB)                       ACTIVE
  B5  leg-phase 3-class (HGB)                  ACTIVE
  B6  directional-pivot 3-class (HGB)          ACTIVE
  B7  leg-sizer (HGB regressor)                ACTIVE -- production entry sizer
  B8  hour-risk (HGB regressor)                ACTIVE
  B9  during-trade remaining-amplitude         ACTIVE -- L5 trade-level (validated 2026-05-17 OOS +$67/day, cross-sample confirmed 2026-05-18 +$66/day CI [+$41,+$94])
  B10 vol-regime sizer (day-level)             ACTIVE -- INVERTED action (boost high-vol, cap low-vol) -- validated 2026-05-18 LATE OOS +$69/day CI [+$7, +$144]
  -----
  B11 RESERVED for next promotion
  B12 RESERVED for next promotion
  -----
  C9  LSTM leg-sizer                           FAILED (was B9 originally, lost to B7 GBM)
  C10 LSTM direct-trade                        FAILED (was B10 originally, OOS Pearson -0.02)
  C11 bad-trade binary-cut detector            FAILED (was B9 briefly 2026-05-17; both v1 + v2 walk-forward 0/N significant)
  C12 imminent-exit binary-preempt             FAILED (0/15 ops sig negative)
  C13 leg-phase phase-gated-pyramid-cap         FAILED (Pearson 0.32-0.48 learnable but action zero incremental)
  C14 vol-regime defensive gate                 FAILED (defensive action BACKWARDS -- high-vol = best days; learned to invert action surface -> promoted v2 -> B10)
  C15 pyramid-confidence attenuator             FAILED (AUC 0.883 strong signal but attenuation hurts at all recalls -- 88.4% of B9 pyramids pay off, attenuating any is net loss)

**Two-stage workflow (effective 2026-05-18)**:
  STAGE 1 - candidate: new during-trade retarget research takes lowest available C-slot.
    Build, walk-forward IS, OOS-sealed test. If validates, STAGE 2; else stays as C-artifact.
  STAGE 2 - promotion: validated candidate gets renamed to lowest available B-slot.
    File rename (git mv c{N}_* b{M}_*), internal reference update, memory update.

**Why:** Contiguous B-numbering preserves a clean operational view of
the production stack. C-prefix preserves research artifacts (we don't
delete failed experiments — they teach us what doesn't work) without
polluting the production numbering.

**How to apply:** When the user mentions B{N}, assume production
candidate unless prefixed with C. When proposing a new model, take the
LOWEST available B-slot (i.e., reclaim freed slots from C-renamed
predecessors before extending the sequence).

Related: [[project_during_trade_b_stack.md]] (B9 = first during-trade
model in development).


## Source: project_cleanup_backlog.md

---
name: cleanup backlog — architectural debt
description: Accumulated refactoring tasks from the breakthrough sessions (2026-03-16 to 2026-03-18)
type: project
---

## Architecture Cleanups

1. ~~**Make continuous IS→OOS the default**~~ — DONE (2026-03-18). `--continuous` flag removed.
   Default pipeline runs IS→checkpoint→OOS. `--oos` = resume from checkpoint.

2. **Unify data folders** — one `DATA/ATLAS/` with all data. Oracle coverage boundary determined
   by Phase 1 discovery range, not folder structure. No copy/paste to expand.

3. **Remove `oos_mode` behavioral branches** — 50 references, most are naming. Replace with
   `phase` label ('is'/'oos') that only affects file names and oracle availability.

4. **Delete IS inline forward pass** — 2,000 lines with pattern_map that's now dead code.
   IS uses compressed + peak (same as OOS). The `else` branch after `if True:` is unreachable.

5. **Three-layer separation** — data injector / engine+oracle / reporting. Oracle should
   be fully decoupled from engine (post-processing, not inline).

6. **Remove `_print_oos_comparison`** — OOS2 deleted, this function is dead.

7. **Remove dead methods** — `run_strategy_selection()` (~line 4854), `run_oos3_standalone()` (~line 2459),
   `_print_oos_comparison()`. All callers deleted 2026-03-18.

8. **Dead imports** — wave_rider, quantum_field_engine references throughout.

9. **Unused config fields** in TradingConfig — audit and remove.

10. **Old gate stats tracking** — nobody reads the gate funnel counters.

## Code Cleanups

11. **`if True:` wrapper** on line ~1566 — remove the wrapper, just use the code directly.

12. **Duplicate peak detection state tracking** — two places update `_peak_prev_pc/fm`.
    Should be one place, updated every bar.

13. **Dashboard chart save** — PostScript files accumulate. Add cleanup or age-out.

14. **Unicode chars** — remaining emoji/unicode in log messages that crash cp1252.

15. **`.gitignore`** — trade_replays JSON files are 60MB+. Should be gitignored.

## Data Pipeline

16. **Resample automation** — when live aggregator saves new 15s/1s, auto-resample
    to other TFs. Currently manual (`tools/` script).

17. **1s bar persistence** in live — bar_aggregator only persists 15s. Need 1s for
    1s worker on restart.

18. **TBN parquet loading** — loads from ATLAS parquets when available, falls back
    to resampling. Verify it's working correctly (was just added).

## Future Avenues (not bugs)

19. **Loose OOS gate** — pattern_type alone (no cascade/struct) → 3x trades but
    lower $/trade. Needs quality filter before enabling.

20. **Phase 1 discovery using peak detection** — rebuild templates around reversal
    patterns instead of z-score threshold crossings.

21. **Maintenance window discovery** — run Phase 1 on daily data during CME halt.

22. **Weekend recalibration** — weekly oracle + brain retrain from live trades.


## Source: project_conditional_probability_table_2026_05_21.md

---
name: conditional-probability-table-2026-05-21
description: The conditional probability table — a 4-entry diagnostic if-this-then-that table over zigzag-leg events, built 2026-05-21; diagnostic asset, not yet in production
metadata:
  type: project
---

A diagnostic research thread: an empirical conditional-probability table that
answers "if-this-then-that" questions about zigzag-leg events.

**Why:** User reframe 2026-05-21 — "I don't want to catch more, I want to be
able to DIAGNOSE that." The goal is not a new trading model but a probability
table — the ATR zigzag pinpoints events (chop, fakeout, leg age), the table
attaches probability estimates. A local-LLM (Llama) layer to chain the
if-this-then-that reasoning was discussed and back-burnered.

**How to apply:** This is a DIAGNOSTIC asset — it has NOT changed production
code. When a future session proposes a leg-structure trading rule, check the
table first; several "obvious" edges are already measured and turn out weak.

Four entries (reports in `reports/findings/oos_bad_days/2026-05-21_*`, tools
at `tools/`):
- Entry 1 `conditional_probability_table.py` — chop begets chop: after 3
  consecutive low-range legs P(next low) = 65% vs 33% base. Chop is sticky.
- Entry 2 `leg_chop_survival.py` — early chop forecasts a ~2x longer leg
  (fixed-early-window; v1 was confounded).
- Entry 3 `trend_continuation.py` — after a chop/fakeout the preceding trend
  continues only ~53-56% vs ~50-52% base (~+4pp). WEAK. The mechanical
  counter-trend entry is slightly wrong, but the directional edge is too
  small for a B1-7 gate to fix profitably (B1-B6 augmentation already tested
  at -$76/day; direction is not the lever).
- Entry 4 `leg_age_hazard.py` — leg-death hazard is HUMP-shaped: peaks at a
  ~5-min danger window, then declines. NOT monotone exhaustion — a pure
  time-stop is unsupported; only a danger-window-aware check is backed. The
  steep early arm is partly the min_bars=36 (3-min minimum leg) floor.

Method discipline for this thread: [[zigzag-conditional-table-confounds]]
(five confounds caught and fixed during the arc).

Candidate next entry (entry 5): amplitude-expended survival — P(leg
continues) given it has already moved P points, the price-distance analog of
entry 4's time hazard. Grew out of [[oos-bad-days-2026-05-21]] but is its own
thread (diagnose, not lift bad days).


## Source: project_context_filter_vs_tier.md

---
name: Architectural distinction — context filters vs tiers (locked 2026-05-09)
description: Robustness filters and macro-pivot detectors are context layers conditioning when tiers fire, not tiers themselves
type: project
---

**Locked 2026-05-09 evening** after the FadeAtBand rejection.

## The distinction

```
TIER             = a strategy that fires ENTRIES with a direction
                   (FADE_CALM, KILL_SHOT, RIDE_AGAINST, FreightTrain etc.)

CONTEXT FILTER   = a gate/conditioner that decides WHEN tiers should fire
                   or WHEN their entries are valid
                   (4 robustness filters, CRM macro detector, regime label,
                    hurst, swing_noise, time-of-day, calendar event)
```

Tiers PRODUCE trade signals. Context filters CONDITION which signals are
allowed through to execution.

## Why the distinction matters

When I tested FadeAtBand "with its filters" as a single tier, I conflated
two things:
1. The entry rule (5s touches 15m ±2σ → fade to 5m mean) — this had no edge
2. The 4 robustness filters — these DID gate macro-event blowups

Treating them as one bundle made the verdict "FadeAtBand is rejected"
which is misleading. The correct verdict is:
- ENTRY RULE rejected
- FILTERS survive as reusable context components

The CRM macro-pivot detector v2 is the same: it's not a strategy, it's a
context filter that wraps any reversion strategy to suppress macro-impulse
days.

## Implications for system design

**Next-iteration entries** should be built as: entry rule + library of
context filters that can be turned on/off independently.

```
Strategy = Entry rule  ⊗  Context filter stack
         = (5s touches band)  ⊗  (hurst<0.60)  ⊗  (require_divergence)
                              ⊗  (CRM not in macro-impulse)
                              ⊗  (regime ≠ DOWN_SMOOTH)  ...
```

This lets us:
1. Test entry rules in isolation (no filter contamination)
2. Test filters in isolation (apply same filter to multiple entry rules,
   measure incremental contribution)
3. Build a context-filter LIBRARY — each filter is a reusable bool function

## What this means for the 9-layer probability stack

The 9 ExNMP tiers were tier-level strategies, but the empirical first-passage
probability table that's pending should be ORGANIZED AS A FILTER GRID:

```
P(reversion succeeds | state) = base table
                              + state filter   (current 3-body state)
                              + regime filter  (UP/DOWN/FLAT × SMOOTH/CHOPPY)
                              + macro filter   (CRM not in impulse)
                              + tod filter     (US morning vs lunch vs close)
                              + cal filter     (FOMC/NFP/CPI off)
                              + hurst filter
                              + sn filter
```

Each filter is a conditional on the table. The fully-conditioned cell tells
us "P(this entry succeeds given THIS exact context)". A tier becomes a
named selection of filter values.

## What we keep from FadeAtBand experiment

As context filters (proven on 2026-05-09):
- `hurst_5m < 0.60` — gates trend regimes
- `max_counter_trend_vel = 25.0` — gates against-strong-momentum
- `require_divergence` between 1m and 5m means — confirms reversion structure
- `confirm_bars = 6` — confirms persistent band-touch

The macro-gate IS the 4-filter stack here. CRM detector is the next-level
filter to add (impulse-day suppression). These compose multiplicatively in
the probability stack.

## Naming convention

When adding to `training_iso_v2/` or any future framework:
- Tiers go under `strategies/`
- Context filters should live under a new `filters/` directory
- Filters expose a single signature: `def __call__(state) -> bool`
- Tests apply filters as `entry_signal AND f1(state) AND f2(state) AND ...`

Nothing in the codebase enforces this yet — but starting today, every new
"check" we add must be classified as either an entry-rule component or a
context filter, and lived in the right place.


## Source: project_counterfactual_engine.md

---
name: Counterfactual Engine (Goat Brain)
description: Every trade and skip spawns phantom trades with alternative parameters. The optimization surface emerges from real-time what-if analysis. Foundation for the goat brain.
type: project
---

Every decision spawns phantom trades testing alternative thresholds:
- Skip → phantom enters anyway with different gate settings
- Trade → phantoms test different exit thresholds (giveback 20%-70%)
- Each phantom tracked for 80 bars (20 min), same exit cascade

After N phantoms complete, optimal parameter values emerge from the data.
System adapts thresholds in real-time. No offline optimization needed.

**Why:** The cat/crow/monkey learn WHAT to trade. The goat learns HOW to tune
ALL parameters simultaneously from continuous counterfactual evaluation.

**How to apply:**
- IS: synchronous (data in _states_map). Report at end of IS.
- OOS: validates IS-learned parameters hold out of sample.
- Live: async background workers. Auto-tune or flag for human review.
- Spec: `docs/specs/COUNTERFACTUAL_ENGINE.md`

**Evolution:** Cat (regime) → Crow (seeds) → Monkey (CNN) → Goat (counterfactual).
The goat doesn't replace the others — it TUNES them.


## Source: project_delta_architecture.md

---
name: Delta vs Absolute Architecture
description: ALL engine computations are cumsum-based (absolute). This is why OOS != Live. Delta-based engine would fix parity permanently.
type: project
---

ALL MarketState fields depend on cumulative history (bar count from start):
- F_momentum = kp*error + ki*CUMSUM + kd*derror (ki*cumsum grows forever)
- DMI = Wilder smoothing (exponential cumsum)
- ADX = double smoothed cumsum
- P_at_center = regression window (history-dependent)
- velocity = derivative of cumsum
- sigma = regression band width (history window)

**Why:** PID controller integral term (ki * cumsum) accumulates over all bars.
OOS starts after 379K IS bars -> deep cumsum. Live starts from zero -> shallow.
Same bar, same price, different F_momentum (218 vs 12).

**How to apply:**
- Short-term: ATLAS warmup loads pre-computed states (deployed 2026-03-20)
- Long-term: refactor engine to output bar-to-bar DELTAS instead of absolute values
- Crow brain features: use deltas, not absolute MarketState values
- CNN input: trajectory of changes over 10 bars, not endpoint values
- Sensor gates: check "is decreasing?" not "is < threshold"
- This eliminates warmup, pre-computed states, and parity gap permanently


## Source: project_dissect_old_brain_2026_03_07.md

---
name: Dissection of pre-snowflake / star-schema Bayesian brain (worktree at 09cd30d8, 2026-03-07)
description: Concrete file-level mapping of the original Bayesian brain + K-means template architecture from the worktree dissection on 2026-05-09 evening. Maps each old file to the V0 chord-keyed equivalent.
type: project
---

**Worktree**: `c:/tmp/dissect-old-bayesian-brain` (commit 09cd30d8, March 7 2026)
**Pre-snowflake spec**: commit 47fa8b3a doc `docs/JULES_SNOWFLAKE_BASELINE.md` (250 lines)
**Pre-snowflake checkpoint commit referenced as `3d0c1b8`** — that commit is on a deleted branch, not in current refs

## Architecture map (file by file)

### `core/fractal_clustering.py` (658 lines) — the K-means template engine

The substrate red flag concentrated here.

**16-D feature vector per pattern** (the obscured multi-D concept):
```
[|z_score|, |velocity|, |momentum|, coherence,
 log2(tf_seconds), depth, parent_is_roche,
 self_adx, self_hurst, self_dmi_diff,
 parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
 self_pid, self_osc_coh]
```

Mixed semantics: physics measures + fractal-tree position (depth, parent context)
+ orientation flags (root_is_roche) + multi-TF coherence. K-means on this blob
produced centroids that no longer corresponded to recognizable patterns.

**`PatternTemplate` dataclass — the star-schema fact row**:
```
template_id            cluster id
centroid               16-D K-means centroid
member_count           # patterns in this cluster
patterns               list of original PatternEvents
physics_variance       cluster tightness
transition_map         {next_template_id: probability}  (Markov)
expected_value         (WR * AvgWin) - (LossR * AvgLoss)
outcome_variance       std(PnL)
avg_drawdown           mean MAE
risk_score, risk_variance
stats_win_rate         fraction with |oracle_marker| >= 1
stats_expectancy       mean(mfe - mae)
stats_mega_rate        fraction with |oracle_marker| == 2
long_bias, short_bias  fraction of positive/negative markers
```

**Recursive K-means**: a parent cluster splits into children if z-variance still
high AND children have >= MIN_SAMPLES_PER_CLUSTER members; depth capped at 5.

**Pre-snowflake spec fix** (commit 47fa8b3a, Feb 21 2026):
1. Raise MIN_PATTERNS_FOR_SPLIT from 20 to 30
2. Replace within-sample silhouette gate with adj-R² gain check
   (split accepted only if weighted_children_adj_R² - parent_adj_R² >= 0.05)
3. Add avg_mfe_bar / p75_mfe_bar fields to template (time-scale of MFE peak)
4. TEMPLATE_MIN_MEMBERS_FOR_STATS raised 5 -> 20

The adj-R² gain test is a BAYESIAN-FLAVOR overfit gate — only split if predictive
power genuinely improves. Today's r2adj_5m primitive uses the same statistical
concept (adjusted R² of a linear fit) but applied at 5min cadence to 5s closes
rather than to the MFE prediction model.

### `core/bayesian_brain.py` (462 lines) — the brain

```
table[StateVector]            -> {wins, losses, total}     # state-keyed
dir_table[(tid, direction)]   -> {wins, losses, total}     # template x direction
dir_bias[tid]                 -> {long_w, long_l, short_w, short_l}
```

`update(TradeOutcome)` writes to all three tables on each completed trade.
TradeOutcome has: state vector, entry/exit prices, pnl, result, exit_reason,
direction, template_id.

The `befdc2df` (March 8 2026) commit added per-template-direction PnL tracking:
`get_expected_pnl(tid, side)` returns running average $/trade, used to classify
upcoming trades as pos-EV / neg-EV / unknown.

### `core/execution_engine.py` (933 lines) — the gate cascade

Phase-1 gates (run on ALL candidates):
- Gate 0       headroom, physics, noise filtering (multiple sub-rules:
                 noise / r3_struct / r3_snap / r4_nightmare / r4_struct /
                 hurst / momentum / tunnel)
- Gate 0.5     extra filter
- Gate 1       distance-to-centroid (must match a known template)
- Gate 2       fractal depth filter

Phase-2 gates (run on the COMPETITION WINNER only):
- Direction cascade
- Gate 3       conviction
- Gate 4       momentum alignment

The "9 gates" the user described maps to gate 0 (with 5+ sub-rules counted as
distinct gates) + gates 0.5 / 1 / 2 / 3 / 4 ≈ 9 gating decisions.

Cluster match (Gate 1) uses scaled-Euclidean to nearest centroid on the 16-D
feature vector; tid -> pattern_library entry holds the brain's posteriors.

### `core/state_vector.py` (80 lines) — the brain keys

Compact representation of market state as a single hashable vector — the
`table[StateVector]` key in bayesian_brain.

### `core/three_body_state.py` — substrate

Where today's "3-body" terminology was established (M_close, M_high, M_low).
The framework lived through the 2026-05-09 work even after the original brain
was deleted.

### `core/timeframe_belief_network.py` — TBN

Multi-TF worker consensus. Maps to the modern `core/tbn` infrastructure.

### `core/quantum_field_engine.py` — physics-paradigm engine

Purged in commit 841e5dc6 ('refactor: metaphor purge, code consolidation,
CPU path removal, compressed replay'). Used physics terms (tunnel probability,
analytical fields) that were removed when CLAUDE.md banned physics metaphors
in production code.

## Mapping to V0 (chord-keyed Bayesian table)

| old file/concept | replaced by |
|------------------|-------------|
| 16-D feature vector (mixed semantics) | 5-axis primitive chord (statistical, single-meaning per axis) |
| K-means recursive splitting | quantile bucketing (canonical cells, no centroid averaging) |
| MIN_SAMPLES_PER_CLUSTER 30 + adj-R² fission gate | hierarchical shrinkage cell -> axis-marginal -> universal |
| PatternTemplate star schema (template_id keyed) | bayesian_table.py cells (chord-tuple keyed) |
| transition_map (Markov next-template) | DEFERRED - not needed for V0 |
| `bayesian_brain.table[StateVector]` | replaced by chord lookup |
| `dir_table[(tid, direction)]` | (chord, tier) cells in `training_iso_v2/bayesian_table.py` |
| `dir_bias[tid]` | per-tier direction bias rolled into per-tier $/trade per cell |
| Gate 0 (headroom/physics) | NORMAL-branch context filter on z, hurst, momentum |
| Gate 1 (cluster match) | OBSOLETE - chord IS the cluster identity |
| Gate 2 (depth) | DEFERRED - fractal depth not in current substrate |
| Gate 3 (conviction) | per-cell P_cascade or per-tier expected $/trade gate |
| Gate 4 (momentum align) | maps to slope/curv axis combination in chord |

## Outstanding questions for the V0 build

1. Do we need the transition_map (Markov next-cell) feature? The original brain
   tracked which template tended to fire next; could be useful for sequence-aware
   gating.
2. Do we restore the 'depth' / fractal-tree-position context? The V0 chord drops
   it, but the original brain considered fractal depth meaningful.
3. The original brain stored a `risk_score` that combined drawdown + variance
   into a single scalar; do we want this as a derived field per chord cell?

## Cleanup

To remove the worktree when dissection is done:
```
git worktree remove c:/tmp/dissect-old-bayesian-brain
```

The worktree adds no risk to main. Reading is read-only; even if we accidentally
modify files in the worktree they're disconnected from any branch ref.


## Source: project_during_trade_b_stack.md

---
name: project-during-trade-b-stack
description: Re-conceiving B-models as during-trade state estimators rather than entry-time snapshots. L5 execution-layer paradigm enabled by the trade trajectory dataset built 2026-05-17.
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User insight 2026-05-17 (after building IS trade-trajectory dataset):
"a lot of B should be made during trades not at entry"

This re-conceives the entire B-model stack. Every existing B-model
(B1-B8) is currently a SNAPSHOT-AT-ENTRY predictor. The architectural
shift: each could be retargeted as a TIME-VARYING state estimator
that updates as the trade unfolds.

**Why this is structurally correct**: data processing inequality.
P(outcome | entry features + trajectory to K) >= P(outcome | entry features alone).
More information can only help (or be ignored).

**During-trade analogs of existing B-models**:

| Pre-entry B | During-trade analog | Operational action |
|---|---|---|
| B1 pivot-imminent | "is opposite pivot forming?" given K held | Early exit (B-cut/B9) |
| B2 fakeout | "am I being faked out NOW?" | Cut + flag direction |
| B4 pivot-region | "is opposite pivot-zone forming around me?" | Tighten exit |
| B5 leg-phase | "where in MY leg am I now?" | Time exit / pyramid timing |
| B6 directional pivot | "next pivot direction from HERE" | Cut or flip |
| B7 leg sizer | REMAINING amplitude from here (B10 candidate) | Pyramid, partial exit |
| B8 hour-risk | 60min risk update at K | Position cap adjust |

Each produces a different operational signal — cut, flip, pyramid, tighten,
hold. The full L5 execution layer is a multi-channel in-flight monitor,
not just one model.

**Enabler**: `reports/findings/regret_oracle/trade_trajectory_IS.parquet`
(74,976 rows = 17,748 legs x 5 K horizons {5, 10, 30, 60, 120} in 5s units).
Built 2026-05-17 from `is_hardened_legs.csv` (17,767 legs / 275 days,
$690/day flat IS baseline).

**Why:** During-trade B-models reduce variance of selector decisions and
unlock execution actions (cut/flip/pyramid) that snapshot-at-entry models
cannot access. The trajectory dataset enables this; before it existed,
training data was unavailable. The morning's trail-tightening failure
(2026-05-17 early) used composite signals at entry, NOT during-trade
features — different paradigm.

**How to apply**:
- L5 execution layer in the regret-oracle 6-layer architecture
- Adds streaming feature requirement: V2 features at entry+K, K+1, ...
  for K in {5, 10, 30, 60, 120} bars (5s units) while position is open
- Requires hybrid NT8 + Python sidecar with per-K state queries
  (not just R-trigger event triggers)
- Each during-trade B must walk-forward validate within IS, OOS held out

**Risk to manage**: more snapshots per leg = more degrees of freedom =
more overfitting potential. With 17k legs x 5 K horizons x 184 V2 features,
spurious patterns are easy to fit. Walk-forward + bootstrap CI mandatory
on every retargeted B. Per [[feedback_quantile_selection_overfit]] —
same risk class.

**Sequenced execution plan**:
1. Walk-forward CI on B-cut (B9, V2-only, target exit_pnl<-$50). If CI
   positive, paradigm is validated on ONE task.
2. Retarget B7 to remaining-amplitude (call it B10). Same trajectory
   dataset, different target. Second proof point.
3. If 1+2 succeed: systematically retarget B1/B5/B6 to during-trade.
   ~2 weeks of work for full L5 execution layer.
4. If step 1 fails (CI crosses zero): kill paradigm, don't waste 2 weeks.

**Current state (2026-05-17 evening)**: diagnostic on V2-only at K=30-60
shows AUC 0.89-0.94 with naive +$43-53/day on one val fold for B-cut.
NO CI yet. Walk-forward is the next step.

Related: [[project_zigzag_calibration]] (Python pipeline calibration),
[[feedback_quantile_selection_overfit]] (overfit risk), [[user_collaboration_protocol]]
(don't pivot mid-experiment — finish B9 walk-forward before retargeting).


## Source: project_fade_at_band_rejected.md

---
name: FadeAtBand v1 rejected (chart-validated framework, IS-validation killed it)
description: 5s-touch-15m-2sigma fade rule with 4 robustness filters; IS net -$17/day disqualifies despite OOS +$29/day driven by 6-day fluke
type: project
---

**STATUS**: REJECTED 2026-05-09. Do not deploy. Do not "fix" without changing the entry rule.

**Hypothesis tested**: at 5s grid, when price touches 15m ±2σ_close band,
fade to 5m M_close. Robustness filters: hurst<0.60, max_counter_trend_vel=25,
require_divergence between 1m and 5m means, confirm_bars=6.

**Result**:
```
                IS (261 days)              OOS (68 days)
total_PnL       -$4,506                    +$1,964
$/day           -$17.27                    +$28.88
$/trade         -$0.18                     +$0.30
PF-WR           negative                   +0.07 (break-even)
```

**Why OOS looked positive but IS killed it**:
- 6 OOS FLAT_SMOOTH days delivered +$1,650 of the +$1,964 OOS total
- IS FLAT_SMOOTH (53 days): only +$0.13/trade — 22× weaker
- 3 of 6 regime cells (UP_CHOPPY, FLAT_CHOPPY, DOWN_CHOPPY) flip sign IS↔OOS
- Textbook overfit fingerprint per `feedback_quantile_selection_overfit.md`

**What we keep from this experiment**:
1. The 4 robustness filters DO suppress macro-event blowups
   - 2026_03_03 (v1 reversion lost -$1,324): FadeAtBand made +$51, n=125
   - 2026_02_12 macro-pivot day: FadeAtBand made -$3.50 (flat)
   - These filters are **reusable for any reversion strategy** that needs
     macro-day suppression
2. The 2D regime split shows real structure but day-aggregate labels are
   not exploitable as an intra-day filter; signs flip across IS/OOS

**What does NOT work**:
- Entry rule "5s touches 15m ±2σ → fade to 5m mean" with k=2.0
- Default exit suite tuned on this rule (TargetMeanReached, ZSeRetracement,
  Z15sOvershoot, MFEPriceTarget, MFEArmedGiveback) — calibration is noise
  on top of a non-edge entry
- Regime-router approach using `regime_2d` daily labels — no regime is
  stably positive across both IS and OOS

**Next-iteration ideas (NOT yet tested)**:
- k_sigma=2.5 or 3.0 (rarer, higher-conviction band touches)
- Use M_close ±3σ_close OUTER WALL trigger (~1.5% of bars, much rarer)
  instead of M_high/M_low ±2σ
- Build empirical first-passage probability table conditioned on
  (state, regime, hurst, sn, tod, dow, cal_event) FIRST, then design
  entry rule against the cells with stable positive structure on IS

**Pickles preserved as overfit research artifact**:
  training_iso_v2/output/is_FADE_AT_BAND.pkl
  training_iso_v2/output/oos_FADE_AT_BAND.pkl
  training_iso_v2/output/{is,oos}_FADE_AT_BAND_regret.pkl

**Lesson reinforced**: small-sample OOS positivity (68 days) does NOT
justify shipping when IS (261 days) is net negative. The CLAUDE.md
anti-doom-cascade rule cuts both ways — don't claim positive edge from
+$28.88/day when PF-WR is +0.07.


## Source: project_feature_tree.md

---
name: Feature tree — 3 levels max, organic expansion like colors
description: Features built organically from 2 primaries (Price, Time) through 3 layers max. Each feature must name its parent measurements and the question it answers.
type: project
---

## Primaries (can't be derived)
- **Price** — where the market is
- **Time** — when

## Secondary (one operation on primaries)

**Kinematic** (mix two primaries):
- velocity = dPrice/dTime (how fast?)
- DMI diff = directional movement from Price highs vs lows (who's winning?)
- volume = market reaction to Price at Time (the mass — not truly independent)

**Statistical** (characterize one primary's distribution):
- mean(Price, window) = center
- std(Price, window) = spread / volatility
- median, skew, kurtosis — shape descriptors

## Tertiary (mix secondaries, or primary + secondary)

- acceleration = dVelocity/dTime (is force changing?)
- z-score = (Price - mean) / std (how unusual is this price?)
- fib_position = (Price - low) / (high - low) (where in range?)
- variance_ratio = std(short) / std(long) (trending or reverting?)
- price × volume = is the move backed by participation? (heavy or light?)
- DMI × volume exhaustion = DMI extreme + volume collapse (reversal coming?)
- session_phase = f(Time) categorical (when in the day?)
- higher_tf_z = z-score at 1h scale (structural position)

## Rules
1. **3 levels max.** If a question can't be answered in 3 layers from Price and Time, it's the wrong question.
2. **Name the parents.** Every feature must trace to its parent measurements.
3. **Name the question.** If you can't state in one sentence what it measures, it doesn't belong.
4. **No brown paint.** Don't mix 5 things at once. Mix two, see what you get, then mix again.
5. **Volume is secondary, not primary.** It's the market's reaction to Price at Time — a mass/force measurement, not independent.
6. **Statistical features are secondary.** std, mean, variance characterize the SHAPE of a primary — one operation, one base.
7. **Test before adding level 4:** What question does it answer that level 3 can't? If none, you don't need it.

**Why:** Old system had level 4+ features (F_momentum = PID of z-score, coherence = entropy of z-score std) that answered questions nobody asked (r≈0). Features should expand organically like colors — primary → secondary → tertiary — each layer intentional.

## Implemented: 7 Features × 10 TFs = 70D (`core/grounded_features.py`)
| Feature | Level | Question |
|---------|-------|----------|
| dmi_diff | 2 | Who's winning? |
| dmi_gap | 2 | How dominant? |
| volume_rel | 2 | Participation vs normal? |
| dir_volume | 3 | Does volume back the direction? |
| velocity | 2 | How fast? |
| z_se | 3 | Statistically significant deviation? |
| price_accel | 3 | Is force changing? |

TFs: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1D, 1W
Higher TFs encode lookback — no sliding window needed.

## Noise & Coherence (rebuilt)
- Noise = std(dP) at scale below your horizon. One TF's signal = next TF's noise.
- Coherence = agreement count across grounded directional features. High = late, Low = edge.
- Variance ratio = std(short)/std(long) — replaces Hurst AND ADX.

## Liquidity (latent variable)
- Can't measure, only observe effects (volume spike, price rejection, time at price)
- Fibonacci ≈ where algos cluster orders. Volume profile = direct measurement.
- Three-body analogy: Price/Volume/Time interacting, solved by pattern matching.

## TCN v5 Architecture (future)
- Input: (10 TFs, 7 features) as 2D matrix, NOT flattened
- TCN convolves across TF axis with dilations — discovers cross-TF relationships
- Replaces K-means as template matcher. See `memory/research_tcn_v5.md`


## Source: project_frozen_sfe_cache.md

---
name: Frozen SFE cache bug (fixed 2026-04-16)
description: LiveFeatureEngine SFE cache keyed on valid_idx alone collided after 5000-bar trim. Stale SFE state since mid-Feb. Every live session before fix traded on frozen features.
type: project
---

# Frozen SFE cache bug — fixed 2026-04-16

**What happened:** `training/live_feature_engine.py`'s `_compute_features`
cached SFE state keyed on `valid_idx` (count of valid bars up to current
timestamp). After the bar store hit its 5000-bar trim limit, every new
bar's `valid_idx` stayed at 5000 (newest always at end after trim), so
the cache returned stale SFE state from whichever bar first reached 5000.

**Observable symptom:** In the 2026-04-16 Phase 6 live session, z was
frozen at +3.95 for 1,346 bars (2 hours) during an MTF_BREAKOUT trade.
Only 2 unique z values across the entire trade (-2.07 and +3.95). Every
z-dependent exit (`fade_mean_reached`, `fade_z_expanding`, oscillation
tracking) never fired.

**Broken since mid-February.** Every live session from mid-Feb through
2026-04-16 was trading on frozen features. All live PnL from that window
is noise relative to the actual engine signal.

**Fix:**
```python
# BEFORE: cache key = valid_idx (collides after trim)
if cached and cached[0] == valid_idx:

# AFTER: cache key = (valid_idx, latest_bar_ts) — invalidates on new bar
if cached and cached[0] == (valid_idx, latest_bar_ts):
```

**Also fixed in same session:**
- `_find_today_start` used UTC midnight — fixed to match batch
  `get_day_start` file boundaries
- 5000-bar trim cap removed (root cause, not symptom)
- Mock mode `exclude_day` extended to `day_name >= exclude_day` so LFE
  store doesn't contain bars from the replay day or later
- Accumulators reset in mock mode to avoid stale carry-over

**First honest live session post-fix**: $900 peak PnL on first 2-hour
trade — frozen z had been hiding real edge. (Session was later stopped
when chain trades died on a bad MTF_BREAKOUT signal.)

**How to apply:** Any future caching in LFE or live components must key
on `(logical_state, latest_bar_ts)` — state identity alone isn't enough
if the state re-populates the same slot.


## Source: project_honest_baseline_2026_04_17.md

---
name: Honest baseline (post-lookahead-fix)
description: After fixing build_dataset lookahead, baseline dropped from $740/day to -$164/day. All 8 tiers are ~50% counter-flip coin flips. Peak-physics exits disproved.
type: project
---

# Honest baseline — 2026-04-17

## The fix that changed everything

`training/build_dataset.py` had lookahead bias in higher-TF aggregation:
```python
# BEFORE (lookahead — picks TF bar whose START is at target_ts,
#         but OHLCV aggregates forward 5s bars to bar end)
idx = np.searchsorted(tf_ts, target_ts, side='right') - 1

# AFTER (only uses completed bars — shift back by period)
idx = np.searchsorted(tf_ts, target_ts - period, 'right') - 1
```

Feature folders also reorganized — now `DATA/ATLAS/FEATURES_5s/` and
`DATA/ATLAS_OOS/FEATURES_5s/` instead of top-level `DATA/FEATURES_79D_1m/`.

**Why:** every 2025 OOS number with `DATA/FEATURES_79D_1m/` is
contaminated. Previous $740/day baseline was lookahead, not edge.

**How to apply:** Any analysis using features from before 2026-04-17 must
be re-run on the new features. The old feature directory is gone.

## Honest baseline numbers

After the fix, full pipeline: **-$164/day IS on 348 days of 2025**.
Chains alone accounted for $157/day of loss — they amplified bad signals.

Per-tier isolated (no chains, no catch-all):

| Tier | N | $/day |
|---|---|---|
| RIDE_AGAINST | 39,721 | -$11 |
| FADE_CALM | 24,039 | -$16 |
| MTF_BREAKOUT | 5,961 | +$4 |
| FADE_AGAINST | 4,532 | +$5 |
| KILL_SHOT | 4,411 | -$2 |
| CASCADE | 1,270 | +$6 |
| MTF_EXHAUSTION | 233 | +$9 |
| FREIGHT_TRAIN | 34 | +$61 (n too small) |

Every tier at the noise floor.

## KILL_SHOT peak physics disproved

Path-level analysis on 2,043 trades with peak > $3:
- 1m velocity flips against trade at peak: **3.3%**
- 1m acceleration flips: **0.2%**
- Wick on other side (>30% jump): **6.8%**
- Largest Cohen-d across peak: **0.19** (1m_wick_ratio)

**There is no detectable physics of the peak.** The peak is a statistical
maximum over noise. Back-test confirmed: natural exit (+$11.61/trade)
beats every physics-based rule including 50% trail (+$3.40), fixed
targets, velocity/accel flips.

Implication: KILL_SHOT loss is an **entry filter** problem, not exits.
~1,720 trades with peak ≤$3 are structural losers.

## All tiers are coin flips on direction

Regret analysis on all 8 isolated tier pickles:

| Tier | % counter-flip | Counter WR |
|---|---|---|
| All tiers | ~49% | ~40% |

**~50% counter-flip means no directional edge.** Regret labels are
meaningful (counter-labeled trades have 38-45% actual WR, so regret
ranks correctly), but the SAME/COUNTER boundary is near-random in
91D feature space.

nn_v2 on NMP worked because NMP was 30-35% counter — there was real
direction. Here every tier is a coin flip dressed as physics.

Oracle upper bound (flip at exit, no peak-chasing): +$2,183/day pooled.
Realistic CNN at 65% accuracy: ~$900-$1,300/day — IF separability exists.

## Data caveat — regret LOOKAHEAD

`training/regret.py` has `LOOKAHEAD = 360` commented "30 min at 5s
resolution" but loads 1m price data → actual window is 6 hours. Every
trade's "optimal" becomes "hold 6 hours and catch the biggest swing."
99% of best_action labels are same_extended or counter_extended.

`flip_at_exit` is still clean (uses actual exit bar, no peak-chasing).
Raw counter-flip % may be slightly inflated by the 6-hour window. Need
to cap LOOKAHEAD to 15-30 min (or use 5s prices) and re-run.

## Decision tree for next session

1. **Fix regret.py LOOKAHEAD** (30 min) — remove 6-hour distortion first.
2. **Test CNN separability on FADE_CALM** (biggest, 24k trades): if CNN
   clears 58%+ OOS on SAME/COUNTER, full pipeline viable. If plateaus
   at 52-54%, tiers are truly dead.
3. **If (2) fails**: rebuild tiers from data (corrected-trade clustering)
   instead of hand-crafted physics gates.

## Tools

- `tools/run_tier_isolated.py` — isolate each tier, no chains/catch-all
- `tools/killshot_peak_physics.py` — path reconstruction + peak physics
- `tools/regret_on_isolated.py` — regret per tier, verdict table

## Reports

- `reports/findings/2026-04-17_killshot_peak_physics.md`
- `reports/findings/2026-04-17_iso_regret.md`


## Source: project_imr_trade_exit.md

---
name: I-MR control chart for trade exit detection
description: Replace Brownian/fixed giveback with I-MR on per-trade price path — detects regime shifts without shape assumptions
type: project
---

Replace all giveback threshold logic with I-MR control chart on per-trade price path:
- Replay each trade's 15s bars from entry to exit
- Compute I-MR on the price series DURING the trade
- "In control" = hold (normal variation for THIS trade)
- "Out of control" = exit (regime shift detected)

**Why:** Current giveback assumes V-motion reversal. Real trades staircase, drift,
spike-fade, or cascade. Fixed/Brownian thresholds fire on consolidation (false alarm)
or miss slow drifts (late exit). I-MR adapts to the trade's actual behavior.

**Why I-MR specifically:**
- No shape assumption (works for V, staircase, drift, cascade)
- Control limits adapt to the trade's own volatility (MR = moving range)
- Detects the MOMENT behavior changes, not after a fixed % giveback
- Already validated in the system (I-MR regime segments tool)
- Six Sigma foundation — this is process monitoring applied to trades

**How to apply:**
1. Build `tools/imr_trade_replay.py` — replay IS trades with per-bar I-MR
2. For each trade: compute I-MR on close prices, detect first out-of-control bar
3. Compare: I-MR exit bar vs actual exit bar vs optimal exit bar (oracle MFE bar)
4. If I-MR catches reversals earlier → wire into exit engine as primary giveback
5. The MR (moving range) naturally handles staircase (high MR → wide limits)
   vs spike (low MR → tight limits)

**Prerequisite:** Need 15s bar data accessible during forward pass for per-trade
I-MR computation. Currently only have entry/exit/peak in trade log.


## Source: project_intraday_primitive_registry.md

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


## Source: project_l5_sim_deploy_2026_05_19.md

---
name: l5-sim-deploy-2026-05-19
description: "L5 stack built + wired into engine_v2 for Phase-1 SIM deploy. Phase-1 1c OOS lift +$42/day NOT SIGNIFICANT (CI [-$39,+$118]). Deploy to SIM only, not real money."
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

L5 stack (V2 features + B7/B9/B10) is wired into `live/engine_v2.py`
behind opt-in `--engine-mode l5` flag. Default `blended` mode unchanged.

**Why:** User went to sleep 2026-05-18 directing autonomous build for SIM
deploy 2026-05-19. Constraint: 1 contract/position max ($100 equity
buffer per additional contract). Phase-1 collapses B7/B9/B10 sizing
surfaces to filters: B7 = binary skip; B9 = binary CUT at K=5; B10 =
risk regime modulating B7+B9 thresholds (NOT skip-day per user's
"tighter limits not full skip" guidance).

**How to apply:**
  - Production money deployment GATED. OOS Phase-1 delta +$42/day NOT
    statistically significant vs FLAT 1c (CI crosses zero). Do NOT
    promote to real money until SIM data refines thresholds.
  - SIM deployment IS appropriate -- zero financial risk, gives
    real-time data on V2 streaming + zigzag pivot timing + B7/B9
    inference latency.
  - Launch: `python -m live.engine_v2 --engine-mode l5 --mock` first
    (mock replay validates), then `--engine-mode l5` for real SIM.
  - Pre-flight: MUST run `tools/sourcing/build_cross_day_features.py`
    daily before launch (B10 day-mode needs today's row); then
    `python tools/preflight_check.py` (7 checks).
  - If L5 misbehaves on SIM: drop the `--engine-mode l5` flag to fall
    back to existing BlendedEngine. Zero rollback cost.

**Key files** (paths -- check via `git status` to find rename/move):
  - `training/live_feature_engine_v2.py` -- streaming 185D V2 vector
    on-demand via `get_v2_vector(ts)`. Subclass of LiveFeatureEngine.
    300/300 parity vs batch on 2026_05_06 at 1e-6 tol.
  - `live/l5_decider.py` -- evaluate(state) -> DecisionBatch with
    zigzag state machine + B7/B9/B10 inference.
  - `tools/forward_pass_1contract.py` -- IS-calibrated Phase-1 OOS
    forward pass. Locked thresholds: B7 skip>=1.90, B9 cut<+5
    (normal); +0.2/+10 tighter on cautious mode.
  - `tools/preflight_check.py` -- pre-session SIM checklist.
  - `docs/Active/LIVE_L5_ARCHITECTURE.md` -- thin-wrapper architecture.

**OOS gating numbers** (51 days, IS-calibrated thresholds locked,
2026-05-19 sleep-run):
  FLAT 1c:          $+454/day CI [+$261, +$664]
  Phase-1 stack:    $+496/day CI [+$311, +$693]
  Delta vs FLAT:    $+42/day  CI [-$39, +$118]  NOT SIG

**UPDATE 2026-05-19 LATE-MORNING -- Mock validated against forward pass:**
  - 5-day mock May 11-15: $+3,582 vs OOS Phase-1 forecast $+3,716
    (gap $-27/day = 3.6%). Engine is decision-identical.
  - 20/20 EXACT entry+exit price match on May 11 spot-check.
  - Streaming pivot detector now ports the FULL detect_swings logic
    (min_bars=36, ATR median of 42 TRs, 5-day pre-warmup); per-day
    pivot capture 98% week-aggregate.
  - --pivot-source=replay mode added for regression testing.

**Related:** [[live-l5-architecture-thin-wrapper]],
[[user-collaboration-protocol]], [[oos-only-for-nn-validation]]


## Source: project_live_wiring.md

---
name: Live wiring — PhysicsEngine DEPLOYED in sim
description: PhysicsEngine wired into live and running on NT8 sim as of 2026-03-22. AdvanceEngine rename complete.
type: project
---

**COMPLETED 2026-03-22:**
1. Rename BarProcessor → AdvanceEngine (14 files, commit 00228fb5)
2. Deleted dead weight: history_replay.py, replay_bridge.py, atlas_loader.py
3. Wired PhysicsEngine into live_engine.py (commit 9bf32cb8)
4. Fixed: belief_network None guards, 1s bar routing, unrealized PnL, session report append
5. Added verbose decision logging, skip diamonds, deferred FLIP confirmation

**How to run:** `python -m live.launcher --physics`
- Auto-finds latest seed JSON in DATA/regime_seeds/
- Forces anchor_tf='1m'
- Skips TBN/brain/pattern library — just SFE + seeds
- SL: 40 ticks (10 points MNQ) at 1s resolution
- FLIP: deferred re-entry waits for close FILL (prevents 274-contract bug)

**Known gap: NOT PRICE AWARE**
- See memory/project_physics_price_aware.md
- 12 features are all physics, none encode position in higher TF structure
- Adding 1h z-score + 1h fm_sign would be biggest improvement

**How to apply:** Collect sim data, then enrich seeds with higher-TF features.


## Source: project_llm_news_intensity_2026_05_21.md

---
name: LLM News-Intensity Feature for DRS (Phase A scaffold)
description: Dev scaffold built 2026-05-21 to score FOMC/CPI/NFP press releases via local Llama-3.1-8B and inject the score into DRS as a feature; nothing run yet; canonical pipeline byte-identical until validated.
type: project
originSessionId: 2111be4b-5103-4d2d-a1e0-249c678bdcde
---
**Fact.** 2026-05-21 late session: built a fully isolated dev scaffold for an
LLM-scored news-intensity feature feeding DRS. **Nothing has been executed.**
Production paths are byte-identical to before this session.

**Why.** 2026-05-18 DRS canonical verdict: OOS sealed Pearson +0.139 CI
[-0.047, +0.451] — lower bound crosses zero, naive sizing loses -$333/day
CI [-$523, -$163]. The binary event flags (`is_fomc/cpi/nfp/opex`) each
contribute $0 of permutation ΔMAE — they tell DRS an event happened but
not whether it was hawkish/dovish/on-consensus/shock. The continuous LLM
score is meant to subsume them and push OOS lower CI strictly positive.
The verdict itself flagged "LLM-scored news headline intensity" as the
next research direction.

**How to apply.**

1. **Do not touch canonical paths during dev.** The user's hard requirement.
   `tools/sourcing/build_cross_day_features.py`, `drs_canonical_gbm.py`,
   `DATA/CROSS_DAY/cross_day_features{,_with_target}.parquet`,
   `drs_canonical_gbm.pkl`, `forward_pass_full_stack.py`, B10 day-multiplier
   — all unchanged. Everything new lives in:
   - `tools/sourcing/llm_news/` (self-contained module)
   - `tools/sourcing/build_cross_day_features_v2.py` (augmenter, NOT a copy)
   - `tools/sourcing/drs_canonical_gbm_v2.py` (mirror of canonical, dev I/O)
   - `DATA/CROSS_DAY/dev/` (all outputs)
   - `research/llm_news_intensity/` (DMAIC + PDCA + findings)

2. **Execution order (user runs these).**
   - `huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
     Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models`
     → rename to `models/llama-3.1-8b-instruct-q4_k_m.gguf`
   - `pip install llama-cpp-python[cuda]` (or pre-built wheel if build fails)
   - `python -m tools.sourcing.llm_news.cli fetch`
   - `python -m tools.sourcing.llm_news.cli test-synthetic`  ← MUST PASS
   - `python -m tools.sourcing.llm_news.cli score`
   - `python tools/sourcing/build_cross_day_features_v2.py`
   - `python tools/sourcing/drs_canonical_gbm_v2.py`
   - Read `research/llm_news_intensity/findings/{date}_phase_a_results.md`

3. **Phase A gate (single-shot, all 3 required).**
   - IS WF Pearson lower CI ≥ +0.098 (baseline lower bound — must not regress)
   - OOS sealed Pearson lower CI > 0 (was -0.047)
   - `news_intensity_today` ΔMAE ≥ +$30/day (beats `vix_close_prior` at +$17/day)

4. **Decision tree on gate fail.**
   - IS lifts, OOS doesn't → LLM memorized 2024-2025 statements. Kill.
   - IS doesn't lift → prompt eng issue or 8B too small. Iterate prompt
     OR upgrade to Qwen2.5-14B Q4 (~9GB VRAM) or Mistral-Small-24B Q4 (~14GB).
   - ΔMAE under threshold → signal exists but too weak; kill.
   - Score std < 1.5 → don't even retrain DRS; prompt eng failed.

5. **Phase B (cycle_02.md) is only created if Phase A passes.** Adds
   `news_intensity_prior` (yesterday's PM releases, 14:00 ET FOMC etc).
   Bootstrap CI on delta_Pearson(A+B vs A only) must exclude 0.

6. **Promotion to main is a SEPARATE user-approved session.** Includes a
   B10 regression check (re-run forward_pass_full_stack.py IS+OOS; OOS
   $/day delta must be within ±$5/day noise floor). Plan section
   "Promotion to main" has the 10-step procedure.

**Critical files for next session to read first**:
- `~/.claude/plans/i-would-like-to-jaunty-spindle.md` (approved plan)
- `research/llm_news_intensity/project.md` (DMAIC frame)
- `research/llm_news_intensity/cycle_01.md` (PDCA — predictions written
  pre-run, do not edit retroactively per MEMORY PDCA rule)

**Backout cost**: delete `tools/sourcing/llm_news/`, both `_v2.py`,
`DATA/CROSS_DAY/dev/`, `research/llm_news_intensity/`, the `models/` entry
in `.gitignore` (line just added), `tools/sourcing/__init__.py`. Zero
production regression possible.

**Pushback rationale (worth remembering for future LLM-in-system asks)**:
- In-the-loop LLM trade veto: REJECTED. Latency, non-determinism, kills
  CI/bootstrap discipline.
- Babysit / monitoring LLM: marginal value over halt-after-N + DRS + B10 +
  blowout sim + parity checks already in place.
- News intensity scorer: the ONE LLM use case with a clean $/day proof
  path (goes in DRS as a feature, bootstrappable, OOS-validatable).
- LLM-as-feature is the right paradigm; LLM-as-decider is the wrong one.


## Source: project_loose_gate_avenue.md

---
name: OOS compressed gate loosening avenue
description: Removing cascade/struct from OOS gate → 3x trades, $12K OOS (excl Feb 9 outlier) but lower $/trade
type: project
---

Removing `cascade_detected OR structure_confirmed` from OOS compressed signal gate:
- 44% of bars have pattern_type but no cascade/struct — previously blocked
- Loosening: 5,087 trades, $40K OOS ($12K excl Feb 9 outlier), $2.52/trade
- Tight (current): 1,691 trades, $6.7K OOS, $3.99/trade

**Why deferred:** Lower $/trade ($2.52 vs $6.16 previous best) means commissions
and slippage eat more of the edge in live. Need to filter low-quality pattern
trades before loosening.

**How to apply when ready:**
1. Loosen the gate (pattern_type alone)
2. Add quality filter: coherence > 0.60, or ADX > 20, or 5m DMI agrees
3. This should keep the extra trades but filter noise entries
4. Re-validate OOS excluding outlier days


## Source: project_market_hierarchy.md

---
name: Market Hierarchy — The Three Bodies as Market Participants
description: The three-body model maps to a social hierarchy of market participants. Gods (institutions) set gravity, demi-gods (algos) enforce bands via PID, avatars (prop) amplify, mortals (retail) navigate. The system is the leaf reading the wind.
type: project
---

## The Three Bodies = Market Participant Hierarchy

| Body | Who | Timeframe | Force | What They Do |
|------|-----|-----------|-------|-------------|
| **Gods** | Institutions, sovereign funds, central banks | 1h-1D+ | Gravity F = -θ·z·σ | Set fair value (μ). Move mysteriously. Their agenda IS the regression mean. |
| **Demi-Gods** | HFT, market makers, execution algos | 1m-15m | PID: Kp·e + Ki·∫e + Kd·de/dt | Execute divine agenda. Kp = band response, Ki = spring, Kd = wick dampening. |
| **Avatars** | Prop firms, systematic traders | 5m-1h | Momentum F_momentum | Align with algos. Follow flow. Amplify moves. |
| **Mortals** | Retail. Us. | 15s-5m | Noise σ·dW | At the whim of the system. The leaf in the wind. |

## The Nightmare Protocol = Be the Leaf That Knows the Wind

We can't control gods, out-compute demi-gods, or out-capitalize avatars.
But we can **read the gravitational field they create** and navigate it.

The leaf doesn't fight the wind — it reads the pressure differentials and
finds the path of least resistance. That's what the z-score, forces, and
wave function probabilities compute: the pressure map of the market.

## Resonance Cascade = When All Participants Align

The cascade happens when gods, demi-gods, and avatars move in the same
direction simultaneously. The PID controllers (demi-gods) stop defending
the bands and JOIN the flow. Roche limits shatter because there's no
resistance to enforce them.

**Why:** disable TP during cascade — the demi-gods aren't providing
resistance anymore, they're accelerating the move. Fading it = stepping
in front of a freight train where ALL participants are the train.

## How to Apply: The system should ask THREE questions every bar

1. **What are the gods doing?** → Macro regression slope + Hurst.
   If gods are flat (H≈0.5), play mean reversion (Nightmare Protocol).
   If gods are trending (H>0.55), look for cascade alignment.

2. **Are the demi-gods enforcing or joining?** → PID response.
   If PID is active (wicks at bands, mean-reverting), respect Roche limits.
   If PID is absent (clean breakouts, no wicks), cascade may be forming.

3. **Where are the avatars flowing?** → Momentum + volume.
   If aligned with gods + demi-gods absent = resonance cascade.
   If fighting gods = chaos / untradeable / stay flat.

**Why:** This maps to checking coherence across the hierarchy, not
just across timeframes. The coherence score should weight by
PARTICIPANT TYPE, not just by TF duration.


## Source: project_midleg_entry_2026_05_20.md

---
name: midleg-entry-research
description: Mid-leg / missed-signal late-join research — REJECTED. Zigzag legs are sequential, engine runs at 99.9% utilisation, no parallel-signal population exists.
metadata:
  type: project
---

Mid-leg entry (late-joining an R-trigger the live engine missed) was
researched 2026-05-20 (autonomous sleep-run, 51-day sealed OOS) and
**REJECTED — do not build.** Full report:
`reports/findings/regret_oracle/2026-05-20_midleg_entry_research.md`.

**The structural fact (the durable takeaway):** hardened zigzag legs are a
*sequential partition* of the price path — leg N exits at a pivot, leg N+1
is born from that pivot. 99.8% of consecutive-leg gaps are exactly 0. A
1-contract greedy engine runs at **99.9% utilisation** and catches **2,922
of 2,926** OOS legs; it misses only 4 in 51 days, and all 4 are losers.
There is **no population of legs missed because the engine was busy.**

**Why:** the zigzag produces one leg at a time; legs essentially never run
in parallel. This is structural — it will NOT change with more data or with
contract scaling. Mid-leg / missed-signal / parallel-position / "catch the
signal we lost" ideas are all moot for the same reason.

**How to apply:** if a future session proposes catching missed signals,
late-joining, or running parallel positions on this zigzag pipeline — point
at this finding first. The premise (overlapping signals) is false. The one
legitimate "add to a running leg" action is pyramiding the leg you are
ALREADY in, which B9's continuous sizing already covers.

**Experiment results:**
- E1 Fork 1 (B9-gated, *unconstrained*): +$303/day @ K=5 [CI +160,+457] —
  +EV but over a population that does not exist live. Note this is the value
  of B9-*gating* late entries, not new money (baseline already enters at K=0).
- E2 Fork 2 (B1-B6 pivot-structure augmentation): −$76/day @ K=5 [CI −158,−2]
  significantly negative; neutral-to-harmful at all K. B1-B6 ARE
  in-distribution mid-leg (trained on all 1m bars), but the B9 GBM already
  extracts pivot-structure from the raw V2 features — stacked B1-B6
  predictions add only noise (same as the rejected lead-in PCA, see
  [[feedback_leadin_pca_rejected]]).
- E3 (position-constrained 1c sim): incremental −$1/day, 4 late-joins / 51d.
- E4 (overlap): 99.9% utilisation, 99.8% gaps = 0.

**What the "lost signals" the user sees actually are:** cold-start — ~7
first-of-session legs/day lost to the ~20-min (240-bar) feature-engine +
zigzag warmup (the engine cannot compute during warmup), or deliberate B7
skips. The fix is **same-day catch-up**, NOT late-join: at startup replay
TODAY's elapsed bars through `detect_swings` + pre-warm the V2 feature engine.
NOT prior-day priming — the forward pass runs the detector PER-DAY anchored at
the day's first bar, so prior-day state would diverge from it. The cold-start
divergence was measured BOUNDED 2026-05-21 (`tools/zigzag_coldstart_divergence.py`:
0/216 never-resync, resync median 27 min) — not significant, low priority.


## Source: project_mtf_counter_proposal.md

---
name: MTF Two-Layer Counter-Proposal (SUPERSEDED)
description: 29D MTF architecture superseded by nn_v2 79D pipeline with 3-CNN system
type: project
---

## SUPERSEDED (2026-04-03)

The MTF Two-Layer Counter-Proposal (29D features, Direction + Duration CNNs) was
superseded by the nn_v2 pipeline which uses 79D features and a 3-CNN system:
- CNN Flip (direction at entry, 70.6%)
- CNN Hold (hold/exit during trade, 94.8%)
- CNN Risk (cut losers early)

Result: $613/day OOS vs the counter-proposal's untested spec.

**Why:** nn_v2 was built from clean data discovery, not theoretical architecture.
**How to apply:** Spec at `docs/Active/COUNTER_PROPOSAL_MTF_TWO_LAYER.md` is historical only.


## Source: project_new_system_design.md

---
name: New System Design (79D + NN + Half-Life)
description: Architecture spec for replacing advance engine — 79D features, strategy router NN, half-life exit
type: project
---

## Status: DESIGNED, NOT YET IMPLEMENTED (2026-04-03)

## Architecture
- **79D feature vector**: 10 features x 6 TFs + 3 helpers x 6 TFs + time_of_day
- **Strategy Router NN**: 79D → direction + hold duration (half-life)
- **Unified Exit**: envelope decay with NN-predicted half-life, modulated by survival score + giveback
- **Execution**: 5s atomic bar, 1m decision anchor

## Key Specs
- `docs/Active/FEATURE_VECTOR_79D_SPEC.md` — full feature vector + NN architecture
- `docs/Active/EXIT_MATH_ANALYSIS.md` — exit module math, unified exit concept, markers for resumption

## Core Insight
5s is the Planck constant. TFs are aggregation windows. The NN learns signal half-life — how many 5s bars of noise to hold through before the edge decays.

## Three Exits → One Function
1. Envelope Decay: `exp(-ln2 * t / hl)` — the base decay
2. Survival Stop: `room * trend * conviction * alignment * momentum` — modulates HL
3. Peak Giveback: volume-adaptive threshold — accelerates HL

Unified: NN predicts initial HL, survival score compresses it live, giveback is emergency accelerator.

## Why Advance Engine Was Abandoned
- 4,721 trades, ALL template -100 (peak reversal), 50.1% WR = coin flip
- Zero template matches in forward pass
- Templates cluster on 16D features the ticker never uses
- Nightmare ticker (simple rules, z_se + vr) found real edges
- Too many moving parts: peak detect, 1m sensor, cat brain, 7 exit types

## Next Steps
1. Implement `extract_79d()` from SFE states
2. Build training label generator (forward PnL at 6 durations per bar)
3. Train strategy router NN
4. Wire unified exit (envelope + survival + giveback)
5. Test on OOS Feb 2026 clean data


## Source: project_nn_v2_system.md

---
name: nn_v2 3-CNN Trading System
description: Current active system — 79D features, NMP+blended engine, 3 CNNs, $613/day OOS on clean data
type: project
---

## nn_v2 3-CNN System (2026-04-07)

### Results
- IS: $620/day, 96% win days
- OOS: $613/day, 91% win days, $22/trade
- BASE_NMP improved from $0.30/trade to $7.90/trade with CNN flip (26x)

### Architecture
```
ticker (1s) → aggregator (all TFs) → SFE → 79D features → NMP → blended engine → 3 CNNs
```

**Blended Engine** (`nn_v2/nightmare_blended.py`):
- Cascade tier: aggressive z-based
- Kill shot tier: |z|>2 + vr<1 + wick rejection (96% WR, $42/day)  
- Base NMP tier: standard z_se>2, vr<1

**Three CNNs**:
1. **CNN Flip** (`cnn_flip.py`): 70.6% direction accuracy from 6×13 TF grid at entry
2. **CNN Hold** (`cnn_hold.py`): 94.8% accuracy, 98.9% HOLD, 69.6% EXIT
3. **CNN Risk** (`cnn_risk.py`): cuts losers (0% WR on cuts = correct)

### Key Learnings
- Trees exhausted at 55% direction — CNN sees cross-TF patterns trees can't
- Regret is the teacher, CNN is the student (skip trees/rules/books)
- Entry-only CNN > path CNN (path data adds noise)
- CNN distillation: primary split = 15m_wick_ratio (same as kill shot)
- Zero crossing pattern: odd = winner, even = loser

### Next Step: Stage 2
1. Run regret on CNN-flipped trades
2. Extract 79D at optimal entry points
3. Train Stage 2 CNN on new entry physics
4. Cluster → validate → expand ExNMP roster

**Why:** This is the first system with positive OOS on clean data ($613/day).
**How to apply:** All new work builds ON TOP of nn_v2. No changes to legacy core/.


## Source: project_nq_goal.md

---
name: Goal — MNQ to NQ in 3 months (by 2026-06-23)
description: User's target is to graduate from MNQ ($0.50/tick) to NQ ($5/tick) once system is proven profitable. Same system, 10x capital requirement.
type: project
---

**Goal:** Trade NQ instead of MNQ by 2026-06-23 (3 months from 2026-03-23).

**Why:** MNQ and NQ are the same instrument — same price, same order book, same features.
The system doesn't change. Only the tick value changes ($0.50 → $5.00, 10x).
The user needs capital to sustain ~$400 of noise (two back-to-back 40-tick SL hits on NQ).

**Prerequisites:**
1. Proven profitability on MNQ (current step — PhysicsEngine in sim)
2. Grounded 13-feature set validated on OOS
3. Price-aware exits (Fibonacci/volume profile)
4. Consistent positive daily PnL for 30+ trading days
5. Account equity sufficient to handle NQ drawdowns ($5K+ buffer)

**How to apply:** Don't rush to NQ. The system must be profitable on MNQ first.
Every improvement (features, exits, regime detection) applies to both.
The graduation is a capital decision, not a system decision.


## Source: project_oos_bad_days_2026_05_21.md

---
name: oos-bad-days-2026-05-21
description: The OOS bad days resist every day/session-level fix — 4 levers ruled out. The deployed B-stack (+$175/day on bad days) is the validated mitigation.
metadata:
  type: project
---

The "lift the OOS bad days" research (2026-05-21 autonomous run; DMAIC project
`research/oos_bad_days/project.md`) converged on a strong negative.

**The fact.** The OOS bad days (14/51 days negative, −$3,570, 8 worst = 85% of
the loss) are NOT preventable by any day-level, session-level, or prior-day
signal. Bad days = the RTH cash session (ET 09–15, ≈100% of the $/day) failing
to trend — chopping. Four levers ruled out, rigorously (IS-discover /
OOS-confirm):
- hour-of-day skipping — not significant on $/day;
- intraday cumulative-P&L session-stop — fails like the per-trade stop (81%
  OOS recovery; −$79/day CI [−154,−22] SIGNIFICANT loss; negative days 14→20);
- bad-day clustering — daily P&L is iid day-to-day (lag-1 IS +0.06 p=0.24 /
  OOS −0.01 p=0.96); prior-day P&L is not a signal;
- (prior work) DRS session-start macro prediction failed OOS; per-trade
  drawdown stop rejected (76% recover).

**What works:** the deployed B-stack (B7/B9/B10) shaves bad days **+$175/day,
CI [+$98, +$269], significant** (the day-level MEASURE); good-day delta −$9
(not sig). The bad days are already the B-stack's main job. The
non-significant headline +$42/day hid this asymmetric protection.

**Why:** chop is not predictable ahead of time at the day level. One real
regularity — the chop regime (leg amplitude) IS autocorrelated day-to-day (IS
+0.275 / OOS +0.485) — does not propagate into bad-day clustering, and B10
already exploits the vol regime.

**How to apply.** Do NOT re-propose a day-level bad-day predictor, an intraday
P&L stop, a per-trade drawdown stop, or a prior-day-outcome filter — all
rejected with OOS evidence. The residual bad days are the irreducible cost of a
trend zigzag in choppy markets. The only avenues with a real prior: (1) a
CAUSAL ATR study — a wider zigzag whipsaws less in chop, needs a causal
streaming pass (the FLAT sweep is contaminated, see
[[flat-pipeline-cross-param]]); (2) strengthening B9 — per-leg continuous
amplitude sizing, the only action template that has beaten the R-trigger.
RTH-only is a defensible pure risk control (~15% less daily std, ~3 fewer bad
days, ≈$/day-neutral) but not a fix. Tools: `oos_bad_day_characterize.py`,
`oos_hourly_characterize.py`, `oos_intraday_stop_analysis.py`,
`oos_bad_day_autocorr.py`.


## Source: project_original_bayesian_brain_architecture.md

---
name: Original Bayesian Brain architecture (Feb 2026, deleted/superseded — substrate was wrong, design was right)
description: 7-step pipeline scanning IS to build trade templates, K-means distill to canonical set, mount on Bayesian brain, track via oracle + regret, fire via 9-gate cascade. The architecture worked; templates were "obscured / multi-dimensional concepts" which was the red flag.
type: project
---

**Documented 2026-05-09 evening from user description.**
**Source files were deleted in commit 23db222f ("chore: delete dead engine code")**
but the architecture is the basis for the V0 build. Reviving the design with
corrected substrate (primitive chord instead of K-means'd multi-D template).

## The original 8-step pipeline

```
1. SCAN IS              walk in-sample data bar by bar
2. BUILD TEMPLATES      extract trade-pattern signatures around each entry
3. K-MEANS DISTILL      cluster template signatures into a smaller canonical set
4. SCRUB                clean noisy / low-confidence clusters
5. MOUNT ON BRAIN       load distilled templates as Bayesian-table keys;
                        each (template, direction) cell holds win/loss + $/trade
6. ORACLE + REGRET      post-hoc: did the trade fire the right way?
                        regret = counterfactual analysis on each decision
                                 (what would have happened with other actions)
7. 9-GATE CASCADE       at each candidate entry, 9 ordered gates each gates
                        the trade; only signals that pass all 9 gates fire
8. OOS VALIDATION       carry the IS-fit brain forward on out-of-sample data,
                        re-run the 9-gate cascade against the brain's posteriors,
                        measure how well the IS-derived edge generalized
                        (this is the "did the substrate hold" check that any
                        IS-fit table requires before deployment)
```

## Why it failed (the red flag, in hindsight)

The "obscured / multi-dimensional concepts" were the templates themselves.
Step 3 (K-means on multi-dimensional template signatures) collapsed distinct
real patterns into clusters whose CENTROIDS no longer represented the
patterns clearly.

Concretely:
- Templates were built from many features at once (a 16D, 23D, or 79D
  feature vector at the entry bar)
- K-means partitioned those vectors into K buckets
- The centroid of a K-means bucket is not a recognizable pattern — it's
  an averaged blob of all the patterns that happened to land near each
  other in 16D / 23D / 79D space
- Two physically different setups could land in the same bucket (false
  merge) or one setup could split across buckets (false split)
- Bayesian table keyed on bucket id therefore averaged unrelated trades
- Outcome estimates were noisy; gate-cascade signals fired on patterns
  that didn't actually exist as coherent regimes

This is the substrate problem. The architecture was right; the keys were
abstract enough that they no longer corresponded to discoverable patterns.

## Why the V0 build (chord-keying) corrects this

The 5 primitive chord axes from `tools/event_bucket_15m_crm.py` are NOT
abstract:

  slope_q       (5 bins)   sign + magnitude of 1h-lookback slope of M_close_15m
  curvature_q   (3 bins)   sign + magnitude of slope-of-slope
  z_close_q     (5 bins)   (5s_close − M_close_15m) / SE_close_15m
  sigma_rank_q  (5 bins)   rolling 60min percentile of SE_close_15m
  r2adj_q       (5 bins)   R² adjusted of 5min linear fit to 5s closes

Each axis is a single statistical measurement with a clear physical
interpretation. The chord (slope_q3, curv_q1, z_q4, sigma_q2, r2adj_q5)
describes a SPECIFIC, recoverable pattern: 'no trend, decelerating curve,
price slightly above mean, low band-width, smooth/predictable'.

There is no centroid averaging. Two events with the same chord WERE
empirically in the same statistical phase. Bayesian table keyed on chord
therefore aggregates trades that genuinely share context.

Up to 5×3×5×5×5 = 1,875 cells from 5,340 IS macro events ≈ mean 3
events/cell — needs hierarchical shrinkage (cell → axis-marginal →
universal), which is already implemented in
`training_iso_v2/bayesian_table.py`.

## What survived from the original architecture

Almost everything except the K-means step:

| step in original         | mapping in V0                                                                  |
|--------------------------|--------------------------------------------------------------------------------|
| 1. scan IS               | same — walk 5,340 IS macro events                                              |
| 2. build templates       | replace with: extract 5-axis chord at event entry                              |
| 3. K-means distill       | DROP — quantile bucketing already produces canonical cells                     |
| 4. scrub                 | replace with: hierarchical shrinkage of thin cells toward parent               |
| 5. mount on brain        | extend `training_iso_v2/bayesian_table.py` keying                              |
| 6. oracle + regret       | reuse — `training_iso_v2/regret.py` already produces per-event labels          |
| 7. 9-gate cascade        | maps to: meta-router + per-tier filter cascade (the 2026-05-09 architectural lock) |
| 8. OOS validation        | OOS sign-stability per axis + per-cell P(_) divergence test; re-run meta-router and tier-filter cascade on the OOS macro-event population (already in todo list as the validation pass) |

The 9-gate cascade became the 9 ExNMP tiers (FADE_CALM, FADE_MOMENTUM,
RIDE_CALM, RIDE_MOMENTUM, FADE_AGAINST, RIDE_AGAINST, KILL_SHOT, CASCADE,
FREIGHT_TRAIN — see `training_iso_v2/strategies/`). Each tier IS a
parameterized gate. The new meta-router (Level 1, P_cascade-based)
selects which tiers are even eligible, then per-tier filters (Level 2)
gate firing.

## What this means for the user's pattern recognition

The user's instinct ('original template work will work but not in the
context we were doing') was correct. The exploratory work today produced:

  - Statistical primitive axes that don't average across distinct patterns
  - Event-segmented bucketing on 1h HL ±3σ macro events (3,431 IS)
  - Oracle-driven failure-mode framing
  - Meta-router architecture that maps to the original 9-gate role
  - Hierarchical shrinkage that replaces the K-means distill step

All of which corrects the SUBSTRATE problem that killed the original brain
without changing the architectural design.

## Files / commits to reference

- Feb 2026 brain (deleted): commit befdc2df modified `core/bayesian_brain.py`
  which tracked `(StateVector, template_id, direction) -> {wins, losses, $/trade}`
- Feb 2026 templates: commits 8232a1c6 (TF-bucketed clustering), 82fc8478
  (counter-trend template analysis), 4f9334c6 (trade duration fractal analysis)
- Deletion: commit 23db222f ('chore: delete dead engine code -- core/ down to 5 files')
- Modern Bayesian table: `training_iso_v2/bayesian_table.py` (regime keying — to be replaced by chord)
- Modern regret: `training_iso_v2/regret.py` (oracle + counterfactual labels)


## Source: project_parity_b9_horizon_2026_05_20.md

---
name: parity-b9-horizon-2026-05-20
description: First L5 SIM run was an engine bug (B9 fired 12x late), not a strategy failure. Lesson - parity-check live runs vs the forward pass before interpreting them.
metadata:
  type: project
---

The first live L5 SIM run (2026-05-20) came in at −$379. A parity check
against the offline forward pass found this was an **engine bug, not the
strategy**: the validated strategy on that exact day was ≈ −$59 (a flat day).

**The bug (now fixed):** `l5_decider` fired the B9 during-trade cut at
`pos.bars_held == 5`. `core/ledger.py` computes `bars_held = (ts−entry_ts)//60`
— elapsed **minutes** — so B9 fired 5 MINUTES after entry. But the B9 model
`b9_remaining_amplitude_K5.pkl` was trained for K=5 in **5-second-bar units**
(`build_trade_trajectory_dataset.py`: `bar_ts = entry_ts + K*5` → 25 s). B9
fired 12× too late, every trade. Fixed: l5_decider now fires B9 off a
5-second-bar counter (`_bar_count − entry_bar_count >= B9_K`).

**Durable trap — units.** `bars_held` (core/ledger) is in MINUTES; the
B9/trajectory K horizons are in 5-SECOND-bar units. Anything timing a
during-trade action against a K horizon must use 5s-bars, NOT `pos.bars_held`.
core/ledger's `//60` was deliberately left as-is — the blended engine
(sim/training) still depends on `bars_held`=minutes.

**Durable lesson — methodology.** Before interpreting a live SIM run's P&L,
**parity-check it**: run the same period through the offline forward pass
(`tools/mock_week_runner.py` mock; `tools/parity_live_vs_forward.py` for price
parity) and diff the decisions. A live run can diverge wildly from the
validated strategy via engine bugs — the headline P&L is meaningless until
parity holds. The user's instinct ("if there's no parity it's back to the
drawing board") was right and caught the bug.

**Mock tool bug (also fixed):** in mock mode, fills stamped `fill_time` with
wall-clock instead of the replayed bar-time, so `entry_ts` was wall-clock and
`bars_held` went negative — B9 never fired in the mock at all. Fixed in
`mock_bridge.py` (`_last_bar_sent_ts`).

**State (2026-05-20):** `engine_v2.py` refactored to **zigzag-only** — the
blended-engine path is removed from the LIVE engine (Phase 1 of
[[zigzag-only-refactor]] / `docs/JULES_ZIGZAG_ONLY_REFACTOR.md`; verified
byte-identical mock output). BlendedEngine still exists for sim/training.
Phase 2 reframed (2026-05-21): prior-day zigzag priming is **REJECTED** — the
forward pass (`build_zigzag_pivot_dataset.py`) runs the detector **per-day**,
anchored at each day's first bar, NOT on prior-day state. The cold-start
divergence study (`tools/zigzag_coldstart_divergence.py`, 54 days / 216
samples) measured the real gap: a mid-session cold-start vs the day's-first-bar
run is a **BOUNDED TRANSIENT** — 0/216 never-resync, resync median 27 min,
median 0 mis-tiled legs, ~$103 mean mis-tiled amplitude (a $ proxy). Not
significant. Correct (optional, low-priority) fix = **same-day catch-up**:
replay today's elapsed bars through the zigzag at session start.


## Source: project_physics_engine.md

---
name: Physics Engine — alt live engine ($132/day baseline)
description: New live engine based on K-NN trajectory matching against enriched seeds. Separate from BarProcessor. $132/day OOS proven.
type: project
---

Physics Engine is the alternative live engine replacing BarProcessor for trading.

**Why:** BarProcessor captures 0.9% of available PnL ($1,844 OOS). Physics funnel captures $5,138 ($132/day) on same data. 72x improvement.

**Config (proven OOS, no lookahead):**
- 12-feature trajectory: fm, z, dmi_p, dmi_m, adx, vel, vol, hurst, P_center, coherence, sigma, pid
- 10-bar lookback window (1m bars)
- K=20 nearest seeds from 38K enriched IS library
- Consensus > 0.65 for direction
- Coherence < 0.6 (TF disagreement = real reversal)
- Magnitude > p25 (rolling window, seeded from IS)
- Hold from matched seed median duration (3-20 bars)

**How to apply:**
- Build as `core/physics_engine.py` (NOT modifying bar_processor.py)
- Load enriched seeds at startup from `DATA/regime_seeds/auto_seeds_all_*.json`
- Pre-compute seed trajectory matrix + normalization from IS data
- Per bar: update trajectory buffer → match → filter → enter/exit
- Wire into `live/live_engine.py` as alternative to BarProcessor

**Architecture:**
- Naming: "engine" not "processor"
- Separate from existing system (competing, not replacing)
- Uses StatisticalFieldEngine for state computation (same CUDA pipeline)
- Does NOT use: TBN, brain, cat, gates, exit cascade
- DOES use: enriched seed library, K-NN matching, coherence filter

**Enriched seeds location:** `DATA/regime_seeds/auto_seeds_all_20260322_154729.json` (426MB)
**Backup:** `checkpoints/backup_seeds_run_20260321/enriched_seeds.json`


## Source: project_physics_price_aware.md

---
name: PhysicsEngine price-awareness gap
description: PhysicsEngine K-NN has no price structure awareness — same physics at top vs bottom of range produces same match. Biggest edge improvement opportunity.
type: project
---

PhysicsEngine's 12 features are all statistical (fm, z, dmi, adx, vel, vol, hurst, P_center, coherence, sigma, pid). None encode WHERE in the price structure the bar sits.

**Why:** Two identical trajectories match the same seeds, but one at the top of a 1h range (no room) and one at the bottom (full room to run). This is why OOS is $264/day not $500+. The user identified this as the primary edge gap on 2026-03-22 during first live sim test.

**Features to add (research line):**
1. 1h z-score — position in hourly regression band
2. 1h F_momentum sign — with or against hourly trend
3. Distance to recent 1h pivots (ATR-normalized)
4. MTF alignment score (how many TFs agree on direction)

**Critical nuance (user insight 2026-03-22):**
Physics leads ENTRY correctly — momentum building, exhaustion forming. But EXIT (funnel flip)
is also purely physics. It doesn't know if the peak happened at a structural price level
(real reversal) or in the middle of nowhere (noise pause). Result: exits too early at noise
peaks, too late at real structural peaks.

**Implication:** Peak seeds and trend seeds MUST stay separate.
- Trends = physics events (PhysicsEngine handles these)
- Peaks = price events (need price structure: bands, pivots, Fibonacci)
- PhysicsEngine should NOT handle peak exits — it needs a price-aware exit layer

**How to apply:**
- Enrich seeds with higher-TF state at entry time (1h z, 1h fm, 15m z)
- Add to TRAJ_KEYS in physics_engine.py (expand from 12 to 15-16 features)
- For EXITS: add price-relative features (distance to regression bands, prior pivots)
- Re-run OOS to validate improvement
- Related: docs/Active/RESEARCH_MTF_POSITION.md


## Source: project_probabilistic_system.md

---
name: Probabilistic System Architecture
description: 4-brain cascade + peak-to-trend lifecycle + evolving CNN + proven exits — the next-gen trading system
type: project
---

## Probabilistic System (designed 2026-03-30)

Full spec: `docs/Active/PROBABILISTIC_4BRAIN_SPEC.md`

### Architecture
- **Frozen Base CNN**: ProbabilisticTrajectory (22D → 10 horizons × P(long)) — perception, never changes
- **Evolving Trade CNN**: copy of Base, fine-tunes from live outcomes (real + ghost trades from crow)
- **Bayesian Table**: IS→OOS→Live calibration cascade (`core/brain_cascade.py`)
- **Templates**: grow from trade seeds (not pre-built offline)

### Trade Lifecycle: Peak → Trend → Reversal
- Peak detector fires → CNN confirms trajectory → ENTER
- P(direction) sustained across horizons → phase = TREND, trail loosely
- Near horizons (n+1..n+3) weaken while far (n+7..n+10) still strong → REVERSAL IMMINENT
- Near horizons flip (< 0.5) → EXIT before reversal hits
- Each closed trade becomes a seed → seeds cluster into templates → exit params improve

### Key Insight
- Exit mechanisms already work (trail, giveback, envelope, SL) — they just need good entry direction
- CNN provides direction + trajectory shape → exits parametrized by trajectory
- Ghost trades (crow/phantom system) provide 10x more training data for live CNN evolution

### Files Built
- `core/brain_cascade.py` — CalibrationBrain + BrainCascade (4-layer)
- `core/probabilistic_engine.py` — ProbabilisticTradingEngine with lifecycle phases
- `training/train_probabilistic_forward.py` — IS/OOS forward pass

### Still TODO
- Phase D: `live/prob_launcher.py`
- Replay buffer for live CNN fine-tuning (real + ghost trades)
- Wire crow/phantom into replay buffer
- Divergence check: Trade CNN vs Base CNN → rollback if worse

### CNN TF Mismatch Found (2026-03-30)
TradeCNN was trained on 1m data but live fed it 15s bars. Fixed: live now aggregates 4×15s→1m before CNN prediction.


## Source: project_quantum_reconnect.md

---
name: Quantum State Reconnection — Critical Architecture Direction
description: The system was designed as a probabilistic quantum model. Gates broke it. Reconnect wave function to scoring. DO NOT FORGET THIS.
type: project
---

## CRITICAL: The System is Quantum — Not Statistical

The Bayesian-AI trading system was designed from first principles as a **quantum-mechanical
model of price behavior**. Price is a particle in a three-body gravitational field. The
"statistical" renaming was a metaphor purge that kept the math but disconnected the
conceptual framework from the decision logic.

**Why:** User's key insight (2026-03-15): "Price behaves like the electron position in an
atom — you can't actually know where it is because by the time you observe it, it's in
another place. That's the reason the old names are quantum and Roche — this should be
a probabilistic system."

## The Three Theories

1. **Nightmare Protocol** — O-U mean reversion + Roche limits (2σ/3σ boundaries)
2. **Three-Body Problem** — Macro/Meso/Micro alignment solved probabilistically via wave function
3. **Resonance Cascade** — Cross-TF synchronization → disable TP, ride the shockwave

## What's Computed Every Bar But UNUSED in Scoring

These are in `MarketState` RIGHT NOW but disconnected from entry/exit decisions:
- `prob_center`, `prob_upper`, `prob_lower` — wave function collapse probabilities
- `entropy_normalized` — chaos measure (high = three-body misalignment)
- `coherence` — superposition measure (low = collapsed = decisive)
- `tunnel_probability` — P(mean reversion) via O-U Monte Carlo
- `reversion_probability` — used in tunnel gate but not scoring

## What's Missing

- **Lyapunov exponent** — λ<0 fade edges, λ>0 ride breakout (not computed)
- **Fractal diffusion** — σ = σ_base × (v_micro/v̄_macro)^H (not implemented)
- **PID control force** — Kp*e + Ki*∫e + Kd*de/dt (term_pid exists but unused in forces)
- **is_resonance_cascade** flag — cross-TF ADX>30 + Hurst>0.55 + aligned DMI
- **Cascade mode** — suppress TP, trail via survival_stop only until Hurst decays

## What Gates Broke

The gate cascade imposes **binary decisions on continuous probabilities**:
- Hurst < 0.50 → BLOCK (but 0.49 and 0.51 are the same state)
- Conviction < 0.48 → BLOCK (rubber stamp, barely filters)
- Distance > 3.0 → BLOCK (proximity ≠ probability)

Should be: **P(success) = f(wave_function, tunnel_prob, coherence, conviction, funnel)**
No gates. One probability. Best P(success) wins.

## Reference Documents (all in docs/reference/)

1. `quantum_field_engine_original.py` — 919 lines, the pre-purge engine with all math
2. `QUANTUM_STATE_REFERENCE.md` — Gemini's theoretical spec (three theories + cascade protocol)
3. `NIGHTMARE_PROTOCOL.md` — Master equation: dX = O-U + fractal_diffusion + PID + jump
4. `THREE_THEORIES.md` — My mapping of quantum concepts to current codebase

## The Decision Funnel (validated by research)

- Funnel tree trained on I-MR auto seeds: **85% accuracy** predicting regime starts
- Key features: tight price_range (65%) + high sigma (17%) = energy coiling → breakout
- This IS the pre-resonance state detection
- Spec: `docs/Active/SPEC_DECISION_FUNNEL.md`

## Next Session Action Items

1. **Run fresh**: `python training/trainer.py --fresh --lookback` (brain stale)
2. **Read quantum_field_engine_original.py** — map which wave function computations
   survive in current statistical_field_engine.py
3. **Add is_resonance_cascade** to MarketState or BeliefState
4. **Replace gate cascade scoring** with P(success) from wave function probabilities
5. **Implement cascade mode** — suppress TP when resonance detected

## PFMEA Top Issues (from this session)

| RPN | Issue | Status |
|-----|-------|--------|
| 648 | Score competition ignores conviction | OPEN — needs P(success) scoring |
| 392 | Conviction gate after competition | OPEN — wrong cascade position |
| 336 | Giveback IS→OOS flip | INVESTIGATE |
| 300 | Brain reject dead (0 blocked) | OPEN |
| 243 | BreakevenLock 4-tick activation | **FIXED** (TrailingStop rewrite) |

## User Preferences (from this session)

- **Challenge ideas HARD** — the user expects resistance, not compliance
- **Workers decide** — no artificial thin-market skips, let the system filter
- **SL is last resort** — tolerance interval from MAE distribution, not fixed
- **Probabilistic, not deterministic** — no magic numbers, no binary gates
- **The physics names were correct** — don't strip the quantum vocabulary


## Source: project_regret_six_layer_architecture.md

---
name: regret-six-layer-architecture
description: The 6-layer architecture of the regret-oracle research arc — L1 oracle (done), L2 direction (done), L3 Bayesian archetypes (pending), L4 selector (missing), L5 execution (missing), L6 validation (missing). Use this frame to organize new work and identify gaps.
metadata:
  type: project
---

The regret-oracle research arc is organized as a 6-layer architecture.
When new work is proposed, identify which layer it touches before building.

**L1 — Trade Seeds via Regret (DONE)**
Tool: `tools/regret_daisy_chain_oracle.py`
Output: `reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv`
Status: 7,925 sequential non-overlapping trades, 2025 full IS, 5s base TF,
60-min window cap. Sequential ceiling = $1,046,546/yr (~$4,500/day under the
1-hour-budget premise). This is the LABEL UNIVERSE — what's theoretically
capturable. It is NOT a strategy; it requires lookahead (centered-window
extrema detection + best-extreme-in-1h window).

**L2 — Direction Discrimination (DONE)**
Tools: `regret_kway.py`, `regret_stratified.py`, `regret_distribution_eda.py`,
`regret_feature_regression.py`, `regret_pair_clusters.py`,
`regret_triplet_clusters.py`, `regret_pair_regression.py`,
`regret_triplet_regression.py`
Findings: signed_mfe target pivot was decisive ([[signed-mfe-pivot]]).
R² saturates at ~0.35 ([[kway-r2-saturation]]). Stratified k=2 matches
unstratified k=5. Direction-callable cells at 43-59% rate; per-cell
accuracy 82-86% in callable cells; 93% in extreme cells (>90% one side).
Findings doc: `reports/findings/regret_oracle/2026-05-16_direction_signal_kway.md`.

**L3 — Bayesian Table via N-D Trajectory (PROTOCOL LOCKED 2026-05-16, BUILD PENDING)**
Spec: `research/bayesian_archetypes/project.md` (DMAIC frame).
See [[bayesian-archetypes-pending]] for locked decisions + deferred open questions.
The L3 build will add: PCA-line clustering, hierarchical r-ladder,
trade-decay tracking that unifies entry classifier + exit signal +
duration prediction + Bayesian posterior update for direction.

**L4 — Selector / Strategy (MISSING)**
Real-time entry trigger that combines L2 cell-gates, L3 cluster matches,
or an ensemble. Open architectural question: discrete-cell selector (L2-only)
vs trajectory-match selector (L3-driven) vs both.

**L5 — Live Execution Model (MISSING)**
Costs ($0.50/tick × 2-4 ticks slippage + $1-2 commission per round-turn on
MNQ), spread, halt-after-N-losses, intra-day DD caps. Per CLAUDE.md
anti-doom-cascade rule: report under multiple cost/intervention assumptions.

**L6 — Validation (MISSING)**
2026 OOS run of L1+L2+L3 cell-gates and cluster-tables. Day WR + mode $/day
+ 95% bootstrap CI per CLAUDE.md protocol. Per-cell / per-cluster
sign-stability OOS check. Live-vs-sim gap measurement (historical gap ~$680/day
per Day-1 v1.0 evidence per CLAUDE.md).

**Architectural observation**: L2 and L3 answer different questions
(L2 = discrete cell classifier; L3 = trajectory matcher with decay).
They are complementary, not substitutes. The natural test when L3 lands:
does L3 add edge over L2 alone? If yes → ensemble. If no → L2 cell-gate
is good enough and L3 was conceptually rich but practically redundant.


## Source: project_resonance_cascade.md

---
name: Resonance Cascade Hypothesis
description: Multi-TF peak agreement detects crashes/rallies. Each TF pair (child+parent) validates peaks as real or fake. Full cascade (5/5 pairs agree) = crash/rally.
type: project
---

Every TF has a parent that validates its peaks:
- 15s -> 1m, 1m -> 5m, 5m -> 15m, 15m -> 1h, 1h -> 4h

When ALL pairs agree on direction = resonance cascade = extreme trend.
When <3 pairs agree = chop. Feb 9 was full 5/5 SHORT cascade.

A "trend" is NOT a separate concept. It's:
1. The decay of peaks in one direction over time
2. The distance between two macro-scale peaks
3. Visible through rolling MFE per direction at micro scale

**Research needed**: `tools/resonance_cascade_research.py`
- Run peak detection on ALL TFs simultaneously
- Count pair agreement per bar
- Correlate with next N bars' direction
- If 5/5 agreement predicts 90%+ accuracy = cascade detector works

**Why:** This would turn Feb 9 from -$1K to +$5K by riding the crash.
Same peak detection code, just applied at multiple scales simultaneously.

**How to apply:**
- Cat brain reads macro peak state (1h/4h) for regime direction
- When cascade detected, ALL micro peaks forced to cascade direction
- Peak still provides timing (bounce entry), cascade provides direction
- Ref: Half-Life resonance cascade (Dr. Freeman), Nightmare Protocol fractal


## Source: project_rl_pivot_2026_05_28.md

---
name: project-rl-pivot-2026-05-28
description: Project pivoted from supervised CNN/blended stack to an RL engine (PW-CRL); the deletion left dangling imports in live + run.py
metadata: 
  node_type: memory
  type: project
  originSessionId: c2947774-6098-4d63-8985-876610818bcd
---

As of commit `4b658e2a` (pushed 2026-05-28), the project pivoted from the
supervised CNN / blended / "nightmare" trade-management stack to a
**reinforcement-learning engine** branded "Parallel Worlds Curriculum RL"
(PW-CRL): a CNN+LSTM DQN with V-trace off-policy correction, a hindsight-regret
"shadow queue", curriculum learning (`EXIT_NMP → ENTRY_NMP → YOLO`), and an
8-agent DOE sweep. Lives in `training/rl_engine/`. Architecture doc:
`rl_whitepaper.md` (repo root). `training/` was reorganized: shared helpers →
`training/utils/`, CNN/GBM trainers → `training/trainers/`, `regret` is now a
package (`training/regret/`), new `training/strategies/zigzag.py`.
`training/build_dataset_v2.py` → `core_v2/build_dataset.py`.

DELETED in the same commit: `cnn_entry/exit/flip/hold/risk/trade_manager.py`,
`nightmare*.py`, `nightmare_blended.py`, `compute_features.py`, `ai.py`,
`physics_labels.py`, `model.py`, `memory.py`, `feature_processor.py`,
`run_baseline.py`, `forward_blended.py`, `live_feature_engine*.py`.

**Why:** per `rl_whitepaper.md`, supervised exit models destroy net PnL by
cutting winners — the first 5 min of a $400 winner is statistically identical to
a loser, so a supervised model amputates the right tail. RL learns exits
directly from the PnL reward. The RL engine is **mid-training as of 2026-05-27**
(curriculum segment 10, manual overfit/LR interventions) — NOT yet a deployed
production path.

**How to apply:**
- CLAUDE.md "Key Files" / "Active Work", `docs/memory/MEMORY.md`, and
  `AGENTS.ini` (`status = production blended pipeline`) all predate this pivot —
  treat them as stale on the CNN/blended pipeline until updated.
- **Known regression (introduced 4b658e2a):** the deletions left ~15 active
  files importing now-deleted modules — `live/live_engine.py`,
  `live/maintenance.py`, `live/diagnostic_run.py`, `training/run.py`, plus
  `tools/util/{hypothesis_test,blended_test}.py`,
  `tools/archive/lookahead_impact.py`, `tools/exits/giveback_analysis.py`,
  `tools/eda/sunday_hourly_eda.py`, `tools/data/validate_sfe_parity.py`, and
  `tests/test_{engine_evaluate,sim_executor}.py` import deleted
  `nightmare`/`nightmare_blended`/`compute_features`/`ai`/`physics_labels`.
  These ImportError on invocation. The `regret` imports are FINE (package
  re-exports resolve). OPEN QUESTION (not yet resolved with user): is the
  blended layer intentionally retired (so this is acceptable transitional debt
  and `engine_v2.py` supersedes `live_engine.py`) or does the live path need
  these modules restored/rewired?
- The Telegram bridge tooling (`telegram_*.py`, `push_alert.py`) now reads the
  bot token from `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars. The token
  that was hardcoded should be treated as compromised (user advised to rotate).

See [[feedback-cli-script-false-orphans]] — the same "no Python imports ≠ no
callers" caution applies in reverse here: deletions without grepping importers
broke live code.


## Source: project_roadmap_q2.md

---
name: Q2 2026 Roadmap — PhysicsEngine (bandaid) + AdvanceEngine (real build)
description: 3-month roadmap to NQ graduation. Two parallel tracks: PhysicsEngine runs now for data/funding, AdvanceEngine gets the full grounded rebuild.
type: project
---

## DMI Flipper (running NOW, funds research — $208/day backtest)
- [x] Deployed to sim (2026-03-25, Sim101, bridge v6.8.1)
- [x] Cross mode: smoothed DMI zero-cross + TP=10t repeating + SL=40t
- [x] Safety lock: refuses non-sim accounts
- [x] Connection loss: bridge sends CONNECTION_LOST, Python flattens + stops
- [x] Bridge reduced to 3 TFs (1s/1m/1h) — 83% less memory
- [ ] Collect 30+ days of data for statistical validation
- Backtest: $208/day (14 months), $82/day OOS, 29.8% WR, 3:1 win/loss

## AdvanceEngine V2 (the real build — grounded templates)

### V2 — K-Means Templates (BUILT, ready to run)
- [x] 70D grounded features (7 features × 10 TFs): `core/grounded_features.py`
- [x] Template matcher with CNN-ready interface: `core/template_matcher.py`
- [x] Training pipeline: `training/advance_v2_trainer.py`
- [x] Full lookahead labeling (Phase 3)
- [x] OOS validation (Phase 5)
- [x] Per-template configs (SL/TP/direction/hold)
- [x] Trade marker logger for inspection
- [ ] Run `python -m training.advance_v2_trainer --phase all`
- [ ] Compare to DMI flipper baseline ($208/day)

### V3 — Brain + Templates (next)
- Load old brain (pre-lookahead-fix) OR train new brain on IS
- Brain provides direction per template state
- DMI provides timing (when to trade)
- Templates provide recognition (what state is this)

### V4 — CNN Direction Predictor (research)
- Simple 1D CNN: 70D input → LONG/SHORT
- Custom loss: maximize PnL not accuracy
- Train IS (6mo), validate OOS (8mo)
- Must beat K-means baseline

### V5 — TCN Multi-Resolution (research)
- Input: (10 TFs, 7 features) as 2D matrix — NOT flattened
- TCN convolves across TF axis with dilations
- Learns cross-TF relationships automatically
- Greedy layer training: add layers until no OOS gain
- See `memory/research_tcn_v5.md`
- Reference: `examples/Overview.md`

### Goal: Merge + Graduate (Month 3)
- Best version replaces DMI flipper as single live engine
- 30+ days consistent profitability on MNQ sim
- Graduate to NQ (same system, 10x capital)
- Target: 2026-06-23

## Operational Costs & Break-Even
- Claude API: $100/month
- Power: ~$40/month
- Daily target: **$60/day** ($1,320/month) to cover costs + incentive
- On MNQ: 120 ticks/day profit, or ~4 clean trades at $15 average
- On NQ: 12 ticks/day profit — trivial if system works
- First session (2026-03-22): $1,495 gross, $264 without outlier
- Biggest leak: ORPHAN_FLATTEN (-$572) + bad SL entries (-$354) = -$926 lost to plumbing

## Key Principles (from 2026-03-22 session)
- Base measurements: Price, Time, Volume — everything else is derived
- DOE principle: measure the PROCESS (market) not the EQUIPMENT (system)
- Derivatives OK at any order if each step answers a nameable question
- The market is NOT Brownian — edge = deviations from random
- Variance ratio = the Brownian test (replaces Hurst + ADX)
- Liquidity is latent — measure effects (volume at price, rejection)
- Templates were premature optimization — raw K-NN at 38K is fast enough
- Optimize for exhaustion detection, not direction prediction
- Everything grounded in probability with sample size


## Source: project_template_revival_via_chord_keying.md

---
name: Templates / Bayesian table — the engine was right, the KEY was wrong (2026-05-09)
description: Existing hierarchical Bayesian table at training_iso_v2/bayesian_table.py is architecturally correct; the failure was keying by day-aggregate regime label (biased). V0 swaps the key to primitive chord and reuses the engine.
type: project
---

**Discovery 2026-05-09 evening, after architecting V0 from scratch:**

The exploratory work today (event-segmentation, 15m CRM characterization,
oracle-driven failure-mode framing, meta-router architecture, statistical
labels) made me think we were inventing a new system. We weren't.

`training_iso_v2/bayesian_table.py` already implements:

- Per-cell posteriors:
    WR (Beta), EV/$/trade (Normal), peak_$ (Normal), MAE_$ (Normal),
    time-to-peak (Normal), capture-ratio (Beta)
- Hierarchical shrinkage: cell -> tier-only -> universal (Empirical
  Bayes; thin cells borrow from parent).
- Built offline from regret labels (oracle-derived per-trade outcomes).
- Used as the substrate for the adaptive exit-threshold optimizer.

This is the ENGINE the V0 work needed. What blocked it from being
useful before was the KEYING:

| | OLD (existing) | NEW (V0) |
|---|---|---|
| key | `regime_idx` (1 of 6 day-aggregate labels) | primitive chord = (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q) |
| cardinality | 6 | up to 5*3*5*5*5 = 1,875 |
| bias | label computed from same metrics it segments | bar-level, no day-aggregate, no lookahead |
| parent | tier-only | axis-marginal (5 parents instead of 1) |

The architecture stays. The shrinkage / posteriors / Empirical Bayes
/ outputs all stay. We swap the index, not the engine.

## What V0 build now consists of (≈80% reuse)

1. Add a `chord_index(state) -> tuple` function that produces the
   primitive chord at-bar from the 5 CRM features computed in
   `tools/event_bucket_15m_crm.py`.
2. Modify `bayesian_table.py` keying:
   - Cell key: chord tuple instead of regime_idx
   - Parent key: per-axis marginal (5 parents); cell shrinks toward
     each axis-marginal weighted by axis informativeness
   - Universal: same as before
3. Add a fit() pass that consumes the 5,340 IS macro events with
   their chord at entry, computes per-cell posteriors with shrinkage.
4. Add live-time lookup function chord_lookup(chord_index) ->
   {P_cascade, P_continuation_60m, expected_max_z, expected_duration,
    per_tier_oracle_$/trade}.
5. (NEW) Compute P_cascade per cell from oracle event resolution
   (e.g., duration > 30min OR max_z >= 4) — this is the meta-router
   signal that didn't exist in the old (regime, tier) table.

## What this means for the session work

- The exploratory work today wasn't wasted; it identified the missing
  primitive substrate (the 5 CRM-derived axes) that the engine needed.
- The architectural locks (filters vs tiers, statistical labels,
  meta-router two-level decision) all map cleanly onto extending
  `bayesian_table.py`.
- We're not building a new system. We're swapping the cell-key in
  an existing system that already has the math right.

## Connections / lineage

- `core/bayesian_brain.py` (legacy) — predecessor probability table
- `training_v2/bayesian_table.py` — V2 variant (regime keying)
- `training_iso_v2/bayesian_table.py` — current canonical implementation
- `training_iso_v2/regret.py` — produces the per-trade oracle outcomes
  that the table is fit on
- `tools/event_bucket_15m_crm.py` — produces the new chord substrate
- `DATA/ATLAS/regime_labels_2d.csv` — DEPRECATED as a key (biased)

## Why the user's pattern brain was right

Pre-2026-05-09 attempts to use templates / probability tables failed
not because the math was wrong but because the cells were too coarse
(6 day-aggregate regimes) AND the cells were biased (regime label
computed from same metrics that determine outcomes). The exploratory
work today produced the unbiased, fine-grained substrate the engine
needed. Templates ARE the right architecture; we just had to do the
substrate work first.


## Source: project_tradecnn_baseline.md

---
name: TradeCNN Baseline (SUPERSEDED)
description: TradeCNN $1,609/day OOS was on NT8 phantom spike data — superseded by nn_v2 3-CNN system
type: project
---

## SUPERSEDED (2026-04-03)

TradeCNN $1,609/day OOS was achieved on NT8 data that contained phantom spikes.
Clean Databento data turned +$4,350 into -$2,427. The edge was fake.

The nn_v2 3-CNN system ($613/day OOS on clean data) is the current baseline.

**Why:** Historical context only. Do not reference TradeCNN numbers as targets.
**How to apply:** All baselines must be on clean Databento data, not NT8 exports.


## Source: project_tunnel_slippage.md

---
name: tunnel probability = live slippage model
description: tunnel_prob serves dual purpose — entry scoring (P reach TP) AND live fill probability (slippage through order book barrier)
type: project
---

Tunnel probability (`reversion_probability` in current code) maps to live trade slippage:
- Original: P(price reaches target band without hitting SL) via O-U Monte Carlo
- Live extension: P(fill at expected price) given order book depth + volatility

**Why:** At high sigma, order book thins (HFT algos pull liquidity), spreads widen.
The "barrier" the order must tunnel through becomes thicker. Low tunnel_prob =
expect slippage, reduce size or skip.

**How to apply:**
- Factor tunnel_prob into position sizing (Kelly × tunnel_prob)
- Log expected vs actual fill price in live trade logger
- Calibrate tunnel_prob → slippage model from live fill data over time
- At 1σ (PID trance): thick book, clean fills, tunnel_prob high
- At 3σ+ (cascade): thin book, slippage likely, tunnel_prob low


## Source: project_useful_v2_signals.md

---
name: Useful V2 signals (chart-validated 2026-05-08)
description: Signals confirmed visually/empirically as informative on 2026_02_12; signals confirmed as redundant or noise
type: project
---

Validated by visual inspection on 2026_02_12 (OOS best #5 — clean morning rally + afternoon crash). Confirmed by SPC/I-MR analysis across 277 IS days where applicable.

**REVERSION / 3-BODY FRAMEWORK (locked 2026-05-09)**

Terminology:
- pivot ≡ inflection ≡ bar where direction changes
- reversion = the new leg that begins at the pivot (NOT mean-reversion in the OU sense)
- trend-follow / ride = trade WITH the move
- Each leg is a reversion (relative to the prior leg's direction)

3-body envelope:
- M_close (blue) = center / target / close-volatility
- M_high (green) = upper bar-extreme regression mean
- M_low (red) = lower bar-extreme regression mean
- Each anchor has its own σ envelope (±1, ±2, ±3)
- All 3 acts as elastic anchors — no force-free equilibrium

Goldilocks operating levels:
- ±1σ: too common (~32%) — no edge
- ±2σ: ★ PRIMARY trigger (~5% of bars) — real reversion edge
- ±3σ: extreme (~1.5%) — often regime-shift signal
- M_close ±3σ_close: outer-wall confirmation (rarest)

Mixed-regime asymmetric TF design:
- HL anchors at SLOW TF (15m, 1h, 4h) — rare extreme triggers
- CRM at FAST TF (1m, 5m) — responsive target
- At 1h-HL ±2σ, P(continuation) ≈ 0 (reversion structurally certain)

**MACRO-EVENT PROBLEM** (2026-05-09 finding):
The reversion framework FAILS during 2-3 hour impulse phases. Real-time
distinction between "in macro-event impulse" and "calm regime" is the
missing operational gate. 2026_03_03 lost $1,324 with reversion fires into
a 3hr crash. OOS forward-pass-honest = −$40/day — not because strategies
are bad, but because no impulse-phase suppressor exists.

CRM flatten-then-pivot detector (designed): state machine NORMAL →
DIRECTIONAL → FLATTENED → PIVOT → IMPULSE → STABILIZING → NORMAL,
walked bar-by-bar with no lookahead. Needs 5-min monitor window after
pivot to filter wiggles vs real impulses.

Compression-precedes-expansion validated empirically (1h, 50 IS days):
σ-rank ≈ 0.42 at T-60min before HIGH+ extremes (compressed),
rising to 0.63 by T-15min. Tradable early-warning lead time.

**STRUCTURAL REQUIREMENTS — every probability table must condition on**:
1. Time-of-day (`L0_time_of_day` available; reversion edge varies wildly
   across session: lunch highest, opens lowest)
2. Day-of-week (Friday vs Monday have different regimes)
3. Calendar event (FOMC/NFP/CPI = scheduled-impulse days; very different
   distributions; need external flag)
4. Current-day-state-so-far (already had impulse? quiet morning? Multiple
   impulses?). Day-state machine needed across the session.

These conditioning axes MUST be in the design from the start — adding
them later changes all bin populations and invalidates earlier calibrations.

**TODO (deferred work)**

- **Empirical breach probability lookup** (deferred): build a per-bar `P(price breaches +k·σ band in next N bars)` lookup. Walk IS data, group by `(z_se_bucket × regime × N × tod × dow × cal_event)`, count the fraction of forward windows where `max(z_se) ≥ k`. Replaces the broken `reversion_prob_w` (which is saturated by NMP entry-gate selection bias).
- **Macro-event detector v2** with 5-min monitor window for impulse confirmation
- **9-layer probabilistic stack** — fuse single-layer probabilities into compound conviction
- **Replay of 2026_02_12 + 2026_03_03** with macro-gate active to quantify expected impulse-day damage avoided

**USEFUL signals — keep / wire**

- `L2_1m_price_mean_15` — 1m regression mean. Hugs price tightly. Tradable as the "immediate fair value" anchor.
- `L2_5m_price_mean_9` — 5m regression mean. Smooth, less jitter. Reference for "where 1m wants to revert to."
- `L2_15m_price_mean_12` — 15m regression mean. Strategic direction line — its slope (over 1h lookback, Q75 magnitude) is the day-scale regime signal.
- `L2_5m_price_sigma_9` — 5m SE bands at ±1σ/±2σ/±3σ/±4σ. Cone width is itself a regime signal: tight cone = calm (fade-friendly), expanding cone = transitioning, wide cone = volatile.
- **1m-5m mean divergence** (`mean_1m − mean_5m`): per-bar continuous tradable signal. Cross above +Q75 = SHORT entry; cross below −Q75 = LONG entry; snapback through zero = exit.
- **15m mean slope (1h lookback)**: strategic gate. Sign of the slope at Q75 magnitude defines trade direction bias for the period.
- **15m mean curvature** (slope-of-slope): magnitude marks sharpness of pivot; combined with slope sign-change pinpoints inflection points. Q75 of |curvature| filters drift-throughs from real pivots.
- `L2_*_vol_mean_w` + `L2_*_vol_sigma_w` — **regime-change detector ONLY, NOT a direction signal**. Vol spikes look identical at rally peaks and crash legs (both = "something is happening"). Use as CONVICTION multiplier on directional signals or as a REGIME gate ("event day vs normal day"), never as a buy/sell trigger by itself.
- `L2_*_vol_velocity_w` + `L2_*_vol_accel_w` — **LEADING pre-pivot timing signal (direction-agnostic)**. Visually validated on 2026_02_12: at the 14:00-14:30 macro pivot all four volume-derived features (vol_mean, vol_sigma, vol_velocity, vol_accel) spike together BEFORE price breaks. Not a "which way" signal — a "WHEN" signal: tells you a pivot is imminent. Combine with σ-rank compression (T-30 to T-60min) for high-confidence early warning. **Multi-signal pre-pivot pattern**:
  1. σ-rank compression (T-30 to T-60min): bands narrow
  2. vol_velocity rising (T-N min): activity ramping
  3. vol_accel spiking (T-1 min): activity acceleration
  4. CRM slope flattening (T-K to 0): direction losing conviction
  5. CRM slope flips (T = 0): pivot fires
  6. vol_mean / σ explode (T+): impulse confirms

  Refined volume rule: **volume is direction-agnostic for WHICH WAY price will go — but vol_velocity/vol_accel are direction-agnostic LEADING indicators of WHEN a pivot is about to happen.**
- `L3_1m_swing_noise_15` — chop/calm gauge. Spikes 4-5x baseline at vol regime transitions. Matches SE band expansion visually.

**REDUNDANT — drop**

- `L2_*_vwap_w` at all TFs — visually identical to `price_mean_w` at the same TF. Pick one, drop vwap.
- `L2_4h_price_mean_18` — barely moves intraday (-9.5 pts on a -408 pt day). Useless at intraday timescale.
- `L2_1h_price_mean_12` line itself — step function at TF cadence, lagged. Drop the line, but its computed slope is useful.

**NOISE — don't use as primary signals**

- `L1_1m_price_velocity_1b` and L2 velocity at 1m / shorter — too noisy on its own. Sign agreement with longer TFs is required.
- `L3_1m_hurst_15` — jitters around 0.5 boundary. Not stable enough alone; would need smoothing.
- `L3_1m_reversion_prob_15` — saturated near 1.0 at NMP-qualifying bars (selection bias). Selection bias: rprob is high at entries (because NMP gates on it) and stays high through MFE. Can't use to detect "thesis dying."

**STRUCTURAL FACT — TF-anchored features**

V2 features at TF=1h, 4h, 1D update at TF cadence and hold flat between updates. At 5s sampling, 90-97% of signed bar-to-bar MR values are zero. **For velocity/slope/MR analysis on TF-anchored features, sample at TF cadence or compute slopes over rolling lookbacks at 5s cadence (not bar-to-bar diffs).**

**THREE-ROLE COMPOSITE FRAMEWORK** (validated visually on 2026_02_12):

```
Strategic gate (15m mean slope, 1h LB, Q75)
  green-shaded → LONG bias allowed
  red-shaded   → SHORT bias allowed
  un-shaded    → either / no-trade

Tactical entry (1m − 5m divergence)
  cross +Q75 → SHORT signal (allowed only in red regime)
  cross −Q75 → LONG signal  (allowed only in green regime)

Tactical exit
  divergence crosses back through zero (snapback)
```

Additional gates available but not yet wired:
- volume regime (vol_sigma expansion = step aside / wider stops)
- swing_noise (chop in calm regime → fade ok; chop in volatile → pass)

**Charts of record**:
- `chart/2026_02_12_regression_means.png` — price + 1m/5m/15m means + 5m SE bands + slope panel + 1m-5m divergence panel + strategic shading + inflection markers
- `chart/2026_02_12_other_features.png` — V2 feature panorama (vol, swing, z, hurst, accel)


## Source: project_v2_features_eda_stack.md

---
name: V2 Features × Price Descriptive EDA Stack
description: Seven-layer descriptive EDA tools characterizing v2 feature behavior across regimes & price phenomena. No fitting. State-fingerprint findings (4h chord cells 100% pure for FLAT_SMOOTH/CHOPPY).
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
# V2 Features × Price — Descriptive EDA Stack

**Established:** 2026-05-03.
**Status:** 7 layers built. All run on IS only (208 days, 47k 5m bars). No fitting; pure characterization. Substrate for future regime-conditional strategy work.

## The 9 layers (updated 2026-05-03 evening)

| Layer | Tool | Question answered |
|---|---|---|
| **TF sweep** (NEW) | `tools/v2_features_tf_sweep_eda.py` | For each (concept, TF), what's the regime-separation strength? Which concepts INVERT sign across TFs? |
| **Context** (NEW) | `tools/v2_features_context_eda.py` | Which features don't predict price directly but RESHAPE other features' price relationships? |
| A1 single, current | `tools/v2_features_regime_eda.py` | Per-feature × per-regime distributions; which features separate which regimes (Cohen-d) |
| A2 pair, current | `tools/v2_features_pairwise_eda.py` | Which pairs interact non-additively; which (X_q, Y_q) cells have extreme price reactions |
| B1 single, lookback | `tools/v2_features_lookback_eda.py` | When feature X does pattern P over N bars (mono/spike/reversal), how does price react |
| B2 pair, lookback | `tools/v2_features_lookback_pair_eda.py` | Joint patterns: when two features both spike up, both fall mono, etc., how does price react |
| Chord (triplet) | `tools/v2_features_chord_eda.py` | Music metaphor: 3-feature combinations whose joint quantile cells encode recognizable state fingerprints |
| Visual overlay | `tools/v2_features_overlay_viz.py` | Per-regime per-day PNGs: price + 12 features + chord stacked for eye-pattern recognition |
| Volume × variation | `tools/v2_features_volume_variation_eda.py` | 4 quadrants of (vol, var) — fakeout (LOW_VOL_HIGH_VAR), compression (HIGH_VOL_LOW_VAR), capitulation, dead zone |

## Structural finding (2026-05-03 evening)

**The composite signal can't be additive.** Contextualization analysis showed `body` at 1h FLIPS sign in its correlation with forward return depending on `vol_sigma_w`'s quantile (corr ranges −0.108 to +0.039). A model that averages or sums them loses this. The right composite framework is **conditional**: target's sign depends on modifier's quantile bin. Two regime-router levels emerge:
- Day-level: route between strategies based on regime_2d (UP_SMOOTH, FLAT_CHOPPY, etc.)
- Bar-level: route target sign based on modifier quantile (intra-bar contextualizer-router)

These are independent layers; both apply.

## TF inversion findings (2026-05-03 evening)

The same concept's regime relationship CHANGES character with timescale:
- **bar_range**: −0.18 (5s) → −0.24 (5m) → +0.18 (1D). Intraday wide range = sells; daily wide range = bull rally. Sign flip at 1D.
- **vol_velocity_w**: ~0 across short TFs → −0.21 at 1D. Only signals at macro TF (capitulation pattern).
- **price_accel_w**: 0 at 5s/5m → +0.55 at 1D. Acceleration only meaningful when smoothed over multi-day window.
- **vol_mean_w**: −0.44 at 1h → +0.07 at 1D. Inverts at the macro boundary.

Universal directional carriers (no sign flip across TFs): price_velocity_w (+1.25 at 1h), price_velocity_1b (+0.52), body (+0.51), vol_sigma_w (−0.41), vwap_w / price_mean_w (+0.26 at 5s, decays).

## Headline findings

### Single-feature relationships
- `L2_5m_price_velocity_w` r = +0.86 with past N-bar return — it IS the recent move
- `L2_1h_price_velocity_w` separates DOWN_SMOOTH from UP_SMOOTH at Cohen-d = −1.25 (strongest single regime separator)
- Forward correlations all <0.1 — confirms what Analyses B/M/N showed: bar-to-bar direction is unpredictable from any single feature

### Pairwise current
- `L1_5m_velocity_1b × L1_5m_body` → WR ranges 42-75% across 9 cells via pure quantile binning. Recovers 70% direction accuracy from 2 features, no model fit.
- `L1_15m_velocity_1b × L1_15m_body` → 35pp WR spread (32-67%)
- Pattern: velocity × body at same TF captures "directional bar agreement" — both push same way → continuation; disagree → reversion.

### Lookback single
- `L1_5m_velocity_1b / 6 / RISING_MONO` → 69.6% WR (n=46), +38.1 tick fwd
- `L1_5m_body / 6 / RISING_MONO` → 57.9% WR (n=38), +41.1 tick fwd
- `L1_15m_body / 3 / SPIKE_DOWN` → 59.9% WR, +5 ticks (mean reversion)
- 6-bar monotonic momentum on 5m bar-shape = strong continuation; single-bar 15m spikes = mean revert.

### Lookback pair
- `(L2_1m_velocity_w, L1_5m_velocity_1b) / 6 / REVERSAL_AGREE_DOWN` → 71.0% WR (n=31), +21 tick fwd. Coordinated bearish moves bounce.
- `SPIKE_BOTH_UP at 1h+5m` → +13.7 tick continuation 56% WR (n=200+)

### Chord (3-feature regime fingerprint)
- **`L1_4h_body + L1_4h_velocity_1b + L3_4h_z_low_w`** has cells:
  - cell(1,2,0) = **100% FLAT_SMOOTH** (n=96, fwd −7.5)
  - cell(2,1,1) = **100% FLAT_CHOPPY** (n=96, fwd −22.3)
  - cell(1,0,0) = 88% FLAT_SMOOTH (n=96, fwd +10.8, WR 61%)
- 4h-TF features dominate purity rankings — 4h is the regime-identifying timescale
- The chord IS the regime classifier; deterministic, no fitting needed

### Volume × variation
- LOW_VOL × HIGH_VAR (fakeout territory): 15m_vol_mean × 15m_price_sigma cell n=486, mean_fwd **+28.7 ticks** at 53% WR. Low participation + dispersion → bounce.
- HIGH_VOL × LOW_VAR (compression): 4h_vol_mean × 4h_swing_noise cell n=636, **70% FLAT_CHOPPY purity**. Highest single-cell concentration in the entire stack.
- All 4 corners FLAT_CHOPPY-dominated (regime distribution bias). Differentiation is in forward return: fakeout averages +3.1 fwd, dead zone averages 0.

## Methodology pattern (consistent across layers)

1. **Pruning via shortlist**: Layer 1 produces top-K shortlist (Cohen-d ranking for fingerprint hunting; lookback_corr for momentum). Layers 2/B1/B2/chord use that shortlist.
2. **Quantile binning** preserves rank information without parametric assumptions; 3 quantiles per feature (sometimes 5).
3. **Min cell support** (50 bars typical) filters noise from rare-event combinations.
4. **All layers IS-only** (208 days, 47k bars). Future OOS validation = retest the high-WR/high-purity cells on the 71 OOS days.
5. **Visual overlay tool** complements the math — eye sees co-activation patterns the statistics summarize.

## What this UNLOCKS for next session

Five follow-up directions, in order of value:

1. **Compute exact conditional rules** from top contextualizer pairs. For each (modifier, target, TF) with high lift, output the explicit rule: "when modifier in Q3, flip target's sign; when in Q0, use as-is". Directly tradable filter — and the only composite framework that captures the structural finding (target sign FLIPS based on modifier quantile).

2. **OOS validation of EDA findings**. The 70% WR patterns, 100% chord cells, and contextualizer flips need to hold on the OOS 71 days to be tradable. Re-run B1/B2/chord/vol-var/context with `--split OOS`. Patterns that survive = real; patterns that disappear = sample-specific noise.

3. **State-fingerprint deployment**. Translate the top chord cells into a NinjaTrader rule. Example: when `4h_body in Q1, 4h_velocity_1b in Q2, 4h_z_low_w in Q0` → tag as FLAT_SMOOTH state. Route to FLAT_SMOOTH-appropriate strategy (zigzag bleed-score filter from prior chop-edge work).

4. **TF-axis × contextualizer cross**: do contextualizer effects ALSO invert across TFs? E.g., does `vol_sigma_w` modify `body` in opposite directions at 5m vs 1h? Single-line extension to existing context tool.

5. **Regime-stratified rerun**. Every layer has or can gain a `--by-regime` flag. The 4h chord found cells 100% pure for FLAT_*; what about UP_SMOOTH or DOWN_CHOPPY? Need stratified analysis to find chord cells specific to each regime.

## Anti-patterns ruled out

- 4-feature chords: combinatorial cost too high, sample sizes drop below useful thresholds (n<10).
- Multi-feature lookback (3+ features × 4 windows × 8 patterns each): same problem — would characterize rare events with too few samples for inference.
- Volume features in models without the framing: they don't track price velocity directly. They describe activity intensity. Used as quadrant-binners (LOW_VOL/HIGH_VOL × LOW_VAR/HIGH_VAR), they DO describe state — but as regression inputs to predict price they're useless.
- "Mixing colors" beyond layer 3: each additional dimension narrows the population to noise. Layer 3 (chord) is the natural depth limit.

## Files

- 7 tools under `tools/v2_features_*.py`
- 7 output dirs under `reports/findings/v2_features_*` and `reports/findings/v2_volume_variation/`
- See `docs/daily/2026-05-03.md` for the full session writeup
- Commits: `0a0229aa`, `3645f1ec`, `aa205bd2`, `bccea3c1`, `ba09337c` + (today's vol×var commit)


## Source: project_v2_flip_rule_discovery.md

---
name: V2 NMP regime-direction flip rule discovery
description: 2026-05-04/05 — recreated legacy ExNMP discovery in V2; per-(regime × direction) cross-tab found a flip-rule splitter; per-cell continuous-feature filters overfit on OOS.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

Recreation of the legacy 9-tier ExNMP discovery methodology (2026-04-06 to
04-18) in V2-native form, run on 19,106 NMP-only IS trades.

Legacy methodology was a 4-step recursion:
  1. Run base NMP entry → trades
  2. Find the splitter axis where trade outcomes diverge most
  3. Sub-classify entries by that axis
  4. Add new entry types when EDA reveals signals NMP misses

Step 2 in V1 found: velocity at entry, CNN flip direction, 1h alignment,
wick rejection. Step 2 in V2 found: **regime × direction interaction**.

## Findings

### 1. Single-column V2 features at entry have NO direction-flip signal

Cohen's d across all 185 V2 columns (FADE_BETTER vs FLIP_BETTER cohorts):
maximum |d| = 0.040, **0/25 features survived walk-forward**. The legacy
CNN flip's 70.6% accuracy came from cross-feature patterns + features
(wick_ratio, dmi_diff) that aren't in the V2 entry vector.

### 2. Categorical (regime × direction) IS the splitter

Per-(regime, direction) cross-tab on 19,106 IS NMP trades:

| regime | dir | $/trade | fade_peak | flip_peak | verdict |
|---|---|---:|---:|---:|---|
| UP_SMOOTH | long | +$2.56 | $91 | $59 | KEEP |
| UP_SMOOTH | short | -$1.67 | $60 | $93 | **FLIP** |
| UP_CHOPPY | long | +$1.25 | $96 | $75 | KEEP |
| UP_CHOPPY | short | -$0.69 | $82 | $107 | **FLIP** |
| DOWN_SMOOTH | long | -$3.35 | $69 | $126 | **FLIP** |
| DOWN_SMOOTH | short | +$2.93 | $119 | $68 | KEEP |
| (others wash or weak) | | | | | |

Rule: **flip NMP trades that go AGAINST the dominant regime direction.**

### 3. Validation status

| Experiment | Delta/day | 95% CI | Significant? |
|---|---:|---|---|
| IS apples-to-apples re-sim | +$59.49 | [+$22, +$97] | YES |
| IS walk-forward (70/30) | +$88.30 | [+$21, +$165] | YES |
| OOS re-simulation | +$68.10 | [-$50, +$264] | no |
| **OOS engine (apples-to-apples)** | **+$1.66** | **[-$48, +$55]** | **no** |

**Re-simulation overestimated the engine impact by ~40×** because it
ignored:
- Trade-time displacement (flipped trades hold 470+ bars, blocking entries)
- ZSeReversal exit firing on bar 1 (the bug we found and fixed — see below)
- The true OOS distribution being thinner than IS

### 4. Critical bug: ZSeReversal exit kills flipped trades

The `ZSeReversal` exit assumes the trade direction matches the FADE thesis
(entry at extreme z, exit when z reverts past 0). Flipped trades enter in
the RIDE direction — z is already on the "wrong side" at entry, so the
rule fires bar 1.

**Fix:** `ZSeReversal.evaluate` now returns None when `position.entry_tier in
{'NMP_FLIP', 'MA_ALIGN', 'NMP_RIDE'}` or `position.extras['flipped_from']`
is set.

After fix: NMP_FLIP went $0.00/trade → +$5.66/trade IS, +$1.49/trade OOS.

### 5. Per-cell continuous-feature filters OVERFIT on OOS

Within-cell EDA on each (regime × direction): 9 of 12 cells had
walk-forward-surviving top discriminators (Cohen's d 0.11-0.34).
Examples:
- DOWN_SMOOTH × long: `L1_1m_bar_range`, d=+0.344
- UP_CHOPPY × long: `L2_5m_price_velocity_9`, d=+0.241

Built `FilteredRegimeAwareReversion` — gates entry on the cell's top
feature using empirical threshold from IS train (70%). Validated inside
IS at +$X/day. **OOS result: -$19.85/day** (filter hurt). Bootstrap CI
[-$59, +$17], 40% of days hurt vs 31% helped.

**This is the 2026-05-03 OOS-overfit lesson playing out exactly.**
70/30 walk-forward inside IS is NOT enough validation; continuous-feature
quantile thresholds break on true OOS hold-out.

## Production state (2026-05-05)

- **Active strategy: `RegimeAwareReversion` (NMP_REGIME)** — base NMP entry
  + (regime × direction) flip rule, no continuous-feature filters
- ZSeReversal fix preserved
- Production thresholds: `training_v2/output/thresholds_prod.json`
- Flip cells: `{(UP_SMOOTH, short), (UP_CHOPPY, short), (DOWN_SMOOTH, long)}`

## OOS performance breakdown (REVERSION-only on 2026 OOS data)

| variant | $/day | day-WR | $/trade |
|---|---:|---:|---:|
| Base NMP (no thr, no flip) | $27.35 | 50% | $0.41 |
| + Prod thresholds | $46.07 | 49% | $0.73 |
| **+ Prod thresholds + Flip rule** | **$47.71** | **50%** | **$0.79** |
| + Filters (REJECTED) | $23.82 | 46% | $0.64 |

Threshold tuning contributes +$18.72/day (most of the gain). Flip rule
adds another +$1.64/day. Filters subtract -$23.89/day.

## Mode-tuned thresholds (2026-05-05) — structural tradeoff, not strict win

After the mode-vs-mean per-regime analysis revealed that typical-trade
regret was $44-64 (vs mean $100-160), tested mode-tuned thresholds:
`tp_pts=$5, sl_pts=$48, gb_min=$5, gb_keep=0.55, time_stop=480 bars`.

OOS apples-to-apples (NMP_REGIME flip rule, both threshold sets):

| | Production | Mode-tuned | delta |
|---|---:|---:|---:|
| OOS $/day | $48.43 | $57.78 | +$9.36 |
| OOS day-WR | 50% | 59% | +9pp |
| Median day delta | — | -$14 | (negative — typical day worse) |
| 95% CI on delta | — | [-$26, +$45] | not significant |
| Days mode beat | — | 29/67 (43%) | minority |

Pattern is bimodal: mode wins big on production's catastrophe days
(+$313 to +$520 on top 5 days where prod was -$54 to -$253) but loses
big on production's trend-trade days (-$188 to -$382 on top 5 where
prod made +$114 to +$624). Mode-tuned is **not strictly better** —
it's a different shape of returns: smoother, lower variance, lower
mean trend capture.

Verdict: keep both as options. Production for max-mean preference,
mode-tuned for max-median-robustness preference. Direction-flip lever
unchanged (still NMP_REGIME).

### Z-band anchor (rejected)

Tested per-cell continuous z-feature thresholds (within UP/DOWN
regimes the global d=0.30-0.41 suggested actionable). Walk-forward
70/30 IS split: only 1 of 12 cells survived (FLAT_SMOOTH × long, the
weakest cell ironically). Per-cell within-cohort z thresholds suffer
the same overfit pattern as previous continuous-feature filters —
the regime × direction CATEGORICAL signal exists but doesn't carry
threshold-level information. The existing flip rule is the right
granularity.

## Per-regime regret asymmetry (2026-05-05 FullRegretLabel analysis)

Stratified `regret_full.py` output by (regime × direction). The
direction asymmetry — `pct(same_extended) − pct(counter_extended)` —
quantifies how strongly each cell leans toward fade vs ride:

| cell | sm_extended % | ct_extended % | asymmetry | verdict |
|---|---:|---:|---:|---|
| UP_SMOOTH × long | 58.9% | 40.6% | +18.3% | KEEP |
| UP_SMOOTH × short | 41.6% | 57.7% | -16.1% | FLIP |
| UP_CHOPPY × long | 56.8% | 43.0% | +13.8% | mixed |
| UP_CHOPPY × short | 41.8% | 57.4% | -15.7% | FLIP |
| DOWN_SMOOTH × long | 40.1% | 59.6% | -19.5% | FLIP |
| DOWN_SMOOTH × short | 59.7% | 39.7% | +20.0% | KEEP |
| DOWN_CHOPPY × long | 45.0% | 54.4% | -9.4% | borderline FLIP |
| DOWN_CHOPPY × short | 56.1% | 43.4% | +12.7% | borderline KEEP |
| FLAT_CHOPPY × long | 49.7% | 49.7% | +0.0% | mixed (no signal) |
| FLAT_CHOPPY × short | 50.6% | 48.8% | +1.8% | mixed |
| FLAT_SMOOTH × long | 50.3% | 48.6% | +1.7% | mixed |
| FLAT_SMOOTH × short | 50.9% | 48.4% | +2.5% | mixed |

Three structural facts:
1. **Direction asymmetry exists only in UP/DOWN regimes** (FLAT cells are
   within ±2.5% of 50/50). The flip rule was right to leave FLAT alone.
2. **Production flip cells are confirmed** by independent regret-asymmetry
   analysis (the three flipped cells match the three highest-asymmetry cells).
3. **30.2% of all trades** (5,764 of 19,106) live in cells with |asym| >= 15%.
   That's the maximum realistic flip-rule lever. The other 70% need
   exit-side, filter-side, or vol-feature levers.

## Per-regime regret leak ranking

| regime | n | mean regret | early_entry_gain | capture |
|---|---:|---:|---:|---:|
| DOWN_SMOOTH | 1,989 | $159.43 | $217.25 | 2% |
| UP_CHOPPY | 1,604 | $144.96 | $200.44 | 5% |
| FLAT_CHOPPY | 7,921 | $140.54 | $189.70 | 3% |
| DOWN_CHOPPY | 1,209 | $133.17 | $182.94 | 2% |
| UP_SMOOTH | 3,023 | $123.05 | $167.50 | 3% |
| FLAT_SMOOTH | 3,360 | $98.93 | $134.34 | 2% |

Capture is 2-5% across ALL regimes — the legacy "97% of profit thrown away"
finding is uniform. FLAT_CHOPPY at 7,921 trades and $140 regret/trade
represents $1.1M of theoretical regret on this single regime — but with zero
direction asymmetry, flip-rule can't address it.

## Code artifacts

- `training_v2/strategies/regime_aware.py` — RegimeAwareReversion (production)
- `training_v2/strategies/filtered_nmp.py` — FilteredRegimeAwareReversion (rejected)
- `training_v2/tier_discovery.py` — FADE/FLIP/CHOP classification + Cohen's d
- `training_v2/full_feature_eda.py` — global feature ranking + Spearman + quintile
- `training_v2/within_cell_eda.py` — per-cell feature ranking
- `training_v2/cell_filters.py` — filter learner (output OVERFIT)
- `training_v2/flip_rule_validation.py` — re-simulation validation
- `training_v2/output/cell_filters.json` — REJECTED filters (for reference)

## Anti-patterns confirmed

1. **Re-simulation overestimates engine impact.** Engine has state-driven
   exits that fire differently than path-walking simulation. Always test
   strategy changes via the actual engine, not just regret-replay.
2. **Walk-forward inside IS is NOT a substitute for true OOS hold-out.**
   Continuous-feature quantile thresholds that survive 70/30 IS split
   still overfit when applied to a date-disjoint OOS sample.
3. **ZSeReversal-style fade-thesis exits must be tier-aware.** Any RIDE
   trade (direction-flipped, MA_ALIGN, etc.) needs to skip fade-exit rules.

## Audit: training_iso V2/ is misnamed

2026-05-05 audit found `training_iso V2/` (folder with space) is V2 in
name only — 9 of 11 .py files import from `core.features`,
`core.statistical_field_engine`, or `training.sfe_ticker` (all V1).
`nightmare_iso.py` uses 91D V1 indices. Same anti-pattern as the original
training_v2/ before the rebuild. Flagged for cleanup, not yet addressed.

`training_RM_physics_v2/` is correctly V2-pure (4 of 6 files; nn_direction.py
and __init__.py have no V2 import but no V1 import either).


## Source: project_v2_iso_pipeline.md

---
name: V2-native iso pipeline (training_iso_v2/)
description: 2026-05-05 — V2-native isolated tier pipeline with the 9 legacy ExNMP tiers ported to V2 column names + an OU-aware reversion-decay exit rule. Replaces misnamed `training_iso V2/`.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

`training_iso_v2/` (no space, clean name) is a V2-native ISOLATED tier pipeline.
Same V2 substrate as `training_v2/` (V2Ticker, V2 features by name, no V1
conversion, no caches) but each tier runs in its OWN engine with its OWN
ledger — N parallel single-strategy engines on one bar stream.

Replaces `training_iso V2/` (with space), which was misnamed: 9 of 11 .py
files imported from `core.features` / `core.statistical_field_engine` /
`training.sfe_ticker` (V1). The new folder is fully V2-pure.

## Architecture additions over training_v2

- **`iso_orchestrator.py`**: runs N engine instances in parallel on one
  ticker. Each tick is dispatched to all engines simultaneously. Each
  engine has its own ledger; one engine's open position does NOT block
  another's entry. Total $/day = sum across N tier accounts.
- **`ticker.py`**: extended V2Ticker. Loads 5m / 15m / 1h OHLCV in addition
  to 5s/1m. Provides `state.ohlcv_5m / ohlcv_15m / ohlcv_1h` as the most
  recent CLOSED higher-TF bar (lookahead-free via `searchsorted(... ts -
  period, side='right') - 1`).
- **`state.py`**: BarState extended with `ohlcv_5m`, `ohlcv_15m`, `ohlcv_1h`.
- **`wicks.py`**: pure-OHLCV directional wick math
  (`upper_wick_ratio`, `lower_wick_ratio`, `directional_wick`). Pure market
  data — V2-pure because OHLCV isn't V1-specific.

## 9 V2-native tier ports

| tier | trigger (V2-native) |
|---|---|
| FADE_CALM | NMP seed + `\|L2_1m_price_velocity_w\| < 5.0` |
| FADE_MOMENTUM | NMP seed + `\|L2_1m_price_velocity_w\| >= 5.0` |
| RIDE_CALM | NMP seed + (regime, dir) in DEFAULT_FLIP_CELLS + `\|vel_1m\| < 5.0`, FLIP direction |
| RIDE_MOMENTUM | NMP seed + flip cell + `\|vel_1m\| >= 5.0`, FLIP |
| FADE_AGAINST | NMP seed + `sign(L2_1h_price_velocity_w)` opposes fade direction |
| RIDE_AGAINST | NMP seed + 1h opposing, FLIP direction |
| KILL_SHOT | NMP seed + `directional_wick(5m)>=0.50 + directional_wick(15m)>=0.45` |
| CASCADE | NMP seed + multi-TF wick + 1h velocity ALIGNED with fade |
| FREIGHT_TRAIN | `\|vel_1m\|>=10` + `swing_noise_1m<=100` + `hurst_1m<=0.5`, FADE the velocity |

NMP seed (V2): `\|L3_1m_z_se_15\|>=1.8 + L3_1m_reversion_prob_15>=0.55`.
Direction: `z>0 -> short; z<0 -> long`.

All thresholds are V1-style defaults; expect to recalibrate against V2
unit distributions on full IS (FADE_* tiers worked on smoke; KILL_SHOT /
CASCADE / FREIGHT_TRAIN had 0 entries — wick/velocity bars in V2 may need
different cutoffs).

## OUReversionDecay exit (new in iso pipeline)

Background: NMP entry depends on `reversion_prob_w`, which is the OU
first-passage probability from `z_se` — given current OU calibration
(theta, sigma), probability the price returns to band within the window.
A high entry rprob means the decay rate was strong enough to predict
reversion.

The OU calibration UPDATES each bar. If during a trade the current rprob
DROPS materially below its entry value, the OU decay rate weakened —
the mean-reversion thesis is dying.

Rule: `OUReversionDecay(tf='1m', decay_factor=0.6)` exits when
`current_rprob <= entry_rprob × 0.6` (configurable). Skipped for RIDE
tiers (the thesis isn't reversion).

Wired through `_entry_extras` in `run_iso.py` which captures
`entry_reversion_prob` at trade open into `position.extras`.

## Smoke test (2025_06_15)

13 total trades across 9 tiers, +$325.50 (single day). FADE_CALM, FADE_MOMENTUM,
FADE_AGAINST all 100% WR (small samples). RIDE_CALM and RIDE_AGAINST losing
(expected — flip cells have +/-18% asymmetry but per-trade is mixed). KILL_SHOT,
CASCADE, FREIGHT_TRAIN had 0 entries — thresholds likely need V2-unit recalibration.

## Run commands

```
# Full IS — all 9 tiers in parallel
python -m training_iso_v2.run_iso --is

# OOS
python -m training_iso_v2.run_iso --oos

# Subset of tiers
python -m training_iso_v2.run_iso --is --tiers KILL_SHOT,CASCADE,FREIGHT_TRAIN

# With production thresholds
python -m training_iso_v2.run_iso --is \
    --thresholds training_v2/output/thresholds_prod.json

# Single day smoke
python -m training_iso_v2.run_iso --smoke
```

## Open work

- Calibrate KILL_SHOT/CASCADE wick thresholds for V2 (legacy used 0.83/0.77 V1;
  V2 wicks computed from raw OHLCV may need different cutoffs)
- Calibrate FREIGHT_TRAIN extreme_velocity (legacy 100 V1, current 10 V2 default)
- Run iso pipeline on full IS + OOS, compare per-tier $/day vs legacy 2026-04-18
  iso baseline
- Investigate OUReversionDecay impact: run with vs without, see if OU exit
  reduces round-trip losses (the 64% of losers that had peak >$20 then gave back)
- Migrate the V1 `training_iso V2/` (with space) to deprecation; this is the
  successor


## Source: project_v2_ma_alignment_directional.md

---
name: V2 MA-Alignment Directional System
description: Active research direction — multi-TF MA alignment as a directional filter. Beats fitted composites on lift; deterministic rule, no overfit risk. Pending: tradeable exit rules + joint with L-magnitude.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
# V2 MA-Alignment Directional System

**Established:** 2026-05-01.
**Status:** Best directional signal found across 5 composite framings. Operational rule defined. Next session = parallel/joint work.

## The rule

For each of 8 TFs (5s, 15s, 1m, 5m, 15m, 1h, 4h, 1D) at every 5m decision bar:

```
vote_TF = +1 if close_5m > L2_<TF>_vwap_w
         -1 if close_5m < L2_<TF>_vwap_w
          0 if within tolerance (default 0.5 ticks)
```

Sum the votes → `alignment_score` ∈ [-8, +8]. Decision:
- LONG  if `alignment_score >= +7`
- SHORT if `alignment_score <= -7`
- FLAT  otherwise

**Test-set performance** (full year, last 20% as test, 14,827 non-flat 5m bars):
- 70.5% direction accuracy on 20% of bars (+17.6% lift over 52.9% baseline)
- Combined VWAP+PriceMean `|score|>=15`: 70.6% on 19% (+17.8%)

**Why:** ship the deterministic rule. No model file, no train/test risk, walk-forward stable by construction.

## Comparison with the alternatives ruled out

| Approach | Acc | Coverage | Lift |
|---|---:|---:|---:|
| **MA align combined ≥15** | **70.6%** | **19%** | **+17.8%** |
| **MA align vwap ≥7** | **70.5%** | **20%** | **+17.6%** |
| Standalone L \|pred\|>20 | 70.4% | 45% | +10.6% |
| Single-horizon 5-voter \|w\|>5 | 62.6% | 26% | +9.8% |
| 5-voter L-aggregator \|w\|>5 | 59.8% | 13% | +9.6% |
| 2-voter (1m+5m) \|w\|>5 | 58.8% | 1.9% | +8.6% |
| Quantile composite (Q_0.25>0/Q_0.75<0) | n/a | 0% | collapsed |

MA alignment matches L on accuracy at half the coverage **but with 67% more lift over baseline** (+17.6 vs +10.6). The lift gap is the real headline — L's high accuracy is partly inflated baseline (59.8%); MA achieves 70% on a 52.9% balanced baseline.

## Why-this-works (operationalization of the user's distributional intuition)

User framing earlier: "treat price as a probability field — by the time we measure it, the electron is somewhere else". Operational read:
- Bar-to-bar direction (Analysis B) → R²=0.0003. Coin flip.
- Conditional mean of signed MFE (Analysis L) → R²=0.06-0.11, gates to 70-73%. Distributional.
- **MA alignment**: discrete approximation of "is the conditional distribution unambiguously on one side of zero". Each TF's vwap_w is a smoothed-price reference; price-vs-vwap is a sign of recent trend at that timescale. When 7+ of 8 TFs agree, the conditional distribution is concentrated on one side.

The autoregressive features (vwap_w, price_mean_w) carry signal as **direct comparators** (price - vwap_w) but NOT as regression inputs (where they trivially correlate with current price level and don't help direction). The user's counter-proposal correctly identified this distinction.

## TFs that carry the signal

Single-feature accuracy (just one TF's vwap comparison):

| TF | VWAP solo | Lift | Smoothing window |
|---|---:|---:|---|
| 5s | 53.5% | +0.6% | 9 bars (45s) |
| 15s | 53.7% | +0.8% | 12 bars (3 min) |
| 1m | 55.8% | +2.9% | 15 bars (15 min) |
| 5m | 59.0% | +6.2% | 9 bars (45 min) |
| **15m** | **61.7%** | **+8.9%** | **12 bars (3 hours)** |
| **1h** | **61.6%** | **+8.7%** | **12 bars (12 hours)** |
| 4h | 55.2% | +2.3% | 18 bars (3 days) |
| 1D | 54.2% | +1.4% | 5 bars (5 days) |

**The 3-hour-to-12-hour smoothing window is where the signal lives.** Shorter (5s-1m) is too noisy; longer (4h-1D) is too coarse for a 5m decision cadence.

## Where the data lives

- Tool: `tools/v2_composite_ma_alignment.py`
- Outputs: `reports/findings/v2_composite_ma_alignment/`
  - `per_tf_signals.csv` — per-TF accuracy for vwap and mean comparators
  - `vwap_alignment.csv` — score thresholds 1-8 for vwap-only
  - `mean_alignment.csv` — score thresholds 1-8 for price_mean-only
  - `combined_alignment.csv` — score thresholds 1-16 for vwap+mean combined
  - `summary.md` — narrative
- Commit: `7dae2585`
- Source feature schema: `DATA/ATLAS/FEATURES_5s_v2/L2_<TF>/` (vwap_w + price_mean_w columns)

## Trend-direction regression (2026-05-02)

`tools/v2_regress_trend_direction.py` — predicts day net_move from 5m bar features. Day-level 60/20/20 split via `regime_labels_2d.csv`.

| Metric | Value |
|---|---:|
| Test R² | 0.032 |
| Bar-level direction acc | 63.2% (+11.9% lift) |
| Day-level direction acc | **74.6%** (+22.5% lift, 71 days) ← new high water mark |

**Per-regime (validates regime-conditional hypothesis):**

| Regime | Bar acc |
|---|---:|
| UP_SMOOTH | **87.6%** |
| UP_CHOPPY | 74.5% |
| DOWN_SMOOTH | 72.7% |
| FLAT_CHOPPY | 58.1% |
| DOWN_CHOPPY | **43.9%** ← model misses these |
| FLAT_SMOOTH | **42.3%** ← noise floor |

The model implicitly classifies "is this a trending day?" — succeeds 73-88% on trend cells, fails 42-58% on chop/flat. Magnitude is conservatively biased (mean_pred ~ 0.5 × mean_actual on trend cells). This is the empirical proof for regime-conditional strategy selection.

## Next session — parallel and then joint

### Parallel track 1 — MA alignment exit rules
The 70.5% direction accuracy is on the entry signal only. Open questions:
- With this filter, what's the average MFE/MAE of qualifying trades?
- What stop and target sizing makes the alignment-filtered signal tradeable?
- Does the alignment HOLD across the trade window, or does it flip mid-trade?
- If alignment flips → exit signal? Or does flipping during a winning trade mean "let it run"?

Tools to build/extend:
- Modify `v2_composite_ma_alignment.py` to also report MFE/MAE distribution per signal (alignment_score → outcome stats)
- Test alignment-flip-as-exit vs fixed-stop exits

### Parallel track 1.5 — 6-class regime classifier (NEW, from net_move result)
The trend-direction regression at 74.6% day-level acc shows the model IS implicitly classifying regimes (87.6% UP_SMOOTH, 42% FLAT_SMOOTH). Make this explicit:
- Train a 6-class classifier on `regime_2d` from bar features
- Compare confusion matrix vs the regression's per-regime accuracy
- Use as the regime gate in the joint router

### Parallel track 2 — L-model refinement
Even with MA alignment dominating on lift, the L approach has higher coverage (45% vs 20%). Options to push L further:
- Drop the autoregressive features (vwap_w, price_mean_w) from the regression inputs entirely. They're trivially correlated with price; removing forces the model onto genuinely predictive features (z_se, hurst, swing_noise, vol_accel_w, etc.).
- Feature selection on the 185D v2 schema (Lasso, mutual info, etc.).
- Replace OLS with a richer base learner (GBM, MDN) — but watch for overfit at low sample counts.

### Joint (after both parallel tracks)
Combine MA alignment FILTER (when can we trade?) with a magnitude estimator (how big is the expected move?). Sketch:
- MA alignment says LONG → only consider long trades
- L (or refined model) gives expected MFE magnitude → use for position sizing or skip if magnitude too small
- Pair filter strictness with magnitude threshold for tradeable subset

This converges the two best signals into one system. The MA filter handles direction with high confidence; the magnitude estimator sets risk/sizing.

## Connection to prior regime work (2026-05-01 update)

**MA alignment IS a regime classifier.** When `alignment_score >= 7`, the day is by definition in a strong-trend regime (UP_SMOOTH or DOWN_SMOOTH per the 2D taxonomy). When alignment is mixed, you're in chop / transitional.

This connects to two prior threads:
- [feedback_chop_edge_regime_filter.md](feedback_chop_edge_regime_filter.md) — the zigzag counter-trend strategy WINS on chop (+$89/day) and LOSES on trend (-$95/day). Two-feature classifier (`prior_range`, `range_compression`) discriminates with d_OOS=0.77/0.78. Filter rule turns -$552 into +$5,000-6,000.
- `tools/atlas_regime_labeler.py` (2026-04-29) — labels all 348 ATLAS days as UP/DOWN/CHOP/QUIET/TRANSITIONAL. Output: `DATA/ATLAS/regime_labels.csv`.

**The right joint system is regime-conditional strategy selection:**
- HIGH alignment → trend-following (today's MA-align direction)
- LOW alignment + chop conditions → zigzag counter-trend (prior bleed_score filter)
- TRANSITIONAL / mixed → skip

## 2D label system (2026-05-01)

Built `tools/atlas_regime_labeler_2d.py` — extends the existing daily labels with:
- `direction_axis` ∈ {UP, DOWN, FLAT}
- `variation_axis` ∈ {SMOOTH, CHOPPY}
- `regime_2d` = combined (e.g., "UP_SMOOTH")
- `split` ∈ {IS, VAL, OOS} (60/20/20 by date)

Output: `DATA/ATLAS/regime_labels_2d.csv` (348 days). Distribution:

| Regime | Total | IS | VAL | OOS |
|---|---:|---:|---:|---:|
| UP_SMOOTH | 63 | 34 | 17 | 12 |
| UP_CHOPPY | 23 | 15 | 4 | 4 |
| DOWN_SMOOTH | 44 | 25 | 9 | 10 |
| DOWN_CHOPPY | 19 | 12 | 2 | 5 |
| FLAT_SMOOTH | 71 | 51 | 13 | 7 |
| FLAT_CHOPPY | 128 | 71 | 24 | 33 |

UP_CHOPPY and DOWN_CHOPPY have thin OOS samples (4-5 days) — caveat for stat-sig analysis on those cells.

Loader:
```python
from tools.atlas_regime_labeler_2d import load_regime_labels
df = load_regime_labels()
oos_up_smooth = df[(df.split == 'OOS') & (df.regime_2d == 'UP_SMOOTH')]
```

Use this as the substrate for ALL future regime-conditional analysis — MA alignment, L-model, zigzag, any composite all evaluated through the same lens.

## Anti-patterns ruled out (do NOT revisit unless new info)

- **Naive cross-TF L voting at common cadence** → fails because horizons don't align (1m predicts 1h ahead, 1h predicts 8h ahead).
- **Apples-to-apples single-horizon refit with handicapped voters (each voter sees only its TF's 23 features)** → still loses to standalone L using all 185 features. Voting handicapped models can't recover what one full model exploits.
- **Strict quantile composite (Q_0.25>0 / Q_0.75<0)** → too stringent under quick-mode GBM (50 trees), collapses to 100% FLAT. Would need either 200+ trees or a softer rule.
- **Stripping autoregressive features from L's inputs** → was Claude's premature suggestion. User's better counter-proposal (use them as alignment signals, not regression inputs) was correct. The features ARE useful when used the right way.

## Single-rule one-liner (for NT8)

```
LONG  if (Close > VWAP_15s_w AND Close > VWAP_1m_w AND ... AND Close > VWAP_1D_w with at least 7 of 8 true)
SHORT if (Close < VWAP_<TF>_w with at least 7 of 8 true)
FLAT  otherwise
```

Approx 20 lines in NT8. Each TF's vwap_w is computed in `core_v2.StatisticalFieldEngine.compute_L2(df, tf)` — the formula is a rolling N-period VWAP per `core_v2.N_BASE` (5s=9, 15s=12, 1m=15, 5m=9, 15m=12, 1h=12, 4h=18, 1D=5).


## Source: project_v2_native_training.md

---
name: V2-native training pipeline (training_v2/)
description: Clean-slate V2-native training built 2026-05-04 — reads core_v2.features directly, no V1 conversion shims. Production state, components, and key files.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

`training_v2/` is the V2-native training pipeline rebuilt from scratch on
2026-05-04. The previous `training_v2/` (V1-shape engine reading a V2-derived
compat cache) was archived to `training_v2_archive/` to preserve the +$78/OOS
calibration work.

**Why:** The user pushed back hard on the V1-shape engine + compat cache. The
new training_v2/ consumes V2 layered features (185D = L0 + 8 TFs × 23 from
`core_v2.features.load_features()`) directly. No conversion. No `v1_compat`.
No 91D V1 vector. No `_1M_OFFSET` indices.

## Architecture

```
training_v2/
├── ticker.py              V2Ticker / MultiDayV2Ticker — yields BarState per 5s bar
├── state.py               BarState dataclass + REGIME_VOCAB
├── v2_cols.py             canonical V2 column-name helpers (z_se_w, vwap_w, body...)
├── ledger.py              Position + ClosedTrade dataclasses
├── engine.py              first-signal-wins entry, first-match exit, threshold injection
├── exits.py               HardStop, TakeProfit, Giveback, ZSeReversal, SwingNoiseSpike,
│                          RegimeFlip, TimeStop. Per-tier thresholds via position.extras.
├── regime_router.py       day-level eligibility (placeholder, all-allowed default)
├── strategies/
│   ├── base.py            Strategy ABC + EntrySignal
│   ├── ma_align.py        MAAlignTrendFollow (7-of-8 vwap_w alignment, 5m close)
│   ├── reversion.py       ReversionFromExtreme (V2 NMP: |z_se_w|≥1.8 + reversion_prob≥0.55)
│   └── velocity_body.py   VelocityBodyChord — KILLED (lottery-day artifact)
├── regret.py              replays each trade's price path → peak/MAE/capture/optimal-exit labels
├── bayesian_table.py      hierarchical posteriors per (regime, tier) — for inspection
├── threshold_optimizer.py grid-search exit thresholds (legacy approach — use sparingly)
├── threshold_bayesian.py  PRINCIPLED — derives thresholds from regret distributions via
│                          formulas (q_tp, q_sl, ttp_factor knobs). Preferred over grid search.
├── tier_discovery.py      flip-signal discovery (FADE_BETTER vs FLIP_BETTER classification + Cohen's d)
├── cnn/                   V2DirectionCNN (8×23 grid + L0 + regime embed → 3-class softmax)
└── run.py                 CLI: --is/--oos --strategies --thresholds --cnn ...
```

## Key principles

1. **No V1 dependencies.** Engine reads V2 columns by canonical name (e.g.,
   `state.get('L3_1m_z_se_15')` via `v2_cols.z_se_w('1m')`). No `_1M_OFFSET`
   indexing.
2. **First signal wins** (user's spec). Strategies evaluated in declaration
   order; first non-None EntrySignal opens the trade.
3. **Per-tier thresholds via `position.extras['thresholds']`.** Engine looks
   up `(regime, tier)` at trade open and copies the threshold dict into the
   position's extras. Exit rules read from there at runtime.
4. **Bayesian thresholds derived, not searched.** `threshold_bayesian.py`
   computes thresholds as quantile/mean functions of cell-grouped regret
   distributions. No grid search overfitting.

## Production state (2026-05-04)

**Strategies (active):**
- MA_ALIGN — 7-of-8 vwap_w alignment, fires on 5m close
- REVERSION — V2-native NMP, fires on 1m close

**Strategies (killed):**
- VEL_BODY_CHORD — see `feedback_outlier_day_optimizer.md`. Negative on 67/68
  OOS days, positive only on 2026-03-20 ($1,333/contract range day).

**Production thresholds:** `training_v2/output/thresholds_prod.json`
(per-tier Bayesian-derived; see `threshold_bayesian.py --group-by tier`).

| tier | TP | SL (capped) | giveback arms | gb_keep | max hold |
|---|---:|---:|---:|---:|---:|
| MA_ALIGN | +$26 | -$50 | $41 | 30% | 41.7m |
| REVERSION | +$27 | -$50 | $41 | 39% | 39.8m |

**Last measured OOS performance** (68 OOS days, MA+REV only, prod thresholds):
- $54.82/day (+$3,728 total)
- Day-WR 57% (vs 51% baseline)
- 95% CI on delta vs baseline: [-$5.29, +$62.89] — NOT statistically significant

## Pipeline (full run sequence)

```
# 1. Generate IS trades (no CNN, no thresholds — baseline)
python -m training_v2.run --is --strategies MA_ALIGN,REVERSION

# 2. Build regret labels
python -m training_v2.regret --trades training_v2/output/is.pkl \
    --out training_v2/output/regret_is.pkl

# 3. Derive Bayesian per-tier exit thresholds
python -m training_v2.threshold_bayesian --regret training_v2/output/regret_is.pkl \
    --out training_v2/output/thresholds_prod.json --group-by tier

# 4. Build Bayesian table (inspection only)
python -m training_v2.bayesian_table --regret training_v2/output/regret_is.pkl \
    --out training_v2/output/bayesian_is.pkl

# 5. Train V2 direction CNN (deferred until used as filter+entry)
python -m training_v2.cnn.train

# 6. Forward pass IS with thresholds + CNN
python -m training_v2.run --is --strategies MA_ALIGN,REVERSION \
    --thresholds training_v2/output/thresholds_prod.json \
    --cnn training_v2/output/cnn/direction_cnn.pt

# 7. OOS validation
python -m training_v2.run --oos --strategies MA_ALIGN,REVERSION \
    --thresholds training_v2/output/thresholds_prod.json \
    --cnn training_v2/output/cnn/direction_cnn.pt
```

## Open questions / known gaps

1. **Directional wicks not in entry feature vector.** They CAN be computed
   per-bar from 1m/5m/15m OHLCV (legacy `directional_wicks_batch` math from
   `core_v2/v1_compat.py` is still valid — pure OHLCV math, not V1-specific).
   Currently the entry_v2 vector is the 185D V2 layered features only. If we
   want to use directional wicks as discriminators, we need to add them.

2. **No sub-tier discrimination.** REVERSION fires for any `|z_se_w|≥1.8`
   entry — no further sub-classification (legacy had 9 sub-tiers under NMP).
   See `project_9tier_discovery_v2.md` for what we tried and what failed.

3. **CNN not yet trained.** The V2DirectionCNN (8×23 grid + L0 + regime
   embed → 3-class softmax) is built but not trained. Expected to push past
   the +$55/day OOS ceiling once trained.

## Why we trust this architecture

- VEL_BODY_CHORD failure was *detected* by the bootstrap CI doom-cascade
  check, not silently shipped (anti-doom rule worked)
- Threshold-tuning ceiling is honestly reported with CIs, not inflated
- Median-day vs total objective comparison made the IS-overfit pattern
  concrete (see `feedback_outlier_day_optimizer.md`)
- Cell-granularity sweep (regime / tier / tier × regime) showed all three
  give within $2/day of each other — robust framework


## Source: project_weekend_recalibration.md

---
name: weekend recalibration loop
description: Future concept — weekly oracle recalibration from live trades (export → oracle → retune → deploy Monday)
type: project
---

Weekend recalibration loop (NOT yet implemented — future roadmap item):
1. Friday close: export week's live trades via nt8_to_parquet
2. Saturday: run oracle on completed week (forward-looking labels now available)
3. Oracle reveals: which templates worked, direction accuracy, exit quality
4. Recalibrate: brain updates, gate thresholds, giveback/trail params
5. Monday open: deploy recalibrated system

**Why:** Walk-forward validation applied to live. Train on days 1→N-1, trade day N.
Weekly cycle gives oracle access to completed data without lookahead.

**How to apply:** Don't build this until live trading is stable. Prerequisite:
unified BarProcessor (OOS = live code), stable quantum scoring, proven OOS edge.
When ready, create `tools/weekend_recalibration.py` pipeline script.


## Source: project_zigzag_calibration.md

---
name: project-zigzag-calibration
description: "Python zigzag pipeline calibration — ATR×4 dynamic R, 1m ATR source, 5s pivot detection. NT8 strategy must match these to produce comparable trades."
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

The Python zigzag pipeline that produced the `composite_forward_pass_hardened.csv` baseline ($475/day flat, 87% Day WR) uses these EXACT calibration parameters. The NT8 strategy must match them or NT8 fills will not be comparable to Python sim.

Source of truth: `tools/build_zigzag_pivot_dataset.py` + `tools/_viz/auto_swing_marker.py:detect_swings`

**Calibration**:
- ATR period: **14**
- ATR source: **1m bars** (NOT the pivot-detection series)
- ATR multiplier: **4.0**  ← "we made it dynamic" — was static, now ATR×4
- Pivot detection bars: **5s closes**
- min_reversal_ticks = `max(4, round(atr_pts / tick_size * 4))`  → in points = `max(1.0, atr_pts × 4)`
- min_bars between pivots: **36** (= 3 minutes at 5s)
- Tick size: 0.25, dollar/point: $2 (MNQ)

**Why:** Why: User visual-calibrated this on 14-month MNQ data; ATR×4 produces zigzags that match real swings rather than noise. Static R values (e.g. R=30) produce different pivot counts and trade timestamps; not comparable.

**How to apply:** When recommending NT8 strategy params for parity testing against Python sim, ALWAYS:
1. Verify the NT8 strategy can compute ATR on a 1m series and detect pivots on a 5s series independently
2. Set ATR period 14, multiplier 4.0, MinR floor 1.0pt (= 4 ticks)
3. Set pivot TF to 5s if the strategy supports it
4. If the NT8 strategy lacks separate ATR/pivot timeframes (single-series ATR), there is a structural mismatch — flag it, propose a fix (use ZigZagATR.cs indicator as input), do NOT recommend static R or single-TF dynamic R as "raw"

**Current NT8 state**:
- `docs/nt8/ZigZagATR.cs` indicator has correct architecture (AtrTfMinutes=1, AtrPeriod=14, AtrMult=4.0, ZigZagTfSeconds=5, MinBars=36)
- `docs/nt8/ZigzagRunner_v1.0.cs` (deployed as `ZigzagRunner.cs`) uses STATIC R=30 → does not match Python
- `docs/nt8/ZigzagRunner_v1.0.8-RC.cs` has UseDynamicR but defaults wrong (AtrLookbackBars=60, AtrMultiplier=5.0, MinRPoints=5.0) AND ATR is computed on the pivot series (not a separate 1m series) → does not match Python

**The gap**: no strategy currently matches Python. Either patch v1.0.8-RC to add a separate 1m ATR series and fix defaults, OR build a new strategy that consumes ZigZagATR.cs indicator as the pivot source. The latter is cleaner and is the foundation for the hybrid build (see [[user_collaboration_protocol]]).

Related: [[feedback_no_human_regime_terms]] (translate metaphors), [[project_v2_iso_pipeline]] (V2 pipeline calibration).


## Source: research_liquidation_anchoring.md

---
name: Liquidation Pool Anchoring — The Linchpin Discovery
description: The three-body model works BECAUSE liquidation pools are the real gravitational bodies, not statistical regression. This is the missing piece that makes the physics predictive, not just descriptive. DO NOT LOSE THIS.
type: project
---

## THE INSIGHT (validated by Opus)

The three-body quantum physics model isn't pseudoscience — it's predictive
WHEN anchored to real market structure instead of statistical abstractions.

### The Problem with Current Anchoring

The regression mean (μ) FOLLOWS price. It's derived from price history.
Using it as the gravitational center means the "gravity" is chasing the
particle instead of attracting it. This is why the system works but doesn't
predict — it describes where price HAS BEEN, not where it WILL GO.

### The Solution: Liquidation Pools as Gravitational Bodies

Liquidation pools are WHERE stops cluster. They represent REAL concentrated
order flow that price is attracted to. They are the actual gravitational
bodies in the three-body model:

| Physics Model | Statistical Anchor (current) | Liquidation Anchor (proposed) |
|---------------|----------------------------|------------------------------|
| Center of mass (μ) | Regression mean (follows price) | Major liquidation pool (price follows IT) |
| Upper singularity (+2σ) | Statistical band (arbitrary) | Nearest long liquidation cluster |
| Lower singularity (-2σ) | Statistical band (arbitrary) | Nearest short liquidation cluster |

### Why This Maps to Real Physics

- **Stops cluster at levels** = mass concentration (gravitational body)
- **Price pulled toward levels** = gravitational attraction (F = -θ·z·σ)
- **Bounce off levels** = repulsive force (1/r³ — the Roche limit)
- **Break through levels** = escape velocity (resonance cascade)
- **Level absorbed (stops liquidated)** = gravitational body consumed → new orbit

### Why This is Predictive

Statistical regression tells you: "price is 2σ from the mean" → so what?
Liquidation anchoring tells you: "price is approaching a $50M stop cluster
at 25,100 — gravitational pull is real, the money IS there"

The SAME z-score has different meaning depending on whether it's approaching
a liquidation pool (predictive) or just far from a regression line (descriptive).

### Validation Path

**Phase 1** (manual): Mark 8-10 liquidation levels manually per session.
Anchor the three-body model to them. Backtest: does WR improve?

**Phase 2** (CNN detection): Train CNN to detect liquidation levels from
order flow / volume profile / historical stop-out patterns.

**Phase 3** (level-aware patterns): Pattern templates contextualized by
proximity to liquidation levels. Same template + near level = different
P(success) than same template + no level.

### Expected Impact

- Current WR with statistical anchoring: 67.3%
- Opus estimate with level anchoring: **70-75% WR**
- The improvement comes from the CENTER being predictive (price goes TO it)
  instead of descriptive (price came FROM it)

### Data Sources for Liquidation Levels

- **Volume profile**: high-volume nodes = institutional levels
- **Open interest changes**: where new positions are being built
- **Historical stop-out zones**: where price reversed sharply (stops triggered)
- **Round numbers**: psychological levels where retail clusters stops
- **CME settlement data**: daily settlement prices act as magnets

### Integration with Existing System

The three-body forces don't change — F_gravity, F_repulsion, F_momentum
use the same formulas. Only the ANCHOR POINTS change:

```python
# Current: statistical anchoring
center = regression_mean(prices, window=21)
upper_sing = center + 2 * regression_sigma
lower_sing = center - 2 * regression_sigma

# Proposed: liquidation anchoring
center = nearest_major_liquidation_pool(current_price)
upper_sing = nearest_long_liquidation_cluster(current_price)
lower_sing = nearest_short_liquidation_cluster(current_price)
```

Everything downstream (z-score, forces, wave function, entropy, tunnel
probability) computes identically but now measures distance from REAL
structure instead of statistical abstractions.

### Status

- **Validated by**: Opus (confirmed as "the missing piece")
- **Implementation**: Not started — research line for future session
- **Priority**: HIGH — this is the linchpin that makes the physics model
  go from descriptive (67% WR) to predictive (75%+ WR)
- **Dependency**: Needs liquidation level data source (manual first, CNN later)


## Source: research_pid_trance.md

---
name: PID Trance at 1-sigma — HFT control loop traps price
description: At 1σ from regression mean, HFT algos maintain a PID control loop that traps price in a trance. Signals at 1σ are the control system maintaining equilibrium, not tradeable setups. Real signals are at 2σ+ where PID loses control.
type: project
---

## The Observation (Moises)

At 1σ distance from the regression mean, price gets caught in a **PID trance** —
the HFT market-making algorithms are actively controlling price within this band.

- **Kp (Proportional)**: Price at +1σ → algos sell proportionally → price reverts
- **Ki (Integral)**: Price stays at +0.5σ too long → accumulated error → algos push harder
- **Kd (Derivative)**: Price moves quickly toward 1σ → algos dampen the velocity (wicks)

The result: oscillation within ±1σ that looks like patterns but is actually the
control system maintaining equilibrium. Trading it = trading against the demi-gods.

## Connection to Scoring

If `term_pid` is high (strong PID control), signals at 1-2σ should be penalized
in P(success). If `term_pid` is low/zero (PID absent), signals at 1-2σ might be
early cascade indicators (demi-gods stepped away).


## Source: research_pre_entry_pace.md

---
name: R3 Pre-Entry Pace Filter
description: Research line — use pace/momentum estimate at entry time to filter noise trades before committing capital
type: project
---

## R3: Pre-Entry Pace Filter

**Discovered**: 2026-03-11, from trade anatomy analysis

### Problem
43% of IS trades are counter-trend scalps (43.0%) or noise (10.8%). The system profits
on many of them because giveback catches micro-spikes, but the entries were wrong —
we're trading statistical noise and getting bailed out by fast exits.

Example: TID=55 LONG trade had oracle_mfe=3.5 ticks ($0.875 real move), but the system
saw a 44-tick spike, captured $13 via giveback, and looked like a winner. In truth the
entry should never have happened — the oracle saw no real move.

### Insight
Pace (tick_progress / time_progress) currently only runs during a position. But the
same concept could gate entries:

- **Pre-entry momentum**: Is current price movement consistent with the template's
  expected MFE trajectory? A template that expects +50 ticks over 8 minutes should
  show early momentum in the right direction at entry time.
- **Velocity alignment**: If `particle_velocity` is flat or adverse at entry, the
  template match is probably noise — the statistical pattern matched but the market
  isn't actually moving.
- **Noise spike detection**: A 44-tick spike in 30 seconds on a template that expects
  8 minutes to peak is a noise spike, not the pattern developing. The pace ratio
  (tick_progress >> time_progress) could flag this as "too fast to be real."

### Implementation Sketch
At entry time, before committing:
```python
# Already available at entry:
velocity = state.particle_velocity  # dp/dt
F_net = state.F_net                 # d²p/dt²
template_mfe = lib_entry['p75_mfe_ticks']
template_resolve = lib_entry['avg_mfe_bar'] * 4  # in 15s bars

# Expected pace: template_mfe / template_resolve = ticks/bar expected
expected_pace = template_mfe / max(1, template_resolve)

# Current pace proxy: velocity (ticks/bar already)
# If velocity << expected_pace and no acceleration, skip
if abs(velocity) < expected_pace * 0.3 and sign(F_net) != sign(velocity):
    skip("pre-entry pace too low")
```

### Scope
- Gate in `execution_engine.py` direction cascade or as new gate after conviction
- Needs: velocity, F_net (already in MarketState), template MFE stats (already in lib_entry)
- Risk: might filter real slow-developing trades that eventually peak
- Validate: compare oracle_mfe distribution of filtered vs passed trades

### Priority
After R1 (TF-bucketed clustering) and 4x timeframe fix validation. This is an
entry-side improvement — separate from exit tuning.


## Source: research_tcn_v5.md

---
name: research_tcn_v5
description: TCN architecture for AdvanceEngine v5 — multi-TF learned pattern recognition replacing K-means
type: project
---

## TCN / Dilated 1D CNN — AdvanceEngine V5

**Why:** K-means (v2) manually flattens 70D features. TCN learns which TF relationships matter.

**Architecture:**
- Input: (10 TFs, 7 features) = (10, 7) matrix — NOT flattened
- Conv1D across TF axis with dilations (1, 2, 4) — sees adjacent to full TF range
- Flatten → Dense → sigmoid → LONG/SHORT
- Each dilation level discovers cross-TF patterns (1m+3m, 1m+1h, 1s+1W)

**Key design decisions:**
- Keep TFs separated as raw base features, let TCN figure out relationships
- Don't pre-compute 70D — that's manual feature engineering the TCN should learn
- Hurst is low (0.004), market isn't truly fractal — it's regime-switching with persistence
- TCN handles regime-switching naturally via dilated receptive fields
- If some TFs are noise (2m, 3m), TCN learns zero weights — self-pruning

**Layers vs Lookback distinction:**
- Layers = computation depth (how many feature combinations)
- Lookback = how many bars of history
- Higher TFs ENCODE lookback (1h DMI = 60 bars of context)
- So lookback = 1 bar of (10, 7) snapshot — TCN convolves across TF axis, not time axis
- This makes it closer to an MLP with structured input than a traditional temporal CNN

**Greedy layer training:**
- Start 1 layer, measure OOS, add layer, measure, stop when no gain
- Each layer adds a level of TF interaction discovery
- Feature tree says 3 levels max → expect 3 layers optimal

**Training:**
- PyTorch, train on IS (6 months), validate on OOS (8 months) — fixed ratio
- Custom loss: maximize PnL, not accuracy
- ONNX export for inference

**Hardware:** RTX 3060 12GB (training), GTX 1060 3GB (inference if needed)

**Prerequisites (must complete first):**
1. K-means v2 baseline (AdvanceEngine V2) — establishes the number to beat
2. 70D feature extraction validated on OOS
3. IS/OOS ratio fixed (6/8 months)

**Reference:** `examples/Overview.md` — external research on TCN vs LSTM vs Transformer for multi-scale signals


## Source: research_telescoping_tf.md

---
name: Telescoping TF entry scope
description: Research line - use 1m for entry primitive decisions, 15s/5s/1s as price ticker for exits, macro TFs via 192D context
type: project
---

Entry primitive TF architecture — telescoping scope concept:

1. **Macro scan** (1h/15m/5m): ZigZag identifies developing structure. Captured via 192D context vector.
2. **Setup recognition** (1m): 10-bar lookback matches entry primitive. 1m has enough bars for real swing geometry.
3. **Entry confirmation** (15s): micro price action confirms the 1m setup is playing out.
4. **Exit management** (15s/5s/1s): fast TFs as price ticker for giveback/envelope.

**Why:** 10 bars at 15s = 2.5 min (too short for setup recognition). 10 bars at 1m = 10 min (meaningful behavioral context). The 192D context already carries multi-TF info, so macro structure doesn't need to be in the lookback geometry.

**Implication:** Entry primitives could cluster exclusively at 1m (single pool, not per-TF), with 192D context differentiating macro setups. This would produce denser, more meaningful clusters.

**Status:** Research line — deferred. Current implementation uses per-TF entry clustering. Revisit after first full build + validation.

**How to apply:** When validating entry primitives, check if per-TF clusters at 15s/30s are meaningful or just noise. If they collapse, this is the fix.


## Source: waveform_research.md

# Waveform Screening & Seed Library Research

> Referenced from MEMORY.md. Full journal: `docs/reference/RESEARCH_JOURNAL.txt`

## Waveform Screening (active research)
- **Journal**: `docs/reference/RESEARCH_JOURNAL.txt` — full methodology + insights
- **Key insight**: segment on price/time FIRST, layer 16D physics on top
- Price I-MR: I=close, MR=signed bar-to-bar change, regimes from UCL breaks
- Default mode = price I-MR only; `--full` = adds 16D fractal pipeline
- Signed MR: every change is a potential pattern, sign flips = direction changes
- **Analyses completed**: A-H (screening), I (seed classification, 20 primitives),
  J (adaptive R² sub-types, IQR quality gate, R² ceiling 0.88),
  K (direction prediction: 70.6% accuracy, +16.5% lift, 4h/1h dominant),
  L-P (various), Q (signed magnitude histogram + paired 192D profiles)
- **Key K findings**: 1h_hurst #1, 4h_osc_coh #2, 15m base TF barely top-10
- **Q status**: Sign-first split working. Adaptive sigma + IQR fallback added.
  Need `--analysis-days 120` to get enough samples (default 7d only gives ~460 bars).
  CLI: `--start X` skips to analysis X, `--cache file.npz` saves/loads feature matrix
- **Integration spec**: `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (5 parts)
  Part 1: seed_library.py (20 shapes), Part 2: 4h worker, Part 3: shape direction P0.5,
  Part 4: GradientBoosting 176D direction model, Part 5: live engine wiring

## Seed Library & Live Worker Architecture (KEY DECISION)
- **The waveform analysis is OFFLINE research** — too slow for live trading
- **Output = a pre-built SEED LIBRARY**: shape templates (mathematical functions
  with fitted parameters) + price models, serialized as a lookup table
- **Workers receive the library at startup**, NOT the raw analysis code
- **Live workflow**: observe N bars -> delta from entry -> match to seed library
  (closest function fit) -> get shape type + predicted direction + magnitude
- **This replaces templates.pkl**: instead of DBSCAN clusters, workers get
  named mathematical shapes (V-reversal, ramp, sigmoid, etc.) with parameters
- **Price model (92% R²) enriches** the shape match for entry precision
- The seed library is the bridge between offline research and live execution

## Shape-First Design Direction
- **Pipeline**: DMI pre-split -> I-MR(DMI diff) -> DBSCAN(vol+ADX), 2D clustering, 16D identity
- **Shape-first**: seed functions (ramp, V, sigmoid) replace DBSCAN clusters
- **Laplacian = shape identifier**: d²p/dt² discriminates shape types


## Source: tier_building_playbook.md

# Tier Building Playbook — Consolidated Methodology

**Last updated: 2026-04-18.** This is the working methodology for building,
fixing, and iterating trading tiers in the Bayesian-AI isolated pipeline.
Supersedes `feedback_tier_three_questions.md` (now a subsection here).

The playbook is organized by the order you use it: data integrity first,
then EDA, then entries, then exits, then the anti-patterns that burn time.

---

## 1. Data Integrity Checklist (do this before anything)

Before running any EDA, verify data is honest. **Most "edge" in backtests
is bugs.** The 2026-04-17 lookahead fix alone turned +$740/day into
-$164/day. Every analysis below assumes these checks pass:

- **Lookahead bias**: `searchsorted(timestamps, target_ts, side='right') - 1`
  must subtract the period before lookup for higher-TF aggregation.
  Anything that peeks at in-progress bars contaminates features.
- **Clean price data**: phantom spikes in NT8 data cost $4,350 of fake
  edge on one analysis (2026-04-03). Use Databento for IS, NT8 only for
  OOS-2 live parity. Validate: max price jump, contract continuity, zero-volume gaps.
- **Feature parity**: training features must match live-engine features
  cell-by-cell (target: 100% parity). On 2026-04-14 we moved from 7.5%
  parity to 100% by switching to `LiveFeatureEngine`. Check
  `training/live_feature_engine.py`.
- **SFE cache**: cache key must include `(valid_idx, latest_bar_ts)` not
  just `valid_idx` (2026-04-16 frozen features bug — collision after
  5000-bar trim left stale state since mid-Feb).
- **Regret LOOKAHEAD cap**: `training_iso/regret.py` defaults to 6h
  counterfactual — use 10min-before-entry / 30min-after-exit bounds.

When in doubt, run `tools/validate_data.py` before touching a tier.

---

## 2. The Three-Question Method (core tier-fixing loop)

Given an iso trade pickle (`training_iso/output/trades/iso_is.pkl`) tagged
with `entry_tier`, ask three questions in order. Each produces at most one
rule. Measure between rules — don't stack blindly.

### Q1 — Are the entries right?

**Method:** peak-bucket analysis. Split tier trades by `peak` (the max
PnL reached during the trade, in $):
- `peak <= 0`: direction wrong from bar 1
- `0 < peak <= 5`: barely worked
- `5 < peak <= 20`: partial success
- `peak > 20`: direction solidly right

**Decision tree:**
- **>80% peak > $20**: direction is correct. Move to Q3.
- **<50%**: direction is inverted. **Flip direction** in the tier's
  fire function (`return 'short' if z > 0 else 'long'` becomes the
  opposite). Rerun single-tier isolated pipeline. THEN move to Q3.
- **50-80%**: mixed. Don't flip yet. Move to Q3; reconsider later.

**Observed examples:**
| Tier | % peak > $20 | Decision |
|---|---:|---|
| TREND_FOLLOWER | 84.8% | Solid — kept direction |
| RIDE_AGAINST (ride dir) | 66.5% | Mixed; later flip still helped |
| KILL_SHOT | 62.4% | Mixed — kept |
| MTF_BREAKOUT | 66.9% | Mixed |
| NMP_FADE | 52.7% | Noise (catch-all) |

### Q2 — Hold-time cliff (natural timescale)

**Method:** bucket trades by held minutes, report WR and $/trade per
bucket. Look for the boundary where buckets flip sign.

**Observed natural timescales:**
| Tier | Cliff | Interpretation |
|---|---:|---|
| TREND_FOLLOWER | 60 min | Fade thesis timescale |
| RIDE_AGAINST | 15 min | 1h-vel reversal plays out fast or not at all |
| KILL_SHOT | 30 min | Wick-rejection intermediate |
| MTF_BREAKOUT | (inverted) | Trend tier — 300+min is BEST bucket |
| NMP_FADE | N/A | No tier-specific timeout helps |

**Rule derivation:** set `max_hold_min = <cliff>`. Exit trades past that
threshold as `<tier>_timeout`.

**Don't apply universally.** Catch-all tiers (NMP_FADE) and trend tiers
(MTF_BREAKOUT) are harmed by timeouts.

### Q3 — Peak signature (the universal exit rule)

**Method:** for each trade, find the bar where `peak_pnl` is maximum.
Extract the feature vector at that bar. Compute `delta = feat_peak -
feat_entry` per feature. Normalize by entry std dev. Rank by `|d/σ|`.

**The universal result** (confirmed on 4 tiers so far):
| Feature | d/σ range | Meaning |
|---|---:|---|
| `1m_p_at_center` | +9.9 to +12.8 | Price at regression mean |
| `1m_reversion_prob` | +0.8 to +1.2 | OU model expects reversion |
| `1m_variance_ratio` | -0.5 to -1.8 | Chaotic regime settled |

**Universal rule of three:**
```
Exit when:
  1m_p_at_center      > 0.35
  AND 1m_reversion_prob > 0.80
  AND 1m_variance_ratio < 1.0
```
**Plus amplitude gate** (ALWAYS needed): `peak_pnl >= $10`. Without it
the rule fires on 90%+ of trades at tiny peaks, giving back everything.

**When to skip the peak rule** (negative findings):
- **NMP_FADE** (catch-all, mean winner peak $69): early peak-exit
  shortens winners more than saving losers. Inverse-only is best.
- **MTF_BREAKOUT** (trend tier): 254 "never worked" trades bleed to
  inverse at -$82/trade, eating the $21K peak wins. Also best with
  inverse-only.

**Rule of thumb:** if winners' mean peak is under ~$80, peak rule
probably hurts. Tiers with mean winner peak > $130 benefit (CASCADE
+$225, NMP_RIDE +$X, RIDE_AGAINST pre-flip +$213, TREND_FOLLOWER after
flip).

---

## 3. Phantom Entry (the fizzle protector)

**The idea (2026-04-06 original, 2026-04-18 reimplemented).** When tier
conditions fire, don't enter — create a PENDING signal and watch. Enter
only if price moves `N` ticks in our direction within `M` minutes.
Cancel otherwise.

**Why it works**: sets with the conditions that fire split into
- **Will confirm quickly**: setup is working, enter at slightly worse
  price, keep most of the edge
- **Won't confirm**: fizzle — skip entirely, skip the resulting loss

The 2-tick/3-tick/4-tick filter CUTS LOSERS MORE than it cuts winners
because losers typically don't move favorably at all — the "confirmation"
is exactly the thing losers fail to do.

**Observed optimum on RIDE_AGAINST**: 4-tick confirm / 2-minute window.
Sweep showed 2→3→4 ticks progressively improved; 5+ eroded winners.

**Implementation**: `self._pending` slot on the engine. In `on_state`,
before entry check, update pending. If confirmed → enter at CURRENT
price. If expired → clear. See `nightmare_iso.py`.

---

## 4. Entry Relaxation Principle (the counter-intuitive one)

**Once phantom is in place, relax other entry filters. Phantom does the
fizzle work — feature-based filters become redundant or over-restrictive.**

Observed on RIDE_AGAINST:
- Remove `5m_bar_range > 55`: +$498 (was tuned for ride direction)
- Remove `h1_against_fade` tier-overlap guard: +$1,444
- Lower `h1_vel > 3` to `> 2`: +$498
- Tighten phantom `> 3 → 4 ticks`: +$52

**Principle:** with phantom catching the market-level confirmation,
feature filters that attempt to PREDICT the same thing (will this trade
work?) become noise. Keep only filters that define WHAT tier this is
(the physics setup), not WHETHER it will work.

**Watch out for over-filtering signatures:**
- Peak-rule $/trade per-exit is high (+$20+) but total tier is flat/neg
- Trade count is low relative to tier's natural universe
- A sweep of any filter threshold shows negative gradient (looser = better)

---

## 5. Direction Flip (the 38%-WR tell)

**If a tier's WR is significantly below 50%, flip the direction first.**
The prior that the 91D feature set predicts direction at entry with >58%
is BROKEN (2026-04-17 CART experiment, OOS acc 46.5% = worse than majority
baseline). Most of what looks like "direction signal" is noise.

BUT — persistent <45% WR across thousands of trades IS a signal: the
tier's direction RULE is systematically wrong. Flip.

**Observed:**
| Tier | Pre-flip WR | Post-flip WR | Status |
|---|---:|---:|---|
| RIDE_AGAINST | 38% | 65% | Flipped 2026-04-18 |
| FREIGHT_TRAIN | 20% | (nuked) | 5 trades, no stat power |

**Not all low-WR tiers want flipping** — sometimes they just have bad
exits (winners give back). Check peak-bucket first: if >80% had peak
> $20, direction's fine, fix exits.

---

## 6. Chain Positions (multi-entry per tier)

**Feature (2026-04-18):** each tier engine can hold up to `max_chains`
concurrent positions. When tier fires again in SAME direction while
already in position and under cap → open chain.

**Observed effect at chains=4** (full engine, no tier-specific tuning):
- NMP_RIDE: 284 → 967 trades (+$26,808)
- NMP_FADE: 6,070 → 18,218 (+$11,960)
- CASCADE: 112 → 199 (+$4,306)
- MTF_BREAKOUT: 866 → 1,611 (+$4,607)
- **Engine total: $21,090 → $73,760 (+$43,448 = +$157/day)**

**Chain-hurt tiers:** TREND_FOLLOWER (-$3,813), MTF_EXHAUSTION (-$3,679).
Peak-based tiers that work on "first entry at extreme" — later chains
enter at worse prices.

**For tier FIXING**, run `chains=1` (isolated single-position) so per-tier
WR measurements are honest. Chains are a separate multiplier to apply at
the end.

---

## 7. Five EDA Questions Beyond the Three

From journal accumulation; useful for deeper dives.

### 7a. Peak-reacher vs non-reacher separators

For tiers where peak rule fires on 60-70% of trades (rest go to timeout/
inverse at a loss), split by "reached $10 peak" vs "didn't" and Cohen-d
entry features. The separators are candidate ENTRY FILTERS. (Note: post-
2026-04-18, phantom entry may subsume this; try phantom first.)

### 7b. Higher-TF state at entry (2026-04-11 finding)

Universal pattern across tiers: **winners enter when higher TFs are calm
or aligned; losers enter when higher TFs are racing against them.** Check
`5m_velocity`, `1h_velocity`, `1h_z` in lookback window pre-entry.

### 7c. Resonance Cascade (2026-04-06 finding)

Multi-TF alignment amplifies edge:
- 1m z extreme = "enter"
- 5m/15m wick rejection = "confirmed"
- 1h z aligned = "amplified"

Base KILL_SHOT: 96% WR $16/tr. + 1h alignment |hz|>1.5: 97% WR **$24/tr**.
Each TF layer adds confidence AND $/trade.

### 7d. Chop is universal loser signal (2026-04-14 finding)

Top Cohen-d across 89K trades:
- `15m_wick_ratio` d = -0.27 (high wick = chop = loss)
- `1h_wick_ratio` d = -0.26
- Per-tier chop-d: RIDE_AGAINST -0.43, FADE_CALM -0.36, FADE_AGAINST -0.42

**High wick_ratio at higher TFs = caution.** Consider as a filter.

### 7e. Gravity well (2026-04-14 finding)

`z_high` / `z_low` are not just historical extremes — they're **attractor
wells** that pull price toward the regression mean. Deeper historical
extreme = stronger gravity.
- Long near 1h floor = trading WITH gravity (good)
- Long near 1h ceiling = trading AGAINST gravity (bad)

**Feature to watch**: `15m_z_high` / `15m_z_low` — winners enter with
deeper highs in their direction (d ≈ +0.08, weak but present).

---

## 8. Exit Physics Beyond the Peak Rule

### 8a. Breakeven lifespan (2026-04-06)

For kill-shot / fade trades with peak > $5:
- 43-59% of trades NEVER break even — they're permanent moves
- Peak takes 5-7 hours on those
- Lifespan (time at profit before reversion) median 50-85 min

**Implication:** trailing stops at modest giveback preserve most $.

### 8b. Trailing stop optimization (2026-04-06)

| Giveback | $/trade | WR | Peak Capture |
|---|---:|---:|---:|
| 10% | $19.6 | 98% | 74% |
| 25% | $20.6 | 98% | 62% |
| 70% | $25.5 | 93% | 35% |

Sweet spot: 10-25% giveback. User stripped trail mechanic 2026-04-18
in favor of peak rule + timeout — but a soft trail might be worth
re-adding for tiers without a clean peak signature.

### 8c. Winner MAE (2026-04-16)

- **97% of winners DIP NEGATIVE FIRST** before becoming winners
- Median dip: -$8.50
- 90% of winners recover from dips ≤ -$17
- Hard stops at -$50 kill 25% of winners; at -$150 kill 0.2%

**Rule:** don't stop winners early. Patience is an edge. No hard stop
above -$150 unless physics says exit.

### 8d. 5m alignment exit patience (2026-04-09)

Split trades by 5m velocity at entry:
- 5m WITH fade: 549 trades, **85% WR**, $35.5/trade
- 5m AGAINST fade: 3,801 trades, 63% WR, $14.5/trade

Use as EXIT PATIENCE (not entry filter):
- Aligned → hold longer (high confidence reversion)
- Opposed → exit faster (lower confidence)

### 8e. Thesis-validity (today)

Exit when the entry condition DECAYS below its threshold (not when the
opposite extreme fires). Examples:
- RIDE_AGAINST: h1_vel flipped sign → thesis gone (sign flip, not wait for |h1_vel|>3 on other side)
- TREND_FOLLOWER: |1m_vel| drops below entry threshold → momentum dead
- KILL_SHOT: 5m_wick decays → wick rejection gone

Note: direction-of-thesis matters. For fades (RIDE_AGAINST post-flip),
h1_vel decay is FAVORABLE, not unfavorable — thesis-validity must be
defined against pos direction, not entry conditions blindly.

---

## 9. Anti-Patterns (burned time in these)

### 9a. Don't trust $740/day baselines

The nn_v2 blended $740/day baseline (2026-04-09) was **lookahead-inflated**.
Post-fix 2026-04-17: same pipeline → -$164/day. Any "historical baseline"
before 2026-04-17 should be considered contaminated unless the feature
file timestamp is post-lookahead-fix.

### 9b. Don't start with CART on winners vs losers

We tried (2026-04-17). CART found `1m_reversion_prob > 0.866` = 86%
oracle capture, looked clean... Turns out `regret.py::correct_trades`
replaces `entry_79d` with approach-buffer features (99.7% of trades).
The CART learned FEATURE LEAKAGE, not signal. After joining to actual
iso entry features, OOS accuracy dropped to 46.5% (worse than majority).

**Use the three-question method (blunt but honest).** Only CART after
you've verified data isn't polluted.

### 9c. Direction at entry is RANDOM on 91D

Hard-proven on 2026-04-17 regime-discovery work. Day-stratified CART
can't beat majority baseline. Don't chase "counter-flip" tiers without
a physics reason (like a mechanically-flipped direction rule).

### 9d. Don't use time-of-day filters

Lazy. Hour-of-day correlates with the real physics cause (liquidity,
session chop), but the filter doesn't generalize. Find the FEATURE that
spikes in those hours and filter on THAT. The 2026-04-17 user note:
"time filter admits we don't understand the physics."

### 9e. Peak rule isn't universal

Applied blindly it hurts catch-all tiers (NMP_FADE, MTF_BREAKOUT).
Rule: if winner mean peak < ~$80, peak rule probably costs more than
it saves. Stick to inverse-signal.

### 9e-bis. Phantom entry isn't universal either (2026-04-18 discovery)

Phantom + short timeout (≤15m) = safe. RIDE_AGAINST had 15m timeout,
phantom was a huge win. MTF_BREAKOUT had no timeout, phantom + inverse
exit helped (+$1,020).

Phantom + long timeout (60m+) = dangerous. TREND_FOLLOWER has 60m
timeout. Adding phantom went +$495 → **-$1,814** (-$2,309 swing).
When phantom-confirmed trades fail, the 60m timeout lets losses bleed
for the full window: 80 trades at **-$166/tr** = -$13K disaster.

**Pattern:** phantom entry trades off worse entry price for fizzle
protection. Works when timeout is short enough to cap the downside
quickly. Fails when slow exit lets the worse-entry-price compound.

**Rule:** if tier's natural timescale is > 15-20 minutes, DON'T add
phantom unless you also shorten the timeout.

### 9f. Timeout isn't universal

KILL_SHOT had bimodal holds pre-peak-rule (0-10m wins, 300+min wins,
60-300m losers). A timeout between 30-300m would cut winners.
Always do Q2 (hold-time bucket) before imposing a timeout.

### 9g. Don't skip negative findings

When a rule DOESN'T help, that's valuable information. Record it in
the journal with the data that proves it. Future sessions will ask
"why don't we do X?" and the journal answers.

---

## 9bis. Peak-Bucket Framework (2026-04-19 discoveries)

The bucket framework reframes exits around **peak-ticks promotion**
rather than single-threshold cuts. Trades move through buckets over
their lifetime; each bucket has its own physics and its own exit rules.

### 9bis-a. Peak buckets (ticks-native, TV=$0.50)

| Bucket | Ticks | Dollars | Meaning |
|---|---:|---:|---|
| NOISE | 0–4 | $0–2 | bar-oscillation noise |
| FAKE | 5–9 | $2.50–4.50 | small move, no conviction |
| MARGINAL | 10–19 | $5–9.50 | real but small |
| REAL | 20–39 | $10–19.50 | significant directional move |
| STRONG | 40–79 | $20–39.50 | major move, thesis worked |
| DOMINANT | 80+ | $40+ | captured full range |

Ticks are the instrument-native unit. Dollar thresholds drift with
contract value; ticks don't.

### 9bis-b. Bucket position predicts outcome (KILL_SHOT_INVERSE)

Bar 10+ observation: trades in REAL+ bucket are **0% losers**. Once a
trade promotes to REAL, it's essentially a guaranteed winner. The
bucket IS the forecast.

Corollary: trades still in NOISE/FAKE past bar 10 are bleeders with
>90% loss probability. Cut without hesitation.

### 9bis-c. Multi-exit by bucket (2026-04-19)

One entry setup → multiple exit rules, one per bucket signature. Each
bucket has its own physics:

- **REAL bucket** (KILL_SHOT_INVERSE, 170 winners): dominant signature
  = directional extension (`|15s_dmi_diff|` large + `|5m_z_se|` large +
  `|1m_dmi_diff|` large). GMM split it by direction sign — same
  signature on positive/negative sides.
- **STRONG bucket** (62 winners): extreme 15s z-extension
  (`15s_z_low > 0.38 AND 15s_z_se > 1.03 AND 15s_z_high > 1.77`). All
  three z values high = price at upper band across the short TF.
- **DOMINANT bucket** (42 winners, biggest $/trade): **higher-TF calm**
  (`1h_bar_range < 367 AND 1h_vol_rel < 1.03 AND 1h_p_at_center > 0.615`).
  Totally different physics than REAL — peaks happen when 1h regime is
  quiet. Captures the big-move tail.

Implementation pattern: sorted rules, first-match-wins in `_check_exit`.
No-progress cut fires first (before rules), peak-signature rules next,
timeout last.

### 9bis-d. Parallel inverse-direction tier (2026-04-19)

For setups where WR is just modestly above 50%, running a parallel
tier with **flipped direction** on the SAME trigger captures the
continuation side. Worked for KILL_SHOT: the `KILL_SHOT_INVERSE` tier
(same trigger, opposite direction) produced 528 trades at 62% WR and
+$1,373 (vs KILL_SHOT normal at 58% WR / -$454). Both tiers share the
same setup; they split on direction thesis.

Caveat: this works when the underlying trigger has ambiguous direction
physics (wick could reverse OR continue). For triggers with strong
direction conviction (e.g. TREND_FOLLOWER already flipped), inverse
doesn't add value.

### 9bis-e. Data-defined milestone thresholds (2026-04-19)

At each bar checkpoint N, sweep peak_ticks thresholds and pick the value
that maximizes Youden's J (winner-retention − loser-retention). The
argmax is the physics-defined milestone for bar N.

KILL_SHOT_INVERSE milestone findings:
- bar 2-10: T ≈ 12 ticks (moderate separators, J=0.38–0.45)
- bar 15: T=7 ticks (J=0.63, 85% W keep / 78% L cut)
- bar 20: T=18 ticks (J=0.67, 67% W keep / **100% L cut**)

Strong rules (J ≥ 0.20) ship directly. Weak (J < 0.10) is noise.

This is preferred over hand-picked thresholds because it's adaptive to
the tier's specific distribution — some tiers have wider W/L gaps at
bar 5, others at bar 15.

### 9bis-f. POST-PHYSICS: MAE stop (risk management, deferred)

2026-04-19 `big_loss_physics.py` finding: BIG_LOSS trades (pnl < −$50)
cost the engine ~$356K IS (−$128/trade over 2,219 trades). Every single
one crosses −$40 MAE by median bar 9. Only 14% of winners ever dip to
−$40. A universal −$40 MAE stop estimated to net **+$166K / +$600/day**.

**NOT part of physics work.** Physics = peak signatures, cliff cuts,
cluster-based exits (all derived from FEATURE state). MAE stop is
capital-protection risk management — flat $-threshold, no physics
insight. Deferred to a separate live-risk-mgmt pass.

### 9bis-g. Loser peak signatures (next lever, not yet implemented)

~25% of losers reach MARGINAL or higher bucket before reversing
("round-trippers"). Clustering their peak-state features could surface
"peak-about-to-reverse" signatures → proactive break-even exits.

Population size: ~70 trades total across KILL_SHOT + INVERSE. Small
but workable. Same pipeline (PCA + GMM) as winner clustering.

---

## 10. Tools Reference

Living tools, updated 2026-04-18:

- **`tools/tier_eda.py --tier NAME`** — parameterized Q1/Q3 analysis with
  segment breakdown, feature separators, regime shift. Writes markdown
  to `reports/findings/tier_eda_<TIER>_<ts>.md`.
- **`tools/daily_hourly_pnl.py`** — mode/WR/hourly breakdown for revenue
  framing. Target metric: $/active-hour, not $/day total.
- **`tools/path_features_eda.py`** — 12-bar 5s path metrics (efficiency,
  R², reversal count) on pre-entry window. Found nothing stronger than
  `1m_bar_range` already in 91D (deprecated but retained).
- **`tools/slope_eda.py`** — OLS β+γ on 30-bar windows per TF. The β
  (velocity of regression mean) was d=-0.31 moderate signal for
  TREND_FOLLOWER. Infra available via `IsoEngine._slope_1m(ts)` but not
  yet consumed.
- **`tools/tier_exit_physics.py --tier NAME`** — comprehensive exit-design
  report: cohort summary, peak timing, bar-N trajectory, fork bar,
  give-back from peak, slope β at entry/peak/exit, cut-rule scan, entry
  discrimination, peak signature. Writes `reports/findings/exit_physics_<TIER>.md`.
- **`tools/peak_bucket_lifecycle.py --tier NAME`** — bar-by-bar bucket
  heatmap (NOISE → DOMINANT) + per-bucket peak-signature clustering
  (GMM on 91D peak-state features). Writes `peak_lifecycle_<TIER>.md`.
  Requires features-intact pickle (single-tier run < 12K strip threshold).
- **`tools/milestone_thresholds.py --tier NAME`** — data-defined
  peak-ticks milestone per bar (Youden's J sweep). Writes
  `milestone_thresholds_<TIER>.md`. Physics-derived cut gates, no
  hand-picked thresholds.
- **`tools/loser_cliff_eda.py --tier NAME`** — legacy focused cliff tool
  (Q2 hold-time separator). Superseded by `milestone_thresholds.py`
  but kept for the entry-feature Cohen d output.
- **Ad-hoc Python** for Q2 (path pnl trajectories) and Q3 (entry→peak
  feature deltas). Pattern is consistent: load `iso_is.pkl`, filter by
  `entry_tier`, compute, print.

---

## 11. Current Tier Scorecard (2026-04-18 end of day)

With all current rules, chains=1:

| Tier | N | WR | $/trade | $/day | Status |
|---|---:|---:|---:|---:|---|
| NMP_RIDE | 299 | 51% | +$40.01 | +$43 | ✓ Winner, keep |
| FADE_AGAINST | 326 | 47% | +$23.19 | +$27 | ✓ Winner, keep |
| MTF_EXHAUSTION | 125 | 47% | +$23.12 | +$10 | ✓ Winner, small sample |
| CASCADE | 113 | 55% | +$42.84 | +$17 | ✓ Winner, small sample |
| NMP_FADE | 8,174 | 55% | +$1.36 | +$40 | ✓ Workhorse |
| RIDE_AGAINST | 4,204 | 65% | +$0.70 | +$11 | ✓ Fixed today |
| TREND_FOLLOWER | 780 | 68% | +$0.82 | +$2 | ✓ Fixed earlier |
| KILL_SHOT | 519 | 67% | -$1.36 | -$3 | ~ Break-even |
| MTF_BREAKOUT | 943 | 40% | -$1.18 | -$4 | ⚠ Marginal (trend tier) |
| **Engine total** | **14,683** | — | — | **+$143** | — |

Pre-session baseline: $76/day. Session lift: **+$67/day**.

Target: $60/active-hour = ~$300/day. Still work to do — likely via
applying phantom entry pattern to other tiers + chain multiplier.


## Source: tier_building_playbook_v2.md

---
name: V2-native tier development playbook
description: Step-by-step methodology for deriving tiers in the V2-native training_v2 pipeline using multi-axis regret analysis. Companion to tier_building_playbook.md (V1-shape iso pipeline).
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
# V2-Native Tier Development Playbook

**Established 2026-05-05 from the regret-based discovery cycle on 19,106 NMP IS trades.**

This playbook is the working methodology for deriving, validating, and
rejecting candidate tiers in the V2-native `training_v2/` pipeline.
Supersedes `feedback_tier_three_questions.md` for V2 work; companion to
`tier_building_playbook.md` (which covers the V1-shape iso pipeline).

The core insight: **multi-axis regret analysis is the LABEL GENERATOR
that drives tier discovery.** Without it you're guessing; with it, every
sub-tier is a falsifiable hypothesis with a measured baseline.

---

## 0. Pre-requisites

Before running any tier work, verify:

- **V2 features built**: `DATA/ATLAS/FEATURES_5s_v2/{L0, L1_<TF>, L2_<TF>, L3_<TF>}/YYYY_MM_DD.parquet` for all IS+OOS days
- **2D regime labels**: `DATA/ATLAS/regime_labels_2d.csv` from `tools/atlas_regime_labeler_2d.py`
- **No lookahead in V2 SFE**: `core_v2/statistical_field_engine.py` has the lookahead audit baked in
- **Engine sanity**: `python -m training_v2.run --smoke` (single-day no-thresholds, no-CNN)

---

## 1. The 6-Step Discovery Cycle

### Step 1 — Run the seed entry alone

Base NMP / REVERSION trigger only. No CNN, no thresholds, no other strategies.

```
python -m training_v2.run --is  --strategies REVERSION
python -m training_v2.run --oos --strategies REVERSION
```

This gives the **honest baseline** the rest of the cycle is measured against.
Save the trade pickles — they're the input to every subsequent step.

**Sanity check**: count WR should be near 50-55%, $/trade near $0. If WR is
much higher, the trigger is too restrictive (not enough trades to discover
sub-tiers from). If WR is much lower, your trigger is broken.

### Step 2 — Run multi-axis regret on the trades

```
python -m training_v2.regret_full --trades training_v2/output/nmp_only.pkl \
    --out training_v2/output/regret_full_nmp.pkl
```

`regret_full.py` produces `FullRegretLabel` per trade with:

| label | meaning |
|---|---|
| `same_early_best` | best PnL strictly before actual exit |
| `same_at_exit` | actual exit (baseline) |
| `same_extended_best` | best PnL after the actual exit |
| `counter_early_best` | best counter PnL in bars 0-3 (flip-and-go) |
| `counter_at_exit` | counter PnL at the actual exit bar |
| `counter_extended_best` | best counter PnL after the actual exit |
| `best_action` | argmax over the 6 options above |
| `regret` | best_pnl − actual_pnl |
| `early_entry_gain` | extra $ if we'd entered at the best earlier bar |

These labels are what tells you *what the trade SHOULD have been*. The
distribution of `best_action` is the first signal: if 50% of trades are
`counter_early` or `counter_extended`, you have a direction-flip splitter.
If 40% are `same_extended`, your exit is firing too early.

### Step 3 — Identify splitter axes

Three discovery angles, in order of OOS-survival likelihood:

#### 3a. Categorical splitters (highest survival rate)

Cross-tab pivot the trades by **categorical** entry features × the
counterfactual best-direction:

```python
df.groupby(['regime_idx', 'direction']).agg(
    n=('actual_pnl', 'size'),
    actual=('actual_pnl', 'mean'),
    fade_peak=('peak_pnl', 'mean'),
    flip_peak=lambda l: (-l).mean()  # = -mae_pnl
)
```

Look for cells where `flip_peak >> fade_peak` (or vice versa) — those are
flip-rule candidates. **The 2026-05-04 V2 discovery: 3 of 12
(regime × direction) cells qualified for flip:**
- `(UP_SMOOTH, short)` → flip to long
- `(UP_CHOPPY, short)` → flip to long
- `(DOWN_SMOOTH, long)` → flip to short

Categorical splitters survive walk-forward + true OOS far more reliably
than continuous-feature filters because the cells have hundreds-thousands
of trades each.

#### 3b. Continuous-feature filters (medium survival rate, requires care)

For each (regime, direction) cell, run within-cell winner-vs-loser EDA
on all 185 V2 columns (`training_v2/within_cell_eda.py`). Look for
features with Cohen's d ≥ 0.2 that survive 70/30 walk-forward.

**Confirmed pattern (2026-05-05): 9 of 12 cells had walk-forward-surviving
top features inside IS, but they FAILED on true OOS hold-out.** The
`FilteredRegimeAwareReversion` strategy lost -$19.85/day OOS. Continuous
quantile thresholds remain a known overfit trap (see
`feedback_quantile_selection_overfit.md`, `feedback_high_vol_harness_failed.md`).

Use continuous filters only as second-pass refinements, never as primary
splitters, and demand both walk-forward IS *and* date-disjoint OOS lift
before shipping.

#### 3c. Time / day-of-week splitters

Group by hour-of-day and weekday. Bleed concentration in specific hours
indicates a **structural condition** (volatility expansion at NY open, etc.)
that should be diagnosed BEFORE acting on the time bucket.

**Important: DO NOT filter on time directly.** Time is correlated with
volatility — find the underlying volatility/structural feature and condition
on that. The 2026-05-05 NY-mid-session bleed turned out to be entry vol
expansion, not "time of day".

### Step 4 — Build the candidate as a strategy variant

For each splitter discovered, write a thin strategy class that wraps
the seed and applies the rule. Examples in `training_v2/strategies/`:

- `regime_aware.py` — flip rule (categorical splitter)
- `filtered_nmp.py` — per-cell quality filter (continuous splitter — REJECTED)

Each variant gets its own name (e.g., `NMP_REGIME`, `NMP_FILTERED`) so
the trade pickle's `entry_tier` field carries the experiment identity for
downstream analysis.

### Step 5 — Validate at THREE levels

A candidate must clear all three to ship:

#### 5a. IS apples-to-apples re-simulation

Use `simulate_exit` from `training_v2/regret.py` to apply the same threshold
policy to baseline vs candidate. This isolates the *signal* contribution
from threshold-tuning effects.

#### 5b. IS walk-forward

Train on first 70% of IS days, validate on last 30%. Compute bootstrap CI
on per-day delta. Required: CI lower bound > 0.

**Caution: walk-forward inside IS is NECESSARY but NOT SUFFICIENT.** Many
overfit rules survive 70/30 IS but break on date-disjoint OOS. Continuous
filters in particular often pass IS-WF and fail OOS.

#### 5c. True OOS engine run

Run the actual engine on the OOS days. Compute bootstrap CI on
(candidate $/day − baseline $/day). Required: CI not catastrophically
negative; ideally CI > 0 (significant) but at minimum CI lower bound
not far below 0.

```
python -m training_v2.run --oos --strategies NMP_VARIANT \
    --thresholds training_v2/output/thresholds_prod.json
```

### Step 6 — Ship, hold, or reject

| IS-WF CI | OOS CI | verdict |
|---|---|---|
| significant > 0 | significant > 0 | **SHIP** as production |
| significant > 0 | positive but CI includes 0 | hold; collect more OOS data |
| significant > 0 | clearly negative | **REJECT — overfit** |
| not significant | any | **REJECT — no signal** |

Ship adds a tier to production; reject removes the experimental code.
**"Hold" should never be production**; if a tier needs more data, run the
engine more before shipping.

---

## 2. The Bleed-Cause Causal Pattern

When a tier underperforms in a specific zone (hour bucket, regime, etc.),
the analysis MUST distinguish symptom from cause. The 2026-05-05 NY-bleed
investigation showed:

```
Symptom: hours 14-17 UTC lose money
Apparent cause: "NY mid-session is bad"
Actual cause: 4-8x volume/range expansion at entry, fades fail when
              extremes extend further
```

Workflow:

1. Identify the bleed cluster (autopsy by hour/regime/tier)
2. Compare V2 feature distributions between bleed-zone and profit-zone
   trades (`training_v2/bleed_cause_analysis.py`)
3. Look for STRUCTURAL feature differences with large Cohen's d
4. **The structural feature, not the cluster label, is the lever**

For NY mid-session: filter on `L2_1m_vol_mean_15` or `L1_5m_bar_range`,
NOT on `hour_utc`. The vol features generalize to other volatility-spike
periods (Fed days, news, etc.) that hour buckets miss.

---

## 3. Anti-Patterns (Confirmed Failures)

### 3.1 Re-simulation overestimates engine impact

`simulate_exit` walks the regret pnl_path and ignores state-driven exits
(`ZSeReversal`, `SwingNoiseSpike`). The 2026-05-04 flip-rule estimate
was +$68/day OOS via re-sim; actual engine produced +$1.66/day. **A 40×
discrepancy.**

**Always validate by running the engine, not just by re-simulating.**

### 3.2 Walk-forward IS is not OOS

Continuous-feature thresholds that pass 70/30 IS-WF (9 of 12 cells in
the 2026-05-05 within-cell EDA) failed on true 2026 OOS hold-out
(-$19.85/day). The 70/30 IS-WF data has temporal structure that lets
overfit rules sneak through.

**Demand date-disjoint OOS, not just walk-forward inside IS.**

### 3.3 Mean-based thresholds overshoot fat-tail distributions

Vol-adaptive exits (2026-05-05) used Bayesian-derived TPs based on the
*mean* peak per vol bin. Q5 high-vol mean peak was $144 — but the typical
trade peak was much lower (fat-tailed). TP set at $51 missed most trades.

**For fat-tail distributions, use lower quantiles (q_05 to q_15) for TP
or stick to median-based formulas.**

### 3.4 Direction flip in volatility-expansion zones

Hypothesis: "if fades fail in high vol, flip to ride direction". The data
rejected this — peaks are symmetric across directions in high-vol bins
(d ≈ 0.05-0.10 between fade_peak and flip_peak). High-vol just means
**big peaks for both directions** — not a directional bias.

**A direction-flip rule needs the regime/state to favor one side
asymmetrically. Symmetric volatility expansion is not such a state.**

### 3.5 Tier-aware exit rules are non-negotiable for ride trades

`ZSeReversal` (mean-reversion exit) fired bar 1 on every flipped trade,
killing the flip rule entirely until fixed (`ZSeReversal.RIDE_TIERS`
guard, 2026-05-04). Any new fade/ride distinction MUST audit the existing
exit rules for direction-asymmetric assumptions.

---

## 4. Reference: V2-Native Discovery Cycle Artifacts

| Tool | Output | Purpose |
|---|---|---|
| `training_v2/run.py --is/--oos` | `is.pkl`, `oos.pkl` | Raw trade lists |
| `training_v2/regret_full.py` | `FullRegretLabel` pickles | Multi-axis regret labels |
| `training_v2/regret.py` | `RegretLabel` pickles | Simple peak/MAE labels (legacy compat) |
| `training_v2/tier_discovery.py` | flip/winner/peak EDA | Categorical splitter discovery |
| `training_v2/within_cell_eda.py` | per-cell feature ranking | Continuous filter discovery |
| `training_v2/full_feature_eda.py` | global feature ranking + Spearman | First-pass scan |
| `training_v2/cell_filters.py` | filter JSON | Filter learner (REJECTED 2026-05-05) |
| `training_v2/flip_rule_validation.py` | walk-forward + OOS CI | Categorical splitter validator |
| `training_v2/loser_autopsy.py` | per-zone bleed report | Loser-pattern analysis |
| `training_v2/bleed_cause_analysis.py` | Cohen's d zone-vs-zone | Causal-mechanism identifier |
| `training_v2/threshold_bayesian.py` | thresholds JSON | Per-cell exit derivation |
| `training_v2/strategies/regime_aware.py` | NMP_REGIME class | Production flip rule |

---

## 5. Tier Lineage & Status (2026-05-05)

| tier | source | status | notes |
|---|---|---|---|
| REVERSION | V2 NMP seed | live | base entry condition, all sub-tiers descend from here |
| MA_ALIGN | EDA (vwap alignment) | live | 7-of-8 vwap_w alignment |
| VEL_BODY_CHORD | EDA (chord) | **killed** | 2026-05-04: lottery-day artifact |
| **NMP_REGIME** | regret + flip rule | **production** | RegimeAwareReversion |
| NMP_FILTERED | within-cell filter | **rejected** | 2026-05-05: OOS overfit |
| (vol-adaptive exits) | exit-side variant | **rejected** | 2026-05-05: fat-tail overshoot |

The discovery cycle on REVERSION found one categorical sub-tier (NMP_REGIME)
in 1 day. Continuous filters on top failed; exit-side variants failed.
**Next discovery candidates** worth running the cycle on:
- 4h-context entry (RIDE if 4h vel aligned with z direction)
- Multi-TF velocity exhaustion (legacy MTF_EXHAUSTION analog)
- Wick-rejection quality filter (legacy KILL_SHOT analog) — needs OHLCV math, not in V2 entry vector

Keep the 6-step cycle disciplined: each candidate is a 4-6 hour effort,
most reject, but the ones that survive are real.
