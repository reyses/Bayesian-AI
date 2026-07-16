# The "template thing" identified + its pattern layer in the league; resurrection proposal
**Doc:** 086 · **Date:** 2026-07-16 · **Author:** Claude · **Status:** FINAL
**Trigger:** Moises — "way back Claude built a template thing that made templates of
actual events along the year (kind of a black box), saved as a branch — check what
it was" + "I'm sure the templates did like doji and hammer patterning."

## 1. What it was (recovered from git + PROJECT_HISTORY)
The **fractal K-means pattern-template engine** (Feb-Mar 2026, the original
Bayesian-brain era):
- **Event layer** (`core/pattern_utils.py` + `core/cuda_pattern_detector.py`):
  candlesticks DOJI (body<0.1×range), HAMMER (lower shadow>2×body, upper<0.1×range,
  body<0.3×range), ENGULFING_BULL/BEAR (exact 2-bar engulf), with a strict priority
  cascade (doji > hammer > engulfing); plus geometric COMPRESSION/WEDGE/BREAKDOWN.
  Moises' memory confirmed exactly.
- **Template layer** (`core/fractal_clustering.py`, 658 lines): each event embedded
  in a 16-D vector (|z|, |velocity|, momentum, coherence, log2(tf), fractal depth,
  parent context, multi-TF alignment, hurst, dmi, ...) → RECURSIVE K-means (split to
  depth 5) → `PatternTemplate`s: centroid, members, EV, WR, MFE/MAE, long/short
  bias, Markov transition_map (which template follows which).
- **Brain** (`core/bayesian_brain.py`): win/loss tables keyed by template×direction.
- **Black-box diagnosis** (history's own words): K-means on the mixed-semantics 16-D
  blob "produced centroids that no longer corresponded to recognizable patterns."
- **Where**: the branch (pre-snowflake checkpoint `3d0c1b8`) is DELETED; the code is
  intact at commit **09cd30d8** (2026-03-07); files removed from main 2026-04-07
  (23db222f). Not in safe/v740. Era caveat: all its NUMBERS are lookahead-tainted
  (pre-2026-04-17); only the mechanism is reusable.

## 2. Pattern layer ported to the league (streams #38-39)
Formulas verbatim from @09cd30d8, on 1m buckets, emitted at bar close, RTH-gated:
```
PTRN-ENGULF N=29917 AUC 0.616 base 0.62 || 0.51 / 0.63 / 0.72 [0.71,0.74]
PTRN-HAMMER N= 4484 AUC 0.615 base 0.55 || 0.43 / 0.57 / 0.65 [0.61,0.68]
```
- ENGULF direction is in the formula; aligned 0.62 — an engulfing bar is a
  continuation event (the ride>fade law again).
- HAMMER = classic bullish reading (DECLARED adaptation: legacy kept patterns as
  state flags; direction was learned by the brain).
- DOJI skipped as a stream (directionless — fabrication risk); candidate combiner
  FEATURE. Geometric patterns not yet ported (need the highs/lows geometry read).

## 3. Resurrection proposal — the 2024-frozen template stream (the complicated one)
Faithful causal design, sized as the FIRST OPUS-WORKER TRIAL under the delegation
ladder (Claude specs + verifies; worker executes):
1. Recover `fractal_clustering.py` + `pattern_utils.py` from 09cd30d8 into
   `research/nt8_catalog/templates_v0/` (read-only reference).
2. Rebuild the event layer causally on 2024 (events = pattern fires across TFs);
   embed with the 16-D vector recomputed from raw bars (V1 formulas; no feature-
   store dependency; SEGMENT-FIREWALL-clean — no future context).
3. Fit the recursive K-means on **2024 events only**; FREEZE centroids + per-
   template direction-bias tables.
4. Stream: 2025-26 events assigned to nearest frozen centroid; direction = the
   template's frozen 2024 bias (long_bias vs short_bias); value = bias strength;
   through the standard league eval.
5. Kill-points: if template assignment is unstable (nearest-centroid margin ~0)
   or 2024 biases don't transfer, report the honest null — the league already
   shows the named streams carry the signal; the templates must beat a "no
   clustering, just the pattern type" baseline (PTRN-* streams = that baseline).

## 4. State
League = 39 streams; combiner refreshed (see reports/combiner_preview.md).
All rows parquets saved. Queue: Opus worker (template resurrection §3 or economic
conversion — Moises picks priority), Sonnet worker (overfit-decay).
