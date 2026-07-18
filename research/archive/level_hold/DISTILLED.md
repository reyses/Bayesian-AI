---
name: distilled-level_hold
description: Frozen band levels do NOT act as special S/R — the bounce is generic mean-reversion; real vs phantom levels hold equally often.
metadata: {type: distilled, topic: level_hold, status: dead}
---
## Verdict
Tested the user's live observation that frozen regression-band levels act as
recurring S/R (`README.md`, Figure_1). Causal freeze-then-touch design with a
phantom-level null (jittered ±4-16 ticks) isolates "this level is special"
from generic mean-reversion. Result: real ≈ phantom at every barrier scale —
no level family clears significance. Three follow-on probes (touch-history
fatigue, long-memory pivot landing, slope-persistence) also came back null
or too weak to trade.
## Key numbers (with CIs where they exist)
- P(hold|touch), 6 days IS, ~15-17k resolved touches/config (`LEVEL_HOLD_FINDINGS_2026-07-07.md`):
  8 ticks: real 0.61-0.63 vs phantom 0.62-0.64; 20 ticks: real 0.58-0.63 vs phantom 0.58-0.61;
  40 ticks: real 0.51-0.59 vs phantom 0.50-0.57 (both → 0.5). R20 slow_extreme real 0.625
  [0.602,0.647] vs phantom 0.594 [0.574,0.614] (`level_hold_R20.txt`, best positive, CIs overlap);
  fast_extreme flips negative at R40: real 0.550 vs phantom 0.574 (`level_hold_R40.txt`).
- touch_history_breakout, 63 days (`touch_history_R20.txt`/`_R40.txt`): P(break) by prior-visit
  count flat — R20: 0.433/0.391/0.406/0.411 for {0,1,2,3+}; R40: 0.474/0.440/0.453/0.439 — no
  monotonic fatigue signal, CIs overlap. Sigma-scaled zone variants (`touch_history_sigma{0.5,1.0,2.0}_R20.txt`)
  shift the base rate (0.33/0.21/0.12) but stay flat across visit buckets.
- pivot_level_proximity, 63 days, N=3339 pivots (`pivot_level_proximity_thr20.txt`): real fraction
  near a morning-frozen level vs Monte-Carlo null (400 shifts): tol4 0.068 vs 0.061 [0.052,0.069]
  p_emp=0.043; tol8 0.130 vs 0.118 [0.107,0.129] p_emp=0.020; tol12 0.188 vs 0.171 [0.157,0.185]
  p_emp=0.007 — small but marginal. 6-day thr40 subset not significant (p_emp 0.13-0.91, N=144).
- churn_slope_persistence, 63 days, N=81967 (`churn_slope_persistence.txt`): base rate P(up)=0.511;
  P(same-dir continuation) by slope bin 0.470/0.471/0.496/0.507/0.506 (down-drive→up-drive) — all
  ≈ coin-flip, wiggle/|net| ≈ 2.2 across bins.
## Graveyard / never-retry (if any)
- Level-value-as-S/R (real vs phantom holds equally) — falsified at 8/20/40-tick scales, N≈17k touches/config.
- Touch-count "fatigue" (more prior visits → more likely to break) — no signal, R20 and R40 both flat.
- Directional-churn-is-rideable (slope persistence) — null, P(same dir) stays at coin-flip across all 5 slope bins.
## Reusable assets
- `tools/level_hold_study.py` — core freeze/touch/outcome engine + phantom-level null (`atlas`, `rolling_ols_bands`, `first_outcome`, `wilson` helpers imported by the other 3 tools).
- `tools/touch_history_breakout.py` — prior-visit-count vs P(break), sigma-scaled zone variant.
- `tools/pivot_level_proximity.py` — long-memory pivot-landing test with Monte-Carlo shift null.
- `tools/churn_slope_persistence.py` — slope-to-noise bin vs forward-continuation probe.
## Data locations
- `DATA/ATLAS/{5s,1m}` parquet (repo root, gitignored) — the only data dependency, via `atlas()` helper.
## Open threads
- Confluence untested: P(hold|touch, approach-curvature) — edge may live in level+curvature+Markov
  confluence (user's actual practice), not level alone, if hold-rate spreads by approach geometry.
- pivot_level_proximity's marginal positive (p_emp 0.007-0.043, N=63 days) never re-run with more
  data or a multiple-comparisons correction (3 tolerance thresholds tested).
## Sources
- research/level_hold/README.md
- research/level_hold/reports/LEVEL_HOLD_FINDINGS_2026-07-07.md
- research/level_hold/reports/level_hold_R20.txt, level_hold_R40.txt
- research/level_hold/reports/touch_history_R20.txt, touch_history_R40.txt, touch_history_sigma{0.5,1.0,2.0}_R20.txt
- research/level_hold/reports/pivot_level_proximity_thr20.txt, pivot_level_proximity_thr40.txt
- research/level_hold/reports/churn_slope_persistence.txt
- research/level_hold/tools/{level_hold_study,touch_history_breakout,pivot_level_proximity,churn_slope_persistence}.py
## Archive recommendation
ARCHIVE — every hypothesis tested (level-specialness, touch fatigue, slope persistence) came back null
against its own null model; only pivot-proximity long-memory has a marginal, uncorrected positive too
weak to justify keeping this live. The open thread (confluence w/ approach geometry) belongs to a
successor study, not a reason to keep this folder active.
