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
