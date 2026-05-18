# Session Log — 2026-05-17 build

Chronological log of decisions, dead ends, and pivots that led to the
final architecture. Useful for outside reviewers to understand WHY
each component is shaped the way it is.

## Phase 1: Direction signal validation

**Question**: can V2 features predict the next-bar direction better than
a streaming zigzag indicator?

- Built `direction_signal_accuracy.py` to compare raw trend3 classifier
  vs DMI-smoothed vs leg_direction truth.
- Result: raw 65.86% accuracy at 66% coverage; smoothed 61.48% at 99.9%
  coverage. Delta -5.51pp, CI [-7.72, -3.75] — smoothing hurts.
- Built `live_zigzag_baseline.py` as causal indicator. Result: indicator
  alone hits 64.85% per-day accuracy at 97.6% coverage.
- **Paired delta of raw trend3 vs live ZZ: +1.80pp, CI [-1.13, +4.40] —
  NOT statistically significant.**

**Conclusion**: V2 features can't beat a 30-line zigzag indicator on
hindsight direction. Direction is essentially solved by indicator
construction.

## Phase 2: Forward-looking targets reveal real V2 edge

**Question**: if direction is solved, what CAN V2 predict that indicator can't?

- Built B1 (`train_b1_pivot_imminent.py`) — pivot-imminent binary
  classifier at K=1,3,5,10 minutes.
- Result: K=10 thr=0.85 → 78.1% precision, 4% coverage, AUC 0.716.
  Indicator can't predict pivot timing — V2 features can.
- Built B2 (`train_b2_fakeout.py`) — fakeout classifier per pivot.
- Result: K=10 thr=0.70 → 66% precision, 19% coverage, AUC 0.695.

**Conclusion**: V2 features have real predictive signal for FORWARD-
LOOKING targets (pivot timing, fakeout) where indicator has none.

## Phase 3: Reframing — region vs imminence

**Question**: B1 predicts FORWARD-only. Can a symmetric (±W around any
pivot) classifier extract more signal?

- Built B4 (`train_b4_pivot_region.py`) — symmetric region classifier
  at W=30,60,120,300 seconds.
- Result: W=300s thr=0.85 → 79% precision at 10% coverage. Beats B1 K=10
  at same precision but 2.5× more coverage.

**Insight**: post-pivot signal (volume residuals, mean-reversion) is
real and B1 was missing it. Symmetric framing captures more.

## Phase 4: Leg-phase 3-class

**Question**: can we tell EARLY-leg from LATE-leg directly?

- Built B5 (`train_b5_leg_phase.py`) — 3-class EARLY/MID/LATE.
- Result: OOS accuracy 38.6% (below MID-baseline 46.4%). Weak.
- BUT: P(MID) > 0.60 has 71% precision at 0.25% coverage. Diversity is
  the value, not standalone accuracy.

**Insight**: EARLY/LATE confusion is severe — features around pivots
look symmetric (volume spike, z-extreme appear both before and after).
B5 contributes signal diversity to the composite, not standalone power.

## Phase 5: Directional pivot — B6

**Question**: which DIRECTION will the next pivot flip?

- Built B6 (`train_b6_directional_pivot.py`) — 3-class NO/LONG/SHORT
  pivot incoming.
- Result: K=10 thr=0.70 → 53-57% precision per direction at 5% coverage.

**Insight**: by zigzag construction, next leg is opposite of current.
But knowing the DIRECTION the next pivot will form lets us pre-place
the right side of trade management.

## Phase 6: Composite cloud + trade-management experiments

Built `pivot_probability_cloud.py` to combine B1+B4+B5 into 8 action zones.

Tested trade-management interventions:

### Trail-tightening — FAILED
- Naive: -$44/leg, CI strictly negative
- Naive + B6 directional + hysteresis (sweep 18 configs): best -$0.29/leg, still negative
- **Structural reason**: R-trigger is structurally optimal for trail exits. Tighter trail trips on legitimate pullback noise.

### Target-placement — FAILED
- Fixed-distance target with R-trigger fallback (sweep 12 configs):
  all CIs strictly negative.
- **Structural reason**: any target before the actual peak is too early; target above peak never hits.

### Position sizing — WORKED ✓
- Hand-coded aggressive (zone × B6): +$626/day (+15.6%) on oracle entries.
- **The entry-time composite predicts leg AMPLITUDE, not direction.**

## Phase 7: B7 — GBM leg sizer

**Question**: can a GBM beat hand-coded sizing rules?

- Built B7 (`train_b7_leg_sizer.py`) — regressor on leg amplitude in R units.
- Result: MAE only 0.7% better than baseline. Pearson 0.22 — weak but monotonic.
- Sizing schemes tested:
  - `gbm_ev`: size = max(predicted_R - 1, 0) clipped to [0, 3]
- Result: **+$1,155/day vs hand_aggressive, CI [+$851, +$1,474]** — statistically significant. Oracle entries: $5,781/day.

**Insight**: weak MAE doesn't matter for sizing — monotonic ranking is
sufficient. B7 is the single highest-leverage model in the stack.

## Phase 8: Honest forward pass — R-trigger entries

**Question**: those numbers are oracle-entry. What's realistic with
indicator-based entries?

- Built `composite_forward_pass.py` (peeky version): $1,613/day on
  formula `leg_amplitude - 2R - friction`.
- Built `composite_forward_pass_hardened.py` (honest version): detect
  R-trigger fire on streaming 5s closes, use actual exit price.
- Result: $475/day flat, $927/day with gbm_ev sizing. Paired delta
  gbm_ev vs flat: +$452/day, CI [+$278, +$654].

**~50% reduction from peeky version** — the R-trigger entry/exit
"tax" gives back about half the oracle edge. But the architecture
remains positive and significant.

## Phase 9: Mode analysis — heavy-tail honesty

**Question**: mean of $927 might be inflated by right-tail days. What's typical?

- Computed histogram mode ($25 bin), KDE mode, GPD fit per scheme.
- Result: **KDE mode $468/day** for gbm_ev (vs mean $927).
- Days >$200: 24/31 (77%).
- Worst day: -$598. Best day: +$4,575.

**Insight**: strategy has heavy right tail. Mean is right-tail
inflated. Realistic typical-day = $400-500. Mean only achievable across
many days; individual days much more variable.

## Phase 10: Trying to fix bad days — FAILED

User asked: "can we hit 32/32 days >$200?"

- Tested hardened-exit sweep (SL + TP + daily cap × amp_min). 480 configs.
  **All hardening makes things WORSE** — bad days remain bad, mean drops.
- Built B8 (`train_b8_hour_risk.py`) — predict next 60-min P&L.
  OOS Pearson 0.22.
- Tested hour-gated forward pass. **Best: 25/31 days >$200, mean $1,505/day, min day -$838.**

**Conclusion**: 32/32 days >$200 isn't reachable with V2-feature-only
signals. Bad days look identical to good days at decision time. Would
need cross-day features (overnight, calendar, intermarket).

## Phase 11: Babysit deployment plan

Designed first-week recipe:
- Day 1-2: 0.25× size
- Day 3-5: 0.50× if normal
- Day 6-10: 0.75× scaling
- Day 10+: 1.0× if all checks passed

With manual pause triggers:
- 3 losing legs in 15 min
- Day cumulative <-$150 by leg 5
- B8 prediction <-$100 + already down

## What's untested vs what's validated

**Validated OOS**:
- Direction signal accuracy: 65.86% raw trend3 (matches indicator, no edge)
- B1-B6 pivot-related signals: AUC 0.69-0.74 OOS
- B7 monotonic calibration: predicted-amp_R ranks actual-amp_R
- Hardened sizing edge: +$452/day vs flat, CI strict positive
- Mode-of-day: $468 KDE mode, $627 median, $927 mean

**Untested / future work**:
- Live deployment (no live data yet)
- Day-level regime classifier (overnight/calendar features)
- Streaming engine validation (training_iso_v2)
- Partial exits / pyramiding
- Risk caps + Kelly sizing
- B7/B8 retraining on R-trigger-bar features

## Mental model for outside reviewer

Think of this as **two layers**:

1. **Indicator layer** (zigzag with R = 4×ATR): provides the entry/exit
   structure. Direction is given by leg alternation. R-trigger is the
   exit. Naked, this is ~$475/day on MNQ — modest positive expectation.

2. **GBM layer** (B7 amp + B8 hour risk): predicts which legs are big
   and which hours are good. Adds ~$450/day via sizing.

The GBM layer doesn't replace the indicator — it's a *multiplier* on
top of the indicator's natural P&L.

This is the cleanest mental model. Most of our experiments validated
or rejected sub-components of this architecture; the final shape is
what survived.
