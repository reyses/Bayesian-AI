# AI label length — MEASURED (the horizon n, grounded not guessed)
**Doc:** 076 · **Date:** 2026-07-15 · **Author:** Claude (executor) · **Status:** RESULTS
North star (Moises): build signals that improve DETECTION of the AI labels. Step 1: measure
the labels' length so the detector horizon n and target displacement are grounded in the
ground truth, not guessed. Source: DATA/ai_cusp_picks (golden auto-labeler v2).

## Numbers (576 days, 25,680 labels, ~47/day median; 50/50 long/short)
| quantity | mode | median | mean | p25 | p75 | p90 |
|---|---|---|---|---|---|---|
| **DURATION (min)** | **12** | **21** | 28 | 12 | 36 | 57 |
| DURATION (5s bars) | 113 | 252 | 336 | 148 | 435 | 679 |
| **DISPLACEMENT (pts)** | **21** | **37** | 54 | 22 | 66 | 112 |

Duration histogram (min): 5-10:3762 · 10-15:4527 · **15-30:8520** · 30-60:6263 · 60-120:2053
(<5 min: ~390 total; the labeler almost never marks a sub-5-min trade.)

## Read (grounds every detector's horizon)
1. **A "good trade" lasts ~12-21 min (mode-median), displaces ~21-37 pts.** THIS is the
   horizon n and the target size — measured, not guessed. The earlier "2-3 min ambient
   period" (recovery_dynamics) is the OSCILLATION clock; a LABEL is ~7-10x longer -> a
   label is precisely a move that OUTLASTS the ambient clock. Confirms the persistence
   definition: a label = a leg that trends past the return-clock.
2. **~47 labels/day** but that is the ORACLE ceiling (best-bar entries, both directions,
   overlapping). A real detector firing 6-8 actionable/day is aiming at a SUBSET of these.
3. **Detector horizon should be set to the label scale**: measure "right direction" over
   ~15 min / ~250 bars (the label median), NOT a fixed 60s or the 2-3 min ambient clock.
   The doc-072 ADX candidate (SMA 84 = 7-min window) is well inside this; good.
4. **Displacement mode 21 pts** ~ matches a zigzag ATR(14)x4 leg scale -> confirms the
   zigzag-leg == label connection (doc 075). Calibrate zigzag ATR-mult so leg size ~ 21-37 pts.

## Next
- Set the detector persistence horizon n = label median (~15 min / 250 bars) as the default
  "did it trend" window for the directional-accuracy metric.
- Then: measure each detector's DETECTION overlap with labels (does an ADX/zigzag signal sit
  within a label window) + directional agreement -> the "improve detection of AI labels" goal.
Tool: tools/measure_ai_label_length.py (inline this turn; save on next touch).
FPS untouched.
