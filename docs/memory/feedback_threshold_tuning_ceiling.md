---
name: Exit-threshold tuning has a ceiling — entries are the bottleneck
description: 2026-05-04 — adaptive exit thresholds (per regime, per tier, or per regime×tier) all give ~$28/day OOS uplift. Cell granularity is a wash. To break past, fix entries, not exits.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The rule

Adaptive exit thresholds doubled baseline OOS PnL ($27 → $55/day, day-WR
51%→57%) — but the uplift saturates around **+$28/day** regardless of cell
granularity (regime-only, tier-only, or tier × regime). To go further, the
lever is **entries**, not exits.

## Evidence (2026-05-04 V2-native pipeline)

Same 68 OOS days, MA_ALIGN + REVERSION strategies (VEL_BODY_CHORD removed),
Bayesian-derived thresholds varying only by cell-granularity choice:

| variant | $/day | total | day-WR | 95% CI on delta vs baseline |
|---|---:|---:|---:|---|
| baseline (no thr) | $27.43 | +$1,865 | 51% | reference |
| regime-only (6 cells) | $54.82 | +$3,728 | 57% | [-$5.29, +$62.89] |
| **tier-only (3 cells)** | **$56.99** | **+$3,875** | 57% | [-$4.56, +$66.78] |
| tier × regime (18 cells) | $55.57 | +$3,779 | 57% | [-$4.71, +$61.92] |

Pairwise differences between groupings: all <$2/day. The framework is
robust; the ceiling is structural.

**95% CI lower bound stays around -$5 across all three variants.** Just below
statistical significance at n=68 OOS days. The day-WR bump (51→57%) is
reliable — the dollar magnitude is too small to clear noise.

## Why this happens

The legacy V1 system had similar behavior: exit improvements helped
incrementally, but the dollar lift came from CNN flip prediction (70.6%
direction accuracy) and adding new entry types (KILL_SHOT, CASCADE,
PEAK_ExNMP). Exits trimmed losers; entries created winners.

In V2-native land, only **2 strategies** are active (MA_ALIGN + REVERSION).
Their entries cover narrow patterns:
- MA_ALIGN: 7-of-8 vwap_w alignment (~20% of 5m bars)
- REVERSION: |z_se_w|≥1.8 (extreme stretches)

A lot of intermediate market behavior — moderate trends, breakouts,
exhaustion bars, absorption, multi-TF momentum exhaustion — is uncovered.

## How to apply

1. **Don't tune exits past +$28/day-uplift expectation.** It's a ceiling,
   not a starting point.
2. **The bottleneck is entries.** Three levers to break past:
   - Train CNN as filter+entry (rejects bad entries, spawns CNN-originated)
   - Add more strategies (EXHAUSTION, BREAKOUT_4H, COMPRESSION, etc.)
   - Apply contextualizer (sign-flip rules conditional on modifier quantile)
3. **Production config**: `training_v2/output/thresholds_prod.json`
   (per-tier Bayesian-derived). This is the locked exit configuration; we
   move to the entry side next.
