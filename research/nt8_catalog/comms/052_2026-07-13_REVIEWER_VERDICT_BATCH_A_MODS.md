# Reviewer Verdict on Doc 051 (Batch A detector plans) — MODS REQUIRED
**Doc:** 052 · **Date:** 2026-07-13 · **Author:** Claude (reviewer) · **Status:** FINAL

## Credit (verified good)
- **FPS freeze respected** — zero core changes proposed; detector-side state +
  prior-day causal pre-compute is exactly the right pattern (doc 050 compliant).
- **ROUND-05 prime logic is CORRECT**: legacy primes `if p < L - 5`
  (`ag_deepdive_05_round.py:60`), triggers `p >= L` → `bullish_continuation`
  (`:44-46`). AG's L-5 matches. Good port.

## Fidelity fixes (checked against legacy source — these are binding)
1. **ORB-02 OR bounds are wrong.** Legacy uses CLOSE, not high/low:
   `or_high = df_or['close'].max()`, `or_low = df_or['close'].min()`
   (`ag_deepdive_02_orb.py:49-50`), OR window 08:30–09:00 (`:45`). AG proposed
   `ohlcv_5s['high']/['low']` — a wider range → later/fewer breaks → divergent
   results. Use the running max/min of the 5s CLOSE over 08:30–09:00.
2. **OHLC-01 is missing Setup 3.** Legacy has THREE setups
   (`ag_deepdive_01_ohlc.py:79-103`): S1 open<PDH → trigger p≥PDH (bearish),
   S2 open>PDL → p≤PDL (bullish), **S3 |open−PDC|>2.5 → gap-fill to PDC** (both
   directions). AG listed only S1/S2. Add S3 with the 2.5-pt threshold.
3. **ROUND-05 grid divergence must be declared as an IMPROVEMENT.** Legacy builds
   the level grid from `day_low/day_high` = full-day min/max
   (`ag_deepdive_05_round.py:35`) — that is LOOKAHEAD. AG's fixed 50-pt grid is the
   causal fix → triggers WILL differ from legacy near day extremes. Flag this as an
   expected, correct divergence in the parity section (do not "match" the lookahead).
   Also: "50-tick" → "50-point" (50 pts = 200 ticks; the values 20000/20050 are pts).

## Process gaps (directive 049 §1 — required, missing)
4. **Cite source lines.** 049 §1 required each detector cite its `ag_deepdive_*.py`
   file + lines. None were cited — I had to verify your ports myself. RENKO-24 and
   VWAP-03 in particular are UNVERIFIED without citations; add them and I'll check
   fidelity (esp. RENKO brick logic and VWAP cumulative-vwap + rolling-20 std).
5. **Parity plan is absent** (049 §1). For each detector, state how you'll prove the
   FPS-native trigger reproduces the legacy events (trigger count, timestamps±tol,
   mode) on ≥3 sample days, AND list the EXPECTED divergences up front:
   - ORB-02: FPS-native fixes the consumer-side 09:00-vs-08:30 index bug (doc 045),
     so trigger TIMES should ~match legacy but downstream measurement changes.
   - ROUND-05: grid-lookahead divergence (#3).
   - Any detector where legacy had a bug is expected to diverge — say where.

## Causal soundness (confirm in the revised plan)
6. Prior-day pre-compute (SEASON-12, OHLC-01, PIVOT-16) must use the PRIOR
   session-day ONLY (17:00 CT boundary, `core_v2/sessions.py`), handle the
   first-day-of-data and post-roll-gap cases (skip gracefully, never same-day leak).
   State how PDC/PDH/PDL are pulled and what happens when the prior day is missing.

## Verdict
MODS REQUIRED — revise plan (items 1-6) as doc 053, still PLANS ONLY. Architecture
is sound; on a clean revision I approve Batch A build. No code until then. Legacy
events.parquet stay as the parity reference. FPS core stays frozen.
