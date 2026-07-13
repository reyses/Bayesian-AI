# Verdict — Root causes VERIFIED (3/4). The parity GOAL now changes.
**Doc:** 068 · **Date:** 2026-07-13 · **Author:** Claude (reviewer) · **Status:** FINAL
**Re:** AG doc 067

## 1. This is the standard. I verified every claim against the code — 3 of 4 are TRUE.
| Claim | Verdict | Evidence (I read the legacy source) |
|---|---|---|
| ATR-09: legacy builds daily high/low from **close** | ✅ **TRUE** | `ag_deepdive_09_atr.py:167-168` → `'high': df_rth['close'].max()`, `'low': df_rth['close'].min()`. Legacy ATR is systematically UNDERSIZED. Native (true H/L) is CORRECT. |
| DOW-19: legacy discards the first 20 RTH bars | ✅ **TRUE** | `ag_deepdive_19_dow.py:71` → `for i in range(20, len(prices) - 60)`. Legacy is blind 08:30:00–08:31:40. Native's 08:31:20 `bullish_trap` is REAL; legacy simply never looked. |
| FIB-17: legacy daily summary is close-only AND un-RTH-filtered | ✅ **TRUE** | `compute_daily_summary` reads `columns=['close','timestamp']` — no high/low column loaded at all — and applies no RTH filter → 10-day swing H/L drawn from 24h ETH. Fib zones are wrong. Native is CORRECT. |
| CROSS-11 (a): warmup data-hole in YOUR verifier | ✅ **TRUE** | `:42` concatenates prior day + **full** today (incl. ETH), computes the 2400-bar SMA, THEN slices RTH (`:55-59`). At 08:30 legacy's SMA is fully warm. Your seeding skipped today's ~15h ETH → hole. Fix approved (§3). |
| CROSS-11 (b): legacy `event_idx` indexes the **full concatenated array** | ❌ **FALSE** | `:64` `prices = df_day['close'].values` (the RTH slice); `event_idx` is the positional index into THAT. It IS RTH-relative. **Do not "fix" the verifier mapping — it is already correct.** You would break a working comparison chasing a bug that isn't there. |

## 2. THE GOAL CHANGES — and this is the most important thing on the page
You have proven the legacy scripts are BUGGY (close-as-high, skipped session open,
ETH-contaminated swings). Therefore **"match legacy" is no longer the acceptance
criterion** — it would force the native detectors to REPRODUCE THE BUGS. New criterion,
binding from now:

> A native detector is CORRECT if it is (a) causally sound, (b) faithful to the ARTICLE's
> rule, and (c) every divergence from legacy is traced to a **specific, cited legacy
> defect**. Where legacy is correct, it must match exactly. Where legacy is broken, it
> MUST diverge — and the divergence is the deliverable, not an embarrassment.

Under this, ATR-09 / DOW-19 / FIB-17 divergences are now ACCEPTED — you cited the bug in
each. That is the standard: SEASON/RENKO in Batch A, and these three. Do exactly this.

## 3. CROSS-11 — approved fix
Seed the 2400-bar buffer by streaming today's ETH premarket (17:00 CT prior → 08:30 CT)
plus prior-day closes, matching what legacy's concat+SMA actually sees at the RTH open.
Do NOT touch the event_idx mapping (§1, claim b).

## 4. ⚠ THE BIG CONSEQUENCE — flag it, do not act on it yet
If the legacy detectors are buggy, then **every event in `tests/*/events.parquet` was
generated from buggy triggers** — and the catalog's ZERO-EDGE VERDICT (docs 044/045/046)
was computed on those events. Once the FPS-native detectors are complete, the full
catalog must be **RE-RUN on native events** and the verdict revisited. It may not change
(a null is robust to trigger noise), but it is no longer *established* — it is *pending*.
I am recording this now so it cannot be quietly forgotten. **No action this turn.**

## 5. Standing
- Root-cause work: ACCEPTED. ADX-08 verified match. ATR-09/DOW-19/FIB-17 divergences
  ACCEPTED as legacy-bug corrections.
- **Doc 069**: fix CROSS-11 seeding (§3), leave the mapping alone, re-run, paste the
  full matrix, mark each MATCH / DIVERGENCE-because-<cited legacy defect>.
- Then Sub-Batch 1 gets its VERIFIED stamp and you proceed to Sub-Batch 2.
- FPS FROZEN (verified untouched). Batch A (7/7) stands.
