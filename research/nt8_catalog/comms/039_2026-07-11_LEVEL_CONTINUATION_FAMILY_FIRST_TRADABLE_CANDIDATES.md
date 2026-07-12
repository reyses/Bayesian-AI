# The Level-Continuation Family — first both-year tradable candidates
**Doc:** 039 · **Date:** 2026-07-11 · **Author:** Claude (executor) + Moises (visual inspection) · **Status:** FINAL

Moises inspected the 20-test gallery and flagged PIVOT-16 ("clear, just flip the
signal") and ROUND-05 ("clear winner"). Both check out quantitatively — and they are
the SAME mechanism from two independent dossiers.

## 1. PIVOT-16 FLIPPED (breach-continuation at S1/R1)
Article claims bounce; reality: the touch continues through the level ~86% of the
time. Flip = sell the S1 touch / buy the R1 touch, take-profit at the old stop
distance (+10..12), DISASTER STOP (simulated properly off stored MFE = flip's
adverse excursion; stop-fill assumed at level, no gap slippage):
| stop | 2024 EV (day-CI) | 2025 EV (day-CI) | WR | mode |
|---|---|---|---|---|
| 20 | **+8.07 [+6.00,+10.04]** | **+9.05 [+6.80,+11.20]** | .86/.87 | +11/+12 |
| 30 | +6.81 [+4.17,+9.19] | +7.97 [+5.31,+10.50] | .86/.87 | +11/+12 |
| 50 | +4.81 [+1.27,+8.12] | +6.30 [+2.46,+9.81] | .86/.88 | +11/+12 |
Robust to stop choice; ~1.1 events/day; worst loss = the stop, exactly.

## 2. ROUND-05 (already the breach-continuation dossier post-audit)
Its DOC "EV" was peak-MFE (unrealizable). Rebuilt as a REAL trade from stored
MFE/MAE with the WORST-CASE rule (if both target and stop were touched in the
window, count it as STOPPED):
| target/stop | 2024 EV (day-CI) | 2025 EV (day-CI) |
|---|---|---|
| +10/−10 | +2.21 [+1.16,+3.18] | +1.19 [+0.00,+2.33] |
| +10/−20 | +3.84 [+2.75,+4.84] | +3.70 [+2.33,+5.02] |
| +15/−15 | +4.07 [+2.79,+5.35] | +3.24 [+1.72,+4.89] |
| +20/−20 | **+5.43 [+3.95,+6.90]** | **+6.61 [+4.76,+8.37]** |
Significant BOTH years in ALL four configs under worst-case ordering. ~1 event/day.

## 3. Why this is believable (and what it is)
- Two INDEPENDENT dossiers, same mechanism: **price reaching a watched level
  (floor pivot, 00/50 round number) tends to continue THROUGH it** — the inverse
  of the articles' bounce claims, and exactly the liquidity/stop-cascade thesis
  (AUDIT-ACC-01 flagged ROUND-05's original bounce test as opposite-of-article).
- Both-year day-block significance, parameter-robust, worst-case assumptions,
  no F-space model in the loop (these are UNCONDITIONAL — the discriminator is
  optional icing, not load-bearing).
- Moises' visual read found it; the numbers confirmed it. Distributions-first works.

## 4. Honest sizing + caveats (do not oversell)
- MNQ $2/pt: PIVOT-16-flip ≈ +$16-18/day/contract; ROUND-05 ±20 ≈ +$11-13/day.
  Real but small per contract; friction (~1-2 ticks RT) eats ~10-20% of it.
- Stop fills assumed clean (no gap-through slippage); 5m/60-bar windows only.
- Both-hit ordering unknown for ROUND-05 → worst case used (real EV likely higher).
- NOT deployed, NOT NT8-tested; next gate per house rules = forward SIM parity.
- ATR-09 buy/sell check (Moises' question): dossiers ARE two-sided; per-side
  models did NOT change ATR-09's result (disasters split across both sides) —
  the pooling concern was real in principle but not the binding issue there.

## 5. Next
1. VWAP-03 + MACD-07 get the same realizable-trade + disaster-stop treatment
   (same reversion family, conversion signatures already visible).
2. PIVOT-16-flip + ROUND-05 → proper event-level backtest through the offline
   forward pass (path-accurate stops), then SIM.
3. F-space discriminator as an OPTIONAL filter on top (doc-027), not a gate.
