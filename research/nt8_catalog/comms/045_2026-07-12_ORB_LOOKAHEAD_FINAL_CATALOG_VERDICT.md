# ORB-02 was lookahead — FINAL catalog verdict: zero
**Doc:** 045 · **Date:** 2026-07-12 · **Author:** Claude (executor) · **Status:** FINAL
**Retracts:** the ORB-02 candidate (docs 044) — the catalog's last survivor.

## How it was caught — Moises' no-stops ruling did it
The exploration-level horizon run (no stops/targets, raw drift per Moises'
directive that management must not censor exploration) showed ORB-02 with
**%>0 = 1.00 at +30m in BOTH years** — a 100% win rate, which is impossible.
The barrier version had MASKED this tell (truncation hides distribution shape);
the naked distribution exposed it in one glance.

## The bug (third variant of the index-space class)
`ag_deepdive_02_orb.py` computes `event_idx` inside its **09:00–15:15 session
slice** (it alone slices at 09:00; verified all other 23 dossiers slice 08:30).
The runner/explorer mapped indices against the 08:30 RTH array → every ORB entry
landed **exactly 30 minutes BEFORE its true trigger**, inside the opening range,
with `mode` = the breakout direction that only happens later. Mechanical
lookahead; +30m %>0=1.00 by construction.

## True ORB-02 numbers (offset corrected, +360 bars)
| horizon | 2024 | 2025 |
|---|---|---|
| 5m | +0.17 [−2.96,+3.40] | −3.41 [−8.57,+1.88] |
| 15m | +1.76 [−3.41,+6.85] | −0.82 [−9.69,+9.13] |
| 30m | +6.44 [+0.11,+12.90] | −4.87 [−14.82,+4.97] |
| 1h | +8.02 [−0.35,+16.89] | −2.05 [−14.77,+10.85] |
%>0 ≈ 0.48–0.55 everywhere. 2025 negative throughout. DEAD.

## FINAL CATALOG VERDICT
- Management level (FPS barrier run, 444k trades): ORB was the only gate-passer
  → now retracted → **zero candidates**.
- Exploration level (horizon run, 55,469 events, no censoring): ORB held the only
  both-year drift cells → now retracted → **zero both-year drift anywhere**.
- **All 24 NinjaTrader catalog concepts: no realizable edge, no raw post-event
  drift, both years, canonical engine.** The catalog is a closed, fully honest
  null — its value is the audited event datasets, the FPS tooling (128-156k
  bars/s), the conversion-signature taxonomy, and six caught artifact classes
  now documented as institutional scar tissue (index-space × 3, stored-excursion
  semantics, unrealizable-peak, label-definition leakage).

## Standing corrective actions
1. Dossier scripts must export `entry_ts` (bar timestamp at trigger) so no
   consumer ever maps indices across slice conventions again. (AG-queue item.)
2. `%>0 = 1.00` (or any impossible perfection) is now a pre-registered AUTO-FAIL
   tell in every future screen — check the distribution before the mean.

## Process record
Caught within ~2 hours of the claim, by the exploration-level view Moises
insisted on. Every layer of today's stack (stops removed → distribution visible
→ impossibility flagged → script audited → offset proven → retraction) is in
comms/043-045 with artifacts. Zero false claims survive in the record.
