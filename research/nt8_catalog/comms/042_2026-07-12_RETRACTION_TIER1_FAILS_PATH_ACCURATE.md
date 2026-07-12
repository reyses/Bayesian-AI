# RETRACTION — Tier-1 candidates FAIL the path-accurate backtest
**Doc:** 042 · **Date:** 2026-07-12 · **Author:** Claude (executor) · **Status:** FINAL
**Retracts:** the Tier-1 "tradable candidates" of docs 039/041.

## Path-accurate replay (real RTH 5s bar sequence, stop-first-in-bar, day-block CIs)
| Candidate | screen claim | path EOD | path 5m box | path 15m box |
|---|---|---|---|---|
| ROUND-05 ±20 | +5.4 / +6.6 SIG | −0.78 ns / +1.40 ns | −0.70 / +1.84 ns | −0.64 / +1.70 ns |
| PIVOT-16 flip T10/S20 | +8.1 / +9.1 SIG | −1.45 ns / +0.54 ns | −1.11 / +0.59 ns | −1.38 / +0.54 ns |
Every variant: CIs cross zero; 2024 mostly negative. **Both candidates are DEAD as
trades.** Tool: `tools/ag_phase5_backtest_path.py` (+ timebox variant inline).

## Root cause of the false screen
The realizable screen (doc 041) trusted the STORED MFE/MAE columns. The replay shows
stops firing ~5-10x more often than the stored MAE implies (PIVOT p95(mae)=8.0 vs
replay stop-rate ~35-40%): the stored excursions are measured from a different
reference/window than an actual entry at the trigger bar close (a semantics class
this audit hit repeatedly — docs 016/026/029). Compounded by the scratch-at-0
assumption. **Lesson (binding): stored-excursion screens are hypothesis generators
ONLY; nothing is called tradable except from bar-sequence replay.** The excursion
screen (ag_phase5_realizable_screen.py) is demoted to a pre-filter.

## What survives, honestly
- The catalog after full Phase-5: **zero validated tradable strategies.** Tier-2
  sub-friction drifts and the pierce-then-bounce anatomy REVERT TO HYPOTHESES until
  re-established path-accurately.
- What the effort produced that is real: the audited event datasets (24 dossiers,
  article-faithful, raw magnitudes), the aligned F-space extraction stack, the
  conversion-signature split (reversion family converts, trend family doesn't),
  the entry-discriminator ceiling result, and a validated review/verification
  process that caught every false positive BEFORE money — including this one,
  one hour after it was journaled and before any NT8 step.

## Process note
Moises ordered journal-first before this backtest precisely so a mid-session stop
wouldn't lose state. The same discipline means the retraction is documented an hour
after the claim. Journals corrected by APPEND (nothing deleted).

## Next options (Moises' call)
1. Pierce-then-bounce anatomy study, path-accurate from bar sequences directly
   (the two-phase level behavior may still be real — the SCREEN was the lie, the
   raw distributions still show the shape).
2. Close the nt8_catalog program at "no unconditional or level-trade edge; catalog
   = feature/context source for the main NMP/RL programs" and bank the tooling.
