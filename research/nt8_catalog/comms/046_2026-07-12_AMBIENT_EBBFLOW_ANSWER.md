# Ambient ebb/flow answer — catalog events are AMPLITUDE markers, not direction
**Doc:** 046 · **Date:** 2026-07-12 · **Author:** Claude (executor), question by Moises · **Status:** FINAL

## Moises' question
Unstopped-to-EOD, every entry shows "good MFE" from the day's oscillation (his
recovery_dynamics: 71% revert <15m, 91% oscillation, price returns to anchor even
in trends). Are the catalog events just landing in ordinary ebb/flow?

## Measurement (day-matched, then hour-matched ambient baselines)
Ambient 15m MFE (any anchor, direction-neutral): median ~20-22 pts = the free
oscillation. Events vs WHOLE-DAY ambient: most dossiers ~2x ambient. But events
cluster in high-vol hours → re-matched by SAME DAY + SAME HOUR:

| dossier | excess vs hour-matched ambient (15m) |
|---|---|
| ORB-02 | **+13.9 [+10.6,+17.5]** |
| ROUND-05 | +9.0 [+5.9,+12.1] |
| VWAP-03 | +8.7 [+5.2,+12.2] |
| RSI-06 | +7.8 [+4.4,+11.3] |
| VA-13 / VWMA-10 / VP-01 / MACD-07 | +5.2..+6.5 (all CI>0) |
| PIVOT-16 / SQZ-04 / OHLC-01 / HNS-22 / ZONE-21 | ≈ ambient |
| SAR-23 / TUNNEL-20 / DOW-19 | +0.5..0.7 (tiny, huge N) |
| ATR-09 | -1.4 (fully explained by time-of-day) |

## Answer + reading
1. **No — the surviving family does NOT sit in ordinary ebb/flow.** After removing
   the diurnal confound, the reversion/level/divergence family marks moments where
   the next 15 minutes breathe +5..+14 pts MORE than that hour's normal — with the
   DIRECTION a coin flip (signed drift ~0 everywhere; doc 045 verdict stands).
2. **The catalog signals are volatility-TIMING features, not directional trades.**
   This coheres with everything: the conversion-signature family split (docs 038),
   recovery_dynamics' "breathing amplitude regime", and the B10 vol-regime lineage.
   Their honest value = amplitude/timing CONTEXT for the main system (sizing,
   regime gates), not standalone entries.
3. **Remaining confound (named, untested): volatility clustering.** Events select
   locally-moving moments; GARCH alone predicts some inherited amplitude. Decisive
   next drill: trailing-vol-MATCHED ambient (compare events only to same-day
   anchors with similar trailing 15m amplitude). If excess survives THAT, these
   are genuine amplitude-expansion predictors beyond persistence.

## Program status
Catalog close-out thesis refined: zero directional edge (doc 045) + a real
direction-free amplitude-marking family → the catalog's product is a set of
volatility-timing context features for the NMP/RL line, pending the vol-matched
confirmation.
