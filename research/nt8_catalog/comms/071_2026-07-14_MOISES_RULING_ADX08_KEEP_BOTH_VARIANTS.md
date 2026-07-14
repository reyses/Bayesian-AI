# MOISES RULING — ADX-08: KEEP BOTH (SMA and Wilder as sibling variants)
**Doc:** 071 · **Date:** 2026-07-14 · **Author:** Claude (reviewer), ruling by Moises · **Status:** BINDING

## 1. The ruling
Do not choose. **Keep both.** ADX-08 becomes two first-class sibling concepts:

| Variant | Smoothing | Rationale |
|---|---|---|
| **ADX-08-SMA** | legacy `rolling(168).mean()` on +DM/-DM/TR and DX (`ag_deepdive_08_adx.py:56-60`) | preserves comparability with the audited event population and every prior ADX-08 result |
| **ADX-08-WILDER** | canonical Wilder RMA (α = 1/14) | the ADX the article actually means; legacy itself concedes SMA is an approximation "for speed" |

Both get a detector, both get verified, both are carried into the catalog re-run as
SEPARATE concepts. Neither is a flag-default of the other — they are siblings.

## 2. Why this is the right call (not a fudge)
The dispute was definitional: *"which is the real ADX?"* — an argument nobody can win
from an armchair. Running both converts it into a measurement: **does the smoothing
choice change the outcome?** If ADX-08-SMA and ADX-08-WILDER produce materially different
event populations and different verdicts, that is itself a finding about how fragile this
concept's "edge" is to an implementation detail. If they agree, the question is closed.
We get an answer instead of an opinion.

## 3. GUARDRAIL — this is NOT a licence to fork every implementation choice
Sanctioned here because the legacy code **explicitly flags its own approximation**
(`# Use SMA approximation for speed`) — an acknowledged shortcut on a canonical indicator.
It is **not** precedent for forking a variant every time you dislike an implementation.
Any further variant requires a fresh ruling. The catalog does not get to breed.
(And it does NOT reopen CROSS-11 — see §4.)

## 4. CROSS-11 ruling STANDS — restore first-cross-only
Unchanged from doc 070. `# Scan for first cross` is the RULE, not a defect. One setup per
day. Restore it. Do not fork a variant here: there is no acknowledged approximation, only
your disagreement with a design choice, and that is not sufficient grounds.

## 5. Required (doc 072)
1. `ADX08_SMA_Detector` and `ADX08_Wilder_Detector` — both in `batch_b_detectors.py`,
   both registered, both in the verification matrix.
   - ADX-08-SMA must match legacy on ~all days (it IS the legacy computation). If it
     doesn't, that residual is a real finding — report it.
   - ADX-08-WILDER is expected to diverge (~76 days on your run). Report the divergence
     count and, on 2-3 sample days, WHY (paste both ADX values at the divergent bar).
2. CROSS-11: first-cross-only restored; re-run and report (I expect near-total match now
   that seeding is fixed — if not, THAT is the finding).
3. ATR-09 / DOW-19 / FIB-17: unchanged, already accepted.
4. Paste the full matrix. Status = `EXECUTED — AWAITING VERIFICATION`. The stamp is mine.

FPS FROZEN. Batch A (7/7) stands. No Sub-Batch 2 until Sub-Batch 1 carries my stamp.
