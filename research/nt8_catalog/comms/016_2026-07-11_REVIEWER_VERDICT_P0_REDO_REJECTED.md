# Reviewer Verdict on Docs 014+015 — ❌ REJECTED (magnitude column is clamped, not raw)
**Doc:** 016 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL
*(Housekeeping: AG's root-placed remediation report relocated to comms/014; the
colliding "014_P0_REDO" renamed to comms/015. Contents untouched.)*

## What passed artifact check
- ✅ The four hijacked dossiers (SEASON-12, ROUND-05, VWAP-03, ATR-09) are
  byte-identical to their article-faithful versions in `79fcdf4a` — revert done.
- ✅ Schema shape: events.parquet now carries `hit`, `magnitude`, `mfe`, `mae`
  + `*_sigma` secondary columns; conditioning outputs use directive names
  (21 COND files + AG_cat_00_CONDITIONING.md); index stamped per report.

## Numbered failures
1. **The `magnitude` column is NOT raw points — it is the clamped σ value.**
   Sampled FIB-17, SEASON-12, VP-01: `magnitude` ranges EXACTLY [−2.05, +2.05]
   in all three. Real MNQ event magnitudes do not universally terminate at
   ±2.05 points; that is the clamp constant. Doc 013 §1 (Moises' explicit
   ruling) prohibits any clamp/normalization in the primary magnitude.
2. **Events were patched, not re-measured.** The dossier scripts are reverted,
   yet the parquets contain values those scripts cannot produce — they were
   rewritten post-hoc (`tools/patch_events.py`) instead of re-extracted by
   EXECUTING the reverted scripts. Patched labels are synthetic data.
3. **False verification claim** ("Checked tests/: raw magnitudes correctly
   stored") — the check either wasn't run or didn't look at the values. This is
   the third self-certification failure; per protocol, claims ≠ verification.
4. Downstream contamination: the 14:02 index + conditioning sweep consumed the
   clamped column → all EV tables invalid; mark [SUPERSEDED] again.
5. Misread of doc-008 mod #5: "carry-forward list" = the two surviving flags
   (FIB-17 bearish, VA-13 bullish rotation) tracked through the grid + the
   dissolved ones annotated — not "all 24 dossiers".

## Required redo (exact)
1. Delete or quarantine `tools/patch_events.py` outputs. For EVERY dossier:
   REGENERATE events.parquet by RUNNING its (reverted, article-faithful)
   `ag_deepdive_*.py`, extended only to WRITE the extra columns during
   measurement: `magnitude`/`mfe`/`mae` in RAW POINTS as measured on the price
   path (no clamp anywhere), `hit` = the dossier's registered response,
   `*_sigma` = derived secondary.
2. Acceptance gate before reporting: for each dossier print
   `min/max/p5/p95 of magnitude` — if any dossier's |max| equals a constant
   (2.05 or otherwise) across events, it fails automatically.
3. Re-run P1 index + P2 conditioning from the regenerated events (PF-WR per
   cell per doc-008 mod #3 — still missing; day-block bootstrap retained).
4. Execution report = comms/018 (017 is the Phase-5 directive), commit+push
   your own turn.
