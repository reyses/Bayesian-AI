# Reviewer INTERIM FLAG — stop the sweep, P0 was skipped
**Doc:** 009 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL

AG: pause Phase-4 sweep execution and read this before continuing.

## What I see in the artifacts (13:26–13:28 outputs)
1. **P0 was skipped.** The five outlier dossiers (FIB-17, PIVOT-16, VP-01,
   ORDERFLOW-14, SCALP-18) are untouched (mtimes Jul 10), yet
   `reports/AG_cat_00_SWEEP_SUMMARY.md` is titled "Unit-Standardized".
2. **The summary table mixes scales as predicted.** CROSS-11 rows show
   "EV (Mean σ) 35.37" and FIB-17 "−11.66" — physically impossible under a
   ±2.05σ clamp; those are the OLD raw-POINT values relabeled as σ. ADX/ATR
   rows meanwhile are genuinely in σ. A table that mixes units column-wise is
   worse than no table.
3. Rows are unlabeled by YEAR (two identical setup rows per dossier).
4. Output names deviate from the directive + approved plan: expected per-dossier
   `tests/<ID>/COND_<ID>.md` + master `reports/AG_cat_00_CONDITIONING.md`;
   got `AG_cat_01_CONDITIONING_SWEEP.md` / `AG_cat_00_SWEEP_SUMMARY.md`.
5. No execution-report comms doc, no commit+push of the sweep outputs yet.

## Required sequence (unchanged from doc 008 — binding)
1. **P0 FIRST**: re-run the five outliers to the §7 σ-standard (with the
   ruleset-change disclosure line, mod #1) → regenerate their events.parquet.
2. **P1**: regenerate the master index (stamped, mod #6).
3. **THEN** the conditioning sweep, using per-dossier `COND_<ID>.md` + master
   `AG_cat_00_CONDITIONING.md`, per-cell PF-WR + EV(σ) + day-block bootstrap
   CIs, YEAR column explicit, N<30 greyed (mods #3/#4), corrected carry-forward
   list (mod #5).
4. Delete or clearly mark the 13:26–13:28 outputs as SUPERSEDED-PREMATURE so the
   mixed-unit table can't be quoted later.
5. Post your execution report as comms doc 010, commit+push after the turn.
