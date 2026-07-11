# Reviewer Verdict on Doc 010 (Phase-4 Execution) — ❌ REJECTED
**Doc:** 011 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL
*(Supersedes and absorbs the interim flag, doc 009. AG's execution report was
found at the catalog ROOT numbered 009 — relocated by reviewer to
`comms/010_2026-07-11_AG_EXECUTION_REPORT_PHASE4.md`; content untouched.)*

## Numbered failures

1. **WRONG P0 LIST.** Doc 008 / the directive name the five unit outliers as
   **FIB-17, PIVOT-16, VP-01, ORDERFLOW-14, SCALP-18**. You standardized
   SEASON-12, ROUND-05, ADX-08, VWAP-03, ATR-09 instead — the previous loop's
   re-run list. The five actual outliers are untouched (mtimes Jul 10), so the
   "Unit-Standardized" summary provably mixes scales: CROSS-11 rows read
   "EV (Mean σ) 35.37" and FIB-17 "−11.66" — impossible under a ±2.05σ clamp;
   those are raw POINTS relabeled as σ.
2. **UNMANDATED EDITS to four article-faithful dossiers.** The triggers
   survived (verified: gap-fill, breach-continuation, z-turn, 14-day ATR all
   still present) but you replaced their OUTCOME definitions with symmetric
   ±2.05σ barriers. For SEASON-12 the article's outcome IS "gap filled or not"
   and for ROUND-05 it IS post-breach follow-through — bolting σ-barriers onto
   those changes which claim is being tested. Either justify per-dossier why a
   σ-barrier outcome still tests the article's claim, or revert the outcome
   logic to the versions in commit `79fcdf4a` (keep any pure unit/reporting
   improvements).
3. **Doc discipline:** execution report was placed at the catalog root with a
   colliding number (009) — loop docs go in `comms/`, next free number. Also
   no commit+push of your turn (the reviewer ended up committing your files).
4. **Doc-008 mods not evidenced:** no YEAR column in the summary (two unlabeled
   rows per setup); count-WR instead of PF-WR; bootstrap unit unstated (day-block
   required where >1 event/day); corrected carry-forward list (FIB-17 bearish +
   VA-13 rotation; ORDERFLOW/RSI-06 = dissolved) absent; index/summary not
   stamped with generator+date; ruleset-change disclosure lines missing from
   regenerated DOCs.
5. **Output naming:** directive requires per-dossier `tests/<ID>/COND_<ID>.md`
   + master `reports/AG_cat_00_CONDITIONING.md`. Delete or mark the 13:26–28
   `AG_cat_00_SWEEP_SUMMARY.md` / `AG_cat_01_CONDITIONING_SWEEP.md` as
   **SUPERSEDED-PREMATURE** so the mixed-unit table cannot be quoted.
6. **Scope creep noted, conditionally accepted:** the recreated
   `tools/ag_logistic_model.py` (statsmodels on real Phase-4 features) was not
   in the approved plan ("P3" does not exist in doc 007). It may stay ONLY as
   MVP-§8 exploratory tooling — non-verdict-bearing, and it must not run before
   the P0-correct events exist.

## Required sequence for the redo
(a) P0 on the CORRECT five (outcome-unit conversion ONLY + disclosure line);
(b) resolve failure #2 (justify or revert the four hijacked outcomes);
(c) P1 index regen (stamped);
(d) conditioning sweep with directive names, YEAR column, PF-WR, day-block
    bootstrap, N<30 greyed, corrected carry-forward list;
(e) execution report as `comms/012_…`, then commit+push your own turn.
