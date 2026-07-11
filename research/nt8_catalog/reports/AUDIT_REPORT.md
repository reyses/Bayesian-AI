# Phase 3B Folder Discipline & Protocol Audit
**Date:** 2026-07-10

This document contains an audit of the `nt8_catalog` directory against the active Phase 3B Joint Bayes directives (`AG_PHASE3B_JOINT_BAYES.md`) and the Master Validation Protocol (`MASTER_VALIDATION_PROTOCOL.md`).

## 1. Folder Discipline Violation (Migration Incomplete)
The directives mandate a strict **"Test Dossier" Architecture**: every concept must be isolated in its own self-contained folder inside `tests/` (e.g., `tests/VWAP-03_Session_VWAP/`), which should include the script, traces, reports, and graphical assets. 

**Finding:** The migration to this architecture is incomplete. While 18 subdirectories exist under `tests/`, many of the execution scripts and markdown reports are still located in the legacy flat `tools/` and `reports/` directories. For example, `tools/ag_cat_01_vwap_pullback.py` and `reports/AG_cat_01_VWAP_Pullbacks.md` exist separately from the `tests/VWAP-03_Session_VWAP` directory, directly violating the instruction to avoid flat shared directories.

## 2. Copy-Paste Errors in the Master Index
In `reports/AG_cat_00_INDEX.md`, the master summary table for the Phase 3B evaluation contains copy-paste errors. Under the **"what it measures"** column, every single row—including APZ, Squeeze State, Candle Shapes, and MA Crossover—incorrectly lists `"VWAP Touch"`.

## 3. Master Validation Protocol Deviations (e.g., VWAP-03)
A review of the test dossier for `tests/VWAP-03_Session_VWAP` shows that while it successfully avoids the RTH and magnitude lookahead bugs, it still violates core statistical rules flagged in `REVIEW_AG.md`:
- It continues to use the simple empirical Win Rate rather than the required PF-based WR.
- It fails to properly output the Mean with the 95% Bootstrap CI (though `DOC_03_Session_VWAP.md` displays an EV 95% CI, indicating a discrepancy between the report and its review).

## 4. Joint Bayesian Model State
The Phase 3B Joint Bayesian Logistic Regression has been run (`tools/ag_joint_bayes_model.py`) and recorded in `reports/AG_Joint_Model.md`. The model trained on 82,102 events and shows significant lift in the top posterior decile (+26.30 pp over the base rate). However, keeping these files in the top-level `reports/` and `tools/` folders technically violates the strict self-contained folder rule.

## 5. Untested Concepts Still Pending
Per `TESTED_VS_UNTESTED.md`, several key concepts (Volume Profile POC, 30-min Opening Range Break, Seasonality) were incorrectly bucketed with dead strategies and remain completely untested. These require isolated runs to evaluate them as gates on the 9-13 CT niche.
