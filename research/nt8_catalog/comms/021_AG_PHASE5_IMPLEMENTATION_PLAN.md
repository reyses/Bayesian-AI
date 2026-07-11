# Phase-5 Implementation Plan: F-space Binary Logistic
**Doc:** 021 · **Date:** 2026-07-11 · **Author:** Antigravity

## 1. Objective
Implement the Phase 5 F-Space Logistic Regression evaluation using telescoping fractal ladders. This plan adheres strictly to Doc-017, generating three distinct models (Entry, Pre-Exit, Post-Exit) for each surviving/tracked edge, evaluating out-of-sample (2024 -> 2025) predictability via stepwise degree expansion.

## 2. Telescoping Ladder Construction
The feature matrix will contain ~23 temporal slots per anchor.

**Anchors (3 Independent Models):**
- **`PhE_`**: Entry (ends at $t_e$, backward looking)
- **`PhXit_`**: Pre-Exit (ends at $t_x$, backward looking)
- **`PhPost_`**: Post-Exit (starts at $t_x$, forward looking - diagnostic only)

**Slots (23 total):**
- **1s Tier (5 slots)**: Immediate micro-structure (interpolated/extracted from base 5s where necessary). *Features: micro-return, micro-range.*
- **5s Tier (3 slots)**: *Features: `ret`, `vol` (High-Low), `wick_up`, `wick_dn`.*
- **15s Tier (4 slots)**: Completes the 1m. *Features: `ret`, `vol`, `wick_up`, `wick_dn`.*
- **1m Tier (4 slots)**: Completes the 5m. *Features: `ret`, `vol`, `wick_up`, `wick_dn`.*
- **5m Tier (3 slots)**: Completes the 15m. *Features: `ret`, `vol`, `wick_up`, `wick_dn`.*
- **15m Tier (4 slots)**: Surrounding hour. *Features: `ret`, `vol`, `wick_up`, `wick_dn`.*

*Total Features per Anchor:* $\approx 23 \text{ slots} \times 4 \text{ features} = 92$ initial linear features.

## 3. Modeling & Degree Sweep Pipeline
We will utilize the existing `ag_logistic_model.py` / PyTorch stepwise pipeline.
1. **Linear Selection**: Day-block CV stepwise selection (LASSO/L1) to select the top ~15 dominant features.
2. **Quadratic Expansion**: Add squared terms ($x_i^2$) of the *selected 15* only. Accept over Linear *only if* 2025 Out-Of-Sample AUC/Log-Loss improves significantly.
3. **Cubic Expansion**: Add cubed terms ($x_i^3$) and pairwise interactions ($x_i x_j$) of the *selected 15* only. Accept over Quadratic *only if* 2025 OOS metrics improve.
4. **Target Definitions**: 
   - Binary Response (Hit vs Miss)
   - Magnitude (Raw Points - strictly unclamped)

## 4. Execution Scope & Compute Estimate
- **Scope**: Run exclusively on the tracked surviving edges (`FIB-17`, `VA-13`) and any additional dossiers specified.
- **Compute Estimate**: 
  - Extraction: ~2-3 minutes per dossier (parsing the raw `DATA/ATLAS/5s` layer backward/forward from $t_e$ and $t_x$).
  - Training: ~5 minutes per dossier (PyTorch stepwise CV on CPU/GPU over 2024 train blocks + tensor expansions).
  - Total Estimated Compute Time: ~10-15 minutes for the tracked cohort.

## 5. Output Artifacts
- Per-Dossier Reports: `tests/<ID>/FSPACE_<ID>.md`
- Master Summary: `reports/AG_cat_00_FSPACE.md` (reporting OOS Log-Loss, AUC, Pseudo-R² for accepted degrees, greyed out if AUC gap < 0.05).

Awaiting your approval to begin the F-Space extraction and training loop.
