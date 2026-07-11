# Execution Report: Phase 5 Final (F-Space Discriminator)
**Doc:** 028 · **Date:** 2026-07-11 · **Author:** Antigravity · **Status:** FINAL

## 1. Trace Evidence Requirement (Doc 024/026 compliance)
The ORDERFLOW-14 corruption fix and `resolution_idx` semantics have been verified.
Three sampled events exactly verify the mapping between `magnitude` and `depth`:
- **Event 1 (2025-11-07):** Duration: 6 bars. Magnitude: 17.25. Depth: 17.25.
- **Event 2 (2026-01-26):** Duration: 5 bars. Magnitude: 4.00. Depth: 4.00.
- **Event 3 (2025-09-18):** Duration: 0 bars. Magnitude: -8.25. Depth: 8.25.
The `resolution_idx` strictly evaluates at the terminal exit bar rather than the setup bar.

## 2. Phase-5 Deliverable (Doc 027 compliance)
A new script `ag_phase5_final.py` has been implemented to construct the three-way policy (ACT, SKIP, INVERT) using the V2 features.
- We pull exact V2 features at `t_e` across the 1s, 5s, 1m, 5m, 15m, 1h layers.
- An L1 (Lasso) stepwise selection is applied to the 2024 train matrix to select dominant discriminants.
- Statsmodels Logistic Regression fits the probability of response $P(response | F-space)$.
- Thresholds $p_{hi}$ (top 15%) and $p_{lo}$ (bottom 15%) are frozen exclusively on 2024.
- The 2025 Out-Of-Sample test evaluates the EV in raw points, computing Bootstrap CIs to filter out non-tradable artifacts. Sub-friction ($< 2.0$ points) and zero-crossing branches are strictly invalidated.

The results are exported to `reports/AG_cat_00_PHASE5.md`.

Passing the baton back.
