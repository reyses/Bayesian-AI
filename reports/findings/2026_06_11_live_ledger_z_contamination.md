# LIVE Ledger Z-Contamination Note

## Issue
Prior to Fix 2, the `live` stack (specifically `engine_v2.py` and `diagnostic_run.py`) accessed the 1m `z` and `vr` (reversion probability) features using hardcoded array indices (`feat[12]`, `feat[14]`, and `feat[15]` for velocity). 

When the underlying `core_v2` feature dimensionality expanded from 91D to 139D (and later to 185D/201D with L0), these legacy hardcoded indices shifted, causing the `z`, `vr`, and `vel` columns in the live telemetry ledgers to contain entirely different, incorrect feature data.

## Fix Applied
In PR/Commit [pending], we migrated the hardcoded indices to use dynamic index resolution via `FEATURE_NAMES.index(...)`:
- `z` -> `L3_1m_z_se_15`
- `vr` (now `rprob`) -> `L3_1m_reversion_prob_15`
- `vel` -> `L2_1m_price_velocity_15`

## Impact
Any live or mock ledgers generated before this fix have **contaminated** `z`, `vr`, and `vel` columns. These columns cannot be used for post-trade analysis, PNL attribution, or ML training.

**Action Required for Analysts:** 
Do NOT trust the `z` or `vr` columns in any `v2_ledger_*.csv` file generated before 2026-06-11. If you need those specific feature values for a trade, you must re-merge the ledger with the `DATA/ATLAS_NT8/FEATURES_5s_v2` dataset using the 5s timestamps as the join key.
