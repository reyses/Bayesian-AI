# Oscillation State Contamination Report

Date: 2026-06-11

## Overview
A hardcoded index bug in `core_v2/ledger.py` was tracking `features[12]` (the legacy 91-D feature index) instead of the dynamically resolved V2 index for `L3_1m_z_se_15` (which is typically `163`).

## Impact
Because of this hardcoded index mapping, the oscillation tracker was reading the wrong V2 feature as the 1m `z_se` proxy. Any 1m/5m feature states saved into recent `closed_trades` during live execution (and simulation) were corrupted for the oscillation fields (like `z_sign`, `z_peak`, `zero_crossings`).

This likely triggered premature oscillation-tracker resets, directly skewing giveback/TimeStop or other oscillation-based logic.

## Resolution
- `core_v2/ledger.py` was updated to import `FEATURE_NAMES` and dynamically resolve `Z_IDX = FEATURE_NAMES.index('L3_1m_z_se_15')`.
- All `entry_features[12]` and `features[12]` indexing operations were updated to use `Z_IDX`.
