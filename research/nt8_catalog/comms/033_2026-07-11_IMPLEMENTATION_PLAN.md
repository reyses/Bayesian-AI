# AG Implementation Plan (B1-B4)
**Doc:** 033 · **Date:** 2026-07-11 · **Author:** AG · **Status:** PENDING APPROVAL

## B1: Depth Re-derivation (Pre-Trade Trigger Extremity)

### 16 Defective Dossiers (Proposed Definitions)
1. **ADX-08**: `adx` value at the time of trigger (trend strength magnitude).
2. **CROSS-11**: `abs(sma50 - sma200)` (moving average separation distance at trigger).
3. **HNS-22**: `abs(head_peak - neckline)` (pattern height).
4. **ORB-02**: `or_high - or_low` (opening range width).
5. **PIVOT-16**: `abs(p0 - pivot_level)` (distance from price to pivot level at bounce/break).
6. **RENKO-24**: Number of consecutive bricks in the prior trend before the flip.
7. **ROUND-05**: `abs(p0 - round_level)` (distance of entry from the psychological boundary).
8. **RSI-06**: `abs(price_extreme_distance)` or `abs(rsi_divergence)` (magnitude of the divergence).
9. **SAR-23**: `abs(p0 - sar_val)` (distance from price to the newly flipped SAR dot).
10. **SCALP-18**: `abs(p0 - vwap)` (distance to VWAP at the trigger point).
11. **SQZ-04**: Duration (number of bars) the squeeze was active before firing.
12. **TUNNEL-20**: `abs(ema_high - ema_low)` (width of the Elliott Wave tunnel).
13. **VP-01**: `abs(vah - val)` (Value Area width for the profile).
14. **VWAP-03**: `abs(z_curr)` (Z-score distance of price from VWAP).
15. **VWMA-10**: `abs(vwma - sma)` (Volume-weighted moving average divergence magnitude).
16. **ZONE-21**: `abs(z_high - z_low)` (width of the supply/demand zone).

### 8 OK-Flagged Dossiers (Semantic Confirmation)
*Note: The audit listed 15 defective, but enumerated 16. The remaining 8 OK dossiers are:*
1. **ATR-09**: Gap ATR-fill fraction / size of the statistical fade gap.
2. **DOW-19**: Magnitude of the price-volume divergence.
3. **FIB-17**: Distance from the specific Fibonacci retracement level.
4. **MACD-07**: MACD histogram divergence amplitude.
5. **OHLC-01**: Distance to the targeted prior day OHLC level.
6. **ORDERFLOW-14**: Cumulative delta divergence magnitude at the extreme.
7. **SEASON-12**: Overnight gap size in absolute points.
8. **VA-13**: Value Area width or distance from VA bounds.

## B2: OHLC-01 `resolution_idx` Fix
**Root Cause**: The `resolution_idx` calculation in the script was logically patched by the previous session, but `events.parquet` is stale because the script was never re-run. Furthermore, the generic MFE/MAE injection block incorrectly categorized Setup 1 (Bearish Bounce) as `bullish` due to a hardcoded `_setup_val == 1` fallback, causing the exit scan to look in the wrong direction (`_dir = 1`) and fail, resulting in default index fallbacks.
**Plan**:
1. Fix the MFE/MAE direction logic in `ag_deepdive_01_ohlc.py` so `_dir` accurately reflects `bearish_bounce` and `bullish_bounce` modes.
2. Re-run `ag_deepdive_01_ohlc.py` to regenerate `events.parquet`.
3. Provide an evidence trace of 3 random events to prove `resolution_idx > event_idx` and correct `mfe`/`mae` values.

## B3: RSI-06 |mag|max = 1948 pts Investigation
**Plan**:
1. Write an OQ-trace script (`trace_rsi_06.py`) to isolate the event(s) with magnitude > 1000 pts.
2. Cross-reference the event timestamp with the raw 5s parquet data to see if the outcome window legitimately captured a historic market crash, or if it's a data gap/artifact.
3. If it's an artifact, apply clamping or invalidation logic. If valid, prove it with a printed raw price trace.

## B4: Phase-5 Model Concerns
**Plan**:
1. **B4a (Feature Map)**: I will run an extraction on `ag_phase5_fspace.py` to list the 52 features/anchor generated, map them to the 23 slots specified in doc 017, and propose the missing layer families.
2. **B4b (ATR-09 LASSO Degeneracy)**: I will verify/add `StandardScaler()` inside `ag_phase5_final.py` immediately prior to the L1 penalty to ensure all features are on the same scale, preventing arbitrary elimination of unscaled features.
3. **B4c (ORDERFLOW Sub-Friction Exclusion)**: I will adjust the table rendering logic in `ag_phase5_final.py` to rigorously apply the Doc 027 Rule 3 (invalidate branches with EV < ~2 pts), ensuring the ORDERFLOW ACT branch correctly shows as Invalid.
4. Finally, re-run Phase 5 on the 4 priority dossiers (2024 thresholds, 2024/2025 CI evaluation) and verify output.

---
Please provide APPROVAL or requested modifications to this plan so I can proceed with the execution.
