# AG Implementation Plan: Phase 4 & Honest Sweep Summary

Per the closed loop confirmation in `006_2026-07-11_TASK_COMPLETE_LOOP_CLOSED.md`, the next catalog work is the Phase-4 conditioning sweep directive and the Honest Sweep Summary. 

Here is my execution plan for your approval:

## 1. P0: Unit Standardization (The 5 outliers)
I will modify the following 5 scripts to divide the computed `magnitude`, `mfe`, and `mae` by the trailing 1m regression residual `sigma`, and clip the results at ±2.05σ before calculating hits and EV:
- `tests/FIB-17_Confluence/ag_deepdive_17_fib.py`
- `tests/PIVOT-16_Floor_Levels/ag_deepdive_16_pivots.py`
- `tests/VP-01_Volume_Profile/ag_deepdive_01_vol_profile.py`
- `tests/ORDERFLOW-14/ag_deepdive_14_orderflow.py`
- `tests/SCALP-18_VWAP_EMA/ag_deepdive_18_scalp.py`
*(I will then re-run these 5 scripts to regenerate their `events.parquet` and `DOC_*.md` files under the §7 standard).*

## 2. P1: Honest Sweep Summary
I will build a script `tools/generate_master_index.py` that parses the 18 `DOC_*.md` files. It will replace the stale table in `reports/AG_cat_00_INDEX.md` with the true EV findings to reflect the honest summary that there are NO unconditional stable positive responses, only negative EV inversions.

## 3. Phase 4: Conditioning Sweep
I will build the master conditioning script `tools/ag_phase4_conditioning.py`. For each dossier with an `events.parquet`, the script will:
1. Load the events and fetch the corresponding daily `DATA/ATLAS/5s` parquet files.
2. Compute the 4 specified conditioners:
   - **Hour-of-day:** {pre-7, 7–9, 9–11, 11–13, 13–15, 15+} CT.
   - **Regime (ER trailing 60m):** {churn, mixed, trend} terciles.
   - **Volatility state (trailing 1m sigma percentile):** {low, mid, high} terciles.
   - **Event depth (trigger magnitude in σ):** terciles.
3. Group events by these 4 dimensions and evaluate the hit-rate and EV (with bootstrapped 95% CIs) for both 2024 and 2025.
4. Filter for **replicated** windows (significant in same direction in BOTH years, N >= 30).
5. Append `COND_<ID>.md` into the respective dossier folders.
6. Aggregate all replicated windows into a final ranked list in `reports/AG_cat_00_CONDITIONING.md`.

## Open Questions for Reviewer
- **Event Depth Approximation:** For time-based anomalies like SEASON-12 where there is no obvious "event magnitude", I plan to use the pre-market gap size as the event depth. Is this acceptable, or should I skip the Event Depth conditioner for non-applicable setups?

Please provide a VERDICT (APPROVED/REJECTED) and any necessary corrections so I can proceed to execution.
