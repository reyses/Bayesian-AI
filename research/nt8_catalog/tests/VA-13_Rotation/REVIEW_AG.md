# Agent Review — VA-13_Rotation

**Verdict: Accurately represents standard Value Area Rotation (80% Rule) setups; core statistical bugs avoided.**

## 1. Article Strategy Adherence
The script `ag_deepdive_13_va.py` computes the prior day's Volume Profile (70% Value Area) and implements standard Market Profile "Value Area Rules." It correctly requires the current day to open *inside* the prior day's Value Area. If price probes the VAH and fails, returning to the POC, it triggers a bearish runner (Setup 1). If it probes the VAL and returns to the POC, it triggers a bullish runner (Setup 2). This faithfully tests standard Value Area rotation mechanics.

## 2. Statistical & Execution Accuracy
*   **Session Open & RTH:** Correctly restricts scanning to the regular day session (08:30 to 15:15 CT).
*   **Magnitude Lookahead:** The `path` calculation accurately computes realizable magnitude bounded by the 60-minute horizon without referencing future un-realizable extremes. Lookahead bug avoided.
*   **Sigma Definition:** Uses standard `rolling_ols_bands` (trailing 1m residual sigma).
*   **Null Controls:** Bypassed (user-authorized override for probability counting).

## 3. Deviations from Core Rules
*   **Win Rate Math:** Continues to use empirical Win Rate (Wins / Total Trades) rather than the required PF-based WR.
*   **Magnitude Reporting:** Reports the median magnitude but fails to report the Mode and the Mean with 95% Bootstrap CI.
