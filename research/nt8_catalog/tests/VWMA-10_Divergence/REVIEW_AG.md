# Agent Review — VWMA-10_Divergence

**Verdict: VWMA/SMA crossover correctly calculated; core statistical bugs avoided.**

## 1. Article Strategy Adherence
The script `ag_deepdive_10_vwma.py` computes a 20-minute (240-period on 5s) Volume-Weighted Moving Average (VWMA) and a matching Simple Moving Average (SMA). It triggers a bullish runner when the VWMA crosses above the SMA (indicating volume is driving price upward more than the time-based average) and a bearish runner when the VWMA crosses below the SMA. This accurately implements standard volume divergence crossover logic.

## 2. Statistical & Execution Accuracy
*   **Session Open & RTH:** Correctly restricts scanning to the regular day session (08:30 to 15:15 CT).
*   **Magnitude Lookahead:** The `path` calculation accurately computes realizable magnitude bounded by the 60-minute horizon without referencing future un-realizable extremes. Lookahead bug avoided.
*   **Sigma Definition:** Uses standard `rolling_ols_bands` (trailing 1m residual sigma).
*   **Null Controls:** Bypassed (user-authorized override for probability counting).

## 3. Deviations from Core Rules
*   **Win Rate Math:** Continues to use empirical Win Rate (Wins / Total Trades) rather than the required PF-based WR.
*   **Magnitude Reporting:** Reports the median magnitude but fails to report the Mode and the Mean with 95% Bootstrap CI.
