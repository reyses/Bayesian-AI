# Agent Review — SQZ-04_Volatility_Squeeze

**Verdict: Accurately reproduces TTM Squeeze mechanics; core statistical bugs avoided.**

## 1. Article Strategy Adherence
The script `ag_deepdive_04_squeeze.py` implements the classic "TTM Squeeze" logic (John Carter). It computes 20-period Bollinger Bands and 20-period Keltner Channels (using a proxy ATR). The "squeeze" is defined precisely as the Bollinger Bands contracting entirely inside the Keltner Channels. A breakout is triggered when price closes outside the Bollinger Band while the squeeze is (or just was) active. This is a faithful transcription of the original volatility expansion strategy.

## 2. Statistical & Execution Accuracy
*   **Session Open & RTH:** Correctly restricts scanning to the regular day session (08:30 to 15:15 CT).
*   **Magnitude Lookahead:** The `path` calculation accurately computes realizable magnitude bounded by the 60-minute horizon without referencing future un-realizable extremes. Lookahead bug avoided.
*   **Sigma Definition:** Uses standard `rolling_ols_bands` (trailing 1m residual sigma).
*   **Null Controls:** Bypassed (user-authorized override for probability counting).

## 3. Deviations from Core Rules
*   **Win Rate Math:** Continues to use empirical Win Rate (Wins / Total Trades) rather than the required PF-based WR.
*   **Magnitude Reporting:** Reports the median magnitude but fails to report the Mode and the Mean with 95% Bootstrap CI.
