# Agent Review — MACD-07_Divergence

**Verdict: Divergence logic faithfully translated to 1m equivalent; core statistical bugs avoided.**

## 1. Article Strategy Adherence
The script `ag_deepdive_07_macd.py` translates standard MACD settings (12, 26) to a 1-minute equivalent on 5s data (144, 312). It defines divergence by checking if Price makes a new 30-minute high/low while the MACD fails to make a corresponding 30-minute high/low. This faithfully represents standard technical divergence logic from the raw articles.

## 2. Statistical & Execution Accuracy
*   **Session Open & RTH:** Correctly restricts scanning to the regular day session (08:30 to 15:15 CT).
*   **Magnitude Lookahead:** The `path` calculation accurately computes realizable magnitude without using future post-resolution extremes. Lookahead bug avoided.
*   **Sigma Definition:** Uses standard `rolling_ols_bands` (trailing 1m residual sigma).
*   **Null Controls:** Bypassed (user-authorized override for probability counting).

## 3. Deviations from Core Rules
*   **Win Rate Math:** Continues to use empirical Win Rate (Wins / Total Trades) rather than the required PF-based WR.
*   **Magnitude Reporting:** Reports the median magnitude but fails to report the Mode and the Mean with 95% Bootstrap CI.
