# Agent Review — ORDERFLOW-14

**Verdict: Accurately represents standard Order Flow setups; core statistical bugs avoided.**

## 1. Article Strategy Adherence
The script `ag_deepdive_14_orderflow.py` uses pre-calculated cumulative delta to test two classic order flow setups:
1.  **Delta Divergence:** Scans for extreme cumulative delta divergence percentiles (< 10th percentile for bearish fades, > 90th percentile for bullish fades). This correctly models divergence between price and aggressive order flow.
2.  **Trapped Traders:** Looks for positive delta (net buying) when the price is down (buyers trapped, bearish runner), and negative delta (net selling) when the price is up (sellers trapped, bullish runner). This maps faithfully to standard trapped-trader order flow dynamics.

## 2. Statistical & Execution Accuracy
*   **Session Open & RTH:** Correctly restricts scanning to the regular day session (08:30 to 15:15 CT).
*   **Magnitude Lookahead:** The `path` calculation accurately computes realizable magnitude bounded by the 15-minute horizon without referencing future un-realizable extremes. Lookahead bug avoided.
*   **Sigma Definition:** Uses standard `rolling_ols_bands` (trailing 1m residual sigma).
*   **Null Controls:** Bypassed (user-authorized override for probability counting).

## 3. Deviations from Core Rules
*   **Win Rate Math:** Continues to use empirical Win Rate (Wins / Total Trades) rather than the required PF-based WR.
*   **Magnitude Reporting:** Reports the median magnitude but fails to report the Mode and the Mean with 95% Bootstrap CI.
