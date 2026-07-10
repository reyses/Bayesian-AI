# Agent Review — ATR-09_Statistical_Fade

**Verdict: Strategy aligns with article logic; core statistical bugs (lookahead, wrong open) avoided.**

## 1. Article Strategy Adherence
The script `ag_deepdive_09_atr.py` targets `Open + 1 Daily ATR` and `Open - 1 Daily ATR` to trigger bearish and bullish fades, respectively. This correctly implements the "Statistical ATR fade" concept (the "90% rule" where markets rarely exceed 1 full daily ATR from the open without fading). The logic faithfully maps to the underlying hypothesis.

## 2. Statistical & Execution Accuracy
*   **Session Open & RTH:** Correctly anchors the daily open to 08:30 CT and restricts to the day session.
*   **Magnitude Lookahead:** The `path` calculation checks up to the resolution bar and stops, or uses the end of the 60-minute horizon without computing an impossible hindsight max/min. No lookahead bug.
*   **Sigma Definition:** Adheres to the framework standard using `rolling_ols_bands` trailing 1m residual sigma.
*   **Null Controls:** Bypassed (user-authorized override for probability counting).

## 3. Deviations from Core Rules
*   **Win Rate Math:** Continues to use empirical Win Rate (Wins / Total Trades) rather than the required PF-based WR.
*   **Magnitude Reporting:** Reports the median magnitude but fails to report the Mode and the Mean with 95% Bootstrap CI. 
