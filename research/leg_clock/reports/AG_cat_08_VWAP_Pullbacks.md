# Concept Report: 08_VWAP_Pullbacks

## 1. Definition
- **Concept:** 08_VWAP_Pullbacks
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** -0.3037 pts (Gap: -0.5808)
- **Short Signal 15m Return:** -0.1142 pts (Gap: 0.3912)
- **Combined Edge Gap:** -0.1895 pts

## 3. Economics Test
- **Average Trades / Day:** 27.87
- **Gross PnL / Day:** $-3.56
- **Net PnL / Day (4 ticks round-trip cost):** $-59.20
- **95% Bootstrap CI (Net $/day):** [$-66.77, $-51.41]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-66.77, -51.41] includes 0 or is highly negative.
