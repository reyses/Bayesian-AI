# Concept Report: 07_MA_Crossover

## 1. Definition
- **Concept:** 07_MA_Crossover
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 0.6225 pts (Gap: 0.3454)
- **Short Signal 15m Return:** 0.2221 pts (Gap: 0.0549)
- **Combined Edge Gap:** 0.4004 pts

## 3. Economics Test
- **Average Trades / Day:** 22.65
- **Gross PnL / Day:** $-1.14
- **Net PnL / Day (4 ticks round-trip cost):** $-46.39
- **95% Bootstrap CI (Net $/day):** [$-53.41, $-39.19]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-53.41, -39.19] includes 0 or is highly negative.
