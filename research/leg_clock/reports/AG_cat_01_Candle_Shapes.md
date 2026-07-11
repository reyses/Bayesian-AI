# Concept Report: 01_Candle_Shapes

## 1. Definition
- **Concept:** 01_Candle_Shapes
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 0.7239 pts (Gap: 0.4468)
- **Short Signal 15m Return:** 0.4285 pts (Gap: -0.1515)
- **Combined Edge Gap:** 0.2953 pts

## 3. Economics Test
- **Average Trades / Day:** 86.26
- **Gross PnL / Day:** $-16.59
- **Net PnL / Day (4 ticks round-trip cost):** $-189.65
- **95% Bootstrap CI (Net $/day):** [$-202.51, $-177.81]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-202.51, -177.81] includes 0 or is highly negative.
