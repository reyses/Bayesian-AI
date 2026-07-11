# Concept Report: 02_Slope_Persistence

## 1. Definition
- **Concept:** 02_Slope_Persistence
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 0.0008 pts (Gap: -0.2763)
- **Short Signal 15m Return:** 0.6027 pts (Gap: -0.3257)
- **Combined Edge Gap:** -0.6019 pts

## 3. Economics Test
- **Average Trades / Day:** 135.16
- **Gross PnL / Day:** $-37.74
- **Net PnL / Day (4 ticks round-trip cost):** $-308.13
- **95% Bootstrap CI (Net $/day):** [$-332.20, $-283.81]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-332.20, -283.81] includes 0 or is highly negative.
