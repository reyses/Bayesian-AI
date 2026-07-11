# Concept Report: 03_Band_Bounce

## 1. Definition
- **Concept:** 03_Band_Bounce
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 0.7916 pts (Gap: 0.5145)
- **Short Signal 15m Return:** -0.2916 pts (Gap: 0.5686)
- **Combined Edge Gap:** 1.0832 pts

## 3. Economics Test
- **Average Trades / Day:** 69.29
- **Gross PnL / Day:** $7.24
- **Net PnL / Day (4 ticks round-trip cost):** $-131.22
- **95% Bootstrap CI (Net $/day):** [$-142.57, $-119.72]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-142.57, -119.72] includes 0 or is highly negative.
