# Concept Report: 11_VolProfile_POC

## 1. Definition
- **Concept:** 11_VolProfile_POC
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 4.2757 pts (Gap: 3.9986)
- **Short Signal 15m Return:** -1.6595 pts (Gap: 1.9366)
- **Combined Edge Gap:** 5.9352 pts

## 3. Economics Test
- **Average Trades / Day:** 157.42
- **Gross PnL / Day:** $-8.95
- **Net PnL / Day (4 ticks round-trip cost):** $-323.27
- **95% Bootstrap CI (Net $/day):** [$-352.78, $-295.55]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-352.78, -295.55] includes 0 or is highly negative.
