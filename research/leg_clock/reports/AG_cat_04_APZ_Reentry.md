# Concept Report: 04_APZ_Reentry

## 1. Definition
- **Concept:** 04_APZ_Reentry
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 1.4598 pts (Gap: 1.1828)
- **Short Signal 15m Return:** -0.3779 pts (Gap: 0.6549)
- **Combined Edge Gap:** 1.8377 pts

## 3. Economics Test
- **Average Trades / Day:** 46.57
- **Gross PnL / Day:** $0.16
- **Net PnL / Day (4 ticks round-trip cost):** $-93.02
- **95% Bootstrap CI (Net $/day):** [$-101.92, $-84.57]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [-101.92, -84.57] includes 0 or is highly negative.
