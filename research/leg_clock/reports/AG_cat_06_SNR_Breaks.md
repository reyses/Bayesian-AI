# Concept Report: 06_SNR_Breaks

## 1. Definition
- **Concept:** 06_SNR_Breaks
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 0.0000 pts (Gap: -0.2770)
- **Short Signal 15m Return:** 0.0000 pts (Gap: 0.2770)
- **Combined Edge Gap:** 0.0000 pts

## 3. Economics Test
- **Average Trades / Day:** 0.00
- **Gross PnL / Day:** $0.00
- **Net PnL / Day (4 ticks round-trip cost):** $0.00
- **95% Bootstrap CI (Net $/day):** [$0.00, $0.00]

## 4. Verdict
**NOISE**
*Reasoning:* Bootstrap CI [0.00, 0.00] includes 0 or is highly negative.
