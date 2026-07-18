# Concept Report: 10_Footprint_Imbalance

## 1. Definition
- **Concept:** 10_Footprint_Imbalance
- **Methodology:** Causal forward-pass (no lookahead). Block-bootstrap over 536 days.

## 2. Existence Test (Null Control)
- **Baseline Expected 15m Return:** 0.2770 pts
- **Long Signal 15m Return:** 0.8813 pts (Gap: 0.6043)
- **Short Signal 15m Return:** -1.4490 pts (Gap: 1.7260)
- **Combined Edge Gap:** 2.3303 pts

## 3. Economics Test
- **Average Trades / Day:** 0.65
- **Gross PnL / Day:** $18.63
- **Net PnL / Day (4 ticks round-trip cost):** $17.23
- **95% Bootstrap CI (Net $/day):** [$12.27, $22.99]

## 4. Verdict
**REAL**
*Reasoning:* Bootstrap CI is strictly positive and gap (2.33) >= 0.10
