---
name: Data validation before modeling
description: Always validate data quality against ground truth BEFORE training any models or running analysis
type: feedback
---

Data validation must be the FIRST step, not an afterthought.

**Why:** We trained multiple CNN models (29D, trajectory, direction) and ran oscillation research on ATLAS data that contained corrupted bars (fake highs/lows from aggregation errors). Bad data produces bad labels which produce bad models. The 1m bar with a fake high of 24628 (actual was 24429) was only caught visually on a chart.

**How to apply:**
1. Before ANY model training or analysis, run `python tools/validate_data.py`
2. After any data update (NT8 export, upsampling, ATLAS merge), re-validate
3. The 1s data is ground truth — all higher TF bars must have OHLCV within 1s range
4. After validation fixes, delete cached features (`.npy`) and rebuild
5. Add validation as step 0 in every pipeline that touches ATLAS data

**Rule:** No model trains on unvalidated data. No analysis runs on unvalidated data. Period.
