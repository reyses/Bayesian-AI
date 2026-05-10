# Pivot direction NN — v2 (185D)

Generated: 2026-04-23T20:40:21
Feature spec: 185D (1 L0 + 8 TFs x 23 per-TF)
Trades: 8195  Train 5251 Val 498 Test 2229
Skipped: missing_day_features=0, nan_or_warmup=217
Train win%: 60.1%  Val: 58.0%  Test: 60.4%
Best val AUC: 0.5196
Test AUC: 0.5160  accuracy: 0.604

## v1 baseline (for comparison)

v1 test AUC (91D features): 0.63 per session memory
v2 test AUC (185D features): 0.5160

## Prediction-calibration on test set

| prob bucket | N trades | actual win% |
|---|---:|---:|
| 0.5 – 0.6 | 978 | 59.0% |
| 0.6 – 0.7 | 1251 | 61.6% |
