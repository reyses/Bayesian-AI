# Pivot direction NN — v2 (185D)

Generated: 2026-04-23T18:13:07
Feature spec: 185D (1 L0 + 8 TFs x 23 per-TF)
Trades: 299  Train 175 Val 36 Test 86
Skipped: missing_day_features=0, nan_warmup=2
Train win%: 64.6%  Val: 72.2%  Test: 68.6%
Best val AUC: 0.7538
Test AUC: 0.6039  accuracy: 0.686

## v1 baseline (for comparison)

v1 test AUC (91D features): 0.63 per session memory
v2 test AUC (185D features): 0.6039

## Prediction-calibration on test set

| prob bucket | N trades | actual win% |
|---|---:|---:|
| 0.5 – 0.6 | 77 | 67.5% |
| 0.6 – 0.7 | 9 | 77.8% |
