# Pivot direction NN

Generated: 2026-04-22T17:19:14
Trades: 21300  Train 14843 Val 3000 Test 3457
Best val AUC: 0.6226
Test AUC: 0.6344  accuracy: 0.584

## Prediction-calibration on test set

| prob bucket | N trades | actual win% |
|---|---:|---:|
| 0.0 – 0.3 | 256 | 12.5% |
| 0.3 – 0.4 | 346 | 31.2% |
| 0.4 – 0.5 | 706 | 45.9% |
| 0.5 – 0.6 | 1616 | 51.7% |
| 0.6 – 0.7 | 533 | 63.4% |
