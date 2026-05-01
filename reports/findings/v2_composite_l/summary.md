# V2 Composite Directional (L-Aggregator) — 2026-05-01 14:55 UTC

**Voter TFs:** ['1m', '5m', '15m', '1h', '4h']

**Common cadence:** `5m` (73918 bars in test window)

**Baseline (majority class):** 50.2%

## Per-TF (own cadence)

 tf      n  accuracy  baseline     lift   pred_std  pred_mean_abs
 1m 370729  0.620003  0.582175 0.037828  20.271386      13.669029
 5m  74099  0.623166  0.598321 0.024845  34.981895      24.116654
15m  24691  0.621725  0.564254 0.057470  64.093619      43.288813
 1h   6175  0.686478  0.654899 0.031579 111.511616      79.582730
 4h   1648  0.709951  0.686286 0.023665 220.204865     162.662997

## Per-TF (projected onto 5m cadence)

 tf  n_active  pct_data  accuracy     lift
 1m     73764     100.0  0.524253 0.021921
 5m     73764     100.0  0.525826 0.023494
15m     73764     100.0  0.525066 0.022735
 1h     73764     100.0  0.534136 0.031804
 4h     73764     100.0  0.522626 0.020294

## Composite: majority vote (stratified)

 min_agree     n   pct_data  accuracy     lift
         1 73764 100.000000  0.530869 0.028537
         2 73764 100.000000  0.530869 0.028537
         3 73764 100.000000  0.530869 0.028537
         4 50146  67.981671  0.542616 0.040284
         5 24561  33.296730  0.564716 0.062385

## Composite: strict-all

     label  n_active  pct_data  accuracy     lift  n_long  n_short  long_acc  short_acc
strict-all     24561  33.29673  0.564716 0.062385   15615     8946  0.564649   0.564833

## Composite: magnitude-weighted (stratified by |weighted_sum|)

 threshold_w     n   pct_data  accuracy     lift
         0.0 73764 100.000000  0.528537 0.026205
         1.0 54328  73.651104  0.538525 0.036193
         2.0 36993  50.150480  0.550401 0.048070
         3.0 24277  32.911718  0.568357 0.066025
         5.0  9626  13.049726  0.598379 0.096048

## Composite: confgated-majority (|pred|>10.0)

                  label  n_active  pct_data  accuracy     lift  n_long  n_short  long_acc  short_acc
confgated_majority_10.0     66895 90.687869  0.535107 0.032775   37646    29249  0.533656   0.536976
