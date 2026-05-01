# V2 Composite Directional (L-Aggregator) — 2026-05-01 23:11 UTC

**Voter TFs:** ['1m', '5m']

**Common cadence:** `5m` (74199 bars in test window)

**Baseline (majority class):** 50.2%

## Per-TF (own cadence)

tf      n  accuracy  baseline     lift  pred_std  pred_mean_abs
1m 370729  0.620003  0.582175 0.037828 20.271386      13.669029
5m  74099  0.623166  0.598321 0.024845 34.981895      24.116654

## Per-TF (projected onto 5m cadence)

tf  n_active  pct_data  accuracy     lift
1m     74045     100.0  0.523898 0.022108
5m     74045     100.0  0.525248 0.023459

## Composite: majority vote (stratified)

 min_agree     n  pct_data  accuracy     lift
         1 57047  77.04369  0.531895 0.030105
         2 57047  77.04369  0.531895 0.030105

## Composite: strict-all

     label  n_active  pct_data  accuracy     lift  n_long  n_short  long_acc  short_acc
strict-all     57047  77.04369  0.531895 0.030105   32408    24639  0.532461    0.53115

## Composite: magnitude-weighted (stratified by |weighted_sum|)

 threshold_w     n   pct_data  accuracy     lift
         0.0 74045 100.000000  0.526018 0.024229
         1.0 34651  46.797218  0.548758 0.046968
         2.0 13813  18.654872  0.564179 0.062389
         3.0  5708   7.708826  0.568500 0.066711
         5.0  1370   1.850226  0.587591 0.085802

## Composite: confgated-majority (|pred|>10.0)

                  label  n_active  pct_data  accuracy     lift  n_long  n_short  long_acc  short_acc
confgated_majority_10.0     55593 75.080019  0.533772 0.031983   30936    24657  0.535169   0.532019
