# V2 Single-Horizon Composite — 2026-05-01 18:06 UTC

**Common cadence:** `5m`
**Voter TFs:** ['1m', '5m', '15m', '1h', '4h']
**Target:** signed MFE × regime direction at 5m (lookahead=24 bars)
**Baseline (avg majority class):** 52.9%

## Per-voter

 tf  n_train  n_test   r2_test  accuracy_test  baseline_acc      lift
 1m    44519   14841 -0.013531       0.546668      0.528729  0.017939
 5m    44519   14841 -0.001127       0.578500      0.528729  0.049771
15m    44519   14841  0.001627       0.595967      0.528729  0.067238
 1h    44519   14841 -0.029245       0.541476      0.528729  0.012746
 4h    44519   14841 -0.057632       0.514432      0.528729 -0.014297

## Composite — majority vote

 min_agree     n   pct_data  accuracy     lift
         1 14828 100.000000  0.561910 0.033180
         2 14828 100.000000  0.561910 0.033180
         3 14828 100.000000  0.561910 0.033180
         4 11091  74.797680  0.580831 0.052102
         5  6383  43.046938  0.609431 0.080702

## Composite — strict-all

   n  pct_data  accuracy     lift
6383 43.046938  0.609431 0.080702

## Composite — magnitude-weighted

 threshold_w     n   pct_data  accuracy     lift
         0.0 14828 100.000000  0.562652 0.033922
         1.0 13220  89.155651  0.570575 0.041845
         2.0 11267  75.984624  0.579835 0.051105
         3.0  8870  59.819261  0.588613 0.059884
         5.0  3807  25.674400  0.626478 0.097748

## Composite — confgated (|pred| > 20.0)

 threshold     n  pct_data  accuracy     lift
      20.0 10502 70.825465  0.570844 0.042114
