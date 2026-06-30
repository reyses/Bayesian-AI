# Direction pre-screen — causal F-space at regime start -> forward direction (B2T tiled)
train days ['2024_02_20', '2024_02_21', '2024_02_22'] -> test days ['2024_02_23', '2024_02_26'] (disjoint) | HistGradientBoosting | horizons [30, 60, 120, 300]s

 horizon |  REAL AUC    acc |  NULL AUC |  AUC gap | nTest(real)
----------------------------------------------------------------
     30s |     0.493   50% |     0.485 |   +0.008 | 1754 (P_up=49%)
     60s |     0.511   51% |     0.534 |   -0.023 | 1754 (P_up=50%)
    120s |     0.498   50% |     0.517 |   -0.019 | 1754 (P_up=51%)
    300s |     0.505   50% |     0.524 |   -0.020 | 1754 (P_up=54%)

VERDICT: REAL ~ null ~ 0.5 at all horizons -> NO forecastable direction; do NOT build Mamba.
(AUC 0.5 = chance. Need REAL AUC > ~0.52 AND clearly above the Fourier null to claim signal. n=5 days, 1 null draw.)
