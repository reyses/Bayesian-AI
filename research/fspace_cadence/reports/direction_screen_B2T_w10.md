# Direction pre-screen — FEATURES_RUN_B2T | wait=10s (anchor = regime start + wait) -> forward direction
train days ['2024_02_20', '2024_02_21', '2024_02_22'] -> test days ['2024_02_23', '2024_02_26'] (disjoint) | HistGradientBoosting | horizons [30, 60, 120, 300]s

 horizon |  REAL AUC    acc |  NULL AUC |  AUC gap | nTest(real)
----------------------------------------------------------------
     30s |     0.493   50% |     0.524 |   -0.031 | 1754 (P_up=48%)
     60s |     0.490   50% |     0.510 |   -0.021 | 1754 (P_up=50%)
    120s |     0.486   49% |     0.506 |   -0.020 | 1754 (P_up=51%)
    300s |     0.485   49% |     0.526 |   -0.041 | 1754 (P_up=54%)

VERDICT: REAL ~ null ~ 0.5 at all horizons -> NO forecastable direction; do NOT build Mamba.
(AUC 0.5 = chance. Need REAL AUC > ~0.52 AND clearly above the Fourier null to claim signal. n=5 days, 1 null draw.)
