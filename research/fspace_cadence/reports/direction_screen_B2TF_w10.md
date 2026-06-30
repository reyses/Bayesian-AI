# Direction pre-screen — FEATURES_RUN_B2TF | wait=10s (anchor = regime start + wait) -> forward direction
train days ['2024_02_20', '2024_02_21', '2024_02_22'] -> test days ['2024_02_23', '2024_02_26'] (disjoint) | HistGradientBoosting | horizons [30, 60, 120, 300]s

 horizon |  REAL AUC    acc |  NULL AUC |  AUC gap | nTest(real)
----------------------------------------------------------------
     30s |     0.484   48% |     0.522 |   -0.038 | 1754 (P_up=48%)
     60s |     0.489   49% |     0.531 |   -0.042 | 1754 (P_up=50%)
    120s |     0.514   52% |     0.519 |   -0.005 | 1754 (P_up=51%)
    300s |     0.474   48% |     0.505 |   -0.031 | 1754 (P_up=54%)

VERDICT: REAL ~ null ~ 0.5 at all horizons -> NO forecastable direction; do NOT build Mamba.
(AUC 0.5 = chance. Need REAL AUC > ~0.52 AND clearly above the Fourier null to claim signal. n=5 days, 1 null draw.)
