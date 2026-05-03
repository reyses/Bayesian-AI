# V2 Volume x Variation EDA — 2026-05-03 16:25 UTC

**Base TF:** `5m`  **Split:** `IS`
**TFs analyzed:** ['5m', '15m', '1h', '4h']
**Quantiles:** 3
**Forward N:** 12 bars

## 4-corner summary (averaged across all (vol, var) pairs)

| Corner | Avg mean_fwd | Avg WR | Avg dom. pct | Top regimes |
|---|---:|---:|---:|---|
| LOW_VOL_HIGH_VAR | +3.11 | 52.2% | 45% | FLAT_CHOPPY(72) |
| HIGH_VOL_LOW_VAR | +0.66 | 52.6% | 40% | FLAT_CHOPPY(66), FLAT_SMOOTH(6) |
| HIGH_VOL_HIGH_VAR | +1.30 | 52.1% | 41% | FLAT_CHOPPY(72) |
| LOW_VOL_LOW_VAR | -0.00 | 52.0% | 39% | FLAT_CHOPPY(69), FLAT_SMOOTH(3) |

## LOW_VOL x HIGH_VAR top 15 (your question — fakeout territory)

 tf          vol_feature          var_feature           corner    n dominant_regime  dominant_pct  mean_fwd  win_rate
15m    L2_15m_vol_mean_w L2_15m_price_sigma_w LOW_VOL_HIGH_VAR  486     FLAT_CHOPPY      0.613169 28.704218  0.526749
15m   L2_15m_vol_sigma_w L2_15m_price_sigma_w LOW_VOL_HIGH_VAR  513     FLAT_CHOPPY      0.536062 27.274854  0.576998
 5m     L2_5m_vol_mean_w  L3_5m_swing_noise_w LOW_VOL_HIGH_VAR  606     FLAT_CHOPPY      0.617162 23.497525  0.572607
 5m    L2_5m_vol_sigma_w  L3_5m_swing_noise_w LOW_VOL_HIGH_VAR  912     FLAT_CHOPPY      0.564693 16.583059  0.535088
 5m     L2_5m_vol_mean_w  L2_5m_price_sigma_w LOW_VOL_HIGH_VAR  453     FLAT_CHOPPY      0.611479 13.926600  0.565121
15m    L2_15m_vol_mean_w L3_15m_swing_noise_w LOW_VOL_HIGH_VAR 1166     FLAT_CHOPPY      0.543739  7.399014  0.583190
 1h     L2_1h_vol_mean_w      L1_1h_bar_range LOW_VOL_HIGH_VAR 3059     FLAT_CHOPPY      0.475319  7.077721  0.536777
 5m    L2_5m_vol_sigma_w  L2_5m_price_sigma_w LOW_VOL_HIGH_VAR  710     FLAT_CHOPPY      0.504225  6.865493  0.522535
 1h   L1_1h_vol_accel_1b  L2_1h_price_sigma_w LOW_VOL_HIGH_VAR 5737     FLAT_CHOPPY      0.397420  6.689995  0.548370
 5m     L2_5m_vol_mean_w      L1_5m_bar_range LOW_VOL_HIGH_VAR  675     FLAT_CHOPPY      0.557037  6.121111  0.527407
 4h L2_4h_vol_velocity_w      L1_4h_bar_range LOW_VOL_HIGH_VAR 4033     FLAT_CHOPPY      0.345400  5.442475  0.512770
15m    L2_15m_vol_mean_w     L1_15m_bar_range LOW_VOL_HIGH_VAR  835     FLAT_CHOPPY      0.532934  5.308982  0.566467
 1h    L2_1h_vol_sigma_w      L1_1h_bar_range LOW_VOL_HIGH_VAR 3227     FLAT_CHOPPY      0.498915  4.953750  0.515029
 1h L2_1h_vol_velocity_w  L2_1h_price_sigma_w LOW_VOL_HIGH_VAR 4140     FLAT_CHOPPY      0.370773  4.718478  0.530193
 1h    L2_1h_vol_accel_w  L2_1h_price_sigma_w LOW_VOL_HIGH_VAR 7058     FLAT_CHOPPY      0.380986  4.489126  0.536271

## HIGH_VOL x LOW_VAR top 15 (compression / breakout candidates)

 tf           vol_feature          var_feature           corner    n dominant_regime  dominant_pct  mean_fwd  win_rate
 4h      L2_4h_vol_mean_w  L3_4h_swing_noise_w HIGH_VOL_LOW_VAR  636     FLAT_CHOPPY      0.698113  9.617138  0.553459
 4h      L2_4h_vol_mean_w  L2_4h_price_sigma_w HIGH_VOL_LOW_VAR  900     FLAT_CHOPPY      0.466667  8.196944  0.490000
 4h     L2_4h_vol_sigma_w  L2_4h_price_sigma_w HIGH_VOL_LOW_VAR 1188     FLAT_CHOPPY      0.525253  7.892045  0.533670
 4h     L2_4h_vol_sigma_w  L3_4h_swing_noise_w HIGH_VOL_LOW_VAR  636     FLAT_CHOPPY      0.698113  6.046384  0.599057
15m     L2_15m_vol_mean_w L2_15m_price_sigma_w HIGH_VOL_LOW_VAR  991     FLAT_SMOOTH      0.317861  5.734864  0.544904
 4h L1_4h_vol_velocity_1b      L1_4h_bar_range HIGH_VOL_LOW_VAR  804     FLAT_SMOOTH      0.343284  4.856654  0.554726
 1h L1_1h_vol_velocity_1b      L1_1h_bar_range HIGH_VOL_LOW_VAR 1752     FLAT_CHOPPY      0.349315  4.340753  0.531393
 4h      L2_4h_vol_mean_w      L1_4h_bar_range HIGH_VOL_LOW_VAR 1884     FLAT_CHOPPY      0.407643  3.492304  0.534501
 4h     L2_4h_vol_sigma_w      L1_4h_bar_range HIGH_VOL_LOW_VAR 2328     FLAT_CHOPPY      0.500000  3.334944  0.560567
 1h     L2_1h_vol_sigma_w      L1_1h_bar_range HIGH_VOL_LOW_VAR 2978     FLAT_CHOPPY      0.423439  2.499832  0.532236
15m     L2_15m_vol_mean_w     L1_15m_bar_range HIGH_VOL_LOW_VAR 1524     FLAT_CHOPPY      0.412073  2.482119  0.530840
 1h      L2_1h_vol_mean_w      L1_1h_bar_range HIGH_VOL_LOW_VAR 3652     FLAT_CHOPPY      0.411555  2.166073  0.529025
 1h L1_1h_vol_velocity_1b  L2_1h_price_sigma_w HIGH_VOL_LOW_VAR 5643     FLAT_CHOPPY      0.359383  1.834574  0.547049
 1h    L1_1h_vol_accel_1b      L1_1h_bar_range HIGH_VOL_LOW_VAR 2291     FLAT_CHOPPY      0.350502  1.712134  0.532082
 4h     L2_4h_vol_accel_w  L2_4h_price_sigma_w HIGH_VOL_LOW_VAR 4643     FLAT_CHOPPY      0.418479  1.698955  0.535645
