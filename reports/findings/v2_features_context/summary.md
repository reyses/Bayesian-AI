# V2 features contextualization EDA — 2026-05-03 17:07 UTC

For each (modifier, target, TF) pair: bin modifier into 4 quantiles; within each bin, compute the target's correlation with forward return. The PAIR's score = range of target_corr_fwd values across modifier bins.

**TFs:** ['1h']  **Quantiles:** 4  **Forward N:** 12  **Split:** IS

## Top 30 (modifier x target x TF) by context_lift_corr

tf         modifier            target  n_bins  min_target_corr_fwd  max_target_corr_fwd  context_lift_corr  context_amplification
1h   vol_velocity_w     price_accel_w       4            -0.034138             0.126545           0.160682               0.109236
1h      vol_sigma_w     price_accel_w       4            -0.069524             0.083433           0.152958               0.071442
1h      vol_sigma_w              body       4            -0.108286             0.038888           0.147174               0.106410
1h   vol_velocity_w  price_velocity_w       4            -0.055010             0.088736           0.143746               0.080783
1h      vol_sigma_w price_velocity_1b       4            -0.099081             0.038070           0.137152               0.097252
1h    swing_noise_w              body       4            -0.086817             0.047556           0.134373               0.082394
1h      vol_accel_w     price_accel_w       4            -0.048191             0.084665           0.132855               0.066025
1h   vol_velocity_w          z_high_w       4            -0.067626             0.065196           0.132823               0.067418
1h   vol_velocity_w    price_accel_1b       4            -0.039825             0.092853           0.132677               0.087936
1h    swing_noise_w            z_se_w       4            -0.054206             0.078384           0.132590               0.075720
1h      vol_accel_w  price_velocity_w       4            -0.063919             0.067752           0.131671               0.063650
1h    swing_noise_w price_velocity_1b       4            -0.081522             0.046987           0.128509               0.079098
1h      vol_accel_w              body       4            -0.103973             0.024137           0.128110               0.092796
1h      vol_accel_w    price_accel_1b       4            -0.102442             0.023846           0.126288               0.093100
1h      vol_accel_w price_velocity_1b       4            -0.101212             0.019281           0.120493               0.088471
1h    swing_noise_w          z_high_w       4            -0.035732             0.082189           0.117921               0.078918
1h    swing_noise_w     price_accel_w       4            -0.059069             0.048900           0.107968               0.047364
1h        bar_range          z_high_w       4            -0.047950             0.059396           0.107346               0.057926
1h          hurst_w     price_accel_w       4            -0.011696             0.094768           0.106464               0.091496
1h    swing_noise_w           z_low_w       4            -0.032150             0.072661           0.104812               0.052988
1h          hurst_w              body       4            -0.044598             0.055445           0.100043               0.048789
1h   vol_velocity_w           z_low_w       4            -0.069069             0.028966           0.098036               0.056161
1h          hurst_w price_velocity_1b       4            -0.042937             0.052955           0.095892               0.046023
1h   vol_velocity_w            z_se_w       4            -0.055569             0.039122           0.094690               0.046045
1h    price_sigma_w          z_high_w       4            -0.034254             0.057522           0.091776               0.044498
1h reversion_prob_w     price_accel_w       4             0.002078             0.087146           0.085068               0.085068
1h       vol_mean_w     price_accel_w       4            -0.015693             0.067858           0.083551               0.064256
1h          hurst_w    price_accel_1b       4            -0.049628             0.033104           0.082732               0.044645
1h      vol_sigma_w    price_accel_1b       4            -0.049486             0.032066           0.081552               0.046301
1h    swing_noise_w    price_accel_1b       4            -0.078425             0.000558           0.078983               0.077868

## Top 30 by amplification

tf         modifier            target  n_bins  min_target_corr_fwd  max_target_corr_fwd  context_lift_corr  context_amplification
1h   vol_velocity_w     price_accel_w       4            -0.034138             0.126545           0.160682               0.109236
1h      vol_sigma_w              body       4            -0.108286             0.038888           0.147174               0.106410
1h      vol_sigma_w price_velocity_1b       4            -0.099081             0.038070           0.137152               0.097252
1h      vol_accel_w    price_accel_1b       4            -0.102442             0.023846           0.126288               0.093100
1h      vol_accel_w              body       4            -0.103973             0.024137           0.128110               0.092796
1h          hurst_w     price_accel_w       4            -0.011696             0.094768           0.106464               0.091496
1h      vol_accel_w price_velocity_1b       4            -0.101212             0.019281           0.120493               0.088471
1h   vol_velocity_w    price_accel_1b       4            -0.039825             0.092853           0.132677               0.087936
1h reversion_prob_w     price_accel_w       4             0.002078             0.087146           0.085068               0.085068
1h    swing_noise_w              body       4            -0.086817             0.047556           0.134373               0.082394
1h   vol_velocity_w  price_velocity_w       4            -0.055010             0.088736           0.143746               0.080783
1h    swing_noise_w price_velocity_1b       4            -0.081522             0.046987           0.128509               0.079098
1h    swing_noise_w          z_high_w       4            -0.035732             0.082189           0.117921               0.078918
1h    swing_noise_w    price_accel_1b       4            -0.078425             0.000558           0.078983               0.077868
1h    swing_noise_w            z_se_w       4            -0.054206             0.078384           0.132590               0.075720
1h      vol_sigma_w     price_accel_w       4            -0.069524             0.083433           0.152958               0.071442
1h   vol_velocity_w          z_high_w       4            -0.067626             0.065196           0.132823               0.067418
1h      vol_accel_w     price_accel_w       4            -0.048191             0.084665           0.132855               0.066025
1h       vol_mean_w     price_accel_w       4            -0.015693             0.067858           0.083551               0.064256
1h      vol_accel_w  price_velocity_w       4            -0.063919             0.067752           0.131671               0.063650
1h        bar_range    price_accel_1b       4            -0.065526             0.007051           0.072577               0.062362
1h        bar_range          z_high_w       4            -0.047950             0.059396           0.107346               0.057926
1h   vol_velocity_w           z_low_w       4            -0.069069             0.028966           0.098036               0.056161
1h reversion_prob_w    price_accel_1b       4            -0.056031             0.016156           0.072187               0.054858
1h    swing_noise_w           z_low_w       4            -0.032150             0.072661           0.104812               0.052988
