# Cross-TF same-concept (Layer Cross-TF 1) - 2026-05-03 22:49 UTC

## Term-structure smoothness ranking

          concept  mean_adjacent_TF_corr  mean_far_TF_corr     decay  n_pos_offdiag  n_neg_offdiag  n_zero_offdiag  has_sign_flips
       vol_mean_w               0.760112          0.168979  0.591133             28              0               0           False
      vol_sigma_w               0.648456          0.112925  0.535531             28              0               0           False
 price_velocity_w               0.496845         -0.001294  0.498138             14              0              14           False
    swing_noise_w               0.753217          0.405779  0.347438             28              0               0           False
         SE_low_w               0.631912          0.315586  0.316327             28              0               0           False
    price_sigma_w               0.652070          0.336941  0.315129             28              0               0           False
        SE_high_w               0.663555          0.349680  0.313875             28              0               0           False
   vol_velocity_w               0.310999          0.012917  0.298081              9              4              15            True
        bar_range               0.705795          0.448672  0.257123             28              0               0           False
             body               0.245263          0.001038  0.244226             10              2              16            True
price_velocity_1b               0.242154          0.002616  0.239538             10              1              17            True
    price_accel_w               0.127066          0.002408  0.124659              7              4              17            True
           z_se_w               0.107271          0.004005  0.103266              4              7              17            True
  vol_velocity_1b               0.098705         -0.000537  0.099242              5              4              19            True
      vol_accel_w               0.086492          0.010002  0.076490              4              2              22            True
 reversion_prob_w               0.073265          0.000670  0.072595              4              0              24           False
     vol_accel_1b               0.048626         -0.011530  0.060157              3              5              20            True
     price_mean_w               0.997779          0.985564  0.012214             28              0               0           False
           vwap_w               0.997830          0.986991  0.010839             28              0               0           False
   price_accel_1b              -0.067019          0.002458 -0.069477              0              5              23           False
         z_high_w              -0.094623          0.001604 -0.096227              1              8              19            True
          hurst_w              -0.112866          0.000127 -0.112992              1              7              20            True
          z_low_w              -0.114536          0.000472 -0.115008              1              8              19            True

**Decay** = mean_adjacent_TF_corr - mean_far_TF_corr.
High decay = smooth term structure (each TF only relates to its neighbors).
Low decay = rigid (the concept is consistent across all TF distances).
Has_sign_flips = some TF pairs have opposite-sign correlation to others — TF inversions exist.
