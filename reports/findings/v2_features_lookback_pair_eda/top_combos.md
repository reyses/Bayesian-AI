# Pairwise lookback combos — 2026-05-03 08:08 UTC

**Base TF:** `5m`  **Split:** `IS`  **Forward N:** 12
**Baseline WR:** 52.1%

## Top combos by WR lift

               x_feature               y_feature  window             pattern                                                 pair    n  freq_pct  win_rate  wr_lift  mean_fwd    std_fwd  mean_mfe  mean_mae
  L2_1m_price_velocity_w L1_5m_price_velocity_1b       6 REVERSAL_AGREE_DOWN   L2_1m_price_velocity_w__x__L1_5m_price_velocity_1b   31  0.066057  0.709677 0.188353 21.266129  48.176020 40.137097 29.435484
  L2_5m_price_velocity_w  L2_1m_price_velocity_w       6        BOTH_FALLING    L2_5m_price_velocity_w__x__L2_1m_price_velocity_w   60  0.127853  0.616667 0.095342 10.745833  65.119487 48.895833 45.291667
             L1_15m_body L1_1h_price_velocity_1b       6     SPIKE_BOTH_DOWN              L1_15m_body__x__L1_1h_price_velocity_1b  177  0.377166  0.598870 0.077545  5.593220  69.338620 41.871469 41.278249
  L2_5m_price_velocity_w L1_5m_price_velocity_1b       6   REVERSAL_AGREE_UP   L2_5m_price_velocity_w__x__L1_5m_price_velocity_1b   32  0.068188  0.593750 0.072425 -0.437500  48.517872 33.687500 43.968750
          L3_15m_z_low_w    L2_15m_price_accel_w      12       SPIKE_BOTH_UP              L3_15m_z_low_w__x__L2_15m_price_accel_w  562  1.199685  0.583630 0.062305  4.138790  65.762021 41.185943 40.591192
L1_15m_price_velocity_1b L1_1h_price_velocity_1b       6     SPIKE_BOTH_DOWN L1_15m_price_velocity_1b__x__L1_1h_price_velocity_1b  179  0.381427  0.581006 0.059681  4.744413  69.050268 41.765363 42.226257
             L1_15m_body              L1_1h_body       6     SPIKE_BOTH_DOWN                           L1_15m_body__x__L1_1h_body  179  0.381427  0.581006 0.059681  4.286313  69.275404 41.093575 42.114525
L1_15m_price_velocity_1b L1_1h_price_velocity_1b      12     SPIKE_BOTH_DOWN L1_15m_price_velocity_1b__x__L1_1h_price_velocity_1b  224  0.477317  0.571429 0.050104  5.341518  81.063358 48.632812 50.426339
L1_15m_price_velocity_1b              L1_1h_body      12     SPIKE_BOTH_DOWN              L1_15m_price_velocity_1b__x__L1_1h_body  226  0.481579  0.570796 0.049472  5.112832  80.790016 48.280973 50.384956
             L1_15m_body              L1_1h_body      12     SPIKE_BOTH_DOWN                           L1_15m_body__x__L1_1h_body  232  0.494364  0.568966 0.047641  4.252155  80.468192 47.092672 50.468750
             L1_15m_body L1_1h_price_velocity_1b      12     SPIKE_BOTH_DOWN              L1_15m_body__x__L1_1h_price_velocity_1b  229  0.487971  0.567686 0.046361  4.593886  80.941113 47.665939 50.936681
  L2_5m_price_velocity_w              L1_1h_body       6     SPIKE_BOTH_DOWN                L2_5m_price_velocity_w__x__L1_1h_body  271  0.577468  0.564576 0.043251  7.060886  60.625295 41.489852 40.819188
L1_15m_price_velocity_1b              L1_1h_body       6     SPIKE_BOTH_DOWN              L1_15m_price_velocity_1b__x__L1_1h_body  181  0.385689  0.563536 0.042211  3.461326  68.975085 40.997238 43.042818
  L2_5m_price_velocity_w L1_1h_price_velocity_1b       6     SPIKE_BOTH_DOWN   L2_5m_price_velocity_w__x__L1_1h_price_velocity_1b  272  0.579599  0.562500 0.041175  6.603860  59.734113 40.948529 40.714154
    L2_15m_price_accel_w L1_5m_price_velocity_1b       6       SPIKE_BOTH_UP     L2_15m_price_accel_w__x__L1_5m_price_velocity_1b  514  1.095272  0.562257 0.040932  4.438716  78.422744 38.445525 38.363813
              L1_1h_body    L2_15m_price_accel_w      12       SPIKE_BOTH_UP                  L1_1h_body__x__L2_15m_price_accel_w  185  0.394213  0.562162 0.040837  5.424324  75.208485 46.248649 35.390541
 L2_15m_price_velocity_w L1_5m_price_velocity_1b       6       SPIKE_BOTH_UP  L2_15m_price_velocity_w__x__L1_5m_price_velocity_1b  493  1.050523  0.559838 0.038513  5.362069  88.418334 41.627282 38.769270
         L3_15m_z_high_w    L2_15m_price_accel_w      12       SPIKE_BOTH_UP             L3_15m_z_high_w__x__L2_15m_price_accel_w  889  1.898613  0.559055 0.037730  6.463161  72.825917 44.762936 40.053993
 L1_1h_price_velocity_1b L1_5m_price_velocity_1b      12       SPIKE_BOTH_UP  L1_1h_price_velocity_1b__x__L1_5m_price_velocity_1b  213  0.453877  0.558685 0.037361 13.726526 101.576907 43.960094 32.038732
              L1_1h_body L1_5m_price_velocity_1b      12       SPIKE_BOTH_UP               L1_1h_body__x__L1_5m_price_velocity_1b  217  0.462401  0.557604 0.036279 13.807604 100.781025 43.979263 31.788018
    L2_15m_price_accel_w L1_5m_price_velocity_1b      12       SPIKE_BOTH_UP     L2_15m_price_accel_w__x__L1_5m_price_velocity_1b  244  0.519934  0.557377 0.036052  7.088115 102.887491 45.372951 41.827869
 L1_1h_price_velocity_1b    L2_15m_price_accel_w      12       SPIKE_BOTH_UP     L1_1h_price_velocity_1b__x__L2_15m_price_accel_w  183  0.389951  0.557377 0.036052  6.355191  76.418595 47.125683 35.114754
 L2_15m_price_velocity_w              L1_1h_body       6     SPIKE_BOTH_DOWN               L2_15m_price_velocity_w__x__L1_1h_body  300  0.639264  0.556667 0.035342  5.967500  71.742042 43.840000 40.826667
 L2_15m_price_velocity_w              L1_1h_body      12     SPIKE_BOTH_DOWN               L2_15m_price_velocity_w__x__L1_1h_body  408  0.869398  0.551471 0.030146  7.272059  69.320042 43.636642 41.525123
  L2_5m_price_velocity_w    L2_15m_price_accel_w      12       SPIKE_BOTH_UP      L2_5m_price_velocity_w__x__L2_15m_price_accel_w  818  1.747320  0.551345 0.030020  0.552873  87.863622 41.742359 43.040954
  L2_5m_price_velocity_w              L1_1h_body      12     SPIKE_BOTH_DOWN                L2_5m_price_velocity_w__x__L1_1h_body  494  1.052654  0.550607 0.029282  3.188765  63.983170 39.719130 42.315283
  L2_5m_price_velocity_w L1_1h_price_velocity_1b      12     SPIKE_BOTH_DOWN   L2_5m_price_velocity_w__x__L1_1h_price_velocity_1b  494  1.052654  0.550607 0.029282  2.578947  63.206671 39.005061 42.313765
 L2_15m_price_velocity_w L1_1h_price_velocity_1b       6     SPIKE_BOTH_DOWN  L2_15m_price_velocity_w__x__L1_1h_price_velocity_1b  298  0.635002  0.550336 0.029011  6.429530  70.848080 43.959732 40.331376
 L2_15m_price_velocity_w         L3_15m_z_high_w      12       SPIKE_BOTH_UP          L2_15m_price_velocity_w__x__L3_15m_z_high_w 1040  2.216114  0.547115 0.025790  6.076442  69.908050 43.818269 40.991346
 L2_15m_price_velocity_w L1_1h_price_velocity_1b      12     SPIKE_BOTH_DOWN  L2_15m_price_velocity_w__x__L1_1h_price_velocity_1b  405  0.863006  0.545679 0.024354  7.603704  68.624422 43.702469 41.077160

## Top combos by |mean_fwd|

               x_feature               y_feature  window             pattern                                                pair    n  freq_pct  win_rate   wr_lift   mean_fwd    std_fwd  mean_mfe  mean_mae   abs_fwd
  L2_5m_price_velocity_w L1_5m_price_velocity_1b       6 REVERSAL_AGREE_DOWN  L2_5m_price_velocity_w__x__L1_5m_price_velocity_1b   14  0.029832  0.428571 -0.092753  26.410714  74.717648 55.285714 31.696429 26.410714
  L2_1m_price_velocity_w L1_5m_price_velocity_1b       6 REVERSAL_AGREE_DOWN  L2_1m_price_velocity_w__x__L1_5m_price_velocity_1b   31  0.066057  0.709677  0.188353  21.266129  48.176020 40.137097 29.435484 21.266129
  L2_5m_price_velocity_w  L2_1m_price_velocity_w       6         BOTH_RISING   L2_5m_price_velocity_w__x__L2_1m_price_velocity_w   50  0.106544  0.520000 -0.001325 -16.980000  90.463896 37.290000 56.290000 16.980000
              L1_1h_body L1_5m_price_velocity_1b      12       SPIKE_BOTH_UP              L1_1h_body__x__L1_5m_price_velocity_1b  217  0.462401  0.557604  0.036279  13.807604 100.781025 43.979263 31.788018 13.807604
 L1_1h_price_velocity_1b L1_5m_price_velocity_1b      12       SPIKE_BOTH_UP L1_1h_price_velocity_1b__x__L1_5m_price_velocity_1b  213  0.453877  0.558685  0.037361  13.726526 101.576907 43.960094 32.038732 13.726526
 L1_1h_price_velocity_1b L1_5m_price_velocity_1b       6       SPIKE_BOTH_UP L1_1h_price_velocity_1b__x__L1_5m_price_velocity_1b  191  0.406998  0.528796  0.007471  13.705497 108.088700 45.303665 33.400524 13.705497
              L1_1h_body L1_5m_price_velocity_1b       6       SPIKE_BOTH_UP              L1_1h_body__x__L1_5m_price_velocity_1b  193  0.411260  0.523316  0.001991  13.546632 107.805346 45.532383 33.734456 13.546632
  L2_5m_price_velocity_w  L2_1m_price_velocity_w       6        BOTH_FALLING   L2_5m_price_velocity_w__x__L2_1m_price_velocity_w   60  0.127853  0.616667  0.095342  10.745833  65.119487 48.895833 45.291667 10.745833
  L2_1m_price_velocity_w L1_5m_price_velocity_1b       6   REVERSAL_AGREE_UP  L2_1m_price_velocity_w__x__L1_5m_price_velocity_1b   32  0.068188  0.531250  0.009925  -8.031250  86.756257 45.414062 54.460938  8.031250
  L2_1m_price_velocity_w         L3_15m_z_high_w      12     SPIKE_BOTH_DOWN          L2_1m_price_velocity_w__x__L3_15m_z_high_w  436  0.929063  0.486239 -0.035086  -7.845183  62.417975 36.319954 51.303899  7.845183
 L1_1h_price_velocity_1b    L2_15m_price_accel_w       6     SPIKE_BOTH_DOWN    L1_1h_price_velocity_1b__x__L2_15m_price_accel_w  187  0.398474  0.545455  0.024130   7.712567  66.802997 43.768717 43.457219  7.712567
 L2_15m_price_velocity_w L1_1h_price_velocity_1b      12     SPIKE_BOTH_DOWN L2_15m_price_velocity_w__x__L1_1h_price_velocity_1b  405  0.863006  0.545679  0.024354   7.603704  68.624422 43.702469 41.077160  7.603704
           L3_15m_z_se_w    L2_15m_price_accel_w      12     SPIKE_BOTH_DOWN              L3_15m_z_se_w__x__L2_15m_price_accel_w 1182  2.518698  0.505076 -0.016249  -7.367386  82.300585 39.440778 53.506345  7.367386
 L2_15m_price_velocity_w              L1_1h_body      12     SPIKE_BOTH_DOWN              L2_15m_price_velocity_w__x__L1_1h_body  408  0.869398  0.551471  0.030146   7.272059  69.320042 43.636642 41.525123  7.272059
    L2_15m_price_accel_w L1_5m_price_velocity_1b      12       SPIKE_BOTH_UP    L2_15m_price_accel_w__x__L1_5m_price_velocity_1b  244  0.519934  0.557377  0.036052   7.088115 102.887491 45.372951 41.827869  7.088115
  L2_5m_price_velocity_w              L1_1h_body       6     SPIKE_BOTH_DOWN               L2_5m_price_velocity_w__x__L1_1h_body  271  0.577468  0.564576  0.043251   7.060886  60.625295 41.489852 40.819188  7.060886
  L2_5m_price_velocity_w L1_1h_price_velocity_1b       6     SPIKE_BOTH_DOWN  L2_5m_price_velocity_w__x__L1_1h_price_velocity_1b  272  0.579599  0.562500  0.041175   6.603860  59.734113 40.948529 40.714154  6.603860
             L1_15m_body    L2_15m_price_accel_w      12     SPIKE_BOTH_DOWN                L1_15m_body__x__L2_15m_price_accel_w 1315  2.802105  0.506464 -0.014861  -6.536312  81.195665 42.196578 54.988213  6.536312
         L3_15m_z_high_w    L2_15m_price_accel_w      12       SPIKE_BOTH_UP            L3_15m_z_high_w__x__L2_15m_price_accel_w  889  1.898613  0.559055  0.037730   6.463161  72.825917 44.762936 40.053993  6.463161
 L2_15m_price_velocity_w L1_1h_price_velocity_1b       6     SPIKE_BOTH_DOWN L2_15m_price_velocity_w__x__L1_1h_price_velocity_1b  298  0.635002  0.550336  0.029011   6.429530  70.848080 43.959732 40.331376  6.429530
 L2_15m_price_velocity_w L1_1h_price_velocity_1b      12       SPIKE_BOTH_UP L2_15m_price_velocity_w__x__L1_1h_price_velocity_1b  356  0.758593  0.544944  0.023619   6.378511  85.060381 42.738062 39.365871  6.378511
L1_15m_price_velocity_1b    L2_15m_price_accel_w      12     SPIKE_BOTH_DOWN   L1_15m_price_velocity_1b__x__L2_15m_price_accel_w 1318  2.808498  0.506829 -0.014496  -6.365706  81.224699 42.227807 54.735774  6.365706
 L1_1h_price_velocity_1b    L2_15m_price_accel_w      12       SPIKE_BOTH_UP    L1_1h_price_velocity_1b__x__L2_15m_price_accel_w  183  0.389951  0.557377  0.036052   6.355191  76.418595 47.125683 35.114754  6.355191
  L2_5m_price_velocity_w  L2_1m_price_velocity_w       6   REVERSAL_AGREE_UP   L2_5m_price_velocity_w__x__L2_1m_price_velocity_w  137  0.291930  0.452555 -0.068770  -6.301095  59.856048 37.532847 54.260949  6.301095
              L1_1h_body    L2_15m_price_accel_w       6     SPIKE_BOTH_DOWN                 L1_1h_body__x__L2_15m_price_accel_w  191  0.406998  0.544503  0.023178   6.213351  68.158204 43.000000 44.689791  6.213351
