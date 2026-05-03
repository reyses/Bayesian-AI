# V2 Pairwise Feature × Price EDA — 2026-05-03 07:50 UTC

**Base TF:** `5m`
**Split:** `IS`
**Top-K from Layer 1:** 20 (ranked by `lookback_corr`)
**Quantiles:** 3 per feature
**Forward N:** 12 bars (60 min)

## Layer 1 shortlist

- `L2_5m_price_velocity_w`
- `L2_1m_price_velocity_w`
- `L1_15m_price_velocity_1b`
- `L2_15m_price_velocity_w`
- `L1_15m_body`
- `L3_15m_z_se_w`
- `L3_15m_z_high_w`
- `L3_15m_z_low_w`
- `L1_1h_body`
- `L1_1h_price_velocity_1b`
- `L2_15m_price_accel_w`
- `L1_5m_price_velocity_1b`
- `L1_5m_body`
- `L1_1h_price_accel_1b`
- `L2_1h_price_accel_w`
- `L3_1h_z_se_w`
- `L2_15s_price_velocity_w`
- `L2_1h_price_velocity_w`
- `L3_15m_SE_high_w`
- `L1_1m_body`

## Top pairs by interaction RMS

Higher = more non-additive structure (the pair tells you more than the sum of its marginals).

               x_feature               y_feature                                             pair  interaction_rms  max_abs_mean_fwd  max_win_rate  min_win_rate  wr_spread  n_total
              L1_1h_body L1_1h_price_velocity_1b           L1_1h_body__x__L1_1h_price_velocity_1b        12.206036         30.166667      0.604348      0.375000   0.229348    46893
 L1_5m_price_velocity_1b              L1_5m_body           L1_5m_price_velocity_1b__x__L1_5m_body        10.359014         31.031250      0.750000      0.416667   0.333333    46915
L1_15m_price_velocity_1b             L1_15m_body         L1_15m_price_velocity_1b__x__L1_15m_body         5.163709         10.461443      0.666667      0.318182   0.348485    46911
    L1_1h_price_accel_1b        L3_15m_SE_high_w        L1_1h_price_accel_1b__x__L3_15m_SE_high_w         4.119034          9.433777      0.544688      0.468512   0.076175    46881
 L2_15m_price_velocity_w     L2_1h_price_accel_w  L2_15m_price_velocity_w__x__L2_1h_price_accel_w         2.731701          6.963141      0.537377      0.483668   0.053709    46749
              L1_1h_body     L2_1h_price_accel_w               L1_1h_body__x__L2_1h_price_accel_w         2.582305          6.517188      0.566320      0.494494   0.071826    46749
           L3_15m_z_se_w          L3_15m_z_low_w                 L3_15m_z_se_w__x__L3_15m_z_low_w         2.528002          8.080242      0.577982      0.497444   0.080537    46881
          L3_15m_z_low_w    L1_1h_price_accel_1b          L3_15m_z_low_w__x__L1_1h_price_accel_1b         2.499619          5.950518      0.552041      0.487827   0.064214    46881
 L2_15m_price_velocity_w              L1_1h_body           L2_15m_price_velocity_w__x__L1_1h_body         2.424774          5.875101      0.551894      0.478663   0.073231    46878
 L2_15m_price_velocity_w    L1_1h_price_accel_1b L2_15m_price_velocity_w__x__L1_1h_price_accel_1b         2.394720          4.667361      0.558183      0.489792   0.068392    46878
 L1_1h_price_velocity_1b     L2_1h_price_accel_w  L1_1h_price_velocity_1b__x__L2_1h_price_accel_w         2.348847          5.303617      0.558389      0.502416   0.055973    46749
           L3_15m_z_se_w         L3_15m_z_high_w                L3_15m_z_se_w__x__L3_15m_z_high_w         2.339131          4.475490      0.542640      0.495992   0.046648    46881

## Top pairs by WR spread

WR spread = max_cell_WR − min_cell_WR. Larger = pair more strongly differentiates winning from losing bars.

               x_feature               y_feature                                      pair  interaction_rms  max_abs_mean_fwd  max_win_rate  min_win_rate  wr_spread  n_total
L1_15m_price_velocity_1b             L1_15m_body  L1_15m_price_velocity_1b__x__L1_15m_body         5.163709         10.461443      0.666667      0.318182   0.348485    46911
 L1_5m_price_velocity_1b              L1_5m_body    L1_5m_price_velocity_1b__x__L1_5m_body        10.359014         31.031250      0.750000      0.416667   0.333333    46915
              L1_1h_body L1_1h_price_velocity_1b    L1_1h_body__x__L1_1h_price_velocity_1b        12.206036         30.166667      0.604348      0.375000   0.229348    46893
              L1_1h_body            L3_1h_z_se_w               L1_1h_body__x__L3_1h_z_se_w         1.121527          2.977273      0.566366      0.440476   0.125889    46773
 L1_1h_price_velocity_1b            L3_1h_z_se_w  L1_1h_price_velocity_1b__x__L3_1h_z_se_w         1.061270          2.438051      0.553609      0.444367   0.109242    46773
           L3_15m_z_se_w          L3_15m_z_low_w          L3_15m_z_se_w__x__L3_15m_z_low_w         2.528002          8.080242      0.577982      0.497444   0.080537    46881
              L1_1h_body    L1_1h_price_accel_1b       L1_1h_body__x__L1_1h_price_accel_1b         1.811361          3.540068      0.567333      0.490092   0.077241    46881
    L1_1h_price_accel_1b        L3_15m_SE_high_w L1_1h_price_accel_1b__x__L3_15m_SE_high_w         4.119034          9.433777      0.544688      0.468512   0.076175    46881
  L2_5m_price_velocity_w           L3_15m_z_se_w  L2_5m_price_velocity_w__x__L3_15m_z_se_w         2.100183          8.274099      0.558909      0.485138   0.073772    46881
 L2_15m_price_velocity_w              L1_1h_body    L2_15m_price_velocity_w__x__L1_1h_body         2.424774          5.875101      0.551894      0.478663   0.073231    46878
              L1_1h_body     L2_1h_price_accel_w        L1_1h_body__x__L2_1h_price_accel_w         2.582305          6.517188      0.566320      0.494494   0.071826    46749
            L3_1h_z_se_w  L2_1h_price_velocity_w   L3_1h_z_se_w__x__L2_1h_price_velocity_w         1.543456          5.355577      0.547438      0.477051   0.070387    46761
