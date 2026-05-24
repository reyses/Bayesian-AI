**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# V2 feature chords (3-feature regime fingerprints) — 2026-05-03 16:15 UTC

**Base TF:** `5m`  **Split:** `IS`
**Top-K:** 10 -> C(K,3) = 120 triplets
**Quantiles:** 3
**Min cell support:** 50 bars

## Top chords by purity

Higher purity = cells reliably map to one specific regime (the chord encodes a regime fingerprint).

                     x                       y                       z  chord_purity  chord_signal  chord_entropy  n_cells
            L1_4h_body L1_4h_price_velocity_1b           L3_4h_z_low_w      0.546164     10.718750       1.502313       17
          L3_4h_z_se_w              L1_4h_body L1_4h_price_velocity_1b      0.545258     10.718750       1.516497       17
L2_1h_price_velocity_w              L1_4h_body L1_4h_price_velocity_1b      0.522987     10.180000       1.614515       16
            L1_4h_body L1_4h_price_velocity_1b L2_15m_price_velocity_w      0.517338      5.141442       1.628003       16
            L1_4h_body L1_4h_price_velocity_1b       L2_1D_vol_sigma_w      0.505941     10.718750       1.690694       14
          L3_4h_z_se_w          L3_4h_z_high_w           L3_4h_z_low_w      0.488628     25.642593       1.775076       20
          L3_4h_z_se_w  L2_1D_price_velocity_w          L3_4h_z_high_w      0.485724     31.829545       1.788287       24
            L1_4h_body L1_4h_price_velocity_1b          L3_4h_z_high_w      0.480732      9.146214       1.754231       14
L2_1h_price_velocity_w            L3_4h_z_se_w           L3_4h_z_low_w      0.473977     14.904225       1.832972       24
L2_1h_price_velocity_w            L3_4h_z_se_w  L2_1D_price_velocity_w      0.465873     24.138502       1.903447       27
            L1_4h_body L1_4h_price_velocity_1b  L2_1D_price_velocity_w      0.465735     10.123750       1.802234       14
L2_1h_price_velocity_w            L3_4h_z_se_w  L2_4h_price_velocity_w      0.465659     30.443268       1.922394       27
            L1_4h_body L1_4h_price_velocity_1b  L2_4h_price_velocity_w      0.465189     10.123750       1.821367       14
          L3_4h_z_se_w  L2_1D_price_velocity_w           L3_4h_z_low_w      0.464763     26.865000       1.864820       24
          L3_4h_z_se_w L1_4h_price_velocity_1b           L3_4h_z_low_w      0.464373     17.286250       1.854192       22

## Top chords by signal

Max cell |mean_fwd| × min(n/200, 1). Higher = at least one cell has a strong, well-supported price reaction.

                      x                       y                       z  chord_purity  chord_signal  chord_entropy  n_cells
 L2_1h_price_velocity_w  L2_4h_price_velocity_w       L2_1D_vol_sigma_w      0.433601     45.736655       1.994135       27
           L3_4h_z_se_w          L3_4h_z_high_w  L2_4h_price_velocity_w      0.453403     43.642500       1.912142       24
 L2_1h_price_velocity_w  L2_1D_price_velocity_w           L3_4h_z_low_w      0.435500     37.641705       1.959816       27
           L3_4h_z_se_w          L3_4h_z_high_w L2_15m_price_velocity_w      0.448929     36.486407       2.001403       23
 L2_1h_price_velocity_w  L2_4h_price_velocity_w           L3_4h_z_low_w      0.446145     35.313139       1.956257       27
           L3_4h_z_se_w          L3_4h_z_high_w       L2_1D_vol_sigma_w      0.463673     33.585648       1.891693       24
           L3_4h_z_se_w  L2_1D_price_velocity_w          L3_4h_z_high_w      0.485724     31.829545       1.788287       24
 L2_1h_price_velocity_w            L3_4h_z_se_w          L3_4h_z_high_w      0.451614     31.140741       1.933225       22
 L2_1h_price_velocity_w          L3_4h_z_high_w  L2_4h_price_velocity_w      0.444489     30.534361       1.989327       27
 L2_1h_price_velocity_w            L3_4h_z_se_w  L2_4h_price_velocity_w      0.465659     30.443268       1.922394       27
L1_4h_price_velocity_1b  L2_4h_price_velocity_w       L2_1D_vol_sigma_w      0.437947     29.706186       2.010560       27
             L1_4h_body  L2_4h_price_velocity_w       L2_1D_vol_sigma_w      0.441773     29.706186       2.005090       27
         L3_4h_z_high_w L2_15m_price_velocity_w           L3_4h_z_low_w      0.440875     29.438429       2.012230       24
 L2_1h_price_velocity_w          L3_4h_z_high_w           L3_4h_z_low_w      0.464292     29.121454       1.844500       25
           L3_4h_z_se_w L1_4h_price_velocity_1b          L3_4h_z_high_w      0.439510     28.255952       1.929146       21
