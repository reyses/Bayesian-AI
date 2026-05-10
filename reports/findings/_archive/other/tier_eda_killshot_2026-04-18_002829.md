# KILL_SHOT tier EDA — 2026-04-18_002829

KILL_SHOT classifier: `wick_5m > 0.83 AND wick_15m > 0.77 AND NOT h1_aligned`

Trades matched: **514** of 16,242 iso trades (3.2%)

## Segment summary

| Segment | N | MeanPnL | Total | MeanPeak | MedianPeak | MeanGiveback | MedHeld |
|---|---:|---:|---:|---:|---:|---:|---:|
| winner | 121 | $+18.14 | $+2,194 | $+33.62 | $+28.50 | $+15.48 | 5m |
| small_loser | 202 | $-1.21 | $-245 | $+14.02 | $+13.00 | $+15.23 | 2m |
| mid_loser | 131 | $-8.86 | $-1,161 | $+8.21 | $+7.50 | $+17.08 | 2m |
| tail_loser | 60 | $-28.13 | $-1,688 | $+5.80 | $+4.50 | $+33.93 | 4m |

## Exit reason by segment

- **winner**: `halflife_trail_1s`:93, `halflife_trail`:12, `fade_oscillation_decay`:10, `fade_oscillation_center`:4, `ride_regime_shift`:2
- **small_loser**: `halflife_trail_1s`:89, `fast_reverse_1s`:56, `halflife_trail`:28, `fade_oscillation_decay`:10, `fade_oscillation_center`:9
- **mid_loser**: `fast_reverse_1s`:72, `halflife_trail_1s`:28, `halflife_trail`:20, `fade_oscillation_center`:4, `fade_oscillation_decay`:3
- **tail_loser**: `fast_reverse_1s`:47, `halflife_trail`:7, `fade_oscillation_decay`:3, `fade_p_center`:2, `halflife_trail_1s`:1

## Top 30 feature separators (winner vs tail_loser, |Cohen d|)

| Feature | Winner median | Tail median | Cohen d |
|---|---:|---:|---:|
| `15s_bar_range` | 16.0000 | 12.0000 | +0.351 |
| `5m_z_high` | 0.7883 | 0.3466 | +0.346 |
| `15m_bar_range` | 69.0000 | 72.0000 | +0.319 |
| `5m_bar_range` | 46.0000 | 49.0000 | +0.309 |
| `1m_bar_range` | 40.0000 | 42.5000 | +0.309 |
| `1h_bar_range` | 192.0000 | 192.0000 | +0.305 |
| `1m_dmi_gap` | 9.8480 | 8.6627 | +0.304 |
| `1m_wick_ratio` | 0.2609 | 0.2361 | +0.303 |
| `15s_vol_rel` | 1.7815 | 1.5208 | +0.286 |
| `5m_velocity` | 0.0000 | 0.1250 | +0.249 |
| `1h_wick_ratio` | 0.5352 | 0.6319 | -0.238 |
| `15m_z_high` | 0.4372 | 0.1157 | +0.233 |
| `15s_wick_ratio` | 0.3333 | 0.4083 | -0.226 |
| `1h_vol_rel` | 0.4244 | 0.4309 | +0.221 |
| `5m_z_se` | 0.1154 | -0.2243 | +0.219 |
| `1h_reversion_prob` | 0.9814 | 0.9779 | -0.210 |
| `5m_wick_ratio` | 0.9286 | 0.9040 | +0.199 |
| `15s_z_high` | 0.4113 | 0.2062 | +0.194 |
| `1h_z_low` | -0.6264 | -0.1641 | -0.189 |
| `15m_wick_ratio` | 0.8838 | 0.8595 | +0.185 |
| `1h_dmi_gap` | 9.3347 | 10.0791 | -0.185 |
| `15s_acceleration` | -0.2500 | 0.7500 | -0.179 |
| `15s_reversion_prob` | 0.9172 | 0.9330 | -0.175 |
| `1D_vol_rel` | 1.0766 | 1.1146 | -0.164 |
| `1m_velocity` | -2.7500 | -4.0000 | +0.156 |
| `5m_dir_vol` | 0.0000 | 0.0346 | -0.155 |
| `1D_p_at_center` | 0.6319 | 0.5879 | +0.151 |
| `15m_p_at_center` | 0.5913 | 0.6387 | -0.149 |
| `15m_reversion_prob` | 0.9749 | 0.9791 | -0.146 |
| `15m_vol_rel` | 0.7972 | 0.8061 | +0.142 |

## Peak timing per segment

| Segment | Peak bar (median) | Peak→close bars (mean) |
|---|---:|---:|
| winner | 41 | 10.6 |
| small_loser | 12 | 17.3 |
| mid_loser | 0 | 26.3 |
| tail_loser | 0 | 55.0 |

## Regime shift: feature mean at entry → peak → exit

| Segment | Feature | Entry | Peak | Exit | Δ entry→exit |
|---|---|---:|---:|---:|---:|
| small_loser | `1m_wick_ratio` | 0.2658 | 0.4740 | 0.5129 | +0.2471 |
| small_loser | `5m_wick_ratio` | 0.9156 | 0.7437 | 0.6993 | -0.2163 |
| small_loser | `15m_wick_ratio` | 0.8844 | 0.8416 | 0.8165 | -0.0679 |
| small_loser | `1h_wick_ratio` | 0.5615 | 0.5652 | 0.5621 | +0.0006 |
| small_loser | `1m_variance_ratio` | 0.5347 | 0.6143 | 0.6041 | +0.0694 |
| small_loser | `5m_variance_ratio` | 0.3515 | 0.3592 | 0.3569 | +0.0054 |
| small_loser | `1m_p_at_center` | 0.0669 | 0.2841 | 0.4063 | +0.3394 |
| small_loser | `5m_p_at_center` | 0.6157 | 0.5541 | 0.5534 | -0.0623 |
| small_loser | `1m_z_se` | -0.1906 | -0.0778 | -0.0343 | +0.1563 |
| small_loser | `5m_z_se` | -0.1774 | -0.1718 | -0.2182 | -0.0408 |
| small_loser | `15m_z_se` | -0.0599 | -0.0524 | -0.0548 | +0.0050 |
| small_loser | `1h_z_se` | -0.0128 | -0.0177 | -0.0101 | +0.0027 |
| small_loser | `1m_velocity` | -0.4319 | 0.0804 | 0.3230 | +0.7550 |
| small_loser | `5m_velocity` | -0.1906 | -0.2599 | -0.4530 | -0.2624 |
| mid_loser | `1m_wick_ratio` | 0.2964 | 0.4232 | 0.5860 | +0.2896 |
| mid_loser | `5m_wick_ratio` | 0.9085 | 0.7989 | 0.6495 | -0.2590 |
| mid_loser | `15m_wick_ratio` | 0.8878 | 0.8484 | 0.8071 | -0.0806 |
| mid_loser | `1h_wick_ratio` | 0.5853 | 0.5883 | 0.5804 | -0.0048 |
| mid_loser | `1m_variance_ratio` | 0.5136 | 0.5591 | 0.6041 | +0.0904 |
| mid_loser | `5m_variance_ratio` | 0.3630 | 0.3636 | 0.3646 | +0.0016 |
| mid_loser | `1m_p_at_center` | 0.0693 | 0.2139 | 0.4123 | +0.3430 |
| mid_loser | `5m_p_at_center` | 0.6108 | 0.5735 | 0.5183 | -0.0925 |
| mid_loser | `1m_z_se` | 0.4717 | 0.3901 | 0.1181 | -0.3537 |
| mid_loser | `5m_z_se` | -0.1509 | -0.0381 | 0.0738 | +0.2248 |
| mid_loser | `15m_z_se` | 0.0118 | 0.0529 | 0.0547 | +0.0429 |
| mid_loser | `1h_z_se` | 0.0574 | 0.0504 | 0.0607 | +0.0033 |
| mid_loser | `1m_velocity` | 1.2366 | 0.5191 | -0.3664 | -1.6031 |
| mid_loser | `5m_velocity` | -0.4046 | 0.5878 | 1.0000 | +1.4046 |
| winner | `1m_wick_ratio` | 0.2913 | 0.4881 | 0.4887 | +0.1973 |
| winner | `5m_wick_ratio` | 0.9218 | 0.6741 | 0.6536 | -0.2683 |
| winner | `15m_wick_ratio` | 0.8870 | 0.8199 | 0.7947 | -0.0922 |
| winner | `1h_wick_ratio` | 0.5499 | 0.5543 | 0.5528 | +0.0029 |
| winner | `1m_variance_ratio` | 0.5274 | 0.6361 | 0.6443 | +0.1168 |
| winner | `5m_variance_ratio` | 0.3954 | 0.3915 | 0.3900 | -0.0053 |
| winner | `1m_p_at_center` | 0.0675 | 0.3889 | 0.4121 | +0.3447 |
| winner | `5m_p_at_center` | 0.6204 | 0.5842 | 0.5726 | -0.0478 |
| winner | `1m_z_se` | -0.1796 | -0.0097 | -0.0762 | +0.1034 |
| winner | `5m_z_se` | 0.0791 | 0.0651 | 0.0333 | -0.0458 |
| winner | `15m_z_se` | -0.0308 | -0.0614 | -0.0738 | -0.0429 |
| winner | `1h_z_se` | 0.0213 | 0.0288 | 0.0254 | +0.0041 |
| winner | `1m_velocity` | 0.0393 | -0.1570 | -0.4793 | -0.5186 |
| winner | `5m_velocity` | 0.3182 | 0.8616 | 0.8450 | +0.5269 |
| tail_loser | `1m_wick_ratio` | 0.2414 | 0.3653 | 0.4972 | +0.2558 |
| tail_loser | `5m_wick_ratio` | 0.9117 | 0.8648 | 0.5377 | -0.3740 |
| tail_loser | `15m_wick_ratio` | 0.8744 | 0.8700 | 0.7294 | -0.1449 |
| tail_loser | `1h_wick_ratio` | 0.6049 | 0.6049 | 0.6072 | +0.0023 |
| tail_loser | `1m_variance_ratio` | 0.4962 | 0.5055 | 0.6669 | +0.1707 |
| tail_loser | `5m_variance_ratio` | 0.3814 | 0.3812 | 0.3814 | +0.0001 |
| tail_loser | `1m_p_at_center` | 0.0671 | 0.1584 | 0.3191 | +0.2520 |
| tail_loser | `5m_p_at_center` | 0.6364 | 0.6017 | 0.4530 | -0.1834 |
| tail_loser | `1m_z_se` | -0.1840 | -0.1351 | -0.2801 | -0.0961 |
| tail_loser | `5m_z_se` | -0.0906 | -0.1713 | -0.0642 | +0.0264 |
| tail_loser | `15m_z_se` | -0.1059 | -0.1158 | -0.0334 | +0.0725 |
| tail_loser | `1h_z_se` | 0.1261 | 0.1261 | 0.1532 | +0.0271 |
| tail_loser | `1m_velocity` | -3.2333 | -2.8833 | -0.4083 | +2.8250 |
| tail_loser | `5m_velocity` | -0.2875 | -0.7750 | -2.7542 | -2.4667 |
