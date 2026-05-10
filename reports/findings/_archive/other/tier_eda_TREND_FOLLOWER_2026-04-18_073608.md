# Tier EDA — TREND_FOLLOWER — 2026-04-18_073608

N=479 trades.

## Segments

| Segment | N | MeanPnL | Total | MeanPeak | MedPeak | MeanGiveback | MedHeld |
|---|---:|---:|---:|---:|---:|---:|---:|
| winner | 279 | $+164.34 | $+45,852 | $+205.74 | $+135.00 | $+41.39 | 134m |
| small_loser | 8 | $-0.19 | $-2 | $+43.44 | $+47.75 | $+43.62 | 124m |
| mid_loser | 17 | $-9.32 | $-158 | $+53.53 | $+49.00 | $+62.85 | 118m |
| tail_loser | 175 | $-272.22 | $-47,638 | $+63.50 | $+41.50 | $+335.72 | 354m |

## Exit reasons

- **winner**: `trend_follower_inverse`:209, `end_of_day`:70
- **small_loser**: `end_of_day`:4, `trend_follower_inverse`:4
- **mid_loser**: `end_of_day`:13, `trend_follower_inverse`:4
- **tail_loser**: `trend_follower_inverse`:99, `end_of_day`:76

## Top 30 separators (winner vs tail)

| Feature | Winner median | Tail median | Cohen d |
|---|---:|---:|---:|
| `15m_vol_rel` | 0.9769 | 0.8945 | +0.246 |
| `15s_reversion_prob` | 0.9475 | 0.9448 | +0.220 |
| `5m_velocity` | -7.7500 | -3.0000 | -0.200 |
| `15m_dmi_diff` | -0.5015 | -2.0454 | +0.194 |
| `1h_wick_ratio` | 0.6244 | 0.5577 | +0.177 |
| `1m_z_low` | -2.5102 | -2.3203 | -0.175 |
| `1h_dmi_diff` | -2.1680 | -2.1193 | +0.174 |
| `5m_dir_vol` | -0.8017 | -0.4940 | -0.169 |
| `1h_hurst` | 0.7220 | 0.6939 | +0.168 |
| `5m_z_se` | -0.5956 | -0.1914 | -0.163 |
| `1h_bar_range` | 226.0000 | 246.0000 | -0.161 |
| `1m_z_se` | -2.1968 | -2.1535 | -0.157 |
| `1h_z_se` | 0.1532 | 0.0000 | +0.153 |
| `1D_variance_ratio` | 0.4054 | 0.4271 | -0.147 |
| `5m_vol_rel` | 1.5155 | 1.3264 | +0.142 |
| `5m_bar_range` | 128.0000 | 127.0000 | -0.139 |
| `15s_dir_vol` | -0.8143 | -0.8715 | +0.139 |
| `1m_dmi_diff` | -17.1034 | -11.6898 | -0.135 |
| `1m_z_high` | 0.1040 | 0.7454 | -0.126 |
| `1h_z_low` | -0.5627 | -0.6370 | +0.125 |
| `15m_z_high` | 0.6964 | 0.6775 | +0.122 |
| `1m_dir_vol` | -2.3606 | -1.8078 | -0.119 |
| `1m_wick_ratio` | 0.2167 | 0.2073 | +0.117 |
| `1m_variance_ratio` | 1.2410 | 1.2403 | -0.116 |
| `15s_z_high` | 0.3566 | 0.4781 | -0.115 |
| `15s_dmi_diff` | -26.3305 | -21.6911 | -0.115 |
| `15s_p_at_center` | 0.3384 | 0.3215 | +0.114 |
| `1m_hurst` | 0.7687 | 0.7872 | -0.111 |
| `1h_reversion_prob` | 0.9715 | 0.9709 | +0.110 |
| `1m_acceleration` | -5.0000 | -0.5000 | +0.110 |

## Regime shift (entry -> peak -> exit)

| Segment | Feature | Entry | Peak | Exit | Δ |
|---|---|---:|---:|---:|---:|
| winner | `1m_wick_ratio` | 0.2587 | 0.4488 | 0.3381 | +0.0794 |
| winner | `5m_wick_ratio` | 0.3945 | 0.4677 | 0.4214 | +0.0269 |
| winner | `15m_wick_ratio` | 0.5768 | 0.5544 | 0.5978 | +0.0210 |
| winner | `1h_wick_ratio` | 0.6076 | 0.5790 | 0.5846 | -0.0230 |
| winner | `1m_variance_ratio` | 1.3112 | 0.8616 | 1.0557 | -0.2555 |
| winner | `5m_variance_ratio` | 0.4144 | 0.5022 | 0.4596 | +0.0452 |
| winner | `1m_p_at_center` | 0.0490 | 0.3479 | 0.2613 | +0.2123 |
| winner | `5m_p_at_center` | 0.3845 | 0.4531 | 0.4478 | +0.0633 |
| winner | `1m_z_se` | -0.6653 | 0.1805 | 0.2449 | +0.9102 |
| winner | `5m_z_se` | -0.4469 | 0.2612 | 0.0725 | +0.5194 |
| winner | `15m_z_se` | -0.0132 | 0.0842 | -0.0785 | -0.0653 |
| winner | `1h_z_se` | 0.1361 | 0.1687 | 0.0764 | -0.0596 |
| winner | `1m_velocity` | 0.7841 | 2.3271 | 4.8916 | +4.1075 |
| winner | `5m_velocity` | -7.5421 | 9.1470 | 3.9220 | +11.4642 |
| small_loser | `1m_wick_ratio` | 0.3175 | 0.4331 | 0.3336 | +0.0161 |
| small_loser | `5m_wick_ratio` | 0.4039 | 0.3875 | 0.5076 | +0.1037 |
| small_loser | `15m_wick_ratio` | 0.5647 | 0.6328 | 0.7759 | +0.2112 |
| small_loser | `1h_wick_ratio` | 0.4909 | 0.5502 | 0.7025 | +0.2117 |
| small_loser | `1m_variance_ratio` | 1.3688 | 0.8494 | 1.0383 | -0.3306 |
| small_loser | `5m_variance_ratio` | 0.3155 | 0.6838 | 0.4777 | +0.1622 |
| small_loser | `1m_p_at_center` | 0.0304 | 0.3301 | 0.3241 | +0.2937 |
| small_loser | `5m_p_at_center` | 0.3468 | 0.3384 | 0.5332 | +0.1864 |
| small_loser | `1m_z_se` | 0.5411 | -0.9923 | -0.3311 | -0.8722 |
| small_loser | `5m_z_se` | -0.0716 | -0.2752 | -0.1452 | -0.0736 |
| small_loser | `15m_z_se` | 0.0184 | 0.0270 | 0.0077 | -0.0108 |
| small_loser | `1h_z_se` | 0.0029 | -0.3202 | -0.2374 | -0.2403 |
| small_loser | `1m_velocity` | 13.4688 | -10.5312 | 1.3750 | -12.0938 |
| small_loser | `5m_velocity` | -9.3438 | 1.0625 | -4.9375 | +4.4062 |
| mid_loser | `1m_wick_ratio` | 0.2266 | 0.5085 | 0.5788 | +0.3523 |
| mid_loser | `5m_wick_ratio` | 0.3469 | 0.6643 | 0.7169 | +0.3700 |
| mid_loser | `15m_wick_ratio` | 0.5540 | 0.5020 | 0.6549 | +0.1009 |
| mid_loser | `1h_wick_ratio` | 0.4656 | 0.4470 | 0.5571 | +0.0915 |
| mid_loser | `1m_variance_ratio` | 1.3776 | 0.5477 | 0.5926 | -0.7850 |
| mid_loser | `5m_variance_ratio` | 0.4835 | 0.5576 | 0.2495 | -0.2339 |
| mid_loser | `1m_p_at_center` | 0.0312 | 0.5467 | 0.4957 | +0.4645 |
| mid_loser | `5m_p_at_center` | 0.2025 | 0.5843 | 0.5252 | +0.3228 |
| mid_loser | `1m_z_se` | 1.1172 | -0.0230 | -0.6187 | -1.7359 |
| mid_loser | `5m_z_se` | 0.0053 | 0.1591 | -0.0680 | -0.0734 |
| mid_loser | `15m_z_se` | -0.5720 | 0.2583 | 0.0745 | +0.6466 |
| mid_loser | `1h_z_se` | -0.0374 | 0.0423 | 0.6153 | +0.6527 |
| mid_loser | `1m_velocity` | 28.3382 | -6.3382 | -2.6765 | -31.0147 |
| mid_loser | `5m_velocity` | 3.6618 | 3.5000 | -2.9265 | -6.5882 |
| tail_loser | `1m_wick_ratio` | 0.2394 | 0.4820 | 0.4336 | +0.1941 |
| tail_loser | `5m_wick_ratio` | 0.3727 | 0.4953 | 0.4936 | +0.1208 |
| tail_loser | `15m_wick_ratio` | 0.5810 | 0.5391 | 0.5702 | -0.0108 |
| tail_loser | `1h_wick_ratio` | 0.5645 | 0.5758 | 0.5329 | -0.0316 |
| tail_loser | `1m_variance_ratio` | 1.3452 | 0.8026 | 0.9004 | -0.4448 |
| tail_loser | `5m_variance_ratio` | 0.4178 | 0.5708 | 0.3139 | -0.1039 |
| tail_loser | `1m_p_at_center` | 0.0460 | 0.4046 | 0.3419 | +0.2958 |
| tail_loser | `5m_p_at_center` | 0.4050 | 0.3974 | 0.4295 | +0.0244 |
| tail_loser | `1m_z_se` | -0.2461 | 0.0076 | -0.0470 | +0.1991 |
| tail_loser | `5m_z_se` | -0.1830 | -0.1123 | -0.0264 | +0.1566 |
| tail_loser | `15m_z_se` | -0.0722 | -0.0591 | -0.0573 | +0.0149 |
| tail_loser | `1h_z_se` | -0.0463 | 0.0555 | 0.0273 | +0.0735 |
| tail_loser | `1m_velocity` | 0.5986 | -0.1729 | 0.6171 | +0.0186 |
| tail_loser | `5m_velocity` | 0.6543 | 1.6100 | 4.6671 | +4.0129 |
