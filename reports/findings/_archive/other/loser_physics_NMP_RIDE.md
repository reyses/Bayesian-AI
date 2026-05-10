# Loser Physics — NMP_RIDE

**Trades:** 513  **Losers:** 242  **Winners:** 271  **WR:** 52.8%

## Top features separating LOSERS from winners (|Cohen d|)

Positive d = feature value is *higher* in losers. Negative = *lower* in losers.

| Feature | Cohen d | loser regime |
|---|---:|---|
| 15s_z_se | -0.222 | LOW |
| 15s_z_low | -0.212 | LOW |
| 1m_hurst | -0.208 | LOW |
| 15m_z_low | -0.206 | LOW |
| 15m_dir_vol | +0.204 | HIGH |
| 15s_z_high | -0.201 | LOW |
| 1m_wick_ratio | -0.188 | LOW |
| 5m_variance_ratio | +0.167 | HIGH |
| 1m_p_at_center | -0.163 | LOW |
| 15s_dmi_gap | -0.163 | LOW |

## Loser regimes ranked by net-delta-if-flipped

For each loser regime: split trades on the regime feature, count flippable losers (those with valid counter-direction peaks in the regret output), and compute the PnL delta we'd get by flipping direction on those losers specifically.

| Regime | N | loser N | WR | flippable losers | flip $/trade | net $ if flipped |
|---|---:|---:|---:|---:|---:|---:|
| 5m_variance_ratio > 0.2541 | 256 | 129 | 49.6% | 129 (100%) | $+84.68 | $+14,640 |
| 15s_z_low < -0.9940 | 256 | 130 | 49.2% | 130 (100%) | $+85.58 | $+14,560 |
| 15s_z_high < 0.5822 | 256 | 130 | 49.2% | 130 (100%) | $+86.13 | $+14,542 |
| 15s_z_se < -0.4176 | 256 | 134 | 47.7% | 134 (100%) | $+82.70 | $+14,490 |
| 1m_hurst < 0.6977 | 256 | 126 | 50.8% | 126 (100%) | $+78.60 | $+13,379 |
| 15m_z_low < -0.7993 | 256 | 131 | 48.8% | 131 (100%) | $+76.86 | $+13,299 |
| 15s_dmi_gap < 33.0783 | 256 | 127 | 50.4% | 127 (100%) | $+77.80 | $+12,894 |
| 15m_dir_vol > 0.0694 | 256 | 131 | 48.8% | 131 (100%) | $+74.20 | $+12,853 |
| 1m_wick_ratio < 0.2195 | 256 | 132 | 48.4% | 132 (100%) | $+75.31 | $+12,836 |
| 1m_p_at_center < 0.0459 | 256 | 131 | 48.8% | 131 (100%) | $+69.43 | $+11,968 |
