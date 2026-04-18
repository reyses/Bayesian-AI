# Loser Physics — NMP_FADE

**Trades:** 9,369  **Losers:** 4,524  **Winners:** 4,845  **WR:** 51.7%

## Top features separating LOSERS from winners (|Cohen d|)

Positive d = feature value is *higher* in losers. Negative = *lower* in losers.

| Feature | Cohen d | loser regime |
|---|---:|---|
| 1m_variance_ratio | +0.076 | HIGH |
| 15s_z_se | +0.058 | HIGH |
| 1D_wick_ratio | +0.056 | HIGH |
| 15s_dmi_diff | +0.056 | HIGH |
| 1h_vol_rel | -0.053 | LOW |
| 1m_z_se | +0.052 | HIGH |
| 15s_dir_vol | +0.052 | HIGH |
| 15s_z_low | +0.047 | HIGH |
| 1m_z_low | +0.047 | HIGH |
| 1m_z_high | +0.045 | HIGH |

## Loser regimes ranked by net-delta-if-flipped

For each loser regime: split trades on the regime feature, count flippable losers (those with valid counter-direction peaks in the regret output), and compute the PnL delta we'd get by flipping direction on those losers specifically.

| Regime | N | loser N | WR | flippable losers | flip $/trade | net $ if flipped |
|---|---:|---:|---:|---:|---:|---:|
| 1m_variance_ratio > 0.4004 | 4,684 | 2,351 | 49.8% | 2,351 (100%) | $+84.70 | $+282,880 |
| 1m_z_high > 0.9452 | 4,684 | 2,324 | 50.4% | 2,324 (100%) | $+81.56 | $+271,434 |
| 1D_wick_ratio > 0.5360 | 4,646 | 2,286 | 50.8% | 2,286 (100%) | $+82.12 | $+267,926 |
| 1m_z_se > -2.0350 | 4,684 | 2,326 | 50.3% | 2,326 (100%) | $+80.26 | $+267,444 |
| 1m_z_low > -2.2125 | 4,684 | 2,324 | 50.4% | 2,324 (100%) | $+79.70 | $+264,830 |
| 15s_z_se > -0.5063 | 4,684 | 2,314 | 50.6% | 2,314 (100%) | $+79.30 | $+262,485 |
| 15s_dmi_diff > -6.2414 | 4,684 | 2,315 | 50.6% | 2,315 (100%) | $+79.17 | $+262,317 |
| 15s_z_low > -1.0282 | 4,684 | 2,330 | 50.3% | 2,330 (100%) | $+78.39 | $+261,288 |
| 15s_dir_vol > 0.0000 | 4,297 | 2,108 | 50.9% | 2,108 (100%) | $+83.26 | $+250,968 |
| 1h_vol_rel < 0.4459 | 4,684 | 2,302 | 50.9% | 2,302 (100%) | $+54.04 | $+176,106 |
