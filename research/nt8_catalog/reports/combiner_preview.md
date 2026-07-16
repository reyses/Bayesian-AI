# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 156119 fires (68792 train 2024 / 87327 test 2025+26) across 13 streams
- pooled OOS AUC **0.685** (test base agreement 0.533)
- coefs (standardized): pivot_age_min=+0.089, sig_with_leg=+0.431, tod=+0.070, inter=-0.158, consensus=+0.103, is_ADX08=+0.017, is_ATR09=-0.459, is_CROSS11=+0.017, is_DOW19=-0.258, is_OHLC01=-0.005, is_ORB02=+0.195, is_PIVOT16=-0.152, is_ROUND05=+0.219, is_SEASON12=-0.003, is_TUNNEL20=+0.082, is_VWAP03=-0.180, is_VWMA10=+0.032, is_ZIGZAG=+0.482

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 8733 | 0.28 | 0.25 | [0.23,0.26] |
| 1 | 8733 | 0.35 | 0.36 | [0.34,0.38] |
| 2 | 8732 | 0.44 | 0.43 | [0.42,0.44] |
| 3 | 8733 | 0.49 | 0.46 | [0.45,0.48] |
| 4 | 8733 | 0.53 | 0.52 | [0.51,0.54] |
| 5 | 8732 | 0.57 | 0.54 | [0.52,0.56] |
| 6 | 8733 | 0.62 | 0.58 | [0.57,0.60] |
| 7 | 8732 | 0.67 | 0.66 | [0.65,0.67] |
| 8 | 8733 | 0.72 | 0.73 | [0.71,0.74] |
| 9 | 8733 | 0.82 | 0.80 | [0.79,0.81] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 15664 | 0.49 |
| 1-2 | 48328 | 0.54 |
| 3-5 | 19439 | 0.56 |
| 6+ | 3896 | 0.51 |