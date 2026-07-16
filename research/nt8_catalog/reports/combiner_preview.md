# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 447433 fires (187636 train 2024 / 259797 test 2025+26) across 25 streams
- pooled OOS AUC **0.675** (test base agreement 0.510)
- coefs (standardized): pivot_age_min=+0.084, sig_with_leg=+0.458, tod=+0.025, inter=-0.154, consensus=-0.073, is_ADX08=+0.027, is_ATR09=-0.266, is_CROSS11=+0.020, is_CURVE=+0.087, is_DOW19=-0.110, is_FIB17=-0.004, is_HNS22=+0.003, is_MACD07=-0.422, is_OHLC01=+0.011, is_ORB02=+0.129, is_PIVOT16=-0.081, is_RENKO24=+0.157, is_ROUND05=+0.217, is_RSI06=-0.526, is_SAR23=-0.046, is_SCALP18=-0.035, is_SEASON12=+0.008, is_SQZ04=+0.010, is_TUNNEL20=+0.118, is_VA13=-0.006, is_VP01=-0.004, is_VWAP03=-0.056, is_VWMA10=+0.033, is_ZIGZAG=+0.321, is_ZONE21=+0.071

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 25980 | 0.16 | 0.17 | [0.16,0.18] |
| 1 | 25980 | 0.38 | 0.38 | [0.37,0.40] |
| 2 | 25979 | 0.46 | 0.43 | [0.42,0.44] |
| 3 | 25980 | 0.48 | 0.46 | [0.45,0.48] |
| 4 | 25980 | 0.50 | 0.50 | [0.49,0.51] |
| 5 | 25979 | 0.54 | 0.53 | [0.52,0.54] |
| 6 | 25980 | 0.61 | 0.59 | [0.58,0.60] |
| 7 | 25979 | 0.65 | 0.64 | [0.63,0.66] |
| 8 | 25980 | 0.67 | 0.66 | [0.65,0.67] |
| 9 | 25980 | 0.75 | 0.74 | [0.73,0.75] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 2551 | 0.56 |
| 1-2 | 37402 | 0.57 |
| 3-5 | 103218 | 0.52 |
| 6+ | 116626 | 0.48 |