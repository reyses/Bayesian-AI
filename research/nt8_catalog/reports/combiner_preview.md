# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 469219 fires (198159 train 2024 / 271060 test 2025+26) across 27 streams
- pooled OOS AUC **0.678** (test base agreement 0.506)
- coefs (standardized): pivot_age_min=+0.082, sig_with_leg=+0.453, tod=+0.015, inter=-0.150, consensus=-0.106, is_ADX08=+0.028, is_ATR09=-0.250, is_CROSS11=+0.020, is_CURVE=+0.090, is_DOW19=-0.101, is_FIB17=-0.003, is_HNS22=+0.004, is_MACD07=-0.407, is_NMP=-0.170, is_NMPEXT=+0.062, is_OHLC01=+0.012, is_ORB02=+0.126, is_PIVOT16=-0.077, is_RENKO24=+0.158, is_ROUND05=+0.222, is_RSI06=-0.508, is_SAR23=-0.039, is_SCALP18=-0.034, is_SEASON12=+0.009, is_SQZ04=+0.010, is_TUNNEL20=+0.121, is_VA13=-0.005, is_VP01=-0.003, is_VWAP03=-0.048, is_VWMA10=+0.034, is_ZIGZAG=+0.314, is_ZONE21=+0.072

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 27106 | 0.15 | 0.16 | [0.15,0.17] |
| 1 | 27106 | 0.35 | 0.37 | [0.36,0.38] |
| 2 | 27106 | 0.45 | 0.43 | [0.41,0.44] |
| 3 | 27106 | 0.47 | 0.45 | [0.43,0.46] |
| 4 | 27106 | 0.49 | 0.50 | [0.49,0.51] |
| 5 | 27106 | 0.53 | 0.53 | [0.52,0.54] |
| 6 | 27106 | 0.60 | 0.58 | [0.57,0.59] |
| 7 | 27106 | 0.65 | 0.64 | [0.62,0.65] |
| 8 | 27106 | 0.67 | 0.66 | [0.65,0.67] |
| 9 | 27106 | 0.74 | 0.74 | [0.73,0.75] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 2044 | 0.59 |
| 1-2 | 31859 | 0.58 |
| 3-5 | 103489 | 0.52 |
| 6+ | 133668 | 0.47 |