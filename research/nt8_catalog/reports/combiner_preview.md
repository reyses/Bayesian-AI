# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 522412 fires (221974 train 2024 / 300438 test 2025+26) across 35 streams
- pooled OOS AUC **0.677** (test base agreement 0.511)
- coefs (standardized): pivot_age_min=+0.091, sig_with_leg=+0.468, tod=+0.025, inter=-0.144, consensus=-0.054, is_ADX08=+0.024, is_ATR09=-0.234, is_CROSS11=+0.017, is_CURVE=+0.077, is_DOW19=-0.106, is_FIB17=-0.004, is_HNS22=+0.003, is_MACD07=-0.392, is_NMP=-0.160, is_NMPLAMBDA=+0.050, is_NMPTCASCADE=-0.024, is_NMPTFADEAGN=-0.008, is_NMPTFADECALM=-0.067, is_NMPTFREIGHT=+0.099, is_NMPTKILLSHOT=-0.019, is_NMPTMTFBRK=+0.089, is_NMPTMTFEXH=+0.065, is_NMPTRIDEAGN=+0.100, is_OHLC01=+0.009, is_ORB02=+0.116, is_PIVOT16=-0.075, is_RENKO24=+0.150, is_ROUND05=+0.194, is_RSI06=-0.487, is_SAR23=-0.047, is_SCALP18=-0.032, is_SEASON12=+0.007, is_SQZ04=+0.009, is_TUNNEL20=+0.105, is_VA13=-0.007, is_VP01=-0.004, is_VWAP03=-0.056, is_VWMA10=+0.030, is_ZIGZAG=+0.290, is_ZONE21=+0.064

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 30044 | 0.17 | 0.17 | [0.16,0.18] |
| 1 | 30044 | 0.36 | 0.37 | [0.36,0.38] |
| 2 | 30044 | 0.45 | 0.43 | [0.42,0.44] |
| 3 | 30043 | 0.47 | 0.47 | [0.46,0.48] |
| 4 | 30044 | 0.49 | 0.49 | [0.48,0.50] |
| 5 | 30044 | 0.53 | 0.52 | [0.51,0.53] |
| 6 | 30043 | 0.62 | 0.59 | [0.58,0.61] |
| 7 | 30044 | 0.66 | 0.64 | [0.63,0.66] |
| 8 | 30044 | 0.67 | 0.66 | [0.65,0.67] |
| 9 | 30044 | 0.76 | 0.75 | [0.74,0.76] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 1154 | 0.58 |
| 1-2 | 21167 | 0.57 |
| 3-5 | 93842 | 0.52 |
| 6+ | 184275 | 0.50 |