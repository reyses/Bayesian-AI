# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 1071848 fires (461656 train 2024 / 610192 test 2025+26) across 55 streams
- pooled OOS AUC **0.676** (test base agreement 0.531)
- coefs (standardized): pivot_age_min=+0.098, sig_with_leg=+0.430, tod=+0.087, inter=-0.140, consensus=+0.263, is_ADX08=+0.010, is_ATR09=-0.164, is_CROSS11=+0.006, is_CTXER=-0.060, is_CURVE=+0.025, is_DOW19=-0.100, is_EXITKMDR=-0.259, is_EXITTIMESTOP=-0.001, is_FIB17=-0.007, is_HNS22=-0.002, is_MACD07=-0.294, is_NMP=-0.130, is_NMP9CASCADE=-0.016, is_NMP9FADEAGAINST=+0.047, is_NMP9FADECALM=-0.036, is_NMP9FADEMOM=-0.040, is_NMP9FREIGHT=+0.037, is_NMP9KILLSHOT=-0.034, is_NMP9RIDEAGAINST=+0.084, is_NMP9RIDECALM=+0.051, is_NMP9RIDEMOM=+0.046, is_NMPLAMBDA=+0.006, is_NMPTCASCADE=-0.020, is_NMPTFADEAGN=-0.010, is_NMPTFADECALM=-0.078, is_NMPTFREIGHT=+0.034, is_NMPTKILLSHOT=-0.023, is_NMPTMTFBRK=+0.050, is_NMPTMTFEXH=+0.033, is_NMPTRIDEAGN=+0.032, is_OHLC01=-0.003, is_ORB02=+0.073, is_PIVOT16=-0.058, is_PROPTURN=+0.025, is_PROPTURNP=-0.019, is_PTRNENGULF=+0.056, is_PTRNHAMMER=+0.001, is_RENKO24=+0.076, is_ROUND05=+0.079, is_RSI06=-0.364, is_SAR23=-0.058, is_SCALP18=-0.025, is_SEASON12=+0.001, is_SQZ04=+0.005, is_TMPL0=+0.252, is_TUNNEL20=+0.047, is_TURNCLIMAX=-0.043, is_TURNHA=+0.041, is_TURNSWEEP=-0.014, is_VA13=-0.011, is_VP01=-0.007, is_VWAP03=-0.072, is_VWMA10=+0.012, is_ZIGZAG=+0.193, is_ZONE21=+0.034

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 61020 | 0.22 | 0.20 | [0.20,0.21] |
| 1 | 61019 | 0.41 | 0.38 | [0.37,0.39] |
| 2 | 61019 | 0.46 | 0.44 | [0.43,0.45] |
| 3 | 61019 | 0.51 | 0.47 | [0.46,0.48] |
| 4 | 61019 | 0.56 | 0.52 | [0.51,0.53] |
| 5 | 61019 | 0.60 | 0.58 | [0.57,0.59] |
| 6 | 61019 | 0.65 | 0.62 | [0.61,0.63] |
| 7 | 61019 | 0.69 | 0.66 | [0.65,0.66] |
| 8 | 61019 | 0.74 | 0.69 | [0.68,0.70] |
| 9 | 61020 | 0.82 | 0.75 | [0.73,0.77] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 27 | 0.59 |
| 1-2 | 797 | 0.50 |
| 3-5 | 12247 | 0.48 |
| 6+ | 597121 | 0.53 |