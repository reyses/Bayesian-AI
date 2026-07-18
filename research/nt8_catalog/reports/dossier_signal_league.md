# Dossier signal league — direction agreement with AI labels
(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50)

## ZIGZAG
- N=4852 (train 2266 / test 2586), OOS AUC **0.556**, test base 0.96
- P-terciles: low: 0.95 [0.94,0.97] N=862 | mid: 0.96 [0.95,0.97] N=862 | high: 0.97 [0.96,0.98] N=862
- coefs: {'pivot_age_min': -0.065, 'sig_with_leg': 0.0, 'value': 0.095, 'tod': 0.032, 'inter': -0.065}
## ORB-02
- N=539 (train 258 / test 281), OOS AUC **0.436**, test base 0.97
- P-terciles: low: 0.98 [0.95,1.00] N=94 | mid: 0.96 [0.91,0.99] N=93 | high: 0.97 [0.93,1.00] N=94
- coefs: {'pivot_age_min': 0.545, 'sig_with_leg': 0.471, 'value': -0.218, 'tod': 0.822, 'inter': -0.159}
## SEASON-12
- N=521 (train 248 / test 273), OOS AUC **0.618**, test base 0.48
- P-terciles: low: 0.40 [0.30,0.51] N=91 | mid: 0.37 [0.29,0.47] N=91 | high: 0.66 [0.56,0.76] N=91
- coefs: {'pivot_age_min': 0.093, 'sig_with_leg': 1.0, 'value': -0.189, 'tod': 0.0, 'inter': -0.447}
## VWAP-03
- N=29577 (train 14684 / test 14893), OOS AUC **0.604**, test base 0.41
- P-terciles: low: 0.31 [0.29,0.34] N=4965 | mid: 0.40 [0.38,0.42] N=4963 | high: 0.50 [0.48,0.53] N=4965
- coefs: {'pivot_age_min': 0.245, 'sig_with_leg': 0.559, 'value': -0.139, 'tod': 0.213, 'inter': -0.622}
## OHLC-01
- N=619 (train 289 / test 330), OOS AUC **0.841**, test base 0.48
- P-terciles: low: 0.07 [0.03,0.12] N=110 | mid: 0.59 [0.50,0.68] N=110 | high: 0.77 [0.70,0.85] N=110
- coefs: {'pivot_age_min': -0.021, 'sig_with_leg': 2.097, 'value': -0.922, 'tod': -0.028, 'inter': 0.171}
## PIVOT-16
- N=324 (train 145 / test 179), OOS AUC **0.939**, test base 0.05
- P-terciles: low: 0.00 [0.00,0.00] N=60 | mid: 0.00 [0.00,0.00] N=59 | high: 0.15 [0.07,0.25] N=60
- coefs: {'pivot_age_min': 0.29, 'sig_with_leg': 0.981, 'value': -0.914, 'tod': 0.028, 'inter': -0.461}
## ROUND-05
- N=44332 (train 15215 / test 29117), OOS AUC **0.623**, test base 0.63
- P-terciles: low: 0.52 [0.50,0.54] N=9706 | mid: 0.63 [0.62,0.65] N=9705 | high: 0.75 [0.73,0.76] N=9706
- coefs: {'pivot_age_min': 0.111, 'sig_with_leg': 0.552, 'value': 0.0, 'tod': 0.082, 'inter': -0.21}
## CROSS-11
- N=504 (train 237 / test 267), OOS AUC **0.616**, test base 0.66
- P-terciles: low: 0.55 [0.44,0.65] N=89 | mid: 0.65 [0.55,0.74] N=89 | high: 0.76 [0.66,0.85] N=89
- coefs: {'pivot_age_min': -0.048, 'sig_with_leg': 0.296, 'value': 0.395, 'tod': -0.057, 'inter': -0.056}
## VWMA-10
- N=540 (train 258 / test 282), OOS AUC **0.714**, test base 0.63
- P-terciles: low: 0.36 [0.28,0.46] N=94 | mid: 0.76 [0.67,0.84] N=94 | high: 0.78 [0.69,0.86] N=94
- coefs: {'pivot_age_min': 0.104, 'sig_with_leg': 0.932, 'value': 0.162, 'tod': -0.279, 'inter': -0.366}
## DOW-19
- N=36842 (train 17325 / test 19517), OOS AUC **0.610**, test base 0.38
- P-terciles: low: 0.28 [0.27,0.29] N=6506 | mid: 0.38 [0.37,0.39] N=6505 | high: 0.49 [0.47,0.50] N=6506
- coefs: {'pivot_age_min': 0.075, 'sig_with_leg': 0.474, 'value': -0.022, 'tod': -0.04, 'inter': -0.117}
## TUNNEL-20
- N=35228 (train 16755 / test 18473), OOS AUC **0.604**, test base 0.59
- P-terciles: low: 0.49 [0.47,0.50] N=6158 | mid: 0.59 [0.58,0.60] N=6157 | high: 0.68 [0.67,0.69] N=6158
- coefs: {'pivot_age_min': 0.044, 'sig_with_leg': 0.408, 'value': 0.14, 'tod': 0.0, 'inter': -0.067}
## ATR-09
- N=882 (train 424 / test 458), OOS AUC **0.500**, test base 0.01
- P-terciles: low: 0.01 [0.00,0.03] N=153 | mid: 0.01 [0.00,0.03] N=152 | high: 0.01 [0.00,0.02] N=153
- coefs: {'pivot_age_min': -0.216, 'sig_with_leg': -0.025, 'value': 0.303, 'tod': -0.535, 'inter': -0.021}
## SAR-23
- N=37184 (train 17615 / test 19569), OOS AUC **0.618**, test base 0.44
- P-terciles: low: 0.33 [0.32,0.34] N=6523 | mid: 0.44 [0.42,0.45] N=6523 | high: 0.56 [0.55,0.57] N=6523
- coefs: {'pivot_age_min': 0.079, 'sig_with_leg': 0.437, 'value': 0.163, 'tod': 0.044, 'inter': -0.087}
- **SQZ-04**: N=168 — too few signals (raw agree 0.58)
## RSI-06
- N=14967 (train 7116 / test 7851), OOS AUC **0.515**, test base 0.04
- P-terciles: low: 0.04 [0.03,0.05] N=2617 | mid: 0.04 [0.03,0.05] N=2617 | high: 0.05 [0.04,0.06] N=2617
- coefs: {'pivot_age_min': 0.055, 'sig_with_leg': 0.237, 'value': 0.104, 'tod': 0.021, 'inter': -0.109}
## MACD-07
- N=9781 (train 4678 / test 5103), OOS AUC **0.552**, test base 0.05
- P-terciles: low: 0.05 [0.04,0.06] N=1701 | mid: 0.04 [0.03,0.05] N=1701 | high: 0.07 [0.06,0.08] N=1701
- coefs: {'pivot_age_min': 0.057, 'sig_with_leg': 0.292, 'value': -0.071, 'tod': -0.086, 'inter': -0.166}
- **SCALP-18**: N=53 — too few signals (raw agree 0.02)
## RENKO-24
- N=198560 (train 75048 / test 123512), OOS AUC **0.611**, test base 0.55
- P-terciles: low: 0.44 [0.42,0.45] N=41171 | mid: 0.55 [0.54,0.56] N=41170 | high: 0.65 [0.64,0.67] N=41171
- coefs: {'pivot_age_min': 0.109, 'sig_with_leg': 0.524, 'value': 0.0, 'tod': 0.058, 'inter': -0.181}
- **FIB-17**: N=140 — too few signals (raw agree 0.33)
## ZONE-21
- N=3451 (train 1376 / test 2075), OOS AUC **0.584**, test base 0.63
- P-terciles: low: 0.58 [0.54,0.62] N=692 | mid: 0.61 [0.57,0.64] N=691 | high: 0.72 [0.68,0.76] N=692
- coefs: {'pivot_age_min': -0.069, 'sig_with_leg': 0.508, 'value': 0.018, 'tod': -0.015, 'inter': -0.071}
## VP-01
- N=283 (train 134 / test 149), OOS AUC **0.732**, test base 0.36
- P-terciles: low: 0.12 [0.04,0.20] N=50 | mid: 0.43 [0.29,0.57] N=49 | high: 0.52 [0.38,0.66] N=50
- coefs: {'pivot_age_min': 0.009, 'sig_with_leg': 0.969, 'value': 0.377, 'tod': -1.67, 'inter': -0.264}
- **VA-13**: N=166 — too few signals (raw agree 0.33)
- **HNS-22**: N=193 — too few signals (raw agree 0.57)
## CURVE
- N=26368 (train 12549 / test 13819), OOS AUC **0.606**, test base 0.55
- P-terciles: low: 0.45 [0.44,0.47] N=4607 | mid: 0.54 [0.52,0.55] N=4606 | high: 0.66 [0.64,0.68] N=4606
- coefs: {'pivot_age_min': 0.017, 'sig_with_leg': 0.347, 'value': -0.098, 'tod': -0.006, 'inter': -0.056}
## ADX08
- N=1359 (train 688 / test 671), OOS AUC **0.660**, test base 0.58
- P-terciles: low: 0.39 [0.30,0.49] N=224 | mid: 0.62 [0.56,0.68] N=223 | high: 0.74 [0.65,0.83] N=224
- coefs: {'pivot_age_min': -0.388, 'sig_with_leg': -1.24, 'value': -0.024, 'tod': 0.079, 'inter': 0.553}
## CTXER
- N=23259 (train 11041 / test 12218), OOS AUC **0.561**, test base 0.41
- P-terciles: low: 0.36 [0.34,0.37] N=4073 | mid: 0.41 [0.40,0.43] N=4072 | high: 0.47 [0.46,0.49] N=4073
- coefs: {'pivot_age_min': -0.051, 'sig_with_leg': -0.137, 'value': -0.237, 'tod': 0.022, 'inter': 0.106}
## EXITKMDR
- N=22296 (train 10357 / test 11939), OOS AUC **0.576**, test base 0.13
- P-terciles: low: 0.10 [0.09,0.10] N=3980 | mid: 0.13 [0.11,0.14] N=3979 | high: 0.16 [0.15,0.18] N=3980
- coefs: {'pivot_age_min': 0.025, 'sig_with_leg': -0.073, 'value': -0.152, 'tod': -0.026, 'inter': -0.012}
## EXITTIMESTOP
- N=2870 (train 1363 / test 1507), OOS AUC **0.533**, test base 0.51
- P-terciles: low: 0.50 [0.45,0.54] N=503 | mid: 0.49 [0.44,0.53] N=501 | high: 0.56 [0.51,0.60] N=503
- coefs: {'pivot_age_min': -0.049, 'sig_with_leg': 0.297, 'value': -0.066, 'tod': 0.033, 'inter': -0.047}
## NMP
- N=10388 (train 4989 / test 5399), OOS AUC **0.639**, test base 0.27
- P-terciles: low: 0.14 [0.12,0.15] N=1800 | mid: 0.32 [0.30,0.34] N=1799 | high: 0.36 [0.34,0.38] N=1800
- coefs: {'pivot_age_min': 0.057, 'sig_with_leg': 0.622, 'value': -0.16, 'tod': -0.048, 'inter': -0.105}
- **NMP9CASCADE**: N=71 — too few signals (raw agree 0.17)
## NMP9FADEAGAINST
- N=1133 (train 737 / test 396), OOS AUC **0.547**, test base 0.76
- P-terciles: low: 0.76 [0.68,0.83] N=132 | mid: 0.67 [0.58,0.75] N=132 | high: 0.85 [0.79,0.91] N=132
- coefs: {'pivot_age_min': -0.143, 'sig_with_leg': 0.221, 'value': 0.163, 'tod': 0.183, 'inter': 0.044}
## NMP9FADECALM
- N=828 (train 368 / test 460), OOS AUC **0.561**, test base 0.29
- P-terciles: low: 0.23 [0.18,0.30] N=154 | mid: 0.30 [0.22,0.37] N=152 | high: 0.34 [0.25,0.42] N=154
- coefs: {'pivot_age_min': 0.255, 'sig_with_leg': 0.306, 'value': -0.015, 'tod': 0.232, 'inter': -0.156}
## NMP9FADEMOM
- N=525 (train 204 / test 321), OOS AUC **0.634**, test base 0.21
- P-terciles: low: 0.08 [0.04,0.14] N=107 | mid: 0.28 [0.19,0.38] N=107 | high: 0.25 [0.17,0.34] N=107
- coefs: {'pivot_age_min': -0.503, 'sig_with_leg': 0.252, 'value': 0.049, 'tod': 0.112, 'inter': 0.421}
## NMP9FREIGHT
- N=1472 (train 412 / test 1060), OOS AUC **0.638**, test base 0.85
- P-terciles: low: 0.79 [0.74,0.83] N=354 | mid: 0.84 [0.80,0.88] N=353 | high: 0.94 [0.91,0.96] N=353
- coefs: {'pivot_age_min': 0.318, 'sig_with_leg': 0.774, 'value': 0.156, 'tod': 0.164, 'inter': -0.082}
## NMP9KILLSHOT
- N=329 (train 155 / test 174), OOS AUC **0.635**, test base 0.17
- P-terciles: low: 0.10 [0.03,0.19] N=58 | mid: 0.16 [0.07,0.25] N=58 | high: 0.26 [0.16,0.36] N=58
- coefs: {'pivot_age_min': 0.174, 'sig_with_leg': 0.597, 'value': -0.105, 'tod': 0.059, 'inter': -0.01}
## NMP9RIDEAGAINST
- N=3969 (train 1977 / test 1992), OOS AUC **0.641**, test base 0.79
- P-terciles: low: 0.69 [0.65,0.73] N=664 | mid: 0.79 [0.76,0.83] N=664 | high: 0.88 [0.86,0.91] N=664
- coefs: {'pivot_age_min': 0.446, 'sig_with_leg': 0.872, 'value': 0.02, 'tod': -0.001, 'inter': -0.529}
## NMP9RIDECALM
- N=1865 (train 968 / test 897), OOS AUC **0.603**, test base 0.78
- P-terciles: low: 0.74 [0.68,0.79] N=299 | mid: 0.75 [0.70,0.80] N=299 | high: 0.86 [0.82,0.90] N=299
- coefs: {'pivot_age_min': 0.025, 'sig_with_leg': 0.405, 'value': 0.039, 'tod': 0.113, 'inter': 0.111}
## NMP9RIDEMOM
- N=1142 (train 451 / test 691), OOS AUC **0.636**, test base 0.81
- P-terciles: low: 0.73 [0.66,0.78] N=230 | mid: 0.81 [0.76,0.86] N=231 | high: 0.89 [0.85,0.93] N=230
- coefs: {'pivot_age_min': 0.121, 'sig_with_leg': 0.749, 'value': 0.529, 'tod': 0.049, 'inter': -0.263}
## NMPLAMBDA
- N=10793 (train 5256 / test 5537), OOS AUC **0.574**, test base 0.54
- P-terciles: low: 0.46 [0.44,0.49] N=1846 | mid: 0.56 [0.53,0.58] N=1845 | high: 0.61 [0.58,0.63] N=1846
- coefs: {'pivot_age_min': 0.054, 'sig_with_leg': 0.289, 'value': 0.17, 'tod': 0.029, 'inter': -0.042}
## NMPTCASCADE
- N=669 (train 401 / test 268), OOS AUC **0.514**, test base 0.43
- P-terciles: low: 0.47 [0.35,0.58] N=90 | mid: 0.33 [0.23,0.43] N=89 | high: 0.49 [0.37,0.62] N=89
- coefs: {'pivot_age_min': 0.082, 'sig_with_leg': 0.604, 'value': -0.085, 'tod': -0.259, 'inter': -0.406}
## NMPTFADEAGN
- N=892 (train 567 / test 325), OOS AUC **0.638**, test base 0.41
- P-terciles: low: 0.24 [0.14,0.34] N=109 | mid: 0.50 [0.37,0.61] N=108 | high: 0.50 [0.38,0.61] N=108
- coefs: {'pivot_age_min': 0.138, 'sig_with_leg': 0.651, 'value': 0.123, 'tod': -0.023, 'inter': -0.162}
## NMPTFADECALM
- N=21034 (train 9787 / test 11247), OOS AUC **0.676**, test base 0.42
- P-terciles: low: 0.25 [0.23,0.26] N=3749 | mid: 0.43 [0.41,0.45] N=3749 | high: 0.59 [0.57,0.60] N=3749
- coefs: {'pivot_age_min': 0.269, 'sig_with_leg': 0.648, 'value': -0.39, 'tod': -0.014, 'inter': -0.315}
## NMPTFREIGHT
- N=4575 (train 941 / test 3634), OOS AUC **0.582**, test base 0.75
- P-terciles: low: 0.69 [0.66,0.72] N=1212 | mid: 0.76 [0.73,0.79] N=1211 | high: 0.81 [0.79,0.84] N=1211
- coefs: {'pivot_age_min': 0.232, 'sig_with_leg': 0.652, 'value': 0.25, 'tod': 0.023, 'inter': -0.335}
## NMPTKILLSHOT
- N=2931 (train 1350 / test 1581), OOS AUC **0.552**, test base 0.40
- P-terciles: low: 0.33 [0.29,0.38] N=527 | mid: 0.44 [0.39,0.49] N=527 | high: 0.42 [0.38,0.47] N=527
- coefs: {'pivot_age_min': 0.084, 'sig_with_leg': 0.048, 'value': -0.013, 'tod': 0.02, 'inter': 0.03}
## NMPTMTFBRK
- N=2167 (train 1024 / test 1143), OOS AUC **0.632**, test base 0.80
- P-terciles: low: 0.69 [0.64,0.75] N=381 | mid: 0.82 [0.77,0.87] N=381 | high: 0.87 [0.83,0.91] N=381
- coefs: {'pivot_age_min': -0.146, 'sig_with_leg': 0.255, 'value': 0.175, 'tod': 0.171, 'inter': 0.133}
## NMPTMTFEXH
- N=840 (train 412 / test 428), OOS AUC **0.635**, test base 0.79
- P-terciles: low: 0.71 [0.64,0.78] N=143 | mid: 0.81 [0.74,0.87] N=142 | high: 0.86 [0.80,0.92] N=143
- coefs: {'pivot_age_min': 0.393, 'sig_with_leg': 0.46, 'value': 0.815, 'tod': 0.0, 'inter': -0.205}
## NMPTRIDEAGN
- N=20690 (train 9611 / test 11079), OOS AUC **0.656**, test base 0.61
- P-terciles: low: 0.45 [0.43,0.48] N=3693 | mid: 0.62 [0.60,0.65] N=3693 | high: 0.75 [0.73,0.77] N=3693
- coefs: {'pivot_age_min': 0.318, 'sig_with_leg': 0.778, 'value': 0.031, 'tod': 0.02, 'inter': -0.461}
## PROPTURN
- N=30928 (train 14386 / test 16542), OOS AUC **0.636**, test base 0.57
- P-terciles: low: 0.46 [0.45,0.47] N=5514 | mid: 0.54 [0.52,0.55] N=5514 | high: 0.71 [0.69,0.72] N=5514
- coefs: {'pivot_age_min': -0.021, 'sig_with_leg': 0.318, 'value': -0.299, 'tod': -0.031, 'inter': -0.031}
## PROPTURNP
- N=210999 (train 79629 / test 131370), OOS AUC **0.689**, test base 0.50
- P-terciles: low: 0.33 [0.32,0.34] N=43790 | mid: 0.48 [0.48,0.49] N=43790 | high: 0.69 [0.68,0.70] N=43790
- coefs: {'pivot_age_min': 0.073, 'sig_with_leg': 0.544, 'value': -0.81, 'tod': -0.043, 'inter': -0.179}
## PTRNENGULF
- N=29917 (train 14648 / test 15269), OOS AUC **0.616**, test base 0.62
- P-terciles: low: 0.51 [0.50,0.53] N=5090 | mid: 0.63 [0.62,0.64] N=5089 | high: 0.72 [0.71,0.74] N=5090
- coefs: {'pivot_age_min': 0.031, 'sig_with_leg': 0.418, 'value': 0.243, 'tod': 0.057, 'inter': -0.046}
## PTRNHAMMER
- N=4484 (train 2075 / test 2409), OOS AUC **0.615**, test base 0.55
- P-terciles: low: 0.43 [0.39,0.47] N=803 | mid: 0.57 [0.53,0.61] N=803 | high: 0.65 [0.61,0.68] N=803
- coefs: {'pivot_age_min': 0.193, 'sig_with_leg': 0.469, 'value': -0.071, 'tod': -0.067, 'inter': -0.25}
## TMPL0
- N=157113 (train 73899 / test 83214), OOS AUC **0.631**, test base 0.68
- P-terciles: low: 0.56 [0.55,0.57] N=27738 | mid: 0.68 [0.67,0.69] N=27738 | high: 0.79 [0.78,0.80] N=27738
- coefs: {'pivot_age_min': 0.025, 'sig_with_leg': 0.023, 'value': 0.522, 'tod': -0.003, 'inter': -0.043}
## TURNCLIMAX
- N=2415 (train 1184 / test 1231), OOS AUC **0.556**, test base 0.33
- P-terciles: low: 0.29 [0.24,0.34] N=411 | mid: 0.31 [0.26,0.36] N=410 | high: 0.39 [0.34,0.43] N=410
- coefs: {'pivot_age_min': 0.043, 'sig_with_leg': -0.041, 'value': 0.185, 'tod': 0.15, 'inter': 0.048}
## TURNHA
- N=51793 (train 24738 / test 27055), OOS AUC **0.615**, test base 0.57
- P-terciles: low: 0.46 [0.44,0.47] N=9019 | mid: 0.58 [0.57,0.59] N=9017 | high: 0.68 [0.67,0.69] N=9019
- coefs: {'pivot_age_min': 0.021, 'sig_with_leg': 0.303, 'value': 0.328, 'tod': 0.05, 'inter': -0.015}
## TURNSWEEP
- N=2028 (train 1046 / test 982), OOS AUC **0.639**, test base 0.53
- P-terciles: low: 0.39 [0.32,0.45] N=328 | mid: 0.53 [0.46,0.59] N=327 | high: 0.68 [0.62,0.73] N=327
- coefs: {'pivot_age_min': 0.074, 'sig_with_leg': 0.424, 'value': -0.009, 'tod': 0.038, 'inter': -0.182}