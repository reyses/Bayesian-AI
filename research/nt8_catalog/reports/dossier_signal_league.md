# Dossier signal league — direction agreement with AI labels
(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50)

## ZIGZAG
- N=4852 (train 2266 / test 2586), OOS AUC **0.556**, test base 0.96
- P-terciles: low: 0.95 [0.94,0.97] N=862 | mid: 0.96 [0.95,0.97] N=862 | high: 0.97 [0.96,0.98] N=862
- coefs: {'pivot_age_min': np.float64(-0.065), 'sig_with_leg': np.float64(0.0), 'value': np.float64(0.095), 'tod': np.float64(0.032), 'inter': np.float64(-0.065)}
## ORB-02
- N=539 (train 258 / test 281), OOS AUC **0.436**, test base 0.97
- P-terciles: low: 0.98 [0.95,1.00] N=94 | mid: 0.96 [0.91,0.99] N=93 | high: 0.97 [0.93,1.00] N=94
- coefs: {'pivot_age_min': np.float64(0.545), 'sig_with_leg': np.float64(0.471), 'value': np.float64(-0.218), 'tod': np.float64(0.822), 'inter': np.float64(-0.159)}
## SEASON-12
- N=521 (train 248 / test 273), OOS AUC **0.618**, test base 0.48
- P-terciles: low: 0.40 [0.30,0.49] N=91 | mid: 0.37 [0.26,0.47] N=91 | high: 0.66 [0.56,0.75] N=91
- coefs: {'pivot_age_min': np.float64(0.093), 'sig_with_leg': np.float64(1.0), 'value': np.float64(-0.189), 'tod': np.float64(0.0), 'inter': np.float64(-0.447)}
## VWAP-03
- N=29577 (train 14684 / test 14893), OOS AUC **0.604**, test base 0.41
- P-terciles: low: 0.31 [0.29,0.34] N=4965 | mid: 0.40 [0.37,0.42] N=4964 | high: 0.50 [0.48,0.53] N=4964
- coefs: {'pivot_age_min': np.float64(0.245), 'sig_with_leg': np.float64(0.559), 'value': np.float64(-0.139), 'tod': np.float64(0.213), 'inter': np.float64(-0.622)}
## OHLC-01
- N=619 (train 289 / test 330), OOS AUC **0.841**, test base 0.48
- P-terciles: low: 0.07 [0.03,0.13] N=110 | mid: 0.59 [0.50,0.67] N=110 | high: 0.77 [0.69,0.85] N=110
- coefs: {'pivot_age_min': np.float64(-0.021), 'sig_with_leg': np.float64(2.097), 'value': np.float64(-0.922), 'tod': np.float64(-0.028), 'inter': np.float64(0.171)}
## PIVOT-16
- N=324 (train 145 / test 179), OOS AUC **0.939**, test base 0.05
- P-terciles: low: 0.00 [0.00,0.00] N=60 | mid: 0.00 [0.00,0.00] N=59 | high: 0.15 [0.07,0.25] N=60
- coefs: {'pivot_age_min': np.float64(0.29), 'sig_with_leg': np.float64(0.981), 'value': np.float64(-0.914), 'tod': np.float64(0.028), 'inter': np.float64(-0.461)}
## ROUND-05
- N=44332 (train 15215 / test 29117), OOS AUC **0.623**, test base 0.63
- P-terciles: low: 0.52 [0.50,0.54] N=9706 | mid: 0.63 [0.62,0.65] N=9705 | high: 0.75 [0.73,0.76] N=9706
- coefs: {'pivot_age_min': np.float64(0.111), 'sig_with_leg': np.float64(0.552), 'value': np.float64(0.0), 'tod': np.float64(0.082), 'inter': np.float64(-0.21)}
## CROSS-11
- N=504 (train 237 / test 267), OOS AUC **0.616**, test base 0.66
- P-terciles: low: 0.55 [0.46,0.65] N=89 | mid: 0.65 [0.55,0.75] N=89 | high: 0.76 [0.67,0.85] N=89
- coefs: {'pivot_age_min': np.float64(-0.048), 'sig_with_leg': np.float64(0.296), 'value': np.float64(0.395), 'tod': np.float64(-0.057), 'inter': np.float64(-0.056)}
## VWMA-10
- N=540 (train 258 / test 282), OOS AUC **0.714**, test base 0.63
- P-terciles: low: 0.36 [0.27,0.45] N=94 | mid: 0.76 [0.67,0.84] N=94 | high: 0.78 [0.68,0.86] N=94
- coefs: {'pivot_age_min': np.float64(0.104), 'sig_with_leg': np.float64(0.932), 'value': np.float64(0.162), 'tod': np.float64(-0.279), 'inter': np.float64(-0.366)}
## DOW-19
- N=36842 (train 17325 / test 19517), OOS AUC **0.610**, test base 0.38
- P-terciles: low: 0.28 [0.27,0.29] N=6506 | mid: 0.38 [0.37,0.40] N=6505 | high: 0.49 [0.47,0.50] N=6506
- coefs: {'pivot_age_min': np.float64(0.075), 'sig_with_leg': np.float64(0.474), 'value': np.float64(-0.022), 'tod': np.float64(-0.04), 'inter': np.float64(-0.117)}
## TUNNEL-20
- N=35228 (train 16755 / test 18473), OOS AUC **0.604**, test base 0.59
- P-terciles: low: 0.49 [0.47,0.50] N=6158 | mid: 0.59 [0.58,0.61] N=6157 | high: 0.68 [0.67,0.69] N=6158
- coefs: {'pivot_age_min': np.float64(0.044), 'sig_with_leg': np.float64(0.408), 'value': np.float64(0.14), 'tod': np.float64(0.0), 'inter': np.float64(-0.067)}
## ATR-09
- N=882 (train 424 / test 458), OOS AUC **0.500**, test base 0.01
- P-terciles: low: 0.01 [0.00,0.03] N=153 | mid: 0.01 [0.00,0.03] N=152 | high: 0.01 [0.00,0.02] N=153
- coefs: {'pivot_age_min': np.float64(-0.216), 'sig_with_leg': np.float64(-0.025), 'value': np.float64(0.303), 'tod': np.float64(-0.535), 'inter': np.float64(-0.021)}