# Dossier signal league — direction agreement with AI labels
(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50)

## TURN-HA
- N=51793 (train 24738 / test 27055), OOS AUC **0.615**, test base 0.57
- P-terciles: low: 0.46 [0.44,0.47] N=9019 | mid: 0.58 [0.57,0.59] N=9017 | high: 0.68 [0.67,0.69] N=9019
- coefs: {'pivot_age_min': 0.021, 'sig_with_leg': 0.303, 'value': 0.328, 'tod': 0.05, 'inter': -0.015}
## TURN-SWEEP
- N=2028 (train 1046 / test 982), OOS AUC **0.639**, test base 0.53
- P-terciles: low: 0.39 [0.32,0.45] N=328 | mid: 0.53 [0.46,0.59] N=327 | high: 0.68 [0.62,0.73] N=327
- coefs: {'pivot_age_min': 0.074, 'sig_with_leg': 0.424, 'value': -0.009, 'tod': 0.038, 'inter': -0.182}
## TURN-CLIMAX
- N=2415 (train 1184 / test 1231), OOS AUC **0.556**, test base 0.33
- P-terciles: low: 0.29 [0.24,0.34] N=411 | mid: 0.31 [0.26,0.36] N=410 | high: 0.39 [0.34,0.43] N=410
- coefs: {'pivot_age_min': 0.043, 'sig_with_leg': -0.041, 'value': 0.185, 'tod': 0.15, 'inter': 0.048}
## EXIT-KMDR
- N=22296 (train 10357 / test 11939), OOS AUC **0.576**, test base 0.13
- P-terciles: low: 0.10 [0.09,0.10] N=3980 | mid: 0.13 [0.11,0.14] N=3979 | high: 0.16 [0.15,0.18] N=3980
- coefs: {'pivot_age_min': 0.025, 'sig_with_leg': -0.073, 'value': -0.152, 'tod': -0.026, 'inter': -0.012}
## CTX-ER
- N=23259 (train 11041 / test 12218), OOS AUC **0.561**, test base 0.41
- P-terciles: low: 0.36 [0.34,0.37] N=4073 | mid: 0.41 [0.40,0.43] N=4072 | high: 0.47 [0.46,0.49] N=4073
- coefs: {'pivot_age_min': -0.051, 'sig_with_leg': -0.137, 'value': -0.237, 'tod': 0.022, 'inter': 0.106}
## EXIT-TIMESTOP
- N=2870 (train 1363 / test 1507), OOS AUC **0.533**, test base 0.51
- P-terciles: low: 0.50 [0.45,0.54] N=503 | mid: 0.49 [0.44,0.53] N=501 | high: 0.56 [0.51,0.60] N=503
- coefs: {'pivot_age_min': -0.049, 'sig_with_leg': 0.297, 'value': -0.066, 'tod': 0.033, 'inter': -0.047}