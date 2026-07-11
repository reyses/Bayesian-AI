# Document ID: DOC-SEASON-12
**Title:** Deep Dive #12: Seasonality / Day of Week Effects
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Weekday Gap-Fills (>5pts).

## Probability of Fill (Hit Rate)

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | Fill Prob | 95% CI |
|---|---|---|---|---|---|---|
| 1 | Mon | 49 | 0.55 | 26.07 | 0.55 | [0.41, 0.69] |
| 2 | Tue | 49 | 0.69 | 28.88 | 0.69 | [0.55, 0.82] |
| 3 | Wed | 50 | 0.50 | 10.77 | 0.50 | [0.36, 0.64] |
| 4 | Thu | 51 | 0.65 | 8.78 | 0.65 | [0.51, 0.78] |
| 5 | Fri | 49 | 0.57 | 38.38 | 0.57 | [0.43, 0.71] |

#### Contrast vs Monday (2024)
| Day | Contrast (Day - Mon) | 95% CI | Significant? |
|---|---|---|---|
| Tue | +0.142 | [-0.041, +0.327] | No |
| Wed | -0.049 | [-0.252, +0.149] | No |
| Thu | +0.094 | [-0.087, +0.278] | No |
| Fri | +0.022 | [-0.163, +0.224] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | Fill Prob | 95% CI |
|---|---|---|---|---|---|---|
| 1 | Mon | 42 | 0.43 | 33.39 | 0.43 | [0.29, 0.57] |
| 2 | Tue | 43 | 0.60 | 14.88 | 0.60 | [0.47, 0.74] |
| 3 | Wed | 45 | 0.78 | 19.42 | 0.78 | [0.64, 0.89] |
| 4 | Thu | 47 | 0.70 | 10.34 | 0.70 | [0.57, 0.83] |
| 5 | Fri | 42 | 0.55 | 31.84 | 0.55 | [0.40, 0.69] |

#### Contrast vs Monday (2025)
| Day | Contrast (Day - Mon) | 95% CI | Significant? |
|---|---|---|---|
| Tue | +0.175 | [-0.035, +0.387] | No |
| Wed | +0.347 | [+0.160, +0.535] | Yes |
| Thu | +0.273 | [+0.069, +0.475] | Yes |
| Fri | +0.121 | [-0.095, +0.333] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-SEASON-12_distributions.png)