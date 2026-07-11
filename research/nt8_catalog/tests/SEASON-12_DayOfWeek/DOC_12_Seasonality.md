# Document ID: DOC-SEASON-12
**Title:** Deep Dive #12: Seasonality / Day of Week Effects
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Weekday Gap Fades (>5pts). Ruleset changed from bespoke exit to symmetric ±2.05σ (§7 standard) for cross-dossier comparability; pre-standard results in comms/ docs 001–005 + git history.

## Probability of +2.05σ (Hit Rate)

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | Fill Prob | 95% CI |
|---|---|---|---|---|---|---|
| 1 | Mon | 49 | 0.57 | 2.01 | 0.57 | [0.43, 0.71] |
| 2 | Tue | 49 | 0.71 | 2.01 | 0.71 | [0.59, 0.84] |
| 3 | Wed | 50 | 0.70 | 2.01 | 0.70 | [0.58, 0.82] |
| 4 | Thu | 51 | 0.61 | 2.01 | 0.61 | [0.47, 0.75] |
| 5 | Fri | 49 | 0.49 | -2.01 | 0.49 | [0.35, 0.63] |

#### Contrast vs Monday (2024)
| Day | Contrast (Day - Mon) | 95% CI | Significant? |
|---|---|---|---|
| Tue | +0.142 | [-0.041, +0.327] | No |
| Wed | +0.126 | [-0.056, +0.311] | No |
| Thu | +0.038 | [-0.146, +0.222] | No |
| Fri | -0.082 | [-0.286, +0.103] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | Fill Prob | 95% CI |
|---|---|---|---|---|---|---|
| 1 | Mon | 42 | 0.55 | 2.01 | 0.55 | [0.40, 0.69] |
| 2 | Tue | 43 | 0.37 | -2.01 | 0.37 | [0.23, 0.51] |
| 3 | Wed | 45 | 0.64 | 2.01 | 0.64 | [0.49, 0.78] |
| 4 | Thu | 47 | 0.45 | -2.01 | 0.45 | [0.32, 0.60] |
| 5 | Fri | 42 | 0.50 | -2.01 | 0.50 | [0.36, 0.64] |

#### Contrast vs Monday (2025)
| Day | Contrast (Day - Mon) | 95% CI | Significant? |
|---|---|---|---|
| Tue | -0.178 | [-0.387, +0.035] | No |
| Wed | +0.098 | [-0.108, +0.303] | No |
| Thu | -0.102 | [-0.310, +0.106] | No |
| Fri | -0.046 | [-0.262, +0.167] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-SEASON-12_distributions.png)