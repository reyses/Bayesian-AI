# Document ID: AG-DOC-VP-01 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #1: Volume Profile Trading Strategies
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Target VAH/VAL or 20pt/10pt Stop). Unclamped Magnitude.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 23 | 0.22 | -10.89 | **-5.34** | [-10.63, 0.94] | No |
| 2 | Naked POC Test | 14 | 0.36 | -9.94 | **0.01** | [-7.98, 9.64] | No |
| 3 | Trend Runner | 98 | 0.28 | -10.68 | **-3.06** | [-5.95, 0.07] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 20 | 0.25 | -13.23 | **-5.65** | [-11.89, 3.01] | No |
| 2 | Naked POC Test | 7 | 0.43 | -16.14 | **17.59** | [-6.75, 45.07] | No |
| 3 | Trend Runner | 71 | 0.27 | -11.01 | **-4.10** | [-7.70, -0.19] | Yes |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-VP-01_distributions.png)