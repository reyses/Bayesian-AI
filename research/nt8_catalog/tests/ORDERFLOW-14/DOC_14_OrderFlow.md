# Document ID: DOC-14-OrderFlow (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #14: Order Flow & Cumulative Delta
**Status:** Completed (Single Block Validated)
**Ruleset:** Trapped Delta / Divergence at Swings. 3.0$\sigma$ Target / 3.0$\sigma$ Stop.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1379 | 0.51 | -299.17 | **-533.45** | [-871.64, -206.13] | Yes |
| 2 | Trapped Traders at Peak | 5855 | 0.51 | 335.66 | **-339.95** | [-501.67, -178.06] | Yes |

### Results for 2026
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 230 | 0.52 | -4.23 | **0.39** | [-0.98, 1.75] | No |
| 2 | Trapped Traders at Peak | 1081 | 0.52 | -466.13 | **0.04** | [-71.05, 72.26] | No |

### Results for All Data (6-Month Single Validation Block)
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1609 | 0.51 | -299.17 | **-460.02** | [-743.86, -166.17] | Yes |
| 2 | Trapped Traders at Peak | 6936 | 0.52 | -508.59 | **-287.01** | [-421.12, -153.29] | Yes |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-14-OrderFlow_distributions.png)