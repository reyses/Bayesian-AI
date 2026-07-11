# Document ID: DOC-14-OrderFlow (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #14: Order Flow & Cumulative Delta
**Status:** Completed (Single Block Validated)
**Ruleset:** Trapped Delta / Divergence at Swings. 3.0$\sigma$ Target / 3.0$\sigma$ Stop. (Expanding min_periods=4050 for p10/p90 thresholds; 4049 initial rows dropped for warm-up).

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1527 | 0.50 | -2.38 | **-1.61** | [-4.64, 1.42] | No |
| 2 | Trapped Traders at Peak | 5686 | 0.51 | 5.02 | **-0.93** | [-2.40, 0.51] | No |

### Results for 2026
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 232 | 0.52 | -4.23 | **0.53** | [-0.79, 1.86] | No |
| 2 | Trapped Traders at Peak | 1078 | 0.52 | 1.73 | **0.44** | [-0.71, 1.58] | No |

### Results for All Data (6-Month Single Validation Block)
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1759 | 0.50 | -2.38 | **-1.36** | [-3.90, 1.25] | No |
| 2 | Trapped Traders at Peak | 6764 | 0.52 | 5.02 | **-0.71** | [-1.91, 0.49] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-14-OrderFlow_distributions.png)