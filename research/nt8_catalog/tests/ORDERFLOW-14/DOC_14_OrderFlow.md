# Document ID: DOC-14-OrderFlow (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #14: Order Flow & Cumulative Delta
**Status:** Completed (Single Block Validated)
**Ruleset:** Trapped Delta / Divergence at Swings. 3.0$\sigma$ Target / 3.0$\sigma$ Stop. (Expanding min_periods=4050 for p10/p90 thresholds; 4049 initial rows dropped for warm-up).

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1486 | 0.51 | -4.84 | **-0.03** | [-0.60, 0.53] | No |
| 2 | Trapped Traders at Peak | 5584 | 0.52 | 3.98 | **0.01** | [-0.21, 0.24] | No |

### Results for 2026
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 232 | 0.52 | -4.23 | **0.52** | [-0.84, 1.87] | No |
| 2 | Trapped Traders at Peak | 1075 | 0.52 | 3.92 | **0.46** | [-0.02, 0.93] | No |

### Results for All Data (6-Month Single Validation Block)
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1718 | 0.51 | -4.84 | **0.05** | [-0.48, 0.57] | No |
| 2 | Trapped Traders at Peak | 6659 | 0.52 | 3.98 | **0.09** | [-0.11, 0.30] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-14-OrderFlow_distributions.png)