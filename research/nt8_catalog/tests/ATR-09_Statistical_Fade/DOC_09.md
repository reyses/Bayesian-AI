# Document ID: AG-DOC-ATR-09
**Title:** Deep Dive #9: Statistical ATR fade (True 14-day ATR Sweep)
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Ruleset changed from bespoke exit to symmetric ±2.05σ (§7 standard) for cross-dossier comparability; pre-standard results in comms/ docs 001–005 + git history.

## Expected Value (EV)

### Results for 2024
| Setup (Threshold) | Description | N | WR(>2.05σ)% | Excursion (Mode) | EV (Mean σ) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 108 | 0.52 | 2.01 | **0.08** | [-0.30, 0.46] | No |
| 50.0% | Bullish Fade | 100 | 0.49 | -2.01 | **-0.04** | [-0.45, 0.37] | No |
| 75.0% | Bearish Fade | 76 | 0.49 | -2.01 | **-0.06** | [-0.54, 0.38] | No |
| 75.0% | Bullish Fade | 74 | 0.53 | 2.01 | **0.12** | [-0.33, 0.58] | No |
| 100.0% | Bearish Fade | 28 | 0.39 | -2.01 | **-0.44** | [-1.17, 0.29] | No |
| 100.0% | Bullish Fade | 46 | 0.50 | -2.01 | **0.00** | [-0.62, 0.62] | No |

### Results for 2025
| Setup (Threshold) | Description | N | WR(>2.05σ)% | Excursion (Mode) | EV (Mean σ) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 92 | 0.54 | 2.01 | **0.17** | [-0.22, 0.58] | No |
| 50.0% | Bullish Fade | 91 | 0.57 | 2.01 | **0.29** | [-0.11, 0.70] | No |
| 75.0% | Bearish Fade | 52 | 0.50 | -2.01 | **0.00** | [-0.55, 0.55] | No |
| 75.0% | Bullish Fade | 61 | 0.38 | -2.01 | **-0.50** | [-0.97, -0.03] | Yes |
| 100.0% | Bearish Fade | 24 | 0.62 | 2.01 | **0.51** | [-0.34, 1.20] | No |
| 100.0% | Bullish Fade | 47 | 0.51 | 2.01 | **0.04** | [-0.57, 0.65] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-ATR-09_distributions.png)