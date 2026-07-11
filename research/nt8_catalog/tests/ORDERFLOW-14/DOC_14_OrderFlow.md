# Document ID: AG-DOC-ORDERFLOW-14
**Title:** Deep Dive #14: Order Flow & Cumulative Delta
**Status:** Completed (Single Block Validated)
**Ruleset:** Trapped Delta / Divergence at Swings. 3.0$\sigma$ Target / 3.0$\sigma$ Stop. (Expanding min_periods=4050 for p10/p90 thresholds; 4049 initial rows dropped for warm-up).

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1527 | 0.50 | -2.38 | **-1.61** | [-4.46, 1.38] | No |
| 2 | Trapped Traders at Peak | 5686 | 0.51 | 5.02 | **-0.94** | [-2.36, 0.47] | No |

### Results for 2026
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 232 | 0.52 | -4.23 | **0.53** | [-0.81, 1.86] | No |
| 2 | Trapped Traders at Peak | 1078 | 0.52 | 1.73 | **0.43** | [-0.67, 1.55] | No |

### Results for All Data (6-Month Single Validation Block)
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Delta Divergence at Peak | 1759 | 0.50 | -2.38 | **-1.33** | [-3.90, 1.21] | No |
| 2 | Trapped Traders at Peak | 6764 | 0.52 | 5.02 | **-0.73** | [-1.94, 0.49] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-14-OrderFlow_distributions.png)

## Diagnostic OQ Trace (Interleaved Symbol Bug)
The original order flow data contained interleaved symbols/contracts, causing physically impossible sigma spikes (e.g. subtracting an NQ price from an ES price). Sorting the data by time fixes the variance window and produces realistic magnitudes.

```text
--- TRACE BEFORE FIX (Unsorted) ---
Max sigma: 6462.93
Event at index 20 (Mode: bearish_bounce): p0 = 23472.50, magnitude = 18.75, max_sigma in path = 5.98
Event at index 94 (Mode: bullish_runner): p0 = 23505.75, magnitude = -7.75, max_sigma in path = 2.93
Event at index 159 (Mode: bearish_bounce): p0 = 23495.50, magnitude = -6.50, max_sigma in path = 6.19

--- TRACE AFTER FIX (Sorted) ---
Max sigma: 65.81
Event at index 20 (Mode: bearish_bounce): p0 = 23472.50, magnitude = 18.75, max_sigma in path = 5.98
Event at index 94 (Mode: bullish_runner): p0 = 23505.75, magnitude = -7.75, max_sigma in path = 2.93
Event at index 159 (Mode: bearish_bounce): p0 = 23495.50, magnitude = -6.50, max_sigma in path = 6.19
```