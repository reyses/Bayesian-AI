# Nightmare EDA — Problem Exit Deep Dive

**Run**: reports/findings/nightmare_2026-01-06_to_2026-02-09_trades.csv
**Date**: 2026-04-02 20:54
**Period**: 2026-01-06 to 2026-02-09
**Total trades**: 2087 | **Total PnL**: $-2,427.00

## All Exit Types — Overview

| Exit | N | WR | Total PnL | Avg PnL | Avg Hold |
|------|---|----|-----------|---------| ---------|
| end_of_day | 27 | 25.9% | $-516.00 | $-19.11 | 12.1 |
| lambda_flip | 80 | 48.8% | $-644.50 | $-8.06 | 16.3 |
| max_hold_profit | 182 | 78.0% | $4,390.50 | $24.12 | 26.1 |
| mean_reached | 424 | 69.3% | $-5,130.50 | $-12.10 | 23.2 |
| profit_hold_exit | 926 | 53.8% | $-40.00 | $-0.04 | 5.8 |
| trend_breakeven_protect | 246 | 0.0% | $-2,670.50 | $-10.86 | 6.5 |
| trend_exhausted | 119 | 24.4% | $-2,839.50 | $-23.86 | 11.5 |
| trend_protect_profit | 83 | 100.0% | $5,023.50 | $60.52 | 22.8 |

---
## 1. mean_reached

**Hypothesis**: Mean-reversion trades that "reach" the mean but had
large adverse excursions before getting there, resulting in tiny wins
and oversized losses when the mean is reached at a bad time.

### PnL + Hold Time
| Metric | Value |
|--------|-------|
| Trades | 424 |
| Total PnL | $-5,130.50 |
| Mean PnL | $-12.10 |
| Median PnL | $7.00 |
| Std Dev | $70.55 |
| Min | $-487.50 |
| P10 | $-79.70 |
| P25 | $-10.38 |
| P75 | $16.00 |
| P90 | $32.40 |
| Max | $166.00 |
| Mode (top 3) | [(6.0, 28), (8.0, 24), (12.0, 19)] |
| Hold Mean | 23.2 bars |
| Hold Median | 8.0 bars |
| Hold Range | 2-192 bars |

### Entry Conditions
| Feature | Mean | Median | P25 | P75 |
|---------|------|--------|-----|-----|
| z_se | -0.065 | 0.234 | -5.762 | 5.741 |
| vr | 0.429 | 0.400 | 0.281 | 0.558 |
| lam | -0.571 | -0.600 | -0.719 | -0.442 |
| dmi_1m | -1.566 | -1.424 | -6.657 | 3.638 |
| trend_15 | 0.308 | -0.375 | -6.750 | 6.750 |

### Winners vs Losers at Entry

- Winner avg PnL: $18.27, avg hold: 6.6 bars, avg peak: $21.52
- Loser avg PnL: $-80.79, avg hold: 60.9 bars, avg peak: $0.07

| Feature | Winner Mean | Loser Mean | Delta |
|---------|------------|------------|-------|
| z_se | 0.199 | -0.661 | +0.860 |
| vr | 0.431 | 0.426 | +0.005 |
| lam | -0.569 | -0.574 | +0.005 |
| dmi_1m | -1.340 | -2.078 | +0.737 |
| trend_15 | 0.110 | 0.758 | -0.648 |

### Direction Breakdown
| Direction | N | PnL | WR | Avg PnL |
|-----------|---|-----|----|---------|
| LONG | 209 | $-2,539.50 | 67.9% | $-12.15 |
| SHORT | 215 | $-2,591.00 | 70.7% | $-12.05 |

### Strategy Breakdown
| Strategy | N | PnL | WR | Avg Hold |
|----------|---|-----|----|----------|
| cautious_reversion | 34 | $-469.50 | 70.6% | 23.3 |
| open_ride | 10 | $123.00 | 80.0% | 12.4 |
| reversion | 380 | $-4,784.00 | 68.9% | 23.5 |

### Time of Day
| Hour (UTC) | N | PnL | Avg PnL | WR |
|------------|---|-----|---------|-------|
| 00:00 | 20 | $361.50 | $18.07 | 100.0% |
| 01:00 | 13 | $-148.00 | $-11.38 | 69.2% |
| 02:00 | 16 | $-497.00 | $-31.06 | 56.2% |
| 03:00 | 23 | $-456.50 | $-19.85 | 65.2% |
| 04:00 | 20 | $-60.50 | $-3.02 | 70.0% |
| 05:00 | 29 | $-162.50 | $-5.60 | 79.3% |
| 06:00 | 17 | $-301.00 | $-17.71 | 52.9% |
| 07:00 | 27 | $-80.00 | $-2.96 | 74.1% |
| 08:00 | 14 | $-8.50 | $-0.61 | 78.6% |
| 09:00 | 23 | $-113.00 | $-4.91 | 69.6% |
| 10:00 | 32 | $-1,327.50 | $-41.48 | 59.4% |
| 11:00 | 16 | $-89.50 | $-5.59 | 62.5% |
| 12:00 | 13 | $-464.50 | $-35.73 | 69.2% |
| 13:00 | 11 | $-296.50 | $-26.95 | 36.4% |
| 14:00 | 6 | $26.00 | $4.33 | 66.7% |
| 15:00 | 15 | $737.50 | $49.17 | 93.3% |
| 16:00 | 14 | $-293.00 | $-20.93 | 71.4% |
| 17:00 | 19 | $-1,267.50 | $-66.71 | 52.6% |
| 18:00 | 21 | $-451.50 | $-21.50 | 66.7% |
| 19:00 | 19 | $-312.00 | $-16.42 | 63.2% |
| 20:00 | 17 | $163.00 | $9.59 | 70.6% |
| 21:00 | 23 | $-385.00 | $-16.74 | 69.6% |
| 23:00 | 16 | $295.50 | $18.47 | 87.5% |

**Worst 3 hours:**
  - H10: 32 trades, $-1328, WR=59%
  - H17: 19 trades, $-1268, WR=53%
  - H02: 16 trades, $-497, WR=56%

### Day of Week
| Day | N | PnL | Avg PnL | WR |
|-----|---|-----|---------|-------|
| Monday | 91 | $-1,778.00 | $-19.54 | 64.8% |
| Tuesday | 71 | $-985.00 | $-13.87 | 70.4% |
| Wednesday | 77 | $-621.00 | $-8.06 | 72.7% |
| Thursday | 100 | $-546.50 | $-5.46 | 72.0% |
| Friday | 83 | $-1,372.50 | $-16.54 | 66.3% |
| Sunday | 2 | $172.50 | $86.25 | 100.0% |

---
## 2. trend_breakeven_protect

**Hypothesis**: Trend rides that were profitable, then trend weakened
and PnL went negative. The exit fires when was_profitable=True but
pnl<=0. These are trades that should have exited earlier (at trend_protect_profit)
but missed the window.

### PnL + Hold Time
| Metric | Value |
|--------|-------|
| Trades | 246 |
| Total PnL | $-2,670.50 |
| Mean PnL | $-10.86 |
| Median PnL | $-6.50 |
| Std Dev | $14.07 |
| Min | $-106.00 |
| P10 | $-28.75 |
| P25 | $-12.50 |
| P75 | $-3.00 |
| P90 | $-0.50 |
| Max | $0.00 |
| Mode (top 3) | [(0.0, 30), (-4.0, 27), (-6.0, 25)] |
| Hold Mean | 6.5 bars |
| Hold Median | 5.0 bars |
| Hold Range | 3-27 bars |

### Entry Conditions
| Feature | Mean | Median | P25 | P75 |
|---------|------|--------|-----|-----|
| z_se | -0.627 | -2.484 | -9.774 | 10.024 |
| vr | 0.531 | 0.525 | 0.375 | 0.698 |
| lam | -0.469 | -0.475 | -0.625 | -0.302 |
| dmi_1m | -0.908 | -5.690 | -14.272 | 12.819 |
| trend_15 | -1.404 | -15.375 | -29.938 | 25.750 |

### Peak PnL Before Loss
- Mean peak: $18.53
- Median peak: $10.50
- P75 peak: $22.50
- These trades were profitable (peak > 0) but gave it ALL back plus more.
- Average loss at exit: $-10.86
- The gap (peak - exit PnL): $29.39 average giveback

### Direction Breakdown
| Direction | N | PnL | WR | Avg PnL |
|-----------|---|-----|----|---------|
| LONG | 113 | $-1,168.00 | 0.0% | $-10.34 |
| SHORT | 133 | $-1,502.50 | 0.0% | $-11.30 |

### Strategy Breakdown
| Strategy | N | PnL | WR | Avg Hold |
|----------|---|-----|----|----------|
| trend_ride | 246 | $-2,670.50 | 0.0% | 6.5 |

### Time of Day
| Hour (UTC) | N | PnL | Avg PnL | WR |
|------------|---|-----|---------|-------|
| 00:00 | 9 | $-52.00 | $-5.78 | 0.0% |
| 01:00 | 14 | $-85.00 | $-6.07 | 0.0% |
| 02:00 | 8 | $-39.50 | $-4.94 | 0.0% |
| 03:00 | 9 | $-186.50 | $-20.72 | 0.0% |
| 04:00 | 6 | $-80.00 | $-13.33 | 0.0% |
| 05:00 | 3 | $-25.00 | $-8.33 | 0.0% |
| 06:00 | 9 | $-49.00 | $-5.44 | 0.0% |
| 07:00 | 18 | $-120.50 | $-6.69 | 0.0% |
| 08:00 | 15 | $-122.50 | $-8.17 | 0.0% |
| 09:00 | 6 | $-42.00 | $-7.00 | 0.0% |
| 10:00 | 9 | $-61.00 | $-6.78 | 0.0% |
| 11:00 | 7 | $-59.00 | $-8.43 | 0.0% |
| 12:00 | 13 | $-62.50 | $-4.81 | 0.0% |
| 13:00 | 5 | $-41.50 | $-8.30 | 0.0% |
| 15:00 | 19 | $-417.00 | $-21.95 | 0.0% |
| 16:00 | 15 | $-294.00 | $-19.60 | 0.0% |
| 17:00 | 13 | $-166.50 | $-12.81 | 0.0% |
| 18:00 | 13 | $-135.50 | $-10.42 | 0.0% |
| 19:00 | 16 | $-189.50 | $-11.84 | 0.0% |
| 20:00 | 12 | $-161.50 | $-13.46 | 0.0% |
| 21:00 | 11 | $-127.00 | $-11.55 | 0.0% |
| 23:00 | 16 | $-153.50 | $-9.59 | 0.0% |

### Post-Exit Price Movement
*Did price recover after we exited?*

| Lookahead | Recovery Rate | Avg Move (ticks) | Best Avg (ticks) |
|-----------|---------------|------------------|------------------|
| +5 bars | 48.8% would recover | avg move 5.3 ticks | best avg 43.2 ticks |
| +10 bars | 54.1% would recover | avg move 3.9 ticks | best avg 64.0 ticks |
| +15 bars | 50.4% would recover | avg move 4.9 ticks | best avg 78.5 ticks |
| +20 bars | 50.0% would recover | avg move 7.6 ticks | best avg 90.8 ticks |
| +30 bars | 52.8% would recover | avg move 20.5 ticks | best avg 118.1 ticks |

---
## 3. trend_exhausted

**Hypothesis**: Trend rides that got hit by a 15m trend flip.
The trend flipped against the trade direction past MIN_MOVE (10 pts).
Low WR suggests these are entering too late into trends that are
already near exhaustion.

### PnL + Hold Time
| Metric | Value |
|--------|-------|
| Trades | 119 |
| Total PnL | $-2,839.50 |
| Mean PnL | $-23.86 |
| Median PnL | $-35.50 |
| Std Dev | $69.41 |
| Min | $-164.50 |
| P10 | $-92.70 |
| P25 | $-59.75 |
| P75 | $-3.00 |
| P90 | $62.80 |
| Max | $296.00 |
| Mode (top 3) | [(-34.0, 4), (-64.0, 3), (-44.0, 3)] |
| Hold Mean | 11.5 bars |
| Hold Median | 9.0 bars |
| Hold Range | 3-45 bars |

### Entry Conditions
| Feature | Mean | Median | P25 | P75 |
|---------|------|--------|-----|-----|
| z_se | -0.778 | -2.022 | -8.675 | 8.775 |
| vr | 0.464 | 0.437 | 0.307 | 0.617 |
| lam | -0.536 | -0.563 | -0.693 | -0.383 |
| dmi_1m | -2.010 | -5.741 | -12.558 | 8.783 |
| trend_15 | -6.231 | -12.750 | -28.375 | 19.375 |

### Direction Breakdown
| Direction | N | PnL | WR | Avg PnL |
|-----------|---|-----|----|---------|
| LONG | 45 | $-1,034.00 | 24.4% | $-22.98 |
| SHORT | 74 | $-1,805.50 | 24.3% | $-24.40 |

### Strategy Breakdown
| Strategy | N | PnL | WR | Avg Hold |
|----------|---|-----|----|----------|
| trend_ride | 119 | $-2,839.50 | 24.4% | 11.5 |

### Time of Day
| Hour (UTC) | N | PnL | Avg PnL | WR |
|------------|---|-----|---------|-------|
| 00:00 | 7 | $-146.00 | $-20.86 | 14.3% |
| 01:00 | 7 | $-201.50 | $-28.79 | 28.6% |
| 02:00 | 1 | $-13.00 | $-13.00 | 0.0% |
| 03:00 | 3 | $-54.00 | $-18.00 | 33.3% |
| 04:00 | 1 | $-33.50 | $-33.50 | 0.0% |
| 05:00 | 1 | $-25.00 | $-25.00 | 0.0% |
| 06:00 | 5 | $-246.50 | $-49.30 | 0.0% |
| 07:00 | 5 | $28.00 | $5.60 | 80.0% |
| 08:00 | 6 | $-254.00 | $-42.33 | 0.0% |
| 09:00 | 5 | $-15.00 | $-3.00 | 40.0% |
| 10:00 | 2 | $-70.00 | $-35.00 | 0.0% |
| 11:00 | 5 | $-205.00 | $-41.00 | 0.0% |
| 12:00 | 4 | $-144.50 | $-36.12 | 0.0% |
| 15:00 | 16 | $-736.00 | $-46.00 | 25.0% |
| 16:00 | 5 | $-304.50 | $-60.90 | 0.0% |
| 17:00 | 10 | $-177.50 | $-17.75 | 20.0% |
| 18:00 | 7 | $-408.00 | $-58.29 | 0.0% |
| 19:00 | 6 | $93.00 | $15.50 | 66.7% |
| 20:00 | 11 | $412.00 | $37.45 | 54.5% |
| 21:00 | 2 | $-174.00 | $-87.00 | 0.0% |
| 23:00 | 10 | $-164.50 | $-16.45 | 30.0% |

### Post-Exit Price Movement
*Did price continue the trend after we exited?*

| Lookahead | Recovery Rate | Avg Move (ticks) | Best Avg (ticks) |
|-----------|---------------|------------------|------------------|
| +5 bars | 51.3% would recover | avg move -3.5 ticks | best avg 46.1 ticks |
| +10 bars | 49.6% would recover | avg move -0.2 ticks | best avg 69.7 ticks |
| +15 bars | 47.9% would recover | avg move -3.4 ticks | best avg 92.4 ticks |
| +20 bars | 52.9% would recover | avg move 13.5 ticks | best avg 110.4 ticks |
| +30 bars | 54.6% would recover | avg move 4.7 ticks | best avg 138.2 ticks |

---
## Cross-Exit Comparison

**Problem exits**: 789 trades, $-10,640.50
**Other exits**: 1298 trades, $8,213.50
**Net without problem exits**: $8,213.50 (283.22/day)

### Hypothetical: Remove Problem Exits
If these 789 trades never happened:
- Remaining PnL: $8,213.50
- Per day: $283.22
