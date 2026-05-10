# Physics-Based Entry Rules — 2026-04-05

Derived from 3526 profitable corrected trades.
Features ranked by Cohen's d (standardized mean difference vs all trades).

## Global: Winners vs Losers

Features that distinguish winners (0 with |d| >= 0.3):

| Feature | Cohen's d | Winner typical | All typical | Direction |
|---------|-----------|---------------|------------|-----------|

## Per-Strategy Rules (8 groups)

### ExNMP: long_extended
  Trades: 1239 | Avg PnL: $81.0 | Total: $100,306 | Avg hold: 27 bars

  No strongly discriminative features (all |d| < 0.3)

### ExNMP: long_fast
  Trades: 56 | Avg PnL: $18.1 | Total: $1,014 | Avg hold: 2 bars

  **Entry conditions (if/then):**
  ```
    time_of_day > 0.15  # MODERATE d=+0.56 (typical=0.75, range=[0.15, 0.98])
    1h_reversion_prob > 0.00  # WEAK d=+0.44 (typical=0.00, range=[0.00, 0.97])
    1h_wick_ratio > 0.00  # WEAK d=+0.44 (typical=0.00, range=[0.00, 0.80])
    1h_p_at_center > 0.00  # WEAK d=+0.38 (typical=0.00, range=[0.00, 0.59])
    1m_reversion_prob < 0.86  # WEAK d=-0.36 (typical=0.83, range=[0.54, 0.86])
    15s_dmi_diff > -28.86  # WEAK d=+0.35 (typical=5.60, range=[-28.86, 41.81])
    1m_p_at_center < 0.11  # WEAK d=-0.33 (typical=0.09, range=[0.03, 0.11])
    15s_variance_ratio > 0.35  # WEAK d=+0.33 (typical=0.99, range=[0.35, 1.62])
    1h_bar_range > 0.00  # WEAK d=+0.31 (typical=0.00, range=[0.00, 219.00])
    1h_variance_ratio < 1.00  # WEAK d=-0.30 (typical=1.00, range=[0.40, 1.00])
  ```

  **TF alignment:**
     15s: dmi_diff=+0.3, variance_ratio=+0.3
      1m: reversion_prob=-0.4, p_at_center=-0.3
      1h: reversion_prob=+0.4, wick_ratio=+0.4, p_at_center=+0.4, bar_range=+0.3, variance_ratio=-0.3

### ExNMP: long_long
  Trades: 307 | Avg PnL: $55.3 | Total: $16,974 | Avg hold: 12 bars

  No strongly discriminative features (all |d| < 0.3)

### ExNMP: long_medium
  Trades: 152 | Avg PnL: $40.0 | Total: $6,072 | Avg hold: 5 bars

  No strongly discriminative features (all |d| < 0.3)

### ExNMP: short_extended
  Trades: 1196 | Avg PnL: $85.8 | Total: $102,584 | Avg hold: 27 bars

  No strongly discriminative features (all |d| < 0.3)

### ExNMP: short_fast
  Trades: 56 | Avg PnL: $13.7 | Total: $766 | Avg hold: 1 bars

  **Entry conditions (if/then):**
  ```
    time_of_day > 0.11  # MODERATE d=+0.53 (typical=0.69, range=[0.11, 0.99])
    1h_reversion_prob > 0.00  # MODERATE d=+0.51 (typical=0.00, range=[0.00, 0.98])
    1h_wick_ratio > 0.00  # WEAK d=+0.49 (typical=0.00, range=[0.00, 0.76])
    15s_acceleration < 1.62  # WEAK d=-0.44 (typical=-1.00, range=[-16.38, 1.62])
    1m_acceleration < 1.62  # WEAK d=-0.44 (typical=-1.00, range=[-16.38, 1.62])
    15s_reversion_prob < 0.97  # WEAK d=-0.42 (typical=0.92, range=[0.00, 0.97])
    1m_velocity < 2.00  # WEAK d=-0.42 (typical=-0.75, range=[-16.25, 2.00])
    15s_velocity < 2.00  # WEAK d=-0.42 (typical=-0.75, range=[-16.25, 2.00])
    1h_p_at_center > 0.00  # WEAK d=+0.41 (typical=0.00, range=[0.00, 0.65])
    1h_bar_range > 0.00  # WEAK d=+0.38 (typical=0.00, range=[0.00, 258.50])
  ```

  **TF alignment:**
     15s: acceleration=-0.4, reversion_prob=-0.4, velocity=-0.4
      1m: acceleration=-0.4, velocity=-0.4
      1h: reversion_prob=+0.5, wick_ratio=+0.5, p_at_center=+0.4, bar_range=+0.4

### ExNMP: short_long
  Trades: 345 | Avg PnL: $60.1 | Total: $20,738 | Avg hold: 12 bars

  No strongly discriminative features (all |d| < 0.3)

### ExNMP: short_medium
  Trades: 175 | Avg PnL: $38.6 | Total: $6,754 | Avg hold: 5 bars

  No strongly discriminative features (all |d| < 0.3)

## Feature Importance Across All Groups

| Feature | Groups | Avg |d| | Physics |
|---------|--------|---------|---------|
| time_of_day | 2 | 0.54 | time of day |
| 1h_reversion_prob | 2 | 0.48 | reversion probability |
| 1h_wick_ratio | 2 | 0.46 | rejection (wick = indecision) |
| 1h_p_at_center | 2 | 0.39 |  |
| 1h_bar_range | 2 | 0.35 | volatility/risk |
| 15s_acceleration | 1 | 0.44 | momentum change (chop) |
| 1m_acceleration | 1 | 0.44 | momentum change (chop) |
| 15s_reversion_prob | 1 | 0.42 | reversion probability |
| 1m_velocity | 1 | 0.42 | rate of change |
| 15s_velocity | 1 | 0.42 | rate of change |
| 1m_reversion_prob | 1 | 0.36 | reversion probability |
| 15s_dmi_diff | 1 | 0.35 | trend direction/strength |
| 1m_p_at_center | 1 | 0.33 |  |
| 15s_variance_ratio | 1 | 0.33 | regime (trending vs reverting) |
| 1h_variance_ratio | 1 | 0.30 | regime (trending vs reverting) |