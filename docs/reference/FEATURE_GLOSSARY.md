# Feature Glossary

Quick reference for all features used across the system. Each feature answers ONE question.

## 13D Base Features (per bar, any TF)

| # | Name | Measures | Unit | Range |
|---|------|----------|------|-------|
| 0 | dmi_diff | Who's winning — buyers or sellers | DI+ minus DI- | -100 to +100 |
| 1 | dmi_gap | How decisive the trend is | abs(DI+ - DI-) | 0 to 100 |
| 2 | vol_rel | Participation vs average | volume / 30-bar SMA | 0 to 5+ |
| 3 | dir_vol | Directional participation | sign(price_change) × vol_rel | -5 to +5 |
| 4 | velocity | How fast price is moving | price[i] - price[i-1] | unbounded |
| 5 | z_se | Position in the range (standard error) | (price - mean) / SE | -5 to +5 |
| 6 | price_accel | Is price speeding up or slowing down | velocity[i] - velocity[i-1] | unbounded |
| 7 | std_price | Volatility of recent prices | std(30-bar window) | >0 |
| 8 | variance_ratio | Trending vs mean-reverting | std(10-bar) / std(60-bar) | 0 to 3+ |
| 9 | bar_range | How big is this bar | (high - low) / tick | 0 to 100+ |
| 10 | wick_ratio | How much rejection in this bar | 1 - (body / range) | 0 to 1 |
| 11 | vwap_distance | Distance from volume-weighted average | (price - VWAP) / tick | unbounded |
| 12 | time_of_day | Where in the trading session | seconds_since_midnight / 86400 | 0 to 1 |

## Wave Function Features (from MarketState, computed every bar)

| # | Name | Measures | Note |
|---|------|----------|------|
| 13 | P_at_center | Probability price stays near fair value | 3-class softmax |
| 14 | P_near_upper | Probability price moves to upper band | 3-class softmax |
| 15 | P_near_lower | Probability price moves to lower band | 3-class softmax |
| 16 | entropy_normalized | Market certainty (0=decisive, 1=confused) | Shannon entropy / ln(3) |

## Reversion/Breakout Probabilities (from MarketState)

| # | Name | Measures | Note |
|---|------|----------|------|
| 17 | reversion_probability | P(price reverts to mean) | OU first-passage analytical |
| 18 | breakout_probability | P(price breaks through band) | 1 - reversion_probability |

## Level Context Features (from hand-drawn levels)

| # | Name | Measures | Note |
|---|------|----------|------|
| 19 | dist_to_resistance | Ticks to nearest resistance level above | From DATA/levels/ |
| 20 | dist_to_support | Ticks to nearest support level below | From DATA/levels/ |
| 21 | zone_position | Position within current zone (0=at support, 1=at resistance) | Derived |

## Legacy 16D Features (used in standalone_research.py, FractalClustering)

| # | Name | What It Actually Measures | Grounding |
|---|------|--------------------------|-----------|
| 0 | abs(z_score) | Distance from fair value in sigma units | Level 2: derived from Price |
| 1 | log1p(velocity) | Price speed (log-compressed) | Level 2: derived from Price |
| 2 | log1p(momentum) | Buying/selling pressure (velocity × volume) | Level 3: Price × Volume |
| 3 | entropy_normalized | Market certainty | Level 3: from softmax of z-score |
| 4 | log2(tf_seconds) | Timeframe scale | Level 1: Time |
| 5 | depth | Fractal nesting depth | Level 2: derived from Time |
| 6 | parent_is_band_reversal | Was parent pattern a reversal? | Level 3: structural |
| 7 | adx / 100 | Trend strength (0-1) | Level 3: derived from DMI |
| 8 | hurst_exponent | Persistence (>0.5=trending, <0.5=reverting) | Level 3: rescaled range of Price |
| 9 | dmi_diff / 100 | Buyer/seller balance | Level 2: derived from Price |
| 10 | parent_z | Parent TF's z-score | Level 2: cross-TF Price position |
| 11 | parent_dmi_diff | Parent TF's direction | Level 2: cross-TF direction |
| 12 | root_is_roche | Root ancestor at 2σ extreme | Level 3: structural |
| 13 | tf_alignment | Do this TF and root agree? | Level 3: cross-TF agreement |
| 14 | term_pid | Algorithmic control pressure (PID of z-score) | Level 4: machine-specific — USE WITH CAUTION |
| 15 | osc_coherence | Oscillation tightness | Level 3: rolling std of z-score |

## Feature Derivation Levels (from First Principles Framework)

```
Level 1 (Primary):     Price, Time — independent measurements
Level 2 (Kinematic):   velocity, z_score, dmi_diff — 1 step from primary
Level 3 (Combined):    momentum, variance_ratio, entropy — 2 steps
Level 4 (Machine):     term_pid, F_momentum — cumsum-based, machine-specific
                       *** Do NOT use Level 4 in transferable models ***
```

## Key EDA Findings (2026-03-29)

- **z_se is THE level feature** (p=5.6e-18) — hand-drawn levels sit at z-score extremes
- **dmi_diff** is #3 most level-aware (-0.122 correlation with level distance)
- **bar_range + wick_ratio** diverge BEFORE level touches = predictive
- **1m leads 1h** for: dmi_diff, dmi_gap, vol_rel, z_se (direction features)
- **1h leads 1m** for: velocity, price_accel (speed features)
- **variance_ratio** replaces ADX + Hurst with single grounded measurement
