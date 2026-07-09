# 03. Volatility & Momentum Mechanics

This document consolidates institutional concepts regarding momentum oscillators, volatility cyclicality, and multi-dimensional indicator stacks extracted from the NinjaTrader catalog.

## A. Momentum & Divergence

### 1. Composite Indicator Stack (Multi-Dimensional Confirmation)
Professional systems do not rely on a single indicator. They stack distinct dimensions of the market to avoid collinearity.
- **Trend:** Moving Averages (direction).
- **Value/Location:** VWAP (fair price, entry level).
- **Momentum Shift:** MACD (crossovers confirm early trend shift).
- **Exhaustion:** RSI (overbought/oversold limits).
- **Volatility Squeeze:** Bollinger Bands (breakout imminent).
**Bayesian-AI Implementation:** Structure the RL agent's observation space so it receives inputs from all 5 independent categories, ensuring it learns the non-linear relationships.

### 2. RSI Divergence (Leading Reversal Signal)
RSI > 70 is not an automatic sell signal. The true edge is **RSI Divergence**. 
- A bullish divergence occurs when the price makes a Lower Low (LL) but the RSI makes a Higher Low (HL), signaling waning downside momentum.
**Bayesian-AI Implementation:** Write an SFE module `divergence_tracker.py` hooked into the ZigZag pivot tracker. If `Current_Low < Previous_Low` AND `Current_RSI > Previous_RSI`, flip `is_bullish_divergence = True`.

### 3. Momentum Velocity and Acceleration
The Momentum indicator measures price change over a set period.
- **Velocity:** The absolute distance from the zero-line indicates trend strength.
- **Acceleration (Slope):** The slope of the momentum line indicates accelerating/decelerating trend.
- **Divergence:** If price makes a new high, but momentum slopes toward zero, it signals exhaustion.
**Bayesian-AI Implementation:** Track both `Momentum_Value` and `Momentum_Slope` (1st derivative). Divergence between `Price_Slope` and `Momentum_Slope` is a critical causal feature.

### 4. Trend Strength Classification (ADX & DMI)
ADX measures the absolute strength of a trend, regardless of direction. DMI measures the direction.
- **Trend Threshold:** An ADX value strictly > 25 indicates a confirmed, tradable trend.
**Bayesian-AI Implementation:** Use `ADX > 25` as an environment Regime Classifier. The agent switches between a "Trending Sub-Policy" and a "Ranging Sub-Policy" based on this hard threshold.

### 5. Moving Average Crossover Baselines (Golden Cross)
- **Golden Cross:** 50-period SMA crosses above the 200-period SMA (Macro Bullish).
- **Death Cross:** 50-period SMA crosses below the 200-period SMA (Macro Bearish).
**Bayesian-AI Implementation:** Binary feature flag `is_golden_cross` used as a hard filter: long-only strategies active during Golden Cross, short-only during Death Cross.

---

## B. Volatility & Extremes

### 6. Volatility Squeeze (Bollinger Bands + 21 EMA)
Volatility is cyclical. When Bollinger Bands tighten heavily around the 21-period EMA, the market is in a low-volatility accumulation phase. Extreme low volatility inevitably predicts an imminent, violent breakout.
**Bayesian-AI Implementation:** Calculate `Bollinger_Bandwidth = (Upper_Band - Lower_Band) / 21_EMA`. Create a feature `is_volatility_squeeze` when the Bandwidth drops below the 20th percentile of its 100-bar rolling window.

### 7. Statistical ATR Reversals (Fading the Daily Range)
Historical data often shows an instrument stays within its Average True Range (ATR) a high percentage of the time.
- **Reversal (Fading):** If price pushes to the extreme edge of the daily ATR (filling the range) and momentum wanes, fading the move is statistically high probability.
**Bayesian-AI Implementation:** Calculate `Daily_Range_Filled = (Current_High - Current_Low) / Daily_ATR`. If `Daily_Range_Filled > 0.9` and price is at the High of the day, `Go_Long` is heavily penalized, while `Go_Short` becomes favorable.

### 8. Psychological Liquidity Pools and Squeezes
- **Psychological Levels:** Large round numbers (e.g., ES at 5000). Traders place massive clusters of limit and stop orders here.
- **Squeezes:** When price breaches these heavily defended levels, it triggers cascades of stop-losses, resulting in violent acceleration.
**Bayesian-AI Implementation:** Create a `distance_to_round_number` feature. When price breaks through a round number AND volume spikes, interpret this as a high-probability squeeze event (momentum continuation).
