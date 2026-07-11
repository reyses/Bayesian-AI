# 01. Volume & Order Flow Mechanics

This document consolidates the institutional concepts related to Volume Profile, Order Flow Footprints, and Volume-Price Divergence extracted from the NinjaTrader catalog.

## A. Volume Profile Mechanics

### 1. Volume Profile Shapes (P, b, D, B)
The shape of the volume distribution dictates the micro-market regime.
- **P-Shape (Bullish/Short Covering):** High volume at the top, thin tail below. Signals trapped shorts exiting.
- **b-Shape (Bearish/Long Liquidation):** High volume at the bottom, thin tail above. Signals trapped longs capitulating.
- **D-Shape (Balance):** Normal bell curve. Centered Point of Control (POC). Signals balance/choppiness. Institutional accumulation before a breakout.
- **B-Shape (Continuation):** Two D-profiles stacked. Represents a measured move or trend continuation.
**Bayesian-AI Implementation:** We calculate rolling skewness and kurtosis of the volume profile. `Profile_Skew < 0` = P-Shape. `Profile_Skew > 0` = b-Shape. The agent uses this as a regime classifier to bias longs/shorts.

### 2. Value Area & POC Dynamics
Volume Profile plots volume on the Y-axis (price).
- **Value Area (VA):** The price range containing ~70% of the day's volume. Bounded by Value Area High (VAH) and Value Area Low (VAL).
- **Point of Control (POC):** The single price level with the highest volume. Acts as a heavy magnet.
- **Value Area Rotations:** When price breaks into the Value Area from the outside, it has an 80% probability of rotating completely to the other side (e.g., from VAL up to VAH).
- **POC Reversion:** If price moves away from the POC on low volume, it is likely to snap back to the POC.
**Bayesian-AI Implementation:** If current price is inside the Value Area, we are in a **Mean-Reverting Regime**. If `Price crosses above VAL`, action space favors `Go_Long` with `Target = VAH`.

### 3. Context & Retracements
Opening price context vs the Prior Day's Volume Profile.
- **Bullish:** Open is above Prior Value Area but below Prior High. Wait for pullback to Prior POC.
- **Bearish:** Open is below Prior Value Area but above Prior Low. Wait for rally to Prior POC.
- **Runner:** Open is completely outside Prior Profile. Expect trend continuation.
**Bayesian-AI Implementation:** Create boolean features: `is_bullish_open_context`, `is_bearish_open_context`. Feed `distance_to_prior_poc` into the model so the agent learns to trade bounces off this level.

---

## B. Order Flow & Footprints

### 4. Absorption & Trapped Buyers
Volumetric footprint charts show the actual Bid vs. Ask volume at every tick.
- **Absorption:** High-volume rejection at a key support/resistance level. Massive volume is transacted at a single price node, but the price does not break through.
- **Trapped Buyers:** If footprint shows massive *buying* volume stacked at the absolute high of a candle, but price closes lower, the buyers are trapped. They become forced sellers.
**Bayesian-AI Implementation:** We calculate `Delta_at_High`. High positive delta at the absolute high of a bar that closes lower triggers `Trapped_Buyers_Flag = True`.

### 5. Imbalances & Delta Divergence
- **Imbalance:** When Ask volume at a price level is 3x (3:1 ratio) larger than the Bid volume directly below it. Flags where aggressive market orders overwhelmed limit orders.
- **Cumulative Delta Divergence:** A setup occurs when price makes a new high/low, but Cumulative Delta fails to confirm the move. This indicates that aggressive buying/selling is exhausting and failing to push through passive limit orders.
**Bayesian-AI Implementation:** Track local pivots in Price and Delta. If `Price_Pivot(t) > Price_Pivot(t-1)` AND `Delta_Pivot(t) < Delta_Pivot(t-1)`, flag `Delta_Bearish_Divergence = True`.

---

## C. Volume Divergence & Velocity

### 6. VWAP Z-Score Mean Reversion
VWAP acts as the intraday fair value anchor. 
- **Statistical Extreme:** ~95% of trading occurs within 2 standard deviations. A Z-score of +2 or -2 is a statistically uncommon extreme.
- **The Setup:** Wait for price to hit Z-Score ±2, and then wait for the Z-Score to *begin turning back toward zero* before entering. Target is the VWAP.
**Bayesian-AI Implementation:** If `VWAP_Z_Score <= -2` and `Delta_Z_Score > 0` (starting to revert), the RL agent's action space heavily favors `Go_Long`.

### 7. VWMA vs SMA Divergence (Trend Velocity)
- **Trend Confirmation:** If a 20-period Volume-Weighted Moving Average (VWMA) diverges (pulls away) from a 20-period Simple Moving Average (SMA), it proves the trend is driven by heavy volume.
- **Exhaustion:** If the VWMA converges back toward the SMA, volume is dropping out.
**Bayesian-AI Implementation:** `Volume_Velocity_Spread = VWMA(20) - SMA(20)`. A shrinking spread acts as a causal early warning feature for a reversal.

### 8. Price-Volume Divergence (Dow Theory)
We compare current volume to a 20-period SMA of volume.
- **Bullish Confirmation:** Price Up + Volume Up.
- **Bearish Divergence:** Price Up + Volume Down (exhaustion of buyers).
**Bayesian-AI Implementation:** If `Price_Delta > 0` and `Volume_Delta < 0`, flag `Volume_Divergence_Short = True`. Breakouts with this flag are highly probable traps.
