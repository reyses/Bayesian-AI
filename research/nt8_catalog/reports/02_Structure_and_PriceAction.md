# 02. Structure & Price Action Mechanics

This document consolidates institutional mechanics regarding geometric market structure, boundaries, candlestick patterns, and time-filtering extracted from the NinjaTrader catalog.

## A. Intraday Boundaries & Market Phases

### 1. Standard Intraday Boundaries (The "Grid")
Institutional algorithms use fixed reference points generated from the previous session to establish a grid of support and resistance.
- **Prior Day OHLC:** The previous session's Open, High, Low, and Close.
- **Floor Trader Pivots:** Mathematical levels (Pivot, R1, R2, R3, S1, S2, S3) derived strictly from the prior day's High, Low, and Close.
- **Opening Range (ORB):** The High and Low established during the first 30 minutes of the cash session. A breakout of this range signals directional bias.
**Bayesian-AI Implementation:** The RL agent's observation space must include continuous distance metrics to these grid levels (e.g., `distance_to_PriorHigh`, `distance_to_OR_Low`).

### 2. Market Structure Phases (Impulse vs. Pullback)
Trends do not move in straight lines. They consist of:
- **Phase 1 (Impulse/Advance):** Movement in the direction of the macro trend structure.
- **Phase 2 (Pullback/Correction):** Movement opposing the macro trend structure.
**Bayesian-AI Implementation:** We algorithmically label phases by measuring the local slope of a fast moving average against a slow moving average. Entry models should target the exhaustion of Phase 2 to rejoin Phase 1.

### 3. Elliott Wave "Tunnels" (Macro Phase Identification)
Instead of subjectively counting waves, systems use a "Tunnel" of moving averages (e.g., 34-EMA High and 34-EMA Low).
**Bayesian-AI Implementation:** Calculate `Trend_Tunnel = [EMA(High, 34), EMA(Low, 34)]`. The position of price relative to this tunnel (Above = Impulse, Inside = Corrective/Chop, Below = Bearish Impulse) serves as a continuous state-space variable.

### 4. Seasonality (Temporal Features)
Markets exhibit recurring patterns tied to the calendar year (e.g., 5, 15, and 30-year averages).
**Bayesian-AI Implementation:** Encode temporal data into the state space using cyclical transformations: `sin(day_of_year)`, `cos(day_of_year)`, `sin(month_of_year)`, `cos(month_of_year)`.

---

## B. Price Action Validation

### 5. Virgin Supply / Demand Zones
Institutional zones are not historical lines. They are **2 to 5 candles of tight consolidation followed immediately by a sharp, high-volume departure.** A zone is only highly probable on its *first retest* (a "Virgin Zone").
**Bayesian-AI Implementation:** Build a `zone_tracker.py` L4 module that scans for a tight 3-bar standard deviation of high/lows. Triggers if the following bar's price velocity exceeds a high threshold. Store the bounding box and emit a boolean `is_in_virgin_demand`.

### 6. Micro-Structure Candlestick Signals (Exhaustion)
Specific candle shapes combined with location dictate exhaustion.
- **Hammer / Shooting Star:** A long wick (rejection) with a small body, closing near the open.
- **Engulfing:** A large body completely swallowing the prior candle's body.
**Bayesian-AI Implementation:** `Wick_Ratio = (High - Max(Open, Close)) / (High - Low)`. If `Wick_Ratio > 0.6` at a known resistance level, it confirms a Shooting Star.

### 7. Divergence in Classical Patterns (Head & Shoulders)
A true H&S pattern requires **Volume Divergence**, not just geometry.
- Left Shoulder = High Volume. Head = New high in price, but *Lower Volume*. Right Shoulder = Lower high in price, *Even Lower Volume*.
**Bayesian-AI Implementation:** Map this to the ZigZag array. When `High -> Higher High -> Lower High` occurs, check if `Sum(Volume_Leg1) > Sum(Volume_Leg2) > Sum(Volume_Leg3)`.

### 8. Fibonacci Retracements & Extensions
Natural pullbacks frequently stall at the 38.2%, 50%, and 61.8% retracements of the previous swing leg.
**Bayesian-AI Implementation:** Using the `feat_001_leg_state.py` ZigZag leg height, dynamically calculate the 61.8% pullback: `Fib_618 = Leg_High - (Leg_Height * 0.618)`. Pass `distance_to_fib_618` to the RL agent.

---

## C. Time Filtering

### 9. Renko Time-Filtering
Renko charts remove time from the x-axis entirely, plotting bricks only when price moves a specified distance (e.g., 4 ticks).
- **Noise Reduction:** Choppy markets with low volume are compressed into very few bricks, filtering out false signals.
**Bayesian-AI Implementation:** Generate an alternative state-space observation matrix using a Renko transformation instead of Time-Bars to reduce dimensionality during consolidation periods.
