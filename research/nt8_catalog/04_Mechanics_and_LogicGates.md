# 04. Mechanics & Logic Gates

This document consolidates institutional concepts regarding Boolean logic gates, systemic filters, risk management, and dynamic structural stops extracted from the NinjaTrader catalog.

## A. Boolean Logic Gates & Filters

### 1. The Pre-Trade Overtrading Logic Gate
Overtrading is mitigated by enforcing a strict 3-indicator checklist before an order is sent.
- **Size/Stop (ATR):** Is the stop loss scaled to the current ATR, and position size scaled to keep dollar risk constant?
- **Timing (RSI):** Is the RSI in extreme territory (>70 or <30) against the trend? (Filters out FOMO).
- **Conviction (VWAP):** Is price on the correct side of the VWAP relative to the thesis?
**Bayesian-AI Implementation:** Implement this checklist as a rigid `action_mask` or penalty function in the RL Agent. If the agent attempts a long position when `RSI > 70` (chasing) or when `Price < VWAP`, the action is penalized or masked out entirely.

### 2. Reversal vs Pullback Structural Logic
A pullback is a pause; a reversal is a structural shift. The logic gate to confirm a reversal requires:
1. **Structure Break:** Price breaks a prior significant swing high/low.
2. **Volume Expansion:** The reversal move must be accompanied by higher-than-average volume.
3. **Divergence:** MACD or RSI divergence present at the prior high/low.
**Bayesian-AI Implementation:** A complex Boolean flag: `Confirmed_Reversal = (Structure_Break) AND (Volume > SMA_Vol) AND (RSI_Divergence)`. If True, the agent must exit trend-following positions and flip bias.

### 3. Micro-Structure Scalping Dynamics
1-minute scalping relies on speed and structure rather than prediction. The core stack is VWAP, tight EMAs (9, 20), RSI for overextension, and Cumulative Delta for bid/ask aggression.
**Bayesian-AI Implementation:** Construct a composite "Scalp Setup" feature: `price < 9-EMA`, `9-EMA > 20-EMA` (pullback in trend), and `Delta > 0` (buyers stepping in). Enforce hard daily loss limits as an explicit terminal state to prevent overtrading.

---

## B. Risk Management & Dynamic Stops

### 4. Structural Stop Losses & Dynamic Sizing (ZigZag)
Experts use *structural* stop losses (placed strictly beyond the most recent Swing High/Low) and calculate position size based on the distance to that stop to maintain a fixed 1% account risk.
**Bayesian-AI Implementation:** Update the RL Environment (`mamba_rl_env`) to pass the distance to the last `feat_001_leg_state.py` ZigZag pivot. The agent outputs a binary `Enter/Hold/Close` command, and the environment *automatically* sizes the position so that the risk exactly equals 1%.

### 5. Volatility-Adaptive Stops and Targets (ATR)
ATR measures average volatility. It dynamically expands in fast markets and contracts in slow markets.
- **Profit Targets:** Multiples of ATR (Target = Entry + 2*ATR).
- **Stop-Loss:** Beyond the noise (Stop = Entry - 1.5*ATR).
**Bayesian-AI Implementation:** Include rolling ATR in the state space. The Action Space should not dictate fixed tick stops, but multiplier coefficients of the current ATR (e.g., set stop at 1.0 * ATR).

### 6. Parabolic SAR Trailing Logic
The Parabolic SAR is a time-and-price based trailing indicator that accelerates to catch up to the price.
**Bayesian-AI Implementation:** When in a trend-following trade, the RL agent uses a parameterized action `Update_Stop_To_SAR`. This ensures risk is dynamically managed, removing subjective stop-loss placement.

---

## C. Systematic Rulesets

### 7. Systematic Mean Reversion Setup
A mechanical 5-step process for mean reversion:
1. **Baseline:** Use a 20-period SMA as the mean.
2. **Deviation:** Wait for price to deviate significantly above/below the 20-SMA.
3. **Confirmation:** RSI Overbought/Oversold OR price touches the outer Bollinger Band.
4. **Entry:** Go long when price is below average + oversold.
5. **Exit:** Target the 20-period SMA.
**Bayesian-AI Implementation:** Code this as `Agent_MeanReversion` in the SFE. The target is always `current_20_SMA`, creating a dynamic limit order logic.

### 8. Adaptive Price Zones (APZ) for Volatility Mean Reversion
APZ uses a double-smoothed EMA centerline with percentage-based upper/lower volatility bands.
- **Signal:** When price breaches the APZ band and re-enters, it signals a short-term reversal (exhaustion).
**Bayesian-AI Implementation:** Implement the APZ math: `Centerline = EMA(EMA(Close, N))`. `Band = Centerline ± (Volatility_Percentage * Centerline)`. Create a feature `is_apz_exhausted` that flips True when price pierces the outer band and closes back inside.
