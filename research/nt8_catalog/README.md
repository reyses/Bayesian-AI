# NinjaTrader 8 Catalog: Synthesis Index

This directory contains the systematic extraction and synthesis of 463 historically scraped NinjaTrader articles. 
To ensure maximum utility when implementing these concepts into the Bayesian-AI Statistical Field Engine (SFE) and Mamba RL Agent, the extraction has been distilled into 4 core thematic pillars.

## Thematic Pillars

### 1. [Volume & Order Flow Mechanics](01_Volume_and_OrderFlow.md)
Focuses on the actual volume transacted at specific price levels to reveal institutional intent.
- **Key Concepts:** Volume Profile Shapes (P, b, D, B), Value Area & POC Dynamics, Cumulative Delta Divergence, Absorption, Footprint Imbalances (3:1), Trapped Buyers, VWAP Z-Score Mean Reversion, VWMA vs SMA Divergence.

### 2. [Structure & Price Action Mechanics](02_Structure_and_PriceAction.md)
Focuses on the geometric and time-based structures that define the boundaries of the market.
- **Key Concepts:** Market Phases (Impulse/Pullback), Intraday Boundaries (Prior OHLC, Floor Pivots, 30-min ORB), Virgin Supply/Demand Zones, Candlestick Wick Ratios, Head & Shoulders Divergence, Renko Time-Filtering, Elliott Wave Tunnels, Seasonality, Fibonacci Targets.

### 3. [Volatility & Momentum Mechanics](03_Volatility_and_Momentum.md)
Focuses on the cyclicality of expansion/contraction and the speed of price movement.
- **Key Concepts:** Composite Indicator Stacks (avoiding collinearity), RSI Divergence, Momentum Velocity & Acceleration, Trend Strength (ADX > 25), Volatility Squeezes (Bollinger Bandwidth), Statistical ATR Fading (90% rule), Psychological Liquidity Pools, Golden Cross Baselines.

### 4. [Mechanics & Logic Gates](04_Mechanics_and_LogicGates.md)
Focuses on the explicit Boolean logic, risk management math, and systemic rulesets required for execution.
- **Key Concepts:** Pre-Trade Overtrading Logic Gate, Reversal vs Pullback Structural Validation, 1-min Scalping Constraints, ZigZag Structural Stops (Dynamic Sizing), Volatility-Adaptive ATR Stops, Parabolic SAR Trailing, Systematic Mean Reversion Setup, Adaptive Price Zones (APZ).

---

## Guidelines for Implementation
When resetting context or building a new feature in the `pipeline/`, `builders/`, or `tools/` subdirectories, refer directly to the corresponding thematic pillar to pull the exact mathematical logic and causal reasoning required for the codebase.

---

## Phase 3B — Bayesian response sweep (active)
- **AG directive**: `AG_PHASE3B_JOINT_BAYES.md` (supersedes `AG_TASK_catalog_sweep.md`
  per-concept-verdict framing; `AG_PHASE3_REVIEW.md` = the methodology review).
- Tools → `tools/ag_cat_NN_<name>.py` (one standalone run per signal).
- Reports → `reports/AG_cat_NN_<name>.md` + figures in `reports/assets/`;
  master mode-first table → `reports/AG_cat_00_INDEX.md`.
- Prior evidence map: `TESTED_VS_UNTESTED.md`.
