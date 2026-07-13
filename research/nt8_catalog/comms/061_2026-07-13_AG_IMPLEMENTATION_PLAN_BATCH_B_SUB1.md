# AG Implementation Plan: Batch B (Sub-Batch 1: 5 Detectors)
**Doc:** 061 · **Date:** 2026-07-13 · **Author:** AG · **Status:** DRAFT
**Re:** Batch B Sub-Batch 1 (ADX-08, ATR-09, CROSS-11, DOW-19, FIB-17)

## 1. Overview
Per Doc 060, the Batch B rollout is split into manageable sub-batches to ensure depth and correctness. This document covers the first 5 detectors, providing concrete rules, file:line citations, exact FPS inputs (including required out-of-band context), carried state, verified index-space conventions, and parity expectations based on actual `events.parquet` data.

---

## 2. ADX-08 (Trend Gate)
- **Rule Restated:** Computes a 14-day ADX and a 10-day SMA using daily bars. If the 14-day ADX > 25 and yesterday's close > 10-day SMA (Up Trend), it triggers a bullish bounce when the intraday price drops below the 10-day SMA. If the trend is Down (yesterday's close < 10-day SMA), it triggers a bearish bounce when intraday price rallies above the 10-day SMA. Exits are set at a predefined target (swing high/low) or a 10-point stop.
- **Citations:** `ag_deepdive_08_adx.py` Lines 190-205 (Daily context: ADX, SMA, Trend, Swing H/L). Lines 58-94 (Intraday trigger and exit evaluation).
- **FPS Inputs Required:** `core_v2` standard 5s bars. **Foreign Data:** Requires daily High, Low, and Close from the past 14 days to compute ADX, SMA, and Swing High/Low. This must be injected out-of-band into the detector context (FPS core is frozen).
- **Carried State:** Intraday `in_trade` flag, target price, stop price.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4288`).
- **Real Modes:** `bullish_runner`, `bearish_runner` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected on triggers assuming daily foreign context is precisely matched. FPS divergences will occur if target/stop evaluations straddle the EOD truncation boundary differently than the script.

---

## 3. ATR-09 (Statistical Fade)
- **Rule Restated:** Computes a 14-day True ATR using daily data. Intraday, it tracks the `running_high` and `running_low`. If the intraday range exceeds a multiple of the 14-day ATR (Thresholds: 50%, 75%, 100%) and the current price is within 0.25 points of the running high/low, it triggers a mean-reversion fade. Target is 50% of the daily ATR; stop loss is 10 points.
- **Citations:** `ag_deepdive_09_atr.py` Lines 197-207 (14-day True ATR calculation). Lines 37-54 (Intraday threshold triggers). Lines 63-93 (Target/Stop exit).
- **FPS Inputs Required:** `core_v2` standard 5s bars. **Foreign Data:** Requires daily High, Low, and Close of the prior 15 days to calculate True Range and 14-day ATR.
- **Carried State:** `running_high`, `running_low`, `triggered` boolean flags for each threshold (0.5, 0.75, 1.0).
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4846`).
- **Real Modes:** `bearish_fade`, `bullish_fade` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected on exact threshold breaches. Divergences may occur if floating-point drift affects the 0.25 point proximity check to the running extreme.

---

## 4. CROSS-11 (Golden Cross)
- **Rule Restated:** Computes a 50-period and 200-period SMA on continuous 5s bars. Triggers a bullish runner when SMA-50 crosses above SMA-200, and a bearish runner when SMA-50 crosses below SMA-200. Exits the trade upon an opposite cross.
- **Citations:** `ag_deepdive_11_cross.py` Lines 48-53 (SMA definitions and crosses). Lines 75-86 (First cross scan). Lines 101-120 (Opposite cross exit).
- **FPS Inputs Required:** `core_v2` standard 5s bars. 
- **Carried State:** 200-bar rolling buffer for `sma200`, 50-bar rolling buffer for `sma50`, current cross state.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4857`).
- **Real Modes:** `bullish_runner`, `bearish_runner` (Verified from `events.parquet`).
- **Parity Plan:** **Significant divergence expected.** The deepdive script concatenates yesterday's data with today's to compute the 200-period SMA without a warmup gap (Lines 33-42). FPS will require a 200-bar warmup at the start of the RTH session, missing any early-morning crosses that the script catches unless the verifier explicitly seeds the buffers.

---

## 5. DOW-19 (Price Volume Divergence)
- **Rule Restated:** Computes a 20-period SMA of volume and a 20-period SMA of price. Tracks a 10-bar rolling High and Low of price. Triggers a bearish trap if price breaks above the 10-bar high but volume is below its 20-period average. Triggers a bullish trap if price breaks below the 10-bar low on below-average volume. Exits upon mean-reverting to the 20-period price SMA or hitting a 3-sigma stop. Incorporates a 60-bar max hold and a 60-bar cooldown.
- **Citations:** `ag_deepdive_19_dow.py` Lines 38-45 (Indicators). Lines 71-86 (Trigger logic). Lines 100-126 (Exit logic).
- **FPS Inputs Required:** `core_v2` standard 5s bars **including volume**.
- **Carried State:** 20-bar buffers for price and volume, 10-bar buffers for high and low, `cooldown` counter, `in_trade` flag.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4799`).
- **Real Modes:** `bullish_trap`, `bearish_trap` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected after buffer warmup (20 bars). Divergences may occur if trades remain open near EOD, as the script explicitly looks ahead up to 60 bars (Lines 90-92), which may overflow the RTH truncation in standard FPS processing.

---

## 6. FIB-17 (Confluence)
- **Rule Restated:** Requires a 14-day ADX > 25. If the daily trend is UP, it waits for an intraday pullback into the 50%-61.8% Fibonacci retracement zone of the recent 10-day swing high/low (Bullish Bounce). If the trend is DOWN, it waits for a rally into the same Fib zone (Bearish Bounce). Target is the swing extreme; stop is 10 points.
- **Citations:** `ag_deepdive_17_fib.py` Lines 257-282 (Daily context calculations). Lines 69-94 (Intraday trigger). Lines 113-143 (Exit logic).
- **FPS Inputs Required:** `core_v2` standard 5s bars. **Foreign Data:** Daily High, Low, Close of the past 24 days (to compute ADX and Swing H/L). Must be injected out-of-band.
- **Carried State:** `lower_bound`, `upper_bound`, `target`, `stop`, `in_trade` flag.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4741`).
- **Real Modes:** `bullish_bounce`, `bearish_bounce` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected on triggers, assuming identical foreign context injection.

---
## Request for Review
Please review this sub-batch plan. Awaiting `APPROVED — EXECUTE` before writing any detector code.
