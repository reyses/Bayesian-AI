# AG Implementation Plan: Batch B (Sub-Batch 1: 5 Detectors)
**Doc:** 063 · **Date:** 2026-07-13 · **Author:** AG · **Status:** DRAFT
**Re:** Batch B Sub-Batch 1 (ADX-08, ATR-09, CROSS-11, DOW-19, FIB-17)

## 1. Overview
This is the corrected Sub-Batch 1 plan. ADX-08 and CROSS-11 have been corrected after reading their true cited line ranges. ATR-09, DOW-19, and FIB-17 have been re-verified to ensure periods are stated in bars and all logic strictly matches the scripts.

---

## 2. ADX-08 (Trend Gate)
- **Rule Restated:** Computes a 14-minute ADX proxy (168 bars on 5s) and a 20-minute SMA (240 bars on 5s). Triggers a bullish runner when the close crosses above the 240-bar SMA while the 168-bar ADX > 25. Triggers a bearish runner when the close crosses below the 240-bar SMA while ADX > 25. Exits when the ADX drops below 25 (trend exhausted).
- **Citations:** `ag_deepdive_08_adx.py` Lines 44-64 (168-bar ADX and 240-bar SMA indicators). Lines 70-74 (Cross definitions). Lines 96-113 (Intraday trigger). Lines 128-147 (Exit when ADX < 25).
- **FPS Inputs Required:** `core_v2` standard 5s bars. **No foreign data is needed.**
- **Carried State:** 168-bar rolling buffer for ADX computation (including +DI/-DI/TR), 240-bar rolling buffer for SMA, `triggered_bull`/`triggered_bear` boolean flags to limit to one trade per setup per day.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4288`).
- **Real Modes:** `bullish_runner`, `bearish_runner` (Verified from `events.parquet`).
- **Parity Plan:** Exact parity expected after a 240-bar warmup period. The script concatenates the whole day, so early crosses inside the 240-bar window might be missed in FPS unless the verifier seeds the buffers.

---

## 3. ATR-09 (Statistical Fade)
- **Rule Restated:** Computes a 14-day True ATR using daily data. Intraday, it tracks the `running_high` and `running_low`. If the intraday range exceeds a multiple of the 14-day ATR (Thresholds: 50%, 75%, 100%) and the current price is within 0.25 points of the running high/low, it triggers a mean-reversion fade. Target is 50% of the daily ATR; stop loss is 10 points.
- **Citations:** `ag_deepdive_09_atr.py` Lines 197-207 (14-day True ATR calculation from daily H/L/C). Lines 37-54 (Intraday threshold triggers). Lines 63-93 (Target/Stop exit).
- **FPS Inputs Required:** `core_v2` standard 5s bars. **Foreign Data:** Requires daily High, Low, and Close of the prior 15 days to calculate True Range and 14-day ATR. Must be injected out-of-band.
- **Carried State:** `running_high`, `running_low`, `triggered` boolean flags for each threshold (0.5, 0.75, 1.0).
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4846`).
- **Real Modes:** `bearish_fade`, `bullish_fade` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected on exact threshold breaches. Divergences may occur if floating-point drift affects the 0.25 point proximity check to the running extreme.

---

## 4. CROSS-11 (Golden Cross)
- **Rule Restated:** Computes a 50-minute (600 bars) and 200-minute (2400 bars) SMA on continuous 5s bars. Triggers a bullish runner when the 600-bar SMA crosses above the 2400-bar SMA, and a bearish runner when the 600-bar SMA crosses below the 2400-bar SMA. Exits the trade upon an opposite cross.
- **Citations:** `ag_deepdive_11_cross.py` Lines 48-53 (600-bar and 2400-bar SMA definitions and crosses). Lines 75-86 (First cross scan). Lines 101-120 (Opposite cross exit).
- **FPS Inputs Required:** `core_v2` standard 5s bars. 
- **Carried State:** 2400-bar rolling buffer for `sma200`, 600-bar rolling buffer for `sma50`, current cross state.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4857`).
- **Real Modes:** `bullish_runner`, `bearish_runner` (Verified from `events.parquet`).
- **Parity Plan:** **Significant divergence expected.** The deepdive script concatenates yesterday's data with today's to compute the 2400-bar SMA without a warmup gap (Lines 33-42). FPS will require a 2400-bar warmup (~3.3 hours into the 4861-bar RTH session), meaning FPS will naturally miss early-morning crosses that the script catches unless the verifier explicitly seeds the buffers.

---

## 5. DOW-19 (Price Volume Divergence)
- **Rule Restated:** Computes a 20-bar (100s) SMA of volume and a 20-bar SMA of price. Tracks a 10-bar rolling High and Low of price. Triggers a bearish trap if price breaks above the 10-bar high but volume is below its 20-bar average. Triggers a bullish trap if price breaks below the 10-bar low on below-average volume. Exits upon mean-reverting to the 20-bar price SMA or hitting a 3-sigma stop (sigma calculated over 12 bars). Incorporates a 60-bar max hold and a 60-bar cooldown.
- **Citations:** `ag_deepdive_19_dow.py` Lines 38-45 (Indicators: 20-bar SMA, 10-bar H/L). Lines 71-86 (Trigger logic). Lines 100-126 (Exit logic, 3-sigma stop).
- **FPS Inputs Required:** `core_v2` standard 5s bars **including volume**.
- **Carried State:** 20-bar buffers for price and volume, 10-bar buffers for high and low, 12-bar buffer for OLS sigma, `cooldown` counter, `in_trade` flag.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4799`).
- **Real Modes:** `bullish_trap`, `bearish_trap` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected after buffer warmup (20 bars). Divergences may occur if trades remain open near EOD, as the script explicitly looks ahead up to 60 bars (Lines 90-92), which may overflow the RTH truncation in standard FPS processing.

---

## 6. FIB-17 (Confluence)
- **Rule Restated:** Requires a 14-day ADX > 25. If the daily trend is UP (last close > 10-day SMA), it waits for an intraday pullback into the 50%-61.8% Fibonacci retracement zone of the recent 10-day swing high/low (Bullish Bounce). If the trend is DOWN, it waits for a rally into the same Fib zone (Bearish Bounce). Target is the swing extreme; stop is 10 points.
- **Citations:** `ag_deepdive_17_fib.py` Lines 257-282 (Daily context calculations: 14-day ADX, 10-day Swing H/L). Lines 69-94 (Intraday trigger). Lines 113-143 (Exit logic).
- **FPS Inputs Required:** `core_v2` standard 5s bars. **Foreign Data:** Daily High, Low, Close of the past 24 days (to compute ADX and Swing H/L). Must be injected out-of-band.
- **Carried State:** `lower_bound`, `upper_bound`, `target`, `stop`, `in_trade` flag.
- **Index-Space Convention:** RTH (Verified: `event_idx.max() = 4741`).
- **Real Modes:** `bullish_bounce`, `bearish_bounce` (Verified from `events.parquet`).
- **Parity Plan:** Parity expected on triggers, assuming identical foreign context injection.

---
## Request for Review
Please review this corrected sub-batch plan. Awaiting `APPROVED — EXECUTE` before writing any detector code.
