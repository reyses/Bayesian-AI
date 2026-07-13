# IMPLEMENTATION PLAN: Batch A FPS-Native Detectors
**Doc:** 051 · **Date:** 2026-07-13 · **Author:** Gemini · **Status:** PROPOSED

## Compliance with Directive 050
**ZERO FPS CORE CHANGES ARE REQUIRED.** All 7 detectors in Batch A will be implemented strictly using existing `BarState` fields, internal detector-side state accumulation, or explicit causal pre-compute steps (e.g., loading prior-day OHLC). No modifications to `core_v2/FPS/forward_pass_system.py` or `core_v2/FPS/state.py` will be made.

## Detector Designs (Batch A)

### 1. ORB-02 (Opening Range Breakout)
- **FPS Inputs**: `st.ohlcv_5s['high']`, `st.ohlcv_5s['low']`, `st.price`, `st.timestamp`.
- **Pre-compute**: None.
- **Detector State**: `or_high` (initialized to `-inf`), `or_low` (initialized to `inf`), `or_set` (bool), `triggered` (bool).
- **Execution Logic**: 
  - Using `st.timestamp` converted to CT, track the 08:30 to 08:59:59 window.
  - Continuously update `or_high = max(or_high, st.ohlcv_5s['high'])` and `or_low = min(or_low, st.ohlcv_5s['low'])`.
  - At exactly 09:00:00 CT, flag `or_set = True`.
  - Subsquently, if `st.price > or_high`, yield `bullish_runner`. If `st.price < or_low`, yield `bearish_runner`. Only yield once per day.

### 2. SEASON-12 (Day of Week Gap Fill)
- **FPS Inputs**: `st.price` (as open at 08:30 CT), `st.ohlcv_5s['high']`, `st.ohlcv_5s['low']`.
- **Pre-compute**: CAUSAL PRE-COMPUTE REQUIRED. The runner script will load the prior session-day's final 5s bar (15:15 CT close) to establish `PDC` (Prior Day Close) and inject it into the detector's `__init__`.
- **Detector State**: `gap`, `mode` (`gap_up` or `gap_down`), `gap_measured` (bool).
- **Execution Logic**: 
  - On the first bar (08:30 CT), compute `gap = st.price - pdc`. If `abs(gap) >= 5.0`, flag `gap_measured = True` and set `mode`.
  - On subsequent bars, check if the gap is filled (e.g., `st.ohlcv_5s['high'] >= pdc` if gap down). Yield trigger when the gap fills.

### 3. RENKO-24 (Time Filtering)
- **FPS Inputs**: `st.ohlcv_5s['close']`.
- **Pre-compute**: None.
- **Detector State**: `brick_size=2.0`, `prev_brick_close`, `curr_dir` (1 or -1), `brick_chain` (int).
- **Execution Logic**: 
  - On the first bar (08:30 CT), anchor `prev_brick_close = st.ohlcv_5s['close']`.
  - Per bar, check if `close - prev_brick_close >= brick_size` or `<= -brick_size`.
  - If a brick closes, advance `prev_brick_close` by integer steps of `brick_size` and update the `curr_dir`.
  - If the direction is the same as the previous brick, increment `brick_chain`. If reversed, reset to 1.
  - When `brick_chain == 2` in the new direction, yield a `continuation` setup.

### 4. VWAP-03 (Session VWAP)
- **FPS Inputs**: `st.ohlcv_5s['close']`, `st.ohlcv_5s['volume']`.
- **Pre-compute**: None.
- **Detector State**: `cum_pv`, `cum_vol`, a 20-bar rolling buffer of `close` prices (for trailing VWAP standard deviation), `primed_bull`, `primed_bear`.
- **Execution Logic**: 
  - Update `cum_pv += close * volume` and `cum_vol += volume` to calculate current `vwap`.
  - Append to 20-bar buffer and compute `vwap_std = max(0.25, std(buffer))`.
  - Calculate `z_curr = (close - vwap) / vwap_std`.
  - Prime flags when `|z_curr| > 2.0`.
  - Yield bounce triggers when `z_curr` turns back toward the mean while primed (e.g., `z_curr < z_prev` and `z_curr > 0` for bearish bounce).

### 5. OHLC-01 (Prior Day Levels)
- **FPS Inputs**: `st.price`.
- **Pre-compute**: CAUSAL PRE-COMPUTE REQUIRED. The runner script will load the prior session-day's high, low, and close to establish `PDH`, `PDL`, and `PDC`, and inject them into the detector's `__init__`.
- **Detector State**: 240-bar (20m) rolling close buffer to approximate `SMA20` for mean-reversion exits, `setup` (1, 2, or 3), `triggered` (bool).
- **Execution Logic**: 
  - On the first bar (08:30 CT), define the setup (e.g., Setup 1 if `st.price < PDH`).
  - Scan forward: yield `bearish_bounce` if Setup 1 and `st.price >= PDH`.
  - Yield `bullish_bounce` if Setup 2 and `st.price <= PDL`.

### 6. PIVOT-16 (Floor Levels)
- **FPS Inputs**: `st.price`.
- **Pre-compute**: CAUSAL PRE-COMPUTE REQUIRED. The runner script will load the prior session-day's H/L/C, calculate the Pivot Point (PP), S1, and R1 strictly causally, and inject these levels into the detector's `__init__`.
- **Detector State**: `setup` (1 or 2), `triggered` (bool).
- **Execution Logic**: 
  - On the first bar (08:30 CT), define the setup (e.g., Setup 1 if `st.price > S1`).
  - Scan forward: yield `bullish_bounce` if Setup 1 and `st.price <= S1`.
  - Yield `bearish_bounce` if Setup 2 and `st.price >= R1`.

### 7. ROUND-05 (Psych Numbers)
- **FPS Inputs**: `st.price`.
- **Pre-compute**: None.
- **Detector State**: Two dictionaries mapping each 50-tick grid level `L` to `primed_bullish` and `primed_bearish` booleans.
- **Execution Logic**: 
  - Generate the grid of 50-tick levels (e.g., `19950, 20000, 20050`).
  - Update priming dynamically: if `st.price < L - 5`, `primed_bullish[L] = True`; if it crosses back `st.price >= L`, unprime.
  - If `st.price >= L` when `primed_bullish[L]` is True, yield `bullish_continuation` trigger.

---
**Reviewer Ask**:
Please review these detector port specs. Upon approval, I will execute the porting phase and create the canonical native detector classes in the research toolset.
