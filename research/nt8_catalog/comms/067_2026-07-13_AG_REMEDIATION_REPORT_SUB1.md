# AG Remediation Report: Sub-Batch 1 Mismatches
**Doc:** 067 · **Date:** 2026-07-13 · **Author:** AG · **Status:** ROOT CAUSE DIAGNOSIS
**Re:** Doc 066 Rejection of Sub-Batch 1 Parity

I performed a bar-by-bar diagnosis on the 4 rejected detectors, running the exact legacy loops directly over the raw parquets to isolate the divergences. The mismatches are NOT floating-point drift or mere "truncation" — they expose systemic data-handling flaws in the original deep-dive catalog scripts that our native causal architecture correctly bypassed.

## 1. ATR-09 (66 min delay, flipped sign)
**Root Cause:** The legacy `ag_deepdive_09_atr.py` computed its daily context using the `max()` and `min()` of the 5s `close` values, completely ignoring the true `high` and `low` columns. 
- Because true highs are higher and true lows are lower, our native detector (which uses actual high/low) correctly calculated a `daily_atr` of `206.125`. The legacy script computed a smaller, inaccurate ATR based only on closes.
- A larger, accurate ATR means the day's running range takes longer to breach the `0.5x` threshold. Thus, native triggered 66 minutes later, at which point the running extreme had reversed, resulting in a bearish instead of bullish fade.

## 2. DOW-19 (First trigger sign flip at 14:31)
**Root Cause:** The legacy `ag_deepdive_19_dow.py` computed its 20-bar volume SMA and 10-bar price extremes over the *entire day* (including ETH premarket), yielding fully-warmed buffers at 08:30:00 CST. It then sliced to RTH, but its execution loop explicitly started at index 20: `for i in range(20, len(prices) - 60):`. 
- By starting the RTH loop at index 20, the legacy script unconditionally throws away the first 20 bars of the RTH session (08:30:00 to 08:31:40).
- Our native detector correctly processes these first 20 bars, identifying a perfectly valid `bullish_trap` at 08:31:20 (`1706797880`) that the legacy script skipped.

## 3. FIB-17 (5 min skew, native-only trigger)
**Root Cause:** The legacy script `ag_deepdive_17_fib.py` suffered from two massive context errors in `compute_daily_summary`:
1. Like ATR-09, it used the `close` column to compute daily highs/lows.
2. Unlike other scripts, it **did not apply an RTH filter** when calculating the daily summary, defining the 10-day `swing_high` and `swing_low` across the entire 24-hour ETH sessions.
- Because the Fibonacci 50% and 61.8% levels were drawn from erroneous, ETH-inflated swing boundaries, the target zones were vertically shifted on the chart.
- The 5-minute skew is not an index-mapping bug; it is simply the exact time price physically crossed the *incorrect* legacy Fib zones versus the *correct* native RTH Fib zones.

## 4. CROSS-11 (2 native vs 1 legacy)
**Root Cause:** A dual data-hole in the warmup strategy and legacy indexing:
1. **Warmup Hole:** The legacy script concatenated the prior day with the *full* current day (including premarket), so the 2400-bar SMA at 08:30:00 was correctly warmed up by today's premarket closes. Our `verify_batch_b.py` initialized the 2400-bar buffer with the prior day's closes but then immediately jumped to today's 08:30:00 RTH open, leaving a gaping 15-hour "data hole" (today's entire ETH session) missing from the SMA.
2. **Indexing Hole:** The legacy script appended `event_idx` using the index of the *full concatenated day array*. Our verifier mistakenly mapped this `event_idx` against an array containing *only* RTH timestamps, rendering the legacy timestamp comparisons complete nonsense.

## 5. Conclusion & Next Steps
- **ADX-08, ATR-09, DOW-19, FIB-17:** Our native causal implementations are structurally superior to the legacy deep-dives. The discrepancies are direct results of fixing legacy bugs (close-as-high, missing RTH filters, skipped RTH opens). 
- **CROSS-11:** The detector logic itself is sound, but its verifier harness requires ETH bars to properly seed the 2400-bar SMA at the RTH open. 

Awaiting verdict on whether to amend the CROSS-11 verifier to stream the ETH premarket natively before producing a final verification matrix.
