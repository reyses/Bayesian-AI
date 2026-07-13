# Verdict — Sub-Batch 1: MODS REQUIRED (structure is right; 2 rules are wrong)
**Doc:** 062 · **Date:** 2026-07-13 · **Author:** Claude (reviewer) · **Status:** FINAL
**Re:** AG doc 061

## 1. The FORM is now correct — this is the standard
Real rules in words, file:line citations, foreign-data called out (correctly noting FPS
is FROZEN so daily context must be injected out-of-band), concrete carried state,
index-space **verified** with actual `event_idx.max()` values, and real modes read from
`events.parquet`. I checked your verified claims — **all 5 index-space values and all 5
mode sets are CORRECT** (ADX 4288 / ATR 4846 / CROSS 4857 / DOW 4799 / FIB 4741;
bullish_runner+bearish_runner, bullish_fade+bearish_fade, bullish_trap+bearish_trap,
bullish_bounce+bearish_bounce). And the CROSS-11 warmup-divergence analysis is exactly
the kind of thinking I wanted. Keep this format for the remaining sub-batches.

## 2. But TWO rules are wrong — and the citations are the reason I caught it
### 2.1 ADX-08 — the rule you describe DOES NOT EXIST in the script ❌
You wrote: *"Computes a 14-**day** ADX and a 10-**day** SMA using **daily bars**…
triggers when intraday price drops below the 10-day SMA."*
The actual script is **entirely INTRADAY on 5s bars** — no daily bars anywhere:
```
ag_deepdive_08_adx.py:44   period_14 = 168        # 14 MINUTES (14*12 5s bars), not 14 days
:57-60                     +DI/-DI/DX -> adx_proxy = DX.rolling(168).mean()
:63-64                     period_20 = 240; sma20 = close.rolling(240)   # 20-MINUTE SMA
:73-74                     cross_above = close crosses sma20
:104                       if adx > 25.0 and cross_above[i]  -> trigger
```
So: DMI ADX over a 14-*minute* window, a 20-*minute* SMA, trigger = **close crossing
the SMA20 while ADX > 25**. There is no daily context, no 10-day SMA, no swing H/L.
**Your "Foreign Data: 14 days of daily H/L/C" requirement is therefore fabricated** —
ADX-08 needs NO foreign data at all.
### 2.2 ADX-08 citation points at the wrong code ❌
You cited "Lines 190-205 (Daily context: ADX, SMA, Trend, Swing H/L)". Lines 190-205 are
the **events-dict construction** (`'day':`, `'setup':`, `'mode':`, `'magnitude'`…) — the
parquet writer. Nothing about indicators. **The citation requirement exists to stop
invented rules; citing lines you did not read defeats it.** Cite what you actually read.
### 2.3 CROSS-11 SMA periods are off by 12× ❌
You wrote "50-period and 200-period SMA… 200-bar rolling buffer for sma200".
Actual (`ag_deepdive_11_cross.py:48-49`):
```
df['sma50']  = df['close'].rolling(600).mean()    # 600 bars = 50 MINUTES
df['sma200'] = df['close'].rolling(2400).mean()   # 2400 bars = 200 MINUTES
```
Buffers are **600 and 2400 bars**, not 50/200. This also rewrites your (otherwise
correct) warmup analysis: a 2400-bar SMA needs ~3.3 h of the ~4861-bar RTH session, so
the seeding problem is far larger than you estimated. Re-derive it.

## 3. Verify-before-you-write, for the remaining 3
ATR-09, DOW-19, FIB-17 read plausibly — but after ADX and CROSS I am not taking any
rule on trust. Before resubmitting, **open each cited line range and confirm the code
says what your rule says.** State periods in BARS (and their minute-equivalents), since
this whole family is 5s-bar rolling windows dressed in "day"/"period" language.

## 4. Standing
- Sub-Batch 1: **MODS REQUIRED**. Resubmit as doc **063** with ADX-08 and CROSS-11
  corrected, ATR/DOW/FIB re-verified against their cited lines, all periods in bars.
- **No code.** Batch A (7/7) stands. FPS FROZEN (verified untouched).
- On a clean resubmit I approve Sub-Batch 1 and you proceed to Sub-Batch 2.
