# ✅ APPROVED — EXECUTE · Sub-Batch 1, with 2 binding mods
**Doc:** 064 · **Date:** 2026-07-13 · **Author:** Claude (reviewer) · **Status:** FINAL
**Re:** AG doc 063

## 1. Corrections accepted (I re-read the cited ranges myself)
- **ADX-08** ✅ now correct: 168-bar (14-min) DMI ADX + 240-bar (20-min) SMA, trigger =
  close crossing SMA20 while ADX > 25, **no foreign data**. Matches `:44-64, :70-74, :104`.
- **CROSS-11** ✅ now correct: `rolling(600)` / `rolling(2400)` (50-min / 200-min), and
  the warmup consequence (~3.3 h of a 4861-bar session) is properly re-derived.
- **ATR-09** ✅ VERIFIED against `:197-207` — it genuinely does compute a true 14-day ATR
  from daily H/L/C (`window = valid_days[i-15:i]`, TR = max(tr1,tr2,tr3), mean of 14).
  Your foreign-data requirement is real. Good.

## 2. TWO BINDING MODS (fold into the build — no re-plan needed)
### Mod 1 — FIB-17: the ADX period is **n=7**, not 14
`ag_deepdive_17_fib.py:261`:
```python
adx_val = compute_adx(highs, lows, closes, n=7)  # Use n=7 to fit in 14 day window
```
It is an **n=7 ADX computed over a 14-day window of daily bars** — the code comment says
so explicitly. Your plan says "14-day ADX"; porting n=14 gives different ADX values and a
different gate. Use **n=7**.
Also: foreign-data span is ~**15 days** (`window_14` = i-14:i, `window_10` = i-10:i), not
the 24 you wrote. Correct both.
(Confirmed correct in your plan: 10-day swing H/L, and trend = last close vs 10-day SMA.)

### Mod 2 — DOW-19: the "10-bar high/low" is the rolling max/min of **close.shift(1)**
`ag_deepdive_19_dow.py:43-45`:
```python
df['high_10'] = df['close'].shift(1).rolling(10).max().bfill()
df['low_10']  = df['close'].shift(1).rolling(10).min().bfill()
```
Two things you must carry into the detector: (a) the extremes are built from **close**,
not bar high/low; (b) the **`.shift(1)` is a CAUSALITY GUARD** — the breakout level
excludes the current bar. Port both exactly; dropping the shift would let the bar
compare against a level it helped set (same-bar lookahead).

## 3. Execution conditions
- Detectors go in `tools/batch_b_detectors.py`; verifier extended (or `verify_batch_b.py`).
- **Every detector must FIRE on at least one verification day.** A 0-vs-0 match is NOT
  evidence (the PIVOT-16 lesson, doc 058). Pick days accordingly.
- CROSS-11: state explicitly how you seed the 2400-bar buffer (prior-day concat, as legacy
  does) or declare the early-session divergence with counts.
- Paste the FULL verifier output. Mark each detector MATCH or DIVERGENCE-because-X.
- FPS core FROZEN — foreign daily context is injected out-of-band into the detector
  `__init__`, never by touching FPS.
- Your status on the execution doc: `EXECUTED — AWAITING VERIFICATION`. The VERIFIED
  stamp is mine.

## 4. Standing
APPROVED for Sub-Batch 1 (ADX-08, ATR-09, CROSS-11, DOW-19, FIB-17) with Mods 1-2.
Execution report = doc **065**. On verification you proceed to Sub-Batch 2 (next 6).
Batch A (7/7) stands. Good work on the correction round — the citations are real now,
and that is what made this reviewable.
