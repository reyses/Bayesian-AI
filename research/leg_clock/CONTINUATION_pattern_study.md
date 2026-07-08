# CONTINUATION BRIEF — wall-footprint precursor pattern (AUTONOMOUS RESUME)

Written 2026-07-07 for a scheduled autonomous resume (~1h later). This thread
with Moises is very long; on wake, READ THIS + latest `docs/daily/` journal +
`MEMORY.md` first. Do NOT re-derive from scratch.

## Standing rule (agreed with Moises today)
Agree the spec before executing. THIS brief is the agreed direction — it is the
sign-off to execute this research autonomously. Research only: do NOT touch
training / reward / live. Commit + journal each step. Keep chat replies SHORT
(his stated preference).

## The question
What precedes a trend? Characterize the microstructure PATTERN in the
oscillation → trend transition. Specifically (his words): **how many times do
we see the wall-footprint before we see the slow/fast trend, and how do the
wicks and bodies look through the process.**

## Wall footprints (deduced from OHLCV — no order book in history)
- **Absorption wall** = SMALL wick + HIGH volume, repeated at a level (price
  grinds against it and holds; it absorbs contracts). Small wick + LOW volume =
  just quiet, not a wall.
- **Rejection** = BIG wick (price spiked to a level and got slammed back).
- **Defended level** = wick beyond a level BUT close returns to it (failed
  breakout). Footprint only — we canNOT prove active injection vs no
  follow-through (no book); measure the pattern, not the intent.

## Volume handling (settled today)
- Aggregation in the pipeline is correct (summed to higher TFs).
- BUT raw volume is not comparable across TFs. Use a RATE = contracts/second.
- Cross-TF feature = short-window rate / long-window rate (acceleration), plus
  rate vs time-of-day normal. Volume = participation (symmetric; no direction).

## Tasks (in order, autonomous)
1. Trend onset = start of a macro leg (zigzag ~150 ticks on 1m; ~a handful of
   legs/day — matches his eye). Reuse `pivot_level_proximity.zigzag_pivots`.
2. Build a per-bar wall-footprint detector (absorption / rejection / defended)
   from 5s or 1m OHLCV + volume-rate.
3. In the window BEFORE each leg onset (try 30 + 60 min), COUNT footprints and
   measure wick/body evolution (wick:body ratio, body-size trend, footprint
   counts).
4. Compare pre-trend windows vs pre-nothing (oscillation-continues) windows:
   does the footprint pattern differ? Null = random windows. Report the lift +
   a null anchor.
5. Split slow vs fast resulting trend (by leg velocity): do precursor patterns
   differ (his thesis: fast/violent legs look different going in)?
6. OOS: 2024 vs 2025. Distributions / mode, not point estimates. Must beat the
   null or it's noise.

## Data / tools
- `DATA/ATLAS/{5s,1m}/*.parquet` (OHLCV+volume). 2024=259 days, 2025=277.
- Reuse `research/level_hold/tools/level_hold_study.py` (atlas loader,
  rolling_ols_bands w/ return_sigma), `pivot_level_proximity.py` (zigzag),
  `research/leg_clock/tools/*` (leg segmentation, MRL).
- Run via `.venv_wsl/bin/python ...` (WSL). Reports → `research/leg_clock/reports/`.

## What we already know (don't retest)
- Micro-bounce real but unconditional (any line bounces 60%); levels weak
  (+1-2pp); touch-count dead; slope doesn't persist bar-to-bar; leg length =
  momentum/fat-tail (rising MRL) but confirm-then-ride backtest LOSES causally
  (-$60..-200/day OOS). So momentum alone isn't tradeable — the entry FILTER is
  the open problem. THIS study is hunting that filter in candle microstructure.
