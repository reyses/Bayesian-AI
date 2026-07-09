# AG TASK — NT8 catalog sweep: one standalone test per concept

**For: Antigravity.** Self-contained. SUPERSEDES `AG_TASK_churn_structure.md`.
MNQ repo; run `.venv_wsl/bin/python` (WSL); paths repo-root-relative.

## Mission
The user compiled 463 NinjaTrader educational articles into 4 pillars
(`research/nt8_catalog/*.md`). These are BROKER EDUCATION — prior of a real
edge is LOW; if anything survives it's likely via crowd-attention effects
(thousands of retail eyes on the same level). Your job: give each testable
concept its own STANDALONE test and its own report. No cross-contamination
between concepts. A synthesis/combination step happens only AFTER the sweep,
with the user.

## Per-concept protocol (identical template, every report)
1. **Definition** — exact causal math (trailing windows only; sigma-relative
   distances, never fixed ticks).
2. **Existence test** — the concept's signal vs a matched same-day/same-hour
   null, 2024 AND 2025 separately. Signal bar: gap ≥0.10 REAL, 0.05–0.10
   CONDITIONAL, <0.05 NOISE.
3. **Economics test** — the concept as a standalone mechanical rule (its own
   article's entry/exit logic where stated): $/day with day-block bootstrap
   95% CI, PF, trades/day, 4 ticks/round-trip costs. Say "NOT significant"
   when CI includes 0. Both years; a config choice made on one year must
   hold on the other.
4. **Verdict line** — REAL / CONDITIONAL / NOISE / UNTESTABLE, one sentence
   why.
Report file per concept: `research/leg_clock/reports/AG_cat_NN_<name>.md`.
Tools: `research/leg_clock/tools/ag_cat_NN_<name>.py`, standalone-runnable.

## Concept queue (priority order)
01 Prior-day OHLC levels (+ floor pivots R1/S1) — crowd-attention candidate #1
02 30-min opening range break (ORB)
03 Session VWAP z-score mean reversion (VWAP as gate/target)
04 Volatility squeeze: Bollinger bandwidth contraction → expansion break
05 Psychological round numbers (00/50 levels) as liquidity pools
06 RSI divergence at extremes
07 MACD divergence
08 ADX>25 trend gate (as regime filter on a basic rule)
09 Statistical ATR fade ("90% rule" as stated in the article)
10 VWMA vs SMA divergence
11 Golden cross baseline
12 Seasonality / day-of-week effects
13 Value-Area rotation rules (VA-break → target other side) — NOTE: the
   naive "walls" version is already tested: turns sit at LOW-volume nodes /
   OUTSIDE the VA (weak, 2-yr stable); VP entry-gate flips sign across years.
   Test only the ROTATION rule variant, don't redo ours.
14 (tail, only if time) Renko time-filter, Fibonacci targets, H&S divergence.
UNTESTABLE (no tick/BidAsk history — flag, skip): footprint imbalances,
cumulative delta, trapped-buyers.

## Data & hard rules (each has burned this repo)
- `DATA/ATLAS/{5s,1m,5m,15m,1h}/*.parquet`, 2024 (259d) + 2025 (277d).
  Tick 0.25, $0.50/tick. **`DATA/ATLAS_NT8/` is SEALED — never touch.**
- Feature extraction via `core_v2/FPS/forward_pass_system.py` or strictly
  trailing raw-parquet windows. Your previous 0.87 AUC collapsed to 0.55
  when re-extracted leakproof (the event bar had leaked). Never index at or
  after the event bar.
- **LABEL-FREE**: do not use `DATA/ai_cusp_picks/` (hindsight-snapped; known
  traps). Judge on forward price outcomes and dollars.
- Dead list (don't retest as-is): volume-rate buildup, candle wick/body
  shapes, band-level first-touch bounce, zone touch-count, bar-to-bar slope
  persistence, confirm-then-ride at leg scale, APZ re-entry confirmation,
  VP entry-gate.
- Session context if needed: profitable-hours reference = 9–13 CT (measured);
  15:55 CT flatten in any economics sim.

## Deliverables
- One report per concept (template above), committed as you go.
- Final index: `AG_cat_00_INDEX.md` — one verdict line per concept, ranked.
- The survivors list (REAL or CONDITIONAL on BOTH years) — these go to the
  user for the combination phase. Expect most to die; a clean NOISE verdict
  with tight methodology is a fully successful outcome.
