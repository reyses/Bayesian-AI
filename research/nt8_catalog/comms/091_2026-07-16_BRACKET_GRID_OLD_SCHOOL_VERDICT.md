# Old-school SL/TP brackets on the calibrated entries — measured verdict
**Doc:** 091 · **Date:** 2026-07-16 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executor:** Sonnet worker (ladder trial #6). Moises: "we can do it old school with
stop loss and take profit dumb strategies — we have options."

## 1. Reviewer notes on verification (and one error of MINE the worker caught)
- Fills verified at row level: **0/80,246 mismatches** vs the established drift
  convention; residual 0.6% divergences fully explained (5s data gaps; ceiling-vs
  -floor timeout convention) with the worst case being the 2025-04-09 tariff spike.
- **My spec error, caught by the worker**: I quoted +1.75 as the fixed-5m
  reference; +1.75 is the 15m median — the 5m median is +3.25 (doc 088). The
  worker cross-checked the source docs instead of hammering its sim to match my
  wrong number. Exactly the behavior the ladder is supposed to produce.
- Dedup finding (real information): removing same-direction re-fires within 60s
  drops the 5m median from +3.25 to +2.00 — the tight re-fire BURSTS carry extra
  drift. Confirmation freshness is a real state variable (matches the consensus
  and P_hold-freshness threads).

## 2. Sealed old-school results (cell chosen on 2024 ONLY; test quoted)
| pop | cell | TEST mode | median | mean [CI] | PF-WR | mix (stop/target/timeout) |
|---|---|---|---|---|---|---|
| A top-decile | SL20/TP30/60m | **−21.0** | **−20.0** | +2.06 [+1.43,+2.68] | +0.16 | 54/43/3% |
| B bottom-inv | SL20/TP12/15m | **+12.0** | **+12.0** | +0.83 [+0.58,+1.07] | +0.12 | 29/55/16% |
References (no bracket): A fixed-5m mean **+2.87 [+2.02,+3.69]**, PF-WR **+0.21**;
B fixed-5m +1.23 [+0.93,+1.53], PF-WR +0.18.

## 3. Verdict (mode-first, per house rules)
- **Population A's sealed bracket is the outlier-day trap in miniature**: the
  TYPICAL trade loses 20 pts (mode −21, median −20); the positive mean is a fat
  right tail. Not a usable shape — the worker flagged it itself.
- **Brackets do not ADD expectancy on either population**: A bracket mean +2.06
  < plain 5m hold +2.87 (and PF-WR 0.16 < 0.21); B bracket +0.83 < hold +1.23.
- **What brackets DO is reshape**: B's sealed cell turns the distribution into a
  consistency machine (55% of trades end exactly +12, median/mode +12) at the
  cost of 29% −20 stops and lower total EV. Worth knowing for reward design;
  not an edge additive.
- The graveyard rule extends to the NEW population: fixed stops/targets subtract
  vs short holds here too. Old-school lane: measured, closed.
- Caveat: close-based fills (no intrabar H/L) — live SL hits would be worse,
  which only strengthens the negative verdict.

## 4. State of the options board
1. ~~Old school brackets~~ — measured, not additive (this doc).
2. **Turn Catalog** — 16 concepts drafted + verified (TURN_CATALOG_DRAFT.md);
   port queue: TURN-10 HeikinAshi flip, TURN-07 Sweep-and-Reclaim, TURN-06
   climax exhaustion; scorecard = turn dir-recall@±2m vs 0.43 chance + lead/lag.
3. **Sequential (Mamba)** — the remaining lane for the 16× exit gap; handoff
   spec next, with lanes 1-2 as its measured baselines.
Artifacts: reports/bracket_grid.md, bracket_fills.parquet, bracket_run.log,
tools/bracket_grid_sweep.py.
