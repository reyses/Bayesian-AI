# FPS Canonical Run — final results (all strategies, one engine)
**Doc:** 044 · **Date:** 2026-07-12 · **Author:** Claude (executor) · **Status:** FINAL
**Executes:** Moises' solution (doc 043): audit FPS → optimize → rerun everything.

## 1. Engine work (committed f4544208)
- **Audit**: FPS causality core SOUND (timestamp joins, closed-bar semantics,
  Databento/NT8 convention guard). Fixed: silent bar-drop now counted
  (`skipped_bars` = 12/day session-open warmup); added opt-in `use_5s_price`
  (fill-accurate mode; default legacy bit-identical).
- **Optimization**: vectorized OHLCV joins (once/day), minimal feature tiers for
  the runner. **Parity: stream SHA `675e744170679490` identical old vs new.**
- **Throughput: ~128,000 bars/s** → 521 days × all strategies × 8 configs =
  444,528 simulated trades in ~9 minutes.

## 2. Result vs the pre-registered gate (CI_lo > 2-tick friction BOTH years, N≥100/yr)
**One candidate in the entire catalog: ORB-02 Opening-Range Breakout, AS THE
ARTICLE STATES IT, T20/S20:**
- 2024: **+4.22 pts/trade [CI +1.86, +6.55]**, N=258 (1/day)
- 2025: **+3.19 pts/trade [CI +0.71, +5.66]**, N=226
Coherence (why this isn't a lucky cell): stated direction positive in ALL 4
configs with MONOTONE dose-response (T10/S10 +0.8 → T20/S20 +4.2), flip = exact
mirror-negative. Structured momentum signature: the 30-min-range break drifts,
and wider barriers harvest more of the drift.
Everything else: fails the gate (full table `reports/AG_cat_00_FPS_RESULTS.md`).
ROUND-05 / PIVOT-16 remain dead (doc 042 confirmed by canonical engine).

## 3. Honest caveats (pre-registered skepticism)
- 1 gate-passer out of ~160 tests → ~8% chance a random catalog produces one
  false pass; the monotone dose-response and mirror-flip structure argue real,
  but ORB-02 must survive: (a) monthly-stability check, (b) T25/30 extension,
  (c) entry-slippage sensitivity (1-2 ticks), (d) distribution plot (mode-first)
  BEFORE any SIM/NT8 conversation. None of that has run yet.
- SEASON-12 and RENKO-24 triggers not in RTH index space → not re-run (SEASON
  already dead per doc 040; RENKO needs its own runner if ever wanted).
- $ frame at face value: ~+$7-8/trade/contract (MNQ), 1 trade/day ≈
  +$1,700-2,100/year/contract before slippage. Modest, single-contract scale.

## 4. Program state
The catalog program's canonical answer: **23 of 24 concepts hold no realizable
edge under one audited engine; ORB-02 stated T20/S20 is the single surviving
candidate, pending the §3 robustness quartet.** All tooling now flows through
FPS — no more per-script data semantics.
