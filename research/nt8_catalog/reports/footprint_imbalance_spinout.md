# Footprint Imbalance spin-out — STOPPED at Phase 1 (order-flow graveyard)

**Task:** 121 · **Date:** 2026-07-18 · **Drone:** Opus · **Reviewer:** Claude Fable
**Verdict:** **STOP after Phase 1.** The rule requires order-flow (aggressor buy/sell
delta) data, which is the exact input the project REJECTED on 2026-06-28 ("We will
NOT purchase tick data" — the order_flow graveyard). No sealed test was run. Per the
task's explicit data constraint, this is the required action, not a choice.

---

## Phase 1 — EXTRACT

### The exact rule (verbatim)
Source: `research/archive/leg_clock/tools/ag_cat_10_footprint.py`
(class `FootprintImbalanceConcept`), scored by `ag_cat_harness.py::run_sweep`.

```
Concept 10: "Footprint Imbalances (Proxy via Extreme Delta)"

Signal, evaluated ONLY at each 1-minute close, on the 5s bar aligned to that close:
    d = order_flow_delta_5s.delta[ ts_of_1m_close ]   # contracts, aggressor buy − sell
    if d >  +200 contracts  ->  +1  (Long)
    elif d < −200 contracts ->  −1  (Short)
    else                    ->   0  (flat)

Hold: position = signal.shift(1); held one minute (re-evaluated next 1m close).
Forward-return null: 15m forward price change (15 one-minute bars).
```

- **Threshold:** ±200 contracts (a hand-picked round number — NOT tuned, NOT sealed).
- **Timeframe:** signal sampled at 1m closes; underlying feature is 5s delta.
- **Direction thesis:** extreme localized delta = aggressive institutional
  participation → short-term momentum continuation (a RIDE/run bet, not a snap-back).

### Data input — THE BLOCKER
The rule reads `DATA/ATLAS/order_flow_delta_5s.parquet`, columns
`[open,high,low,close,volume,delta,cum_delta,price_change,divergence]`. `delta` is
**aggressor order-flow delta** (signed contract imbalance per 5s bar). This is
order-flow / tick-derived microstructure data.

Two independent reasons this is unavailable for a sealed test:

1. **It is graveyard-REJECTED.** Journal `docs/daily/2026-06-28.md` (Order Flow BREAK):
   True Delta gave only +0.0036 AUC lift over OHLCV wicks AND failed the Fourier
   phase-randomized null (Fold-5 AUC 0.6363 vs 95th-pct null 0.6406). Verdict: BREAK,
   "We will NOT purchase tick data." The Footprint rule is the same data class.
2. **The sealed-test design is physically impossible.** The parquet covers only
   **2025-07-30 → 2026-01-29** (158 calendar days). There is **zero order-flow data in
   2024** — so the ±200 threshold (a free parameter) CANNOT be fixed on 2024 and
   frozen, which the task requires before any single-shot test. (This is the same
   "year coverage" reason ORDERFLOW-14 was excluded from the ladder, doc 038.)

### How the June $18.63 gross / $17.23 net /day was computed — and its rigor gaps
Population: `run_sweep(years=['2024','2025'])` iterated every L0 day, 15m forward
returns at 1m closes, vectorized 1-contract backtest, block-bootstrap (B=1000) over
days. Reported: 0.65 trades/day, gross $18.63/day, net $17.23/day (4t RT cost),
CI [$12.27, $22.99], gap 2.33 pts → "REAL."

**Rigor gaps (all verified this session):**

1. **The killer: 536-day denominator, 6-month numerator.** The sweep averaged over
   **536 L0 days**, but delta data exists for only **158 days**, of which **114 overlap
   the sweep**. So **422 of 536 days (79%) are structural zeros** — `delta_map.get(ts, 0.0)`
   returns 0 on every bar with no data → signal 0 → 0 trades → net 0. These are
   MISSING-DATA zeros, not real flat days. The $17.23/day mean is diluted ~4.7× (true
   per-active-day economics ≈ $17.23 × 536/114 ≈ $81/active-day), and the CI is
   artificially compressed toward the diluted mean by the mass of identical zeros.
   The headline number is not interpretable as either a per-active-day figure or a
   properly-sampled strategy.
2. **No IS/OOS split, no threshold sealing.** ±200 is in-sample on the full 2024–25
   span; never OOS-validated. A single in-sample number.
3. **Weak null.** Baseline = the day's OWN mean 15m forward return; not a
   phase-randomized / permutation null. (When a proper Fourier null was later applied
   to the sibling delta signal, it FAILED — 2026-06-28.)
4. **Arbitrary 1m-close sampling of a 5s feature.** Only 1 of the 12 five-second
   delta bars per minute is inspected (the one at the 1m close). Signal cardinality
   choice, not a physical one.
5. **B=1000 bootstrap** (current standard is 4000); plain day resample, no day-block
   over overlapping windows. (Cost was 4t RT = $2.00, actually MORE conservative than
   the current 2.4t convention — the one gap that runs against the optimistic direction.)

**Net:** even setting the data-availability blocker aside, the June "+$17.23/day REAL"
is a diluted, unsealed, weak-null in-sample artifact. It would not clear current rigor.

---

## Phase 2 — NOT RUN (blocked by the data constraint)

- **League line (FOOTPRINT-IMB):** not ported. Would require order-flow delta at every
  bar over the dossier span; data absent outside the 6-month window and graveyard-
  rejected. No pipeline append made.
- **Entry-fail filter (main event):** not run. The powered-frontier test population is
  2025-26; the delta window (2025-07-30 → 2026-01-29) only partially overlaps it, and —
  decisively — the ±200 threshold cannot be sealed on 2024 (no order-flow data exists
  there). Running it would violate both the "seal free thresholds on 2024" rule and the
  standing REJECTED decision. No runner created.

## Pre-registered bar
Not evaluable — no filtered vs base rate was produced (test not run). Default outcome
under the data constraint: **back to the archive.**

## Card verdict (for the archive)
`10_Footprint_Imbalance`: **REJECTED — order-flow data (graveyard, 2026-06-28); June
"+$17.23/day REAL" is a 536-vs-114-day dilution artifact, unsealed, weak-null.** Do not
resurrect without (a) reversing the no-tick-data decision AND (b) a full 2024 order-flow
history to seal the ±200 threshold — neither exists.

## Deviations
- Did not append to `dossier_signal_pipeline.py` or create `tools/footprint_spinout.py`
  — Phase 1 mandated STOP, so no test code was written (nothing to seal, nothing to run).
  Only this report was produced. Commit: none.
