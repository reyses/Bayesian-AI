# RM Pivot Research — Project (DMAIC)

> One file governs the project. Cycles live in `cycle_NN.md` alongside.
> Reports live in `findings/YYYY-MM-DD_<topic>.md`.

---

## DEFINE

### Problem statement

Per journal entry 2026-04-21 (`docs/daily/INDEX.md`):
- Direction prediction at arbitrary bars = coin flip (91D Cohen d max 0.01)
- **Direction at regression-mean zigzag pivots via `1m_z_se` residual = Cohen d ≈ −2.46, 86% oracle accuracy** (walk-forward)
- Regression cord (honest ceiling) = $2.5–3K/day; NMP captures 2.3% → 40× headroom
- Planned action (literal journal quote): *"build forward pass on regression-line pivots (not price pivots), target 20-30% of regression cord = $500-750/IS, $575-862/OOS"*

### Goal

Build and validate a forward pass on regression-line pivots with direction from residual. Capture 20–30% of RM cord.

### Scope

- **Instrument**: MNQ (1 contract, chains=1 until equity ≥ $600)
- **Data**: `DATA/ATLAS/1m/` + `DATA/ATLAS/FEATURES_5s/` (IS 2025, OOS 2026)
- **TF**: 1m trading timeframe, 5s feature cadence
- **In scope**: RM zigzag pivot detection, residual-direction rule, exit rule, honest trade-profitability
- **Out of scope**: CNN overlays, 9-tier cascade, cross-instrument, leverage

### Metric definitions

- **Trade WR** = (∑profit / |∑loss|) − 1
  - 0 = break-even · +1.0 = profit 2× loss · −0.5 = profit half the loss
- **$/trade** = report BOTH mode (typical) AND mean (tail-inclusive)
- **Day WR** = count-based: winning days / total days

### Success criteria (project-level)

1. IS captured / RM_cord ≥ 15% (entry gate toward 20–30% target)
2. Daily PnL mode ≥ $0, median ≥ $100
3. OOS/IS PnL ratio ≥ 0.7 (no overfit)
4. Trade WR ≥ +0.5 (profit is ≥1.5× loss) on net-$10 trades
5. Mode $/trade ≥ $10 AND Mean $/trade ≥ $10 (slippage + commissions added in Control)
6. Day WR ≥ 60% (winning days / total days in forward pass)

### Kill switch

If after 3 measured improvements WR stays ≤ 52% on net-$10 trades, the RM-pivot premise is wrong. Abandon cycle, journal why, pivot.

### Constraints (user-supplied)

- Starting equity: **$100** (no chains until $600)
- Trade evaluation floor: **$10 net** per trade
- Slippage + commissions + other execution costs: deferred to Control phase (applied AFTER signal + forward pass validated)

---

## MEASURE

Living log of baselines. One subsection per measurement.

### M1 — RM cord length per day (baseline ceiling)

_Planned in Cycle 2 prep. Uses `tools/cord_length_regression.py`._

Pending.

### M2 — Cohen d verification of direction signal at RM pivots

_Cycle 1 — CLOSED, STANDARDIZED. See `cycle_01.md` + `findings/2026-04-22_cohen_d_verify.md`._

**Result**: Signal real and reproducible.

| R | IS \|d\| | OOS \|d\| | Monthly std | Flipped Accuracy |
|---:|---:|---:|---:|---:|
| $2 | 1.92 | 1.87 | 0.17 | 82.8% |
| **$4** | **1.96** | **1.95** | **0.16** | **83.5%** ← picked |
| $6 | 1.83 | 1.94 | 0.25 | 81.8% |
| $10 | 1.56 | 1.67 | 0.28 | 78.3% |

**Corrected direction rule**: LOW pivot → LONG, HIGH pivot → SHORT. (Initial assumption was mean-reversion; data shows trade-the-pivot is correct.)

**Physics explanation**: 60-bar OLS lags turning price. At a peak, price stalls while RM keeps climbing → residual at the HIGH pivot is negative (mean −0.99). Mirror at LOW pivots.

**Carryover to Cycle 2**: enter on confirmed RM pivot at R=$4 using the corrected direction rule.

### M3 — RM pivot frequency per day (at chosen $R)

_Pending._

### M4 — Current iso_is.pkl capture-ratio vs RM cord

_Pending — sanity check on the 2.3% figure from the journal._

---

## ANALYZE

_Populated per cycle after Check/Act. Holds root-cause findings that inform future Improves._

### A1 — Direction accuracy at pivots ≠ tradeable edge

**Source**: Cycle 2 CHECK, 2026-04-22.

Cohen d=1.96 and 83.5% accuracy at RM pivots are real (Cycle 1). But naive pivot-to-pivot forward pass LOSES money at every R tested ($2→$20). Root cause: 2R retracement tax.

- Entry tax = $R (confirmation requires price to move $R in our direction after the extreme)
- Exit tax = $R (next pivot requires $R retracement against us)
- Net capture per trade = (leg amplitude) − 2R
- At all tested R, mean leg amplitude ≤ 2R + small margin → negative or near-zero mean trade

**Implication**: The entry signal is a necessary condition but not sufficient. The exit rule must NOT pay a second R tax. Future cycles must either:
1. Use a non-zigzag exit (e.g., residual-zero cross, fixed TP)
2. Use asymmetric R (enter big, exit small)
3. Filter for trades likely to exceed 2R amplitude

### A1b — Cohen d at pivots was a tautology, not a forecast (added Cycle 3)

Cycle 1's |d|=1.96 measured residual sign ↔ pivot type correspondence. By zigzag definition, pivot type (HIGH/LOW) = direction that just finished — so residual (a function of how price/RM moved lately) always matches. It is a post-hoc label-check, not a prediction.

Cycle 3 tested what trading actually needs — "given a pivot confirms, what direction goes next?" — and got 49-50% across every R and both pivot sources (RM $4, price $40). The |d| signal exists but has no predictive power on future direction.

**Rule to remember**: a high Cohen d between two label-dependent statistics is not evidence of tradeable edge. Tests must use an independent forward target.

### A2 — RM cord capture currently 0.24% of ceiling

At R=$10 (best-case Cycle 2), IS $/day = $7 / ~$2,750 RM cord ceiling = **0.24%** capture. Target is 20-30% = 80-120× away. The entry trigger alone doesn't get us there; we need a much smarter capture per pivot.

---

## IMPROVE (cycles)

| Cycle | File | Status | Hypothesis | Result |
|---|---|---|---|---|
| 01 | [cycle_01.md](cycle_01.md) | **STANDARDIZED** | Cohen d = −2.46 claim is real and reproducible | **PASS** — \|d\|=1.96 IS / 1.95 OOS. Direction rule **inverted**: trade the pivot (LOW→LONG, HIGH→SHORT). 83.5% accuracy. |
| 02 | [cycle_02.md](cycle_02.md) | **ADJUST** (all 7 gates failed) | Naive pivot-to-pivot exit nets ≥ $10/trade on the verified signal | **FAIL** — best R=$10 yields mean $1.06/trade, median −$25, tradeWR +0.02. 2R retracement tax consumes leg amplitude. |
| 03 | [cycle_03.md](cycle_03.md) | **ADJUST** (signal portfolio 2/5) | Direction signal exists at forward horizons after RM pivots | **FAIL** — Q1 direction HR 49.7% IS / 50.3% OOS = coin flip. Also tested price R=$40: 49.9% both. Cycle 1's \|d\|=1.96 was STRUCTURAL, not predictive. Q2 (turning-point) and Q4 (oracle ceiling $215-292) passed. |
| 04 | cycle_04.md (pending) | PLAN pending user pick | Direction predictor search / pivot-as-exit / abandon | TBD |

---

## CONTROL

- **Journal**: `docs/daily/YYYY-MM-DD.md` updated at session boundaries
- **Index**: `research/INDEX.md` row reflects current status + latest cycle
- **Monitoring**: every cycle's Check section compares actual to predicted; drift triggers ANALYZE revisit
- **Commit discipline**: every Standardized cycle = one git commit; every Abandoned cycle = journal entry + MEMORY.md update

### Reproduction

All cycle DO phases have a `Method` block with the exact CLI to re-run. Findings reports carry the same command.
