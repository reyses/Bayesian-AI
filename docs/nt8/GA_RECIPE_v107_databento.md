# Genetic Optimization Recipe — v1.0.7-RC vs v1.0.6-RC on MNQCONT (Databento 14-month)

**Date**: 2026-04-28
**Goal**: find best parameter combo for ZigzagRunner on the full Databento back-adjusted continuous dataset, and validate that dynamic R (v1.0.7-RC) generalizes better than static R (v1.0.6-RC) — without the 32-day window-fit problem.

## Why two GAs (not one)

The 2026-04-28 v1.0.6-RC GA winner (R=45 / SL=90 / Trail=21,0.05) produced
**+$74/day on the 32-day Playback window** but **-$75/day on the full 14-month
sweep** — pure window-fit. Running a fresh GA on the 14 months with v1.0.6-RC
parameters only is the **honest test** of whether static R can survive a longer
window. Running v1.0.7-RC dynamic R on the same 14 months shows whether the
dynamic-R hypothesis is real edge or just adds dimensions to overfit.

## Risk-management framing

This is **NOT a single-shot GA winner ⇒ live trade** workflow. It's:

1. **GA-A** (v1.0.6-RC, ~7-D search) ⇒ candidate A
2. **GA-B** (v1.0.7-RC, ~12-D search) ⇒ candidate B
3. **Hold-out**: re-run both candidates on a withheld 2-month slice (e.g. last 2 months of 2026 ATLAS) WITHOUT re-tuning. Whichever PnL holds up wins.
4. **Walk-forward**: NT8 has built-in Walk-Forward Optimization — gold standard. Optional but recommended for live deployment confidence.

If neither candidate's hold-out PnL ≥ baseline (v1.0.4 = ~+$50/day), DO NOT
go live Thursday with v1.0.7-RC. Stay on v1.0.4.

## Pre-flight checklist

- [x] v1.0.7-RC compiled in NT8 (`docs/nt8/ZigzagRunner_v1.0.7-RC.cs` deployed, MD5 match confirmed `d7b44ae3...`)
- [x] v1.0.6-RC compiled in NT8 (already there)
- [x] MNQCONT custom instrument exists in NT8 Instrument Manager
- [x] Databento monthly tick files imported into MNQCONT (15 months, 31.7M ticks)
- [ ] Optional: confirm tick data integrity by overlaying MNQCONT chart vs MNQ JUN26 chart for an overlapping day

---

## GA-A: v1.0.6-RC parameter sweep on MNQCONT

**Strategy**: `ZigzagRunner_v1.0.6-RC`

**Strategy Analyzer setup**:
- Instrument: `MNQCONT` (or whatever your custom instrument's exact label is)
- Period: 1 minute (chart is independent — this is just the loaded series)
- Time range: full 14-month available (e.g. 2025-01-01 to 2026-02-28). Optionally hold out the last 2 months for OOS validation, run GA on first 12 months only.
- Optimizer: **Genetic** (NOT Default exhaustive — combinatorial explosion)
- Generations: 30-50 (NT8 default 25 may converge prematurely on 7-D)
- Population size: 50
- Stability: 3 (uses median across 3 sims to reduce noise)
- Fitness criterion: **Net Profit** (or Profit Factor if you want robustness — NetProfit is more aggressive)

**Parameter ranges** (these are the search axes for GA-A):

| Parameter | Min | Max | Step | Notes |
|---|---|---|---|---|
| `RPoints` | 15 | 75 | 5 | static R search — covers tested range |
| `MaxUnrealizedLossPoints` | 30 | 150 | 10 | SL sweep (0=disable, 30=tight, 150=loose) |
| `MfeCutBarsAfterEntry` | 0 | 30 | 1 | bar at which to apply MFE-cut |
| `MfeCutThresholdUsd` | 0 | 50 | 5 | dollar threshold for MFE-cut |
| `TrailActivatePoints` | 0 | 50 | 5 | trail arm threshold (0=disable) |
| `TrailGivebackPct` | 0.05 | 0.50 | 0.05 | trail giveback ratchet |
| `Contracts` | 1 | 1 | 0 | hold at 1 (don't optimize sizing) |

**Pin these (do NOT optimize)**:
- `PivotTfSeconds = 60` (1m, baseline)
- `HardSlTfSeconds = 1` (1s SL check)
- `OnHighPivot = Short`, `OnLowPivot = Long` (counter-trend baseline)
- `EodHourUtc = 20`, `EodMinuteUtc = 55`
- `EntryCutoffHourUtc = 20`, `EntryCutoffMinuteUtc = 30`
- `SlippagePoints = 0.25` (1 MNQ tick — keep realistic)
- `CommissionPerRoundtripUsd = 1.90` (matches live broker)

**Expected runtime**: 30-90 min on 12-month MNQCONT depending on CPU.

**Output**: top-10 combos sorted by Net Profit. Capture screenshot or export
to CSV.

---

## GA-B: v1.0.7-RC dynamic R sweep on MNQCONT

**Strategy**: `ZigzagRunner_v1.0.7-RC`

**Same Strategy Analyzer setup as GA-A** but:

**Parameter ranges**:

| Parameter | Min | Max | Step | Notes |
|---|---|---|---|---|
| `UseDynamicR` | true | true | (fixed) | force dynamic on for this GA |
| `RPoints` | 30 | 30 | 0 | fallback during ATR warmup (pin) |
| `AtrLookbackBars` | 20 | 240 | 20 | ATR window length on 1m pivot bars (20m to 4h) |
| `AtrMultiplier` | 0.5 | 15.0 | 0.5 | the core scalar — biggest fitness driver |
| `MinRPoints` | 5 | 30 | 5 | floor on dynamic R |
| `MaxRPoints` | 50 | 300 | 50 | ceiling on dynamic R |
| `MaxUnrealizedLossPoints` | 30 | 150 | 10 | SL sweep (re-optimized vs static R) |
| `MfeCutBarsAfterEntry` | 0 | 30 | 1 | re-optimize relative to dynamic R |
| `MfeCutThresholdUsd` | 0 | 50 | 5 | re-optimize |
| `TrailActivatePoints` | 0 | 50 | 5 | re-optimize |
| `TrailGivebackPct` | 0.05 | 0.50 | 0.05 | re-optimize |

**Why pin `UseDynamicR=true`**: GA-A already covers the static-R case. GA-B
isolates the question "does dynamic R help?".

**Pin same as GA-A**: PivotTfSeconds=60, HardSlTfSeconds=1, OnHighPivot=Short,
OnLowPivot=Long, EOD/cutoff times, slippage, commission.

**Why bigger search space matters**: 12-D ≫ 7-D. Use Genetic mode (NT8 default
exhaustive will not finish). Bump generations to 50 minimum.

**Expected runtime**: 60-180 min on 12-month MNQCONT. Be patient; 12-D needs
generations to converge.

---

## Comparison protocol (after both GAs finish)

### Step 1 — top-3 from each GA

Pull top-3 combos by Net Profit from each. Record:
- All param values
- Net Profit
- Trade count
- Profit Factor
- Max DD
- Win Rate

### Step 2 — hold-out validation

Run each top combo on the **last 2 months of MNQCONT** (which were NOT in the
GA training window). Use **Backtest** mode (not optimizer) — single run,
fixed params.

Record same metrics. **The drop from training to hold-out is the overfit tax.**
Healthy strategy: drop < 30%. Unhealthy: drop > 60% or sign flip.

### Step 3 — head-to-head decision

| Outcome | Decision |
|---|---|
| GA-A and GA-B hold-out both ≥ +$30/day | Pick the one with smaller train→holdout drop (more robust). |
| GA-A holds, GA-B fails | Static R won. Dynamic R doesn't add edge here. Use GA-A combo. |
| GA-A fails, GA-B holds | Dynamic R is real edge. Use GA-B combo. |
| Both fail | **Stay on v1.0.4.** Do not go live with either candidate. Rebuild the EDA hypothesis. |
| Both hold but barely (<$30/day) | Stay on v1.0.4 baseline (+$50/day). Edge isn't worth the complexity. |

### Step 4 — if going live Thursday

- Apply the winning version + winning params to a fresh MNQ JUN26 chart on Sim101
- Let it run 1+ session in Sim before flipping live
- Monitor [v1.0.7-RC COST SUMMARY] in Output panel — the Print line shows the runtime params + roundtrip count + commission accrual

---

## Common gotchas

- **NT8 Optimizer ignores Slippage on the strategy**: even though we drive
  `Slippage = SlippagePoints` in Configure, the optimizer's internal sim may
  use 0 slippage. Verify by looking at the "Strategy" tab inside the optimizer
  result — if `Slippage` field is 0 but our property is 0.25, the optimizer
  won the wrong battle. Fix: explicitly set Slippage in the optimizer toolbar
  too.

- **Commission ≠ NT8 trade-performance commission**: our `CommissionPerRoundtripUsd`
  is a print-only diagnostic. NT8's optimizer fitness uses whatever commission
  template is attached to the account/instrument. If you want commission baked
  into fitness, attach the "Free" template (or your live template) to MNQCONT
  in the Account/Instrument settings BEFORE running the optimizer.

- **GA-B convergence**: 12-D space + 50 generations + pop 50 = 2,500 fitness
  evaluations × 12 months data each. Don't be alarmed by 2-3 hour run times.

- **MNQCONT vs MNQ JUN26 mismatch**: MNQCONT is back-adjusted continuous; it
  has NO contract roll gaps. If your live deployment is on MNQ JUN26, validate
  that the front-month behaves similarly to the continuous near contract
  boundaries (mid-March 2026 was the last roll).

---

## After both GAs

Update `docs/daily/2026-04-28.md` with results:
- GA-A top-3 combos + holdout PnL
- GA-B top-3 combos + holdout PnL
- Decision (which version + which combo wins)
- If go-live Thursday: confirm on Sim101 first

If neither survives hold-out, escalate to walk-forward optimization (NT8's
WFO is the same Strategy Analyzer with "Walk Forward" mode — gold standard
for parameter robustness).
