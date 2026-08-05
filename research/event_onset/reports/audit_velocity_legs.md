# Adversarial audit — `velocity_legs.py` / `velocity_legs.md`

2026-08-04. Independent re-measurement of `research/event_onset/tools/velocity_legs.py`,
commissioned on the owner's suspicion that it measures the wrong thing.
The target file was **not modified**. All numbers below were recomputed from
`DATA/ATLAS/1s`, 138 files → 112 RTH days (2025-01-02 .. 2025-06-19).

Tools: `research/event_onset/tools/audit_velocity_legs.py` (replication +
per-trigger measurement grid), `research/event_onset/tools/audit_velocity_analysis.py`
(controls + statistics). Raw numbers: `research/event_onset/reports/audit_velocity_legs.json`.

**Replication is bit-identical.** Trigger timestamps, `dd`, `disp`, `run`, `mae`,
`mfe` reproduce the seven `impulses_D*_T*.parquet` files with max abs error 0.0 on
every row of every cell. Everything below is a re-measurement of the same events,
not a different dataset.

---

## Verdict table

| item | verdict | one-line |
|---|---|---|
| a. entry-point conflation | **FLAW** (confirmed, not exculpatory) | 62% of the reported heat is created by the entry choice; the alternative anchor is 100% hindsight |
| b. follow window | **FLAW on heat / CLEAN on the coin flip** | `P(run>0)` is horizon-invariant; `p50 heat` is *purely* a 300s-hold statistic (5.75pt at 30s → 24.50pt at 600s) |
| c. correlated samples | **PARTIAL** | overlap is severe but VIF≈1, headline unchanged; the "**seven independent parameterisations**" claim is false (83–100% cross-cell overlap) |
| d. signs / conventions | **CLEAN** (with a reporting flaw) | no inversion; but MFE was computed and **dropped from the report**, and MFE ≈ MAE |
| e. selection / survivorship | **FLAW** ×3 | time axis not enforced (T bars ≠ T seconds); 1.4% truncated follow windows; ties counted as losses |
| f. is 49% a null | **FLAW in framing; the study understated its own result** | the null is 49.79%, not 50%; excess CI includes 0; the *significant* number is −1.20pt/trigger net of friction, which is absent from the report |
| g. the owner's claim | **NOT TESTED by the study; tested here** | impulse **timing** is highly predictable (AUC 0.94); **direction** is not (AUC 0.55); "MAE 0" is a 1-in-40 outcome even with perfect hindsight |

**Bottom line.** The study's headline *conclusion* — chasing a completed
displacement has no edge — survives every correction and in fact gets stronger
when friction is applied and when the grid is extended to owner-sized moves. But
almost every *number* used to argue it is either mis-specified, uncontrolled, or
one-sided, and the report violates three mandatory rules in `CLAUDE.md`
(no 95% CI, no significance statement, no PF-based Trade WR, no friction).
The load-bearing sentence — "which places the entire problem on ENTRY SELECTION
BEFORE THE MOVE" — is reached by elimination inside a false dichotomy, not by
measurement. Item g measures it directly for the first time.

---

## a. ENTRY POINT CONFLATION — FLAW (confirmed; the alternative is not tradeable)

The suspicion is correct in mechanism. Entering at `t` measures *"does a completed
displacement continue?"*, not *"what is a leg?"*.

D10/T60, n=24,409, day-clustered 95% CI, 300s hold:

| entry anchor | P(run>0) | mean run | p50 MAE | p50 MFE | real-time? |
|---|---|---|---|---|---|
| `t−T` (leg START) | 72.69% [71.82, 73.58] | **+15.056** [+13.88, +16.44] | 6.50 | 29.50 | **NO** |
| `t−T/2` (mid) | 62.48% [61.81, 63.16] | **+8.367** [+7.65, +9.17] | 9.75 | 24.50 | **NO** |
| `t` (trigger, as published) | 49.22% [48.52, 49.89] | **−0.315** [−0.812, +0.172] | 17.25 | 17.00 | YES |

Two things follow, and they point in opposite directions.

**(1) The heat number is an artifact of the entry choice.** p50 MAE is 6.50pt at
the leg start and 17.25pt at the trigger — **62% of the published 17.75pt of
"enormous heat" is manufactured by entering after the displacement**, not by the
tape. The report's claim that the zigzag study's 2.25pt was hindsight-inflated and
the true figure is "8x larger" compares two different estimands: the zigzag number
is MAE-until-leg-completion for legs that did complete (doubly future-conditioned);
the velocity number is MAE over a fixed unmanaged 5-minute hold. The like-for-like
bridge is the 6.50pt start-anchor figure.

**(2) The alternative anchor is 100% hindsight and buys nothing.** Decomposition
of the start anchor:

```
mean |disp|                              = 15.303 pt   <- already printed at t
mean run from t−T over 300s              = 15.056 pt
=> post-trigger increment                = −0.246 pt   <- everything after t
mean run from t     over 300s            = −0.315 pt
```

The entire start-anchor "edge" **is the displacement itself**. There is no residue.
The trigger is *defined* by that displacement, so entering at `t−T` is conditioning
on the future — the +15pt is arithmetic, not a signal.

**Which questions are answerable in real time.** Only the trigger anchor.
The obvious rescue — "enter earlier in the leg by using a smaller/faster
threshold" — is already refuted inside the study's own grid: the seven cells span
D/T ratios from 0.167 to 0.667 pt/s (D10/T60 … D20/T30, D10/T15) and **every one
returns 49.0–49.6%**. Entering earlier by lowering the bar does not help, because a
lower bar is just an earlier chase.

---

## b. FOLLOW WINDOW — CLEAN on the coin flip, FLAW on the heat

Sweeping the horizon at the trigger anchor (D10/T60), excess of `P(run>0)` over the
direction-free baseline, day-clustered:

| horizon | excess over baseline | p50 MAE | p50 MFE |
|---|---|---|---|
| 30s | **−1.315pp** [−2.127, −0.549] | 5.75 | 5.25 |
| 60s | **−0.887pp** [−1.559, −0.237] | 8.00 | 7.50 |
| 120s | **−0.897pp** [−1.662, −0.191] | 11.00 | 10.75 |
| 300s (published) | −0.528pp [−1.236, +0.136] | 17.25 | 17.00 |
| 600s | −0.176pp [−0.815, +0.434] | 24.50 | 23.75 |

**The 49% is horizon-invariant** — the suspected artifact is not there. The
opposite is true: the effect is *strongest at short horizons* and decays toward the
null as the window grows. 300s is the shortest horizon at which the adverse
deviation stops being statistically detectable. Shortening the window does not
reveal buried continuation; it reveals slightly more **mean reversion**.

**But the heat headline is entirely a 300s-hold statistic.** p50 MAE runs 5.75 →
24.50pt purely as a function of how long you are told to hold. And the
uncontrolled comparison matters more: at a **random RTH time with a random
direction** (n=24,640), 300s p50 MAE is **13.50pt** and 300s p95 is 58.51pt. The
impulse conditioning lifts p50 heat from 13.50 → 17.25pt, a factor of **1.28**.
"Heat is enormous" is a property of holding MNQ unmanaged for five minutes, not a
property of chasing impulses. The report never computed this control.

---

## c. OVERLAPPING / CORRELATED SAMPLES — PARTIAL

The overlap is as bad as suspected, and it does not matter for the headline.

D10/T60: 24,409 triggers → 13,830 distinct moves (gap ≤120s and same direction),
1.76 triggers/move, max 15; 67% of rows sit in a multi-trigger move; **96% of the
300s follow windows overlap the previous trigger's**.

Yet the day-clustered bootstrap says the correlation does not propagate into the
frequency statistic:

| cell | P(run>0), all rows | P(run>0), 1st per move | VIF | effective n |
|---|---|---|---|---|
| D10/T30 | 49.214% [48.564, 49.859] | 49.193% [48.569, 49.789] | 0.91 | 22,909 |
| D10/T60 | 49.228% [48.533, 49.887] | 49.241% [48.553, 49.912] | 1.16 | 20,961 |
| D15/T30 | 49.084% [48.161, 49.971] | 48.868% [47.948, 49.729] | 1.05 | 11,721 |
| D15/T60 | 49.523% [48.720, 50.289] | 49.525% [48.765, 50.289] | 1.03 | 15,613 |
| D20/T60 | 49.582% [48.552, 50.559] | 49.341% [48.353, 50.304] | 1.10 | 9,545 |
| D10/T15 | 48.989% [48.247, 49.715] | 49.143% [48.431, 49.841] | 0.94 | 17,835 |
| D20/T30 | 49.618% [48.453, 50.683] | 49.051% [47.956, 50.137] | 0.93 | 7,727 |

Deduplicating to one trigger per move moves the headline by ≤0.6pp and never
changes a verdict. Variance inflation is ≈1: the sign of a 300s run from two
starts 60s apart is close to independent, because the run is dominated by the last
part of the window. **The `n` is not meaningfully inflated. Not a flaw.**

The mean run is a different story — it *is* CI-fragile (see item f) and the
report's "median run is NEGATIVE in every cell" survives significance testing in
1 of 7 cells.

### The real correlation flaw is between the CELLS, not within them

The report says: *"Seven independent parameterisations, all a coin flip."* They are
not independent. Fraction of each cell's triggers falling within 60s of a trigger
in another cell:

| | D10/T30 | D10/T60 | D15/T30 | D15/T60 | D20/T60 | D10/T15 | D20/T30 |
|---|---|---|---|---|---|---|---|
| **D10/T30** | 100% | 97.9% | 71.2% | 83.5% | 63.5% | 86.2% | 47.2% |
| **D10/T60** | 90.7% | 100% | 65.5% | 79.0% | 58.9% | 79.8% | 42.9% |
| **D15/T30** | 100% | 99.9% | 100% | 97.1% | 84.4% | 98.6% | 68.4% |
| **D15/T60** | 98.4% | 100% | 82.3% | 100% | 76.7% | 91.8% | 57.8% |
| **D20/T60** | 99.6% | 100% | 93.9% | 100% | 100% | 97.4% | 73.2% |
| **D10/T15** | 98.5% | 98.6% | 79.5% | 88.5% | 70.7% | 100% | 54.6% |
| **D20/T30** | 100% | 100% | 100% | 99.7% | 96.2% | 99.8% | 100% |

Every cell is 90–100% nested inside D10/T30 and D10/T60. **D20/T30 is a 100%
subset.** The seven rows are one population re-measured seven times, and the
report treats their agreement as seven-fold corroboration. It is one observation.

---

## d. DIRECTION AND SIGN CONVENTIONS — CLEAN

No inversion of the `stop_width_study.py` class. Checks run on all 7 cells:

- `mae < 0`: **0** rows. `mfe < 0`: **0** rows.
- Invariant `−MAE ≤ run ≤ +MFE`: **0** violations in 108,187 rows.
- `corr(run, mfe)` = +0.629 … +0.693 (positive, correct). `corr(run, mae)` =
  −0.642 … −0.693 (negative, correct). A swap would flip both signs.
- Brute-force re-derivation from the *P&L path* (`(low−entry)·dd` for the adverse
  side, `(high−entry)·dd` for the favourable side) on 500 randomly sampled
  triggers: **max abs error 0.00** for MAE, MFE and run.
- `dd` matches `sign(disp)` on every row; `dd == 0` is unreachable because
  `|disp| ≥ D > 0`.

### Reporting flaw: MFE was computed and dropped

`day_impulses` computes `mfe` and writes it to the parquet; the report table has no
MFE column. It is nearly identical to MAE:

| cell | p50 MAE | p50 MFE | mean MAE | mean MFE | P(MFE>MAE) |
|---|---|---|---|---|---|
| D10/T30 | 19.00 | 18.50 | 26.79 | 26.27 | 49.0% |
| D10/T60 | 17.75 | 17.25 | 25.13 | 24.94 | 49.2% |
| D15/T30 | 22.75 | 22.25 | 31.86 | 31.18 | 49.0% |
| D15/T60 | 20.75 | 20.25 | 28.94 | 28.76 | 49.5% |
| D20/T60 | 23.75 | 23.00 | 33.16 | 32.32 | 49.0% |
| D10/T15 | 20.50 | 20.00 | 29.16 | 28.29 | 48.9% |
| D20/T30 | 26.50 | 26.25 | 37.25 | 36.67 | 49.8% |

"Heat is enormous (p50 17.75pt)" is half of a symmetric pair. The same window
offers 17.25pt of favourable excursion. Reporting only the adverse half makes an
unbiased distribution look punitive.

For completeness, that symmetry is not hiding an ordering edge. First-touch
bracket test at the trigger (D10/T60, n=24,409), TP-first vs SL-first:

| bracket | horizon | TP-first | SL-first | E gross | E net (−0.89) |
|---|---|---|---|---|---|
| ±5 | 300s | 49.3% | 50.7% | −0.070 | −0.960 |
| ±10 | 300s | 48.1% | 48.8% | −0.074 | −0.964 |
| ±15 | 300s | 43.8% | 44.4% | −0.081 | −0.971 |
| ±20 | 300s | 37.4% | 38.2% | −0.193 | −1.083 |
| ±30 | 600s | 35.7% | 36.3% | −0.157 | −1.047 |

Consistent with the existing repo finding that ±N brackets lose at every width.

---

## e. SELECTION AND SURVIVORSHIP — FLAW ×3

### e1. The time axis is not enforced — `c[i] − c[i−T]` is T **bars**, not T **seconds**

The owner's definition is *"a displacement in price in a given displacement in
time."* The code enforces the price leg and not the time leg. ATLAS 1s bars only
print for seconds that traded; the median RTH day has **21,063 of 21,600** seconds
present. Indexing back `T` positions therefore reaches back **more** than `T`
seconds, and — because a longer elapsed window accumulates more displacement — the
`|disp| ≥ D` filter **preferentially admits the gappiest, i.e. thinnest and least
liquid, windows**.

| cell | nominal T | mean realised | p90 | p99 | max | frac > T | frac > 2T |
|---|---|---|---|---|---|---|---|
| D10/T30 | 30s | 32.28s | 32 | 71 | **734s** | 25.3% | 1.5% |
| D10/T60 | 60s | 64.74s | 65 | 141 | **1,346s** | **38.4%** | 1.5% |
| D15/T30 | 30s | 31.91s | 31 | 64 | 726s | 16.8% | 1.1% |
| D15/T60 | 60s | 64.43s | 63 | 136 | 1,346s | 29.6% | 1.3% |
| D20/T60 | 60s | 64.25s | 62 | 127 | 1,346s | 23.1% | 1.1% |
| D10/T15 | 15s | 16.03s | 16 | 33 | 518s | 13.7% | 1.3% |
| D20/T30 | 30s | 31.41s | 31 | 52 | 703s | 11.6% | 0.6% |

38.4% of "60-second impulses" took longer than 60 seconds; the worst is a
**22-minute** window labelled a 60-second impulse. **Magnitude of the damage is
small** — only 1.1–1.5% exceed 2T, and the badly-inflated tail (>141s, n=244) has
mean run +0.41 vs −0.34 for the clean rows. Real bug, small contaminant. Fix is
`searchsorted(ts, ts[i] − T)` with an elapsed-time guard.

### e2. End-of-mask truncation biases `run` toward zero — and is 100% avoidable

`j1 = min(i + FOLLOW_S, n − 1)` clips at the end of the **RTH-masked** array, so
triggers in the last five minutes get a truncated window. Minimum realised follow
window: **1 second**.

| cell | frac truncated | n | p50 MAE trunc / full | mean \|run\| trunc / full |
|---|---|---|---|---|
| D10/T30 | 1.33% | 278 | 11.25 / 19.00 | 16.00 / 26.32 |
| D10/T60 | 1.38% | 337 | 10.75 / 17.75 | 15.03 / 24.80 |
| D20/T30 | 1.08% | 78 | 14.88 / 26.75 | 22.09 / 36.76 |

Exactly the suspected direction. It is a pure artifact: `RTH1 = 15:30 ET` but the
parquet runs to 20:00 ET, so the follow window can simply cross the mask boundary.
Recomputed with true seconds and no clip, D10/T60 trigger/300s: **p50 MAE
17.25 (vs 17.75), mean run −0.315 (vs −0.307)**. Small.

### e3. No future conditioning in the trigger — CLEAN

`disp` uses only `[i−T, i]`. The cooldown is applied on realised timestamps. The
`range(T, n−1)` bound drops exactly one bar per day (the last), duplicates nothing.
138 files → 112 with RTH data, **0 days dropped by the `< 600` filter**. The "112
val days" label is correct (`audit_pipeline.md`: val = 2025_01_02..2025_06_19).

### e4. Ties are silently counted as not-a-win

`P(run == 0)` = 0.26%–0.43% (closes lie on a 0.25 grid). `P(run>0)` is compared
against an implied 50%, but the correct direction-free null is `(1 − P(tie))/2`.
See item f.

### e5. Population concentration

22.6% of D10/T60 triggers fall in the first hour (uniform: 16.7%); per-day counts
range 10 → 355 (mean 217.9), top-10 days contribute 13.9% of all triggers.

---

## f. IS 49% ACTUALLY A NULL? — the study understated its own result

### The null is 49.79%, not 50%

| cell | P(run>0) [day-clustered] | direction-free null | excess | significant? |
|---|---|---|---|---|
| D10/T30 | 49.214% [48.564, 49.859] | 49.802% | −0.588pp [−1.231, +0.057] | **no** |
| D10/T60 | 49.228% [48.533, 49.887] | 49.785% | −0.557pp [−1.235, +0.092] | **no** |
| D15/T30 | 49.084% [48.161, 49.971] | 49.837% | −0.753pp [−1.687, +0.136] | **no** |
| D15/T60 | 49.523% [48.720, 50.289] | 49.825% | −0.303pp [−1.101, +0.449] | **no** |
| D20/T60 | 49.582% [48.552, 50.559] | 49.838% | −0.257pp [−1.270, +0.724] | **no** |
| D10/T15 | 48.989% [48.247, 49.715] | 49.810% | −0.821pp [−1.560, −0.093] | yes (1 of 7) |
| D20/T30 | 49.618% [48.453, 50.683] | 49.868% | −0.250pp [−1.424, +0.811] | **no** |

Six of seven CIs contain zero; the one that does not is expected at 7 tests. The
correct statement is not "49% ≈ a coin flip" but **"indistinguishable from the
direction-free null"** — which is a stronger and cleaner claim.

### There is no fat positive tail — the negative tail is bigger

D10/T60 run distribution: p1 −99.73, p5 −53.50, p25 −17.50, **p50 −0.25**, p75
+16.75, p95 +52.25, p99 +97.00, std 38.62.

| cell | mean win | mean loss | W/L size | profit factor | **PF-based Trade WR** | p99/\|p1\| | skew |
|---|---|---|---|---|---|---|---|
| D10/T30 | 26.06 | −26.51 | 0.983 | 0.960 | **−0.040** | 0.958 | −0.788 |
| D10/T60 | 24.74 | −24.80 | 0.998 | 0.975 | **−0.025** | 0.973 | +0.690 |
| D15/T30 | 30.86 | −31.57 | 0.978 | 0.948 | **−0.052** | 0.950 | +0.120 |
| D15/T60 | 28.60 | −28.71 | 0.996 | 0.984 | **−0.016** | 0.995 | +0.721 |
| D20/T60 | 31.96 | −32.93 | 0.971 | 0.961 | **−0.039** | 0.967 | −0.608 |
| D10/T15 | 28.24 | −28.77 | 0.982 | 0.950 | **−0.050** | 0.966 | −0.402 |
| D20/T30 | 36.16 | −37.24 | 0.971 | 0.962 | **−0.038** | 0.949 | +0.736 |

`p99/|p1| < 1` in **every** cell: the upper tail is *smaller* than the lower tail.
There is no asymmetry to rescue the frequency. Note also that the skew flips sign
(−0.79, +0.69, +0.12, +0.72, −0.61, −0.40, +0.74) across cells that share **>90%
of their triggers** — proof that the third moment here is noise, not structure.

PF-based Trade WR (the repo's canonical metric, absent from the report) is negative
in all seven cells: gross profit is below gross loss **before friction**.

### The significant number is expectancy net of friction — and it is not in the report

Using the sibling study's `FRICTION = 0.89` pt round-trip:

| cell | mean run (gross) | **mean run NET** | mean FADE net |
|---|---|---|---|
| D10/T30 | −0.534 [−1.283, +0.110] | **−1.424 [−2.173, −0.780]** | −0.356 [−1.000, +0.393] |
| D10/T60 | −0.307 [−0.810, +0.191] | **−1.197 [−1.700, −0.699]** | −0.583 [−1.081, −0.080] |
| D15/T30 | −0.825 [−1.872, +0.068] | **−1.715 [−2.762, −0.822]** | −0.065 [−0.958, +0.982] |
| D15/T60 | −0.228 [−0.915, +0.490] | **−1.118 [−1.805, −0.400]** | −0.662 [−1.380, +0.025] |
| D20/T60 | −0.651 [−1.456, +0.126] | **−1.541 [−2.346, −0.764]** | −0.239 [−1.016, +0.566] |
| D10/T15 | −0.731 [−1.383, −0.099] | **−1.621 [−2.273, −0.989]** | −0.159 [−0.791, +0.493] |
| D20/T30 | −0.717 [−1.663, +0.186] | **−1.607 [−2.553, −0.704]** | −0.173 [−1.076, +0.773] |

Gross mean run is *not* significant in 6 of 7 cells (contradicting the report's
"median run is NEGATIVE in every cell" as if that were a finding). **Net of
friction it is significant in all seven** — −1.12 to −1.72 pt per trigger, every CI
excluding zero. Fading is also negative everywhere. Neither side of the chase is
tradeable, and the report's own conclusion is much better supported by the number
it did not compute.

Per-trigger Sharpe: −0.005 to −0.017.

### The grid never tests a move of the size the owner trades — extended here

`velocity_legs.py` stops at D=20. The 2024_09_16 move it invokes was 168pt.
Extending (trigger anchor, day-count 100–112):

| cell | n | /day | p50 \|disp\| | horizon | excess over null | mean run | **net (−0.89)** |
|---|---|---|---|---|---|---|---|
| D30/T60 | 4,660 | 43.6 | 32.00 | 300s | −0.31pp | −1.23 | −2.12 |
| D40/T60 | 2,229 | 21.9 | 42.25 | 300s | −0.87pp | −2.46 | −3.35 |
| D60/T60 | 661 | 10.5 | 63.00 | 300s | −1.59pp | −4.43 | **−5.32** |
| D30/T30 | 2,742 | 25.9 | 32.00 | 300s | −0.62pp | −1.30 | −2.19 |
| D50/T120 | 2,313 | 23.4 | 53.00 | 300s | +0.71pp | −0.36 | −1.25 |
| D80/T300 | 1,645 | 19.4 | 84.75 | 300s | −2.40pp | −3.26 | −4.15 |

The conclusion holds and **gets worse with size**: bigger completed impulses
mean-revert more (−4.43pt at D60/T60). The study should not have generalised from
a grid that tops out at 20pt, but the generalisation happens to be right.

### Is the D10 population an "impulse" at all?

At a **random** RTH 60s window, `P(|disp| ≥ 10)` = **32.1%**; at 30s, `P(≥10)` =
19.6%. The D10 cells therefore fire on roughly a third of the tape — 217.9
triggers/day is **61% saturation** of the 360 slots the 60s cooldown permits.
Within-window directionality `|disp|/range` is 0.682 for D10/T60 vs **0.455** for
a random 60s window — a mild lift that is largely mechanical (`range ≥ |disp|` by
construction). D20/T30 is defensible (`P(≥20)` at a random 30s window = 4.7%);
D10/T60 is not an event.

---

## g. THE OWNER'S ACTUAL CLAIM — the study never tested it

The report's closing move ("which places the entire problem, again and finally, on
ENTRY SELECTION BEFORE THE MOVE") is **elimination inside a false dichotomy**. The
study measured one option (chase) and assigned the edge to the unmeasured
alternative. It also cites a single 2024_09_16 trade — a date **outside its own
data range** — as the supporting evidence for a claim its 24,409 samples say
nothing about.

Measured here for the first time. For every trigger, features computed **strictly
from bars at or before `t−T`**, versus time-of-day-matched non-impulse controls
(no trigger within ±180s). Day-clustered CIs.

**The trustworthy cell is D40/T60** (2,229 impulses vs 1,920 controls; only 18.9%
of RTH sits inside an exclusion zone). At D10/T60, **94.3% of RTH is inside an
impulse exclusion zone** — a 10pt/60s "impulse" covers nearly the whole session, so
its control set is the quietest 5.7% of the tape and its AUCs are partly circular.
Reported for contrast, not for inference.

### Impulses ARE preceded by an identifiable state — volatility, not compression

| pre-state feature (at `t−T`) | AUC, D40/T60 | Cohen d | impulse median | control median |
|---|---|---|---|---|
| realised vol, 60s | **0.937 [0.907, 0.956]** | +1.365 | 3.16 | 1.30 |
| realised vol, 300s | **0.936 [0.906, 0.955]** | +1.418 | 3.14 | 1.34 |
| range, 300s | **0.915 [0.882, 0.939]** | +1.082 | 84.25 | 32.75 |
| range, 60s | **0.912 [0.878, 0.939]** | +1.173 | 39.25 | 14.50 |
| range, 900s | **0.895 [0.854, 0.926]** | +0.993 | 141.50 | 58.25 |
| volume, 60s | **0.800 [0.767, 0.833]** | +1.031 | 1,817 | 939 |
| volume, 300s | **0.800 [0.762, 0.835]** | +1.073 | 8,909 | 4,810 |
| \|ret\|, 300s | **0.766 [0.727, 0.797]** | +0.732 | 36.25 | 12.62 |
| position in 900s range | 0.420 [0.387, 0.449] | −0.280 | 0.46 | 0.59 |
| **range COMPRESSION (60s/300s)** | 0.519 [0.503, 0.536] | +0.100 | 0.46 | 0.45 |
| **time of day** | 0.445 [0.377, 0.501] (n.s.) | −0.133 | 99 min | 125 min |

*When* an impulse fires is highly predictable — but the mechanism is plain
volatility clustering, not a special setup. Two negatives worth keeping:
**range compression does not precede impulses** (AUC 0.52 / 0.49 — the "coiling"
hypothesis is dead), and **time of day does not separate** once you control for it.

### Direction is NOT predictable — which is the whole problem

Same features, now predicting whether the coming impulse is **up** or **down**:

| feature | AUC, D40/T60 | AUC, D10/T60 |
|---|---|---|
| position in 900s range | **0.415 [0.386, 0.449]** (≡0.585 flipped) | 0.449 [0.440, 0.458] (≡0.551) |
| range, 900s | 0.544 [0.523, 0.563] | ~0.50 |
| range, 300s | 0.539 [0.515, 0.564] | ~0.51 |
| realised vol, 300s | 0.538 [0.516, 0.565] | ~0.50 |
| everything else | 0.50–0.53 | 0.49–0.51 |

The best single pre-impulse directional signal is **0.585 AUC** (low in the
trailing 15-minute range → up impulse; i.e. mean reversion). Consistent with the
existing repo finding that direction sits at ~0.57 AUC and refuses to move.

**The combination is the trap.** The state that predicts an impulse is high
volatility — which is precisely the state in which the 300s heat is maximal.
Anticipating the *timing* delivers you into maximum heat with a 0.55-AUC coin on
*direction*. "Get positioned before the move" is not a free lunch; it is the same
unbiased coin, entered earlier, with more variance.

### "MAE 0" is a 1-in-40 outcome, not a property of leg starts

Testing the anecdote directly — entry at the exact leg start, **perfect hindsight**:

| population | hold | p50 MAE | P(MAE ≤ 0.25) | P(MAE ≤ 1) | P(MAE ≤ 2) | mean run |
|---|---|---|---|---|---|---|
| D10/T60 | 300s | 6.50 | **4.5%** | 11.6% | 21.2% | +15.06 |
| D20/T30 | 300s | 7.00 | **5.6%** | 13.8% | 24.2% | — |
| D40/T60 | 300s | 8.00 | **3.8%** | 10.3% | 19.3% | +45.22 |
| D60/T60 | 300s | 12.25 | **4.4%** | 7.6% | 15.1% | +66.09 |
| D80/T300 | 300s | 10.50 | **2.5%** | 7.1% | 13.4% | +101.65 |

Even with a time machine, a leg-start entry has median heat of 6.5–12.25pt, and
zero heat occurs in 2.5–5.6% of cases. The 2024_09_16 trade is a top-few-percent
draw, not a reproducible signature. Presenting it beside 24,409-sample statistics
as the thing the statistics point toward is the weakest inference in the report.

---

## Corrections to apply if the study is rerun

1. Enforce the time axis: `searchsorted(ts, ts[i] − T)` with an elapsed guard, not
   `i − T`.
2. Let the follow window cross `RTH1` (data runs to 20:00 ET); never clip to the
   masked array end.
3. Report the direction-free baseline `(1 − P(tie))/2`, not an implied 50%.
4. Report the random-time, random-direction control for MAE/MFE. Without it, heat
   numbers are uninterpretable.
5. Report MFE alongside MAE. They are equal to within 0.5pt.
6. Report the day-clustered 95% CI, a significance statement, PF-based Trade WR,
   and expectancy net of the 0.89pt friction — all four are mandatory in
   `CLAUDE.md` and all four are missing.
7. Drop "seven independent parameterisations" — the cells are 83–100% nested.
8. Either extend the grid past D=20 or stop generalising to 168pt moves.
9. Separate what was measured (chasing has no edge) from what was inferred
   (anticipation is where the edge lives). The second now has its own numbers:
   timing AUC 0.94, direction AUC 0.585.
