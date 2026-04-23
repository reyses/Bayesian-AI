# Cycle 03 — Signal capture only (no exit mechanics)

> PDCA cycle inside the RM Pivot research project.
> Ref: [project.md](project.md) · follows Cycle 2 ADJUST

---

## PLAN

### Hypothesis

When an RM zigzag pivot fires and we call direction per Cycle 1's standardized rule (LOW → LONG, HIGH → SHORT), price subsequently moves in the called direction **more often than random** at useful trading horizons. In plain English: when the system says "short," price is genuinely heading down from that point; when it says "long," price is heading up.

### Why this cycle exists

Cycle 2 failed because the exit cost 2R in retracement tax. That failure is about EXIT, not about whether we entered in the right direction. This cycle isolates the **entry direction quality** from any exit mechanics, so we know where the signal stops working:

- If direction is right → exit rule is the whole problem → Cycle 4 fixes exit
- If direction is random → premise is dead → abandon RM pivots as entry trigger

### Change

None to production code. Read-only measurement. New script:
`tools/measure_rm_pivot_entry_direction.py`.

### Method

Per day (IS 2025 + OOS 2026):
1. Compute RM (60-bar rolling OLS on 1m closes) and zigzag at R=$4 (the Cycle 1 sweet spot)
2. At each confirmed RM pivot bar:
   - Record direction called (LOW → LONG, HIGH → SHORT)
   - Entry price = 1m close at confirmation bar
3. For each pivot, look **forward** at horizons H ∈ {5, 10, 15, 30, 60, 120, 240} minutes:
   - Price at entry_ts + H
   - Move = (price[entry_ts+H] − entry_price) × direction_sign
   - Move > 0 = price went our way. Move < 0 = against.
4. Aggregate per horizon:
   - % of entries with move > 0 (direction hit rate)
   - Mean move ($), median move ($)
   - Mean absolute move ($, to characterize volatility)
5. Also compare IS vs OOS and split by pivot type (LOW vs HIGH) to check for asymmetry

### Predicted outcome

| Horizon (min) | Direction hit-rate | Mean signed move |
|---:|---:|---:|
| 5 | 55–65% | +$1 to +$5 |
| 15 | 58–70% | +$5 to +$15 |
| 30 | 60–72% | +$8 to +$25 |
| 60 | 58–68% | +$10 to +$30 |
| 120 | 52–62% (decay) | +$5 to +$25 |
| 240 | 50% ± 5% (random) | near zero |

At least ONE horizon should have hit-rate > 55% and positive mean move in IS AND OOS.

### Success gate (any ONE of the below passes)

- **Direction**: hit-rate ≥ 55% at ≥ 2 horizons in IS **and** OOS
- **Magnitude**: mean signed move ≥ +$5 at ≥ 1 horizon in IS **and** OOS

### Kill gate

- Hit-rate ≤ 50% at ALL horizons in either IS or OOS → signal random
- All horizons show mean signed move ≤ $0 → signal anti-predictive
- IS ≥ 55% but OOS ≤ 50% at same horizon → overfit / regime-dependent

### Output

- `research/rm_pivot/findings/2026-04-22_signal_capture.md`
- `research/rm_pivot/findings/2026-04-22_signal_capture.png`

### Reproduction command

```
python tools/measure_rm_pivot_entry_direction.py
```

---

## DO

Ran `python tools/measure_rm_pivot_entry_direction.py` 2026-04-22.
- 5,856 IS pivots × 12 horizons. 1,672 OOS pivots × 12 horizons.
- Output: `research/rm_pivot/findings/2026-04-22_signal_capture.{md,png}`

---

## CHECK

### Portfolio dashboard

| Q | Sub-question | IS | OOS | Gate | IS | OOS |
|---|---|---:|---:|---|:---:|:---:|
| **Q1** | Direction right at future H? | **49.7%** | **50.3%** | ≥55% at 2+ H | ✗ | ✗ |
| **Q2** | Real turning point (pre-against)? | 90.6% | 91.4% | ≥55% at 2+ H | ✓ | ✓ |
| **Q3** | Mean $ captured per trade? | +$0.19 | −$0.06 | ≥$5 at 1+ H | ✗ | ✗ |
| **Q4** | Oracle-exit ceiling | +$214.69 | +$215.21 | ≥$20 | ✓ | ✓ |
| **Q5** | Daily stacks positive? | 49% DayWR | 57% DayWR | ≥60% | ✗ | ✗ |

**IS: 2/5 gates pass. OOS: 2/5 gates pass.**

### Shape of the signal (crucial)

- **Backward horizons (H=-5 to -30 min)**: pre-trend against our direction is 87-91%, mean signed $-40. That is, AT the pivot we are $40 "deep" into a move against our called direction.
- **Forward horizons (H=+5 to +240 min)**: hit-rate is flat at 48-50% (coin flip), mean signed move ≈ $0 at every horizon.

The pivot IS a real turning point — but the turning point's **future direction is random.**

### What this means (reconciling with Cycle 1)

**Cycle 1's 83.5% accuracy was a tautology, not a tradeable signal.**

Cycle 1 measured: "residual sign at pivot matches pivot type." That's true because pivot type is DEFINED by the direction change; the residual naturally correlates with the already-happened direction change.

Cycle 3 tested what trading needs: "given an RM pivot, what's price going to do NEXT?" Answer: **50/50 coin flip, same as any random bar.**

The 2026-04-21 journal's "Cohen d=−2.46 walk-forward, 86% oracle accuracy" claim is technically correct as a statistical effect size but **does not translate to predictive edge on future direction.** That finding was a restatement of pivot definitions, not a forecast.

### Oracle gap

- Oracle ceiling at H=+240: **$215/trade** (if we could pick the best moment)
- Actual capture at H=+240: −$0.85 (random exit timing)
- Direction guess is 50/50

So $215/trade of movement sits there — but without direction edge, we can't know which way to trade it.

---

## ACT

**Decision: ADJUST — the specific hypothesis fails, but Q2 + Q4 suggest a different hypothesis could work.**

### What's dead

- **RM pivot + residual-sign direction rule as an ENTRY SIGNAL.** Direction prediction at future horizons is 50/50. No matter what R or exit we pick, we cannot beat random direction.

### Cross-source / cross-R confirmation (per user suggestion)

Also tested **price zigzag at R=$40** (matches `charts/cord_length_2025_06_09_R30.png` framing):

| | IS | OOS |
|---|---:|---:|
| Q1 direction HR | **49.9%** | **49.9%** |
| Q2 turning point % | 78% | 80% |
| Q3 mean $/trade | +$0.09 | −$0.87 |
| Q4 oracle $/trade | +$292 | +$248 |
| Q5 Day WR | 47% | 45% |

11,244 IS pivots at price R=$40 — and direction is **still 49.9%**. Sample size is not the problem; the signal simply has no direction edge.

**Conclusion**: the pivot-direction failure is UNIVERSAL across (RM vs price) × (R = $4 to $40). Pivots mark where reversals have HAPPENED; they do not forecast where they GO. Report: `findings/2026-04-22_signal_capture_priceR40.md`.

### What survived

- **Q2**: RM pivots correctly localize where a reversal occurred (90%+ pre-trend against).
- **Q4**: $215/trade of movement exists post-pivot if we could pick direction correctly.

### What this opens

The RM pivot answers "WHERE did something happen" but not "WHICH WAY will it go." So the entry signal is incomplete — it's missing a direction predictor. Options for Cycle 4:

**4a. RM pivot + OTHER direction predictor**: use the pivot as location trigger, then a different feature (wick, volume, higher-TF z, slope) to predict direction. Needs a separate direction-quality measurement per feature.

**4b. Flip the role — pivot as EXIT, not entry**: since pivots mark where trends exhaust, enter DURING a trend (before the pivot), exit at pivot confirmation. Re-frame the whole approach.

**4c. Abandon RM pivots entirely**: accept that this premise is dead, return to `research/INDEX.md` and pick a different research topic or new premise.

### Claude's read

Q2 is genuinely strong. Q4 ceiling is genuinely large. The structural information IS there. But without direction, the signal is "something happened here" which isn't tradeable alone. Honest answer to user: **no, we cannot trade reliably enough with RM-pivot-as-direction-signal.** The portfolio says 2/5 across both IS and OOS.

Recommend Cycle 4 = **4a** (direction predictor search), because:
- Q2/Q4 say the location signal is real — don't throw it away
- Direction predictor search is a contained experiment (Cohen d on candidate features at pivots, forward-return target)
- If no feature has direction edge at pivots, **then** abandon to 4c

But this is Claude's vote. User decides before any more code.

### MEMORY update needed

Add to `docs/memory/MEMORY.md`:
- **Cohen d at pivots measures structural correlation, NOT predictive direction.** Distinguish "did the pivot correctly mark a reversal" (structural) from "will price go up/down next" (predictive). The Cycle 1 finding was the former.
