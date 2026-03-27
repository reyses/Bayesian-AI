# PROPOSAL: Multi-TF Layered CNN — Oscillation-Aware Trading

**Status:** Proposal — exploratory phase first, trading phase earned by findings
**Depends on:** 29D feature pipeline (Phase A complete), ATLAS multi-TF data
**Principle:** Measure first, model second. No CNN trains until we understand what it's learning.

---

## The Problem

The current approach (L1 direction + L2 duration + L3 retreat) treats the market as a
single-resolution signal. It predicts "price goes up" and holds for N bars. But 1m price
action is not a single signal — it's an oscillation riding a trend riding a structure.

A 30-minute hold crosses ~4 oscillation cycles. The model can't tell the difference between
"normal dip in an uptrend" and "the trend just reversed." It enters at random oscillation
phases and exits on fixed rules. Result: 21,000 SL hits.

## The Insight

The market has layered structure visible across timeframes:

```
1h:   ────────────/───────────\──────────────   (structural trend)
15m:  ───/──\──/──────\──/──\────/──\──/─────   (session swings)
5m:   /\/\/\──\/\/\──/\/──\/\/\──/\/\/\──\/─    (swing oscillation)
1m:   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     (execution noise + signal)
1s:   ·····································     (tick-level timing pulse)
```

Each TF answers a different question. The 29D features already capture all of them.
The missing piece is labels that reflect this structure — not "predict 7D state at t+10"
but "what regime are we in, where in the oscillation, should we enter/exit?"

---

## Phase 1: EXPLORATORY (research, no trading)

Measure the market structure from ATLAS data. Each module produces findings that
inform the CNN labels in Phase 2. All outputs saved to `reports/findings/`.

### Module E1: Trend Characterization

**Question:** What does a trend look like at each TF? How long do they last?

**Method:**
- Identify directional segments at 1h, 15m, 5m using regime detection
  (e.g., rolling regression slope sign changes, DMI crossovers, z_se sign persistence)
- For each segment: duration (bars), amplitude (ticks), direction, volume profile
- Distribution of trend durations per TF
- Cross-TF alignment: when 1h and 15m agree, how long does the trend persist?

**Output:**
- Trend duration distributions per TF (histograms + percentiles)
- Amplitude vs duration scatter (is there a minimum move size?)
- Cross-TF agreement persistence (when all TFs align, median trend duration)
- Regime transition signatures (what does a trend reversal look like in the features?)

**Key question answered:** How long can we expect a trend to last? This sets the
upper bound on hold duration — holding past the median trend duration is gambling.

### Module E2: Oscillation Characterization

**Question:** What is the natural oscillation period and amplitude at 1m within a trend?

**Method:**
- Within each trend segment (from E1), analyze 1m price action
- Identify oscillation cycles: local min/max detection (zero-crossings of detrended price)
- Measure: period (bars between troughs), amplitude (ticks peak-to-trough),
  symmetry (rise time vs fall time)
- How does oscillation change with trend strength? With volatility regime?
- Does the ~8 minute natural oscillation hold across months/regimes?

**Output:**
- Oscillation period distribution (expected: peaked around 8 bars/minutes)
- Amplitude distribution (how many ticks is a typical oscillation?)
- Period vs trend strength (do oscillations speed up in strong trends?)
- Phase alignment: do oscillation troughs in 1m align with 5m features?

**Key question answered:** What is the natural rhythm of 1m price? This defines
the entry/exit cycle — enter at trough, exit at peak, period = expected hold.

### Module E3: Cross-TF Oscillation Nesting

**Question:** How do oscillations at different TFs nest inside each other?

**Method:**
- Map oscillation cycles at 1m, 5m, 15m simultaneously
- When a 5m oscillation troughs, what's happening at 1m? (is 1m also at a trough?)
- When 1m oscillation troughs ALIGN with 5m trough = high-conviction entry?
- When 1m oscillation peaks while 5m still rising = just a wobble, hold?

**Output:**
- Cross-TF phase alignment frequency (how often do 1m and 5m troughs coincide?)
- PnL of aligned vs non-aligned entries (is alignment actually predictive?)
- The "nested oscillation" signature in the 29D features

**Key question answered:** Can we use higher-TF oscillation phase to filter 1m entries?

### Module E4: Optimal Entry/Exit Points

**Question:** Given trend + oscillation, where are the best entries and exits?

**Method:**
- At every oscillation trough (from E2), compute the forward PnL if we entered
- At every oscillation peak, compute the PnL if we exited
- Compare: entry at trough vs random entry (how much edge does timing give?)
- Factor in 2s slippage from 1s data (real fill prices)
- What 29D features distinguish "good troughs" from "fake troughs"?

**Output:**
- Edge of oscillation-timed entry vs random (ticks/trade)
- Feature importance: which of the 29D features predict "real trough"?
- Slippage-adjusted PnL distribution for trough entries
- Exit timing: peak exit vs fixed hold vs trend-end exit comparison

**Key question answered:** How much is oscillation timing worth? If the edge is
small, the whole approach is wrong. If it's large, we know exactly what the CNN
needs to learn.

### Module E5: 1s Timing Precision

**Question:** Does 1s data improve entry/exit timing within the 1m oscillation?

**Method:**
- At each identified 1m trough, look at 1s features in the 30s before and after
- Is there a 1s velocity/volume signature that confirms "the trough is happening NOW"?
- Compare: enter at 1m bar close vs enter when 1s confirms trough
- Slippage: how much better are 1s-timed fills vs 1m-bar fills?

**Output:**
- 1s confirmation signature at oscillation troughs (feature pattern)
- Fill improvement: 1s-timed vs 1m-bar entry (ticks saved)
- False trough rate: 1s features that distinguish real vs fake troughs

**Key question answered:** Is 1s worth the complexity, or is 1m resolution enough?

---

## Phase 2: CNN LAYERS (earned by Phase 1 findings)

Only proceed if Phase 1 shows:
- [ ] Trends have measurable, predictable duration at higher TFs
- [ ] 1m oscillation has a consistent period (~8 bars)
- [ ] Cross-TF alignment provides entry edge
- [ ] Oscillation-timed entry beats random entry by > 4 ticks/trade

### L1 — Trend Regime CNN

**Input:** 10 × 29D features (same lookback window)
**Output:** trend_direction (+1/-1), trend_strength (0-1), trend_bars_remaining
**Labels:** From E1 — every bar tagged with its trend segment properties
**Architecture:** StatePredictor backbone, 3-output head
**Primary TF signal:** 1h dmi_diff, 15m velocity, 15m z_se

### L2 — Oscillation Phase CNN

**Input:** 10 × 29D features + L1 output (trend context)
**Output:** oscillation_phase (continuous 0→1, trough=0, peak=0.5)
**Labels:** From E2 — every bar tagged with its phase in the oscillation cycle
**Architecture:** Small MLP on backbone latent + L1 features
**Primary TF signal:** 1m z_se, 1m velocity, 5m z_se

### L3 — Entry Gate CNN

**Input:** L1 output + L2 output + current 29D features
**Output:** P(enter_now)
**Labels:** From E4 — bars at oscillation troughs aligned with trend = 1, else = 0
**Architecture:** Small MLP (~800 params)
**Fires when:** oscillation at trough (L2) + trend confirms direction (L1)

### L4 — Exit Gate CNN

**Input:** L1 output + L2 output + trade context (side, PnL, bars held)
**Output:** P(exit_now)
**Labels:** From E4 — bars at oscillation peaks or trend reversals = 1, else = 0
**Architecture:** Small MLP (~800 params), runs every bar while in trade
**Fires when:** oscillation at peak (L2) + trend weakening (L1), OR trend reversed (L1)

### Guardrails (unconditional, not learned)

- Circuit breaker: 200t absolute max loss (should never fire if L4 works)
- Maintenance flatten
- Daily loss limit
- 2s slippage on every fill (from 1s data)

---

## Phase 2 Simulation

```
Each bar:
  L1: What regime? → trending UP, strong, 45 bars remaining
  L2: Oscillation phase? → 0.15 (near trough, ascending)

  If not in trade:
    L3: Enter now? → P=0.82 (trough + trend aligned) → ENTER LONG

  If in trade:
    L4: Exit now? → P=0.12 (hold — oscillation still rising)
    ...
    L4: Exit now? → P=0.78 (oscillation peaked + trend weakening) → EXIT
```

Expected behavior:
- Enter at oscillation troughs (~every 8 bars during a trend)
- Hold through the half-cycle (4 bars rise)
- Exit at oscillation peak
- Skip entries when trend is weak or reversing
- 5-10 trades/day, each riding one oscillation half-cycle

---

## Build Order

### Phase 1 (1-2 sessions, research only)
1. Build `tools/mtf_oscillation_research.py` with modules E1-E5
2. Run on ATLAS IS data, save findings to `reports/findings/`
3. Review findings — do the oscillation measurements support this architecture?
4. Gate: does oscillation-timed entry beat random by > 4 ticks?

### Phase 2 (2-3 sessions, conditional on Phase 1)
1. Build label pipelines for L1-L4 from Phase 1 measurements
2. Train L1 (trend regime) → validate on IS
3. Train L2 (oscillation phase) → validate on IS
4. Train L3+L4 (entry/exit gates) → validate on IS
5. Full simulation on IS → then OOS

### Phase 3 (1 session, conditional on Phase 2)
1. OOS validation
2. Live integration planning

---

## What NOT to Build

- No fixed SL/TP/trail — exits are learned by L4
- No duration commitment — hold until L4 says exit
- No hand-tuned confidence thresholds — L3 gate is learned
- No separate TF workers — the 29D features encode all TFs in one vector
- No Phase 2 without Phase 1 validation — measure before model

---

## Baselines to Beat

```
DMI flipper (proven):           $208/day (simple, robust)
TradeCNN 13D (baseline):       $1,609/day (direction + fixed exits)
TradeCNN 29D + L2 (current):   $602/day (duration, too many SL hits)
Target:                         > $1,609/day with fewer trades and learned exits
```

---

## Files

| File | Purpose | Phase |
|------|---------|-------|
| `tools/mtf_oscillation_research.py` | E1-E5 research modules | 1 |
| `reports/findings/trend_characterization.md` | E1 findings | 1 |
| `reports/findings/oscillation_characterization.md` | E2 findings | 1 |
| `reports/findings/cross_tf_nesting.md` | E3 findings | 1 |
| `reports/findings/optimal_entry_exit.md` | E4 findings | 1 |
| `training/build_oscillation_labels.py` | Label pipeline for L1-L4 | 2 |
| `core/trend_cnn.py` | L1 + L2 models | 2 |
| `core/trade_gates.py` | L3 + L4 models | 2 |
| `training/train_layered_cnn.py` | Training pipeline for all layers | 2 |
