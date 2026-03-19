# Exit Engine — C&E Matrix + PFMEA
> Generated 2026-03-14 from --lookback run (IS: 8,017 trades, OOS: 1,984 trades)

---

## 1. CAUSE & EFFECT MATRIX

**Y (Output)** = Trade PnL ($/trade)
**Scoring**: Correlation strength 1-10 (10 = strongest effect on Y)

### 1a. Exit Reason → PnL Impact (Empirical)

| # | Exit Module | IS Trades | IS WR | IS $/trade | OOS Trades | OOS WR | OOS $/trade | IS→OOS Decay |
|---|-------------|-----------|-------|------------|------------|--------|-------------|--------------|
| 1 | survival_stop | 781 (9.7%) | 98.2% | **$35.71** | 114 (5.7%) | 98.2% | **$20.77** | -42% |
| 2 | envelope_decay | 10 (0.1%) | 100% | **$30.65** | 2 (0.1%) | 100% | **$50.75** | +66%* |
| 3 | death_hook | 57 (0.7%) | 87.7% | $4.05 | 27 (1.4%) | 85.2% | **$18.70** | +362%* |
| 4 | take_profit | 13 (0.2%) | 100% | $10.00 | 1 (0.1%) | 100% | $10.00 | 0% |
| 5 | regime_decay | 3,562 (44.4%) | 81.5% | $6.51 | 683 (34.4%) | 86.4% | $0.80 | **-88%** |
| 6 | trail_stop/belief_flip | 405 (5.1%) | 87.9% | $6.32 | 30 (1.5%) | 83.3% | $5.48 | -13% |
| 7 | peak_giveback | 42 (0.5%) | 100% | $2.90 | 27 (1.4%) | 77.8% | **-$11.57** | **FLIP** |
| 8 | stop_loss | 3,140 (39.2%) | 89.1% | $0.39 | 1,095 (55.1%) | 100% | $0.50 | +28% |
| 9 | band_urgent | 6 (0.1%) | 83.3% | **-$22.25** | 3 (0.2%) | 0% | **-$22.50** | -1% |
| 10 | maintenance_flat | 1 (0.0%) | 0% | -$11.00 | 2 (0.1%) | 50% | -$7.25 | — |

*Small sample — interpret with caution

### 1b. Input Variables (X) → Exit Quality

| X (Input Variable) | Corr to Y | Affects Exits | Mechanism |
|---------------------|-----------|---------------|-----------|
| **Hold duration (bars_held)** | **10** | ALL | Longer holds = higher $/trade (2-5m = $25.56 vs <30s = $3.61). Most exits are time-gated. |
| **Template SL sizing (sl_ticks)** | **9** | stop_loss, breakeven | SL floor at 4 ticks = $1 protection. Trades stopped out in noise. 39% of IS volume. |
| **MFE achieved (peak_favorable)** | **9** | giveback, breakeven, envelope | No MFE = no giveback activation. Breakeven locks at 4t. Envelope anchors to T0. |
| **ADX strength (macro)** | **8** | regime_decay, envelope, tidal_wave | ADX < 20 = regime_decay fires. ADX slope modulates envelope halflife +-50%. |
| **Hurst exponent (macro)** | **8** | regime_decay | Hurst < 0.50 = regime shift. Primary academic trigger for trend collapse. |
| **DI crossover (macro)** | **8** | regime_decay, belief_flip | DI cross against position = trend reversal confirmation. Two exits use this. |
| **Band position (z-score)** | **7** | band_urgent, death_hook, envelope | Extreme bands (+-2sigma) trigger death_hook. Band exhaustion modulates envelope. |
| **Worker consensus (conviction)** | **7** | belief_flip, watchdog | Conviction flip → belief_flip. Workers against ≥ 5 → watchdog. |
| **Noise ticks (volatility)** | **6** | tidal_wave, giveback, envelope | 4x noise expansion = tidal_wave. Noise floor for envelope. |
| **Brain ePnL** | **6** | survival_stop | Bayesian expected PnL ≤ 0 = edge gone. Requires 3+ observations. |
| **Z-score variance** | **5** | survival_stop | z_var < 0.20 = price flatlined. Structural death signal. |
| **Wave maturity** | **4** | belief_flip (via exit_signal) | Mature wave → tighten_trail signal → indirect exit pressure. |
| **Shape params** | **3** | giveback, envelope | Override giveback threshold + envelope halflife. Only with --shapes/--primitives. |
| **30m slow flip** | **3** | giveback | Tightens giveback by 15pp when 30m worker flips. Sticky per trade. |

### 1c. Critical X's (Pareto)

The **top 3 inputs** that explain ~80% of exit quality variance:

1. **Hold duration** — The single strongest predictor. Short holds (<30s) are coin flips (60% WR IS, 48% OOS). Medium holds (2-5m) are $25.56/trade. The min-hold experiment directly attacks this.

2. **SL sizing** — 39% of IS trades exit via stop_loss at $0.39/trade. These are barely profitable noise captures. A 4-tick SL on a 5-minute hold = guaranteed stop-out. The 40-tick floor in the experiment addresses this.

3. **ADX/Hurst regime state** — regime_decay is 44% of trades. When it fires correctly ($6.51 IS), it's the workhorse. When it fires prematurely, it cuts profitable trades short (explains the 81.5% WR vs 98% for survival_stop).

---

## 2. PROCESS FAILURE MODE & EFFECTS ANALYSIS (PFMEA)

**Scoring Scale:**
- **Severity (S)**: 1-10. Impact on trade PnL if failure occurs.
- **Occurrence (O)**: 1-10. How often this failure mode happens.
- **Detection (D)**: 1-10. How hard it is to detect before damage is done. (10 = undetectable)
- **RPN** = S x O x D. Higher = more critical.

| # | Exit Module | Failure Mode | Effect on Trade | S | O | D | **RPN** | Current Control | Recommended Action |
|---|-------------|-------------|-----------------|---|---|---|---------|-----------------|-------------------|
| 1 | **stop_loss** | SL too tight (4-tick floor) | Stopped out in noise before trade develops. 39% of IS volume at $0.39/trade — barely breakeven. | 8 | 9 | 3 | **216** | sl_min_ticks=4 config | **DONE**: min-hold raises floor to 40 ticks. Consider raising global sl_min_ticks to 8-12. |
| 2 | **stop_loss** | SL too wide (40-tick floor in experiment) | Large single-trade losses ($10) when reversal guardrails fail. Could erode gains from 10+ winning trades. | 7 | 3 | 5 | **105** | regime_decay + belief_flip as early exits | Monitor max single-trade loss in experiment. Add trailing SL once MFE > 20t. |
| 3 | **regime_decay** | Fires prematurely (ADX oscillates near 20) | Cuts profitable trade short. 81.5% WR (lowest of non-noise exits) suggests 18.5% premature fires. | 6 | 7 | 6 | **252** | ADX threshold = 20, requires crossover (prev >= threshold) | Consider ADX smoothing (EMA of ADX) or confirmation bar (2 bars below 20). |
| 4 | **regime_decay** | Fires too late (ADX drops slowly) | Trade bleeds while waiting for threshold cross. Already in drawdown by the time ADX < 20. | 5 | 4 | 7 | **140** | Hurst regime shift as faster signal | Hurst + ADX dual confirmation could catch faster. Validate Hurst fires before ADX in data. |
| 5 | **regime_decay** | DI cross false signal (choppy DI) | Exits on noise DI crossover in ranging market. Especially bad in low-ADX environments. | 5 | 5 | 6 | **150** | Requires bars_held >= 3 | Add ADX minimum for DI cross (only trust DI cross when ADX > 15). |
| 6 | **survival_stop** | Flatline detection too slow (min_bars=10) | Holds dead trade for 10 bars (2.5 min) in chop before recognizing flatline. | 4 | 5 | 4 | **80** | z_var < 0.20 threshold | Acceptable — 10 bars is reasonable patience. |
| 7 | **survival_stop** | ePnL exit requires 3 obs (cold start) | First 3 trades per template can't use Bayesian ePnL exit. Falls back to flatline detection. | 3 | 6 | 3 | **54** | Flatline fallback mode | Acceptable — cold start is inherent. |
| 8 | **peak_giveback** | Giveback IS→OOS flip ($2.90 → -$11.57) | 100% WR in IS but negative in OOS = overfitted threshold. Shape-calibrated thresholds may not generalize. | 7 | 6 | 8 | **336** | Self-tuning (tune_shrink_step=0.05) | **HIGH PRIORITY**: Investigate OOS giveback losses. Are they from wrong-direction trades that peaked briefly? Validate giveback threshold stability. |
| 9 | **peak_giveback** | Threshold too tight (10% giveback_pct) | Exits immediately after any pullback. Doesn't allow normal trade oscillation. Natural MFE → 10% pullback → exit → price continues. | 6 | 5 | 5 | **150** | Self-tuning adjusts dynamically | 10% is aggressive. Consider 20-25% base for min-hold experiment trades. |
| 10 | **envelope_decay** | Halflife too long (40 bars base) | Holds losing trades too long. Envelope barely decays in first 20 bars (50% remaining). | 4 | 3 | 5 | **60** | ADX slope + giveback ratio modulate HL | Acceptable — lazy safety net by design. Giveback is primary exit. |
| 11 | **envelope_decay** | Halflife too short (HL floor = 8) | Decays too fast on short-template trades. Forces exit before trade thesis plays out. | 5 | 3 | 4 | **60** | template_hl_floor=8, hl_mult_floor=0.3 | Acceptable — rare edge case. |
| 12 | **belief_flip** | False flip in choppy conviction | Exits on temporary conviction wobble. Conviction can oscillate bar-to-bar in ranging markets. | 5 | 4 | 6 | **120** | Requires bars_held >= 2, conviction_delta < -0.3 | Consider requiring 2 consecutive bars of flip confirmation. |
| 13 | **belief_flip** | Doesn't fire (conviction never flips) | Trade rides into loss without belief system warning. Trend fades gradually without sharp conviction reversal. | 6 | 4 | 8 | **192** | regime_decay as backup | This is exactly why regime_decay exists — backup for gradual fades. |
| 14 | **death_hook** | Micro ADX never reaches 40 threshold | Death hook never activates. Most trades don't see micro ADX > 40. Only 57 IS fires / 8,017 trades. | 3 | 7 | 3 | **63** | Other exits handle these cases | Acceptable — designed as rare surgical exit. |
| 15 | **band_urgent** | Fires on chop, not reversal | -$22.25/trade (worst performer). Exits at extreme band = often mean-reversion opportunity, not reversal. | 8 | 2 | 7 | **112** | Suppressed during min-hold | **DONE**: Suppressed in min-hold. Consider disabling entirely — negative expected value. |
| 16 | **tidal_wave** | SE expansion threshold (20%) too sensitive | Exits on normal volatility expansion (news, session open). These are often temporary and mean-revert. | 4 | 3 | 5 | **60** | Requires adverse z-score direction | Acceptable — dual condition limits false fires. |
| 17 | **watchdog** | Workers against threshold (5) rarely met | Only fires when 5+ workers disagree. In strong trends, workers align — watchdog is silent when most needed (trend exhaustion). | 4 | 6 | 5 | **120** | Other exits (envelope, giveback) cover this | Consider lowering to 3-4 workers if trade is underwater. |
| 18 | **breakeven** | Lock at 4 ticks too early | Locks breakeven on a 1-tick profit buffer. Natural oscillation trips the adjusted SL. Turns potential winner into scratch. | 5 | 6 | 4 | **120** | be_buffer_ticks=1.0 | Consider raising be_activation_ticks to 8-10 for min-hold trades. |
| 19 | **ALL (cascade)** | No exit fires before SL | All guardrails miss (regime_decay ADX > 20, belief stable, no DI cross). Trade hits 40-tick SL = $10 loss. | 9 | 2 | 9 | **162** | SL as last resort | This is the design — SL is the emergency floor. Acceptable if RPN stays low via low occurrence. |
| 20 | **ALL (min-hold)** | Suppressed exits miss real reversals | During 5-min hold, envelope/giveback/survival suppressed. Real reversal happens but only regime_decay and belief_flip can catch it. | 7 | 4 | 7 | **196** | regime_decay + belief_flip + SL(40t) | **MONITOR in experiment**: Track how many min-hold trades would have exited profitably via suppressed exits. |

---

## 3. PFMEA PRIORITY RANKING (by RPN)

| Rank | RPN | Exit Module | Failure Mode | Status |
|------|-----|-------------|-------------|--------|
| 1 | **336** | peak_giveback | IS→OOS performance flip | INVESTIGATE — threshold may be overfit |
| 2 | **252** | regime_decay | Premature fire (ADX oscillates near 20) | MONITOR — consider ADX smoothing |
| 3 | **216** | stop_loss | SL too tight (4-tick floor) | **FIXED** — min-hold raises to 40 ticks |
| 4 | **196** | ALL (min-hold) | Suppressed exits miss real reversals | EXPERIMENT — track in --min-hold run |
| 5 | **192** | belief_flip | Never fires (gradual fade) | COVERED by regime_decay backup |
| 6 | **162** | ALL (cascade) | No exit fires before SL | BY DESIGN — acceptable |
| 7 | **150** | regime_decay | DI cross false signal in chop | CONSIDER adding ADX gate on DI cross |
| 8 | **150** | peak_giveback | 10% threshold too tight | CONSIDER raising base to 20% |
| 9 | **140** | regime_decay | Fires too late (slow ADX decline) | Hurst is faster backup |
| 10 | **120** | belief_flip | False flip in choppy conviction | CONSIDER 2-bar confirmation |
| 11 | **120** | watchdog | 5-worker threshold rarely met | CONSIDER lowering to 3-4 |
| 12 | **120** | breakeven | Locks too early (4t activation) | CONSIDER 8-10t for min-hold |
| 13 | **112** | band_urgent | Fires on chop, not reversal | **FIXED** — suppressed in min-hold |
| 14 | **105** | stop_loss | SL too wide (40t in experiment) | MONITOR max single-trade loss |

---

## 4. KEY FINDINGS

### Finding 1: Giveback is the #1 Risk (RPN 336)
Peak giveback flips from +$2.90 IS to **-$11.57 OOS**. This is the highest-RPN failure mode.
The 10% giveback threshold may be overfitted to IS MFE distributions. OOS trades have different
MFE profiles (compressed path, no oracle), so the same tight threshold catches noise pullbacks
on trades that would have recovered.

**Action**: After min-hold experiment, compare giveback outcomes IS vs OOS. If OOS giveback
trades show higher MFE-at-exit than IS, the threshold is too tight for OOS conditions.

### Finding 2: Regime Decay Dominance + Fragility (RPN 252)
44% of IS trades exit via regime_decay. It's the workhorse but also the most fragile:
- ADX oscillating near 20 → premature fires (81.5% WR = 18.5% wrong)
- ADX declining slowly → late fires (trade bleeds first)
- DI crosses in chop → false reversal signals

**Action**: Consider regime_decay with confirmation (2 bars below ADX threshold, not just crossing it).

### Finding 3: Stop Loss Volume = Noise Trading (39% of IS)
3,140 trades exit via SL at $0.39/trade. These are effectively noise captures — barely breakeven.
The 4-tick SL floor means any normal 1-point ($0.25 × 4 = $1.00) oscillation triggers the stop.
The min-hold experiment (40-tick floor) directly addresses this by giving trades room.

### Finding 4: Duration is the Master Variable
The C&E matrix confirms hold duration as the #1 input variable (score 10/10).
Every other input variable is downstream of "did the trade have time to develop?"
The min-hold experiment is testing the right hypothesis.

### Finding 5: Min-Hold Risk Profile
The experiment suppresses 8 of 12 exits during the hold period. Only SL (40t), regime_decay,
and belief_flip remain active. The RPN 196 risk is that suppressed exits (especially giveback
and envelope) would have caught real reversals that regime_decay and belief_flip miss.

**Mitigation**: After the run, compare:
- How many trades hit SL during hold period (= guardrail failures)
- What exit would have fired first if not suppressed
- PnL difference between actual exit and what suppressed exit would have given

---

## 5. AUDIT UPDATE (2026-03-15) — Code-Level Exit Cascade Review

### Finding 6: BreakevenLock Was the Real Primary Exit (56% of trades)
**Root cause discovered**: `BreakevenLock` activated after just **4 ticks ($1)** of MFE and
ratcheted SL above entry price. Any 1-tick reversal triggered "stop_loss" at $0.50 profit.
5,174/9,253 IS trades (56%) exited this way — labeled as `stop_loss` but actually breakeven
scalping. The dashboard WR of 80% was real; the 97% WR from signal_log shards was wrong
(signal_log PnL column misaligned with trade outcomes).

**Fix (commit acd08ff)**: Rewrote as `TrailingStop` with MFE-based activation:
- Activation at 80% of template p75_mfe (TF-scaled), floor $10, ceiling $100
- Old: 4 ticks → immediate breakeven lock. New: wait for statistical profit target.

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 21 | **breakeven (old)** | 4-tick activation = instant SL ratchet | 56% of trades exit at $0.50 profit. Prevents any trade from developing. | **9** | **9** | **3** | **243** | **FIXED** — TrailingStop rewrite |

### Finding 7: ExitAction Enum Reuse (Reporting Ambiguity)
Three different exits return `ExitAction.TRAIL_STOP`:
- V-reversal (exit_engine.py line 492)
- Belief flip urgent (belief_flip.py line 29)
- Belief flip DI cross (belief_flip.py line 56)

One exit returns `ExitAction.REGIME_DECAY` but isn't regime decay:
- Tidal Wave (tidal_wave.py line 86)

**Impact**: Trade reports aggregate these together — can't distinguish belief flips from
V-reversals from trailing stops. The prior PFMEA entry for "trail_stop/belief_flip" (row 6)
was actually 3+ different exit mechanisms lumped together.

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 22 | **belief_flip + tidal_wave** | ExitAction enum reuse | Reports can't distinguish exit types. Misattributed PnL per exit. | 4 | 8 | 7 | **224** | OPEN — need unique ExitAction values |

### Finding 8: Peak Giveback Config Mismatch
Constructor default: `giveback_pct=0.70`. TradingConfig default: `giveback_pct=0.10`.
Production uses config (correct), but direct instantiation in tests/tools would use 0.70.

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 23 | **peak_giveback** | Constructor vs config default mismatch | Direct instantiation uses 7x tighter threshold than production | 3 | 2 | 8 | **48** | LOW — production is correct, footgun for tests |

### Finding 9: Watchdog Uses Wrong Reference Field
Watchdog compares MFE progress against `trail_activation_ticks` (line 38), but should use
`anchor_mfe_ticks`. Trail activation is for the trailing stop, not watchdog progress checks.

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 24 | **watchdog** | Wrong reference field (trail_activation vs anchor_mfe) | Watchdog fires at wrong MFE threshold — either too early or never | 5 | 5 | 6 | **150** | OPEN — fix field reference |

### Finding 10: Band Urgent Requires Loss to Fire
Band urgent only fires when `unrealized_ticks < -loss_ticks` (underwater by 2+ ticks).
If multi-TF support breaks but the trade is still in profit, no exit fires.
Thesis invalidation is ignored if PnL is positive.

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 25 | **band_urgent** | Only fires on loss (ignores in-profit reversals) | Structural reversal missed while in profit → gives back gains | 5 | 4 | 7 | **140** | OPEN — consider removing loss gate |

### Finding 11: Hardcoded Thresholds Not in Config
- Belief Flip DI gap: `5.0` (belief_flip.py line 54) — cited as "87% accurate at gap≥5"
- Envelope ADX slope boost: `0.05` (envelope.py line 96)
- Envelope ADX slope penalty: `0.1` (envelope.py line 99)
- Giveback shape blend: `0.7` (giveback.py line 72)
- Survival stop breakeven gate: `current_ticks < 0 → return None` (never fires on losses)

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 26 | **multiple** | Magic numbers not in TradingConfig | Can't tune without code changes. Violates no-magic-numbers rule. | 3 | 8 | 4 | **96** | OPEN — extract to config fields |

### Finding 12: Death Hook Private Member Access
`fractal_exhaust.py` directly accesses `worker._last_tf_bar_idx` and `worker._states` — private
members of TBN workers. If worker internals change, death hook silently breaks.

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 27 | **death_hook** | Fragile coupling to TBN internals | Refactoring TBN workers breaks death hook silently | 4 | 3 | 8 | **96** | OPEN — add public accessor methods |

---

## 6. UPDATED PFMEA PRIORITY RANKING (2026-03-15)

| Rank | RPN | Exit Module | Failure Mode | Status |
|------|-----|-------------|-------------|--------|
| 1 | **336** | peak_giveback | IS→OOS performance flip | **MITIGATED** (sensor fusion thresholds, 2026-03-18) |
| 2 | **252** | regime_decay | Premature fire (ADX oscillates near 20) | **MITIGATED** (sensor enrichment, 2026-03-18) |
| 3 | **243** | breakeven (old) | 4-tick activation = instant SL ratchet | **FIXED** (TrailingStop rewrite) |
| 4 | **224** | belief_flip + tidal_wave | ExitAction enum reuse (reporting ambiguity) | **FIXED** (unique ExitAction per exit, 2026-03-18) |
| 5 | **216** | stop_loss | SL too tight (4-tick floor) | **FIXED** (tolerance interval + cap) |
| 6 | **196** | ALL (min-hold) | Suppressed exits miss real reversals | **REPLACED** by sensor enrichment approach |
| 7 | **192** | belief_flip | Never fires (gradual fade) | **MITIGATED** (sensor enrichment gates noise, lets real flips through) |
| 8 | **162** | ALL (cascade) | No exit fires before SL | BY DESIGN |
| 9 | **150** | regime_decay | DI cross false signal in chop | **MITIGATED** (requires 1+ sensor confirmation) |
| 10 | **150** | peak_giveback | 10% threshold too tight | **REPLACED** by sensor fusion tiered thresholds |
| 11 | **150** | watchdog | Wrong reference field (trail_activation vs anchor_mfe) | OPEN |
| 12 | **140** | regime_decay | Fires too late (slow ADX decline) | Hurst is faster backup |
| 13 | **140** | band_urgent | Only fires on loss (thesis invalidation ignored) | OPEN |
| 14 | **120** | belief_flip | False flip in choppy conviction | **FIXED** (requires 2+ sensors, 2026-03-18) |
| 15 | **120** | watchdog | 5-worker threshold rarely met | CONSIDER lowering to 3-4 |
| 16 | **96** | multiple | Magic numbers not in TradingConfig | OPEN |
| 17 | **96** | death_hook | Fragile coupling to TBN internals | OPEN |
| 18 | **48** | peak_giveback | Constructor vs config default mismatch | LOW |

---

## 7. AUDIT UPDATE (2026-03-18) -- Sensor-Enriched Exits + Peak State Exit

### Context
Research on human seeds + IS peak trades showed:
- **Real peaks**: 1m volume collapses, 1m F_momentum leaves, 1s velocity flips
- **Fake peaks**: 1m volume still flowing, momentum building against trade
- Belief flip was firing on bar 1-2 (stutter) because it had no sensor confirmation
- Tidal wave fired on SE expansion without checking if expansion was against trade
- The exit signal for "the move peaked" is the INVERSE of the entry signal

### Changes Made

**New exit module: `peak_state_exit.py` (inverted entry)**
- Checks: "would the system enter against me right now?"
- 4 sensors: 1s velocity against, 1m volume against, 1m DMI against, 1m F_momentum against
- Full (4/4 sensors) = exit immediately
- Strong (3/4) + giving back 15%+ = exit
- Min hold: 3 bars (peak), 5 bars (template) -- prevents stutter
- Cooldown: 6 bars after exit -- prevents re-entry on same decaying peak
- Own ExitAction: PEAK_STATE_EXIT (separate from PEAK_GIVEBACK)

**Enriched exits (not suppressed -- they now use sensor data to PREVENT false alarms):**
- **belief_flip**: TBN urgent requires 2+ sensors confirming. DI crossover requires 1+ sensor.
  Result: dropped from ~631 fires to 19 in OOS (97% noise reduction).
- **tidal_wave**: SE expansion requires 1+ sensor confirming against trade.
  Result: dropped from ~514 to 207 in OOS.
- **regime_decay**: unchanged (already had higher-TF override + adaptive thresholds)

**TBN enhanced**: `get_exit_signal()` now returns 4 directional sensor signals:
- `vel_1s_against`: 1s velocity flipped against trade direction
- `vol_1m_against`: 1m volume flowing against trade direction
- `dmi_1m_against`: 1m DMI crossed against trade direction
- `fm_1m_against`: 1m F_momentum against trade direction

### New PFMEA Items

| # | Exit Module | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|-------------|-------------|--------|---|---|---|---------|--------|
| 28 | **peak_state_exit** | Sensors never agree (all 4 rarely fire together) | Inverted entry exit never fires; falls through to proxy exits | 5 | 4 | 5 | **100** | MONITOR -- track fire rate in full IS run |
| 29 | **peak_state_exit** | Sensors agree on noise (1s velocity noise) | False exit on 1s noise that happens to align with other sensors | 4 | 3 | 6 | **72** | 1m sensors normalize noise; min peak 4t gate |
| 30 | **belief_flip (enriched)** | Sensor gate too strict (2+ required) | Real belief flip with only 1 sensor confirming gets suppressed | 5 | 3 | 7 | **105** | MONITOR -- compare belief_flip trades before/after |
| 31 | **tidal_wave (enriched)** | Sensor gate masks real tidal wave | SE expansion against trade but no sensors confirm (1m lagging) | 4 | 3 | 6 | **72** | SL is safety net below; regime_decay also catches |

### Updated Rankings (items changed by 2026-03-18 work)

| Orig RPN | New RPN | Exit Module | Change | Reason |
|----------|---------|-------------|--------|--------|
| 336 | **168** | peak_giveback IS->OOS flip | S=7 O=4 D=6 | Sensor fusion thresholds replace static %; less overfit risk |
| 252 | **126** | regime_decay premature | S=6 O=3 D=7 | Sensor enrichment prevents firing without confirmation |
| 224 | **0** | ExitAction enum reuse | RESOLVED | Each exit has unique action: BELIEF_FLIP, TIDAL_WAVE, V_REVERSAL, PEAK_STATE_EXIT |
| 196 | **100** | Min-hold suppression | S=5 O=4 D=5 | Replaced by sensor enrichment (exits not suppressed, just validated) |
| 192 | **96** | belief_flip never fires | S=6 O=4 D=4 | Sensor gate removes noise, lets real flips through -- net more useful |
| 150 | **75** | DI cross false signal | S=5 O=3 D=5 | Sensor confirmation prevents chop-triggered crosses |
| 120 | **0** | belief_flip choppy conviction | RESOLVED | 2+ sensor requirement eliminates conviction wobble exits |
