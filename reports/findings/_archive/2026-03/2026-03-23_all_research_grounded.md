# ALL Research — Grounded Feature Lens
> 2026-03-23 — Every finding reframed through base measurements.
> Rule: nth-layer features OK if they answer a named question.
> If a feature can't state what it measures in one sentence, flag it.

---

## 1. ENTRY DETECTION

### What we know works (grounded)

| Signal | Grounded as | Level | Question answered | Evidence |
|--------|------------|-------|-------------------|----------|
| velocity flip | dP/dt sign change | 2 | Did the move reverse? | 305K flips detected, baseline signal |
| volume collapse | V < 30% of avg(V, 60s) | 2 | Did participation die? | +1.4% accuracy boost (H=4,244) |
| range compression | std(P, short window) | 2 | Was price compressed? | Strongest discriminator (H=5,802) |
| magnitude threshold | abs(P_now - P_20ago) > p75 | 2 | Was the prior move big enough? | Reduces noise but doesn't improve direction |

### What we know works but needs proper grounding

| Signal | Old name | Grounded as | Level | Question | Status |
|--------|----------|------------|-------|----------|--------|
| DMI exhaustion | dmi_extreme + vol_low | dmi_diff extreme + V low | 2×2 cross | Is the buyer/seller battle exhausted? | **KEEP** — catches different exhaustion type than velocity |
| Trend alignment | "aligned vs counter-trend" | sign(vel_1m) vs sign(vel_15m) | 2 vs 2 | Does the structure support this reversal? | **KEEP** — 80% fakeout filter (48pp separation) |
| TF disagreement | "coherence inverted" | count(vel_sign agreement across TFs) | meta of level 2 | Are the timeframes fighting? | **KEEP** — disagreement = real reversal (PF 2.44 fading agreement) |
| Chasing filter | "fm > 20 at entry" | abs(velocity) at entry | 2 | Are we entering a move that's already extended? | **KEEP** — winners enter at vel=13.9, losers at vel=20.8 |

### What should be dropped

| Signal | Old name | Why drop |
|--------|----------|----------|
| P_center rise | "P_at_center > prev * 1.05" | Near-zero separation (delta=+0.06). Level 3 regression probability. Question: "is regression center shifting?" Answer: z-score already tells you this. **REDUNDANT** |
| Coherence threshold | "osc_entropy > 0.55" | Inverted std of z-score (level 4). r=0.001. The REAL coherence is TF velocity agreement. **REPLACED** |
| Brain conviction | "brain.should_fire()" | AUC=0.501 for peaks. Literally random. One counter for all peaks — can't differentiate. **USELESS for peaks** |
| Pattern type matching | "cascade/structure detected" | Template centroids averaged 86 seeds each, destroying information. Raw K-NN is better. **REPLACED by K-NN** |
| Buildup buffer | "10-bar pc_delta + fm_delta" | Deltas of deltas of derivatives. Level 4. The question "did this reversal build over time?" is valid but answered more directly by: did velocity change GRADUALLY (low acceleration) or SUDDENLY (high acceleration)? **REPLACE with acceleration** |

---

## 2. EXIT SIGNALS

### What works (grounded)

| Signal | Grounded as | Level | Question | Evidence |
|--------|------------|-------|----------|----------|
| Volume collapse during trade | V dropping from entry | 2 | Is participation dying? | Entry vol=-57.5 (dying)=WIN, +126.6 (flowing)=LOSS |
| Velocity fade during trade | abs(vel) decreasing | 2 | Is the move slowing? | fm_delta -15=WIN (gentle), -86=LOSS (collapsed against) |
| Hold duration | bars_held × bar_period | 1 (Time) | How long have we been in? | 30s-5m=profitable, 5m+=negative. Strongest single predictor (corr=10) |

### What works but was over-engineered

| Signal | Old name | Grounded version | Question |
|--------|----------|-----------------|----------|
| Sensor fusion (+4/+7/+10 bars) | "both bad at +N bars" | velocity_against AND volume_against at +N bars | Are BOTH base measurements saying get out? **YES — 44pp separation at +10 bars. KEEP.** |
| Envelope decay | halflife-based exponential | std(P, window) shrinking = move compressed | Is the trade range narrowing? Simpler: if std of price in trade drops below entry std, the move is dying. **SIMPLIFY** |
| Giveback | "gave back 65% of MFE" | (peak_price - current) / (peak_price - entry) | How much of the best price did we lose? **KEEP — pure Price measurement, level 2** |

### What should be dropped from exits

| Signal | Old name | Why |
|--------|----------|-----|
| regime_decay | ADX < 20 + Hurst < 0.5 + DI cross | Triple-derived. ADX=H(82), weakest discriminator. Replace with variance_ratio < threshold. **REPLACE** |
| belief_flip | TBN conviction flipped | PF=0.02. 10% WR. Literally anti-signal. **DROP** |
| tidal_wave | 4x noise expansion | PF=0.00. Zero profit factor. **DROP** |
| peak_state_exit | P_center + coherence trigger | PF=0.05. Both features have near-zero signal. **DROP** |
| death_hook | extreme z-score bands | Small sample. z-score > 2σ IS useful info but fires too rarely. **MERGE into SL sizing** |

---

## 3. DIRECTION PREDICTION

### The 8-voter cascade (grounded audit)

| # | Old voter | Grounded? | Keep? | Reason |
|---|-----------|-----------|-------|--------|
| 1 | Pattern type momentum | NO — pattern types are template artifacts | DROP | Replace with: sign(velocity) at detection TF |
| 2 | Parent TF consensus | YES — sign(vel_parent) | **KEEP** | MTF velocity agreement is proven |
| 3 | Child TF micro-reversals | PARTIALLY — mean reversion at lower TF | **KEEP** | But reframe: sign(vel_1s) vs sign(vel_1m) |
| 4 | Band confluence (P_center) | NO — P_center has zero signal | DROP | Replace with: z-score sign (same question, grounded) |
| 5 | DMI alignment | YES — dmi_diff sign | **KEEP** | Buyer/seller battle direction |
| 6 | Velocity sign | YES — dP/dt | **KEEP** | Fundamental |
| 7 | Brain historical bias | NO — AUC=0.501 for peaks | DROP | Will be replaced by seed reliability weighting |
| 8 | Slow TF dominance | PARTIALLY — 4h/1h velocity | **KEEP** | But use velocity sign, not weighted conviction |

**Grounded direction cascade (simplified):**
1. velocity sign at detection TF (which way just flipped?)
2. velocity sign at structure TF (does the bigger picture agree?)
3. dmi_diff sign (who's winning the battle?)
4. z-score sign (are we extended or mean?)

Four voters, all level 2, each from a different angle on Price.

---

## 4. RISK & SIZING

### Grounded audit

| Finding | Grounded? | Translation |
|---------|-----------|-------------|
| Depth 5-7 is sweet spot | YES — this is the aggregation scale | 2-5 min = CLT averaging surfaces signal from 1s noise. **Keep as: optimal aggregation window** |
| Tight range → explosive moves | YES — std(P, window) low = energy stored | **Keep: std_price at entry predicts magnitude** |
| Volume collapse → larger reversal | YES — volume base measurement | **Keep** |
| PID integral building | NO — accumulated z drift | **Replace with: z-score persistently extreme for N bars** |
| Outliers are NET NEGATIVE (-$13K) | YES — fat tail measurement | **Keep: this IS the SL justification. Cap losses at the tail.** |
| SL sizing | YES — but should scale with std(P) | **Improve: SL = f(std_price, window) not fixed 40 ticks** |

### SL should be grounded

Current: fixed 40 ticks. That's a magic number.
Grounded: SL = K × std(price_changes, 60s). K=3 means "3 standard deviations of recent 1-minute volatility." Scales automatically with the regime. Quiet market = tighter SL. Wild market = wider SL.

---

## 5. TF RELATIONSHIPS

### Grounded reframe

**Old:** "1s detection + 1m confirmation. 15s is noise."

**Grounded:** This is an aggregation question. At what CLT averaging level does signal emerge from noise?
- 1s: noise floor (can detect flips but can't predict direction at 50.77%)
- 1m (60× aggregation): signal emerges (K-NN hits 61% WR)
- 5m (300× aggregation): structural peaks visible (today's chart)
- 15s (15× aggregation): not enough averaging to surface signal, too much to keep timing precision. **Dead zone confirmed.**

**Old:** "TF agreement = fakeout. Disagreement = reversal."

**Grounded:** This is velocity coherence across aggregation scales.
- All scales agree → the move is consensus at every resolution → already priced in → late entry
- Scales disagree → structural break propagating → some scales haven't caught up → edge exists

This is the REAL coherence. Not entropy of oscillations. Just: do the velocities at different aggregation levels agree? Count the votes.

**Old:** "Parent TF validation is contrarian."

**Grounded:** When velocity_5m already flipped the same way as velocity_1m:
- The 5m already caught up → the reversal is priced in at the structural level → you're late
- When velocity_5m HASN'T flipped but velocity_1m just did → you're early → edge

This means: the best entries are when the LOWER TF flips first and the HIGHER TF hasn't caught up yet. The propagation delay IS the edge window.

---

## 6. TREND SEEDS

### Grounded reframe

**Old:** "31K ZigZag-detected swings with enriched physics."

**Grounded:** Each seed is a SEGMENT of price action defined by:
- Start: velocity_1m flipped direction (level 2)
- End: velocity_1m flipped again (level 2)
- Duration: Time between flips (level 1)
- Magnitude: Price change during segment (level 1)

The enrichment (12 features per bar) was mostly level 3-4 derivatives.
With grounded features, each seed bar would carry:
- velocity, volume, std_price, dmi_diff (level 2)
- acceleration, z-score, variance_ratio (level 3)

**Old:** "MFE arrives at 74% of trade duration."

**Grounded:** Peak favorable price occurs at 0.74 × hold_time.
This means: the optimal exit is at ~75% of the matched seed's duration.
Not when the funnel flips. Not when velocity reverses. At a FIXED TIME point relative to the seed's historical hold.
Question to test: does time-based exit (exit at 74% of matched hold) beat the funnel flip exit?

---

## 7. THE GOLDEN PATH

### Grounded reframe

**Old:** "$960K realistic ceiling from perfect trend seed following."

**Grounded:** If you could detect every velocity_1m flip and enter the correct direction at each one, holding for the median duration → $960K in 12 months.

System captures 0.9% ($8.4K). The gap is:
1. Detection: missing 99% of flips (gates too tight) → **solved by removing gates, using K-NN**
2. Direction: wrong direction on ~40% of entries → **partially solved by MTF velocity agreement**
3. Exit timing: holding too long or too short → **needs grounded exit (time-based? volume-based?)**

---

## 8. ARCHITECTURE DECISIONS (grounded audit)

| Decision | Grounded? | Status |
|----------|-----------|--------|
| Rolling PID (window=200) | PARTIALLY — still a PID, but window makes it stationary | **REPLACE with velocity (dP/dt). PID was always a noisy velocity.** |
| CUDA-only | YES — computation constraint, not feature | **KEEP** |
| Delta architecture | YES — bar-to-bar changes are grounded | **KEEP — this IS the "measure the process" principle** |
| Templates → K-NN | YES — templates were premature optimization | **DONE — PhysicsEngine already uses raw K-NN** |
| IS thresholds don't transfer | YES — values are regime-dependent | **Validates grounded approach: measure shape (std, ratio), not absolute values** |

---

## 9. BUGS REFRAMED

| Bug | Grounded explanation |
|-----|---------------------|
| Lookahead (state N+1) | Used FUTURE velocity to detect CURRENT peak. Of course it worked — you could see the reversal before it happened. |
| F_momentum cold start (17x divergence) | PID integral accumulated differently with different starting bars. A GROUNDED feature (velocity = dP/dt) has no cold start — each bar's velocity is independent. |
| BreakevenLock scalp | SL at 4 ticks = 1 std of 1s noise. Random noise triggers it. SL should be K × std(P), not fixed ticks. |
| 4x TF mismatch | Counting 1m bars as 15s bars. Aggregation scale confusion. Grounded approach: every feature specifies its time window explicitly. |

---

## 10. SUMMARY — QUESTIONS AND THEIR GROUNDED ANSWERS

| # | Question | Grounded answer | Feature | Level | Proven? |
|---|----------|----------------|---------|-------|---------|
| 1 | Did the move reverse? | velocity sign flipped | velocity | 2 | YES (305K events) |
| 2 | Is participation dying? | volume < 30% of avg | volume | 2 | YES (+1.4% acc, H=4,244) |
| 3 | Was price compressed? | std(P) low before flip | std_price | 2 | YES (H=5,802, strongest) |
| 4 | Is the move speeding up? | d(velocity)/dt | acceleration | 3 | NOT YET (0% improvement in layered test) |
| 5 | Is the battle exhausted? | dmi_diff extreme + vol low | cross signal | 2×2 | YES (+1.0% short-term) |
| 6 | Does structure support it? | vel_1m sign vs vel_15m sign | MTF velocity | 2 vs 2 | YES (80% fakeout filter) |
| 7 | Trending or reverting? | var(short) / var(long) | variance_ratio | 3 | PARTIAL (thin sample in layered) |
| 8 | How long to hold? | matched seed median duration | Time | 1 | YES (30s-5m profitable, 5m+ negative) |
| 9 | When to exit? | vel_against AND vol_against for N bars | sensor fusion | 2+2 | YES (44pp at +10 bars) |
| 10 | How much to risk? | K × std(P, 60s) | dynamic SL | 2 | NOT YET (currently fixed 40 ticks) |
| 11 | Where in the range? | (P - low_5d) / (high_5d - low_5d) | fib_position | 2 | NOT YET (research line open) |
| 12 | Is the move real? | sign(vel) == sign(vol) | price×volume | 2×2 | NOT YET |
| 13 | When in the day? | session phase from Time | session | 1 | NOT YET |
| 14 | How noisy is it? | std(dP, 1s, window) | noise floor | 2 | YES (noise floor concept validated) |

**Proven: 8 of 14. Untested: 6. Zero features without a named question.**
