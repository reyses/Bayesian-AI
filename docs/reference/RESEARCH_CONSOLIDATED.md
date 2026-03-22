# Consolidated Research Findings
> Last updated: 2026-03-22
> Source: All research journals, standalone tools, pivot scanners, gate analysis, exit studies

---

## 1. ENTRY SIGNALS — What Predicts Good Entries

### 1.1 Peak Detection (1s detect / 1m confirm)
- **258,933 pivots** analyzed on full ATLAS (12 months)
- 98% REAL reversals, 1% fakeouts, 1% marginal
- Median hold: 82s (1.4 min) for REAL, 275s (4.6 min) for FAKEOUT
- **Fakeouts are LONGER** — real reversals resolve fast

### 1.2 REAL vs FAKEOUT Discriminators (1m features at entry)

| Feature | REAL | FAKEOUT | Delta | Signal |
|---------|------|---------|-------|--------|
| **1m volume_delta** | +1.5 | -32.2 | +33.7 | Volume dying = real. Volume flowing = fake. |
| **1m F_momentum** | -0.7 | -27.1 | +26.4 | Momentum dead = real. Still building = fake. |
| **1m DMI_diff** | -0.6 | -0.6 | 0.0 | DMI diff alone doesn't separate (need extreme+vol) |
| **15s P_center** | 0.44 | 0.38 | +0.06 | Higher center prob = real |
| **15s coherence** | 0.61 | 0.58 | +0.02 | More orderly = real |

### 1.3 The Exhaustion Pattern (from human seed research)
- Entry signal is NOT the DMI cross (too late, lagging)
- Entry signal IS: **DMI extreme (one side high vs other) + volume collapse**
- The cross CONFIRMS what already happened 2-3 bars ago
- Volume tells you the move is dying BEFORE price confirms it

### 1.4 False Peak vs Real Taper (609 human seeds)

| Feature | False Peak | Real Taper | Separation |
|---------|-----------|-----------|------------|
| Volume 1m (aligned) | +62.5 | +238.7 | Real: 3.8x more volume |
| F_momentum 1m (aligned) | +63.0 | +267.1 | Real: 4.2x stronger |
| |F_momentum| 15s | +43.6 | +119.8 | Real: 2.7x stronger |
| Volume 15s (raw) | +1.3 | +20.3 | Real: 16x more volume |

### 1.5 Peak Template Outcomes (174,085 peaks)

| Outcome | % | Volume | DMI | Range |
|---------|---|--------|-----|-------|
| Reversal | 68.9% | 0.6 (dying) | -0.21 (crossing) | tight |
| Continuation | 29.7% | 14.9 (flowing) | +0.26 (steady) | wide |
| Plateau | 1.4% | 2.3 | +2.06 | medium |

**Top discriminating features (Kruskal-Wallis H-statistic):**
1. Geometric range (H=5,802) — tightest = reversal
2. Volume delta (H=4,244) — dying = reversal
3. F_momentum (H=2,735) — fading = reversal
4. PID output (H=198)
5. DMI Diff (H=109)
6. ADX (H=82) — weakest discriminator

### 1.6 Chasing Filter
- Winners enter at F_momentum = 13.9
- Losers enter at F_momentum = 20.8
- High fm at entry = chasing extended move

---

## 2. EXIT SIGNALS — What Predicts Good Exits

### 2.1 Exit Module Performance (ranked by $/trade)

| Exit | IS Trades | IS WR | IS $/trade | Status |
|------|-----------|-------|------------|--------|
| survival_stop | 781 | 98.2% | $35.71 | BEST performer |
| stop_loss | 3,140 | 89.1% | $0.39 | Noise/scalping (was breakeven bug) |
| regime_decay | 3,562 | 81.5% | $6.51 | Workhorse but fragile |
| peak_giveback | PF 0.32 | 43% | -$3.62 | NET LOSER — too aggressive |
| tidal_wave | PF 0.00 | 0% | -$38.65 | DISABLED |
| belief_flip | PF 0.02 | 10% | -$18.88 | DISABLED |
| peak_state_exit | PF 0.05 | 25% | -$18.85 | DISABLED |

### 2.2 The 1m Exit Signature
**What a GOOD exit looks like (REAL pivot entry = prior trade's exit):**
- 1m volume_delta: +1.5 (near zero = exhausted)
- 1m F_momentum: -0.7 (near zero = force died)
- 1m dmi_diff: -0.6 (near zero = DMI crossing)
- 1m z_score: -0.03 (near zero = returned to center)

**What a BAD exit looks like (exiting too early = fakeout):**
- 1m volume_delta: -32.2 (still flowing)
- 1m F_momentum: -27.1 (still building)

### 2.3 Volume Pattern at Entry → Exit
- **WIN**: entry vol = -57.5 (dying), exit vol = -145.5 (dead)
- **LOSS**: entry vol = +126.6 (flowing), exit vol = -24.0 (still alive)
- Volume collapse IS the exit signal

### 2.4 F_momentum Delta During Trade
- Winners: fm_delta = -15 (gentle fade)
- Losers: fm_delta = -86 (momentum collapsed against)
- When fm drops 30+ from entry → 1m has flipped → exit

### 2.5 Hold Duration Sweet Spot
- **30s-5m: ALL positive** — $54,868 total
- **5m+: ALL negative** — -$34,163 total
- Auto seeds confirm: 5-15 bars at 1m (5-15 min) median hold
- Over-holding past 5 min = TF anchor problem (exit engine stuck on 15s signals)

### 2.6 Sensor Fusion Timing (bar offset from peak)
- At +4 bars: WIN both bad = 13.4%, LOSS both bad = 47.1% (+33.8pp separation)
- At +7 bars: WIN = 8.9%, LOSS = 47.8% (+38.9pp separation)
- At +10 bars: WIN = 17.4%, LOSS = 61.5% (+44.1pp separation)
- Signal GROWS with time — losers deteriorate, winners don't

---

## 3. TF RELATIONSHIPS — Which TF Pairs Work

### 3.1 The Proven Pair: 1s detection + 1m confirmation
- 1s gives exact tick timing (entry precision)
- 1m gives macro exhaustion (volume + DMI extreme)
- 15s is noise for both entry and exit — middle ground helps nothing

### 3.2 TF Agreement = FAKEOUT (contrarian finding)
- Real reversals: 60% TF agreement
- Fakeouts: 77% TF agreement (+17pp)
- When both TFs agree on volume/momentum → continuation → fakeout
- When they DISAGREE → reversal → real entry
- Resonance cascade (4+ TFs peaked same direction): PF 2.44 FADING it

### 3.3 Parent TF Validation is CONTRARIAN
- Parent agreement = move exhausted at BOTH scales = WORSE entry
- Validated peaks perform WORSE than unvalidated
- Led to removing 1m entry gate (under lookahead — need to re-evaluate honestly)

### 3.4 TF Hierarchy for Trades
```
Human Seeds (15m, 1h)     → session structure, major reversals
  Trend Seeds (1m, 5m)    → swing direction, 5-15 min moves
    Peak Seeds (1s)        → entry/exit timing, micro-reversals
```

### 3.5 Slow TF Issues
- 4h worker frozen for hours (incomplete bar = stale state)
- Solution: blend completed bar with forming bar, weighted by maturity %
- 1s/5s/15s DMI = noise. 3m+ DMI = reliable.

---

## 4. TREND SEEDS + PEAK ALIGNMENT

### 4.1 Auto Seeds (Trend Seeds) Profile
- 31,602 seeds across 312 days (101/day)
- TF: 1m, params: min_reversal=30, min_bars=5, max_bars=15
- Median hold: 10 min (11 bars at 1m)
- Median MFE: 55 ticks ($27.50)
- MFE arrives at 74% of trade duration

### 4.2 Peaks Live INSIDE Trend Seeds
- 86% of peak pivots fall inside auto seed time ranges
- 8 peaks per trend seed on average
- Peaks are spread throughout the seed (flat distribution, slight front-loading)

### 4.3 Trend Alignment is THE Edge
- 80% of fakeouts are COUNTER-TREND (peak opposes trend seed)
- Only 7% of fakeouts are trend-aligned
- 48 percentage point separation

| | Aligned | Counter |
|---|---------|---------|
| Real peaks | 55.6% | 44.4% |
| Fakeouts | **7.4%** | **80.4%** |

- Aligned: 95.5t/trade, 76% WR, +$14,241
- Counter: 65.8t/trade, 69% WR, -$3,033
- Ratio: 1.4x edge for trend-aligned

### 4.4 Golden Path (Realistic Ceiling)
- Perfect: $1,601,380 (all trend seeds, known direction)
- Realistic (60% capture): $960,828
- Target: 25% capture = $240K (40 seeds/day of 101 available)
- Lookahead captured 25.3% ($243K)
- Honest system captures 0.9% ($8.4K)
- Gap = gates blocking 99.95% of detections

### 4.5 Detection vs Capture Funnel
```
744,466 peak detections (2,386/day)
  → 183,687 pass gates (25%)
    → 380 final trades (1.2/day)
      → $1,844 captured

Target: 25 trades/day = $50K-$240K
```

---

## 5. DIRECTION PREDICTION

### 5.1 Direction Cascade (8 voters, priority order)
1. Pattern type momentum (bias toward reversal)
2. Parent TF consensus
3. Child TF micro-reversals (mean reversion)
4. Band confluence (P_center position)
5. DMI alignment at discovery TF
6. Velocity sign (1s derivative)
7. Brain historical bias (template win rate)
8. Slow TF dominance (4h/1h weight 5.0, 1s weight 0.1)

### 5.2 Brain Conviction = Random (AUC 0.501)
- Brain adds zero predictive value for peak trades
- One counter for ALL peaks (tid=-100) — can't differentiate
- Template selection works, peak conviction doesn't

### 5.3 SHORT Dominates
- IS: SHORT $6,728 (75% WR) vs LONG $1,694 (69% WR)
- Consistent across all runs

---

## 6. RISK & SIZING

### 6.1 Depth-to-TF Sweet Spot
- Depths 5-7 (2m-5m): $13-17/trade, 72-78% WR
- Depth < 5: noise
- Depth > 10: stale (regime_decay overfits)

### 6.2 Trade Magnitude Predictors
- Tight geometric range at entry → explosive moves
- Volume collapse → larger reversal
- PID integral building → reversal imminent

### 6.3 Outlier Analysis
- 307 outlier trades (|PnL| > $200 in < 3 bars OR hold > 100 bars)
- Outliers are NET NEGATIVE: -$13,347
- Without outliers: $34,147 instead of $8,422
- Fast-big ($400+ in 2 bars): gap events, not repeatable
- Long-hold (200+ bars): failed exit cascade, TF anchor problem

---

## 7. ARCHITECTURE DECISIONS

### 7.1 Rolling PID (the 100x fix)
- Old: cumsum(z_scores) — history-dependent, diverges between IS and live
- New: rolling_mean(z_scores, window=200) — stateless after 200 bars
- IS went from $1,943 to $209,734

### 7.2 CUDA-Only
- CPU path removed 2026-03-08
- No fallback, no CPU testing

### 7.3 Delta Architecture
- ALL engine computations except term_pid are per-bar or rolling-window
- Bar-to-bar changes are history-independent
- This should inform all new feature design (crow, goat, observers)

### 7.4 Two Worker Sets (planned)
- Peak workers (1s): micro timing, entry trigger
- Trend workers (1m): macro direction, hold/exit
- Session workers (1h): daily structure context
- Higher TF peaks (5m-1h): exhaustion detection for trend lifecycle

### 7.5 IS-Derived Thresholds DON'T Transfer to OOS
- Features matter (volume, momentum separate winners)
- But VALUES are wrong (calibrated to IS conditions)
- This is why CNN approach needed — learns thresholds from data

---

## 8. BUGS & FIXES

### 8.1 Lookahead Bug (THE discovery, 2026-03-21)
- IS peak detection used state N+1 (next bar) since codebase was built
- ALL prior IS/OOS numbers were inflated
- Branch `pre-lookahead-fix` preserves old state at commit 09cb6171
- Relative comparisons still valid (both sides had same lookahead)

### 8.2 F_momentum Cold Start
- PID cumsum starts from zero in live (17x divergence from OOS)
- Mitigated: pre-computed warmup states from ATLAS
- Long-term: delta-based engine (no warmup needed)

### 8.3 BreakevenLock Scalp Bug
- 4-tick activation = 56% of IS trades at $0.50 profit
- Fixed: MFE-based TrailingStop (activation at 80% template p75)

### 8.4 4x TF Mismatch
- Oracle bar counts are 1m, consumed as 15s → 4x error
- Fixed: template discovery on 1m anchor, forward pass on 15s

---

## 9. OBSERVER DATA (2026-03-22 overnight run)

### 9.1 Detection Funnel
- 76,931 peak events logged
- 11,668 entered (15.2%), 65,263 skipped
- Trend-aligned: 37,557 peaks (7,914 entered, 29%)
- Counter-trend: 37,059 peaks (3,204 entered, 21%)

### 9.2 Problem: Entering Counter-Trend
- 42% of all entries are counter-trend
- Research says 80% of counter-trend peaks are fakeouts
- Filtering to aligned-only: 7,914 entries with 7% fakeout rate

### 9.3 Problem: Skipping Institutional Moves
- Storm (|fm_1m| > 200): 76% of peaks, 13% entry rate
- Calm markets: 25% entry rate
- Extreme volume: 13% entry rate
- Retail volume: 25% entry rate
- System enters retail/calm and skips institutional/storm — backwards

---

## 10. EACH CONCEPT ANSWERS ITS OWN QUESTION

| Concept | Question | What to Measure |
|---------|----------|-----------------|
| Trend direction | Which way is the river? | 1m DMI extreme (not cross) + volume trajectory |
| Trend strength | How strong is the current? | ADX for sizing (NOT for direction/entry) |
| Noise level | How choppy? | Sigma, geometric range → hold patience, stop width |
| Exhaustion | Is the move done? | 1m volume collapse + DMI extreme → entry/exit timing |
| Volume profile | Institutional or retail? | |vol_1m| magnitude → confirmation |
| Session | Where in the day? | 1h regime → expectations (not filtering) |
| TF agreement | Are TFs aligned? | DISagreement = reversal. Agreement = continuation/fakeout |

---

## 11. BASELINES

| Run | IS PnL | Trades | WR | PF | Notes |
|-----|--------|--------|----|----|-------|
| Lizard (cumsum) | $1,943 | — | — | — | Before rolling PID |
| Cat (no gate) | $209,734 | 34,446 | 70.4% | — | WITH lookahead |
| Cat + exits | $243,190 | — | — | 2.56 | WITH lookahead, peak |
| Seeds + lookahead | $22,565 | 3,264 | 65% | 2.55 | Pre-lookahead branch |
| **Honest baseline** | **$8,422** | **655** | **72.1%** | **2.63** | No lookahead, 2wk warmup |
| Honest + 1m flip | $1,844 | 380 | 56.1% | 2.28 | Chase filter too tight |
| Golden path ceiling | $960,828 | 31,602 | — | — | 60% capture of all trends |
