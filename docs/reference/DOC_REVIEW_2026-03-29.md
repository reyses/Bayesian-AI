# Full Documentation Review — 2026-03-29

## Purpose
Complete audit of every document in the project. Identify hidden gems,
contradictions with current approach, and actionable findings.

---

## HIDDEN GEMS (actionable, forgotten)

### 1. Volume SPIKE at peaks, NOT collapse
**Source:** 2026-03-22 journal, RESEARCH_CONSOLIDATED
**Finding:** Real peaks show 5.5x volume SPIKE at bar +0. Volume COLLAPSE comes
AFTER the peak (bars +1 to +5). Our shape analysis needs to check this — are
we measuring the spike correctly, or looking at the wrong bars?
**Action:** Verify in shape analysis feature profiles

### 2. TF Agreement = FAKEOUT (inverted signal)
**Source:** RESEARCH_CONSOLIDATED
**Finding:** Real reversals have 60% TF agreement. Fakeouts have 77% agreement.
When ALL timeframes agree, it's MORE likely to be a fakeout, not a real move.
**Impact:** We built entry gates requiring "all horizons agree" — this was WRONG.
The disagreement between near and far horizons that we measured in the trajectory
analysis is actually the CORRECT signal (confirmed by this old research).
**Action:** Entry logic should use TF DISAGREEMENT as signal, not agreement

### 3. Hold Duration Sweet Spot: 30s to 5m ONLY
**Source:** RESEARCH_CONSOLIDATED
**Finding:** 30s-5m holds = all positive (+$54K). 5m+ holds = all negative (-$34K).
Over-holding past 5 minutes = TF anchor problem.
**Impact:** L2 DurationPredictor predicted 36-bar holds. That's 36 MINUTES — way
past the sweet spot. The 5-10 bar clamp we tried (5-10 min) was still too long.
At 1m resolution, max hold should be ~5 bars (5 min).
**Action:** Hard cap at 5 bars for 1m resolution trades

### 4. Wave Function Probabilities — Computed, Never Used
**Source:** THREE_THEORIES.md, core/market_state.py
**Finding:** Every MarketState already contains:
  - prob_weight_center, prob_weight_upper, prob_weight_lower
  - P_at_center, P_near_upper, P_near_lower
  - reversion_probability, breakout_probability
  - entropy_normalized
These are FREE features computed every bar but never fed to any model.
**Action:** Add to 13D features or use as level proximity signals

### 5. Detection Funnel: 99.95% of Signals Blocked
**Source:** RESEARCH_CONSOLIDATED
**Finding:** 744K peak detections → 183K pass gates (25%) → 380 trades (0.1%)
→ $1,844 captured. Golden path ceiling: $1.6M perfect, $961K realistic.
Current system captures 0.2% of theoretical maximum.
**Impact:** Gates are catastrophically over-restrictive. The signal exists
(744K detections are real peaks), the gates destroy it.
**Action:** Loosen gates dramatically. Use confidence scoring, not binary gates.

### 6. Delta Architecture Solves Parity
**Source:** 2026-03-20 journal
**Finding:** Absolute feature values diverge between IS/OOS/live because of
different history lengths (cumsum effects). DELTA features (bar-to-bar change)
are history-independent — instant parity without warmup.
**Impact:** All our parity issues (F_momentum 17x divergence, 240-bar warmup)
disappear with delta-based features.
**Action:** Future feature engineering should prefer deltas over absolutes

### 7. Resonance Cascade: 5/5 TF Agreement = Crash/Rally
**Source:** project_resonance_cascade.md
**Finding:** When all 5 TF pairs (1D/4h, 4h/1h, 1h/15m, 15m/5m, 5m/1m) agree
on direction → resonance cascade → crash or rally imminent. Feb 9 was 5/5 SHORT.
**Impact:** This is the STRUCTURAL trend detector. Not for individual trades —
for detecting when the market shifts regime entirely.
**Action:** Build resonance cascade detector as structural bias input

### 8. User's 6 Missing Features (from manual trading system)
**Source:** user_vp_trading_system.md
**Finding:** The user's manual system has 6 concrete features not yet automated:
  1. Headroom gate (room to run before next level)
  2. 4σ wall exit (hard structural stop)
  3. Zone mode switching (different strategy per zone)
  4. Asian session trap skip (low liquidity = fake moves)
  5. Visibility rule (don't trade if 3σ off screen)
  6. Pre-open force check (globex positioning)
**Action:** These are concrete roadmap items from proven manual trading

### 9. DMI Extreme + Volume Exhaustion = Reversal (not DMI Cross)
**Source:** 2026-03-22 journal, RESEARCH_CONSOLIDATED
**Finding:** DMI CROSS is too late (lagging indicator). The FADING of dmi_diff
combined with volume collapse gives 10-20 bar advance warning of reversal.
This is what the EDA shape analysis confirmed: dmi_gap drops before touch.
**Action:** Use dmi_diff TRAJECTORY (fading, not crossing) as reversal signal

### 10. Variance Ratio Replaces ADX + Hurst
**Source:** FIRST_PRINCIPLES_FRAMEWORK
**Finding:** Single grounded measurement: std_short / std_long.
  >1 = trending (short-term more volatile than long-term)
  <1 = mean-reverting
  ≈1 = random walk (decoherent)
Replaces both ADX (trend strength) and Hurst (fractal dimension) with one
number derived from Price only.
**Action:** Already in 13D features as variance_ratio. Use as regime detector.

---

## CONTRADICTIONS WITH CURRENT APPROACH

### A. TF Agreement (Entry Gate)
**Old research says:** TF disagreement = real signal (60% agreement at real reversals)
**Current system does:** Requires all horizons to agree before entry
**Resolution:** Use DISAGREEMENT as signal — confirmed by trajectory horizon
analysis (near-far disagreement spikes at peaks, p=0.0000)

### B. Hold Duration
**Old research says:** 5m+ holds are all negative
**Current system does:** L2 predicts 36-bar holds, clamped to 5-10
**Resolution:** Hard cap at 5 bars (5 min) for 1m, let the trajectory
decay curve decide exit within that window

### C. Volume at Peaks
**Old research says:** Volume SPIKES at peak (5.5x), COLLAPSES after
**Current shape analysis shows:** vol_rel spikes at touch for ALL types
**Resolution:** Need to separate: spike AT touch (entry) vs collapse AFTER
(confirmation). Both are true — they're sequential.

### D. Gate Strictness
**Old research says:** 99.95% of detections blocked, $1,844 of $1.6M captured
**Current system does:** Multiple confidence gates (3.0 threshold, all agree, etc)
**Resolution:** Replace binary gates with continuous confidence scoring.
Trade MORE, smaller, with position sizing based on confidence.

---

## JOURNALS SUMMARY (Feb 28 - Mar 22)

| Date | Key Discovery | Still Valid? |
|------|--------------|-------------|
| Feb 28 | I-MR method, 3 Questions, lookahead bug found+fixed | YES |
| Mar 1 | Pipeline baseline: templates broken, physics carries system | YES |
| Mar 2 | Exit CE matrix, two feedback loops (too_early, too_late) | YES |
| Mar 7 | Band conflict exit study, sensor enrichment | PARTIALLY |
| Mar 8 | Trajectory extrapolation breakthrough, 270-day results | YES |
| Mar 9 | K-NN trajectory (coin flip at execution TF), noise floor 8.8t | YES |
| Mar 10 | (no journal) | — |
| Mar 11 | TF magnitude mismatch, two research lines (R1 bucketed, R2 1m anchor) | YES |
| Mar 12 | V7.0.0: $39,736 IS, $8,200 OOS, TF bucketing works | YES |
| Mar 13 | Exit PFMEA, 13-layer cascade audit | YES |
| Mar 14 | Gate cascade PFMEA | YES |
| Mar 15 | Oscillation research spike (8-min period hypothesis) | CONFIRMED |
| Mar 16 | Level drawing tool conception | YES |
| Mar 18 | Sensor fusion overhaul, 174K peak analysis | YES |
| Mar 19 | ADX/fake peak/peak override all FAILED | YES (proven wrong) |
| Mar 20 | F_momentum parity break, delta architecture insight | YES |
| Mar 21 | Lookahead bug in peak detection, rolling PID fix | YES |
| Mar 22 | First principles rebuild, 13-feature canonical set, volume spike discovery | YES |

---

## REFERENCE DOCS ASSESSMENT

| Document | Lines | Status | Key Value |
|----------|-------|--------|-----------|
| BAYESIAN_AI_MASTER.md | ~500 | KEEP | Architecture foundation, nightmare protocol |
| FIRST_PRINCIPLES_FRAMEWORK.md | ~300 | KEEP | Feature theory, DOE methodology |
| RESEARCH_CONSOLIDATED.md | ~800 | KEEP | Comprehensive findings database |
| RESEARCH_JOURNAL.txt | 3267 | KEEP (indexed) | Full research chronology |
| Playbook for MNQ.md | ~200 | KEEP | MNQ trading reference |

---

## MEMORY DOCS ASSESSMENT (selected)

| Document | Status | Key Value |
|----------|--------|-----------|
| ce_methodology.md | KEEP | C&E analysis framework |
| project_resonance_cascade.md | KEEP | Unfinished research, high potential |
| project_feature_tree.md | KEEP | Feature hierarchy |
| user_vp_trading_system.md | KEEP | User's manual system (6 missing features) |
| user_level_trading.md | KEEP | Level lifecycle theory |
| research_telescoping_tf.md | KEEP | 1m anchor clustering idea |
| research_pid_trance.md | KEEP | PID control loop insight |
| project_delta_architecture.md | KEEP | Future architecture direction |

---

## LEGACY ARCHIVE (docs/archive/legacy/)

| Document | Key Takeaway | Status |
|----------|-------------|--------|
| THREE_THEORIES.md | Wave function probs unused — hidden gem | REVISIT |
| MARKET_HIERARCHY.md | PID control zone model, 1-2σ sweet spot | KEEP as reference |
| CAUSE_AND_EFFECT_MATRIX.md | Gate vs modulator confusion methodology | KEEP |
| REGIME_TRADING_SPEC.md | Alternative DMI regime approach | DECOMMISSION |
| NIGHTMARE_PROTOCOL.md | Level classification (Titans/Architects/Explorers) | REVISIT |
| QUANTUM_DESIGN_INTENT.md | Original physics metaphor intent | ARCHIVE |
| SPEC_DECISION_FUNNEL.md | Entry funnel narrowing | ARCHIVE |
| DEEP_RESEARCH_ENTRY_IMPROVEMENTS.md | Entry improvement proposals | ARCHIVE |

---

## ADDITIONAL FINDS FROM DEEP ARCHIVE REVIEW

### 11. Hurst Gate Blocks 28.4% of Profitable Signals
**Source:** CLAUDE_CODE_EXIT_IMPROVEMENTS.md
**Finding:** 22,660 FN signals blocked by Hurst gate (Gate 0 Rule 5a).
Window=100 bars at 15s = 25 min. Potentially +$25K if relaxed.
**Action:** Build hurst_validation.py confusion matrix at window 50/100/200/400

### 12. Breakeven Lock at 2t MFE = +40% PnL
**Source:** CLAUDE_CODE_EXIT_IMPROVEMENTS.md
**Finding:** IS $82K→$115K, OOS $22K→$31K just by activating BE at 2 ticks MFE.
Currently set to 5,000 ticks (never fires).
**Action:** Lower BE activation to 4 ticks (noise floor derived)

### 13. One-Bar Stutter Trades = -$1,568
**Source:** PEAK_TEMPLATES.md, STATEFUL_PEAK_EXIT.md
**Finding:** 1,239 one-bar trades in OOS from entry stutter. Costs $1,568.
**Action:** Stateful peak monitor: IDLE→ENTER→TRACKING→EXHAUSTED→COOLDOWN

### 14. 13 Code Duplications (6 HIGH severity)
**Source:** JULES_CODEBASE_AUDIT.md
**Finding:** Feature extraction, exit sizing, TBN ticking, trade recording
all duplicated between live_engine.py and trainer.py.
**Action:** Merge into shared modules (prevents silent drift bugs)

### 15. MEMORY.md Misdirects Every Session
**Source:** DOC_STATUS_AUDIT (1).md
**Finding:** Wrong file paths (quantum_field_engine→statistical_field_engine),
ghost references (fractal_dna_tree.py deleted), wrong class names.
**Action:** Surgical update of MEMORY.md paths and references

### 16. IS Lookahead via pattern_map Still Active
**Source:** IS_BARPROCESSOR_REFACTOR.md
**Finding:** IS forward pass uses pre-computed pattern_map from ALL bars
(including future). IS metrics still inflated by lookahead.
**Action:** Delete inline forward pass, use BarProcessor with oracle post-labeling

---

## TOP 10 ACTIONS FROM THIS REVIEW (updated)

1. **FIX: TF agreement is FAKEOUT signal** — invert entry logic
2. **FIX: Hard cap hold at 5 bars (5 min)** — research proves 5m+ loses money
3. **FIX: BE activation at 4t (not 5,000t)** — proven +40% PnL
4. **FIX: Hurst gate relaxation** — 28.4% of profitable signals blocked
5. **ADD: Wave function probabilities** — free features, already computed
6. **ADD: Delta features** — solve parity without warmup
7. **ADD: Stutter suppression** — 1,239 one-bar trades = -$1,568
8. **BUILD: Resonance cascade detector** — structural regime shift
9. **CLEAN: MEMORY.md paths** — misdirects every AI session
10. **CLEAN: 13 code duplications** — prevents silent drift bugs
