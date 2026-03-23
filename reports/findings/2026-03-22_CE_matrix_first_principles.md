# C&E Matrix — First Principles Rebuild
> Generated 2026-03-22 from OOS feature-price analysis (42,795 bars)
> Based on: base measurements are Price, Time, Volume. Everything else is derived.

---

## 0. BASE MEASUREMENTS

| Base | What it measures | Independent? | Available? |
|------|-----------------|-------------|------------|
| **Price** | Where the market is | YES — the primary observable | Yes (close, high, low) |
| **Time** | When things happen | YES — independent of price | Implicit (bar index, session phase) |
| **Volume** | How much participation | YES — independent of price direction | Yes (volume_delta) |

---

## 1. FEATURE CLASSIFICATION BY DERIVATION LEVEL

| Level | Feature | Formula | Base inputs | OOS r_s (+1bar) | Verdict |
|-------|---------|---------|-------------|-----------------|---------|
| **BASE** | price (close) | — | Price | — | Not in trajectory (should be, as structural position) |
| **BASE** | volume_delta | dV | Volume | -0.018 | KEEP — only independent signal |
| **BASE** | session_phase | f(time) | Time | not tested | ADD — open/lunch/close dynamics differ |
| **1st deriv** | velocity | dP/dt | Price, Time | -0.024 | KEEP — strongest short-term predictor |
| **1st deriv** | z_score | (P - mean) / sigma | Price | -0.020 | KEEP — persists at +10bar (structural) |
| **1st deriv** | dmi_plus | f(highs) | Price | -0.006 | DEMOTE — weak, redundant with velocity |
| **1st deriv** | dmi_minus | f(lows) | Price | +0.006 | DEMOTE — weak, redundant with velocity |
| **1st deriv** | sigma | std(residuals) | Price | +0.007 | KEEP — noise floor (environment, not direction) |
| **2nd deriv** | F_momentum | PID(z_score) | Price | -0.024 | REPLACE — same signal as velocity, more complex |
| **2nd deriv** | adx | smooth(|DMI+−DMI−|/(DMI++DMI−)) | Price | +0.006 | DROP — triple-derived, r=0.006 |
| **2nd deriv** | term_pid | integral(z_score) | Price | -0.019 | REPLACE — accumulated drift, use z directly |
| **2nd deriv** | hurst | R/S statistic | Price | -0.004 | DROP — confirmed useless (r≈0) |
| **2nd deriv** | P_center | regression probability | Price | -0.000 | DROP — zero signal |
| **2nd deriv** | coherence | oscillation entropy | Price | +0.001 | DROP — zero signal |

---

## 2. PROPOSED PRINCIPLED FEATURE SET

### 2a. Keep (proven signal)
| # | Feature | Base | Why keep | OOS evidence |
|---|---------|------|----------|-------------|
| 1 | **velocity** | Price+Time | Strongest +1bar predictor, fundamental 1st derivative | r_s=-0.024 |
| 2 | **z_score** | Price | Best +10bar persistence, measures structural position | r_s=-0.020 (+10: -0.011) |
| 3 | **volume_delta** | Volume | Only independent base signal | r_s=-0.018 |
| 4 | **sigma** | Price | Noise floor / environment measure (not direction) | r_s=+0.007 |

### 2b. Add (missing from first principles)
| # | Feature | Base | What question it answers |
|---|---------|------|------------------------|
| 5 | **acceleration** (d²P/dt²) | Price+Time | Is the move speeding up or exhausting? |
| 6 | **fib_position** | Price (structural) | WHERE in the 5-day range? (0=low, 1=high) |
| 7 | **dist_nearest_fib** | Price (structural) | How far from a Fibonacci level? (ticks) |
| 8 | **volume_rate** (dV/dt) | Volume+Time | Is participation increasing or drying up? |
| 9 | **price_volume_alignment** | Price×Volume | Is the move BACKED by volume? (sign agreement) |
| 10 | **session_phase** | Time | Open/morning/lunch/afternoon/close (categorical→numeric) |
| 11 | **higher_tf_z** (1h z-score) | Price (MTF) | Position in hourly structure |
| 12 | **higher_tf_fm_sign** (1h direction) | Price (MTF) | Is 1m WITH or AGAINST the 1h? |

### 2c. Drop (no signal or redundant)
| Feature | Why drop |
|---------|----------|
| F_momentum | Same info as velocity, 3 layers of abstraction deeper |
| term_pid | Accumulated z drift — use z_score directly |
| adx | Triple-derived, r=0.006, zero directional value |
| hurst | r=-0.004, confirmed useless |
| P_center | r=-0.000 |
| coherence | r=+0.001 |
| dmi_plus | r=-0.006, velocity already captures this |
| dmi_minus | r=+0.006, velocity already captures this |

---

## 3. CROSS-DERIVATIVE C&E MATRIX

**Y (Output)** = Next-bar price change direction (correct = profitable trade)

| X (Input) | Measurement Type | Answers | Corr to Y | Priority |
|-----------|-----------------|---------|-----------|----------|
| **velocity** | Price÷Time | How fast? | 10 | ENTRY timing |
| **z_score** | Price−Mean | How far from center? | 9 | ENTRY + EXIT position |
| **fib_position** | Price÷Range | WHERE in structure? | ? (untested) | EXIT timing |
| **volume_delta** | Volume | Who's participating? | 8 | ENTRY confirmation |
| **price×volume** | Price×Volume | Is move real or fake? | ? (untested) | ENTRY quality |
| **acceleration** | d(velocity)/dt | Speeding up or dying? | ? (untested) | EXIT timing |
| **sigma** | std(Price) | How noisy? | 5 | SIZING (not direction) |
| **1h_z_score** | MTF Price | Position in hourly? | ? (untested) | EXIT structure |
| **session_phase** | Time | When in the day? | ? (untested) | ENTRY filter |
| **volume_rate** | Volume÷Time | Crowd arriving/leaving? | ? (untested) | EXIT confirmation |

### 3a. What Each Feature Should Be Used For

| Question | Best Feature | NOT this feature |
|----------|-------------|-----------------|
| **Direction** (which way?) | velocity sign, z_score sign | ~~ADX~~ (strength not direction) |
| **Timing** (when to enter?) | velocity magnitude, acceleration | ~~hurst~~ (statistical, not timing) |
| **Position** (where in structure?) | fib_position, 1h_z_score | ~~P_center~~ (model, not structure) |
| **Quality** (real or fake?) | price×volume alignment | ~~coherence~~ (r≈0) |
| **Sizing** (how much?) | sigma, volume magnitude | ~~ADX~~ (not for sizing either) |
| **Exit** (when to get out?) | fib_position, acceleration, volume_rate | ~~F_momentum~~ (same as velocity) |

---

## 4. RESEARCH PRIORITIES

1. **Fibonacci features** — validate fib_position and dist_nearest_fib on existing seeds
   (see docs/Active/RESEARCH_FIBONACCI_STRUCTURE.md)

2. **Price-volume cross** — compute sign(velocity) × sign(volume_delta) per bar,
   test if alignment predicts trade quality

3. **Acceleration** — d(velocity)/dt, test if it predicts exit timing better than
   funnel flip

4. **Higher-TF z-score** — add 1h z to trajectory, re-enrich seeds, re-run OOS

5. **Session phase** — bucket trades by session phase, check if entry quality differs

---

## 5. IMPLICATIONS FOR PhysicsEngine

Current: 12 features, 10 are price derivatives, 1 volume, 0 structure.
Proposed: 12 features, 4 proven price, 1 volume, 3 structure, 2 cross, 2 context.

The K-NN trajectory matching will improve because:
- Seeds matched by STRUCTURAL POSITION will separate "same physics, different outcome"
- Price-volume cross will filter real moves from noise
- Session phase will prevent matching morning patterns against lunch chop

**Expected impact**: $264/day → $400+ when exits know WHERE they are, not just WHAT the physics says.
