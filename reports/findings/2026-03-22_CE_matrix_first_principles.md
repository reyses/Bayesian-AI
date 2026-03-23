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

**Grounded derivatives** (1 transparent step from base):
| # | Feature | Formula | Base | What question it answers |
|---|---------|---------|------|------------------------|
| 5 | **acceleration** | d(velocity)/dt | Price+Time | Is the move speeding up or exhausting? |
| 6 | **volume_rate** | dV/dt | Volume+Time | Is participation increasing or drying up? |
| 7 | **price_volume_alignment** | sign(velocity) × sign(volume) | Price×Volume | Is the move BACKED by volume? |

**Statistical distribution** (std/variance of base measurements — NOT derivatives, these measure the SHAPE):
| # | Feature | Formula | Base | What question it answers |
|---|---------|---------|------|------------------------|
| 8 | **std_price_changes** | std(dP, window=20) | Price | Realized volatility — quiet vs wild? |
| 9 | **std_volume** | std(V, window=20) | Volume | Is flow steady (institutional) or spiky (retail)? |
| 10 | **variance_ratio** | var(dP, short=5) / var(dP, long=20) | Price | Trending (>1) vs mean-reverting (<1)? Replaces Hurst. |

**Structural position** (price in context, not just price):
| # | Feature | Formula | Base | What question it answers |
|---|---------|---------|------|------------------------|
| 11 | **fib_position** | (P − low_5d) / (high_5d − low_5d) | Price (structural) | WHERE in the 5-day range? (0=low, 1=high) |
| 12 | **higher_tf_z** | 1h z-score | Price (MTF) | Position in hourly structure? |
| 13 | **session_phase** | f(time_of_day) | Time | Open/morning/lunch/afternoon/close? |

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

| X (Input) | Category | Answers | Corr to Y | Priority |
|-----------|----------|---------|-----------|----------|
| **velocity** | grounded deriv (dP/dt) | How fast? | 10 | ENTRY timing |
| **z_score** | grounded deriv (P−mean)/σ | How far from center? | 9 | ENTRY + EXIT |
| **volume_delta** | base (Volume) | Who's participating? | 8 | ENTRY confirmation |
| **fib_position** | structural (P in range) | WHERE in structure? | ? | EXIT timing |
| **price×volume** | cross (Price×Volume) | Real move or fake? | ? | ENTRY quality |
| **acceleration** | grounded deriv (d²P/dt²) | Speeding up or dying? | ? | EXIT timing |
| **std_price_changes** | distribution (std of dP) | Quiet or wild? | ? | SIZING |
| **std_volume** | distribution (std of V) | Steady or spiky flow? | ? | ENTRY quality |
| **variance_ratio** | distribution (var short/long) | Trending or reverting? | ? | REGIME |
| **1h_z_score** | structural (MTF) | Position in hourly? | ? | EXIT structure |
| **session_phase** | base (Time) | When in the day? | ? | ENTRY filter |
| **volume_rate** | grounded deriv (dV/dt) | Crowd arriving/leaving? | ? | EXIT confirm |

### 3a. What Each Feature Should Be Used For

| Question | Best Feature | Category | NOT this feature |
|----------|-------------|----------|-----------------|
| **Direction** (which way?) | velocity sign, z_score sign | grounded deriv | ~~ADX~~ (strength not direction) |
| **Timing** (when to enter?) | velocity magnitude, acceleration | grounded deriv | ~~hurst~~ (ungrounded statistic) |
| **Position** (where in structure?) | fib_position, 1h_z_score | structural | ~~P_center~~ (model, not structure) |
| **Quality** (real or fake?) | price×volume alignment, std_volume | cross + distribution | ~~coherence~~ (r≈0) |
| **Regime** (trending or choppy?) | variance_ratio | distribution | ~~ADX~~ (triple-derived, obscured) |
| **Sizing** (how much?) | std_price_changes, volume magnitude | distribution | ~~hurst~~ (grounded alternative exists) |
| **Exit** (when to get out?) | fib_position, acceleration, volume_rate | structural + deriv | ~~F_momentum~~ (same as velocity) |

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

Current: 12 features — 10 ungrounded price derivatives, 1 volume scalar, 0 structure, 0 distribution.
Proposed: 4 categories grounded in 3 base measurements (Price, Time, Volume):

| Category | Count | Features |
|----------|-------|----------|
| Grounded derivatives | 4 | velocity, z_score, acceleration, volume_rate |
| Distribution (std/var) | 3 | std_price_changes, std_volume, variance_ratio |
| Structural position | 3 | fib_position, higher_tf_z, session_phase |
| Cross (Price×Volume) | 2 | volume_delta, price_volume_alignment |
| **Total** | **12** | |

Principle: derivatives are OK when grounded (1 transparent step from base).
Principle: std/variance measure the SHAPE of base measurements, not another transformation.
Principle: each feature answers exactly ONE question about the market.

The K-NN trajectory matching will improve because:
- **Distribution features** tell the K-NN about the ENVIRONMENT (quiet/loud, trending/reverting)
- **Structural features** separate "same physics, different outcome" by WHERE in the range
- **Cross features** confirm whether moves are backed by participation
- **No redundancy** — 12 features, 12 different questions, 3 independent bases

**Expected impact**: $264/day → $400+ when the trajectory encodes environment + structure + confirmation, not just 10 ways of asking "what did price do?"
