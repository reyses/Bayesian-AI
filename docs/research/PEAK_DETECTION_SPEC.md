# Peak Detection Research Spec

**Date**: 2026-03-17
**Status**: In Progress
**Goal**: Detect trade peaks in real-time to optimize exit timing

---

## What We Know (Data-Validated)

### 1. Quantum State Changes at Peak (IS, 500 trades)
At peak → exit, the MarketState shifts predictably:
- **P_center**: 0.38 → 0.53 (+0.15) — wave function shifts toward mean reversion
- **F_momentum**: -43.8 → 0.0 (+43.8) — kinetic energy dies completely
- **F_reversion**: +1.25 → -0.04 (-1.30) — reversion force exhausts
- **z-score**: -0.70 → +0.02 (+0.72) — price reverts toward center
- **ADX**: 30.7 → 28.9 (-1.7) — trend weakens
- **entropy**: 0.61 → 0.62 (+0.02) — slight increase in chaos

**Implication**: The physics changes BEFORE price fully reverses. P_center
and F_momentum are leading indicators of peak exhaustion.

### 2. Volume at Peak (IS June, 264 trades at 15s, 297 at 1s)

**15s resolution:**
| Volume Pattern | Trades | WR | $/trade |
|---------------|--------|-----|---------|
| Q1 collapse (after < 50%) | 92 | 88.0% | $21.24 |
| Normal (50-150%) | 134 | 73.9% | $12.08 |
| Spike (after > 150%) | 38 | 55.3% | $4.93 |

**1s resolution:**
| Volume Pattern | Trades | WR | $/trade |
|---------------|--------|-----|---------|
| Q1 collapse | 75 | 84.0% | $15.79 |
| Q2 | 74 | 74.3% | $11.81 |
| Q3 | 74 | 71.6% | $10.70 |
| Q4 spike | 74 | 78.4% | $15.51 |

**Key finding**: Volume collapse after peak = best trades (84-94% WR).
The move spent its energy. Volume spike after peak = danger (55% WR,
someone pushing against you).

**Optimal TF for volume**: 1m (institutional flow). 1s is too noisy
(tick-level fills). 15s works but 1m gives cleaner signals.

### 3. Volume by Z-Score Zone (1m, June)
| Zone | Avg Volume | Avg ADX |
|------|-----------|---------|
| Center (|z| < 0.5) | 230 | 23.3 |
| Approach (0.5-1.5) | 260 | 24.0 |
| Roche (1.5-2.5) | 374 | 25.1 |
| Event horizon (2.5+) | 713 | 25.3 |

Event horizon = 3× volume of center. ADX barely changes — volume
is the leading signal, ADX lags.

### 4. ADX-Volume Correlation
- Spearman r = +0.13 (p < 0.001) — significant but weak
- ADX > 30 in 29% of high-volume bars vs 16% of low-volume bars
- Ohm's law analogy: Trend = Volume × Resistance (conceptual, not proven)

### 5. Compression → Spike → Collapse (C-S-C) at 1m
- 351 C-S-C patterns found in June
- 61.2% preceded a reversal (213/348)
- Avg reversal magnitude: 37.9 ticks
- Avg continuation magnitude: 58.9 ticks
- At 1s: pattern is noisier (70.6% WR, worse than without)

### 6. Fake vs Real DMI Crosses (1m, 2,673 crosses)
- Only 37.3% of DMI crosses are real (sustained 20 bars)
- Real crosses: vol_at = 111, gap = 2.69
- Fake crosses: vol_at = 99, gap = 2.11
- High volume ratio at cross: 39.7% real vs 35.0% for low ratio
- Signal exists but weak — needs combination with other features

---

## What We DON'T Know (Needs Proper Math)

### 1. The Giveback Threshold Distribution
Current: Brownian `2σ√N` — hand-waved.
**Needed**: Fit the actual distribution of drawdown-from-peak for IS trades.
What percentile of normal drawdowns does 2σ√N correspond to?
Is the distribution symmetric, fat-tailed, or skewed?
Method: Kernel density estimation on `(peak - current) / peak` for each
bar after the peak. Find the 95th percentile of "normal" giveback.

### 2. The State-Based ePnL Formula
Current: `remain = (room/2) × trend × conviction × alignment × momentum`
**Needed**: Logistic regression or gradient boosting on:
  X = [z_score, hurst, conviction, alignment, F_momentum, P_center, entropy]
  y = is_still_profitable_10_bars_later
Coefficients from data, not hand-tuned multiplicative weights.

### 3. Volume Signal Statistical Test
Current: quartile buckets showing WR differences.
**Needed**: Kolmogorov-Smirnov test on volume distributions before/after peak
for wins vs losses. Permutation test for significance. Effect size (Cohen's d).

### 4. The Inflection Point Model
Current theory: inflection is at z=0 (center) or z=±3σ (Roche limit).
At z=0: outcome depends on F_momentum magnitude.
**Needed**: Map P(reversal | z, F_momentum, volume) empirically.
At what z-score × momentum × volume combination does the reversal
probability exceed 50%? This is a 3D surface, not a 1D threshold.

### 5. Normality of Volume Distribution
Shapiro-Wilk: 97% of trade volume sequences are non-normal.
**Needed**: What distribution DO they follow? Log-normal? Pareto?
This affects which statistical tests are valid for detection.

---

## The Train of Thought: How to Measure Peaks

### Observation 1: Peaks are inflection points
A peak is where price transitions from "moving in trade direction" to
"not moving in trade direction." This is a phase transition, not a
discrete event.

### Observation 2: You can't detect the peak AT the peak
At the exact peak, the trade looks maximally profitable. The signal
that the peak has passed comes AFTER — when the state starts changing.

### Observation 3: What changes after the peak
From the replay data (500 IS trades):
1. P_center rises (probability shifts toward center)
2. F_momentum collapses (kinetic energy → 0)
3. Volume drops (no new orders pushing the move)
4. z-score starts reverting toward 0

### Observation 4: z = 0 is the uncertainty maximum
At z = 0, all three wave function probabilities are roughly equal.
The system can't predict what happens next from state alone.
Volume is the tiebreaker:
- High volume at z=0 → price shoots through (cascade)
- Low volume at z=0 → PID catches it (chop/reversal)

### Observation 5: The fuse analogy (Ohm's law)
Price approaching a structural level with:
- Low volume (low current) → fuse holds → bounce/reversal
- High volume (high current) → fuse blows → breakthrough/cascade
Volume tells you BEFORE the event which outcome is likely.

### Observation 6: The compression-spike-collapse sequence
1. Volume compresses (liquidity thinning — demi-gods pulling bids/offers)
2. Volume spikes (the wall is hit — stops trigger)
3. Volume collapses (energy spent — move is done)
This is the fuse blowing sequence. Detectable on 1m, too noisy on 1s.

### Observation 7: Real vs fake reversals
- Real reversal (brick wall): high volume at the turn, DMI gap > 2.5,
  ADX > 20. The gods changed direction.
- Fake reversal (FOMO): low volume at the turn, DMI gap < 2.0,
  ADX < 20. Retail panic, demi-gods will correct.

### What's Missing
- The rate of change of P_center and F_momentum (derivative, not level)
- A combined score that weights volume + state changes + z-score position
- Calibration: what threshold on the combined score = "exit now" vs "hold"
- I-MR on the trade's own bar series (detect regime shift without assumptions)

---

## Next Steps (Ordered)

1. **I-MR trade replay** — run I-MR on each trade's per-bar price path.
   Detect out-of-control bars objectively. Compare I-MR exit bar vs
   actual exit bar vs optimal (oracle peak bar).

2. **Fit giveback distribution** — compute empirical drawdown-from-peak
   distribution. Set thresholds at data-derived percentiles, not σ√N.

3. **dP_center/dt as exit signal** — compute rate of change of P_center
   during trades. When dP_center/dt > threshold = peak passing. Needs
   per-bar quantum state from trade replays.

4. **Volume collapse as exit modifier** — add 1m volume check to giveback.
   If vol_after < 50% of vol_before = tighten. If vol_after > 150% =
   tighten aggressively (counter-force entering).

5. **3D inflection surface** — fit P(reversal | z, F_momentum, 1m_volume).
   This replaces all the separate threshold checks with a single
   probability surface.
