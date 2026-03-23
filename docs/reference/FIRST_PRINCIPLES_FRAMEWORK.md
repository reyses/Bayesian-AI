# First Principles Framework — Base Measurements & Market Physics
> Session: 2026-03-22 (overnight into 03-23)
> Context: After deploying PhysicsEngine to live sim, deep analysis of WHY features
> work or don't. Rebuilt the entire measurement framework from scratch.

---

## 1. The Problem

PhysicsEngine has 12 features: fm, z, dmi_p, dmi_m, adx, vel, vol, hurst, P_center,
coherence, sigma, pid. OOS result: $264/day, 61% WR.

But 10 of 12 features are transformations of price predicting price. That's circular.
And the engine is not price-aware — same physics at the top of a range vs bottom of
a range produces the same match, even though outcomes are opposite.

The question: what SHOULD we be measuring?

---

## 2. Base Measurements

Only 3 independent observables exist from the market:

| Base | What it measures | Independent? |
|------|-----------------|-------------|
| **Price** | Where the market is | YES |
| **Time** | When things happen | YES |
| **Volume** | How much participation | YES |

Everything else — every indicator, every oscillator, every "signal" — is a
transformation of these three. The question is: how many layers of transformation
before the base signal is buried?

---

## 3. Derivation Levels

### Grounded (1 transparent step from base)
- **velocity** = dP/dt (Price + Time) → how fast?
- **z_score** = (P - mean) / std (Price) → how far from center?
- **dmi_diff** = DMI+ - DMI- (Price highs vs lows) → who's winning?
- **volume_rate** = dV/dt (Volume + Time) → participation changing?
- **acceleration** = d²P/dt² (Price + Time) → speeding up or dying?

### Distribution (measures the SHAPE of base data)
- **std(price changes)** → realized volatility (quiet vs wild)
- **std(volume)** → flow consistency (institutional = steady, retail = spiky)
- **variance ratio** = var(short window) / var(long window) → trending or reverting?

### Over-abstracted (3+ layers, machine-specific)
- **F_momentum** = PID(z_score) where z = (P-regression)/sigma → three layers deep
- **ADX** = smooth(|DI+ - DI-| / (DI+ + DI-)) → double-smoothed ratio of ratio
- **term_pid** = integral(z_score) → accumulated drift
- **oscillation_entropy_normalized** = 1/(1+std(z, 5)) → inverted std of a derivative

The over-abstracted features encode our system's tuning constants (PID gains,
regression windows, smoothing periods), not market properties.

---

## 4. The DOE Principle

From injection molding: when doing a rheology study, you measure **material
properties** (viscosity, shear rate, temperature) — not **machine readings**
(output pressure, screw RPM). Material properties transfer across machines.
Machine readings are specific to one setup.

Applied to trading:
- **velocity** (dP/dt) = material property. Works on MNQ, ES, NQ, any instrument.
- **F_momentum** (PID with kp/ki/kd over regression window=200) = machine reading.
  Change the PID constants, change the window, change the instrument → different values.

**Test**: "Would this feature mean the same thing on ES as on MNQ?"
If no → it's machine-specific → it doesn't belong in the base feature set.

Every feature must have a one-sentence explanation:
"This measures X of Y over Z window" where X = statistical operation, Y = Price/Time/Volume,
Z = defined window. If you can't explain it without reading the code, it's suspect.

---

## 5. What Is Volatility?

**std(dP/dt)** — standard deviation of velocity. Distribution of a first derivative.

At different timeframes it measures different things:
- **std(dP, 1h)** = trend amplitude (the structural envelope)
- **std(dP, 1m)** = noise amplitude (the tradable oscillations)
- **std(dP, 1s)** = noise floor (our measurement limit)

One timeframe's signal is the next timeframe's noise. The ratio between scales
IS the fractal state measurement:
- std(1m) / std(1h) = noise-to-trend ratio = how much is tradable chop vs push

---

## 6. What Is Noise?

Price movement that carries no information about future direction. The random
component — the part that IS Brownian.

But noise depends on your timeframe:
- At 1s: almost everything is noise (bid-ask bounce, single fills)
- At 1m: 1s noise averages out, what remains is tradable mean-reversion
- At 1h: 1m oscillations are noise, what remains is structural trend

**1s is our noise floor** — the resolution limit. Below that we can't see.
`std(dP, 1s, rolling)` is the irreducible randomness at our finest resolution.
When it changes, the regime changed.

---

## 7. The Variance Ratio = Brownian Motion Test

Brownian motion is the **null hypothesis** of price movement: random, independent,
normally distributed steps. It became the foundation of quant finance because
Black-Scholes (1973) assumes it, and the math is clean.

But markets are NOT Brownian: fat tails, volatility clustering, mean reversion,
momentum. Every profitable trade is the market being NOT random.

The variance ratio measures exactly this:
- **Ratio = 1** → random walk (Brownian), no edge, don't trade
- **Ratio > 1** → trending (super-diffusive), ride it
- **Ratio < 1** → mean-reverting (sub-diffusive), fade it

This replaces BOTH:
- **Hurst exponent** (R/S rescaled range statistic — ungrounded, r=-0.004)
- **ADX** (double-smoothed directional ratio — 5 layers of abstraction)

Same question, one step from price.

---

## 8. Why ADX Became Standard

Wilder published it in 1978. It was **first**, not best.

What ADX computes:
1. Directional Movement from bar highs/lows (+DM, -DM)
2. Normalize by True Range
3. Smooth with 14-bar Wilder EMA → DI+, DI-
4. Ratio: |DI+ - DI-| / (DI+ + DI-) → DX
5. Smooth DX with ANOTHER 14-bar EMA → ADX

Five steps. Double-smoothed ratio of a ratio.
Variance ratio answers the same question in one step.

ADX is standard because it got built into every platform and traders memorized
the rules (>25 = trend, <20 = chop). Not because it's optimal.

---

## 9. Coherence Decoded

The old `oscillation_entropy_normalized` was NOT entropy and NOT oscillation.

**Actual formula**: `1 / (1 + std(z_scores, window=5))`

It's inverted short-term volatility of z-score. Misnamed from the quantum era.
OOS correlation: r=0.001 — confirmed zero signal as a standalone feature.

**Grounded coherence** (rebuilt from first principles):
Count how many directional features agree:
- velocity sign
- dmi_diff sign
- volume sign
- z_score sign
- higher_tf direction

5/5 agree → high coherence → move is consensus, already priced in, late entry.
3/5 agree → disagreement → edge forming.
<3 agree → no signal.

Prior research validated this inverted: "TF disagreement = better trades (r=-0.096)."
Disagreement IS the signal — the base measurements are fighting, and the winner
of that fight is the trade direction.

---

## 10. DMI — Grounded but Different from Velocity

DMI measures the **buyer/seller battle** from bar highs and lows.
Velocity measures **center speed** from close to close.
They're both first derivatives of price, but from DIFFERENT components.

Individually weak: r=0.006. But DMI extreme + volume collapse = **exhaustion signal**.
This is the reversal trigger. The cross-signal is the value, not DMI alone.

DMI cross (DI+ crosses DI-) is too late — the reversal already happened.
DMI extreme (one side dominant) + volume drying up = exhaustion BEFORE the cross.

---

## 11. Entropy and Free Energy

- **Entropy** (S) = disorder/randomness. Tends to increase.
- **Free energy** (G = H - TS) = energy available to do work. Decreases as entropy rises.

Applied to markets:
- High entropy = random walk, Brownian motion, no pattern → no profit extractable
- Low entropy = ordered state (trending or reverting) → edge exists, profit extractable
- Free energy = deviation from Brownian = the profit opportunity

The variance ratio literally measures this: how far from maximum entropy
(random walk) is the market right now?

In true decoherence (maximum entropy, ratio ≈ 1):
- Velocity gives you nothing (no trend)
- Z-score gives you nothing (no mean to revert to)
- DMI exhaustion × volume is the ONLY extractable signal — and it's thin

| Market State | Variance Ratio | Edge Source | Difficulty |
|-------------|---------------|-------------|------------|
| Trending | > 1 | velocity, z_score | Easy |
| Mean-reverting | < 1 | z_score, fib_position | Moderate |
| Decoherent | ≈ 1 | DMI × volume only | Hard, thin |

---

## 12. The Market Does NOT Have Normality

Fat tails. Extreme moves happen orders of magnitude more often than normal
distribution predicts. 1987 crash = 20+ sigma event (shouldn't happen once
in the universe's lifetime under Gaussian assumptions).

- Center (~90% of bars): roughly normal → features work here
- Tails: power law → this is where you get killed
- SL is for the tails. Features are for the center.

This IS the Nightmare Protocol: trade the normal center with the features,
protect against the fat tails with the SL. Two separate problems, two
separate solutions.

---

## 13. Liquidity Is a Latent Variable

**Liquidity** = orders waiting to execute at a price. Not volume (that's
transactions already completed). Liquidity is what's SITTING there, unfilled.

**Can't measure directly.** The order book is live — historical data doesn't
capture "500 contracts were sitting at 24,000 at 10:15 AM." Once filled or
cancelled, gone.

**Can measure its effects:**
1. **Volume at price** (Volume Profile) — if lots traded there, orders were there
2. **Price rejection** — price bounces off a level = something absorbed the move
3. **Time at price** (TPO) — price spent time there = accepted value = liquidity
4. **Repeat reaction** — bounces off same level 3x = orders keep refilling

**Fibonacci levels** are a cheap proxy: "given the range, where are orders
LIKELY sitting?" Volume profile is the grounded version: "where did orders
ACTUALLY sit?"

---

## 14. Support/Resistance = Algos Waiting to Cash Out

The "levels" aren't mystical. They're price points where algorithms have
standing orders. Why they persist over months:

- Institutional algo bought at 23,800 → that level is programmed
- When price returns, the algo is there again — same level, same orders
- Hundreds of systems see the same historical volume node → clustering
- The wall SELF-REINFORCES across time because the strategies don't change

**Level lifecycle:**
1. Fresh → just created, untested, full liquidity
2. Confirmed → tested and held, orders refilled (strongest)
3. Weakened → tested multiple times, partially absorbed
4. Broken → price pushed through, liquidity consumed
5. Flipped → old support becomes resistance (new orders from other side)

Levels from 2-3 months ago get retested because the algos that placed
them are still running. The gravity source is persistent.

---

## 15. Three-Body Problem

At any moment, price is pulled by three forces simultaneously:

1. **Momentum** — inertia of the current move (committed buyers/sellers)
2. **Mean reversion** — pull toward fair value (regression center)
3. **Liquidity walls** — invisible mass at price levels (algos waiting)

The three-body problem has no analytical solution. You can't write an equation
that predicts where price goes. That's why the Nightmare Protocol was
overengineered — it tried to solve analytically (Ornstein-Uhlenbeck + PID +
Lyapunov). The market doesn't have a closed-form solution.

The K-NN approach accepts this and sidesteps it: don't solve the equations,
match current observable state against 38K historical cases where you know
what happened next. Numerical solution instead of analytical.

---

## 16. The Nightmare Protocol Was This Framework

The original Nightmare Protocol (Feb 2026) had the right structure:
- Stable regime (λ < 0) = mean-reverting → fade the edges
- Chaotic regime (λ > 0) = trending → ride the breakout
- Lyapunov exponent λ determines the regime

λ IS the variance ratio. λ < 0 = ratio < 1. λ > 0 = ratio > 1.
The "Singularity" IS the fat tail event. The "Roche Limit" IS the
band extreme where probability density approaches zero.

Same framework. Just buried under physics metaphors (gravitational fields,
event horizons, quantum states) that made it impossible to debug when
something broke. The grounded version uses the same math without the
metaphors.

---

## 17. Proposed 13-Feature Set

| # | Feature | Category | Base | Question |
|---|---------|----------|------|----------|
| 1 | velocity | grounded deriv | Price+Time | How fast? |
| 2 | z_score | grounded deriv | Price | How far from center? |
| 3 | acceleration | grounded deriv | Price+Time | Speeding up or dying? |
| 4 | dmi_diff | grounded deriv | Price (H/L) | Who's winning (buyers vs sellers)? |
| 5 | std_price | distribution | Price | Quiet or wild? |
| 6 | std_volume | distribution | Volume | Steady or spiky flow? |
| 7 | variance_ratio | distribution | Price | Trending or reverting? |
| 8 | fib_position | structural | Price | WHERE in range? (proxy for liquidity) |
| 9 | higher_tf_z | structural | Price (MTF) | Position in hourly structure? |
| 10 | session_phase | structural | Time | When in the day? |
| 11 | volume_delta | cross | Volume | Who's participating? |
| 12 | price×volume | cross | Price×Volume | Is the move real? |
| 13 | dmi×vol_exhaustion | cross | Price×Volume | Reversal coming? |

4 categories. 3 independent bases. Each feature answers ONE question.
No redundancy. Every feature explainable in one sentence.

---

## 18. What We're Really Building

A numerical three-body solver that:
1. Measures the market with grounded instruments (not our system's artifacts)
2. Detects deviations from Brownian motion (where the edge lives)
3. Classifies the regime (trending / reverting / decoherent)
4. Matches observable state against historical outcomes (K-NN)
5. Protects capital against fat tails (SL for the non-normal extremes)

The features measure the PROCESS (market), not the EQUIPMENT (our system).
They transfer across instruments, timeframes, and regimes. When something
breaks, you can trace it to which base measurement changed.

That's the DOE principle applied to trading: ground your factors in base
measurements, and the experiment transfers.
