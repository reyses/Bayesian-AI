---
name: Feature tree — 3 levels max, organic expansion like colors
description: Features built organically from 2 primaries (Price, Time) through 3 layers max. Each feature must name its parent measurements and the question it answers.
type: project
---

## Primaries (can't be derived)
- **Price** — where the market is
- **Time** — when

## Secondary (one operation on primaries)

**Kinematic** (mix two primaries):
- velocity = dPrice/dTime (how fast?)
- DMI diff = directional movement from Price highs vs lows (who's winning?)
- volume = market reaction to Price at Time (the mass — not truly independent)

**Statistical** (characterize one primary's distribution):
- mean(Price, window) = center
- std(Price, window) = spread / volatility
- median, skew, kurtosis — shape descriptors

## Tertiary (mix secondaries, or primary + secondary)

- acceleration = dVelocity/dTime (is force changing?)
- z-score = (Price - mean) / std (how unusual is this price?)
- fib_position = (Price - low) / (high - low) (where in range?)
- variance_ratio = std(short) / std(long) (trending or reverting?)
- price × volume = is the move backed by participation? (heavy or light?)
- DMI × volume exhaustion = DMI extreme + volume collapse (reversal coming?)
- session_phase = f(Time) categorical (when in the day?)
- higher_tf_z = z-score at 1h scale (structural position)

## Rules
1. **3 levels max.** If a question can't be answered in 3 layers from Price and Time, it's the wrong question.
2. **Name the parents.** Every feature must trace to its parent measurements.
3. **Name the question.** If you can't state in one sentence what it measures, it doesn't belong.
4. **No brown paint.** Don't mix 5 things at once. Mix two, see what you get, then mix again.
5. **Volume is secondary, not primary.** It's the market's reaction to Price at Time — a mass/force measurement, not independent.
6. **Statistical features are secondary.** std, mean, variance characterize the SHAPE of a primary — one operation, one base.
7. **Test before adding level 4:** What question does it answer that level 3 can't? If none, you don't need it.

**Why:** Old system had level 4+ features (F_momentum = PID of z-score, coherence = entropy of z-score std) that answered questions nobody asked (r≈0). Features should expand organically like colors — primary → secondary → tertiary — each layer intentional.
