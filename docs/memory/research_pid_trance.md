---
name: PID Trance at 1-sigma — HFT control loop traps price
description: At 1σ from regression mean, HFT algos maintain a PID control loop that traps price in a trance. Signals at 1σ are the control system maintaining equilibrium, not tradeable setups. Real signals are at 2σ+ where PID loses control.
type: project
---

## The Observation (Moises)

At 1σ distance from the regression mean, price gets caught in a **PID trance** —
the HFT market-making algorithms are actively controlling price within this band.

- **Kp (Proportional)**: Price at +1σ → algos sell proportionally → price reverts
- **Ki (Integral)**: Price stays at +0.5σ too long → accumulated error → algos push harder
- **Kd (Derivative)**: Price moves quickly toward 1σ → algos dampen the velocity (wicks)

The result: oscillation within ±1σ that looks like patterns but is actually the
control system maintaining equilibrium. Trading it = trading against the demi-gods.

## Connection to Scoring

If `term_pid` is high (strong PID control), signals at 1-2σ should be penalized
in P(success). If `term_pid` is low/zero (PID absent), signals at 1-2σ might be
early cascade indicators (demi-gods stepped away).
