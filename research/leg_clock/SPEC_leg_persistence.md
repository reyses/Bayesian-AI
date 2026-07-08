# SPEC — Leg Persistence Model (fade vs ride)

Agreed with Moises 2026-07-07 under the "agree the spec before executing" rule.
Nothing runs until this is signed off.

## Objective
From CAUSAL features inside a running leg, predict how much longer it lasts —
to classify **fade** (stops soon) vs **ride** (runs). Continuous, not binary.

## Legs
Fixed-reversal zigzag on 1m closes. Threshold ~150 ticks (Moises' macro scale,
~a handful of legs/day). [PARAM to confirm]

## Features (all causal — measured at sample points INSIDE each leg)
1. Elapsed so far: duration (min) + extent (ticks).
2. **Velocity so far** = trailing regression slope (lagged, causal — the cubic /
   regression-mean slope). NOT the hindsight full-leg velocity.
3. **Volume**: running / relative to session.

## Target
- Primary: remaining **DURATION** (time to leg end).
- Secondary: remaining **EXTENT** (price).
Split on purpose — the thesis is about duration, not price delta.

## Thesis under test
A fast / violent leg is mostly shorter in DURATION (fades soon) — not
necessarily smaller in price delta. A slow grind runs longer (ride). Volume
conditions persistence.

## Protocol (honesty rules)
- Train 2024, test **2025 OOS**.
- Continuous prediction (regression), reported as distributions / mode — not a
  single point.
- Two baselines it must beat: (a) elapsed-only, (b) shuffle null. Velocity +
  volume must ADD over elapsed-alone, or they're noise.
- Judge by whether confidence tiers SEPARATE outcomes, not one R²/AUC number.
- Causal features only. No hindsight leg velocity.

## Out of scope (for this test)
Trading/PnL. This measures whether the SIGNAL exists. A backtest comes only
after the signal clears the baselines.

## Sign-off
- [ ] Moises approves → then execute.
