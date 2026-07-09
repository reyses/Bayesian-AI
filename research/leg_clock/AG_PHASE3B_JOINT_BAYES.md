# AG Phase 3B — Joint Bayesian evidence model (supersedes per-concept verdicts)

**User direction (Moises, 2026-07-09):** this is NOT a per-concept
if-this-then-that sweep. ALL NT8 signals are evidence terms in ONE Bayesian
likelihood framework. The deliverable is a per-bar POSTERIOR of the response,
built from all signals jointly — the confluence is the object of study.

## Architecture

1. **Event layer (per bar, all concepts at once)**: every catalog concept
   becomes a boolean/graded event flag computed causally — pivot touch, VWAP
   pullback, APZ touch, exhaustion wick, momentum exhaustion, structure
   break, round-number proximity, squeeze state, ADX regime, divergence
   flags, POC/VA position, etc. One row per bar, one column per signal.
2. **Response (unchanged from the agreed definition)**: first-touch ±k·σ
   (k ∈ {1,2}), symmetric barriers, direction pre-registered per concept.
3. **Per-signal likelihood tables** (kept, but demoted to building blocks):
   P(response | signal) vs matched + phantom nulls — for interpretability
   and for catching degenerate flags. NOT the verdict layer anymore.
4. **The joint model (the actual deliverable)**:
   - Fit a calibrated combiner on ALL flags simultaneously — regularized
     logistic (a discriminative Bayesian update) is the default; report each
     signal's weight = its evidence contribution GIVEN the others.
   - **Correlation is the enemy**: signals co-fire (pivot+VWAP+APZ often the
     same bar). Do NOT multiply marginal likelihood ratios (naive Bayes) —
     that triple-counts shared evidence and manufactures fake confluence.
     Report the signal-correlation matrix alongside the weights.
   - Train 2024 → evaluate 2025 untouched.
5. **Evaluation of the posterior** (in order):
   a. **Calibration**: predicted posterior vs realized frequency, by decile
      (a Bayesian model that says 70% must be right ~70% of the time).
   b. **Tier separation**: response rate in top posterior tiers vs base +
      shuffle-label null. Signal bar: top-tier lift ≥ +10pp REAL.
   c. **Marginal-contribution ranking**: which signals actually move the
      posterior (drop-one-out deltas), i.e., which of the 14 concepts carry
      unique information vs redundant echoes.
   d. **ONE economics test** at the end: trade the top posterior tier only
      (agreed exits, 4t costs, day-block CI, both years). This test — not
      any AUC/calibration number — decides if anything ships.

## Unchanged hard rules
Sigma-relative everything; causal/trailing extraction only (FPS discipline);
label-free; matched + phantom nulls at the per-signal layer; day-block CIs;
both-years; NT8 sealed; delta-from-OHLCV must be disclosed as a proxy.

## Why this supersedes Phase 3-as-proposed
Standalone verdicts measure each whisper alone — we already know most
whispers are <5pp alone. The user's thesis (and his manual method) is that
the EDGE lives in the joint state. The joint model measures exactly that,
while the per-signal tables + drop-one-out ranking still tell us which
concepts earn their place.
