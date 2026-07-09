# FABLE-5 Review — AG Phase 3 (Bayesian Probabilistic Response Sweep)

**Verdict: approved WITH four mandatory amendments.** The reframe is correct —
the articles describe contextual modifiers, not systems, and P(response|event)
is the right object (it is the "existence test" of the original template).
But the design as proposed will manufacture a false positive we have already
killed once. Amendments below are non-negotiable.

## Answer to the review question
First-Touch Probability (target before stop) = YES, aligned — with:

### Amendment 1 — Barriers in sigma, not points
±10 fixed points violates the sigma-relative rule (user-corrected 2026-07-07).
Use symmetric ±k·σ barriers (σ = trailing regression residual or ATR;
k ∈ {1, 2} reported both). Symmetric barriers also give the clean
random-walk reference of 0.5.

### Amendment 2 — MATCHED null, not the unconditional base rate (critical)
Comparing against the unconditional P(target-before-stop) confounds
time-of-day and volatility regime: events cluster in specific conditions.
Required controls:
- **Time-matched null**: random non-event bars, same day + same hour.
- **Phantom-geometry null** for every price-level touch concept (pivot, VWAP,
  APZ, POC): the same "touch" logic on a level jittered ±(4–16)·ticks (or
  sigma-scaled). We measured this exact trap: ANY nearby line "bounces" ~60%
  at short scale — real band levels scored IDENTICAL to phantom lines
  (research/level_hold/, 63 days). An APZ-touch bounce probability without
  the phantom control is mean-reversion base rate wearing a costume.
The reported number is the DELTA vs the matched null, not vs 0.5.

### Amendment 3 — Pre-registered direction per concept
Each event's expected response direction (bounce vs break) is declared in the
report BEFORE measuring, from the article's own claim. No post-hoc sign
picking — with 10+ concepts, sign-fitting doubles the effective comparisons.
If the measured delta has the OPPOSITE sign of the article's claim, that is a
finding (report it), not a license to flip the hypothesis on the same data.

### Amendment 4 — Event hygiene + honest CIs
- Debounce: one event per episode (a touch persisting across N consecutive
  bars = one event; new event only after price leaves the zone by > barrier/2
  or after ≥5 min).
- CI on the delta via DAY-BLOCK bootstrap (events within a day are
  correlated; bar-level bootstrap overstates precision).
- Both years separately; a concept is a survivor only if the delta holds
  sign and magnitude on BOTH.
- Signal bar in probability units vs matched null: ≥ +10pp REAL,
  +5–10pp CONDITIONAL, < +5pp NOISE.

## Also
- **Slope-persistence re-run**: allowed — the exhaustion-at-extreme
  conditioning differs from the version we killed (unconditional bar-to-bar).
- **Delta divergence (ag_cat_09)**: flag your data source. There is no
  bid/ask/tick history in ATLAS; if you are deriving "delta" from OHLCV, say
  exactly how, and label it a proxy. Proxy-delta divergence ≠ the article's
  concept.
- **Endgame honesty**: a surviving probability shift is a FEATURE for the
  combination phase, not a strategy. Deltas do not add linearly; the combined
  stack gets ONE economics test at the end (costs, CI), and that test — not
  the sweep — decides if anything ships.
- Keep NT8 sealed. Keep it label-free. Keep FPS/trailing-only extraction.
