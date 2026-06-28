# THESIS — Reward Design for a Direction-Exhaustion Trading Agent

Canonical reference for the reward/policy workstream. Captures the design and its rationale. Open knobs (density, architecture, weights) are flagged at the end — to be resolved against this frame, not before it.

---

## 1. Foundational premise — what is predictable
Three independent studies (order-flow ablation, 1s segment F-space, the cubic regression work) converge on one result: **forward price *magnitude* is ~unpredictable.** Every test against forward price returns ~zero R²; contemporaneous tests return a false ~1.0 (the identity trap).

What *is* predictable is **direction** and **exhaustion** — and of the two, exhaustion is the more detectable ("better at calling reversals from extended states than continuations").

**Therefore the reward credits *capturing a favorable directional move and exiting at exhaustion* — never price prediction.** This is the axiom the entire design rests on.

## 2. The oscillation substrate
Price oscillates with a measurable period (~7 bars / ~8 min at 1m). Two consequences:
- **Horizon = half the oscillation period.** The "available move" is one half-cycle, ending at the inflection. Predict to the next turn, not beyond.
- **The cubic regression is the analytic form of the thesis:** endpoint **slope = direction**, **curvature = exhaustion**, **inflection = the exit point**. Move shapes (V-reversal, inverted-V, monotonic, chop) are the structural vocabulary.

## 3. Core principle — process-based scoring, not outcome-based
Grade the **decision quality given the situation**, the way you'd grade a trader — not the dollar outcome. A wrong trade **cut fast** lost money but was a **good decision** → small *positive* score. This (a) matches how discipline is actually taught, and (b) gives a cleaner causal signal than raw PnL ("given this went against you, did you manage it well?").

## 4. The reward is a scorecard — three INDEPENDENT components
Mapped one-to-one onto the three-head architecture, each scored separately and **summed (weighted), never multiplied**:

| Component | Head | Question |
|---|---|---|
| **Selectivity** | trade / no-trade | Did you take the high-quality move and skip the wiggle? |
| **Direction** | direction | Of entries, correct side? |
| **Exit / Capture** | exit-timing | `capture_rate` vs the oracle move, with the cut-vs-ride asymmetry (§5). |

**Why additive, not a conditional funnel:** a multiplied chain (entry × direction × exit) lets a bad entry (×0) zero the whole reward, so the exit head gets **no gradient on trades it shouldn't have entered** — downstream skills starve. Additive components mean **every sub-skill trains every episode.** (The conditional funnel is kept — but only as a *diagnostic* and *curriculum* tool, §8.)

## 5. The exit head is the edge — asymmetric PnL
This is where profitability is *manufactured*, and it's likely the highest-value head.

- **Same action, opposite reward by outcome:** early exit on a **wrong** trade → reward (cut the loss); early exit on a **right** trade → penalty (killed capture). Exit is therefore a **discrimination problem** — the head must learn the **causal early signature** that separates "going wrong, cut now" from "developing, hold to inflection" (curvature flip, building adverse excursion, delta divergence).
- **Why it's the edge:** "cut losers short, let winners run" = asymmetric realized R/R = **positive expectancy even at sub-50% direction accuracy.** Since direction is barely predictable, you don't win by being right more often — you win by managing right/wrong trades **asymmetrically.** Mediocre direction + excellent exit asymmetry = profitable.

## 6. Invariant design properties (the load-bearing guarantees)

**Path-independence.** The reward has **no memory of being down** — each decision scored on its own merit, no cumulative "recover the deficit" term. This structurally **kills revenge-trading and bag-holding** at once (both are recovery-pressure pathologies). The agent isn't told "don't bag-hold" — it has *no reason to*.

**Oracle-anchored exit (no hope to hold onto).** Capture is measured against the **oracle's half-cycle move, which objectively ends at the inflection.** Once price inflects, the available move is *locked* — holding longer can't raise capture, only accrue reversal risk. A fresh favorable move is a **new oscillation = a new trade**, not this position recovering. **Hope has nothing to attach to** — the human's "will *this* recover?" is structurally answered "this move is done."

**Objective quality gate — MFE relative to MAE.** Opportunity is defined by **MFE/MAE** (or MFE under an MAE ceiling), not MFE alone — a big move that first went hard against you is a *bad* trade. Critically, **MFE/MAE is a property of the *move*, not your equity** → it is the *objective* R/R, with the equity-subjectivity isolated into position *sizing* (a separate decision). Threshold is **volatility-normalized** (ATR/SE units), not raw ticks, for regime-invariance.

**Regret lives on entry, gated on *knowable* quality.** Opportunity cost (anti-inaction) fires only for **missing a causally-readable, high-MFE/low-MAE setup** — zero regret for sitting out chop or a wiggle. Calibrated to **knowability, not hindsight** (punishing missed un-catchable moves trains a gambler). Because regret is on the *entry* decision and capture is on the *exit*, **regret can never curdle into hold-for-recovery** — the separation is what keeps the two instincts from colliding.

**The leak wall (refined).** The **observation must be causal**; the **reward may use hindsight** (it's computed post-episode, after the action — standard RL). So oracle/MFE/MAE in the *reward* is clean; they must **never** enter the *observation/policy input*. Oracle = label/reward engine, full stop. (This refines the earlier "realized-only" rule, which was right for *online* learning and for keeping MFE out of *features*.)

## 7. Tensions to balance (symmetric failure modes)
- **Wiggle-penalty:** too harsh → agent freezes / under-trades. Balance against a decaying **entropy/exploration bonus** so it samples trades early instead of defaulting to safe-flat.
- **Cut-reward:** too heavy → agent over-cuts everything (the `too_early` pathology, now reward-induced). Hold cut-on-wrong and ride-on-right in tension via a **speed/adversity dimension** — the cut reward **decays with holding time and accumulated MAE**, so *fast* cutting is what's credited.

## 8. Architecture mapping
- **Three heads** (trade/no-trade, direction, exit-timing), each receiving its reward component → clean credit assignment, matches the scorecard exactly.
- **The conditional funnel** ("70% entry → of those 30% good → of those 20% exit") is the **diagnostic dashboard** (extends the existing `ce_methodology` capture buckets) and the **curriculum order**: train **selectivity → direction → exit timing** (no point optimizing exit on trades you shouldn't take). The scorecard hands you the training order for free.

## 9. Anchors in existing work
- **`ce_methodology`** capture buckets (`too_early` / `too_late` / `optimal`, `capture_rate`) = the proven foundation this formalizes.
- **Cubic / orange-line** = the causal direction + exhaustion signal (slope, curvature).
- **Oracle** = the Grip-B post-hoc label/reward engine (defines the available move + capture denominator).
- **Exhaustion features** (z-bands, `reversion_prob`, variance-ratio, curvature, delta-divergence) = the causal antecedents the heads learn to read.

## 10. Open knobs — resolve next, against this frame
1. **Reward density:** terminal `capture_rate` at close vs **dense** per-step curvature-shaped reward.
2. **Architecture:** three separate policy heads vs single policy with the decomposed-but-summed reward.
3. **Component weighting:** relative weights of selectivity / direction / exit.
4. **Entropy schedule** (exploration decay).
5. **Setup-clarity source** for the regret gate: derive from the **cubic** (slope + curvature confidence) or the **oracle's** labeled setup quality.

---
*Axiom: reward direction + exhaustion, process not outcome, path-independent, oracle-anchored, causal observation. Everything downstream is tuning within this frame.*
