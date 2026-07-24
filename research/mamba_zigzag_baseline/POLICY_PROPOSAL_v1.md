# PROPOSAL — Student Policy Training Architecture v1 ("distill-first, leashed-RL")
Status: PROPOSED for external review (both consultants). 2026-07-24.
Self-contained: no repo access assumed. All numbers are measured unless marked assumed.

## 1. SYSTEM CONTEXT
The program trains a fast STUDENT policy (Mamba SSM, ~431 bars/s inference) for
minute-scale MNQ futures ride-management: given an already-taken entry, decide
HOLD/EXIT each bar to capture ride length. Ground truth about the domain, measured
at high power on 23,378 engagements over 282 test days: on quality entries, NO cut
policy beats never-bail — every dumb stop nets −6.9…−3.4 ticks/trade (all CIs ≤ 0);
76% of −$100 drawdown legs recover; winners are captured by DURATION. The edge is
ride-length, not exits-timing or entry-timing.

A frozen LLM TEACHER (qwen3-14B, deterministic logit readout) labels episodes
minute-by-minute with P(exit); its rule-set ("genome") is text, evolved
generationally, each generation gated on held-out episodes against the never-bail
baseline with day-block bootstrap CIs. Gen-0 (3 naive rules) LOST to never-bail
(−12.6 pts/ep, day-block CI [−18.7, −6.4]) by exiting too early (median minute 7
vs oracle 15). Gen-1 (rules + a measured-facts handbook) is being gated now.
Teacher label noise floor is measured: 1.07% decision flips under seed
perturbation, median |ΔP| = 0.

The student's current reward (a hand-built scorecard) exhibits a measured
structural failure. Epoch-0 telemetry: 2,533 trades/day (churn), PF-based
win-rate −0.93, entry discrimination near zero (enters on 85% of signal bars vs
74% of no-signal bars), and reward decomposition: capture ≈ 0, cost = an EXACT
constant −0.30 reward-units × trade count, wiggle penalty ∝ trade count.
**Diagnosis: there is no positive per-trade term.** Every gradient path to higher
reward reduces trade count; the fixed point is the flat (never-trade) policy.
Freeze and churn are the same gradient seen at different training stages — a
2-epoch smoke shows trade rate collapsing 3–7×/epoch toward flat.

## 2. THE PROPOSAL (three layers + sequencing)

### Layer 1 — Imitation as the PRIMARY objective
Per-decision loss: cross-entropy between student action distribution and the
teacher's label, weighted by the teacher's logit margin (confident teacher
labels teach harder; near-tie labels teach softly).
**Why this way:** it is the only positive per-trade learning signal we can
trust. (a) It cannot freeze: matching the teacher pays on every decision, and
the teacher trades. (b) It cannot churn: the student's trade rate is anchored
to the teacher's BY CONSTRUCTION — no reward-tuning needed to police count.
(c) It inherits the gate: the teacher only becomes a distillation target after
beating never-bail on held-out days, so the student imitates a certified edge,
not a hope. Margin weighting exists because the measured noise floor shows
near-tie teacher labels (|P−0.5| small) are seed-unstable — they carry little
information and should carry little gradient.

### Layer 2 — RL as a small KL-leashed correction
After imitation converges: maximize expected net PnL subject to
KL(π ‖ π_imitation) ≤ ε. PnL uses the ratified execution model: $1.00 fixed
round-trip commission + slippage sampled uniformly WITHIN the actual next-1s
bar's range after each decision (per-trade seeded draw → random-in-distribution,
bitwise-reproducible; adverse-side fill is the pre-registered stress variant).
**Why this way:** unconstrained PnL-RL on this reward landscape rediscovers the
freeze/churn gradient or hacks the shaping. The KL leash bounds BOTH failure
modes simultaneously — the policy cannot drift far from a certified-sane anchor
in either direction — and converts "reward must encode all our fears" into
"reward only needs to point uphill locally." ε is pre-registered from teacher
label statistics (see §4), never tuned on test outcomes.

### Layer 3 — Potential-based gradient densifier
Shaping term F = Φ(s′) − Φ(s) with Φ = unrealized mark-to-market.
**Why this way:** the true reward (realized PnL at exit) is sparse — one
terminal signal per ride starves credit assignment across 20–60 decisions.
Potential-based shaping telescopes to zero over any episode (Ng et al. 1999:
provably policy-invariant), so it densifies the gradient WITHOUT creating a
churn or hold incentive. This is the difference between it and the
opportunity-cost / entropy-bonus hacks we explicitly distrust: those change the
optimal policy; this provably does not.

### Sequencing (a dependency, not a schedule)
1. GATE THE TEACHER FIRST. Distilling gen-0 would distill the too-early-exit
   disease. The imitation loss is built only against a generation that beat
   never-bail on held-out days (day-block CI, cleared above the noise floor).
2. GATE-FIRST FOR THE STUDENT TOO: before any RL layer runs, execute the
   teacher's labels through the realistic-fill model. If net PnL ≤ 0, no reward
   architecture fixes it — the signal is absent and the correct response is to
   fix the teacher (or pivot it to entry-veto/sizing), not to tune the student.
3. Meanwhile (cheap, running now in spirit): extend the freeze-trajectory smoke
   to epoch ~8–10 to confirm/refute that the current scorecard's rate collapse
   terminates in flat — closing the loop on the diagnosis.

## 3. WHY NOT THE ALTERNATIVES (reasoning record)
- **Pure PnL-RL:** the measured landscape has no positive per-trade term; with
  realistic costs the gradient points to flat, and with shaping hacks it points
  to churn. Also unfalsifiable failure: a frozen policy is indistinguishable
  from "correctly learned there is no edge" unless the teacher-gate has already
  established the edge exists.
- **Entropy bonuses / exploration rewards:** treat the symptom (no trading) by
  paying for randomness; the policy trades for the bonus, not the market. When
  annealed, freeze returns. Distrusted on principle: any term that pays for
  action-diversity per se changes the optimum.
- **Opportunity-cost penalties on flat:** manufacture churn by construction and
  require a counterfactual ("what you would have made") that is itself a leak
  surface.
- **Reward re-weighting alone (raise capture, lower cost):** re-tunes the same
  degenerate landscape; every weighting either under-prices friction (churn) or
  over-prices it (freeze) because there is no per-decision truth signal in the
  reward — that signal lives in the teacher's labels, which is Layer 1.

## 4. PRE-REGISTRATION (before any training run)
- ε (KL budget) and a two-sided trade-rate band derived from the certified
  teacher's OWN label statistics on training days (e.g., band = teacher trade
  rate ± measured day-to-day spread). Frozen before training; violations abort.
- Evaluation: held-out days only, day-block bootstrap CIs, deltas interpreted
  against the measured teacher noise floor; PF-based win-rate (never count-based).
- Execution model as ratified: $1.00 commission + seeded next-1s-bar slippage;
  adverse-fill stress variant reported alongside.
- Ablations pre-registered: imitation-only vs +shaping vs +leashed-RL; and the
  KL leash sweep is run ONCE on a dev slab, not iterated against the gate.

## 5. KNOWN RISKS / OPEN QUESTIONS (for both consultants to attack)
1. Teacher ceiling: the student can at best match a teacher that itself sits
   well below oracle (gen-1 pending). Is leashed-RL headroom (ε) enough to
   exceed the teacher, or does exceeding require widening ε to where the leash
   stops protecting?
2. Margin weighting uses teacher logit margins that are themselves imperfectly
   calibrated — should weights be calibrated against realized outcomes instead
   (risking circularity)?
3. Distribution shift: the student acts on its own trajectory; imitation data
   is teacher-on-teacher-trajectory. Do we need DAgger-style corrections
   (student states relabeled by the teacher), and at what cost?
4. The 1s-bar slippage model prices fills within one second of decision — is
   that adverse enough for minute-scale decisions in fast tape?
5. Φ = unrealized MTM assumes marks are well-defined mid-ride at 5s cadence —
   any pathology in gap/halt bars?

— End of proposal v1. Feedback welcome on any layer, the sequencing, or the
pre-registration; adversarial readings preferred.
