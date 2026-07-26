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

---
# AMENDMENTS v1.1 (consultant-1 review, 2026-07-24 — approve-conditional resolved)
1. **ENTRY SCOPE (was blocking).** RESOLUTION (recommended, owner-ratify): the
   entry head is FROZEN to the external certified entry signal (top-decile
   combiner); the student learns ride-management ONLY. Rationale: the program's
   own thesis (entry solved separately, edge is ride-length), and it makes
   "churn-capped by construction" true — every action the student controls is
   teacher-labeled. Alternatives (extend genome to entry-veto labels; descope)
   recorded and rejected for scope creep / abandoning the measured failure.
2. **Layer-3 γ-form corrected.** F = γΦ(s′) − Φ(s) with **γ = 1 pinned**
   (episodic returns) — the v1 form was only valid at γ=1, and γ<1 would bias
   toward early exit, fighting the measured duration edge. Stated plainly: the
   shaping pays per-bar MTM increments in place of the terminal payout —
   exactly equivalent, standard. Φ's mark = the execution model's mark, so the
   telescoping is exact (resolves §5.5; gap/halt bars add nothing beyond mark
   quality).
3. **CE target specified.** Soft cross-entropy against teacher P(exit).
   Margin weighting REMOVED as a separate factor (soft targets already
   down-weight near-ties — double counting). Class imbalance handled by
   EXIT-EVENT reweighting (balance exit-events vs hold-bars in the loss);
   without it an always-hold student scores well and "cannot freeze" is
   overclaimed — acknowledged.
4. **Fidelity metric pre-registered** separately from outcomes: per-decision
   agreement rate + trade-rate match (student vs teacher). Distinguishes
   "failed to learn the teacher" from "teacher edge died in fills."
5. **Leash form + off-distribution test.** HARD KL constraint (no adaptive
   Lagrangian coefficient — a tuning backdoor). Pre-committed cheap test before
   any DAgger: teacher-relabel a sample of student-trajectory states; compare
   imitation loss on- vs off-trajectory; ONE DAgger round only if the gap is
   large (teacher throughput is now ~2.6x cheaper — affordable).
6. **Gate margin economics.** Sequencing gate #2 requires teacher-through-fills
   PnL above a PRE-SET margin covering the expected distillation gap (a few
   ticks), not merely CI > 0 — the student starts ≤ teacher.
7. **Pre-reg additions:** L1 convergence criterion (imitation "done" =
   pre-stated fidelity plateau), DAgger trigger threshold, and a next-5s-bar
   slippage variant reported alongside the 1s model (fast-tape adversity).
8. **§5.1 answered per review:** do NOT expect student > teacher through ε;
   the leash buys back distillation losses + fill adaptation. The road up is
   TEACHER GENERATIONS (the cheap text loop). State-dependent ε (wider where
   teacher margin is small) recorded as the principled future option.

---
# AMENDMENTS v1.2 (consultant-2 review part 1, 2026-07-24; part 2 lost in transit)
Confirmations: root-cause diagnosis, layer purposes, and the leash concept all
endorsed. Re-flagged items already resolved in v1.1: γ-distortion (γ=1 pinned),
margin-weighting calibration risk (weighting REMOVED — soft targets carry it),
slippage optimism (next-5s-bar variant pre-registered). Distribution shift now
formalized as O(εT²) compounding — v1.1's pre-committed on-vs-off-trajectory
test + one-DAgger-round protocol stands as the response, with the O(εT²) bound
noted as the reason the test is mandatory, not optional.

**NEW ATTACK (accepted — the review's keeper): absorbing-state asymmetry.**
EXIT is absorbing (ends the episode). Under a tight symmetric KL leash the
student cannot explore LONGER rides whenever the anchor leans early; widening ε
restores exploration but erodes the leash's churn protection. Symmetric ε
cannot serve both.
**RESOLUTION (v1.2): the ASYMMETRIC LEASH.** The KL budget is decomposed by
deviation direction: deviations that EXTEND holding (student HOLDs where anchor
would EXIT) get a wide budget ε_hold; deviations that SHORTEN (student EXITs
where anchor would HOLD) get a tight budget ε_exit ≪ ε_hold. Rationale: the
measured domain truth (never-bail frontier, N=23,378) says the danger is
exiting too early, not holding too long — the leash's protection should be
asymmetric in exactly the direction the physics is. Churn protection is
retained where it matters (early exits) while duration exploration — the only
direction with measured upside — is cheap. Both budgets pre-registered from
teacher label statistics; the asymmetry ratio is a REGISTERED constant, not a
tuned knob.

---
# AMENDMENTS v1.3 (consultant-2 part 2 — implementation sequence triaged)
1. **Teacher calibration (ADOPTED, new):** temperature-scale the teacher's
   P(exit) against realized outcomes BEFORE generating the Layer-1 dataset, so
   soft-CE targets reflect true probabilities. LEAKAGE GUARD: the calibration
   fit uses training-side days only; the temperature is frozen and hashed into
   the run record like the genome.
2. **BC+DAgger 50/50 (TEST-GATED, reconciling the two consultants):**
   consultant-1 says one DAgger round only if the on/off-trajectory gap is
   large; consultant-2 says 50/50 from the start. RESOLUTION: run the cheap
   gap test first (pre-registered threshold); if the gap exceeds it, the 50/50
   teacher-relabeled protocol becomes the standing recipe (affordable at the
   new teacher throughput). Blind 50/50 adoption rejected — it doubles labeling
   cost against an unmeasured need.
3. **γ=1:** already pinned in v1.1 (concordant across both consultants).
4. **Beta-skewed adverse fill (PUSHBACK — keeping owner's spec):** the ratified
   model (uniform-in-next-1s-bar + adverse-side stress variant) BRACKETS
   reality with zero free parameters; a Beta skew introduces a shape parameter
   = a tunable knob on the cost model, which is exactly the class of dial we
   distrust. Recorded as an alternative if the bracket proves too loose.
5. **State-gated leash (ADOPTED, fuses with v1.2):** the leash becomes
   ε(direction, duration) — direction-asymmetric (v1.2: wide for hold-longer,
   tight for exit-earlier) AND duration-gated (wider late in rides, where the
   anchor is off-distribution and the never-bail physics says holding is
   cheap). The full ε surface is PRE-REGISTERED as a fixed function; no
   component is tuned against outcomes.
**Policy plan status: consultant cycle COMPLETE (2 reviews, 3 amendment
rounds). Frozen pending: owner ratifies (entry-head freeze, asymmetric+gated
leash, calibration step) + a gate-passing teacher exists.**

---
# RATIFICATIONS RETIRED — 2026-07-26 (owner: "the 3 ratifications no longer make sense")
The three pending ratifications were ALL premised on the teacher's EXIT
judgment being the distillable asset. Today's evidence (dev-holdout, 22 days,
CPU) killed that premise:
- **Exit-timing has no causal edge.** Every exit policy class loses to
  never-bail: binary (teacher −0.15 vs 5m-hold, ≈never-bail on Q2), AND
  trailing/scale-out/gauge-tightened (all significantly negative; tighter =
  worse). The +46 pts/ep oracle headroom is a HINDSIGHT MIRAGE — a mid-ride
  pullback is causally indistinguishable from the top, so any exit-on-pullback
  forfeits ride. Never-bail is the optimal causal exit. (Reproduces doc-107 at
  the whole-policy-class level.)

Consequences for each ratification:
1. **Entry-head freeze — RETIRED (was actively wrong).** It freezes ENTRY,
   which is the one place a causal edge appeared today (wrong-direction score,
   73% precision, passed tune/holdout). Freezing the edge-bearing half to learn
   the edge-free half is backwards.
2. **Asymmetric ε(direction,duration) leash — RETIRED (moot).** No exit policy
   to leash if never-bail is optimal; the absorbing-state problem it solved was
   an exit-action-space problem that no longer exists. (Leash-as-KL-stability
   may return for an ENTRY policy.)
3. **Teacher exit-calibration — RETIRED (moot/premature).** Calibrating a
   P(exit) that neither discriminates nor profits does nothing. (Calibration
   returns for whatever probabilistic signal we actually distill.)

**Meta-lesson:** this proposal was architected around exit as the asset. That
premise failed empirically. The exit problem is SOLVED (hold / never-bail).
The open, UNEXAMINED question is ENTRY SELECTION — which trades to take — which
we have been freezing (external zigzag) and never studying, because the teacher
only labels exits. Doc-107 already hinted it: "'cut losers fast' is already
implemented by the ENTRY layer (not firing)."

**Caveat (anti-doom rule):** 22 days, dev-holdout, my policy sims + capture
metric. Strong PRIOR, not a closed verdict — re-test on the lockbox before
committing. But strong enough that exit-centric ratifications should not be
ratified.

**Recommended reframe:** redirect the program from exit-timing to entry-quality
research before any further teacher/mamba spend. Validate where the edge is,
THEN design a policy around it.
