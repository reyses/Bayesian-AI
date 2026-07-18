---
name: ride-edge-gate-spec
description: Statistical validity spec v2 for the load-bearing gate (Moises northstar, 2026-07-18) - power-checked, peek-proof, lockboxed, parity-aware. The whole distillation tower passes through here.
metadata:
  type: spec
  gate: ride-edge
  status: draft-v2
  supersedes: v1 (adds Q0 power, Q3 transfer, sequential/alpha-spending, lockbox, ablated-teacher baseline, cost-anchored floor, actuary-bucket unification)
---

# Ride-Edge Gate — Statistical Spec v2

The tower rests on ONE claim: genome shows real edge on the RIDE side of held-out
(both dojos wash on cut). If this gate passes by luck, every downstream spend
(QLoRA, annotation, Mamba distill, RL, ONNX) is built on a mirage. v2 hardens the
gate against the four ways a false pass actually happens: luck (Q1), triviality
(Q2), non-transfer (Q3), and self-deception (Section 0 + leakage gates).

Sign is free; magnitude, robustness, and transfer are the assets.

## 0. Pre-registration & peek discipline (before anything is scored)

- **Freeze this spec in git BEFORE scoring.** The commit hash is the
  pre-registration. Any edit after first scoring = new gate, new alpha.
- **Nominate-then-score.** A genome candidate is nominated on training data only,
  THEN scored on held-out once. No browsing held-out to pick the candidate.
- **Sequential peeking is the default failure here** — evolution runs
  continuously and the gate will be consulted repeatedly across generations.
  Keep an **alpha-spending ledger** in the fossil record: every held-out look is
  logged, and either (a) each generation's look spends alpha (O'Brien-Fleming
  style bounds), or (b) cheap looks use a rotating dev-holdout and the true gate
  is scored on the lockbox only.
- **Lockbox.** Arena reuse of "held-out" episodes across thousands of checkpoint
  evals causes holdout decay. Reserve a terminal lockbox set (most recent
  contiguous days, never touched by evolution, arena, OR gate peeks) opened
  EXACTLY ONCE: at the final pass/fail before QLoRA spend. Log its opening.
- **One bucketing scheme.** Regime strata below = the ACTUARY's coarse
  structural buckets (tier x regime x session), pre-registered. The gate does
  not invent its own bins — private strata are a hidden researcher degree of
  freedom, and sharing buckets makes gate numbers and actuary tallies the same
  currency.

## Q0 — Power check (may the gate even run?)

Before scoring: compute the minimum detectable effect (MDE) at the
pre-registered alpha and day-count of the held-out window, using day-level
units. If MDE > the Q2 cost-anchored floor, the gate is UNDERPOWERED — a pass
would be noise, a fail uninformative. Verdict: "collect more days," not "run it
anyway." Emit the power calc into the gate record.

## Q1 — Is the ride edge real?

1. **Walk-forward, K folds.** Evolve on [0..t], score [t..t+w], roll. Report
   per-fold edge. A single slab invites slab-luck; the claim must survive
   sequential out-of-sample.
2. **Regime-stratified** using the actuary buckets (Section 0). Edge per bucket,
   large-n cells only, aggregate secondary. Edge confined to one bucket = a
   regime bet; either kill or re-scope the thesis TO that regime and re-gate.
3. **Day-aware units.** Independent unit = day/session. Frames are
   autocorrelated; frame-counting fakes n and shrinks CIs. All tests use
   day-level or block bootstrap.
4. **Multiple-testing.** Declare N_trials (every genome/config/threshold ever
   evaluated toward this candidate, including discarded generations). Judge
   against N_trials via BH-FDR across candidates or Bonferroni for the promoted
   one. Prefer deflated Sharpe / PSR over raw t-stat. Alpha pre-registered
   (suggest 0.01) and debited per the Section 0 ledger.
5. **Drift check.** Regress per-fold edge on fold index. Significant negative
   slope = alpha decay in-sample of the gate itself; fresh-fold edge carries the
   verdict, not stale-fold average.

**Q1 pass:** positive ride edge in >= ceil(0.7*K) folds; MT-and-ledger-corrected
significance on aggregate; not single-bucket-confined (or re-scoped); no
significant decay slope; cut side confirmed ~wash (an accidental cut edge
changes the thesis and voids the gate).

## Q2 — Is the teacher edge worth cloning?

The student cannot exceed the teacher; distillation clones teacher ERROR as
faithfully as teacher skill. Measure magnitude, not sign.

1. **Baseline panel** on identical held-out ride episodes: always-ride,
   random-exit, fixed-horizon, current production heuristic, and — new —
   **ablated teacher** (same weights, state channels shuffled/permuted per
   frame). If the ablated teacher retains the edge, the teacher is a fancy
   constant and there is nothing state-dependent to distill. Hard fail.
2. **Cost-anchored floor.** teacher_edge = teacher − best_baseline, day-level
   CI. Pre-register the floor as an equation, not a vibe:
   floor = round-trip costs+slippage + expected distillation loss (parity gap +
   quantization) + safety margin. CI-excludes-zero does NOT pass; CI must clear
   the floor.
3. **Soft-label informativeness.** P(EXIT)/P(ride) labels must discriminate:
   check entropy/dispersion (not degenerate-flat or saturated), AUC of
   per-frame P against realized outcomes, reliability slope (calibration —
   quote it through the actuary's referee, same currency). Flat or miscalibrated
   labels distill into a useless student regardless of teacher win-rate.
4. **Logprob fidelity.** The annotation path is native llama.cpp logprobs on a
   quantized teacher. Verify quantized logprobs track full-precision teacher
   within tolerance on a sample (rank correlation + calibration delta). GGUF
   quantization can silently warp the very distribution being distilled.
5. **Student headroom probe (cheap derisk).** Before committing to Mamba: train
   a trivial student (linear probe / tiny GRU) on a label subsample. If the
   trivial student captures most of teacher_edge, distillation is derisked; if
   it captures ~none, the edge may live in features the student's input space
   cannot express — see Q3 before spending.

## Q3 — Does the edge survive the adapter? (parity → native)

The teacher's edge is measured on LLM frames; Mamba deploys on NATIVE full
production state through the tensor-frames adapter. An edge that lives in
frame-representation artifacts dies in translation — discovered only after the
full distill spend unless gated here.

- Score the teacher (or the Q2 probe student) on PARITY-track tensor
  equivalents of the same episodes. parity_edge / frame_edge = transfer ratio;
  pre-register a minimum (suggest >= 0.7).
- Reconcile the adapter: enumerate what the tensor translation drops or
  distorts vs the frames the teacher was scored on. Every dropped channel is a
  candidate explanation for transfer loss — name them before, not after.
- Arena two-track separation (PARITY vs NATIVE) stays intact downstream; this
  gate only certifies the entry claim: the edge is representational-artifact-free.

## Leakage gates (hard-fail, checked before Q1–Q3 are scored)

- **Causal annotation.** Label(t) uses only state <= t. Unit test: perturb
  future frames, assert label(t) unchanged. Any future-of-day context =
  automatic fail.
- **No fold bleed.** Normalization stats, feature scaling, thresholds — nothing
  fit on data the fold scores on.
- **Nonce-audit parity.** Gate scoring runs through the same nonce-audited
  harness as the arena: one measurement, not two dialects.
- **Holdout hygiene.** Ledger complete (Section 0); lockbox unopened until the
  terminal run; dev-holdout rotation logged.

## Gate record (fossil, committed)

One signed record: spec commit hash; power calc; per-fold + per-bucket ride
edge; N_trials + correction + alpha ledger; deflated significance; drift slope;
teacher_edge vs full baseline panel (incl. ablated) with CIs vs the cost floor;
soft-label diagnostics + logprob fidelity; probe-student headroom; transfer
ratio; leakage pass/fail; lockbox status. Counts flow into the actuary's
register (registrar, never cell-picker). If it fails, the plan dies here —
cheaply, by design.

## One-degree-of-freedom guard (downstream)

Past the gate: evolve ONE thing per generation (genome XOR teacher XOR Mamba
ckpt XOR steering XOR priors), freeze the rest, measure, commit. The gate proves
the edge exists; one-DOF evolution keeps you able to prove what carries it.
