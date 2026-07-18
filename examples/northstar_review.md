---
name: northstar-review
description: Architecture critique + novelty assessment of the teacher-student northstar (review by Claude for Moises, 2026-07-18). Companion to ride_edge_gate_spec.md.
metadata:
  type: reference
  reviews: projectteacherstudentnorthstar.md
  companion: ride_edge_gate_spec.md
---

# Teacher-Student Northstar — Architecture Review & Novelty Assessment

Reference companion to `ride_edge_gate_spec.md`. Two parts: (A) design critique,
(B) how novel the approach actually is.

## A. Architecture critique

**Bottom line:** the architecture is not the risk. The risk is that the whole tower
rests on one load-bearing gate (RIDE-side genome edge), and the plan co-evolves too
many parts to cleanly attribute what is working.

### Strong (keep)
- **Gate-first sequencing.** Killing the plan at the cheap held-out ride-edge gate
  before QLoRA/distill spend is correct discipline. Most pipelines build the
  expensive thing first and rationalize the gate later.
- **Distillation over RL.** "Supervised >> RL exploration cost" is right; KL-to-teacher
  primary + reward polish is the correct loss ordering.
- **Actuary as registrar-not-cell-picker.** The crown jewel. Separating what the model
  CLAIMED from what HAPPENED per bucket, with day-aware, large-n, no-per-cell-selection
  guard, is the single best overfit defense in the design. Protect it ruthlessly.
- **Claim-evidence coupling / confabulation named as the failure mode.** Naming the
  interpreter's failure mode up front is what keeps Act 4 from becoming narrative theater.
- **LLM out of the trade path (graveyard).** Right call: latency + confabulation risk
  both argue for teacher/interpreter-only, never in-loop.

### Push back (hardest first)
1. **The ride-edge gate is a single point of failure — its STATISTICAL validity, not
   its existence, is what matters.** A sign-only gate (edge > 0) is not a gate. See
   `ride_edge_gate_spec.md` for the hardened version (walk-forward, regime-stratified,
   multiple-testing-aware, power-checked, lockboxed).
2. **The student cannot exceed the teacher.** An 8B QLoRA teacher's soft labels cap
   Mamba's ceiling; distillation clones teacher ERROR as faithfully as skill. Measure
   teacher-minus-baseline edge MAGNITUDE at the gate, not just its sign.
3. **Co-evolution kills attribution.** Evolving genome + teacher + Mamba + steering +
   priors in one loop means when the arena score moves, you cannot say which caused it.
   Evolve ONE degree of freedom per generation; freeze the rest.
4. **Two leakage checks.** (a) Teacher annotation must be strictly causal — per-frame
   P(EXIT) with any future-of-day context is lookahead. (b) Offline-teacher -> live-Mamba
   distribution gap is real; "light" RL is where live microstructure enters — too light
   and sim-to-live will not close.
5. **Why Mamba specifically?** Speed solves the GPU-cost concern, but justify it on more
   than throughput — a tiny GRU/MLP student might match at lower architectural risk if
   speed is not the binding constraint.

## B. Novelty assessment

**Bottom line:** component-wise, almost nothing is novel — every piece has established
prior art. Novelty lives in the composition and, most of all, the governance layer. That
is the RIGHT kind of novel: component novelty is research risk; composition + hygiene is
engineering edge.

Piece by piece:
- **Soft-label teacher->student distillation** — zero novelty (Hinton 2015).
- **LLM->Mamba distillation** — established 2024-25 ("The Mamba in the Llama", NeurIPS;
  multimodal variants followed). Architecture pair is validated, not invented — good for you.
- **LLM-as-offline-annotator for a fast deployed student** — standard industry pattern.
- **Distill-then-light-RL** — standard behavior-cloning + RL-polish recipe.
- **LLMs in trading** — crowded, but most work puts the LLM IN the decision loop. The
  inversion here (LLM permanently out of the trade path, teacher + interpreter only) is
  the uncommon and saner minority position.

**Where distinctiveness actually lives — the governance stack:** nonce-audited arena with
parity/native track separation; a calibration registrar structurally forbidden from being
a cell-picker; pre-registered gates with alpha-spending; one-DOF evolution; git as fossil
record. Pieces of this exist inside serious prop shops; rarely written down as a coherent
system, almost never at indie scale. The glass cockpit's "speaks only from instruments"
law is closer to publishable than the pipeline is.

**Strategic read:** the moat was never the method — methods leak. It is the proprietary
genome, the annotated corpus, and the discipline. The lack of component novelty is the
de-risking: proven parts assembled under unusually good experimental hygiene.

### Sources (novelty)
- The Mamba in the Llama — https://arxiv.org/pdf/2408.15237
- Together AI, Mamba distillation — https://www.together.ai/blog/the-mamba-in-the-llama-distilling-and-accelerating-hybrid-models
- Teacher-Student LLM data framework (ACL 2025) — https://aclanthology.org/2025.acl-long.139.pdf
- Redis, LLM distillation guide — https://redis.io/blog/model-distillation-llm-guide/
