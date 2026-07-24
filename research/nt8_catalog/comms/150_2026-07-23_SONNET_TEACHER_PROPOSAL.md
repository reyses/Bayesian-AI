# RESEARCH PROPOSAL — Sonnet through the curriculum (adjudicator → ensemble teacher)
**Doc:** 150 · **Date:** 2026-07-23 · **Author:** Claude (Fable) · **Origin:** Moises (via Telegram) · **Status:** PROPOSED

## Idea (owner)
Run Sonnet-class Claude agents through the SAME episode curriculum as the qwen
teacher — the tiered packet+window harness is teacher-agnostic (exit-dojo
precedent: 200 blind episodes via headless Claude sessions, docs 097/098).

## Honest deltas vs the qwen teacher
1. **No native logits.** The API exposes no logprobs → no true p_exit soft
   labels; instead decision + verbalized confidence (weaker calibration).
   The distillation target changes shape (discrete/verbalized vs continuous).
2. **Non-determinism + model drift.** qwen is seeded/bit-reproducible; Sonnet
   is not, and versions rotate. Acceptable for OFFLINE labeling (the artifact
   freezes once written — consistent with the "LLM-as-feature, never live
   decider" law), but weakens same-run reproducibility claims.
3. **Cost.** qwen is free local; a full 156-episode Sonnet pass ≈ ~50M input
   tokens. Budgeted, deliberate runs only (the no-parallel-cloud-spend rule).

## Phased plan (each phase gates the next)
- **Phase 0 — qwen readout-disagreement experiment (free, ~1h).** ~40 frames
  stratified across the p_exit spectrum; run BOTH readouts (think-bypass logit
  vs reasoned generation); measure disagreement structure. Motivated by the
  10/10 HOLD-flip finding (`reports/teacher_why_2026-07-23.md`) that put the
  gen-0 baseline in question.
- **Phase 1 — Sonnet as ADJUDICATOR (small budget).** On the frames where
  qwen's readouts conflict, Sonnet reasons over the identical tiered context
  as a third opinion. Output: per-frame verdict + rationale. Decides whether
  the artifact is the bypass readout, the reasoned pass, or genuine ambiguity.
- **Phase 2 — ensemble-diversity teacher (budgeted subset).** Sonnet labels a
  stratified episode subset; qwen-vs-Sonnet disagreement becomes an
  UNCERTAINTY channel for the student (labels where teachers agree are
  high-confidence distillation targets).
- **Explicit non-goal:** replacing qwen. Free-local-frozen remains the primary
  teacher premise; Sonnet is measurement + diversity.

## Kill rules (pre-registered)
- Phase 0: if disagreement <10% and unstructured → readout artifact rejected;
  baseline stands; genome mutation proceeds; Phases 1-2 lose urgency.
- Phase 1: if Sonnet sides with the bypass readout on >70% of conflicts →
  reasoned-pass explanations were post-hoc stories; keep the fast readout.
- Phase 2: if qwen/Sonnet agreement is so high the uncertainty channel is
  empty (<5% disagreement) → ensemble adds cost without signal; stop.

## Dependencies / governance
Curriculum episodes only (no lockbox exposure). Phase 2 cost needs an owner
budget number before launch. Results feed doc-149's re-baseline decision and
the gen-1 mutation targets.
