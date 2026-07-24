# teacher_speedup — qwen throughput research (isolated)

**Status:** OPEN (2026-07-23, owner-ratified as its own project for cleanliness/
contamination containment). **Nothing here touches production** — experiments
COPY or IMPORT from `research/dojo_forge/`, never modify it; promotion of any
winner happens only via a numbered comms doc + gate re-verification.

## Scope (from the constraint analysis, comms thread 2026-07-23)
qwen's non-hardware bottlenecks: (1) redundant prefill — ~10k tokens/frame,
~95% shared with the previous frame; (2) the think tax — 300–1500 generated
tokens/decision at 5–10× read cost. Candidate levers, each its own experiment:

| exp | lever | expected | risk |
|---|---|---|---|
| E1 | speculative decoding (qwen3-0.6B draft → 14B verify, llama.cpp native) | 2–3× generation, LOSSLESS | draft-model VRAM (~0.5GB) |
| E2 | anchor KV cache (state save/restore per episode) | ~25–30% prefill | state-restore correctness |
| E3 | KV q8 + n_batch 512 | +4–6 GPU layers / faster prefill | quality drift (must A/B logits) |
| E4 | compact text format (gen-1 text-definition spec pre-study) | up to ~2× tokens/frame | changes label semantics — gate territory |
| E5 | two-speed harness (System-1 logits everywhere; System-2 reasoning on contested frames only) | ~5–10× effective | threshold choice = doc-150 Phase 0 dependency |

## Verification discipline
Every experiment: (a) timing on a FIXED 3-episode bench set (same seeds),
(b) output-equivalence check vs baseline (bitwise for E1/E2; logit-delta budget
for E3; N/A E4/E5 — those are semantic changes and say so), (c) report in
`reports/` with the numbers. Baseline timings to beat: logit harness ~3.8s/frame
@8k ctx (38 layers); reasoned exam ~55s/frame @13k (full offload).

## Layout
- `pipeline/` — experiment harnesses (copies/wrappers, never edits of dojo code)
- `tools/` — benches + timers
- `reports/` — findings (E1_...md etc.) + assets
