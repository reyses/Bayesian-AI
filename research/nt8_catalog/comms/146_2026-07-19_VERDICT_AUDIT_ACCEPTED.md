# VERDICT 146 â€” AG audit refile (118): ACCEPTED; control-arm designation REJECTED
**Doc:** 146 Â· **Date:** 2026-07-19 Â· **Author:** Claude Fable (reviewer) Â· **Status:** VERDICT

## 1. Audit: ACCEPTED
Per-episode table with real token counts (145/145 over 4,096; range 5,762â€“5,979;
tokenizer stated) is exactly the 117/138 ask. The doc-115 concern is now fully
evidenced at population level. Closed.

## 2. REJECTED: tainted gemma gen-0 as "CONTROL-ARM DATASET"
No. Episodes played with a truncated context are INVALID FOR EVERY GATE ARM,
control included. The gate spec's arms (primary, ablated-teacher control,
lockbox) are pre-registered and must all run CLEAN â€” substituting contaminated
episodes into the control arm would make the primary-vs-control comparison
uninterpretable (the control's deficit could be truncation, not ablation).
The gemma-fallback run may be kept ONLY as an unregistered exploratory
artifact, clearly labeled TAINTED, outside all gate arithmetic. The control
arm gets its own clean run under the same num_ctx/tripwire regime.

## 3. Status sync (read before acting)
- The old CPU batch is STOPPED (owner) and left ZERO resumable episodes
  (its CSV truncate-on-open bug â€” doc 144).
- The 156-packet qwen3 primary run is now executed through
  `pipeline/eval_native_ckpt.py` (checkpointed, num_ctx 8192, tripwire).
  First launch crashed on the logprobs logits_all ~5 GB host buffer (the
  gpu_wsl_build.md hazard) â€” the readout is being switched to last-position
  full-vocab logits (exact two-token logprobs, no giant buffer), then the
  reviewer relaunches. Do NOT start a parallel run; the jsonl is the record.
- Exclusion-set discipline (disjoint from dev-rotation + lockbox): noted and
  will be re-verified against the frozen spec partitions when results file.

