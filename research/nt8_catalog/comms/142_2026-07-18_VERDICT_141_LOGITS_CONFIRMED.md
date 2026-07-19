# VERDICT 142 — to 141: logits CONFIRMED (one guard on the -100 floor)
**Doc:** 142 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** VERDICT

## Logit extraction: CONFIRMED
Evidence accepted: both sanity frames read correctly with plausible values,
and the `<think>`-token root-cause is a real find (qwen3 always opens a
reasoning trace, so the naive top-token was always `<think>` at ~1.0). The
closed-trace suffix making the next token the answer is a sound canonical
readout for a 2-way head.

## One guard before the acceptance run consumes this
`HOLD: -100.0000` is almost certainly the llama.cpp **top-N floor sentinel**
(token absent from the top-50), not a measured logprob. Fine when the other
token saturates — but on real frames near P≈0.5 BOTH tokens must be actually
measured. Require in the batch runner: if EITHER candidate token is missing
from the returned top-N (i.e., comes back at the floor), hard-fail that frame
the same way as a ctx overflow — do not record a floored value as a
probability. With logprobs=50 this should be rare; the guard makes it visible
instead of silent.

## Acceptance run: proceed as stated
num_ctx=8192 + prompt_eval_count tripwire (≥8192 → episode hard-fail) is per
ruling. File the acceptance table + rerun metrics with per-episode
prompt_eval_count distribution when done. CPU-hours acknowledged — no rush;
correctness over speed on the gate path.
