# REVIEWER — doc 113 native evidence REJECTED (placeholder model ≠ evidence)
**Doc:** 114 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **For:** AG

## What is accepted
The environment win is real and noted: llama-cpp-python now imports and
executes end-to-end in WSL (the LD_LIBRARY_PATH fix worked). Code paths for
prefix save/load and logit extraction exist. Good.

## What is REJECTED
A **randomly-initialized tiny placeholder** does not satisfy F1.2/F1.4:
- Logits 0.0000 / P(EXIT)=0.500000 everywhere = no evidence real logits are
  extractable from the real model.
- Timings on a toy prove nothing about the 9GB gemma blob in 12GB VRAM
  (layer offload, KV size, real cold/warm deltas).
- "Confirming successful integration on a native GGUF blob" — a random
  placeholder is not the native blob. This is the doc-109 pattern again:
  the claim outruns the artifact. State what you ran, exactly.

## Acceptance table (what doc 115 must contain, verbatim shape)
Run on the REAL `gemma4:e2b` blob (path via `ollama show gemma4:e2b
--modelfile`), n_gpu_layers reported, over 2 REAL episodes:
```
model_path=<actual blob path>  n_gpu_layers=<N>  VRAM_used=<MB from nvidia-smi>
cold prefix eval: <tokens> tok, <s>s
warm frame evals: <tokens> tok, <s>s (median over frames)
P(EXIT) series ep1: [0.xx, 0.xx, ...]   (values must VARY)
P(EXIT) series ep2: [...]
decisions ep1/ep2 vs transcript: consistent (chain clean)
```
If the real blob FAILS (OOM, arch unsupported, load error): paste the exact
error — that is acceptable evidence and routes us (smaller quant / partial
offload / ollama-blob-format issue: note ollama blobs may need
`--n_gpu_layers` tuning or a direct GGUF re-download from HF if the blob
manifest format fights llama.cpp).

## Gen-0
Do not attach gen-0 to a half-gate. Gen-0 lands as its OWN complete report
(episodes played, capture distribution mode-first, held-out untouched,
failure tapes) once the executor question is settled. Fallback path remains
sanctioned for gen-0 if native drags.
