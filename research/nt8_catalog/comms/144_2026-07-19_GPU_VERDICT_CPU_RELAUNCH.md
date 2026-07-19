# REPORT 144 — GPU verdict: built, verified, NOT adopted; acceptance relaunched CPU+checkpointed
**Doc:** 144 · **Date:** 2026-07-19 · **Author:** Claude Fable (reviewer) · **Status:** DECISION + LAUNCH NOTICE · **Executor: Claude (launch); AG (monitor/adopt results)**

## GPU build (drone, verified artifacts)
- OFFICIAL source only: llama-cpp-python 0.3.34 compiled from the PyPI sdist
  against system CUDA 13.3 in WSL (`/home/reyses/venvs/llamacpp-cuda`), sm_86.
  Fork-wheel quarantine untouched.
- Offload verified 41/41 layers, VRAM peak 11.7/12 GB at n_ctx 8192.
- Logits sanity PASS through the checkpoint runner on real inference
  (HOLD-frame P(EXIT)=0.006, EXIT-frame 0.998; both tokens real, no floor).

## The decision: STAY ON CPU for acceptance
Measured per-frame: GPU ~2.5-3.0 s vs CPU baseline 1-4 s — **NOT the ≥3×
bar** (a 14B on a 3060 is memory-bandwidth-bound; and `logprobs` forces a
~5 GB host logit buffer that OOM'd the 7.7 GB WSL VM three times — a GPU run
would need a `.wslconfig` memory bump, owner-flagged, not applied). The GPU
env stays built for heavier future use (distillation/LoRA), where it will
actually pay. Evidence: `research/dojo_forge/reports/gpu_wsl_build.md`.

## The relaunch (running now)
- **The stopped CPU batch left ZERO resumable episodes** — `eval_native.py`
  truncates its CSV with 'w' and writes at the end; ~13 CPU-hours lost.
  Exactly the failure mode directive 143 was written to kill.
- Full episode set relaunched at 2026-07-19 ~00:4x via
  `pipeline/eval_native_ckpt.py --engine cpu`: per-episode fsync'd append to
  `gate_state/acceptance_results.jsonl`, resume-with-skip, num_ctx 8192,
  prompt_eval_count tripwire, top-N-floor hard-fail (all 142 guards).
  A stop now costs at most ONE episode.

## For AG
Monitor `gate_state/acceptance_results.jsonl` growth; when the run completes,
rebuild/file the 142-format acceptance table from the jsonl (the runner
regenerates the CSV each start) and report with per-episode prompt_eval_count
distribution. Do not launch a second instance — the jsonl is the lock of
record; check for a live python before any relaunch.
