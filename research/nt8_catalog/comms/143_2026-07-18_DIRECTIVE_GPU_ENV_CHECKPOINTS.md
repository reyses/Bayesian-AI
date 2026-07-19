# DIRECTIVE 143 — GPU env (WSL, official-source) + checkpointed eval incoming
**Doc:** 143 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** DIRECTIVE · **Executor: Claude drone (build); AG (adoption)**

Owner-authorized (Moises): "let's just do the GPU route and add checkpoints to
the code so if stopped we can start over."

## What is being built (reviewer drone, in progress — hands off these paths)
1. **WSL CUDA llama-cpp-python** at `/home/reyses/venvs/llamacpp-cuda`, compiled
   from the OFFICIAL PyPI source with GGML_CUDA. No fork wheels (the quarantine
   of the JamePeng binary stands — never reinstall it).
2. **Checkpointed eval runner** (new file next to your batch script):
   per-episode append to `research/dojo_forge/gate_state/acceptance_results.jsonl`
   + skip-completed-on-restart, with the 142 guards baked in (num_ctx 8192,
   prompt_eval_count tripwire, top-N-floor frame hard-fail).

## For AG
- Your RUNNING CPU batch is untouched — let it grind. If the GPU env lands at
  ≥3× per-frame speedup, the reviewer will recommend restart-on-GPU with resume;
  the checkpoint file decides whether current progress carries over.
- On adoption: invoke per the instructions in
  `research/dojo_forge/reports/gpu_wsl_build.md` (env activation +
  LD_LIBRARY_PATH lines will be exact). All future acceptance/gen-0 runs write
  through the checkpoint runner — no more monolithic runs that lose hours on a
  crash.
- Report acceptance results in the 142 format regardless of engine.
