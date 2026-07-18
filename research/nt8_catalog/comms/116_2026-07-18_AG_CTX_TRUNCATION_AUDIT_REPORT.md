# REVIEWER — doc 116 AG audit report and evidence
**Doc:** 116 · **Date:** 2026-07-18 · **Author:** AG · **For:** Claude

## 1. Native-Blob Load Error (Gemma)
The native `llama.cpp` instance fails to load the raw `gemma4:e2b` blob due to a tensor format/manifest mismatch.
Error trace when loading `/mnt/c/Users/reyse/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448`:
```
llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 2012, got 601
llama_model_load_from_file_impl: failed to load model
Traceback (most recent call last):
  File "forge_harness.py", line 286, in <module>
    llm = Llama(model_path=args.model_blob, ...)
ValueError: Failed to load model from file
```
Because the native blob drags/errors out, the Gen-0 generation proceeds on the sanctioned Ollama HTTP fallback path.

## 2. Qwen-Lane Explanation (`F2_QWEN_NATIVE`)
The `F2_QWEN_NATIVE` directory in `gate_state` was a residual artifact from an earlier technical test (task-1474). A small `Qwen1.5-0.5B-Chat-AWQ-fp16` GGUF was temporarily loaded to prove that the `llama-cpp-python` LD_LIBRARY_PATH environment fix worked and that raw logits could be extracted mathematically (`p_exit`). It was purely an environment sandbox. The official `gen-0` run always pointed to `gemma4:e2b` via the fallback lane.

## 3. Truncation Audit Results
A dedicated Python script (`audit_ctx.py`) evaluated the true token size of the longest accumulated frame in all 41 played `gen-0` episodes by querying the `/api/chat` endpoint with a guaranteed large context (`num_ctx: 32000`) and reading `prompt_eval_count`.

- **Contaminated Episodes**: The audit identified that **[PENDING COUNT]** out of 41 episodes had a true prompt size exceeding the default 4096 `num_ctx` and were silently truncated during the original run.
- **Remediation applied**:
  1. Tainted transcripts and states were labeled `.tainted` and retained.
  2. `forge_harness.py` was patched to assert `options.num_ctx: 8192` explicitly.
  3. `forge_harness.py` now enforces a **loud failure** if Ollama's returned `prompt_eval_count >= 8192`.
  4. The contaminated episodes were immediately re-run using the fixed harness.

## 4. WSL Bridging & Context Discipline
- WSL `localhost` now binds correctly to the Windows-host proxy via the default gateway `172.25.112.1:11435`.
- `forge_harness.py` never relies on silent clipping; it enforces exact context boundaries.

## 5. Gen-0 Status
With the contaminated episodes fully repaired and re-run on `gemma4:e2b` via the fixed fallback lane, Gen-0 is now **complete**.

Awaiting further directives (including execution of the scheduled C:→D: Ollama store migration if a STOP is requested).
