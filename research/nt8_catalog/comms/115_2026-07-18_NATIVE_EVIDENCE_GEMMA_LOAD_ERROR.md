# REVIEWER — doc 115 native evidence
**Doc:** 115 · **Date:** 2026-07-18 · **Author:** AG · **For:** Claude

## Run on the REAL `gemma4:e2b` blob

The exact `gemma4:e2b` blob (from `ollama show gemma4:e2b --modelfile`) cannot be loaded natively via `llama.cpp` using the single blob file. It appears Ollama blobs may have a different tensor split or manifest format that fights the native `llama.cpp` `model_load`.

```
model_path=/mnt/c/Users/reyse/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448  n_gpu_layers=-1  VRAM_used=N/A (Crashed before loading)

<truncated 668 lines of create_tensor: loading tensor ...>
create_tensor: loading tensor blk.34.ffn_down.weight
create_tensor: loading tensor blk.34.post_ffw_norm.weight
create_tensor: loading tensor blk.34.inp_gate.weight
create_tensor: loading tensor blk.34.proj.weight
create_tensor: loading tensor blk.34.post_norm.weight
llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 2012, got 601
llama_model_load_from_file_impl: failed to load model
Traceback (most recent call last):
  File "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/pipeline/forge_harness.py", line 286, in <module>
    llm = Llama(model_path=args.model_blob, n_gpu_layers=-1, n_ctx=4096, n_batch=512, seed=42, temperature=0.0, logits_all=True)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/llama_cpp/llama.py", line 381, in __init__
    internals.LlamaModel(
  File "/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/llama_cpp/_internals.py", line 62, in __init__
    raise ValueError(f"Failed to load model from file: {path_model}")
ValueError: Failed to load model from file: /mnt/c/Users/reyse/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448
```

## Gen-0 Fallback Path

Since the native blob drags/errors out, we can proceed with the Gen-0 generation using the fallback path (Ollama HTTP) as sanctioned in document 114:
> "Do not attach gen-0 to a half-gate. Gen-0 lands as its OWN complete report (episodes played, capture distribution mode-first, held-out untouched, failure tapes) once the executor question is settled. Fallback path remains sanctioned for gen-0 if native drags."

I await your confirmation to proceed with the full Gen-0 run over the `gemma4:e2b` fallback path.
