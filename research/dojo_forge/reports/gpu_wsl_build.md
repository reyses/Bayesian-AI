# GPU (WSL/CUDA) llama-cpp-python build + crash-safe checkpointed eval

**Date:** 2026-07-19 · **Author:** Opus build drone · **Status:** BUILD DONE, VALIDATED
**Directive:** comms 143 (GPU env + checkpoints) · **Method:** comms 141/142 (native logits)

TL;DR — GPU build **SUCCEEDED** and offload is confirmed, but on this box the
14B/3060 combo is **NOT >=3x faster** than the CPU path (per-frame ~2.5-3.0s vs
CPU 1-4s; memory-bandwidth-bound). **Recommendation: keep the acceptance run on
CPU.** The checkpoint runner works on both engines and is validated end-to-end.

---

## (A) The CUDA build — official source only

| Item | Value |
|---|---|
| Package | `llama-cpp-python==0.3.34` (PyPI sdist, abetlen — matches Windows CPU env) |
| Source | **compiled locally from PyPI sdist** (`--no-binary llama-cpp-python`). No fork wheels. The quarantined JamePeng wheel was NOT touched. |
| Bundled engine | llama.cpp / ggml **0.16.0** (commit `e3546c7`) |
| Venv | `/home/reyses/venvs/llamacpp-cuda` (python 3.12.3, created with `uv venv --seed`) |
| CUDA toolkit | **system** `/usr/local/cuda` = CUDA **13.3.33** (already apt-installed; no pip nvidia packages needed) |
| Host compiler | gcc/g++ 13.3.0 · cmake 4.4.0 (pip) · ninja |
| Build flags | `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=off"`, `CUDACXX=/usr/local/cuda/bin/nvcc`, `CMAKE_BUILD_PARALLEL_LEVEL=4` |
| Target arch | **sm_86** only (RTX 3060, compute 8.6). Load log: `CUDA : ARCHS = 860` |
| Runtime linkage | `libggml-cuda.so` → `libcudart.so.13`, `libcublas.so.13` from `/usr/local/cuda`, `libcuda.so.1` from `/usr/lib/wsl/lib` (driver passthrough) |

Build wall time ~15 min (417 ninja targets, nvcc kernels the slow part). Exit 0.

### Env activation + LD_LIBRARY_PATH (exact lines AG needs)
```bash
source /home/reyses/venvs/llamacpp-cuda/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
(That is the only runtime env needed — the CUDA libs resolve from the system
toolkit. No pip-nvidia `site-packages/nvidia/*/lib` involved on this box.)

---

## (B) Offload + VRAM (verified)

- **`load_tensors: offloaded 41/41 layers to GPU`** with `n_gpu_layers=-1`.
- `CUDA0 model buffer = 8423 MiB`, `KV buffer = 1280 MiB` (n_ctx 8192, f16),
  compute buffer `700 MiB` (no flash) → **307 MiB with `flash_attn=True`**.
- **VRAM peak = 11,668 MiB** (baseline 1,152 → delta **10,516 MiB**), measured via
  `nvidia-smi` sampling during load. This is **near the 12 GB ceiling** — full
  offload leaves ~0.6 GB headroom on top of the Windows desktop's ~1.1 GB.

### OOM fallback ladder (if VRAM-tight or sharing the GPU)
`n_gpu_layers=-1` (all 41) → `35` → `30`. With `flash_attn=True` full offload is
stable for the small logit-readout evals; the danger is HOST RAM, not VRAM (see §D).

---

## (C) Logit sanity — native readout CONFIRMED on GPU

Method (comms 141/142): closed `</think>` trace so the next token is the true
answer; `create_completion(max_tokens=1, logprobs=50)`; read EXIT & HOLD.
**Both candidate tokens returned as real, non-floor logprobs:**

```
[hold-ish] EXIT=-12.1960 (found=True)  HOLD=-4.1183 (found=True)  P(EXIT)=0.0003
[exit-ish] EXIT=-1.2902  (found=True)  HOLD=-3.3930 (found=True)  P(EXIT)=0.8912
```

Real end-to-end through the **checkpoint runner** on CUDA (tiny packets):
```
ep_A f0  P(EXIT)=0.006  (HOLD  — trade just opened, at mean)   logits EXIT=-8.83  HOLD=-3.79
ep_A f1  P(EXIT)=0.998  (EXIT  — +45 ticks, target hit)        logits EXIT=-0.33  HOLD=-6.47
ep_B f0  P(EXIT)=1.3e-5 (HOLD  — fresh, indecisive)            logits EXIT=-15.58 HOLD=-4.31
```
Semantically correct, both tokens measured every frame. Tokenization note: the
answer tokens come back keyed as whole tokens `EXIT` / `HOLD` (the "HOLD → H"
split does not occur at this position because the closed-`</think>` assistant
turn forces a single answer token).

---

## (D) Benchmark + the speed verdict

Model: qwen3:14b Q4_K_M (8.63 GiB) from `/mnt/d/ollama/.../blobs/sha256-a8cc1361…`.

| Metric | GPU (RTX 3060) | Note |
|---|---|---|
| **Cold model load** | **272–338 s** | dominated by the 9.3 GB blob read over the `/mnt/d` 9p filesystem, NOT GPU. One-time per process. |
| **Per-frame eval (~2500 new tok)** | **2.5–3.0 s** (~800–1000 tok/s) | frame0 3.00s, frame1 2.48s, frame2 2.74s on a cumulative prompt |
| Warm re-eval (identical prompt, full KV hit) | 0.008 s | degenerate — real frames always add new tokens |
| First-inference CUDA-graph warmup | ~8 s one-time / process | folded into the first frame |
| Episodes before ctx tripwire | 3 frames | matches design (each frame ~2500 tok, `num_ctx=8192`) |

**CPU baseline (given): 1–4 s per eval.** GPU per-frame 2.5–3.0 s sits **inside**
that band. The RTX 3060 is memory-bandwidth-bound (360 GB/s) on an 8.6 GB model,
so prefill throughput (~900 tok/s) is not dramatically above the Ryzen 5 5600X.

> **Verdict: GPU is NOT >=3x faster for this 14B / single-token-logit workload.**
> Combined with the 5-min cold load (9p) and the host-RAM requirement below,
> **do not migrate the acceptance run to GPU.** Keep it on CPU (Windows, 16 GB).
> The GPU env remains available and is the right target for heavier future use
> (LoRA/distillation), where sustained generation amortizes the load cost.

### The host-RAM trap (important, and a latent bug in `eval_native.py`)
`create_completion(logprobs=…)` **requires `logits_all=True`** (llama.py:1358
raises otherwise). `logits_all=True` allocates an all-positions logit buffer
`n_ctx * n_vocab * 4B ≈ 8192 * 151936 * 4 ≈ 5.0 GB` of **host** RAM. The default
WSL VM here is only ~7.7 GB → this **OOM-crashed the WSL VM** (dxgk driver fault)
three times during long-prompt prefill before the cause was found. The Windows
CPU env (16 GB) is unaffected — which is why AG's CPU batch runs fine.

If a GPU acceptance run is ever wanted, raise WSL RAM first:
`%USERPROFILE%\.wslconfig` →
```ini
[wsl2]
memory=12GB
```
then `wsl --shutdown`. (Not created by this task — flagged for the owner, since
it is a global WSL change. The CPU speed verdict makes it unnecessary for now.)

---

## (E) Crash-safe checkpointed eval — `pipeline/eval_native_ckpt.py`

New file next to `eval_native.py`; the running/stopped batch was never touched.

### Checkpoint file format
`research/dojo_forge/gate_state/acceptance_results.jsonl` — **one JSON line per
COMPLETED episode**, appended immediately with `flush()` + `os.fsync()`:
```json
{"episode_id","engine":"cpu|cuda","model","num_ctx":8192,"tainted","taint_reason",
 "exit_frame","n_frames_evaluated","elapsed_s","ts",
 "frames":[{"frame_idx","p_exit","logit_exit","logit_hold","prompt_tokens","decision"}]}
```
A companion `reports/acceptance_native_gen0.csv` (the 142 table:
`eid,frame_idx,p_exit,prompt_tokens,tainted`) is **rebuilt from the jsonl on every
start**, so it is always consistent with the checkpoint and never a source of truth.

### Resume semantics
On start it reads the jsonl, collects completed `episode_id`s, and **skips** them;
only unfinished episodes run. A truncated final line from a hard kill is tolerated
(that one episode simply reruns). Episode-level granularity: a crash loses at most
the in-flight episode, never prior ones. Kill any time, rerun, it continues.

### 142 guards baked in (engine-agnostic)
1. `num_ctx = 8192`.
2. **ctx tripwire** — `prompt_tokens >= 8192` → frame hard-fail, episode tainted.
3. **top-N floor guard** — if EITHER `EXIT`/`HOLD` is absent from the returned
   top-N logprobs, that frame hard-fails (never records a floored −100 as a
   probability). Episode tainted with reason.

### How AG invokes it
```bash
# CPU (Windows — RECOMMENDED, matches the current confirmed method)
<dojo_forge>\.venv\Scripts\python.exe pipeline\eval_native_ckpt.py --engine cpu

# CUDA (WSL — only if the .wslconfig RAM bump above is applied)
source /home/reyses/venvs/llamacpp-cuda/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python pipeline/eval_native_ckpt.py --engine cuda
```
`--engine` selects `n_gpu_layers` (cpu=0 / cuda=−1, `flash_attn` on for cuda) and
the default blob path (Windows `D:\…` / WSL `/mnt/d/…`). Both engines write the
SAME jsonl, so a CPU run can be stopped and a CUDA run resumes it (or vice-versa).
Extra flags: `--limit N` (run N new episodes), `--dry-run` (mock, no model),
`--model-blob / --n-gpu-layers / --ckpt / --csv / --packets-dir` overrides.

### Validation
- **Dry-run smoke** (mock, Windows py): pass1 ran 2 eps → pass2 skipped those 2,
  ran the next 2 → 4 total; CSV rebuilt consistent; truncated-line tolerated.
- **Real CUDA** (tiny packets, full inference): ran ep_A(2 frames)+ep_B(1) with
  correct P(EXIT) values written to jsonl; **rerun skipped both** ("2 of 2 already,
  0 to run"). End-to-end proven: load → logit extraction → append → resume.

---

## (F) State of the stopped CPU batch's output
`research/dojo_forge/reports/acceptance_native_gen0.csv` currently holds **only
the header row — 0 completed episodes**. `eval_native.py` opens it with mode `'w'`
(truncate-on-start), so its restart/stop wiped any prior progress. **Nothing is
resumable** from it, and the checkpoint runner cannot import it as pre-completed
episodes (empty, and a different per-frame CSV shape vs the per-episode jsonl).
This is exactly the failure the checkpoint runner removes. **Do NOT auto-import**
— per directive, and there is nothing to import anyway. The full 156-episode
acceptance run should be (re)launched through `eval_native_ckpt.py` from scratch;
from here on a stop/crash costs at most one episode.

---

## Blockers / notes for AG
- None blocking. GPU env is built, offloads, and produces correct logits.
- The `/mnt/d` 9p cold-load (~5 min) and WSL 7.7 GB RAM limit are the two GPU
  frictions; both are avoided by staying on CPU, which is also ≈ as fast per frame.
- 156 packets present in `reports/gen0/packets/`. Launch:
  `<.venv>\Scripts\python.exe pipeline\eval_native_ckpt.py --engine cpu`.
