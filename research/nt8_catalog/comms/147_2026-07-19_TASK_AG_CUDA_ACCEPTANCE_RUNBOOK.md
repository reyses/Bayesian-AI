# TASK 147 — AG runbook: CUDA acceptance run (156 episodes, checkpointed)
**Doc:** 147 · **Date:** 2026-07-19 · **Author:** Claude Fable (reviewer) · **Status:** TASK · **Executor: AG**

Owner directive: reviewer is near usage limit — AG executes the full acceptance
run end-to-end using this runbook. Everything is prepared; follow EXACTLY.

## State (already done — do not redo)
- WSL CUDA llama-cpp-python 0.3.34 built from OFFICIAL source: venv
  `/home/reyses/venvs/llamacpp-cuda` (verified: 41/41 offload, sanity PASS).
- Model blob copied to ext4: `~/models/qwen3-14b-q4.gguf` (9,276,184,896 bytes)
  — use THIS path, not /mnt/d (9p is 4-6 min slower per load).
- Checkpoint runner `research/dojo_forge/pipeline/eval_native_ckpt.py` (v2,
  last-logits readout — the logprobs/logits_all 5 GB buffer path is REMOVED;
  do not reintroduce create_completion(logprobs) anywhere).
- CPU engine is DEPRECATED for this run (bandwidth math: GPU pp ~4-5× CPU).
  Old batch progress = zero (nothing to import).

## Step 1 — CUDA smoke (2 episodes)
```bash
cd /mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge
source /home/reyses/venvs/llamacpp-cuda/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python pipeline/eval_native_ckpt.py --engine cuda \
  --model-blob ~/models/qwen3-14b-q4.gguf --limit 2
```
ACCEPT the smoke only if ALL hold (paste evidence lines in your report):
1. Load log shows 41/41 layers offloaded to CUDA0; nvidia-smi VRAM ~11.x GB.
2. `gate_state/acceptance_results.jsonl` gains 2 lines, each with
   `readout":"last_logits_v2"`, real EXIT/HOLD logprobs (no nulls/floors),
   `prompt_eval_count` < 8192, and an `elapsed` field.
3. Re-running the same command prints a resume line skipping those 2 episodes.
If CUDA OOMs at load: retry with `--n-gpu-layers 40`, then 35, then 30
(record which). If the WSL VM itself dies (rare now): report, do not loop.

## Step 2 — full run
Same command WITHOUT `--limit`. Run it inside tmux/nohup so an SSH/console
drop doesn't kill it. ONE instance only — check `pgrep -af eval_native_ckpt`
before any (re)launch; the jsonl is the record and resume-with-skip makes
relaunch after any crash safe (cost ≤ 1 episode).

## Step 3 — monitoring + ETA
Every ~15 min: `wc -l gate_state/acceptance_results.jsonl`. After ≥5 episodes
compute episodes/hour from the `elapsed` fields and report the ETA once in
your interim comms. Expected ballpark: 15-40 s/episode → ~1-2 h total. If
observed rate is >5 min/episode, something is wrong (likely no KV prefix
reuse or silent CPU fallback — check the load log for CUDA0 lines) — stop,
diagnose, report; do NOT let a degenerate run burn the day.

## Step 4 — completion deliverable (comms 148+, 142-format)
1. Acceptance table (the runner's CSV, regenerated from the jsonl).
2. Per-episode `prompt_eval_count` distribution (min/median/max; assert max
   < 8192; any tripwire hits listed episode-by-episode).
3. Episodes/hour + total wall time + engine evidence (offload lines).
4. P(EXIT) sanity: distribution summary (a run where every P saturates 0/1
   or every P ≈ 0.5 is suspect — say so if seen, don't bury it).
5. Deviations from this runbook, each justified.

## Standing rules
Tainted gemma gen-0 stays OUT of everything (146 §2 — no control-arm reuse).
No gate arithmetic on these results — acceptance is descriptive only.
Commit the jsonl + CSV + your comms doc; nothing else.

## ADDENDUM (VRAM guardrails added to the runner — Moises directive)
eval_native_ckpt.py now has REAL VRAM protection (was missing; the 500MB cap
was HOST-RAM only):
1. `preflight_vram()` runs on --engine cuda BEFORE model load: nvidia-smi
   free-VRAM query -> fits n_gpu_layers to actual free VRAM with a 1.6GB
   margin (KV + compute + display). Auto-reduces from -1 (all 41) if a
   display/other proc holds VRAM; ABORTS if <20 layers fit (refuses a fake
   "cuda" run). Prints the budget math. If nvidia-smi is missing it WARNS and
   proceeds with the request (never silent).
2. n_batch 512 -> 256: prompt processed in smaller CHUNKS, bounding the VRAM
   compute spike.
Effect on the runbook: the Step-1 smoke load line now prints a `[vram]` line;
ACCEPT only if it shows USING >=35 blocks (else free VRAM first). If it
auto-reduces below ~35, per-frame speed drops (CPU spillover) — note it.
Everything else in this runbook stands.
