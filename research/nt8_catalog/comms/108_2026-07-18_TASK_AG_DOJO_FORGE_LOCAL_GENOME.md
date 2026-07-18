# TASK 108 — DOJO FORGE: local-model generational dojo (AG execution package)
**Doc:** 108 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Executor:** AG
**Status:** TASK — Claude usage-constrained; AG executes under the Claude⇄AG
review protocol (comms/CLAUDE_AG_REVIEW_PROTOCOL.md: append-only reports, NO
self-certification, artifact-level evidence, verify-then-STOP at every gate).

## 0. Scaffolding adoption (the "Hermes-Moises" way of working — MANDATORY)
Before any code, AG reads and OPERATES UNDER:
- `docs/WOW_TEMPLATE.md` (the generalized WoW — §5 epistemic kit, §7 journals)
- `docs/memory/feedback-session-promote-ritual.md` + `tools/memory_loop/`
  (end-of-session: promote lessons, rebuild the FTS db)
- Journals: update `docs/daily/2026-07-18.md` + one INDEX line +
  RESEARCH_JOURNAL entry at every session end. Comms docs numbered 109+ for
  AG execution reports (append-only).
- Git: work on branch **`forge/genome-v0`** — NEVER commit to main (main is
  reviewer-gated). Commit early/often ON THE BRANCH with descriptive messages.

## 1. Context (read first)
- Doc 107: cut side CLOSED; the program's edge is RIDE-ONLY. The forge's
  target = ride-side hold policy (winners/midflips), metric = capture ratio.
- Doc 104: the memory-loop segments (genome philosophy).
- The dojo sandbox: research/exit_dojo/ (gate, packets, truth format, nonce
  audit). The forge REUSES packet/truth formats and transcript conventions.
- Hardware: RTX 3060 12GB; ollama installed with gemma4:latest (9.6GB) and
  gemma4:e2b (7.2GB) — GGUF blobs in the ollama store
  (`ollama show <model> --modelfile` reveals blob paths).

## 2. PHASE F1 — the local forge harness (build + smoke, then STOP)
New project folder `research/dojo_forge/` (WoW layout: pipeline/ tools/
reports/ README.md). Build `pipeline/forge_harness.py`:
1. **Engine**: llama-cpp-python with CUDA in the WSL venv
   (`/home/reyses/venvs/bayesian-ai` — install there; document the wheel/build
   used). Load the gemma GGUF directly from the ollama blob path
   (config constant; verify with `ollama show gemma4:e2b --modelfile`).
   n_gpu_layers=-1; deterministic: temperature=0, fixed seed.
2. **Prefix KV cache**: the genome+system prompt is identical across a
   generation → build once, `save_state`, `load_state` per episode. MEASURE
   and report the speedup vs cold prompts.
3. **Grammar-forced output**: GBNF constraining every decision to exactly
   `HOLD` | `EXIT: <free-text reason ≤ 40 tokens>`. Zero parse failures by
   construction.
4. **Logprobs**: record P(EXIT) (first-token logit contrast under the
   grammar) EVERY frame → per-episode confidence series in the transcript.
5. **In-process gate**: port dojo_gate's serve/commit/nonce semantics to an
   in-process iterator writing the SAME transcript.jsonl format (nonce chain
   intact) so score tools work unchanged. The model object receives ONLY the
   frame strings — no filesystem, no tools (blindness by construction; state
   this in the README).
6. **Fallback**: if llama-cpp-python fights the gemma arch on this build,
   fall back to the ollama HTTP API (localhost:11434/v1, temperature 0) — a
   config switch, not a redesign. Report which path is live.
**GATE F1 (STOP for review)**: 2 scripted-dummy episodes + 2 real gemma4:e2b
episodes end-to-end; artifact evidence: transcripts w/ nonce chain PASS, the
P(EXIT) series, prefix-cache timing table (cold vs cached), tokens/s.

## 3. PHASE F2 — genome + generation loop (build + gen-0, then STOP)
1. **Genome file** `research/dojo_forge/genome/GENOME.md` (git-versioned on
   the branch; one lesson per line):
   `- [G<gen>.<id>] IF <condition> THEN <action>. | born:g<gen> | ho_record:W-L`
   Sections: WITH-TREND / DIP-HANDLING / CHOP / GENERAL. Seed gen-0 from the
   exit-dojo grammar + doc-107 failure catalog (Claude-authored lessons
   already in reports/full_run/synthesis.md + comms/098/100/107).
2. **Episode source**: ride-side (winner|midflip) fresh 2025-26 days NEVER
   used by any prior dojo (exclusions: pilot 10 + full_run 200 + wrongdir 200
   day lists — build the exclusion set from their selection.json files).
   Held-out: 50 episodes, fixed, played by NO generation during evolution.
3. **Scoring**: capture ratio vs oracle + vs 5m-hold (reuse score_full_run
   conventions); logprob calibration curve (P(EXIT) vs realized outcome).
4. **Generation loop** `tools/run_generation.py`: play N=100 fresh episodes
   with the current genome → write generation report (mode-first stats +
   worst-10 failure tapes extracted verbatim for the distiller).
5. **Distillation**: AG does NOT self-distill lessons in v0. The generation
   report + failure tapes are the handoff artifact; the REVIEWER (Claude, on
   return) distills lesson diffs. (This keeps genome mutations reviewed —
   revisit after two clean cycles.)
**GATE F2 (STOP)**: gen-0 played (100 eps + the 50 held-out baseline run);
report with capture distribution, calibration, failure tapes; genome
committed on the branch; NO lesson edits.

## 4. Acceleration ladder (trigger-gated — do NOT pre-build)
- TensorRT-LLM engine swap ← only if a generation can't finish overnight.
- cuDF/RAPIDS for scans ← only when a scan exceeds ~15 min.
- Nemotron-8B third executor ← at genome-transfer test time (gen ≥ 2).
- LoRA distillation (unsloth) ← only after lessons stabilize 2 consecutive
  generations. Each rung needs its trigger MEASURED and reported first.

## 5. Hard rules (repeat-offenders list)
python3.11 on Windows / WSL venv python for CUDA; bare `python` hangs.
No lookahead: episodes are packets built by the EXISTING telescope builder
(closed-bars-only asserts stay). Truth never enters a prompt. No magic
numbers. Friction 2.4t/RT where economics are computed. Day-block CIs,
mode-first. Commit to the branch only. Report claims = artifact paths.
