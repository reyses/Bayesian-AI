---
name: teacher-student-northstar
description: The program north star (Moises 2026-07-18) - specialized local LLM branch teaches Mamba via soft-label distillation; gates and sequencing
metadata:
  type: project
---

Moises (2026-07-18): "make a specialized gemma branch or qwen for our use case
... then we can teach Mamba which is faster by a bunch to do the actual trading."

Pipeline: genome evolution (free, running) -> GATE: held-out ride-edge proven
-> QLoRA specialize the branch (unsloth, 8B comfortable on 12GB) -> teacher
annotates thousands of days offline (per-frame P(EXIT)/P(ride) SOFT labels;
requires the native llama.cpp logprob path) -> Mamba distills (KL-to-teacher
primary loss, reward polish; far cheaper than RL exploration) -> light RL
fine-tune -> ONNX deploy (rl_whitepaper section 5).

**Why:** LLM = slow smart teacher; Mamba = fast student executor. Distillation
resolves the Mamba GPU-cost concern (supervised >> RL exploration cost).
**Why gated:** both dojos show LLM ~wash on cut side; the plan rests on the
RIDE side showing genome edge. If it never does, the plan dies at the cheap
gate. Related: [[telescope-nested-cadence]], docs 107/108, PRODUCTION_RUN_SPEC.
