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

## Acts 3+4 (added 2026-07-18, Moises)
**Act 3 - the ARENA**: the forge harness is executor-agnostic - Mamba plays the
same nonce-audited gate as the LLMs (tensor frames adapter). One arena for all
contenders (teacher variants, Mamba checkpoints, dumb baselines) on identical
held-out episodes. Two tracks kept distinct: PARITY (Mamba sees the tensor
equivalent of the LLM frames - the honest teacher-vs-student comparison) vs
NATIVE (full production state - the deployable measure). Mamba speed => arena
evaluates checkpoints on thousands of held-out eps => tight student CIs.
Steering-CI enforcement: any steered variant must post an arena score before live.

**Act 4 - the GLASS COCKPIT**: the teacher doubles as Mamba bidirectional
interpreter. OUTBOUND: verbalizes the instrument panel ONLY - named state
channels + P(EXIT) trajectory + mechanical per-channel counterfactual
attributions (computable for a small SSM; seed tool mamba_brain_scanner.py).
LAW: interpreter speaks only from instruments (claim-evidence coupling) -
confabulated "reasons" are the failure mode. INBOUND: compiles Moises-English
into genome lessons/conditioning flags -> steering-CI -> live. Chat surface =
the Telegram bridge. LLM stays OUT of the trade path (graveyard).

**The engine (Moises closing)**: the Hermes-like harness keeps EVERYTHING
slowly evolving - genome, teacher weights, Mamba checkpoints, steering rules -
one loop (play -> score vs dumb baselines on held-out -> distill -> commit ->
next gen), git as the fossil record, reviewer gates as selection pressure.
Nothing finished, nothing drifting unmeasured.

## Act 5 - the ACTUARY (added 2026-07-18, Moises)
The ORIGINAL genesis concept returns: bayesian_brain.py (2026-01-31, HashMap
StateVector->WinRate, recovered in research/exit_lineage_recovered/) becomes
the arena bookkeeper - every play by every executor registers
(state-bucket, action) -> outcome tallies. Three jobs: (1) probability
register P(outcome|tier x regime x action) with count-based CIs; (2)
calibration referee (models CLAIM P, the table records what HAPPENED per
bucket - the glass cockpit quotes both); (3) prior-setter for conditioning
channels. GUARD (graveyard section 3): registrar + calibrator, NEVER a
cell-picker - coarse structural buckets only, large-n cells, day-aware
counts; per-cell selection is the overfit trap.
