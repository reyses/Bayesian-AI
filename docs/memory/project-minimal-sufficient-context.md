---
name: project-minimal-sufficient-context
description: "Moises' context-design principle (2026-07-22) — reduce LLM/model context as much as possible WITHOUT losing data: compress through the TF hierarchy (log-decay retention), never by discarding; governs teacher harness, gen-1 packets, student features"
metadata:
  node_type: memory
  type: project
---

**Minimal-sufficient context (Moises, 2026-07-22, via Telegram bridge — ratified
design philosophy for the teacher-student program):**
"Reduce the context as much as possible WITHOUT losing data."

Compression happens through the timeframe hierarchy, never by discarding:
whatever leaves the fast view is already retained in the coarser TF's
aggregates ("telescopic both ways — the data is kept in the higher TF").

**The concrete ladder (owner-specified):**
- Sub-minute = ACTION context (keep only the current minute's tape);
  1m-and-up = DECISION context (keep ~20 bars of 1m/5m); HTF = STRUCTURE
  (pinned once in the anchor — measured: the packet builder already emits
  1h/4h/1D only in frame 0).
- Log-decay retention per TF: keep only the bars the next TF up hasn't yet
  absorbed (~2 coarser bars' worth): 1s until its 5s closes; 5s → ~6 bars;
  15s → ~8; 1m → ~10; 5m → ~6 … → context is ~CONSTANT at any episode depth
  by construction.

**Where it lands:**
- Teacher harness (implemented 2026-07-22): `research/dojo_forge/pipeline/
  eval_native_tiered.py` — anchor + 20-min 1m/5m history + full current tape;
  measured plateau ~9.8k tokens at any depth (vs 148k unbounded telescope).
  Spec: comms doc 149.
- Gen-1 packet builder + Mamba student feature stream: apply the full
  log-decay ladder at build time (doc 149 §gen-1 principle).

**Why:** the unbounded telescope grew ~2.2k tokens/min and hit ~148k on long
episodes (impossible for any ctx window); fixed-depth truncation censored the
late-ride frames where the pre-registered ride-length edge lives. This
principle gets full depth AND bounded cost, with no information loss in the
sense that matters (each granularity survives at the horizon it's decision-
relevant for). See [[project-telescope-nested-cadence]], [[ONBOARDING]].
