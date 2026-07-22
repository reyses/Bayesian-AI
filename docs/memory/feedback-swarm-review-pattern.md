---
name: feedback-swarm-review-pattern
description: WoW (Moises 2026-07-21) — when N independent items each need a judgment, fan out a swarm of cheap agents (one per item) with a structured-output verdict schema, synthesize into ONE decision table, and let the owner approve before any action
metadata:
  node_type: memory
  type: feedback
---

**Swarm-review-to-decide (ratified by Moises 2026-07-21 — "moving forward you should
have a WoW that talks about this").** The delegation-ladder companion for the specific
shape "I have N independent items and need to decide what to do with each."

## When to reach for it
Any batch of independent items each needing the SAME per-item judgment: root scripts
to keep/archive/delete, files to classify, findings to verify, streams to score,
candidates to triage. If the judgment is identical per item and items don't depend on
each other → swarm it. If it's one entangled decision → don't (a swarm can't see the
whole).

## The pattern
1. **One cheap agent per item** (Sonnet tier — this is the "fan-out searches / mechanical
   verifiable runs" rung of [[feedback-worker-delegation-ladder]]). Reviewer stays the
   orchestrator; workers never mutate anything.
2. **Force structured output** — give each agent a JSON verdict schema
   (`{item, purpose, evidence/referenced_by, category, verdict, confidence, reason}`),
   not prose. Then synthesis is a table, not a read-through.
3. **Ground every field in evidence** — each agent READS its item and GREPS the repo
   itself (independent look), reports subprocess/doc/cron references, not just imports.
   Claim–evidence coupling, same as the ladder.
4. **Bake in the project's own guardrails** — e.g. the CLAUDE.md "false-orphan" rule:
   conservative verdicts, DELETE only when provably dead, uncertain → KEEP/low-confidence.
5. **Reviewer synthesizes ONE decision table, sorted so the obvious KEEPs and safe
   ARCHIVEs separate at a glance.** Nothing moves until the owner signs off on the table.
   The swarm decides nothing — it gathers evidence; the owner + reviewer decide.

## Why
A swarm turns 33 sequential eyeball-judgments into one parallel pass with uniform,
evidence-backed verdicts, at cheap-tier cost — and the structured output means the
reviewer spends tokens on synthesis, not re-reading. It's the ladder's fan-out rung
made concrete for triage/cleanup decisions.

## How to apply (+ gotchas learned 2026-07-21)
- Use the **Workflow** tool (deterministic fan-out) with `parallel()`/`pipeline()` and
  `agent(..., {model:'sonnet', agentType:'general-purpose', schema})`. Requires explicit
  user opt-in ("set a swarm", "use a workflow").
- **Workflow `args` arrives as a STRING, not a parsed array** — `SCRIPTS.map` throws
  "not a function". HARDCODE the item list inside the script, or `JSON.parse(args)`
  defensively. (Cost us one instant-fail run.)
- Give each worker a `.filter(Boolean)` / null-fallback so one dead agent doesn't sink
  the batch; a failed agent → an explicit "review manually" row, never a silent drop.
- Concurrency caps ~16; a 33-item swarm runs in ~2–3 waves — fine, just not instant.
- **Owner-approval gate is non-negotiable for destructive follow-through** (moves/deletes):
  present the table, wait for the go, THEN act — see the NT8 live-deploy gate discipline
  for the same "present diff → explicit approval → act" shape.
