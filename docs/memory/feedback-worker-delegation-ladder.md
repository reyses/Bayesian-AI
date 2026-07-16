---
name: feedback-worker-delegation-ladder
description: Standing ways-of-working protocol (Moises 2026-07-16) — tiered subagent delegation to minimize Fable usage; Fable = spec + verdict only
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 49f1ab8b-f170-41ec-955f-86beb538417f
---

**Worker delegation ladder (ratified by Moises 2026-07-16 — "added as your ways of
working protocol"). Goal: reduce Fable usage as much as possible.**

- **Fable (main session)**: orchestration, written specs, final verdicts, statistical-trap
  checks (lookahead, pseudo-replication), anything touching the frozen FPS core.
  Role = spec → review artifacts, not hands-on building.
- **Opus workers** (Agent tool, model: opus): substantial self-contained builds/analyses
  from a written spec (new tools, studies, interpretation). Light review tax.
- **Sonnet workers**: mechanical verifiable runs — frozen-script sweeps, grids,
  fan-out searches, formatting. Output must be checkable from artifacts alone.
- **Haiku workers**: watchers, timers, polling (ag-watcher pattern).

**Why:** Fable tokens are the scarce resource; the Claude⇄AG numbered-doc protocol
(spec → execute → verify, [[project-claude-ag-review-protocol]]) already proved the
reviewer/executor split works — subagent workers remove the AG failure modes (false
completion claims, protocol drift) while keeping the verification discipline.

**How to apply:** for each new task, pick the LOWEST tier whose silent-error risk is
acceptable; always give workers a written spec + expected-artifact checklist; verify
artifacts before use (a result from unverified plumbing is not a result). It took
Moises three asks to get this answered — when he proposes a process change, answer
DIRECTLY and first, before continuing task work.

**ALT ladder — Fable unavailable (Moises 2026-07-16):** when Fable usage is exhausted
(or the session runs on Opus), **Opus takes the reasoner/orchestrator seat**: writes
the specs, issues verdicts, runs the statistical-trap checks — and delegates the
working tiers exactly as above (Sonnet = mechanical verifiable runs, Haiku =
watchers). Same verification discipline, same numbered-doc protocol; only the top
seat changes. Escalate back to Fable (or flag for Moises) anything touching the
frozen FPS core or a major program-level verdict.
