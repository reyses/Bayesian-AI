---
name: feedback-session-promote-ritual
description: End-of-session learning-loop — promote corrections/patterns/decisions to dated memory, then rebuild the DERIVED FTS mirror; how to query memory before a task
metadata:
  node_type: memory
  type: feedback
---

# Session promote ritual (the memory learning-loop)

Adapted from the Hermes memory-loop blueprint, honoring the owner's constraint:
**SEGMENTS, NOT REWRITES.** This is an ADD-ONLY loop layered on top of the
existing journaling scaffold. It does not restructure, rename, or rewrite any
existing memory/journal file. The four memory tiers already exist here — this
file only adds the *loop* (how insights move from journal → durable memory) plus
a queryable mirror.

## The four tiers (already satisfied — noted, not changed)
- **Stable** (identity / ways of working) → `CLAUDE.md` + `docs/memory/feedback-*.md`,
  `USER_PERSONA_AND_PROTOCOL.md`, `AGENT_FEEDBACK_RULES.md`.
- **Context** (durable facts / decisions / learned patterns) → `docs/memory/MEMORY.md`
  sections + `project-*.md` / `reference-*.md` detail files + `PROJECT_HISTORY.md`.
- **Volatile** (per-session running log) → `docs/daily/YYYY-MM-DD.md`,
  `docs/daily/INDEX.md`, `docs/reference/RESEARCH_JOURNAL.txt`.
- **Loop** (how memory gets written) → *this file*.

## End-of-session checklist (the promote step)
Run at session wrap-up (this is IN ADDITION to the existing HARD-RULES journal
updates — it does not replace them):

**Reusability gate (effective 2026-07-21) — apply BEFORE promoting anything.**
Promote to `docs/memory/` ONLY what a FUTURE session can act on: durable facts,
decisions, preferences, patterns, reusable tools. Litmus test: "would a session
weeks from now reuse this?" No → it stays in the daily journal (`docs/daily/`),
not durable memory. One-off task logs, run-specific numbers, and conversation-scoped
detail are journal material, not memory. This gate also governs pruning: a memory
no future session could reuse is a cleanup candidate (still append-not-delete for
`MEMORY.md` / `PROJECT_HISTORY.md`).

1. **Corrections** the user made this session → one dated line each. Feedback →
   a `docs/memory/feedback-*.md` file (or append a dated note); a factual fix →
   the relevant `MEMORY.md` section (append with date, never delete — CLAUDE.md rule).
2. **Patterns that worked** and are worth repeating → dated line in the matching
   `project-*`/`feedback-*` memory file.
3. **Decisions with lasting impact** → `MEMORY.md` (append, dated) and/or the
   active roadmap.
4. **Unfinished threads** stay in the volatile journal (`docs/daily/`), not promoted.
5. **Single source of truth = `docs/memory/`** (updated 2026-07-21, Linux migration).
   The old dual-copy sync (private `~/.claude/projects/<hash>/memory/` → repo via the
   `pre-commit` hook) is RETIRED: that hook pointed at the Windows project-hash path,
   is a no-op on Linux, and was disabled (`.git/hooks/pre-commit.disabled`). Edit
   `docs/memory/` files directly — they are authoritative. Do NOT rely on any private-
   dir mirror; the Linux private memory dir is empty and unused.
6. **Rebuild the derived mirror** (last, after the markdown edits):
   ```
   python3.11 tools/memory_loop/build_memory_db.py
   ```

## The derived mirror is STRICTLY DERIVED
`docs/memory/memory.db` (SQLite + FTS5) is rebuilt idempotently from the markdown
files by `build_memory_db.py`. It is **never** a source of truth and **never** a
write-target for content. All promotion happens in the markdown files first; the
DB is regenerated from them. It is gitignored. Never edit memory by writing to the
DB — the dual-copy sync hook must never see a DB-born edit (there are none).

## Query memory BEFORE acting on a task (retrieval step)
Instead of loading a whole memory file, FTS-query the mirror for the task's terms:
```
python3.11 tools/memory_loop/query_memory.py "<terms>" [--limit N] [--tier stable|context|volatile]
```
Then read the named `source_file` for full context on any hit. Periodically run
`python3.11 tools/memory_loop/memory_stats.py` to surface recurring-correction
candidates (promote/consolidate them) and stale file references (flag only).

See `tools/memory_loop/README.md` for the full tool contract.
