# tools/memory_loop — derived FTS mirror of the memory corpus

Hermes-style memory augmentation, added as **SEGMENTS, NOT REWRITES**. Nothing
here restructures, renames, or rewrites any existing memory/journal file. It adds
a *derived* queryable mirror + analytics + a promote ritual on top of the existing
journaling scaffold.

## The derived-only rule (read this first)
`docs/memory/memory.db` is **STRICTLY DERIVED**. It is rebuilt idempotently from
the markdown/text files every time `build_memory_db.py` runs (drop + recreate +
reinsert). It is:
- **never** a source of truth — the markdown files are authoritative;
- **never** a write-target for content — you never edit memory by writing to the DB;
- **gitignored** (`docs/memory/memory.db`) — regenerate it, don't commit it.

The dual-copy `MEMORY.md` sync hook must never see a DB-born edit, because there
are none: this tooling only *reads* the source files and *writes* the `.db`.

## Scripts

### `build_memory_db.py` — build the mirror
```
python3.11 tools/memory_loop/build_memory_db.py [--db PATH] [--quiet]
```
Scans (read-only) and emits `docs/memory/memory.db`:
- `docs/memory/*.md` — `MEMORY.md` split by `##` section (one row each, incl. the
  GRAVEYARD / TRAPS / § blocks); every other detail file = one row.
- `docs/memory/archive/*.md` — one row per archived file.
- `docs/daily/INDEX.md` — one row per dated table line.
- `docs/reference/RESEARCH_JOURNAL.txt` — one row per dated entry.

Schema:
```
learnings(id, date, tier, tag, source_file, text)
learnings_fts  -- FTS5 external-content mirror over text (content='learnings')
```
**Tier inference** (stable | context | volatile):
- `volatile` — INDEX rows + RESEARCH_JOURNAL entries (per-session logs).
- `stable` — memory files with frontmatter `type: feedback`, plus
  `USER_PERSONA_AND_PROTOCOL.md`, `AGENT_FEEDBACK_RULES.md`, `ce_methodology.md`,
  and the HARD-RULES / METRIC-DEFINITIONS / USER-PROFILE `MEMORY.md` sections.
- `context` — everything else in `docs/memory` (`type: project`/`reference`,
  `PROJECT_HISTORY.md`, remaining `MEMORY.md` sections, archive dumps).

**Tag**: memory-file stem, or `MEMORY#<section-slug>`, or the INDEX/journal date.
**Date**: explicit for INDEX/journal rows; first date found in the text (else file
mtime) for memory files.

### `query_memory.py` — retrieve before acting
```
python3.11 tools/memory_loop/query_memory.py "<fts terms>" [--limit N] [--tier T] [--full]
```
FTS5 `MATCH`, ranked by relevance. Prints `date | tier | tag | text` one per line
(truncated unless `--full`). Multi-word = implicit AND; supports `"phrase"`, `OR`,
`NOT`, `prefix*`. stdlib `sqlite3` only. Read the hit's `source_file` for full context.

### `memory_stats.py` — analytics
```
python3.11 tools/memory_loop/memory_stats.py [--db PATH] [--jaccard 0.5]
```
Writes stdout + `tools/memory_loop/last_stats.md` (derived, overwritten each run):
- **Recurring-correction candidates** — near-duplicate rows grouped across dates
  (normalized-token Jaccard union-find; clusters spanning ≥2 dates). Surfaces a
  lesson learned/re-learned more than once → promote or consolidate it.
- **Stale-entry report (FLAG ONLY)** — file paths named in *active* memory
  (stable/context, excl. archive) that no longer exist on disk. Never deletes.
  Volatile journal/INDEX + archive dumps are excluded: they log deleted files by
  design, so flagging them just reproduces known project churn.

## Rebuild cadence
- **Session end** — after promoting corrections/patterns/decisions into the
  markdown files (see `docs/memory/feedback-session-promote-ritual.md`), rebuild
  the DB last so it reflects the new markdown.
- **On demand** — before a task, rebuild (cheap) then `query_memory.py` the task's
  terms instead of loading whole memory files.

## Requirements
- `python3.11` (bare `python` hangs in this env). stdlib only — `sqlite3` with FTS5
  (verified present: sqlite 3.45.x). No pandas / no third-party deps.
