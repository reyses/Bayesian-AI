# TASK 104 — Hermes-style memory augmentation, as SEGMENTS (no rewrites)
**Doc:** 104 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Moises: "about the augments go ahead but as segments not rewrites." Source
blueprint: C:\Users\reyse\Downloads\hermes-memory-loop.md (read it; honor its
own guardrail — our existing structure already satisfies tiers 1-3; we adopt
ONLY the additive pieces). HARD CONSTRAINT: existing memory/journal files are
NOT restructured, NOT renamed, NOT rewritten. Add-only.

## Segments to add
1. **Derived FTS mirror** — `tools/memory_loop/build_memory_db.py`:
   - Scans (read-only): docs/memory/*.md (index + detail files + archive),
     docs/daily/INDEX.md (one row per line), docs/reference/RESEARCH_JOURNAL.txt
     (one row per dated entry), plus graveyard/§ blocks inside MEMORY.md.
   - Emits docs/memory/memory.db — SQLite table
     `learnings(id, date, tier, tag, source_file, text)` + FTS5 mirror
     (`content=` external-content pattern). tier ∈ stable|context|volatile
     inferred from source (CLAUDE/MEMORY=stable/context, INDEX/journal=volatile);
     tag = memory-file name / INDEX date / journal date.
   - **DB is strictly DERIVED**: rebuilt idempotently from the files; NEVER a
     write-source for content (the dual-copy sync hook must never see DB-born
     edits). Gitignore the .db (add the line; that gitignore append is allowed).
2. **Query helper** — `tools/memory_loop/query_memory.py "<terms>" [--limit N]`:
   FTS5 MATCH, prints date | tier | tag | text one-per-line. Fast, no deps
   beyond stdlib sqlite3.
3. **Analytics** — `tools/memory_loop/memory_stats.py`: recurring-correction
   candidates (near-duplicate feedback rows across dates via normalized-text
   grouping), stale-entry report (memories naming files that no longer exist —
   flag only, never delete). Output to stdout + tools/memory_loop/last_stats.md.
4. **Promote ritual segment** — ONE new memory detail file
   `docs/memory/feedback-session-promote-ritual.md` (standard frontmatter):
   the end-of-session checklist (corrections / patterns-that-worked / lasting
   decisions → dated entries; then rebuild the DB) + how to query before tasks.
   Plus ONE pointer line APPENDED to MEMORY.md — in BOTH copies
   (C:\Users\reyse\.claude\projects\c--Users-reyse-OneDrive-Desktop-Bayesian-AI\memory\MEMORY.md
   AND docs\memory\MEMORY.md) or the commit hook reverts it.
5. `tools/memory_loop/README.md` — what each script does, the derived-only
   rule, rebuild cadence (session end / on demand).

## Verify then stop
Build the DB from current files; report row counts per source; run 3 sanity
queries ("worker background", "lookahead", "telegram") and show hits; run the
stats tool and report its top recurring-correction candidate. Commit NOTHING.
Do not touch anything outside tools/memory_loop/, the one new memory file,
the two MEMORY.md pointer-line appends, and the .gitignore line.
