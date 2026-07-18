# Hermes-Style Memory Loop — Adaptation Spec (read + merge, do not overwrite)

> For Claude in VS Code: this is a **blueprint**, not a fresh install. The user already has
> journaling scaffolding, saved memories, and documented ways of working. Your job is to
> **map their existing pieces onto the four tiers below and reorganize in place** — preserve
> their content and wording; only restructure, dedupe, and wire the loop.

## Target architecture (the four tiers)

1. **Stable tier — identity & ways of working.** Rarely changes. Role, preferences,
   standing conventions, tone. → belongs in `CLAUDE.md` (top section).
2. **Context tier — long-term memory.** Durable facts, decisions, learned patterns.
   → `memory/learnings.md`, imported via `@` into CLAUDE.md.
3. **Volatile tier — the journal.** Per-session running log, open threads, scratch.
   → their existing journaling scaffold, normalized to `memory/session-log.md`.
4. **Loop tier — how memory gets written.** The rules that move insights from journal →
   long-term memory. → the "Learning-loop protocol" section below.

## Adaptation steps (do these against what already exists)

1. **Inventory.** List every existing instruction/memory/journal file. Do not delete anything.
2. **Classify each block** into tier 1–3 above. Ambiguous → ask the user before moving.
3. **Segment:** move stable "ways of working" to CLAUDE.md; durable memories to
   `learnings.md`; running journal to `session-log.md`. Keep original phrasing verbatim.
4. **Wire imports:** ensure CLAUDE.md pulls the two memory files via `@./memory/...`.
5. **Dedupe:** merge near-duplicate memories; flag conflicts for the user instead of guessing.
6. **Leave a migration note** at the top of any file you restructure, listing what moved where.

## Learning-loop protocol (the tier that makes it self-improving)

At session wrap-up (or on request), promote from journal → long-term memory:
- **Corrections** the user made this session.
- **Patterns that worked** and are worth repeating.
- **Decisions** with lasting impact.
Write each as one dated line in `memory/learnings.md`. Keep unfinished items in
`session-log.md`. Compress older entries periodically to respect token budget.

## Structured memory — SQL + DataFrames (tier 2, queryable form)

Flat markdown is the human-readable source of truth; mirror it into a queryable store so
Claude can search, dedupe, and compress at scale (this is how Hermes uses SQLite + FTS5).

- **Store:** `memory/memory.db` (SQLite). One table `learnings(id, date, tier, tag, text)`
  plus an FTS5 virtual table `learnings_fts` for full-text search over `text`.
- **Journal → DB:** on the loop step, INSERT each promoted learning as a row (don't just
  append to markdown). Markdown files stay as the readable export.
- **Retrieval:** before acting on a task, run an FTS5 query for relevant tags/keywords
  instead of loading the entire memory file — keeps token cost flat as memory grows.
- **DataFrame ops (pandas):** load the table with `pd.read_sql` to:
  - **dedupe** — group by normalized `text`, keep newest.
  - **compress** — cluster by `tag`, summarize old clusters into one row (quarterly).
  - **analyze** — frequency of corrections, recurring patterns, stale entries to prune.
- **Regenerate markdown** from the DB after any bulk op so `learnings.md` stays in sync:
  `SELECT` ordered by date → write back to the file.

### Explicit reference (declarative — describe intent, don't ship a helper)
This section is a **specification of shape and operations**, not code to run as-is.
Claude should generate the actual statements inline when a task needs them.

Schema (the exact shape memory should take):
```
CREATE TABLE learnings (
  id   INTEGER PRIMARY KEY,
  date TEXT NOT NULL,          -- YYYY-MM-DD
  tier TEXT NOT NULL,          -- 'stable' | 'context' | 'volatile'
  tag  TEXT,                   -- topic/domain for scoped retrieval
  text TEXT NOT NULL           -- one-line insight
);
CREATE VIRTUAL TABLE learnings_fts USING fts5(text, content='learnings', content_rowid='id');
```
Named operations (intent → the query/op Claude produces):
- **retrieve** — `SELECT text FROM learnings_fts WHERE learnings_fts MATCH :q` (scoped, not full-load).
- **promote** — `INSERT INTO learnings(date,tier,tag,text) VALUES (...)` on the loop step.
- **dedupe** — `pd.read_sql(...).drop_duplicates("text", keep="last")`, write back.
- **compress** — group by `tag`, summarize stale clusters into one row (quarterly).
- **export** — `SELECT ... ORDER BY date` → regenerate `learnings.md`.

Keep the DB optional: if the user's memory is small, flat markdown alone is fine. Introduce
SQL only when memory volume or retrieval cost justifies it.

## Guardrails
- Preserve the user's existing journaling format if it already works — adapt the tiers to it,
  not the reverse.
- Never fabricate memories. Only record what actually occurred in-session.
- When their current structure already satisfies a tier, note that and change nothing.

## Suggested file map (rename only if the user agrees)
```
CLAUDE.md                     # tier 1 + @imports of the memory files
memory/learnings.md           # tier 2 — long-term (readable export)
memory/memory.db              # tier 2 — SQLite + FTS5 (queryable source)
memory/session-log.md         # tier 3 — journal / volatile
.claude/skills/record-learning/SKILL.md   # optional: on-demand capture
```
