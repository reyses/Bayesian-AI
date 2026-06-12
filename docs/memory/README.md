# Project Memory — `docs/memory/`

> **CONDENSED 2026-06-12.** `MEMORY.md` is the condensed, load-on-start knowledge base (9 sections: program / hard rules / metric formulas / traps / graveyard / wins / architecture / user / pointers). The pre-condense originals of every file here are preserved verbatim in `archive/` (`*_pre-condense_2026-06-12.md`); `PROJECT_HISTORY.md` now opens with a chronological era arc.

Durable, non-chronological knowledge that future sessions need to load fast.
Where the journals (`docs/daily/`) record *what happened*, memory records
*what's true now* and *the rules to follow*.

## Index file

**`MEMORY.md`** is the index. It loads on every session start (per CLAUDE.md
hard rule). Keep it tight — each entry is one line, under ~150 chars:

```markdown
- [Short title](filename.md) — one-sentence hook describing why this matters.
```

Never write memory *content* into `MEMORY.md` itself — only pointers. Content
lives in topic files. `MEMORY.md` has no frontmatter.

## Memory file types

Naming is prefix-coded. The prefix tells you what kind of knowledge you're
about to read.

| Prefix | Meaning | Examples |
|---|---|---|
| `feedback_*` | A rule the user gave Claude, or a hard-won lesson from a mistake. Persists across sessions. | `feedback_v2_only_hard_rule.md`, `feedback_dollar_lift_framing.md` |
| `project_*` | Current state of an ongoing project / experiment. Updates as the work evolves. | `project_forward_pass_unification_2026_05_24.md`, `project_oos_bad_days_2026_05_21.md` |
| `user_*` | Information about how the user works, their domain expertise, preferences. | `user_collaboration_protocol.md` |
| `research_*` | Long-form research notes that don't fit a single finding report. | `research_zigzag_atr_calibration.md` |
| `tier_*` | Tier/strategy-specific facts (mostly historical V1 — being deprecated). | `tier_building_playbook.md` |
| `ce_*`, `waveform_*` | Domain-specific subjects (collaboration engagement, signal waveforms). | rare |

Special files:
- **`MEMORY.md`** — the index. Always loaded.
- **`ROADMAP.md`** — future topics backlog. Not loaded by default — read only
  when planning next work.

## File format

Every memory file has YAML frontmatter:

```markdown
---
name: short-kebab-case-slug
description: One-sentence summary used to decide relevance later. Be specific.
type: feedback           # one of: feedback, project, user, research, tier
originSessionId: <uuid>  # optional, identifies the session it was written in
---

Body content here. Markdown.
```

### Body structure by type

**`feedback_*` files** — lead with the rule, then explain why:
```
**Rule**: <the rule as imperative>

**Why**: <the user's stated reason, often a past incident>

**How to apply**: <when/where to invoke; edge cases>
```

**`project_*` files** — lead with current status, then context:
```
**Status (as of YYYY-MM-DD)**: <one-line state>

**What is this project**: <problem statement, scope>

**What's done**: <bullets>

**What's next**: <bullets>

**Open questions / risks**: <bullets>
```

**`user_*` and `research_*`** — free-form, but with a clear opening
paragraph that future Claude can skim in 5 seconds.

## Linking

Cross-link memories with `[[other-name-slug]]`. The slug is the `name:` field
from the target file's frontmatter. A `[[link]]` that doesn't resolve yet is
fine — it marks a future write.

Example: `See [[feedback-v2-only-hard-rule]] for the reasoning.`

## When to write a new memory

1. **The user gave a rule** ("never do X", "always do Y") → `feedback_*`
2. **A non-obvious lesson emerged from a mistake** → `feedback_*` (write
   *why* the mistake happened, not just "I broke X")
3. **A project state needs to persist** (ongoing experiment, partial result)
   → `project_*`
4. **Surprising user preference confirmed** ("yes, that's the right call")
   → `feedback_*`. Quieter signal than corrections, just as important.

Then add a one-line entry to `MEMORY.md`.

## When to UPDATE an existing memory

- Status changed (project finished, rule refined).
- New evidence overturned the prior conclusion — rewrite or mark deprecated.
- A `[[link]]` finally has its target.

**Always update `MEMORY.md`'s entry** if the file's purpose changed.

## When to DELETE a memory

- The rule got encoded into CLAUDE.md (now enforced at instruction-load time).
- The project finished and the lesson is documented elsewhere.
- A `feedback_*` rule turned out to be wrong and the user said so.

Don't delete to "clean up" — only delete when the memory is actively wrong or
fully superseded. Stale memories are usually better than missing memories.

## What does NOT belong in memory

- **Daily session work** — that goes in `docs/daily/`. Memory is for facts
  that survive the session, not the session itself.
- **Code documentation** — that goes in docstrings or `docs/Active/*.md`.
- **Long data tables** — write those to `reports/findings/`.
- **Speculation about future work** — `ROADMAP.md` is for that.

## Relationship to other state

- **`docs/daily/`** — *what happened*, chronological. See [daily README](../daily/README.md).
- **`docs/memory/` (this folder)** — *what's true now*, topical.
- **`CLAUDE.md`** (project root and `~/.claude/`) — *hard instructions* that
  override everything. Memory advises; CLAUDE.md commands.
- **`Jules_instructions/`** — task briefs for Google Jules runs.
- **`reports/findings/`** — analysis outputs, finding reports.

## Auto-memory (Claude's, not in repo)

Claude also has a file-based auto-memory at
`~/.claude/projects/<project-hash>/memory/`. That's a separate system, not
git-tracked, not visible to the user. The two coexist:
- **Auto-memory** is Claude's working memory across sessions for fast lookup.
- **`docs/memory/`** (this folder) is the project's durable record, reviewed
  by humans, git-tracked.

When in doubt, write to `docs/memory/` first — it's discoverable.
