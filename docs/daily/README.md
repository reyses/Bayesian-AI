# Daily Journals — `docs/daily/`

Per-session change log. The first thing Claude reads at the start of a session
and the last thing Claude writes at the end. Survives context loss; future
sessions read these instead of reconstructing intent.

## Purpose

Per CLAUDE.md hard rule: **after ANY code change (feature, fix, refactor), write
a short exit report.** Daily journals are that report.

They answer four questions for a future reader (Claude or human):

1. **What changed** — narrative of the day's edits/decisions/findings.
2. **What files** — concrete paths so a reader can `git log` the diff.
3. **What to look for in the next run** — verification targets, regression risks.
4. **Expected impact on metrics** — $/day, AUC, run-time, or "zero (refactor)".

## Folder layout

```
docs/daily/
├── README.md          ← this file
├── INDEX.md           ← one-line summary per day, newest first
└── YYYY-MM-DD.md      ← one per active session day
```

## File naming

`YYYY-MM-DD.md`, ISO date. One file per *calendar* day on which work happened.
Multiple sessions on the same day **append** sections to the same file —
do not create `YYYY-MM-DD_v2.md`.

## What a journal entry looks like

A journal is a markdown file with one numbered section per logical chunk of
work. Each section follows the same four-question template:

```markdown
## N. Short title of the chunk

### What Changed
Narrative paragraph(s). State the decision, the surprise, the trade-off.

### What Files Changed
- `path/to/file.py` — one-line summary of the edit
- `path/to/other.py` — same

### What to Look for in the Next Run
- Specific assertion (file exists, import resolves, $/day within CI)
- Failure mode to watch for and the revert command

### Expected Impact on Metrics
**$XX/day** OR **Zero (pure refactor)** OR **TBD until <gate>**.
Always state the expected delta, even if zero.
```

Multi-topic days have multiple sections (§1, §2, …, §N). Each section is
self-contained — a reader who only cares about one chunk doesn't need to read
the others.

## INDEX.md

`INDEX.md` is the **table of contents**. One row per day, **newest first**:

```markdown
| Date | Summary |
|------|---------|
| 2026-05-24 (training reorg + run_strategy.py) | One-paragraph hook. Link: `docs/daily/2026-05-24.md`. |
| 2026-05-22 (trade-outcome suite) | One-paragraph hook. … |
```

When reading `INDEX.md` to get session context (CLAUDE.md hard rule "Start of
session: read `docs/daily/INDEX.md`"), the goal is the hook, not the full
journal. Hook is 200–500 chars max. Drill into the dated file only when the
session needs the detail.

## When to write

- **End of every session** — full journal entry under today's date, plus a
  new `INDEX.md` row at the top.
- **After any code change**, even a one-liner. The change-report rule has no
  size threshold.
- **After a research finding** that won't be redone — capture the verdict and
  the data shape so the next session doesn't replicate the work.

## What does NOT belong in a journal

- The contents of `MEMORY.md` (durable rules, project state) — those go in
  `docs/memory/`. Journals are *chronological*; memory is *topical*.
- Long tables of raw data — write those to `reports/findings/` and link from
  the journal.
- Speculation about future work — that goes in `docs/memory/ROADMAP.md`.
- A copy of the diff — the reader has `git log`.

## Editing old journals

Older journals are **frozen by convention**. If a fact in a prior journal turns
out wrong, do not retroactively edit — write the correction in **today's**
journal with a back-reference (`see 2026-05-20.md §4 — that conclusion was
wrong because …`). The chronological record stays honest.

The only acceptable retroactive edits are typo fixes and broken-link repairs.

## Related files

- `docs/memory/MEMORY.md` — durable, non-chronological knowledge (rules,
  current state, references). See [memory README](../memory/README.md).
- `Jules_instructions/` — task briefs handed to Google Jules runs. See
  [Jules_instructions README](../../Jules_instructions/README.md).
- `docs/reference/RESEARCH_JOURNAL.txt` — long-form research notes, separate
  from these daily journals.
