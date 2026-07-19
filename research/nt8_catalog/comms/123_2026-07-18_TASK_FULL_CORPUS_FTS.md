# TASK 123 — complete the database: full-corpus FTS ingestion + coverage QA
**Doc:** 123 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** TASK (Opus drone)
Moises: "go over all the files that we have and complete the database."
Method ruling: MECHANICAL ingestion (exact, free), not swarm re-summarization
— FTS searches raw text; the judgment layer (DISTILLED cards) already exists.

## Extend tools/memory_loop/build_memory_db.py (append-only sources; keep all
## existing behavior + tags byte-compatible)
New sources, each as PER-SECTION rows (split on ^## like the MEMORY.md
handler; fall back to whole-doc for files without sections; skip empty):
1. research/nt8_catalog/comms/*.md — tier=context, tag=comms:NNN (parse the
   leading number; the program's decision spine).
2. research/*/reports/**/*.md + research/archive/*/reports/**/*.md —
   tier=context, tag=report:<topic>. SKIP research/nt8_catalog/raw_articles*
   (scraped externals — bulk noise) and any assets/ dirs.
3. docs/daily/*.md (full journals, not just INDEX) — tier=volatile,
   tag=journal:<date>.
4. docs/northstar/*.md, docs/nt8/*.md (not archive subdir), docs/Active/*.md,
   docs/WOW_TEMPLATE.md, ROADMAP_LAMBDA_COMPLETION.md, rl_whitepaper.md,
   AGENTS.ini, research/*/README.md — tier=context, tag=doc:<name>.
5. research/dojo_forge/RIDE_EDGE_GATE_SPEC.md + PRODUCTION_RUN_SPEC.md —
   tier=stable (governing specs), tag=spec:<name>.
Guards: dedupe by (source_file, section-slug); total-size sanity (print MB
ingested); a --sources filter arg for partial rebuilds; keep the DB strictly
derived (regenerable, gitignored) — unchanged.

## QA (deliver in reports/… and your final message)
1. Coverage table: rows + MB per source class; total.
2. Ten probe queries with top-hit sanity: "lookahead searchsorted",
   "dumb stop never-bail", "FADEAGN inverted", "telegram bridge token",
   "gate spec lockbox", "warm-start selectivity gap", "wick 0.83",
   "dilution artifact", "$900 giveback", "teacher student distillation" —
   each must surface the RIGHT document class (say which file hit #1).
3. Perf: rebuild time + query latency (must stay sub-second on query).
4. File: tools/memory_loop/coverage_report.md.

## Rules
Edit ONLY build_memory_db.py (+ the coverage report; query_memory.py only if
a --tag filter is trivial to add). RUN SYNCHRONOUSLY; python3.11; commit
NOTHING. Final message: coverage table, the 10 probe results, perf, deviations.
