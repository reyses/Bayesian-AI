# Memory FTS — full-corpus coverage report (doc 123 + code scope-ext)
**Date:** 2026-07-18 · **Builder:** `tools/memory_loop/build_memory_db.py` · **DB:** `docs/memory/memory.db` (strictly derived, gitignored)

Mechanical ingestion only — FTS indexes raw text (docs) + AST-extracted structure
(code). No model-generated summarization anywhere. Rebuild is idempotent
(DROP + CREATE + INSERT); DB is never a source of truth.

## Coverage table (authoritative full rebuild)
Total: **3037 rows**, **4.34 MB** indexed text, DB **7.09 MB** on disk.

| source class | rows | MB (indexed text) | tier | notes |
|---|---:|---:|---|---|
| memory      |  60 | 0.582 | context/stable | MEMORY.md per-§, detail files, agent segments, DISTILLED cards (pre-existing) |
| archive     |   8 | 0.528 | context | docs/memory/archive/*.md (pre-existing) |
| index       | 142 | 0.052 | volatile | docs/daily/INDEX.md dated lines (pre-existing) |
| journal_txt |  40 | 0.088 | volatile | RESEARCH_JOURNAL.txt dated entries (pre-existing) |
| **comms**   | 665 | 0.439 | context | research/nt8_catalog/comms/*.md per-§, tag=comms:NNN |
| **reports** | 692 | 1.196 | context | research/**/reports/**/*.md per-§, tag=report:<topic> (raw_articles + assets SKIPPED) |
| **daily**   | 936 | 1.012 | volatile | docs/daily/YYYY-MM-DD.md full journals per-§, tag=journal:<date> |
| **docs**    | 161 | 0.179 | context | northstar/nt8/Active + singletons + research READMEs, tag=doc:<name> |
| **specs**   |   9 | 0.010 | stable | dojo_forge RIDE_EDGE_GATE_SPEC per-§, tag=spec:<name> |
| **code**    | 324 | 0.251 | context | AST surface per module, tag=code:<repo-rel-path>, **0 parse-failures** |

by tier: context=1896, stable=23, volatile=1118.
Bold = new in this task. All pre-existing source handlers, tags, and tiers are byte-unchanged.

## Skips honored (per spec)
- `research/nt8_catalog/raw_articles*` (560 scraped external .md) — excluded from reports.
- any `assets/` dir — excluded from reports.
- code: `research/archive/**`, `__pycache__`, `venv`/`.venv`, `node_modules`, files >300 KB.

## Code layer (scope-ext) — mechanical AST extraction, per module = 1 row
Extracted VERBATIM, no summarization:
1. module docstring (first 40 lines),
2. every top-level class/function signature (`ast.unparse` of args + return) + first docstring line,
3. path-like string literals (contain `/`|`\` AND an extension or DATA/checkpoints/reports), deduped, max 20.
324 modules indexed, **0 parse-failures**, 0.251 MB. Modules with no docstrings keep signatures only.

## Perf
- Full rebuild wall time: **~2.5 s** (2.46–2.94 s across runs).
- Query latency (13 probes, cold-ish): **mean 0.58 ms, max 1.30 ms** — well sub-second.

## Probe queries (10 doc + 3 code) — #1 hit
FTS5 has no stemmer and punctuation is a query-parser operator; probe strings are
tokenized to bare AND-terms (as `query_memory.py` users phrase them). `--tag`
(new, trivial) narrows to a source class.

| probe | #1 hit file | class OK? |
|---|---|---|
| lookahead searchsorted | docs/daily/INDEX.md (2026-04-17 LOOKAHEAD line) | yes (journal/index) |
| dumb stop never-bail | research/nt8_catalog/comms/100_..._DUMB_STOP_WINS_NET.md | yes (comms) |
| FADEAGN inverted | docs/reference/RESEARCH_JOURNAL.txt (07-18 ExNMP) | yes (journal) |
| telegram bridge token | tools/telegram_bridge/bridge.py | yes (code) |
| gate spec lockbox | docs/northstar/ride_edge_gate_spec.md (spec copy #2) | yes (doc/spec) |
| warm-start selectivity gap | comms/096_..._MAMBA_ANTIFREEZE...md | yes (comms) |
| wick 0.83 | comms/102_..._NMP9_QUANTILE_RETUNE.md | partial — see note |
| dilution artifact | RESEARCH_JOURNAL.txt (footprint spinout); #2 report:nt8_catalog | yes |
| $900 giveback | docs/daily/2026-04-16.md ($900 peak / giveback) | yes (journal) |
| teacher student distillation | docs/northstar/ride_edge_gate_spec.md (Q2) | yes (doc/spec) |
| z_se compute (code) | #1 journal 05-03; code layer: SFE/exits/conversion | see note |
| last_closed_idx (code) | #1 journal 07-17; `--tag code:` → core_v2/build_dataset.py #1 | yes (code) |
| nonce commit serve (code) | comms/111 #1; #2 research/exit_dojo/tools/dojo_gate.py | yes |

### Notes / honest limitations
- **`wick 0.83`**: `0.83` tokenizes to `0`+`83` (unicode61 splits decimals). The literal
  "wick"+"0.83" co-occurs in comms/102, comms/085, AGENT_FEEDBACK_RULES, PROJECT_HISTORY —
  #1 comms/102 legitimately contains both. There is no single "wick 0.83" report; the fact
  lives in feedback/history/comms.
- **Code probes rank below prose by design.** `build_dataset.py` DOES contain `last_closed_idx`
  (in its docstring) and `statistical_field_engine.py` DOES contain `z_se`; they rank #2 only
  because (a) journals discuss the same identifier more densely and (b) FTS5 does not stem, so
  `compute` ≠ `computes`. Constraining with **`--tag code:`** returns the right module #1
  (`last_closed_idx` → build_dataset.py; `nonce commit` → dojo_gate.py). The code layer indexes
  structural surface (docstrings + top-level signatures + IO paths), NOT every in-body identifier —
  a token buried only inside a function body may be absent. This is the intended mechanical scope.

## Deviations from the spec (with reasons)
1. **`rl_whitepaper.md`** — no repo-root copy exists; it was moved to
   `archive/root_2026_06/rl_whitepaper.md` (2026-06 root cleanup). Builder falls back to the
   archive copy so the RL architecture doc is still indexed. Reason: preserve intent (index the whitepaper).
2. **`ROADMAP_LAMBDA_COMPLETION.md`** — no repo-root copy; it lives at
   `docs/Active/ROADMAP_LAMBDA_COMPLETION.md`, already covered by the `docs/Active/*.md` glob. No loss.
3. **`PRODUCTION_RUN_SPEC.md`** — does not exist in `research/dojo_forge/` (only
   `RIDE_EDGE_GATE_SPEC.md`). Glob simply matches nothing for it; specs layer = 9 rows from the one spec.
4. **README tag** — `doc:<name>` would collide across ~10 research READMEs (all basename `README`);
   READMEs are tagged `doc:<parent-dir>` to keep them distinct/meaningful.
5. **`--tag` filter added to `query_memory.py`** — trivial prefix `LIKE` filter (spec permitted this).

## Reproduce
```
python3.11 tools/memory_loop/build_memory_db.py            # full rebuild (~2.5s)
python3.11 tools/memory_loop/build_memory_db.py --sources comms,code   # partial
python3.11 tools/memory_loop/query_memory.py "last_closed_idx" --tag code:
```
