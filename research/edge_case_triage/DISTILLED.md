---
name: distilled-edge_case_triage
description: Built a 40-trade edge-case teaching-set curator (6 archetypes) for Gemma few-shot training, but the 3-way verification loop (Gemini/Claude/human) it was designed around was never filled in.
metadata: {type: distilled, topic: edge_case_triage, status: dead}
---
## Verdict
Goal: curate a small (40-trade) sample spanning distinct trade archetypes from the
full clean-trade population, route it through 3-way verification (Gemini/Claude/human)
to check entry/exit quality, and use the agreed set as a Gemma few-shot teaching set.
The curation tool (`tools/curate_edge_cases.py`) ran and produced the sample + plots +
manifest, but `reports/edge_case_manifest.csv`'s `verify_gemini`/`verify_claude`/
`verify_human` columns are all empty — only the script's own `proposed_*` labels are
filled in. The verification step never happened; `project.md` (DMAIC) is an unfilled
template. No findings/conclusions exist beyond the archetype taxonomy itself.

## Key numbers (with CIs where they exist)
- Full-population (5k trades) archetype counts, from `reports/edge_case_manifest.md`:
  STOPPED=2159, SMALL_LOSS=1270, SMALL_WIN=1064, CLEAN_RIDE=458, GAVE_BACK=454, CHOP=58.
- Curated teaching sample = 40 trades (6/archetype + 5 extremes by net_usd/dur/mfe), per
  `tools/curate_edge_cases.py` (`PER = 6`).
- No CIs anywhere in this topic — it is a labeling/taxonomy exercise, not a stats result.

## Graveyard / never-retry (if any)
none recorded — the topic stalled before reaching a graveyard-worthy conclusion.

## Reusable assets
- `research/edge_case_triage/tools/curate_edge_cases.py` — self-extracts each trade's
  signed price path from `DATA/ATLAS/1s/<day>.parquet` (no dependency on the stale
  contaminated `trade_paths.parquet`), classifies into 6 archetypes via hand-tuned
  net$/MFE/kept-fraction rules, and writes plots + manifest. Reusable if the
  verification loop is ever revived.

## Data locations
- Input: `reports/findings/kalman_clean_trades.csv` (referenced by the tool, not stored
  in this topic folder).
- Path source: `DATA/ATLAS/1s/<day>.parquet` (1s OHLC, columns `timestamp`, `close`).
- Output: `reports/edge_case_manifest.csv` / `.md`, `reports/trade_*.png` (individual
  trade plots), `reports/_contact_sheet.png` (grid overview).

## Open threads
- The 3-way verification (Gemini/Claude/human agreement on entry/exit quality per
  archetype) was never executed — `verify_gemini`/`verify_claude`/`verify_human`
  columns in the manifest are blank. Unknown whether this was superseded by other
  labeling work (e.g. golden-labels efforts referenced elsewhere in MEMORY.md) or
  simply dropped.
- A 7th archetype, `GAP_TRUNCATED` (gap-close trades, dur<=120s), is defined in the
  tool's `archetype()` function (2026-06-16 Gemini-caught rule) but does not appear in
  the curated 40-trade sample or the manifest counts.

## Sources
- research/edge_case_triage/README.md
- research/edge_case_triage/project.md
- research/edge_case_triage/tools/curate_edge_cases.py
- research/edge_case_triage/reports/edge_case_manifest.md
- research/edge_case_triage/reports/edge_case_manifest.csv

## Archive recommendation
ARCHIVE (thin/stalled — curation tool + unlabeled 40-trade sample only, no completed
analysis or findings; the one 3-way-verification question this topic exists to answer
was never answered). Keep `curate_edge_cases.py` noted as reusable if the teaching-set
effort is revived.
