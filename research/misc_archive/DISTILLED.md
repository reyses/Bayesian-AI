---
name: distilled-misc_archive
description: Grab-bag of orphaned scripts/reports swept into one folder during the 2026-06-22 research-org reorg — not a single research effort.
metadata: {type: distilled, topic: misc_archive, status: concluded}
---
## Verdict
No single question — this folder is the overflow bucket from the 2026-06-22
research-folder reorg (every file has the same 2026-06-22 mtime, none newer).
Contains one substantive analysis (F-space selection pattern), one unrelated
tooling design note (Gemini/Antigravity wakeup mechanism, not trading), one
trivial NT8 data-conversion log, and ~15 standalone scripts/tests with no
shared project.md tying them together.

## Key numbers (with CIs where they exist)
- `reports/2026-06-12_fspace_pattern.md` (80,717 segments): real Gini 0.602 vs
  null Gini 0.545 ± 0.000 — concentration barely beats chance. Top-10 features
  = 30.0% of selections, top-20 = 47.9%. Jaccard(top-10 across vol tiers) =
  0.82–1.00 (vol-invariant vocabulary). 94.8% of surviving terms quadratic
  (interaction-heavy = overfit signature). Beta sign-stability 0.51 (coin
  flip) for top-2 features, 0.67–0.76 for z_high/z_low/velocity.
- `reports/nt8_convert_report.txt`: 1s/1m NT8 conversion, 129 days (2025_12_12
  → 2026_06_12). No further content.

## Graveyard / never-retry (if any)
- The fspace-pattern "measurable pattern" claim: TRUE only in the weak sense
  (stable vocabulary reached for); FALSE in the strong sense (no stable,
  simple functional form — interaction-heavy, sign-unstable). Do not build a
  linear "watch indicator X" rule on this.

## Reusable assets
- `tools/audit_fspace_pattern.py` — produced the fspace-pattern report above.
- `tools/test_live_lookahead_parity.py` — live-vs-offline forming-bar parity check (1m/1h/4h).
- `tools/validate_trade_execution.py` — mechanical trade-execution ground-truth validator (tick-level).
- `tools/test_vr_separation.py` — tests whether vr (V1's dropped de-facto gate) separates NMP winners/losers.
- `tools/gemma_calibrate.py`, `tools/gemma_triage.py` — local vision-LLM (Ollama/Gemma3) trade-plot triage + calibration harness.
- `tools/probe_convergence.py`, `tools/generate_candidate_seeds.py` — PCA/KMeans clustering utilities.

## Data locations
- FEATURES_5s_v2 parquet family (referenced by the fspace-pattern audit); no
  dedicated store owned by this folder.

## Open threads
None — the one real finding (fast reversion/extension vocabulary is
vol-invariant) already migrated into the NMP-LAMBDA / λ-completion line
(MEMORY §5); nothing here is unfinished on its own terms.

## Sources
- research/misc_archive/reports/2026-06-12_fspace_pattern.md
- research/misc_archive/reports/nt8_convert_report.txt
- research/misc_archive/reports/Wakeup_MCP_Design_2026-06-13.md
- research/misc_archive/tools/audit_fspace_pattern.py
- research/misc_archive/tools/test_live_lookahead_parity.py
- research/misc_archive/tools/validate_trade_execution.py
- research/misc_archive/tools/test_vr_separation.py

## Archive recommendation
KEEP where it is. It is already an archive by construction (created as the
2026-06-22 reorg's catch-all for files that didn't fit a named topic) — not a
live project with a README/project.md to fold. Moving it under
`research/archive/` would just rename the same bucket; only worth doing if/when
a project-wide `research/archive/` convention is adopted for ALL concluded
topics at once (see the two lineage-recovery folders' cards for that same
question). Flag `Wakeup_MCP_Design_2026-06-13.md` as mis-filed (Gemini/Antigravity
tooling note, not trading research) if anyone reorganizes further.
