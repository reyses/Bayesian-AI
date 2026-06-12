# Research Findings

Pure research findings and data analysis results. No specs, no instructions.
Each file is a snapshot of what we learned, when, and from what data.

## Naming Convention
`YYYY-MM-DD_topic.md`

## Index

| Date | File | Summary |
|------|------|---------|
| 2026-03-07 | `2026-03-07_scalp_timescale.md` | Counter-trend scalps + too-early exits share the same regime (r=0.716). Two root causes: (1) template bias dominance — TF-agnostic 97% SHORT overrides physics even at 4h, (2) timescale mismatch — fast TFs spook exits while slow TFs correctly identify trend. |
| 2026-03-07 | `2026-03-07_brain_aggregation.md` | Full signal flow trace: quantum state -> worker tick -> template+physics blend -> geometric mean -> band confluence -> exit signal. Three problems identified: TF-agnostic template weight, stale bar-close-only updates, exit ignores trend agreement. Three fixes proposed (2-line to Jules-sized). |
