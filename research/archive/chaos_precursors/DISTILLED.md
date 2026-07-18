---
name: distilled-chaos_precursors
description: Fittability deterioration before a chaos block is vol-in-disguise — no independent precursor signal, causally tested IS+OOS.
metadata: {type: distilled, topic: chaos_precursors, status: dead}
---
## Verdict
Asked whether causal fittability (1-R2 of a trailing linear price fit, plus decel/resid_norm
variants) predicts forward chaos (top-quartile forward realized vol) BEYOND plain trailing-vol
persistence. Tested causally (W=60 trailing, H=60 forward, 5s bars) on IS 2024-03 and OOS
2025-03. Result: fittability adds ~zero increment over vol-only in both windows — it is vol in
disguise, not an independent precursor. `research/chaos_precursors/reports/chaos_precursor_causal.md`

## Key numbers (with CIs where they exist)
- IS 2024-03 (n=316611, days=20, base chaos=25.0%): AUC trailing-vol (null) = 0.959.
  unfit solo AUC 0.521, vol+unfit AUC 0.959 (increment +0.000); decel solo AUC 0.501,
  vol+decel AUC 0.959 (increment -0.000); resid_norm solo AUC 0.564, vol+resid_norm AUC
  0.959 (increment -0.000); ALL fittability+vol AUC 0.959 (increment over vol -0.000).
- IS chaos rate by fittability quartile: Q1 fittable 27.5%, Q2 24.8%, Q3 23.8%, Q4 choppy 23.9%.
- OOS 2025-03 (n=205190, days=18, base chaos=82.3%): AUC trailing-vol (null) = 0.922.
  unfit solo AUC 0.528, vol+unfit AUC 0.922 (increment +0.000); decel solo AUC 0.504,
  vol+decel AUC 0.922 (increment +0.000); resid_norm solo AUC 0.557, vol+resid_norm AUC
  0.922 (increment +0.000); ALL fittability+vol AUC 0.922 (increment over vol +0.000).
- OOS chaos rate by fittability quartile: Q1 fittable 84.9%, Q2 81.8%, Q3 81.3%, Q4 choppy 81.2%.
- No CIs reported in the source file (point-estimate AUCs only).

## Graveyard / never-retry
- Fittability-as-chaos-precursor (unfit / decel / resid_norm, solo or stacked) — killed both
  IS and OOS: increment over trailing-vol-only AUC is ~0.000 in every variant. Source:
  `research/chaos_precursors/reports/chaos_precursor_causal.md`.

## Reusable assets
- `research/chaos_precursors/tools/chaos_precursor_causal.py` — causal fittability-vs-vol-null
  test harness (trailing R2 fit, forward-vol chaos target, day-block design); reusable pattern
  for "does X beat the vol-persistence null" questions.
- `research/chaos_precursors/tools/analyze_chaos_precursors.py` — earlier non-causal,
  segment-level precursor probe (day-block bootstrap CI over `PURE_CHAOS`/`PRISTINE`/
  `RECOVERED` segment transitions); precursor to the causal script above.

## Data locations
- `DATA/ATLAS/5s` (Databento, IS 2024-03 + OOS 2025-03) — used by `chaos_precursor_causal.py`.
- `artifacts/stage2_year_segments.json` — segment-level input to `analyze_chaos_precursors.py`
  (not present under this topic folder; not verified to still exist at that path).

## Open threads
none — verdict is a closed negative result.

## Sources
- research/chaos_precursors/reports/chaos_precursor_causal.md
- research/chaos_precursors/tools/chaos_precursor_causal.py
- research/chaos_precursors/tools/analyze_chaos_precursors.py
- research/chaos_precursors/README.md (stub, no content)
- research/chaos_precursors/project.md (DMAIC stub, all sections empty)

## Archive recommendation
ARCHIVE (dead — causal test found no independent signal beyond the vol-persistence null,
in both IS and OOS; README/project.md are empty stubs with no further planned work).
