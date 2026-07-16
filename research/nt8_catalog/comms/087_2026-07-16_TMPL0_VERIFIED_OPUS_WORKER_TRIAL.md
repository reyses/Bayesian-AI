# TMPL0 VERIFIED — template stream in the league; first Opus-worker trial: PASS
**Doc:** 087 · **Date:** 2026-07-16 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executor:** Opus subagent (delegation-ladder trial #1), spec = doc 086 §3.

## 1. Reviewer verification (the rule: a result from unverified plumbing is not a result)
- All artifacts present (builder, 4 recovered legacy sources, parquet 159,498 rows,
  1,020-template JSON, findings, raw eval + build logs). Nothing committed by the
  worker; HEAD untouched. ✓
- **Independent reproduction: byte-identical.** My own evaluate() on the worker's
  parquet: `N=157113 AUC 0.631 base 0.68 || 0.56 [0.55,0.57] / 0.68 / 0.79
  [0.78,0.80]` — matches the worker's pasted output exactly. ✓
- Causality audit: events emitted at the first row of the NEXT bucket (same
  convention as the harness); 21-bar OLS z / Wilder DMI / R-S hurst all trailing;
  fit code and docstring confirm 2024-ONLY fitting, frozen centroids + 2024
  standardization. ✓
- Deviations all declared (10 of 16 dims dropped as non-causal/non-recoverable —
  tree-position and field-engine terms; ADX/Hurst canonical stand-ins; flat
  nearest-centroid membership). Kill-points evaluated, not tuned past. ✓

## 2. TMPL0 result (league stream #40)
```
TMPL0  N=157113  OOS-AUC 0.631  base 0.68 || 0.56 [0.55,0.57] / 0.68 / 0.79 [0.78,0.80]
```
- 203,635 raw events (7 pattern types × 1m/5m/15m) → 1,020 frozen 2024 templates
  → 768 with conviction → ~276 fires/day.
- **Genuinely directional and stable OOS**: terciles monotonic, CIs disjoint;
  assignment margins healthy (median 0.13, zero exact ties).
- **Honest ceiling**: only +0.015 AUC over raw PTRN-ENGULF (0.616) — below the
  0.05 signal bar. The clustering's real content is the per-template frozen 2024
  label-majority direction + conviction (logistic rides `value`, coef 0.52); the
  16-D physics blob collapsed to a 6-D causal core that barely out-discriminates
  a single raw pattern type. Moises' original black-box instinct was right: the
  named streams carry the signal; the clusters mostly repackage it.
- BUT the 0.68 base on 83k OOS fires is itself notable: a 6-D microstructure
  state (z, velocity, TF, adx, hurst, dmi) predicts the active label's direction
  at 0.68 via a frozen nearest-neighbor prior. That is the strongest dense-stream
  base in the league and feeds the combiner.

## 3. Worker-trial verdict (the delegation ladder works)
- Spec → execute → verify loop completed in ~18 min of worker time; zero
  false claims; kill-points respected; every deviation declared with rationale.
  Contrast: the AG loop needed 7+ false-completion catches for similar scope.
- Pattern adopted: Opus workers get judgment-adjacent builds WITH a written spec,
  artifact checklist, kill-points, and mandatory reviewer reproduction.

## 4. State
League = 40 streams (TMPL0 in); combiner rerun in flight. The template-engine
resurrection question (Moises, 2026-07-16) is CLOSED: identified (doc 086 §1),
input layer ported (PTRN-*), full engine resurrected 2024-frozen (this doc).
Queue: economic conversion (next Opus worker), overfit-decay (Sonnet worker).
