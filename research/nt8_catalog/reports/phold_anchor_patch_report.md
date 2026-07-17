# P_hold anchor patch — causal-correction rerun vs prepatch (verdict parity)
2026-07-17. Patch: `phold_exit_model.py` τ-anchors now use `t - BAR_S`
(BAR_S=5) at the 5s base layer (lines 207, 403), so per-minute features are
read at the last bar that CLOSED at/before wall time t — matching the
`_last_closed_idx` convention (build_dataset.py:96). Prior code omitted the
−period shift → features/drift up to 5s fresher than the label check.

Reviewer note: Drone A applied the patch + ran the rerun (both verified), but
stopped before writing this comparison; the reviewer (Fable) produced the
table below directly from the prepatch archive vs the fresh outputs.

## Old (prepatch, ≤5s skew present) vs New (causal)
| metric | prepatch | new (causal) | verdict |
|---|---|---|---|
| FULL AUC | 0.6472 | 0.6381 | — |
| BASELINE AUC | 0.6894 | 0.6846 | — |
| **FULL−BASE delta** | **−0.0422** [−0.0538,−0.0309] | **−0.0465** [−0.0582,−0.0350] | **UNCHANGED** |
| KILL-POINT A (F-space adds signal?) | BELOW 0.05 bar → no | BELOW 0.05 bar → no | **UNCHANGED** |
| flip lead-time (mode / median) | +1.5 / +3.0 min | +1.5 / +3.0 min | **UNCHANGED** |
| fixed-5m capture median | +1.75 | +1.75 | **UNCHANGED** |
| P<0.6 / P<0.5 capture median | −3.00 / −2.75 | −2.75 / −2.75 | **UNCHANGED** |
| KILL-POINT B (open-ended exit earned?) | no | no | **UNCHANGED** |

## Interpretation
- **Every doc-089 verdict survives on clean causal anchors.** Both kill-points
  hold; the 16× oracle exit gap and "exit is sequential → Mamba's job"
  conclusion are unaffected.
- **The correction moved everything in the PREDICTED direction.** Removing the
  ≤5s feature-freshness peek made FULL slightly worse (0.6472→0.6381) and the
  delta slightly MORE negative (−0.0422→−0.0465): the F-space, which reads the
  fresh 5s features, benefited more from the skew than the context-only
  baseline did. Stripping it makes the null *stronger*, not weaker — exactly
  as the audit expected for a shared-anchor advantage that still lost.
- Absolute AUCs drifted ≤0.01; flip lead-time p25 −2.0→−3.0 and mean +0.72→
  +0.44 (marginally earlier warnings, still net-lagging). No conclusion turns.

## Files
- Patched: `tools/phold_exit_model.py` (BAR_S constant + 2 anchor sites).
- Prepatch archive: `reports/phold_exit_model_prepatch.md`, `phold_run_prepatch.log`.
- Fresh: `reports/phold_exit_model.md`, `phold_run.log`, `phold_rows.parquet`.
