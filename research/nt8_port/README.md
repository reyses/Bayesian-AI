# research/nt8_port — Interim NT8 native port (Architecture B, doc 129)

Project folder for the **native NinjaScript port** of the current entry system
(entry P + R-trigger exit + sizing). The **parity harness is the spine**: every C#
component validates bar-by-bar against Python golden vectors. Spec:
`research/nt8_catalog/comms/129_2026-07-18_SPEC_NT8_NATIVE_PORT_B.md`.
Architecture B = mechanical manager, NO cut logic (entry combiner P + R-trigger + sizing).

## Layout
```
research/nt8_port/
  tools/golden_vector_gen.py   # P0 generator (this deliverable)
  golden/<day>.parquet         # 20 reference days, per-1m-bar golden records
  reports/golden_schema.md     # column-by-column spec (implement C# against THIS)
  reports/golden_manifest.json # combiner fit, top-K + coef mass, threshold, day list, per-file sha256
  reports/golden_sanity.md     # per-day fire/entry/pivot counts
  README.md                    # this file
```

## P0 — golden vectors (DONE)
`tools/golden_vector_gen.py` runs the **Python reference decider** over 20 reference
days (10 × 2024, 10 × 2025-26; real RTH label-days, regime-diverse) and emits
per-1m-bar golden records: top-K stream fire states, the pooled combiner P, the entry
decision at the frozen top-decile threshold, and the live R-trigger zigzag pivot state.

Run (repo root; **python3.11** — bare `python` hangs on this box):
```
python3.11 research/nt8_port/tools/golden_vector_gen.py                    # full 20-day run
python3.11 research/nt8_port/tools/golden_vector_gen.py --verify-determinism
```

### What it reuses (no drift)
- `research/nt8_catalog/tools/dossier_signal_pipeline.py` — `DayCtx`, all `GENS`
  causal generators, paths, the streaming context (tail + prior-day) convention.
- `research/nt8_catalog/tools/combiner_preview.py` — `load_pool`, `BASE`, `CONSENSUS_S`;
  the combiner fit is reproduced exactly (2024-train logistic, standardized).
- `research/nt8_catalog/tools/template_stream_builder.py` + `reports/tmpl0_templates_2024.json`
  — TMPL0 (frozen K-means template stream, top-4 by weight).
- `training/strategies/zigzag.py::ZigzagStrategy` — R-trigger state machine (ported verbatim:
  `extreme ± R`, `min_bars_5s = 36`).

### Key facts (see `reports/golden_schema.md` + `reports/golden_manifest.json`)
- **K = 22** top-K streams (cumulative `|coef|` reaches 80.1% of the 3.302 stream-mass at
  stream 22). Grand-total denominator is degenerate (impossible ≥80%), so stream-mass is the
  only well-defined reading. The combiner weight is **diffuse** (top stream RSI06 = 11%).
- **Frozen entry threshold**: `P >= 0.7339` (90th pct of 2024-train pooled P).
- **Determinism**: byte-identical on re-run (verified, sha256 recorded per file).
- **Excluded external streams** (rank far below top-K): `ADX08`, `FOOTPRINTIMB` (meta/circular).
- **Consensus caveat**: fit uses label-window-filtered consensus; generation uses all-fires
  (live-valid) consensus — a documented P-calibration caveat. P is the *full-combiner reference*;
  P1's compact re-fit is the C# P-parity target.

### Data dependencies (repo-root relative, run from root)
- `DATA/ATLAS/5s/*.parquet` (604 days) — the 5s substrate.
- `DATA/ATLAS/FEATURES_1s_v2/L3_1m/<day>.parquet` — `L3_1m_z_se_15` for NMP / NMP9-head streams.
- `DATA/ai_cusp_picks/ai_picks_*_multi.json` — used ONLY to select real RTH trading days
  (labels are NOT part of the golden vectors).
- `research/nt8_catalog/reports/signal_rows_*.parquet` — the combiner training pool (56 streams).
- `research/nt8_catalog/reports/{tmpl0_templates_2024.json, propturn_frozen.json, propturn_p_frozen.json,
  nmp9_retuned_constants.json}` — frozen stream artifacts consumed by the generators.

## Next phases (not built here)
- **P1** entry port: re-fit a compact 2024-sealed model over the K generators + logistic;
  quantile-match thresholds on 2024; C# parity ≥99% decision agreement + P within tolerance.
- **P2** R-trigger native (adapt released ZigzagRunner v1.0 .cs); parity vs golden pivots.
- **P3** SIM parity (multi-gap 0/30/60/100%). **P4** live SIM → micro live (deploy gate per revision).
