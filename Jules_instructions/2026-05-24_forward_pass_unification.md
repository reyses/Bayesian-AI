# 2026-05-24 — Forward Pass Unification

**Status**: Draft 2026-05-24
**Owner**: reyses
**Hard rule input**: V2 features only (see `docs/memory/feedback_v2_only_hard_rule.md`)
**Brief path**: `Jules_instructions/2026-05-24_forward_pass_unification.md` (this file)
**Exit report**: `Jules_instructions/2026-05-24_forward_pass_unification_exit_report.md` (required after run, per `Jules_instructions/README.md`)

---

## Goal

One bar-walking engine, used by all consumers (training, OOS forward pass, live trading), that produces a V2 `BarState` stream — either by reading pre-built `FEATURES_5s_v2/*` parquets (`source='cache'`) or by computing features on the fly from raw OHLCV via SFE (`source='live'`).

Outcome: the lookahead-free discipline lives in one module. Live and offline parity is by construction (same code path produces the bar stream).

---

## Non-goals (this spec)

- Replacing the SFE math itself (it stays — FP just calls it).
- Changing the V2 feature schema or layer families.
- Retraining CNNs (separate task, gates Phase 3).
- Replacing the NT8 bridge — the bridge still pushes 1s bars, FP just owns the loop above it.

---

## The unified interface

```python
# core_v2/forward_pass.py

from core_v2.forward_pass import ForwardPass
from core_v2.FPS.state import BarState  # V2 BarState

# ── Mode 1: pre-built features (validation, training, OOS forward passes)
fp = ForwardPass(
    days=['2025_06_15', '2025_06_16'],
    atlas_root='DATA/ATLAS_NT8',
    source='cache',                    # reads FEATURES_5s_v2/{L0,L1_*,L2_*,L3_*}/
)
for state in fp:
    ...  # state: BarState

# ── Mode 2: compute via SFE on the fly (live trading, fresh days)
fp = ForwardPass(
    days=['2025_06_15'],
    atlas_root='DATA/ATLAS_NT8',
    source='live',                     # raw 5s OHLCV → Aggregator → SFE → BarState
)
for state in fp:
    ...

# ── Mode 3: build features AND persist (replaces build_dataset.py wholesale)
fp = ForwardPass(
    days=['2025_06_15'],
    atlas_root='DATA/ATLAS_NT8',
    source='live',
    write_features_to='DATA/ATLAS_NT8/FEATURES_5s_v2/',
)
for state in fp:
    pass                               # drains the iterator, persists as side effect
```

**Guarantee**: no consumer ever sees a feature value that depended on data with `timestamp > state.timestamp`. This is the only invariant; everything else is implementation detail.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ ForwardPass(days, atlas_root, source, write_features_to?)   │
│                                                              │
│   if source == 'cache':                                      │
│     ── load_features(days)  →  BarState stream  ────────┐    │
│                                                          │   │
│   if source == 'live':                                   │   │
│     ── raw OHLCV parquets                                │   │
│           ↓                                              ├──► yield BarState
│        Aggregator (TF roll-up, internal)                 │   │
│           ↓                                              │   │
│        StatisticalFieldEngine (V2 layered features)      │   │
│           ↓                                              │   │
│        assemble BarState  ──────────────────────────────┘    │
│           ↓ (optional, if write_features_to set)             │
│        persist to FEATURES_5s_v2/{layer_family}/             │
└─────────────────────────────────────────────────────────────┘
```

Internal pieces (no public API):
- `core_v2/aggregator.py` — OHLCV TF roll-up (moved from `training/`).
- `core_v2/statistical_field_engine.py` — SFE math (already in `core_v2/`).
- `core_v2/features.py` — V2 feature names, `load_features()`, layer family schema.
- `core_v2/FPS/state.py` — `BarState` dataclass.

`FeatureProcessor`, `LiveFeatureEngine{,V2}`, `compute_features.py`, the existing `FPS/forward_pass_system.py` body — all get folded into `ForwardPass` or deleted.

---

## Migration phases

Each phase ends with a parity check. No phase ships without a green parity run.

### Phase 0 — Aggregator move (mechanical, cheap)

- `git mv training/aggregator.py core_v2/aggregator.py`
- Update callers (grep for `from training.aggregator`): `feature_processor.py`, `live_feature_engine.py`, `compute_features.py`, `core_v2/build_dataset.py`, `live/live_engine.py`, `live/maintenance.py`.
- **Parity check**: byte-identical features produced by old vs new paths on 1 IS day.
- **Status**: ready to do today.

### Phase 1 — Build `ForwardPass` (V2-only)

- New file `core_v2/forward_pass.py`. Implements `source='cache'` and `source='live'` modes.
- `source='cache'` wraps `load_features()` + OHLCV join (the current `FPS/forward_pass_system.py` body).
- `source='live'` runs `Aggregator → SFE → V2 feature assembly`.
- **Parity check**: `ForwardPass(source='cache')` on day D must yield bar-for-bar identical `BarState` to `ForwardPass(source='live')` on day D, where day D has both raw OHLCV and pre-built V2 features. This is THE test — it proves the live and cache paths agree.
- **Tooling**: `tools/parity_validate.py` already exists; extend it to compare BarState streams.

### Phase 2 — Migrate `nightmare_blended.py` (engine) to V2

- Every `state['features_79d']` access becomes a named-field read on `BarState` (e.g. `state.z_se_1m`, `state.velocity_5m`).
- All hard-coded indices (`_1M_Z_IDX`, `_5M_BODY_IDX`, etc.) deleted.
- Engine signature changes from `evaluate(state: dict)` to `evaluate(state: BarState)`.
- **Parity check**: same OOS days, V1 engine on V1 features vs V2 engine on V2 features — trade-by-trade comparison. Expected: small differences (different feature shapes), document them.
- **Gate**: this phase ships only if the V2 engine's OOS $/day is within the bootstrap CI of the V1 engine on the same days. Otherwise we have a regression to investigate before continuing.

### Phase 3 — Retrain CNNs on V2

- `cnn_entry_direction.py`, `cnn_trade_manager.py` rewritten to consume V2 `BarState` directly (drop V1 91D input layer).
- `physics_labels.py` (label generator) verified to work on V2 stream — likely no change needed since labels come from price, not features.
- **Training run**: user runs `--fresh` per CLAUDE.md (Claude does not run training via Bash).
- **Gate**: new V2 CNN checkpoints must show IS/OOS AUC within bootstrap CI of V1 CNNs on overlapping days. Old V1 `.pt` files move to `training/archive/_v1_checkpoints/`.

### Phase 4 — Live engine migration

- `live/engine_v2.py` loop driven by `ForwardPass(source='live')` instead of its current NT8-bridge-callback loop.
- NT8 bridge pushes 5s OHLCV bars into `ForwardPass`; FP yields `BarState`; engine consumes.
- **Parity check**: SIM run for 1 trading day. Compare trade decisions to a forward pass of the same day's data through the migrated engine. Expected: bit-identical.
- **Highest-risk phase** — this touches production money path.

### Phase 5 — Delete V1

Now nothing imports V1. Delete:
- `core_v2/build_dataset.py` (replaced by `ForwardPass(source='live', write_features_to=...)`).
- `training/feature_processor.py`.
- `training/live_feature_engine.py`, `training/live_feature_engine_v2.py`.
- `training/compute_features.py` (or rewrite as thin V2 wrapper if it has unique logic worth saving).
- `core_v2/features.py::extract_features` (V1 91D constructor). Keep V2 names/loader.
- Old V1 CNN training scripts (`cnn_entry_direction.py`, `cnn_trade_manager.py` once V2 versions exist with the same filenames or new ones).

Update `MEMORY.md` to remove "V1 holdouts" section from `feedback_v2_only_hard_rule.md`.

---

## Parity check infrastructure

`tools/parity_validate.py` exists. We need it to compare two `BarState` streams from the SAME day:
- Same bar count.
- For each bar, same `timestamp`, same `price`, same value on every named V2 field within `1e-9` (or whatever tolerance the SFE math guarantees).

Add `tools/parity_forward_pass.py` if needed — purpose: given a day, produce both `ForwardPass(source='cache')` and `ForwardPass(source='live')` streams, diff them, report first divergence.

---

## Open questions (decide before Phase 2)

1. **`BarState` schema** — does it include all 185 V2 layered features as named fields, or grouped (`state.layer_3.z_se_1m`)? Current `core_v2/FPS/state.py` has a partial schema; needs to be the canonical V2 field surface.
2. **Cross-day history** — `ForwardPass(source='live')` for day D needs the prior 300 bars of higher-TF history to warm up SFE. Today `build_dataset.py` loads N-1 prior days per TF. FP needs the same mechanism. Spec'd in the SFE math; just verify FP preserves it.
3. **Write semantics for Mode 3** — write per-bar (slow, atomic) or buffer per-day and write on day-close? Today `build_dataset.py` buffers per-day. Keep that.
4. **Test data** — do we have at least one day where both raw OHLCV AND pre-built V2 features exist on disk for parity? If yes, identify it. If no, build V2 features for one such day first.

---

## Risks

- **Subtle lookahead introduced during migration**. Mitigation: every phase has a parity check that includes byte-equality on at least one day where the prior pipeline ran. If the byte check fails, do NOT proceed — root-cause first.
- **`nightmare_blended.py` Phase 2 has many touch sites**. Indexed access pattern (`feat[_1M_Z_IDX]`) is used pervasively; the rename pass needs to be mechanical and complete in one commit, not piecemeal.
- **Phase 3 training run is opaque**. Once CNNs are retrained, comparing them to the V1 CNNs is statistical, not byte-equal. Set OOS-CI-overlap as the gate.
- **Phase 4 live migration** — production money. Run in SIM for at least 5 trading days before any real deployment.

---

## What I (Claude) WILL NOT do

- Run training via Bash (per CLAUDE.md).
- Edit production NT8 strategy files (per CLAUDE.md NT8 deploy gate).
- Delete V1 code before its consumers are migrated.
- Skip a parity check to "save time."

---

## What's done as of this draft

- `core_v2/v2_to_v1_inmemory.py` and `tools/build_v2_to_v1_compat_cache.py` deleted (V2→V1 adapters, no longer needed under V2-only rule).
- `training/build_dataset.py` → `core_v2/build_dataset.py` (consolidates under feature system).

## What's next

- Phase 0: `Aggregator` move. ~5 callsite touches. Ready to execute on user approval.
- Phase 1: build `ForwardPass`. Multi-session task; this spec is the design reference.
