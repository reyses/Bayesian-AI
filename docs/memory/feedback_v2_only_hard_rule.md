---
name: feedback-v2-only-hard-rule
description: V2 features (185D layered) are the ONLY supported feature schema. No V1, no hybrid, no custom. All new code targets V2; all V1 code is technical debt to be removed.
metadata:
  type: feedback
---

**Rule**: V2 layered features (185D, per-family parquets under `DATA/ATLAS*/FEATURES_5s_v2/{L0,L1_*,L2_*,L3_*}/`) are the only sanctioned feature schema. Do NOT write code that produces 91D V1 features, do NOT add V1↔V2 adapters, do NOT support both with a flag.

**Why:** User declared this 2026-05-24 during the ForwardPass unification design. Rationale: maintaining two feature schemas means two lookahead-audit surfaces, two parity-validation paths, two CNN training pipelines. The `0c001c1f` baseline-invalidation commit (lookahead in V1 `build_dataset.py`) is exactly the kind of bug duplicate paths invite. Cutting V1 is the lever that prevents that class of bug.

**How to apply:**
- New code: only V2. `BarState` (from `core_v2.FPS.state`) or equivalent V2 dataclass — never `state['features_79d']`.
- Refactoring: when touching V1 code, propose migrating it to V2, not patching it. If V2 migration is out of scope for the task, flag it and stop — don't extend V1.
- Deletions: V1-only files (e.g. `core_v2/v2_to_v1_inmemory.py`, the deleted `build_v2_to_v1_compat_cache.py`) are removable as soon as their last V1 caller is migrated.
- Live engine: live and offline must share the same feature path. The proposed unification is for live to drive bars through `ForwardPass(source='live')` so SFE math is invoked from one place — parity by construction. See [[project_forward_pass_unification_2026_05_24]] for the migration plan.

**Known V1 holdouts as of 2026-05-24 (to be migrated):**
- `training/nightmare_blended.py` — engine indexes `state['features_79d']`
- `core_v2/build_dataset.py` — writes 91D
- `training/feature_processor.py`, `training/live_feature_engine{,_v2}.py`, `training/compute_features.py` — produce 91D
- `live/engine_v2.py` — consumes 91D
- `training/cnn_entry_direction.py`, `training/cnn_trade_manager.py` — trained on 91D (CNN retraining required)
- `core_v2/features.py::extract_features` — V1 91D constructor

**Already V2-native (good):**
- `training/pipelines/v2_native.py`
- `training/sfe_ticker.py` (yields V2 BarState)
- `core_v2/FPS/forward_pass_system.py` (V2 reader)
- `tools/research/features_v2.py`
- `DATA/ATLAS_NT8/FEATURES_5s_v2/` on disk
