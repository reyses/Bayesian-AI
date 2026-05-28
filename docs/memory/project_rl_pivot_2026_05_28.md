---
name: project-rl-pivot-2026-05-28
description: Project pivoted from supervised CNN/blended stack to an RL engine (PW-CRL); the deletion left dangling imports in live + run.py
metadata: 
  node_type: memory
  type: project
  originSessionId: c2947774-6098-4d63-8985-876610818bcd
---

As of commit `4b658e2a` (pushed 2026-05-28), the project pivoted from the
supervised CNN / blended / "nightmare" trade-management stack to a
**reinforcement-learning engine** branded "Parallel Worlds Curriculum RL"
(PW-CRL): a CNN+LSTM DQN with V-trace off-policy correction, a hindsight-regret
"shadow queue", curriculum learning (`EXIT_NMP → ENTRY_NMP → YOLO`), and an
8-agent DOE sweep. Lives in `training/rl_engine/`. Architecture doc:
`rl_whitepaper.md` (repo root). `training/` was reorganized: shared helpers →
`training/utils/`, CNN/GBM trainers → `training/trainers/`, `regret` is now a
package (`training/regret/`), new `training/strategies/zigzag.py`.
`training/build_dataset_v2.py` → `core_v2/build_dataset.py`.

DELETED in the same commit: `cnn_entry/exit/flip/hold/risk/trade_manager.py`,
`nightmare*.py`, `nightmare_blended.py`, `compute_features.py`, `ai.py`,
`physics_labels.py`, `model.py`, `memory.py`, `feature_processor.py`,
`run_baseline.py`, `forward_blended.py`, `live_feature_engine*.py`.

**Why:** per `rl_whitepaper.md`, supervised exit models destroy net PnL by
cutting winners — the first 5 min of a $400 winner is statistically identical to
a loser, so a supervised model amputates the right tail. RL learns exits
directly from the PnL reward. The RL engine is **mid-training as of 2026-05-27**
(curriculum segment 10, manual overfit/LR interventions) — NOT yet a deployed
production path.

**How to apply:**
- CLAUDE.md "Key Files" / "Active Work", `docs/memory/MEMORY.md`, and
  `AGENTS.ini` (`status = production blended pipeline`) all predate this pivot —
  treat them as stale on the CNN/blended pipeline until updated.
- **Known regression (introduced 4b658e2a):** the deletions left ~15 active
  files importing now-deleted modules — `live/live_engine.py`,
  `live/maintenance.py`, `live/diagnostic_run.py`, `training/run.py`, plus
  `tools/util/{hypothesis_test,blended_test}.py`,
  `tools/archive/lookahead_impact.py`, `tools/exits/giveback_analysis.py`,
  `tools/eda/sunday_hourly_eda.py`, `tools/data/validate_sfe_parity.py`, and
  `tests/test_{engine_evaluate,sim_executor}.py` import deleted
  `nightmare`/`nightmare_blended`/`compute_features`/`ai`/`physics_labels`.
  These ImportError on invocation. The `regret` imports are FINE (package
  re-exports resolve). OPEN QUESTION (not yet resolved with user): is the
  blended layer intentionally retired (so this is acceptable transitional debt
  and `engine_v2.py` supersedes `live_engine.py`) or does the live path need
  these modules restored/rewired?
- The Telegram bridge tooling (`telegram_*.py`, `push_alert.py`) now reads the
  bot token from `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars. The token
  that was hardcoded should be treated as compromised (user advised to rotate).

See [[feedback-cli-script-false-orphans]] — the same "no Python imports ≠ no
callers" caution applies in reverse here: deletions without grepping importers
broke live code.
