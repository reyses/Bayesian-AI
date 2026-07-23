# ONBOARDING — read this first (any AI/human getting up to speed)

> Single orientation doc. Points at all memory + the key files + the current
> environment, and flags where the older docs LIE (verified stale, 2026-07-21
> full-repo audit). When a scattered doc disagrees with this file or with the
> code, trust the code, then this file. Indexed into `memory.db` (tag `doc:ONBOARDING`).

## 0. Fast path — read order
1. **`AGENTS.ini`** — repo-root file/command index (but see §6: several entries are stale).
2. **`CLAUDE.md`** — persona (critical-collaborator, real money), hard rules, metric defs. Overrides everything on conflict.
3. **`docs/memory/MEMORY.md`** — always-loaded condensed knowledge base (program/rules/metrics/traps/graveyard/wins/architecture).
4. **`docs/daily/INDEX.md`** — one line per day, newest first. Fastest "what happened lately".
5. **`docs/Active/`** — current roadmaps (MANDATORY per CLAUDE.md).
6. **`docs/northstar/README.md`** — the teacher→student north star (the active program).
7. **Query memory instead of reading whole files**: `python3 tools/memory_loop/query_memory.py "<terms>" [--tier stable|context|volatile] [--tag doc:|comms:|code:]`

## 1. Environment (NEW — 2026-07-21, migrated Windows→native Linux)
- **Interpreter/env**: conda env **`bayesian`, Python 3.12** at `~/miniforge3/envs/bayesian`. Activate: `conda activate bayesian` (or use `~/miniforge3/envs/bayesian/bin/python` directly).
- **Built on ext4**, NOT the repo's NTFS mount (NTFS can't hold venv symlinks). The old `.venv` was deleted.
- **CUDA works**: `torch 2.5.1+cu121`, `torch.cuda.is_available()==True` on an **RTX 3060 (12GB, sm_86, driver 595.71)**. No system CUDA toolkit — torch's bundled cu121 runtime.
- **mamba stack** (from `docs/memory/reference-mamba-ssm-wsl-perf.md` wheel matrix): `mamba_ssm 2.2.4` + `causal_conv1d 1.5.0.post8` (cp312/abiFALSE prebuilt). A `sitecustomize.py` stub aliases `GreedySearch/SampleDecoderOnlyOutput` so `import mamba_ssm` works against transformers 5.x.
- **llama-cpp-python**: the prebuilt wheel SIGILLs (CPU has avx2/fma, no avx512). Correct build = from-source per `research/dojo_forge/reports/gpu_wsl_build.md` (`0.3.34` sdist, `GGML_CUDA=on`, `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_NATIVE=off`, system CUDA). Status: being resolved.
- **RETIRED path**: `/home/reyses/venvs/bayesian-ai` and all `/mnt/c`, `C:/Users/reyse/OneDrive`, `D:\` paths in scripts are dead on this box. See §6.

## 2. Canonical entrypoints (CORRECTED vs the docs)
| Purpose | Command |
|---|---|
| Strategy validation | `python -m training.pipelines.run_strategy --strategy {zigzag,nmp_fade_raw} --target {is,oos} [--analyze]` — **NOT** `training.run_strategy` (that module doesn't exist; CLAUDE.md/AGENTS.ini are wrong) |
| V2 feature build | `python core_v2/build_dataset.py --atlas {DATA/ATLAS,DATA/ATLAS_NT8} --fresh` (user-runs) |
| Zigzag pivot datasets | `python -m training.strategies.zigzag` |
| Live engine | `python -m live.engine_v2 --engine-mode l5 [--mock]` — ⚠️ **CURRENTLY BROKEN at import** (see §6) |
| RL training | `python training/rl_engine/train_historical.py --agent-type {EXIT_NMP,ENTRY_NMP,NMP,YOLO}` (parked behind north star) |
| Mamba seq trainer | `python research/mamba_zigzag_baseline/pipeline/train_mamba_rl_seq.py --days ... --num_episodes N` (⚠️ atlas-root path heuristic needs a Linux fix) |
| Dojo-forge teacher eval | `python research/dojo_forge/pipeline/eval_native_ckpt.py --engine {cpu,cuda}` |
| Viz | `python -m tools.viz.run --plugin <name>` |
| Memory query/rebuild | `python3 tools/memory_loop/query_memory.py "<terms>"` · `build_memory_db.py` |

## 3. Architecture map — the key file per area (read these to understand each)
- **core_v2/** — V2 engine. `features.py` = **the FEATURE_NAMES / N_BASE source of truth (417D, uniform N_BASE=30 for all TFs)**; `statistical_field_engine.py` (regression/z-score kernels, lookahead-risk); `build_dataset.py` (materializer; `_last_closed_idx` is load-bearing causality); `sim_executor.py` (bar-loop driver — **broken**, §6); `sessions.py` (CME session-day boundary).
- **training/** — `pipelines/run_strategy.py` = the real Strategy driver. `strategies/base.py` (Strategy ABC), `strategies/zigzag.py` (only strategy wired to production). Live helpers moved to `archive/training/utils/` (see `training/ARCHIVE_INVENTORY.md`). `rl_engine/train_historical.py` (PW-CRL; parked). NOTE `training/README.md` is EMPTY.
- **live/** — `engine_v2.py` (7-step startup; docstring is trustworthy), `l5_decider.py` (production decision: zigzag/R-trigger + B7/B9/B10 sizing), `config.py` (LiveConfig), `protocol.py` (NT8 TCP wire). Needs NinjaTrader 8 "BayesianBridge" on 127.0.0.1:5199.
- **research/dojo_forge/** — the ACTIVE program. `RIDE_EDGE_GATE_SPEC.md` (frozen governing gate), `pipeline/eval_native_ckpt.py` (acceptance runner), `reports/gpu_wsl_build.md` (the only validated llama build). No README (convention violation).
- **research/mamba_zigzag_baseline/** — `README.md`, `PRODUCTION_RUN_SPEC.md`, `THESIS_reward_design.md`, `pipeline/mamba_env.py`, `pipeline/mamba_rl_network.py`.
- **research/nt8_catalog/** — signal-league spine. `DISTILLED.md` (spine ledger — read FIRST), `MASTER_VALIDATION_PROTOCOL.md`, `tools/dossier_signal_pipeline.py`, `comms/` (152 numbered decision docs, highest-numbered first).
- **DATA/** — `pipeline/build_timeframes.py`, `pipeline/databento_to_atlas.py` (roll calendar), `daily_context/`. See AGENTS.ini `[v2_data_layout]`.

## 4. Memory system (how knowledge persists)
- **Source of truth = markdown** in `docs/memory/*.md` (+ `docs/daily/`, comms, reports). `MEMORY.md` = the always-loaded index.
- **Query layer = derived SQLite** `docs/memory/memory.db` (FTS5, gitignored, rebuildable). Built by `tools/memory_loop/build_memory_db.py`; auto-rebuilt on any `docs/memory/`|`docs/daily/` edit via a PostToolUse hook (`tools/memory_loop/rebuild_on_write.sh`, wired in `.claude/settings.local.json`).
- **Write discipline**: reusable-only (`feedback-session-promote-ritual.md`), single source of truth = `docs/memory/` (the Windows dual-copy sync is retired). Ways-of-working: `feedback-worker-delegation-ladder.md`, `feedback-swarm-review-pattern.md`.

## 5. Data & models (locations)
- **IS** = `DATA/ATLAS/` (Databento). **OOS** = `DATA/ATLAS_NT8/` (NT8 dump = what live trades). Both present/populated. `DATA/ai_cusp_picks/` = label dir.
- **Ollama model store** (dojo teacher) = `/media/moi/WindowsCode/ollama/models/` — 4 GGUFs incl. `qwen3:14b` (the PRIMARY teacher). Set `OLLAMA_MODELS` here if using Ollama; forge_harness can also load the blob directly via llama_cpp.
- Checkpoints in `checkpoints/` (gitignored). `checkpoints/mamba_warmstart.pth` present.

## 6. CRITICAL gotchas — what the stale docs hide (verified 2026-07-21)
1. **🔴 Live engine broken ~24 days**: `live/engine_v2.py` + `core_v2/sim_executor.py` crash at **import** with `ValueError: 'L3_1m_z_se_15' is not in list` — N_BASE went uniform-30 on 2026-06-27 (commit 4330a59e) but hardcoded feature-name strings weren't updated. The "live/real-money path" cannot run. NOT in known_issues. Fix = look up N_BASE dynamically instead of hardcoding window suffixes.
2. **Dimension count**: schema is **417 features / 41 layer families**, NOT the "185D/25 families" in CLAUDE.md + AGENTS.ini, NOR the "139D" in `build_dataset.py`'s banner. `core_v2/features.py` is truth.
3. **Deleted-but-referenced**: `core_v2/strategy_engine.py` (deleted; role now in `run_strategy.py`), `rl_whitepaper.md` (moved to `archive/root_2026_06/`), `training/nightmare_blended.py` + its frozen snapshot (both absent → breaks `pytest`). `training/utils/` and `training/regret/` moved to `archive/training/`.
4. **Wrong entrypoint everywhere**: `training.run_strategy` → use `training.pipelines.run_strategy`.
5. **Interpreter drift**: many docstrings say `python3.11` ("bare python hangs"); on this box use the conda `python` (3.12).
6. **Hardcoded dead paths**: `C:/Users/reyse/OneDrive/...`, `/mnt/c/...`, `D:\Bayesian-AI-data\`, `/home/reyses/venvs/bayesian-ai` appear in many scripts (mamba pipeline atlas-root, DATA cross-day, tools/calibration, research loose scripts). All broken on native Linux.
7. **Missing artifacts**: default sizing pickles `reports/findings/regret_oracle/{b7_leg_sizer,b9_remaining_amp,b10_vol_regime}.pkl` absent → `L5Context.load()` fails on default live run.
8. **numba.cuda** kernels (`core_v2/cuda_statistics.py`, `templates_v0/cuda_pattern_detector.py`) fail (`libnvvm.so` missing) — but those modules are orphaned/unimported, low risk.
9. **Stale READMEs**: `training/README.md` empty; `tools/README.md`, `tools/viz/README.md`, `DATA/pipeline/README.md`, `research/nt8_catalog/README.md` describe pre-consolidation layouts with dead file lists; no README under `research/dojo_forge/`.

## 7. How to trust docs
`AGENTS.ini` and `CLAUDE.md` are authoritative for RULES/persona but **stale on concrete file/dim/entrypoint facts** (last broad update 2026-05-28, predating the N_BASE change and Linux migration). Verify every path against the filesystem and every dimension against `core_v2/features.py`. `docs/daily/INDEX.md` + `docs/memory/` are the freshest. This file is the reconciliation.
