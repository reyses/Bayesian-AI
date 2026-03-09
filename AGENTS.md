# BayesianBridge — Project Instructions
**Snapshot: 2026-03-09 | Commit: 678c9bc**

---

## What This Is

MNQ (Micro Nasdaq futures) algorithmic trading system. Statistical regression bands across 11 timeframes + Bayesian probability learning + recursive K-Means template clustering. Executes via NinjaTrader 8 bridge.

NOT quantum physics. The codebase used physics metaphors historically (three-body, Lagrange, Roche, event horizon). These are ~95% purged. Remaining remnants are in the cleanup backlog. All new code uses statistical/regression language only.

---

## Current Performance (2026-03-08 23:38 run)

| Metric | IS (Jan-Dec 2025) | OOS (Jan-Mar 2026) |
|--------|-------------------|---------------------|
| Trades | 11,881 | 2,977 |
| Win Rate | 56.0% | 58.3% |
| Total PnL | $82,195 | $22,232 |
| Avg PnL/trade | $6.92 | $7.47 |
| Max DD (intraday) | — | $107 |
| Consecutive losing days | 0 | 1 |

Prior baselines in `docs/memory/MEMORY.md` (dated entries, do not overwrite).

---

## Repo Structure

```
Bayesian-AI/                    35,551 Python lines total
├── CLAUDE.md                   Claude Code reads this automatically
├── AGENTS.md                   Agent guidelines (Jules/Claude Code/chat)
├── README.md                   Project overview
│
├── core/                       5,516 lines — engine layer
│   ├── statistical_field_engine.py   489L — CUDA regression, z-scores, MarketState
│   ├── execution_engine.py           976L — gate cascade, direction, sizing
│   ├── exit_engine.py                710L — SL/TP/envelope/giveback/breakeven/flip
│   ├── timeframe_belief_network.py  1018L — 11-TF worker consensus
│   ├── bayesian_brain.py             298L — probability table + direction learning
│   ├── fractal_clustering.py         651L — recursive K-Means templates
│   ├── feature_extraction.py          53L — canonical 16D feature vector
│   ├── market_state.py               255L — MarketState dataclass (per-bar output)
│   ├── cuda_statistics.py            320L — CUDA kernels (Hurst, R/S, ADX, DMI)
│   └── [6 smaller modules]
│
├── live/                       4,388 lines — NT8 bridge + execution
│   ├── live_engine.py               1785L — orchestrator (DECOMPOSITION IN PROGRESS)
│   ├── history_replay.py             524L — compressed forward pass for warmup
│   ├── session_tracker.py            289L — PnL, drawdowns, trade log
│   ├── bar_aggregator.py             295L — 1s → anchor-TF aggregation
│   ├── order_manager.py              302L — order lifecycle
│   ├── ping_pong.py                  117L — flip direction, deferred flips
│   ├── exit_watcher.py                86L — post-exit counterfactual tracking
│   ├── gui_bridge.py                  79L — non-blocking Tk queue
│   └── [6 smaller modules]
│
├── training/                  11,340 lines — pipeline + analysis
│   ├── trainer.py                   4994L — 6-phase pipeline, CLI, reports
│   ├── fractal_discovery_agent.py    830L — template discovery
│   ├── orchestrator_worker.py        617L — per-TF fractal worker
│   ├── trade_analytics.py            562L — Welch t-test, ANOVA, OLS suite
│   └── [10 smaller modules]
│
├── tools/                      — analysis & diagnostics
│   ├── standalone_research.py        — waveform research harness (Analyses A-R)
│   ├── research/                     — subpackage: data, imr, screening, seeds, plots
│   ├── analyze_gates.py              — oracle-driven threshold analysis
│   ├── gate_interaction_matrix.py    — C&E matrix empirical validation
│   ├── nt8_to_parquet.py             — NT8 export → ATLAS converter
│   ├── checkpoint_viewer.py          — inspect brain + library
│   └── archive/                      — retired one-off scripts
│
├── docs/
│   ├── Active/RESEARCH_SPEC_V_TO_FF.md  — 11 async research analyses
│   ├── specs/                        — future feature specs (3 files)
│   ├── archive/                      — completed specs (14 files)
│   ├── reference/                    — C&E matrix + research journal
│   ├── memory/                       — MEMORY.md, ce_methodology.md, waveform_research.md
│   ├── daily/                        — session journals (YYYY-MM-DD.md)
│   └── [ARCHITECTURE, SYSTEM_DESCRIPTION, CHANGELOG, ROADMAP].md
│
├── config/                     — symbols.py (MNQ/MES/NQ tick values)
├── DATA/                       — ATLAS parquets (gitignored except ATLAS_1MONTH)
├── checkpoints/                — model artifacts (gitignored)
└── reports/                    — run outputs (CSVs gitignored, .txt tracked)
```

---

## Entry Points

```bash
# Training pipeline (6 phases: cluster → library → IS → OOS → strategy → report)
python training/trainer.py --fresh --forward-pass

# IS only (skip OOS)
python training/trainer.py --forward-pass --skip-oos

# Fast validation (~3s)
python training/trainer.py --forward-pass --data DATA/ATLAS_1DAY

# Standalone OOS rerun
python training/trainer.py --oos

# Ping-pong mode (continuous flip)
python training/trainer.py --forward-pass --ping-pong

# Live (dry run / real)
python -m live.launcher --dry-run
python -m live.launcher --account Sim101

# Research
python tools/standalone_research.py --data DATA/ATLAS_1WEEK --start U
```

---

## Architecture: How a Trade Happens

### Training Pipeline (6 Phases)
```
Phase 1: Cluster Discovery    → recursive K-Means on 16D features → pattern_library.pkl
Phase 2: Library Build        → per-template oracle stats (MFE, MAE, direction, depth)
Phase 3: IS Forward Pass      → replay 12 months, apply gates, log trades
Phase 4: OOS Forward Pass     → replay 2-3 months unseen, validate IS findings
Phase 5: Strategy Selection   → rank templates by PnL, tier them (T1/T2/T3)
Phase 6: Reports              → IS/OOS summaries, analytics, gate reports
```

### Gate Cascade (ExecutionEngine)
```
Signal detected (TBN consensus ≥ min conviction)
  → Gate 1: Template match (distance < threshold)
  → Gate 2: Brain probability (Bayesian table lookup)
  → Gate 3: Conviction threshold (belief network confidence)
  → Gate 4: Momentum alignment (sign(F_momentum) agrees with direction)
  → Gate 5: Depth minimum (depth ≥ 3 required)
  → FIRE: direction + sizing → ExitEngine.open_position()
```

### Direction Hierarchy (priority order)
```
P0:  Ping-pong live bias (PP mode only)
P1:  Per-cluster logistic regression (signed_mfe_coeff)
P2:  Template aggregate bias (long_bias vs short_bias)
P3:  Live DMI / velocity
P4:  Band confluence (40% weight from TF worker bands)
Override: TBN belief disagrees → flip
```

### Exit Cascade (ExitEngine)
```
SL → TP → BandUrgent → EnvelopeDecay → PeakGiveback → BreakevenLock → BeliefFlip → Hold
  - Trail/MaxHold/Watchdog: DISABLED (envelope handles exits better)
  - Self-tuning: too_early → grow halflife, too_late → tighten giveback
  - Tiered giveback: peak ≥30t → 40%, 16-30t → self-tuned, <16t → disabled
  - 30m worker flip: sticky tighten when slow TF flips against trade
```

### Feature Vector (16D — single source of truth)
```
abs(z), log1p(v), log1p(m), coherence, tf_scale, depth,
parent_ctx, self_adx, self_hurst, self_dmi_diff, parent_z,
parent_dmi_diff, root_is_roche, tf_alignment, self_pid, osc_coh
```

Defined in `core/feature_extraction.py`. Both clustering and TBN delegate to it.

---

## Data Layout

| Path | Contents | Size |
|------|----------|------|
| `DATA/ATLAS/{tf}/YYYY_MM.parquet` | 14 TFs, 12 months (Jan-Dec 2025) | ~2GB |
| `DATA/ATLAS_OOS/{tf}/YYYY_MM.parquet` | Jan-Mar 2026 | ~500MB |
| `DATA/ATLAS_1DAY/` | Jan 2 only — fast validation | ~20MB |
| `DATA/ATLAS_1WEEK/` | Jan 2-10 — screening | ~120MB |
| `DATA/ATLAS_1MONTH/` | Full month — research | tracked |

14 timeframes: 1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1D, 1W, 1M

Tick size: 0.25 | Tick value: $0.50 (MNQ) | Point value: $2.00

---

## Remaining Code Tasks (0 open)

All tasks completed 2026-03-09. Tasks 2-6 executed, Task 7 assessed and declined
(methods too coupled to LiveEngine for clean extraction; `capture` bug fixed instead).
Specs archived to `docs/archive/`.

---

## Research Queue (11 Analyses)

Full spec: `docs/Active/RESEARCH_SPEC_V_TO_FF.md`

| # | Analysis | Input | Priority | Gate |
|---|----------|-------|----------|------|
| V | Trajectory k-NN (8→9th point) | ATLAS | 1 | ≥2pp over U + ≥100 samples at 90% |
| W | Partial bar robustness of U | ATLAS | 2 | <5pp degradation = live-ready |
| EE | Stop loss optimization | Trade logs | 3 | ≥10% OOS PnL improvement |
| FF | Conviction calibration | Trade logs | 4 | Recalibrated AUC ≥ 0.62 |
| BB | Stacked 176D → GBM direction | ATLAS | 5 | OOS direction ≥ 58% |
| CC | E[PnL] tick prediction | ATLAS | 6 | OOS R² ≥ 0.30 |
| AA | Seed shape → direction | ATLAS | 7 | Direction ≥ 58% per shape |
| DD | Level proximity features | ATLAS + logs | 8 | ≥5pp WR near levels |
| Y | Regime-conditional k-NN | ATLAS | 9 | ≥3pp over global U |
| X | Counter-trend scale | Trade logs | 10 | Late-trend WR > 65% + N ≥ 50 |
| Z | Fractal dimensions (Shi 2018) | ATLAS | 11 | Shape classification > 85% |

EE and FF are fast wins — read existing trade logs, run in seconds, no ATLAS needed.

---

## Roadmap (Priority Order)

1. **Pattern Relevance & Living Brain** — next major branch. Relevance scoring per pattern, auto-decay stale templates. Blocks Rolling OOS + Brain Persistence.
2. **Rolling OOS Window** — `--rolling-oos 30`. Blocked by #1.
3. **Partial Bar Aggregation** — blend completed + forming bar by maturity %. Fixes 4h worker staleness.
4. **Live Brain Persistence** — direction_memory.json + exit_tuning.json across weekends. Blocked by #1.

---

## Key Files to Read After Every Run

1. `reports/is_report.txt` — IS summary
2. `reports/oos_report.txt` — OOS summary (validates IS)
3. `checkpoints/trade_analytics.txt` — IS t-tests, ANOVA, OLS
4. `checkpoints/oos_analytics.txt` — OOS version

---

## Conventions

- **CUDA-only** — no CPU fallback (removed 2026-03-08)
- **No physics metaphors** in new code — statistical/regression language only
- **Tick constants** — size: 0.25, value: $0.50, point: $2.00 (MNQ)
- **Progress bars** — tqdm mandatory for any loop > 100 iterations
- **Journal updates** — `docs/daily/YYYY-MM-DD.md` at end of every session
- **MEMORY.md** — append with dates, never overwrite historical entries
- **Syntax check** — `python -c "import ast; ast.parse(open('file.py').read())"` after edits
- **Git** — single `main` branch, all work consolidated
- **Training** — never run via automated bash. Show command, ask user to confirm.
- **NT8 bridge** — bump version + date in both header comment and `BRIDGE_VERSION` const

---

## Agent Roles

**Claude (chat — claude.ai)**: Architecture review, spec writing, research direction, instruction docs. Does not modify code directly.

**Claude Code (VS Code)**: Reads `CLAUDE.md` at session start. Code changes, refactoring, research scripts. Checks `docs/Active/` for current work.

**Jules (VS Code)**: Same as Claude Code. Accumulates learnings in `.Jules/bolt.md` and `.Jules/palette.md`.

**All agents**: No physics metaphors, CUDA-only, journal at session end, surgical MEMORY.md updates.

---

## What NOT to Do

- Add CPU fallback paths
- Reimplement gate logic in live_engine.py (delegates to ExecutionEngine)
- Delete historical entries from MEMORY.md
- Use physics metaphors in new code
- Run training without user confirmation
- Deploy hurst conviction modulation without a working direction model (see RESEARCH_JOURNAL.txt "HURST INTEGRATION CONFLICT" warning)
- Filter by hour-of-day (design decision: automation trades all sessions)

---

## Source Spec Status

| Spec | Status | Location |
|------|--------|----------|
| Research V-FF (11 analyses) | ACTIVE | `docs/Active/` |
| LEVEL_DETECTOR_SPEC | Research absorbed → Analysis DD | `docs/specs/` |
| JULES_EXPECTED_PROFIT_PREDICTOR | Research absorbed → BB + CC | `docs/specs/` |
| JULES_WAVEFORM_SEED_INTEGRATION | Research absorbed → AA + BB | `docs/specs/` |
| 14 completed specs | ARCHIVED | `docs/archive/` |

Implementation phases from absorbed specs are gated on research promotion.
