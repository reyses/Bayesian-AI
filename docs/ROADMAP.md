# Roadmap & Future Work

## COMPLETED (V6.0.0, 2026-03-08)

- **Unified ExitEngine** -- 7-stage cascade (SL→TP→BandUrgent→EnvelopeDecay→PeakGiveback→BreakevenLock→BeliefFlip→Hold) with self-tuning halflife and giveback
- **ExecutionEngine integration** -- gate cascade, direction cascade, sizing. IS/OOS/live share one code path
- **Band confluence entry/exit** -- BandContext per TF worker, get_band_confluence() (Priority 4)
- **Exit improvements (Fixes 1-4)** -- tiered giveback, 30m flip lag, hurst gate, avg_mfe_bar
- **Feature extraction unified** -- core/feature_extraction.py (16D single source of truth)
- **Compressed history replay** -- live/history_replay.py + live/atlas_loader.py
- **Terminology refactor** -- quantum→statistical, ThreeBody→MarketState, full metaphor purge
- **CPU path removed** -- CUDA-only (~310 lines deleted)
- **Pipeline restructured** -- 5→6 phases (Strategy moved after OOS)
- **LiveEngine decomposition** -- exit_watcher, gui_bridge, session_tracker, ping_pong extracted
- **Code consolidation** -- ~600 lines dead code removed, PositionState trimmed, docs reorganized

---

## BRANCH: Pattern Relevance & Living Brain (next major branch)

### Problem
Patterns discovered months ago may not apply to current market regime.
No mechanism to distinguish "still relevant" from "stale contamination."
Weekend purge vs accumulate-forever is a false choice.

### Solution: Oracle-Level Relevance Tracking
Each pattern in pattern_library gets:
- `discovered_date` — when first seen
- `last_confirmed_date` — last time oracle confirmed it worked
- `n_confirmed` / `n_total` — lifetime hit rate
- `recent_confirmed` / `recent_total` — rolling 30-day hit rate
- `relevance_score` = blend of recent + lifetime performance

Relevance formula:
```
relevance = recent_hit_rate * recency_weight + lifetime_hit_rate * (1 - recency_weight)
```

Patterns that keep confirming stay alive, high relevance.
Patterns that stop firing decay and auto-prune.

### What persists vs rebuilds
- Direction biases: PERSIST (most valuable, structural, slow-changing)
- Exit tuning (halflife, giveback): PERSIST (statistical, needs hundreds of trades)
- Depth weights: PERSIST (structural)
- Patterns with ongoing confirmations: PERSIST (real market structure)
- Patterns with no recent confirmations: DECAY and eventually PRUNE

### Files affected
- `core/bayesian_brain.py` — add relevance fields to brain table entries
- `training/trainer.py` — tag discovery_date, update confirmed_date on hits
- `core/execution_engine.py` — weight pattern matches by relevance_score
- `core/fractal_clustering.py` — emit discovery_date on new clusters
- New: `tools/brain_health.py` — pattern age, relevance distribution, stale count

### Production cycle (target workflow)
1. Saturday: export NT8 data, run `nt8_to_parquet.py`
2. `python training/trainer.py --fresh --forward-pass --live-prep`
3. IS trains on all data, OOS validates last month up to Friday
4. Direction + exit tuning carry forward (persisted JSON)
5. Stale patterns auto-pruned by relevance score
6. Monday: go live with clean, relevant brain

### Scope
Full branch — touches brain, trainer, execution engine, clustering, new tooling.
Needs a Jules spec + dedicated branch.

---

## BACKLOG (priority order)

### 1. Rolling OOS Window
- `--rolling-oos 30` flag: OOS forward-start = today - N days
- IS uses everything before that, OOS validates last N days
- Depends on: pattern relevance (otherwise old IS patterns contaminate)

### 2. Partial Bar Aggregation (Timescale Fix)
- **Findings**: `docs/SCALP_TIMESCALE_FINDINGS.md` (2026-03-07)
- Root cause: workers only update on TF bar close (4h worker frozen for hours)
- Exit side: fast TFs (1m) hit local resistance, tighten trail, cause too-early exits
- Entry side: band confluence blend added (40% influence) — needs validation
- **Hourly overlap**: r=0.716 between scalps and too-early exits (same regime problem)
- **Fix**: blend completed bar with forming bar, weighted by bar maturity %
  `effective = completed * (1 - maturity) + partial * maturity`
  Early-bar noise suppressed (maturity ~0.02), late-bar signal trusted (~0.96)
- Scope: worker tick loop, quantum engine partial states, band context interpolation
- Needs Jules spec

### 3. Counter-Trend Scalp Research
- New trade_class column added (correct_dir / counter_trend_scalp / genuinely_wrong / noise)
- 30% of trades are counter-trend scalps ($10,314 profit, oracle timescale mismatch)
- Scalps cluster in bursts (52.8% follow another scalp), MFE is real (68.7 ticks)
- Workers see the trend (57.4% agree with oracle) but template bias overrides
- Partially addressed by band confluence entry blend — run `--fresh --forward-pass` to validate
- Deeper fix: partial bar aggregation (#2 above) addresses exit-side too-early problem

### 3. Extended OOS Data Pipeline
- NT8 tick export to all-TF parquet converter: `tools/nt8_to_parquet.py` (DONE)
- Monthly: export from NT8, convert, extend ATLAS_OOS
- Future: automate via NinjaScript scheduled export

### 4. Live Brain Persistence
- `direction_memory.json` — persists across weekends
- `exit_tuning.json` — persists across weekends
- `depth_weights.json` — persists across weekends
- `--full-purge` flag to wipe everything including accumulated wisdom
- Blocked by: pattern relevance branch

### 5. Multi-Pass Convergence Detection
- `--passes N` implemented, exit engine self-tunes between passes
- Halflife converges to ceiling (60 bars) in 3 passes
- TODO: stop early if params stabilize (delta < threshold)
