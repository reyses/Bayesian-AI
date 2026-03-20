# Roadmap & Future Work

## COMPLETED (V8.0.0, 2026-03-18)

- **Sensor-enriched exits** -- belief_flip requires 2+ sensors, tidal_wave requires 1+. Exits validate with 1s velocity + 1m volume/DMI/momentum before firing. belief_flip dropped 97% noise (631 -> 19 fires).
- **Peak state exit** -- new exit module: "would the system enter against me?" 4 sensors fused. Own ExitAction for tracking.
- **Peak entry gate** -- 1m sensor confirmation before peak entry. Two layers: sensor opposition (blocks LONG-into-crash) + fake peak detection (high volume + momentum = continuation, not reversal).
- **Hard quality filter** -- templates must have >=10 members, >=55% WR, <=10 tick sigma. 176/436 pass.
- **Codebase cleanup** -- 1,700 lines dead code removed, dead imports (33 files), dead methods, unicode fixes.
- **Default pipeline** -- `python training/trainer.py` does IS->checkpoint->OOS. No flags needed.
- **Engine unification** -- no IS/OOS behavioral difference at engine level. Only oracle coverage differs.
- **PFMEA updated** -- 7 exit failure modes mitigated/resolved, 4 new items tracked.

## COMPLETED (V7.0.0, 2026-03-12)

- **TF-bucketed clustering** -- patterns binned by timeframe before K-Means (no scale mixing)
- **OOS compressed per-bar** -- OOS uses same path as live (no discovery, no ancestry features)
- **Phase 7 live replay** -- training calls actual live launcher for integrity test + parity report
- **Live startup simplified** -- no replay warmup, connects straight to NT8 with pre-warmed brain

## COMPLETED (V6.0.0, 2026-03-08)

- **Unified ExitEngine** -- cascade with self-tuning halflife and giveback
- **ExecutionEngine integration** -- gate cascade, direction cascade, sizing
- **Band confluence entry/exit** -- BandContext per TF worker
- **Feature extraction unified** -- core/feature_extraction.py (16D single source of truth)
- **Compressed history replay** -- live/history_replay.py + live/atlas_loader.py
- **Terminology refactor** -- quantum->statistical, full metaphor purge
- **CPU path removed** -- CUDA-only
- **LiveEngine decomposition** -- exit_watcher, gui_bridge, session_tracker, ping_pong

---

## ACTIVE: Live Parity (2026-03-20)

### Problem
OOS: $22.6K, 4,644 trades, $4.87/trade.
Live: 7 trades, $2 profit in 7.5 hours. 303K sensor blocks.

### Root Cause (diagnosed 2026-03-20)
F_momentum uses PID cumsum over full bar history. ATLAS has 417K 1m bars.
Live starts from zero. 17x median divergence. Sensor gate blocks everything.

### Fix (deployed)
Pre-computed states: `python tools/precompute_live_states.py` -> pkl loaded at startup.
1m worker starts with full ATLAS F_momentum. Parity guaranteed.

### Validation (in progress)
All-day live test running. Compare live trades vs OOS for same hours.
NT8 raw data captured to `reports/live/nt8_raw/` for calibration.

---

## ROADMAP: Brain Evolution (established 2026-03-20)

### Stage 1: Lizard Brain (current)
- Win/loss counter per template_id
- BayesianBrain conviction AUC = 0.501 (random, adds no value)
- Serves as gate (should_fire) and direction voter
- **Status**: working but useless for peak trades (one counter for all peaks)

### Stage 2: Crow Brain (8-12 hours)
- k-NN seed matching with 31,605 auto seeds
- Enrich seeds with 10-bar MarketState + 192D context
- Runtime: current state -> 10 nearest seeds -> weighted outcome
- Gives: probability, direction, expected MFE per CONTEXT (not per template)
- **Status**: spec ready, awaiting live parity confirmation

### Stage 3: Monkey Brain (12-18 hours, builds on Crow)
- 1D CNN three-head model trained on enriched seeds
- Heads: P(profitable), P(direction), expected MFE
- 12 channels x 10 bars + 6 context values = 126 inputs
- Replaces: direction cascade (8 voters), exit timing, brain gate
- Spec: `docs/specs/CNN_PEAK_CLASSIFIER.md` (v2, 10 arguments)
- **Status**: spec ready for external validation

### Each stage validates the data pipeline for the next.
### Crow is the safety net -- if CNN fails live, fall back to Crow.

---

## ACTIVE: Research Findings (2026-03-19/20)

### Filters tested and result
| Filter | IS Research | OOS Result | Action |
|--------|-----------|------------|--------|
| ADX chop (<15) | Blocks noise | Blocks profitable trades (PF unchanged) | **Disabled** |
| Fake peak (vol+fm) | 72% precision | Blocked 93% of PnL ($13.6K) | **Disabled** (flag only) |
| DMI at 1m | Expected value | Zero predictive value | **Removed from gate** |
| Buildup (10-bar) | Blocks noise spikes | Blocks 0% of OOS trades | Kept (harmless) |
| Vol+FM sensor gate | Not tested alone | Vol against=$1.67 vs $5.53, FM against=$1.82 vs $5.44 | **Active (only gate)** |
| Peak override | Hold through noise | 5.8% WR, PF 1.01 | **Reverted** |

### Key insight
IS-derived thresholds do NOT transfer to OOS. The features matter (volume, momentum
separate winners from losers) but the threshold VALUES are wrong. This is why the CNN
approach is needed -- it learns thresholds from data, not hand-tuning.

---

## BACKLOG (priority order, updated 2026-03-20)

### 1. 1s Peak Entry with Full+Partial 1m Confirmation
- Peak detection at 1s speed (currently 15s = up to 29.9s latency)
- Full 1m confirmation: completed 1m bar + partial 1m delta
- If 1m fully confirms -> enter at 1s. Partial -> wait for 15s.
- Reduces worst-case entry latency from 29.9s to <1s

### 2. Unify Data Folders
- One `DATA/ATLAS/` with all data (IS + OOS in same directory)
- Oracle coverage boundary determined by Phase 1 discovery range
- No copy/paste to expand dataset

### 3. Extended OOS Data Pipeline
- NT8 tick export to all-TF parquet converter: `tools/nt8_to_parquet.py` (DONE)
- Monthly: export from NT8, convert, extend ATLAS
- Future: automate via NinjaScript scheduled export

### 4. Rolling OOS Window
- `--rolling-oos 30` flag: OOS = last N days, IS = everything before
- Validates current regime relevance, not just historical accuracy

### 5. Live Brain Persistence
- Direction biases, exit tuning, depth weights persist across weekends
- `--full-purge` flag to wipe everything
- Weekly oracle + brain retrain from live trades

### 6. Partial Bar Aggregation
- Workers only update on TF bar close (4h worker frozen for hours)
- Fix: blend completed bar with forming bar, weighted by maturity %
- Scope: worker tick loop, band context interpolation
