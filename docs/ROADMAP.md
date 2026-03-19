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

## ACTIVE: Peak-Based Template Rebuild (next major work)

### Problem
Current templates are built from z-score threshold crossings (arbitrary, lookahead risk).
Direction is unknown -- needs a cascade of 8 voters, which votes LONG into crashes (Feb 9: -$8,496).
Templates have no context about what happens AFTER the pattern.

### Solution: Peak-Derived Conditional Templates
1. **Peak detection finds reversals** in 1m data (no lookahead by design)
2. **10-bar approach shape + context** extracted at each peak (6D geometry + 16D state + 16D delta + 16D slope = 54D)
3. **Outcome classification**: reversal (69%) / continuation (30%) / plateau (1%)
   Research: 174K peaks analyzed. Top separating features: range (H=5802), volume (H=4244), F_momentum (H=2735)
4. **Direction baked in**: peak tells you direction. No cascade needed.
5. **Exit = next entry**: when opposite template fires, you're done.

### Research completed (2026-03-18)
- `tools/peak_template_research.py` -- 174K peaks, feature extraction, clustering, classifier
- RandomForest at 0.65 threshold: 72.2% precision, blocks 35% of fakes
- Top-3 features alone match full 55-feature classifier
- Threshold gate already wired into `bar_processor._1m_confirms_peak()`

### Next steps
1. CNN classifier trained on profit (not labels) -- spec at `docs/specs/CNN_PEAK_CLASSIFIER.md`
2. Phase 1 rebuild: peak discovery replaces z-score discovery
3. Templates carry direction + expected MFE + sensor profile
4. Direction cascade removed for peak entries

### Files
- Spec: `docs/specs/PEAK_TEMPLATES.md`
- CNN spec: `docs/specs/CNN_PEAK_CLASSIFIER.md`
- Research: `tools/peak_template_research.py`
- Data: `reports/findings/peak_template_*.{csv,txt,png}`

---

## BACKLOG (priority order)

### 1. CNN Peak Classifier
- Train CNN on raw 10-bar x 10-channel time series, target = PnL
- Replaces threshold gate with learned model
- Advantages: temporal pattern detection, regime adaptation via retraining
- Spec: `docs/specs/CNN_PEAK_CLASSIFIER.md`

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
