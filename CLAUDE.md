# Bayesian-AI — Claude Code Project Instructions

## What This Is
MNQ futures trading system. Statistical regression bands + Bayesian learning.
NOT quantum physics — the physics metaphors are historical and fully purged.

## Entry Points
- Training: `python training/trainer.py --fresh --forward-pass`
- Live: `python -m live.launcher --dry-run`
- Research: `python tools/standalone_research.py --data DATA/ATLAS_1WEEK`

## Key Files
- `core/statistical_field_engine.py` — regression, z-scores, probability (CUDA)
- `core/execution_engine.py` — gate cascade, direction, sizing
- `core/exit_engine.py` — SL/TP/envelope/giveback exits
- `core/bayesian_brain.py` — probability table + direction learning
- `core/timeframe_belief_network.py` — 11-TF worker consensus
- `core/fractal_clustering.py` — recursive K-Means templates
- `core/feature_extraction.py` — canonical 16D feature vector
- `live/live_engine.py` — NT8 bridge orchestrator
- `live/history_replay.py` — compressed forward pass for warmup
- `training/trainer.py` — main pipeline (6 phases)

## Active Work
- See `docs/Active/RESEARCH_SPEC_V_TO_FF.md` for async research tasks
- See `docs/ROADMAP.md` for future branches

## Conventions
- CUDA-only (no CPU fallback — removed)
- Tick size: 0.25, tick value: $0.50 (MNQ)
- ATLAS parquet: DATA/ATLAS/{tf}/YYYY_MM.parquet
- Checkpoints: checkpoints/ (gitignored)
- Progress bars mandatory (tqdm) for any loop > 100 iterations
- Update `docs/daily/YYYY-MM-DD.md` at end of every session

## Do NOT
- Add CPU fallback paths (CUDA-only decision)
- Reimplement gate logic in live_engine.py (delegates to ExecutionEngine)
- Delete historical entries from MEMORY.md (append with dates instead)
- Use physics metaphors in new code (regression/statistical language only)
- Run training via Bash — say "run with --fresh when ready" and stop
