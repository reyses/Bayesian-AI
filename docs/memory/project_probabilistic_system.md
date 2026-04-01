---
name: Probabilistic System Architecture
description: 4-brain cascade + peak-to-trend lifecycle + evolving CNN + proven exits — the next-gen trading system
type: project
---

## Probabilistic System (designed 2026-03-30)

Full spec: `docs/Active/PROBABILISTIC_4BRAIN_SPEC.md`

### Architecture
- **Frozen Base CNN**: ProbabilisticTrajectory (22D → 10 horizons × P(long)) — perception, never changes
- **Evolving Trade CNN**: copy of Base, fine-tunes from live outcomes (real + ghost trades from crow)
- **Bayesian Table**: IS→OOS→Live calibration cascade (`core/brain_cascade.py`)
- **Templates**: grow from trade seeds (not pre-built offline)

### Trade Lifecycle: Peak → Trend → Reversal
- Peak detector fires → CNN confirms trajectory → ENTER
- P(direction) sustained across horizons → phase = TREND, trail loosely
- Near horizons (n+1..n+3) weaken while far (n+7..n+10) still strong → REVERSAL IMMINENT
- Near horizons flip (< 0.5) → EXIT before reversal hits
- Each closed trade becomes a seed → seeds cluster into templates → exit params improve

### Key Insight
- Exit mechanisms already work (trail, giveback, envelope, SL) — they just need good entry direction
- CNN provides direction + trajectory shape → exits parametrized by trajectory
- Ghost trades (crow/phantom system) provide 10x more training data for live CNN evolution

### Files Built
- `core/brain_cascade.py` — CalibrationBrain + BrainCascade (4-layer)
- `core/probabilistic_engine.py` — ProbabilisticTradingEngine with lifecycle phases
- `training/train_probabilistic_forward.py` — IS/OOS forward pass

### Still TODO
- Phase D: `live/prob_launcher.py`
- Replay buffer for live CNN fine-tuning (real + ghost trades)
- Wire crow/phantom into replay buffer
- Divergence check: Trade CNN vs Base CNN → rollback if worse

### CNN TF Mismatch Found (2026-03-30)
TradeCNN was trained on 1m data but live fed it 15s bars. Fixed: live now aggregates 4×15s→1m before CNN prediction.
