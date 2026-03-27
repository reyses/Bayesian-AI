---
name: TradeCNN Baseline Results
description: TradeCNN StatePredictor $1,609/day OOS with breakeven protection — current best honest result
type: project
---

## TradeCNN StatePredictor — Baseline (2026-03-27)

**Model**: 13D features → CNNBackbone (Conv1d 13→32→64) → Head (64→7 state outputs)
- ~16K params, predicts 7D feature state at t+10 (not direction directly)
- Walk-forward: cold start Day 1 (30 epochs), carry-forward Day 2+ (5 epochs, LR=1e-4)
- Seed=42 required (model unstable without it — different seeds give $0-$736)

**OOS Result (honest)**:
- $1,609/day, 24% WR, 10,571 trades (39 OOS days)
- TRAIL: +$116,164 (8,923 exits)
- BE: $0 (5,370 exits — breakeven protection saved $107K)
- SL: -$53,360 (2,668 exits)
- Walk-forward: 72.3% direction accuracy, 0.24 avg correlation

**Key mechanisms**:
- Breakeven SL: after +5t profit, SL moves to entry (zero downside after confirmation)
- Trailing stop: activate at +10t, trail at 10t from peak (replaced repeating TP)
- Confidence 3.0 filter separates good/bad trades (77.5% pred accuracy on losers vs 98.4% on winners)
- 2s fill delay: actual 1s data lookup for honest slippage

**SL anatomy**: 59% of SL trades never profitable, 22% had peak > 10t (trail should have caught)

**Backup**: `checkpoints/trade_cnn_backup_fast_v1/`

**Why:** This is the honest baseline to beat. Counter-proposal (29D + two-layer) targets improvement.
**How to apply:** Compare all future models against $1,609/day. Any regression = investigate before shipping.
