# CNN Peak Outcome Classifier -- Research Spec

**Priority**: Research (next branch)
**Date**: 2026-03-18

## Why CNN

The threshold gate (volume + momentum) works but is brittle:
- Fixed thresholds don't adapt to regime changes
- Hand-crafted features miss temporal patterns in the 10-bar sequence
- A CNN sees the RAW time series and learns what matters

## Why train on profit, not labels

Classification (reversal/plateau/continuation) is a proxy.
What we care about is: "if I enter this trade, will I make money?"
Train directly on PnL outcome -- the CNN learns to avoid trades
that classify as "reversal" but still lose (e.g., correct direction
but terrible timing, or reversal that only lasts 2 ticks).

## Architecture

```
Input: 10 bars x C channels (C = price + volume + DMI + F_momentum + ...)
       Shape: (batch, C, 10)

Conv1D(C, 32, kernel=3) -> ReLU -> BatchNorm
Conv1D(32, 64, kernel=3) -> ReLU -> BatchNorm
GlobalAvgPool1D -> (64,)
Linear(64, 32) -> ReLU -> Dropout(0.3)
Linear(32, 1) -> output = expected PnL (regression)

Loss: MSE on actual trade PnL
      OR: binary cross-entropy on (PnL > 0) for classification
```

## Data

- 174,085 peaks from IS 1m data (already extracted)
- Each peak has 10-bar lookback with full MarketState per bar
- Label: actual PnL from 10 bars after peak (or reversal MFE)
- Train/val split: TimeSeriesSplit (no lookahead)

## Channels per bar (from MarketState)

1. price (normalized by entry)
2. volume_delta
3. F_momentum
4. dmi_plus - dmi_minus (DMI gap)
5. adx_strength
6. z_score
7. hurst_exponent
8. oscillation_entropy_normalized (coherence)
9. pid_output
10. regression_sigma

Total: 10 channels x 10 bars = 100 values per sample.

## Novelty advantage

- CNN sees patterns humans can't hand-craft
- Temporal ordering matters (volume dropping THEN spiking != spiking THEN dropping)
- Conv kernels naturally capture "what changed over N bars"
- Regularization (dropout, BatchNorm, early stopping) prevents overfit
- Can retrain monthly on new data without code changes

## Sustainability advantage

- No magic thresholds to tune
- Adapts to regime changes via retraining
- Feature engineering is minimal (raw states, not derived metrics)
- PyTorch model is checkpoint-able, versionable, A/B testable

## Integration

- Train offline: `python tools/train_peak_cnn.py`
- Produces: `checkpoints/peak_cnn.pt`
- `bar_processor._1m_confirms_peak()` loads model, runs inference
- If CNN score < threshold: block entry
- Threshold is a TradingConfig field (tunable without code change)

## Risk

- Overfit to IS data patterns that don't generalize
- Mitigation: TimeSeriesSplit validation, walk-forward retraining
- Latency: CNN inference on 100 values is <1ms on GPU, negligible
- Complexity: one more model to maintain, version, debug
