# CNN Market Brain -- Replaces Bayesian Counter
**Priority**: Next major feature
**Date**: 2026-03-20 (updated from 2026-03-18)
**Status**: Spec v2 -- incorporates research from 2026-03-19 sessions

## Executive Summary

We operate a real-money MNQ futures trading system that detects price
reversals ("peaks") and trades them. The system currently uses a
Bayesian counter (win/loss lookup by template ID) as its "brain."
OOS validation shows $14.3K profit over 2 months (PF 2.21 excl outliers).
However, the brain contributes zero predictive value -- OOS conviction
AUC is 0.501 (random). All actual edge comes from peak detection sensors
and entry filters, not the brain.

We propose replacing the counter with a 1D CNN that learns temporal
patterns from 31,605 human-validated trade examples ("auto seeds").
The CNN would handle entry decisions (should I trade?), direction
(LONG or SHORT?), and exit timing (has the pattern exhausted?) in
a single model, replacing the current 8-voter direction cascade and
12-module exit cascade for peak trades.

## Arguments for the CNN approach

### 1. The current brain is provably useless
- BayesianBrain stores `table[-100] = {wins: 1340, losses: 869}` -- ONE counter for ALL peak trades
- Conviction audit (research FF): AUC 0.501 IS, 0.501 OOS -- literally random
- Brain direction bias was a veto with PF 0.00 on some voters (research 2026-03-18)
- Removing the brain from the gate cascade changes nothing -- it's dead weight

### 2. Hand-crafted filters don't transfer IS -> OOS
- "Fake peak" filter (high volume + momentum): blocked 93% of PROFITABLE OOS trades ($13,628)
- ADX chop filter at 15: blocked trades with identical WR to unblocked ($4.22/trade)
- DMI at 1m: zero predictive value (WR 66.9% against vs 66.1% with)
- Every hand-tuned threshold we tested either hurt performance or added nothing
- The features MATTER (volume, momentum separate winners from losers) but the thresholds are wrong

### 3. A CNN learns the right thresholds from data
- Instead of "block when log_vol > 2.5 AND log_fm > 3.0" (wrong for OOS)
- CNN learns: "this specific COMBINATION of volume shape, momentum trajectory, and timing predicts profit"
- Non-linear interactions that no threshold sweep can capture
- Temporal ordering matters: volume dropping THEN spiking != spiking THEN dropping

### 4. We have excellent training data
- 31,605 auto seeds across 312 trading days (full year 2025)
- Human-trained: a trader manually identified reference trades, algorithm generalized
- Each seed has: direction, entry/exit prices, MFE, MAE, duration, 10-bar lookback
- This is SUPERVISED learning with high-quality labels, not unsupervised clustering

### 5. The direction cascade is harmful
- 8 voters with hand-tuned weights: velocity, DMI, brain_bias, band_confluence, etc.
- Research: velocity-based LONG votes have PF 0.47-0.59 (actively lose money)
- Only 1h and 1s workers show real directional edge (+14-15%)
- Peak trades already KNOW the direction (the reversal implies it)
- The cascade overrides peak's implied direction with noise votes
- CNN Head 2 replaces all 8 voters with one learned output

### 6. Peak override failed because sensors lag
- We tried letting peak detection override exit decisions (2026-03-19)
- Result: 5.8% WR, PF 1.01 -- sensors said "hold" while price round-tripped
- The sensor thresholds are static -- they can't adapt to "how fast is this reversal?"
- CNN continuously evaluates "has the pattern changed?" with learned features
- Exit = CNN now predicts opposite direction with high confidence

### 7. Live parity is solvable
- F_momentum diverges 17x between training and live (PID cumsum cold start)
- ATLAS warmup fixes this (already implemented)
- NT8 native DMI/ADX in bridge (guaranteed parity, already implemented)
- CNN trained on ATLAS states + validated against NT8 raw data = verified parity

### 8. Inference speed supports real-time
- 12 channels x 10 bars = 120 values + 6 context values = 126 inputs
- CNN inference on GPU: <1ms per prediction
- Current 15s decision speed: 15,000ms budget per decision
- Even at 1s speed: 1,000ms budget, CNN uses <0.1% of it
- No latency concern

### 9. Sustainability and adaptability
- Monthly retraining on new data: `python tools/train_peak_cnn.py`
- No threshold tuning -- the model learns what matters
- Walk-forward validation catches regime changes
- A/B testable: run CNN alongside current system, compare outcomes
- PyTorch checkpoint is versionable, reproducible, rollback-able

### 10. This is the natural evolution
- Phase 1: z-score threshold entries (removed -- lookahead)
- Phase 2: peak detection (current -- works but coarse)
- Phase 3: CNN brain (proposed -- learned, context-aware, adaptive)
- Each phase replaced hand-crafted rules with learned patterns
- CNN is the logical next step, not a paradigm shift

## Problem with current brain

The BayesianBrain is a win/loss counter per template_id:
- `table[-100] = {wins: 1340, losses: 869}` -- one entry for ALL peak trades
- Can't distinguish good peaks from bad peaks
- Doesn't use market context (volume, momentum, regime, time of day)
- Direction bias is a simple counter, not context-aware
- Research: conviction has AUC 0.501 OOS (random). Brain adds no predictive value.

## What the brain SHOULD do

1. **See** the current market state (10-bar shape + sensors)
2. **Match** against stored experiences (31K auto seeds)
3. **Weight** by context similarity (regime, volume, momentum)
4. **Output** probability + expected direction + expected MFE
5. **Learn** from each trade (update weights)

A human trader does this instantly through pattern recognition. The CNN does it in <1ms.

## Why CNN over alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Current counter** | Fast, simple | No context, no prediction |
| **k-NN seed matching** | Uses 192D context | Slow (31K distance calcs), doesn't generalize |
| **Random Forest** | Handles features well | No temporal patterns, static |
| **CNN** | Learns temporal patterns, generalizes, fast inference | Needs training data, overfit risk |

CNN wins because:
- Temporal ordering matters (volume dropping THEN spiking != spiking THEN dropping)
- Non-linear patterns (profitable ONLY when ADX > 20 AND volume falling)
- Compresses 31K seeds into learned weights -- no distance computation at runtime
- Inference <1ms (fast enough for 1s decisions)
- Online learning possible (update from live trades)

## Why train on profit, not labels

Classification (reversal/plateau/continuation) is a proxy.
Research (2026-03-19): IS labels don't transfer to OOS. "Fake peaks"
blocked 93% of profitable OOS trades. Labels lie. PnL doesn't.

Train directly on PnL outcome -- the CNN learns to avoid trades
that classify as "reversal" but still lose (correct direction
but terrible timing, or reversal that only lasts 2 ticks).

## Architecture (v2)

```
Input: 10 bars x C channels
       Shape: (batch, C, 10)
       + 1m context snapshot (completed + partial)

Temporal Branch:
  Conv1D(C, 32, kernel=3) -> ReLU -> BatchNorm
  Conv1D(32, 64, kernel=3) -> ReLU -> BatchNorm
  GlobalAvgPool1D -> (64,)

Context Branch:
  Linear(context_dim, 32) -> ReLU
  # 1m completed state + partial delta (vol, fm, dmi, adx)

Fusion:
  Concat(temporal=64, context=32) -> (96,)
  Linear(96, 48) -> ReLU -> Dropout(0.3)

Three Heads:
  Head 1: Linear(48, 1) -> sigmoid -> P(profitable)  [0-1]
  Head 2: Linear(48, 2) -> softmax -> P(LONG), P(SHORT)
  Head 3: Linear(48, 1) -> expected MFE (ticks, regression)

Loss: BCE(head1, win/loss) + CE(head2, direction) + MSE(head3, actual_mfe)
      Weighted: 0.5 * profit + 0.3 * direction + 0.2 * mfe
```

## Channels per bar (from MarketState + NT8 native)

Python-computed (from OHLCV):
1. price (normalized by regression center)
2. z_score (distance from center in sigma)
3. F_momentum (PID cumsum -- needs ATLAS warmup for parity)
4. volume_delta (signed by candle direction)
5. velocity (price change rate)
6. oscillation_entropy_normalized (coherence)
7. P_at_center (regression center)
8. regression_sigma (band width)

NT8-native (from bridge, guaranteed parity):
9. dmi (normalized DI diff, -1 to +1)
10. adx (0-100)

Derived:
11. time_of_day (normalized 0-1, captures session patterns)
12. bar_range (high-low, normalized by ATR)

Total: 12 channels x 10 bars = 120 values + context branch.

## Context branch inputs (1m state at entry)

From completed 1m bar + partial 1m delta:
- vol_1m (completed)
- vol_1m_delta (partial - completed, shows DIRECTION of volume)
- fm_1m (completed)
- fm_1m_delta (partial - completed)
- dmi_1m (from NT8)
- adx_1m (from NT8)

Total context: 6 values.

## Training data

Primary: 31,605 auto seeds across 312 days
- Pre-labeled: direction, MFE, MAE, duration
- Human-trained detection algorithm
- Each seed has entry_price, exit_price, lookback_bars=10

Enrichment: extract 10-bar MarketState at each seed timestamp
  python tools/enrich_seeds_for_cnn.py

Secondary: 174K peaks from IS 1m data (already extracted)
- Less reliable than seeds (no human validation)
- Use as augmentation, not primary training data

Validation: TimeSeriesSplit (5 folds, chronological, no lookahead)
OOS test: Feb-Mar 2026 seeds (if available) or holdout month

## Live parity requirements (from 2026-03-19 research)

F_momentum diverges 17x between ATLAS and live due to PID cumsum cold start.
CNN must be trained on ATLAS data AND validated against live data.

Options:
1. Pre-compute states from ATLAS (same as training), use for CNN input
2. Use NT8 native DMI/ADX (guaranteed parity) for context branch
3. ATLAS warmup for live (already implemented)

Hybrid: CNN temporal branch uses Python states (ATLAS-warmed).
CNN context branch uses NT8 native values (DMI, ADX).

## Integration

### Training
```
python tools/train_peak_cnn.py
  --seeds DATA/regime_seeds/auto_swing/auto_seeds_edited_20260313_212432.json
  --atlas DATA/ATLAS
  --output checkpoints/peak_cnn.pt
```

### Inference (replaces BayesianBrain for peak trades)
```python
# In bar_processor._1m_confirms_peak():
cnn_score = self._peak_cnn.predict(bar_sequence, context)
# cnn_score.prob > 0.5 -> enter
# cnn_score.direction -> LONG/SHORT (replaces direction cascade for peaks)
# cnn_score.expected_mfe -> SL/TP sizing
```

### Live
```python
# peak_cnn.pt loaded at startup
# Inference every 15s bar (or 1s when peak fires)
# No direction cascade needed for peak trades
# Brain counter kept for template trades only
```

## Replacing the direction cascade (for peak trades)

Current: 8 voters (velocity, DMI, brain_bias, band_confluence, etc.)
Research: velocity-based voters PF 0.47-0.59 (harmful)
          DMI at 1m adds no value (WR identical with/against)
          Only 1h and 1s workers have real edge (+14-15%)

CNN Head 2 replaces all voters with a single learned direction.
Trained on 31K seeds with known correct direction. No hand-tuned
weights, no voting, no cascade. The CNN sees the full context
and outputs the probability of each direction.

## Replacing the exit timing (for peak trades)

Current: exit cascade with 12 modules, most are noise for peaks
Research: peak override failed (held losers too long)
          "Fake peak" flag useful for exit timing (not entry)

CNN can be queried DURING the trade: "has the situation changed?"
If CNN now predicts the OPPOSITE direction with high confidence,
that's the exit signal. Same model, different question.

This is the "would the system enter against me?" concept, but
with learned pattern recognition instead of fixed sensor thresholds.

## Risk

- Overfit to IS/seed data patterns that don't generalize to live
  Mitigation: TimeSeriesSplit, walk-forward retraining, OOS validation
- Latency: CNN inference on 126 values is <1ms on GPU, negligible
- Complexity: one model to maintain, version, checkpoint
  Mitigation: monthly retraining script, A/B testing framework
- Regime change: CNN trained on 2025 data may not work in 2027
  Mitigation: rolling window retraining, include recent live trades

## Build order

1. `tools/enrich_seeds_for_cnn.py` -- extract 10-bar states at each seed
2. `tools/train_peak_cnn.py` -- train, validate, save checkpoint
3. `core/peak_cnn.py` -- inference wrapper (load model, predict)
4. Wire into `bar_processor._1m_confirms_peak()` as gate
5. Wire into direction decision (replace cascade for peaks)
6. Wire into exit timing (query during trade)
7. A/B test against current system on OOS
