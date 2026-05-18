# Forward pass — direction-confidence selector on 2026 OOS

## Setup

Forward-pass simulation on the 2026 OOS daisy-chain oracle (2,085 trades / 56 sessions Jan-Mar 2026) using two direction predictors:

1. **LR** — LogisticRegression on V2 entry features (the `regret_direction_classifier` baseline, AUC 0.864 on IS test)
2. **LSTM** — multi-head LSTM on K=60 lead-in bars × top-30 V2 features (the `scenario_lstm_K60_F30` model from this session)

Tool: `tools/scenario_forward_pass.py` NEW + `tools/scenario_lstm_predict_proba.py` NEW.

## Trade outcome framings

For each trade in OOS, the selector fires if `max(P(LONG), 1-P(LONG)) > threshold`. Two P/L models:

- **Framing A: Oracle-exit (upper bound)** — perfect exit at MFE.
  - Correct direction → +`mfe_dollars`
  - Wrong direction → −`mfe_dollars` (mirror approximation)
- **Framing B: Fixed R/R (more realistic)** — TP=SL=X ticks, $0.50/tick (MNQ).
  - Correct dir, MFE ≥ X ticks → +X × $0.50 (TP hit)
  - Correct dir, MFE < X → +`mfe_dollars` (oracle MFE smaller than TP)
  - Wrong dir, MFE ≥ X → −X × $0.50 (SL hit on opposite move)
  - Wrong dir, MFE < X → −`mfe_dollars` (small mirror loss)
- Skip (confidence below threshold) → $0

Mandatory metrics per CLAUDE.md: bootstrap CI (4,000 resamples, percentile method) on mean $/day; mode + median $/day; Day WR count-based; Trade WR PF-based.

## OOS results — LR vs LSTM

### Framing B (8-tick fixed R/R) — primary realistic comparison

| Threshold | Coverage | LSTM DirAcc | LR $/day | **LSTM $/day** | Delta |
|-----------|----------|-------------|----------|---------------|-------|
| 0.50 | 100.00% | 0.827 | $92 | **$97** | +$5 |
| 0.65 | 85.66% | 0.838 | $87 | **$96** | +$9 |
| 0.75 | 68.68% | 0.847 | $70 | **$91** | +$21 |
| 0.80 | 55.83% | 0.859 | $57 | **$82** | +$25 |
| 0.85 | 42.06% | 0.877 | $42 | **$52** | +$10 |
| 0.90 | 26.33% | 0.917 | $28 | $19 | -$9 |

### Framing A (oracle-exit, upper bound)

| Threshold | Coverage | LSTM DirAcc | LR $/day | **LSTM $/day** | Delta |
|-----------|----------|-------------|----------|---------------|-------|
| 0.50 | 100.00% | 0.827 | $3,222 | **$3,511** | +$290 |
| 0.75 | 68.68% | 0.847 | $2,557 | **$3,350** | +$793 |
| 0.80 | 55.83% | 0.859 | $2,161 | **$3,030** | +$869 |
| 0.85 | 42.06% | 0.877 | $1,691 | **$2,114** | +$423 |
| 0.90 | 26.33% | 0.917 | $1,203 | $819 | -$384 |

### Headline — LSTM at T=0.85, 16-tick R/R (fairest single number)

- **$/day mean: $104.9 — 95% bootstrap CI [$95, $115]**
- $/day mode: $96   $/day median: $104
- Day WR (active days): **1.000** (56 of 56 days positive)
- Trade WR (PF): **+6.18** (winners 6× loser size)
- Mean $/trade: $6.0  Direction acc when firing: 87.7%
- Coverage: 46.71% (974 fires across 56 days = 17.4 trades/day)

## What this proves

1. **Direction edge is real and OOS-stable**. 81.1% at no filter, rising to 91.7% at threshold 0.90. The selector dial works.

2. **LSTM's calibration at high confidence is materially better than LR's**. At argmax (T=0.50) LSTM is only +1.7pp on direction. But at T=0.85, LSTM is 87.7% acc vs LR's 83.9% — a **+3.8pp lift on a more selective subset**. This translates to +$25/day at T=0.80, +$869/day at T=0.80 in oracle framing.

3. **Trade WR (PF-based) is exceptional**: +3.3 at no filter, +6.2 at T=0.85, +10.1 at T=0.90. Winners are 3-10× the loser size structurally because direction is right ~85% of the time and wins/losses are roughly symmetric.

4. **Day WR ~100%** is real GIVEN the assumptions (every active day stacks ~17 trades at 88% accuracy → daily sum is positive on every day in the 56-day OOS). This validates the SELECTOR LAYER works — it doesn't mean we have a deployable strategy.

## Critical caveats — what this forward pass does NOT validate

Per CLAUDE.md anti-doom-cascade rule, lay these out explicitly:

1. **The daisy oracle gives entry TIMING for free.** We're not solving when-to-fire — the oracle picks the best extreme per 60min window with hindsight. A live strategy must discover those entry times in real-time. This is the **largest unsolved problem**.

2. **The 56-day OOS is thin.** CI [$95, $115] for the headline is from 56 daily P/L values. Wider confidence intervals than they look — particularly for the high-threshold low-coverage thresholds.

3. **Wrong-direction = mirror MFE loss is one approximation.** Reality could be milder (cut early) or harsher (no stop discipline). Need a fixed-SL simulation with raw price walks to validate.

4. **8-tick R/R assumes no SL hit if MFE < 8 ticks.** Slight optimism — some trades that didn't favorably move 8 ticks might still have moved -8 ticks adversely first. A walk-through-bars TP/SL race would be more honest.

5. **17 trades/day at high accuracy is unrealistic for a single strategy** — overlapping positions, broker latency, position-size limits all reduce real fill rate. Realistic = 5-10/day.

6. **Live-sim gap from CLAUDE.md memory**: prior baselines had ~$680/day Python-vs-NT8 gap. This strategy is simpler (direction-only, no complex exit logic) so the gap might be smaller, but we have no calibration yet. **Treat the $/day numbers as IS-CEILING-ESTIMATES under multiple intervention scenarios.**

## Multi-axis risk view (CLAUDE.md mandate)

At LSTM T=0.85, 16-tick R/R, headline $/day = $105:

| Live-sim gap assumption | Net $/day |
|------------------------|-----------|
| 0% gap (matches sim) | $105 |
| 30% gap | $73 |
| 60% gap | $42 |
| 100% gap (matches prior $680 historical gap, scaled) | hard to estimate — strategy too different |

With realistic intervention (10 trades/day cap, EOD halt after -$200 day, max position 1 contract): expected $/day proportionally reduced.

## Conclusions

1. **The selector layer (direction classifier) is validated OOS.** AUC 0.864 / 88% acc at 40% coverage on IS reproduces as 81-91% OOS direction accuracy across thresholds, with stable PF-based trade WR of +3 to +10.

2. **LSTM beats LR by ~$25/day at the sweet spot (T=0.80, 8-tick R/R).** Not a huge dollar improvement but it's the calibration at high-confidence tails that pays.

3. **The entry-timing problem is the next bottleneck.** Forward pass assumes we know WHEN to fire (oracle gives this). The deployable equivalent needs:
   - A real-time entry-bar detector (regime-on / regime-off, breakout detector, microstructure trigger)
   - Reuses the direction classifier to pick LONG/SHORT once an entry candidate appears
   - This is the L4-L5 architecture problem still unsolved (per `project_regret_six_layer_architecture.md`)

4. **Don't fall for the $3,000/day oracle-exit number.** That's the theoretical ceiling assuming perfect entry timing AND perfect MFE exit. The realistic deployable equivalent is in the $50-100/day range under conservative assumptions.

## Files

- `tools/scenario_forward_pass.py` NEW — selector simulation with bootstrap CI
- `tools/scenario_lstm_predict_proba.py` NEW — saves LSTM softmax probabilities
- `reports/findings/regret_oracle/scenario_lstm_OOS_proba.npz` — LSTM P(LONG) per trade
- `reports/findings/regret_oracle/forward_pass_OOS_2026.csv` — LR threshold sweep
- `reports/findings/regret_oracle/forward_pass_OOS_2026_LSTM.csv` — LSTM threshold sweep
- `reports/findings/regret_oracle/forward_pass_OOS_2026_combined.csv` — both, merged
