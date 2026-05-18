# Scenario-based LSTM multi-head classifier — 2026-05-16 (late-late + autonomous)

## Context
User reframe: "if we are going to go the ML approach I think we need to do it
scenario based — we provide a context and ask it to define a direction,
speed p/t, and a duration, we intentionally create speed, trajectory and
duration buckets". Then "lstm nn".

So: predict 4 scenario buckets from lead-in sequence using an LSTM
multi-head model, with IS training and 2026 OOS validation.

## Pipeline built

| Step | Tool | Output |
|------|------|--------|
| 1. OOS daisy oracle | `tools/regret_daisy_chain_oracle.py` | `daisy_chain_OOS_2026.csv` (2,085 trades) |
| 2. Bucket labeling (IS+OOS) | `tools/scenario_bucket_labeler.py` NEW | `daisy_{IS,OOS}_buckets.csv`, `bucket_boundaries_IS.json` |
| 3. V2 join (OOS) | `tools/regret_join_v2_features.py` | `daisy_with_v2_features_OOS_2026.parquet` |
| 4. Sequence dataset | `tools/scenario_sequence_dataset.py` NEW | `scenario_seq_{IS,OOS}_K60_F30.npz` |
| 5. LSTM train+eval | `tools/scenario_lstm_train.py` NEW | `scenario_lstm_K60_F30.{pt,_history.csv,_OOS_predictions.npz,_summary.json}` |
| 6. LR baseline | `tools/scenario_baseline_lr.py` NEW | `scenario_lr_baseline_summary.json` |

## Bucket definitions

- **direction** (2 classes): LONG=1, SHORT=0 (from daisy CSV)
- **duration** (4 classes): quartiles of `time_to_mfe_min`
  - boundaries: 18.17 / 36.92 / 52.67 min
- **speed** (4 classes): quartiles of `mfe_velocity` ($/min)
  - boundaries: $1.93 / $3.43 / $6.67 per min
- **trajectory** (4 classes): operational MAE/MFE ratio cuts at 0.25 / 0.50 / 0.75
  - MAE = max adverse excursion within (entry_ts, exit_ts), computed from
    raw 5s OHLC at `DATA/ATLAS/5s/`
  - distribution heavily imbalanced: T0(CLEAN) 84%, T1 8%, T2 5%, T3 3%

## Architecture

```
Input:  (B, K=60, N=30)  z-scored, IS stats applied to OOS
Trunk:  LSTM(input=30, hidden=64, num_layers=1, dropout=0.3)
Pool:   last hidden state -> (B, 64)
Heads:  4 linear heads (Direction 2-cls, Duration 4-cls, Speed 4-cls, Trajectory 4-cls)
Loss:   sum of per-head weighted cross-entropy (inverse-frequency weights)
Optim:  Adam lr=1e-3 wd=1e-4, batch=128
Train:  50 epochs, early-stop patience 5 on val total loss
Params: 25,486
```

Top-30 features chosen by combined R^2 + Spearman score from
`feature_prune_representatives_IS_full_daisy_v2.txt` — includes:
- L2_15s_price_velocity_12  (best single, R^2=0.248)
- L2_5s_price_velocity_9
- L2_1m_price_velocity_15
- L1_1m_body, L1_15s_body, ...

## Training

Early-stopped at epoch 11 (best epoch 6). ~5 seconds total on RTX 3060.
- val_loss best 4.3636
- Test-set (IS-val) accuracies at best epoch:
  dir 0.839, dur 0.304, spd 0.434, traj 0.458

## Apples-to-apples comparison: LSTM vs LR (entry-only)

| Head | n_cls | Baseline OOS | LR (entry) OOS | LSTM (60-bar) OOS | LR lift | LSTM lift | LSTM vs LR |
|------|-------|--------------|----------------|-------------------|---------|-----------|------------|
| dir  | 2     | 0.501        | 0.810          | **0.827**         | +0.308  | +0.326    | **+0.017** |
| dur  | 4     | 0.261        | **0.330**      | 0.327             | +0.069  | +0.066    | -0.003 |
| spd  | 4     | 0.292        | **0.443**      | 0.400             | +0.151  | +0.108    | **-0.043** |
| traj | 4     | 0.835        | 0.302          | 0.459             | -0.533  | -0.376    | +0.157 (but BELOW baseline) |

### Verdict

**The LSTM's sequence input adds ~1.7pp on direction OOS and zero on the
other heads.** This is far short of justifying the architectural complexity.

Specifically:
- **Direction**: LSTM 82.7% vs LR 81.0% on OOS — marginal +1.7pp lift. The
  sequence does carry SOME additional signal but it's small.
- **Duration**: Effectively tied (32.7% vs 33.0%). No gain from sequence.
- **Speed**: LSTM HURTS by 4.3pp (40.0% vs 44.3%). The sequence is adding
  noise the LSTM overfits to, despite L2 weight decay and dropout.
- **Trajectory**: Looks like a win (+15.7pp) BUT trajectory is so imbalanced
  (84% T0) that the class-weighted models BOTH score below the always-predict-T0
  baseline (83.5%). Forcing predictions of rare classes is hurting both,
  LSTM less so. The "lift" is artifact of the loss-weighting decision, not
  a real signal advantage.

### IS-OOS generalization (no overfit on direction or duration)

LSTM IS-OOS deltas:
- dir: -1.2pp (excellent generalization)
- dur: +2.3pp (better OOS — within noise)
- spd: -3.5pp (mild overfit)
- traj: +0.1pp

LR IS-OOS deltas:
- dir: -1.8pp
- dur: -0.3pp
- spd: -2.6pp
- traj: -4.4pp

Both models generalize well on direction and duration. Both lose modestly
on speed. The IS-OOS gap pattern is similar.

## Information-ceiling reading

The original direction classifier (V2 entry features, single bar, LR with
calibrated probabilities, no class weighting) gave **AUC 0.864 / 88% acc at
40% coverage / 83% acc at 100% coverage**. The new balanced-class LR also
hits 83% at argmax. The LSTM hits 84%.

**The ~83% direction accuracy at argmax on V2 entry features is the
information ceiling.** Adding sequence input nudges it by ~1pp at best.

For speed/duration prediction, V2 entry features deliver:
- duration: 33% (4-class, baseline 26%) — +7pp lift, weak signal
- speed: 44% (4-class, baseline 29%) — +15pp lift, moderate signal

Adding sequence input does NOT improve these — the LSTM matches or slightly
hurts. This means the speed/duration signal lives at the entry bar; it's
not in the lead-in trajectory.

## What this confirms

1. **The lead-in trajectory carries very little additional information.**
   Both the lossy PCA experiment (60/240/720 bar lookbacks) and the raw
   60-bar LSTM ingest the same lead-in window — both fail to improve over
   entry-feature baselines beyond ~1pp.
2. **The information ceiling is in the V2 feature engineering, not the
   model.** Linear, non-linear (LSTM), and probably GBM will all land
   within 1-2pp of each other on direction.
3. **Speed and duration have weak but real predictive structure.** Useful
   for risk management (sizing, target-setting, time-stop) even if not
   strong enough to be a primary entry signal.
4. **Trajectory bucketing as defined (MAE/MFE) is too imbalanced to be
   useful.** 84% of daisy-chain trades are MONOTONIC (MAE=0). Predicting
   trajectory beyond CLEAN/PULLBACK binary is impractical.

## Recommended next moves

1. **Drop the 4-class trajectory head; collapse to binary CLEAN/PULLBACK.**
   That changes the imbalance from 84/8/5/3 to 84/16 — still skewed but
   sane. Or drop trajectory entirely from the multi-head model.
2. **Test GBM with isotonic calibration** on entry features for the same 4
   buckets. Hypothesis: GBM matches or slightly beats LR for speed and
   duration (where weak non-linear interactions might help), ties on
   direction (linear-separable).
3. **Direction-only model with confidence threshold** is still the
   strongest single signal. AUC 0.864 / 88% acc at 40% coverage. Use it
   as the L4 selector. Speed/duration heads feed risk management
   parameters AFTER direction fires.
4. **Stop chasing the lead-in trajectory as a model input.** The ceiling
   is established. Next levers are richer entry features (new TF combos,
   intra-bar microstructure, calendar/event encoding) or a different
   target (e.g., regret-on-skip rather than direction).

## Files

- `tools/scenario_bucket_labeler.py` — NEW
- `tools/scenario_sequence_dataset.py` — NEW
- `tools/scenario_lstm_train.py` — NEW
- `tools/scenario_baseline_lr.py` — NEW
- `reports/findings/regret_oracle/daisy_chain_OOS_2026.csv`
- `reports/findings/regret_oracle/daisy_{IS,OOS}_buckets.csv`
- `reports/findings/regret_oracle/bucket_boundaries_IS.json`
- `reports/findings/regret_oracle/daisy_with_v2_features_OOS_2026.parquet`
- `reports/findings/regret_oracle/scenario_seq_{IS,OOS}_K60_F30.npz`
- `reports/findings/regret_oracle/scenario_lstm_K60_F30.pt` (checkpoint)
- `reports/findings/regret_oracle/scenario_lstm_K60_F30_history.csv`
- `reports/findings/regret_oracle/scenario_lstm_K60_F30_OOS_predictions.npz`
- `reports/findings/regret_oracle/scenario_lstm_K60_F30_summary.json`
- `reports/findings/regret_oracle/scenario_lr_baseline_summary.json`
- `reports/findings/regret_oracle/scenario_comparison_table.csv`
