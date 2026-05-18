---
name: feedback-scenario-lstm-information-ceiling
description: "LSTM multi-head scenario classifier on lead-in sequence ties or barely exceeds LR on entry features — V2 information ceiling, not model class, is the bottleneck"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User proposed scenario-based ML (direction × speed × duration × trajectory buckets) with LSTM as the trunk. Built full pipeline (bucket labeler, sequence dataset, LSTM trainer, LR baseline) and validated on 2026 OOS (2,085 trades, Jan-Mar 2026, fresh oracle from daisy-chain).

**Result table (OOS):**

| Head | n_cls | Baseline | LR (entry-only) | LSTM (60-bar seq) | LSTM - LR |
|------|-------|----------|-----------------|-------------------|-----------|
| dir  | 2     | 0.501    | 0.810           | 0.827             | +0.017 |
| dur  | 4     | 0.261    | 0.330           | 0.327             | -0.003 |
| spd  | 4     | 0.292    | 0.443           | 0.400             | -0.043 |
| traj | 4     | 0.835    | 0.302           | 0.459             | (both below baseline) |

**Findings:**

1. **The V2 entry-feature information ceiling at ~83% direction accuracy holds.** LR and LSTM converge within ~2pp. Linear model with V2 features captures most of the predictable structure; sequence input adds ~1.7pp at best.

2. **Lead-in trajectory carries minimal additional signal.** Confirmed via two architecturally distinct experiments: (a) lead-in PCA centroid+direction concatenated as features (60/240/720 bar — all hurt, see [[feedback-leadin-pca-rejected]]); (b) raw 60-bar sequence into LSTM trunk (this experiment — marginal +1.7pp on direction, zero or negative on other heads).

3. **Speed/duration signal lives at the entry bar, not in lead-in.** LSTM ties LR on duration, HURTS on speed. The V2 multi-TF features at entry already encode the macro setup that would be inferrable from a 5-minute trajectory.

4. **Trajectory bucket from MAE/MFE ratio is too imbalanced** to be useful as a 4-class target — 84% of daisy-chain trades are MONOTONIC (MAE=$0) because the oracle picks the best extreme. Class-weighted loss forces rare-class predictions and both LR and LSTM end up BELOW the always-predict-CLEAN baseline.

**How to apply:**

- **Stop chasing the lead-in trajectory as a model input.** Two architectures (lossy PCA + raw LSTM) both confirm minimal lift.
- **Stop reaching for more complex models on this feature set.** The ceiling is in V2 feature engineering, not LR vs LSTM vs CNN. Don't escalate from LR until features change.
- **Next levers worth trying:**
  - Richer entry features (new TF combos, intra-bar microstructure, calendar/event encoding, session-relative time)
  - Different target (regret-on-skip vs direction)
  - Coarser trajectory bucket (binary CLEAN/PULLBACK)
  - GBM with isotonic calibration on entry features (60-second probe, still untested)
- **The direction classifier (`tools/regret_direction_classifier.py`) is still the L4 selector signal** — AUC 0.864 at 100% coverage, 88% acc at 40% coverage. The LSTM doesn't replace it.

**Doesn't mean LSTM is useless universally** — it might help with sequence-native problems (e.g., predicting exit timing from intra-trade trajectory, or modeling the entry decision itself as a sequence of bars where each could be a candidate). The failure mode here is "predict scenario from past 60 bars" where the past 60 bars don't carry the signal.

Connected: [[feedback-leadin-pca-rejected]], [[feedback-kway-r2-saturation]], [[project-regret-six-layer-architecture]], [[user-collaboration-protocol]].
