# Formal Research Report: Regime Markov Causal Test

### 1. Raw Data Source
- **Location**: `C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/artifacts/stage2_year_segments.json` and `artifacts/regime_buckets.json`. Causal features extracted from `DATA/ATLAS/FEATURES_5s_v2/`.
- **Timeframe**: Chronological segmentation of the 604-day dataset (January 2024 to March 2026), yielding 112,289 structural segments.
- **Granularity**: 1-second segment boundaries and 5-second causal features.

### 2. Data Cleaning & Transformation
- **Sequence Reconstruction**: The segments were sorted strictly chronologically. Using the `regime_buckets.json` dictionary, each segment was labeled with its highest-tier matched regime. Unmatched segments were labeled `NOISE`.
- **Temporal Split**: The ordered array of 112,289 regimes was strictly split into a 70% In-Sample training set and a 30% Out-Of-Sample test set to completely prevent temporal lookahead.

### 3. References
- *Claude's Handover (2026-06-16)*: `comms/HANDOVER_regime_markov_causal_test_2026-06-16.md` which demanded a rigorous, null-controlled evaluation of whether the "Three Universal Laws of Transition" held any genuine predictive power over a base-rate illusion.

### 4. Procedure
The research executed two pre-committed, null-controlled framings:
1. **Framing A (Transition Matrix Predictive Power)**: Evaluated whether the prior completed regime $R_t$ predicts the next regime $R_{t+1}$. A $3030 \times 3030$ transition matrix was fit on the IS data. Its top-1 accuracy on the OOS set was tested against a marginal baseline (always predicting the most common regime, R1).
2. **Framing B (Early Causal Prediction)**: Evaluated whether the structural `volatility_tier` of a *forming* segment could be predicted by extracting the hierarchical SMEP features (Reversion, Kinematics, Bar-shape) exactly at `seg_start + 30 bars` (2.5m). This was tested against a simple trailing-volatility baseline.

Both framings utilized day-block bootstrap resampling (95% CI) and were compared against random-shuffle null controls to establish statistical significance.

### 5. Scripts Used & Locations
- `./regime_markov_causal_test.py`: Reconstructed the temporal sequence and evaluated the transition matrix (Framing A).
- `./regime_causal_earlypredict.py`: Extracted 5s SMEP features and evaluated the Gradient Boosted Classifier for early structural prediction (Framing B).

### 6. Acceptance Criteria
- **Framing A**: The Out-Of-Sample transition matrix accuracy must strictly beat the marginal baseline, the 95% Confidence Interval must exclude 0, and the actual $\Delta$ Accuracy must exceed the 95th percentile of the Sequence-Shuffle Null.
- **Framing B**: The full causal SMEP model must strictly beat the trailing-volatility baseline out-of-sample.

### 7. Results

#### Framing A: Transition Matrix vs Marginal Baseline
* **Real OOS $\Delta$ Accuracy**: `-0.0008` *(The transition matrix was 0.08% WORSE than just blindly predicting R1).*
* **95% Bootstrap CI**: `[-0.0018, -0.0001]`
* **Null A 95th Pct**: `-0.0007`

#### Framing B: Full SMEP Features vs Trailing-Vol Baseline
* **Full SMEP Accuracy**: `0.5230`
* **Trailing-Vol Baseline Accuracy**: `0.5272`
* **Real OOS $\Delta$ Accuracy**: `-0.0042` *(Adding complex structural kinematics performed WORSE than simple volatility).*
* **95% Bootstrap CI**: `[-0.0076, -0.0007]`
* **Null Shuffle 95th Pct**: `0.5275`

> [!WARNING]
> **Base-Rate Illusion Confirmed**
> The "Three Universal Laws of Transition" are entirely a base-rate artifact. Because Regime 1 accounts for 56% of all price action, the transition matrix blindly predicts it. When tested strictly Out-Of-Sample, the transition matrix actually performs *worse* than a naive marginal guess.

### 8. Conclusion
The regime dictionary and its transition matrix have **ZERO** causal predictive power. Early prediction of segment structure using rich kinematic features failed to beat a primitive trailing-volatility baseline.

**DECISION: DEAD as a CAUSAL / LIVE signal.** We will **not** integrate the transition matrix into the
live Kalman engine as a prior (it fails OOS, fails the null, and is firewall-blocked as a forward input).

**Two things survive — both firewall-compliant:**
1. **The feature vocabulary** (velocity / acceleration) — validated, already implemented in the Kalman orange-line.
2. **The regime labels for ORACLE / ceiling diagnostics.** `regime_buckets.json` + the tier labels are
   *non-causal, label-side* quantities — exactly the input an oracle-ceiling analysis needs to bound the
   **achievable** edge under perfect-hindsight regime knowledge (cf. the KT1 oracle ceiling). **Retain them
   as a hindsight measurement / diagnostic; they are dead as a *live* input, not as an *oracle* one.**
   "Retired" means retired from the live path — NOT deleted.
