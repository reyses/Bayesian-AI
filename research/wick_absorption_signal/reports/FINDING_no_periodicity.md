# FINDING: No Structural Periodicity (Broadband Red Noise)

**Date:** June 28, 2026
**Project:** Wick Absorption Signal (Bayesian-AI)
**Stage:** Stage 1 (Oscillation Cadence Check)

## Verdict
There is **no characteristic oscillation period** at any timeframe (5s through 1h). The market behaves as broadband red noise. Any apparent cadences or repeating half-cycles observed previously were artifacts induced by the window-size of the smoothing filters (Yule-Slutsky effect). 

Because there is no structural oscillation, there is no reliable fixed "time-to-peak" or predictable cadence to follow. **All downstream targeting and analysis must pivot to an event-driven, turn-relative framework.**

## Evidence & Methodology
To definitively settle the "is there a cycle?" question, we ran a three-method triangulation across all timeframes (5s, 15s, 1m, 5m, 15m, 1h):

1. **Centered Cubic Sweep (The Locator):** We swept a centered cubic regression window ($N \in \{5, 8, 12, 20, 40\}$) and extracted the average detected period. We then regressed the detected period against the window size $N$. 
   - *Result:* The detected period scaled linearly with the window size (Artifact Slopes between 0.48 and 0.86 bars per window bar). There was no plateau. The cadence was window-driven.
2. **Fourier Periodogram (The Referee):** We extracted the power spectral density (PSD) of the detrended log-returns using Welch/FFT and compared it against a phase-randomized surrogate (red-noise null hypothesis).
   - *Result:* The PSD strictly followed the broadband surrogate. There were zero narrowband peaks protruding above the red-noise threshold on any TF.
3. **Autocorrelation (ACF) Trough:** We calculated the first trough of the ACF.
   - *Result:* The first trough consistently occurred at lag 1 or was effectively negligible, indicating 1-bar microstructure anticorrelation rather than a multi-bar cycle.

### Reconciliation Summary
(See `stage_1_oscillation/reconciliation_table.md` for the full dataset).
All three methods agreed: the market is aperiodic. 

## Architectural Consequences
This finding permanently retires all fixed-horizon constructs, "half-cycle" constants, and cadence assumptions in the architecture.

1. **Event Labeling:** Reversals (turns) are real, but they are *irregular events*. The centered cubic (convex/concave) method remains the best locator of these events, but the significance of a turn is now defined by its **magnitude** (e.g., volatility-normalized price excursion), not by smoothing away noise with a large window.
2. **Targeting (Hazard Framing):** The prediction target shifts from a fixed $N$-bar horizon to **time-to-next-turn** or **is-a-reversal-imminent**. This fundamentally re-frames the RL reward exit head and the Wick EDA into a survival/hazard modeling problem.
