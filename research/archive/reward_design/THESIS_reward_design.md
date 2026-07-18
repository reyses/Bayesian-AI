# Thesis: Reward Design via Causal Physics (UPDATE: BREAK)

## The Final Verdict
This thesis previously considered using tick-level True Delta (Layer 3) to penalize holding through Exhaustion. However, upon subjecting Layer 3 to the full **5-Fold Temporal Walk-Forward and Fourier Null Gauntlet**, the hypothesis has broken.

## The Causal Gauntlet Failure
While True Delta (L3) provided a tiny marginal lift (+0.0036) over free OHLCV wicks (L2) in a raw walk-forward average, it completely **failed the Fourier Phase-Randomized Null**.

- **L3 Fold 5 AUC:** 0.6363
- **Fourier Null 95th pctile (N=20):** 0.6406
- **True Lift over Null:** -0.0043

Any predictive power observed in True Delta is mathematically indistinguishable from random phase noise with the same power spectrum. The "edge" is an illusion.

## The Consequence for RL Reward Design
1. **No Tick Data Required:** We will **NOT** purchase Databento tick data. True Delta provides no causal edge for Exhaustion.
2. **Scrap Order-Flow Specific Penalties:** We will not engineer custom `Exhaustion_Penalty` structures based on micro-structure True Delta. The structural/macro features in the 416D baseline (Layer 1) provide a robust `0.6156` AUC all on their own.
3. **Simplicity:** The RL Agent (Mamba) will rely purely on the existing 416D state representation. The reward loop will focus on larger geometric properties of the trade trajectory (kinematic extensions, max adverse excursions) rather than noisy, non-causal limit-order book artifacts.
