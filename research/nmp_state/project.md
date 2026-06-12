# NMP State Derivation (DMAIC)

## Define
The NMP algorithm relied heavily on variance ratio ($vr$) and a stability exponent ($\lambda$). In V2, $vr$ was dropped entirely, and $\lambda$ was never formally computed (it was hardcoded to 0). This project aims to accurately reconstruct $\hat{\lambda}$ and $vr$ from the V2 feature set and raw closes in order to properly evaluate the true NMP entry criteria over the V2 185D schema, and to recalibrate the associated $z$-score thresholds.

## Measure
We will measure parity between naive, exact logic and our vectorized implementation for accuracy. We will construct a mapping of $z_{15}$ vs $z_{21}$ quantiles to recalibrate threshold $Z^*$. We will also evaluate a cross-TF variance ratio proxy ($vr_{proxy}$) against the exact $vr$ calculation.

## Analyze
Identify an abstain band for $\hat{\lambda}$ based on the distribution of its t-statistic. Validate if $vr_{proxy}$ correlates highly enough (Spearman $\ge 0.8$) to avoid the need to compute $vr_{exact}$ strictly from raw bars.

## Improve
Implement the derivation code in `derive.py` and run empirical validation (`validate.py`) across a subset of 5 days to confirm trigger parity.

## Control
Ensure all logic is strictly causal by utilizing `_last_closed_idx` semantics. Add a full validation report to `reports/findings` and integrate the end result metrics.
